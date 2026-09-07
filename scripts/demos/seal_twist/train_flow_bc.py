#!/usr/bin/env python3
"""Train a 6D-wrench-conditioned Flow Matching BC policy for SealTwist."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "seal_twist"
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

from flow_policy import (  # noqa: E402
    FLOW_ACTION_REPRESENTATION,
    FLOW_OBSERVATION_FIELDS,
    PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX,
    RELATIVE_CAP_POSITION_OBSERVATION_SLICE,
    TOOL_POSITION_OBSERVATION_SLICE,
    WRENCH_OBSERVATION_SLICE,
    SealTwistFlowConfig,
    SealTwistFlowPolicy,
    save_flow_checkpoint,
    trajectory_flow_observation,
)


def action_chunks(action: np.ndarray, horizon: int) -> np.ndarray:
    """Construct one edge-padded future action chunk per observation."""

    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"expected action shape [T, 10], got {action.shape}")
    indices = np.arange(len(action))[:, None] + np.arange(horizon)[None, :]
    indices = np.minimum(indices, len(action) - 1)
    return action[indices].astype(np.float32, copy=False)


def load_episodes(dataset: Path, max_episode_steps: int) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for trajectory_path in sorted(dataset.glob("record_*/trajectory.npz")):
        metadata_path = trajectory_path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("success", False):
            continue
        with np.load(trajectory_path) as trajectory:
            observation = trajectory_flow_observation(
                trajectory, max_episode_steps=max_episode_steps
            )
            action = np.asarray(trajectory["action"], dtype=np.float32)
            stored_wrench = np.asarray(trajectory["wrench"], dtype=np.float32)
        if len(observation) != len(action):
            raise ValueError(f"observation/action length mismatch in {trajectory_path}")
        np.testing.assert_array_equal(
            observation[:, WRENCH_OBSERVATION_SLICE], stored_wrench
        )
        episodes.append(
            {
                "name": trajectory_path.parent.name,
                "observation": observation,
                "action": action,
            }
        )
    if len(episodes) < 2:
        raise ValueError(f"need at least two successful episodes under {dataset}")
    return episodes


def stack_split(
    episodes: list[dict[str, object]], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.concatenate(
        [np.asarray(episode["observation"], dtype=np.float32) for episode in episodes]
    )
    relative_chunks: list[np.ndarray] = []
    for episode in episodes:
        episode_observation = np.asarray(episode["observation"], dtype=np.float32)
        chunks = action_chunks(
            np.asarray(episode["action"], dtype=np.float32), horizon
        ).copy()
        chunks[:, :, :3] -= episode_observation[:, None, :3]
        chunks[:, :, 9] -= episode_observation[
            :, None, PREVIOUS_GRIPPER_WIDTH_OBSERVATION_INDEX
        ]
        relative_chunks.append(chunks)
    chunks = np.concatenate(relative_chunks)
    return observations, chunks


def update_ema(
    ema_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            target = ema_state[name]
            detached = value.detach()
            if torch.is_floating_point(target):
                target.mul_(decay).add_(detached, alpha=1.0 - decay)
            else:
                target.copy_(detached)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("exp_report/bottle/dataset"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("exp_report/bottle/checkpoints/flow_bc_6d_force.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("exp_report/bottle/training/flow_bc_6d_force.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=512)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", "--hidden_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-episode-steps", "--max_episode_steps", type=int, default=900)
    parser.add_argument("--val-fraction", "--val_fraction", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument(
        "--endpoint-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for t=0 one-step endpoint consistency. This stabilizes Flow "
            "decoding on the small SealTwist demonstration set."
        ),
    )
    parser.add_argument(
        "--wrench-force-std-floor",
        type=float,
        default=0.25,
        help="Normalization floor in N for Fx/Fy/Fz; prevents contact-noise amplification.",
    )
    parser.add_argument(
        "--wrench-torque-std-floor",
        type=float,
        default=0.02,
        help="Normalization floor in N*m for Tx/Ty/Tz.",
    )
    parser.add_argument(
        "--tool-position-noise-std-m",
        type=float,
        default=0.002,
        help=(
            "Training-only Gaussian perturbation of measured tool xyz. The opposite "
            "perturbation is applied to relative cap xyz so the observation remains "
            "geometrically consistent and the policy learns closed-loop correction."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.horizon < 1:
        parser.error("epochs, batch-size, and horizon must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must lie in (0, 1)")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("ema-decay must lie in [0, 1)")
    if args.endpoint_loss_weight < 0.0:
        parser.error("endpoint-loss-weight must be non-negative")
    if args.wrench_force_std_floor <= 0.0 or args.wrench_torque_std_floor <= 0.0:
        parser.error("wrench normalization floors must be positive")
    if args.tool_position_noise_std_m < 0.0:
        parser.error("tool-position-noise-std-m must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    episodes = load_episodes(args.dataset, args.max_episode_steps)
    order = np.random.default_rng(args.seed).permutation(len(episodes))
    validation_count = max(1, int(round(len(episodes) * args.val_fraction)))
    validation_indices = set(order[:validation_count].tolist())
    train_episodes = [
        episode for index, episode in enumerate(episodes) if index not in validation_indices
    ]
    validation_episodes = [
        episode for index, episode in enumerate(episodes) if index in validation_indices
    ]
    train_observation, train_action = stack_split(train_episodes, args.horizon)
    validation_observation, validation_action = stack_split(
        validation_episodes, args.horizon
    )

    observation_mean = torch.from_numpy(train_observation.mean(axis=0)).float()
    observation_std = torch.from_numpy(train_observation.std(axis=0)).float().clamp_min(1.0e-4)
    force_stop = WRENCH_OBSERVATION_SLICE.start + 3
    observation_std[WRENCH_OBSERVATION_SLICE.start : force_stop].clamp_min_(
        args.wrench_force_std_floor
    )
    observation_std[force_stop : WRENCH_OBSERVATION_SLICE.stop].clamp_min_(
        args.wrench_torque_std_floor
    )
    flat_train_action = train_action.reshape(-1, train_action.shape[-1])
    action_mean = torch.from_numpy(flat_train_action.mean(axis=0)).float()
    action_std = torch.from_numpy(flat_train_action.std(axis=0)).float().clamp_min(1.0e-4)
    train_x = (torch.from_numpy(train_observation).float() - observation_mean) / observation_std
    train_y = (
        torch.from_numpy(train_action).float() - action_mean[None, None]
    ) / action_std[None, None]
    validation_x = (
        torch.from_numpy(validation_observation).float() - observation_mean
    ) / observation_std
    validation_y = (
        torch.from_numpy(validation_action).float() - action_mean[None, None]
    ) / action_std[None, None]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_x, validation_y),
        batch_size=max(args.batch_size, 1024),
        shuffle=False,
        num_workers=0,
    )

    config = SealTwistFlowConfig(
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    model = SealTwistFlowPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    ema_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    validation_model = SealTwistFlowPolicy(config).to(device)
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_flow_total = 0.0
        train_endpoint_total = 0.0
        train_count = 0
        for observation_batch, action_batch in train_loader:
            observation_batch = observation_batch.to(device, non_blocking=True)
            action_batch = action_batch.to(device, non_blocking=True)
            if args.tool_position_noise_std_m > 0.0:
                raw_position_delta = torch.randn(
                    (len(observation_batch), 3),
                    dtype=observation_batch.dtype,
                    device=device,
                ) * float(args.tool_position_noise_std_m)
                observation_batch = observation_batch.clone()
                observation_batch[:, TOOL_POSITION_OBSERVATION_SLICE] += (
                    raw_position_delta
                    / observation_std[TOOL_POSITION_OBSERVATION_SLICE].to(device)
                )
                observation_batch[:, RELATIVE_CAP_POSITION_OBSERVATION_SLICE] -= (
                    raw_position_delta
                    / observation_std[RELATIVE_CAP_POSITION_OBSERVATION_SLICE].to(device)
                )
                action_batch = action_batch.clone()
                action_batch[:, :, :3] -= (
                    raw_position_delta[:, None, :]
                    / action_std[:3].to(device)[None, None, :]
                )
            noise = torch.randn_like(action_batch)
            time = torch.rand(len(action_batch), device=device)
            flow_loss = model.flow_matching_loss(
                observation_batch, action_batch, noise=noise, time=time
            )
            endpoint_velocity = model.vector_field(
                observation_batch,
                noise,
                torch.zeros(len(action_batch), device=device),
            )
            endpoint_loss = torch.mean((noise + endpoint_velocity - action_batch).square())
            loss = flow_loss + args.endpoint_loss_weight * endpoint_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            update_ema(ema_state, model, args.ema_decay)
            train_total += float(loss.item()) * len(observation_batch)
            train_flow_total += float(flow_loss.item()) * len(observation_batch)
            train_endpoint_total += float(endpoint_loss.item()) * len(observation_batch)
            train_count += len(observation_batch)
        scheduler.step()

        validation_model.load_state_dict(ema_state)
        validation_model.eval()
        validation_total = 0.0
        validation_flow_total = 0.0
        validation_endpoint_total = 0.0
        validation_samples = 0
        validation_generator = torch.Generator(device=device).manual_seed(args.seed + 100_000)
        with torch.inference_mode():
            for observation_batch, action_batch in validation_loader:
                observation_batch = observation_batch.to(device)
                action_batch = action_batch.to(device)
                noise = torch.randn(
                    action_batch.shape,
                    generator=validation_generator,
                    device=device,
                    dtype=action_batch.dtype,
                )
                time = torch.rand(
                    len(action_batch), generator=validation_generator, device=device
                )
                flow_loss = validation_model.flow_matching_loss(
                    observation_batch, action_batch, noise=noise, time=time
                )
                endpoint_velocity = validation_model.vector_field(
                    observation_batch,
                    noise,
                    torch.zeros(len(action_batch), device=device),
                )
                endpoint_loss = torch.mean(
                    (noise + endpoint_velocity - action_batch).square()
                )
                loss = flow_loss + args.endpoint_loss_weight * endpoint_loss
                validation_total += float(loss.item()) * len(observation_batch)
                validation_flow_total += float(flow_loss.item()) * len(observation_batch)
                validation_endpoint_total += float(endpoint_loss.item()) * len(observation_batch)
                validation_samples += len(observation_batch)
        train_loss = train_total / max(train_count, 1)
        train_flow_loss = train_flow_total / max(train_count, 1)
        train_endpoint_loss = train_endpoint_total / max(train_count, 1)
        validation_loss = validation_total / max(validation_samples, 1)
        validation_flow_loss = validation_flow_total / max(validation_samples, 1)
        validation_endpoint_loss = validation_endpoint_total / max(validation_samples, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_flow_loss": train_flow_loss,
                "train_endpoint_loss": train_endpoint_loss,
                "validation_loss_ema": validation_loss,
                "validation_flow_loss_ema": validation_flow_loss,
                "validation_endpoint_loss_ema": validation_endpoint_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in ema_state.items()
            }
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[FLOW BC] epoch={epoch + 1:03d}/{args.epochs} "
                f"train={train_loss:.6f} val_ema={validation_loss:.6f} "
                f"flow={validation_flow_loss:.6f} endpoint={validation_endpoint_loss:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    probe_count = min(2048, len(validation_x))
    probe_observation = validation_x[:probe_count].to(device)
    probe_target = validation_y[:probe_count].to(device)
    probe_generator = torch.Generator(device=device).manual_seed(args.seed + 200_000)
    probe_noise = torch.randn(
        probe_target.shape,
        generator=probe_generator,
        device=device,
        dtype=probe_target.dtype,
    )
    with torch.inference_mode():
        decoded = model.decode(
            probe_observation, probe_noise, integration_steps=24, solver="heun"
        )
        normalized_rmse = torch.sqrt(torch.mean((decoded - probe_target).square()))
        decoded_raw = decoded * action_std.to(device)[None, None] + action_mean.to(device)[None, None]
        target_raw = (
            probe_target * action_std.to(device)[None, None]
            + action_mean.to(device)[None, None]
        )
        first_action_rmse = torch.sqrt(
            torch.mean((decoded_raw[:, 0] - target_raw[:, 0]).square(), dim=0)
        )

    metadata = {
        "dataset": str(args.dataset.resolve()),
        "training_seed": args.seed,
        "train_episode_names": [str(episode["name"]) for episode in train_episodes],
        "validation_episode_names": [
            str(episode["name"]) for episode in validation_episodes
        ],
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "training_objective": "straight_path_conditional_flow_matching",
        "path_definition": "x_t=(1-t)*standard_gaussian_noise+t*normalized_action_chunk",
        "velocity_target": "normalized_action_chunk-standard_gaussian_noise",
        "ema_decay": args.ema_decay,
        "endpoint_loss_weight": args.endpoint_loss_weight,
        "six_dimensional_force_source": "trajectory.wrench = [Fx,Fy,Fz,Tx,Ty,Tz]",
        "gripper_state_alignment": (
            "previous sampled action width; causal and available at deployment"
        ),
        "action_representation": FLOW_ACTION_REPRESENTATION,
        "wrench_force_std_floor_n": args.wrench_force_std_floor,
        "wrench_torque_std_floor_nm": args.wrench_torque_std_floor,
        "tool_position_noise_std_m": args.tool_position_noise_std_m,
    }
    save_flow_checkpoint(
        args.checkpoint,
        model=model,
        observation_mean=observation_mean,
        observation_std=observation_std,
        action_mean=action_mean,
        action_std=action_std,
        max_episode_steps=args.max_episode_steps,
        metadata=metadata,
    )
    report = {
        "policy": "seal_twist_6d_wrench_conditional_flow_matching",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": config.__dict__,
        "observation_fields": list(FLOW_OBSERVATION_FIELDS),
        "wrench_observation_indices": [
            WRENCH_OBSERVATION_SLICE.start,
            WRENCH_OBSERVATION_SLICE.stop,
        ],
        "action_representation": FLOW_ACTION_REPRESENTATION,
        "wrench_force_std_floor_n": args.wrench_force_std_floor,
        "wrench_torque_std_floor_nm": args.wrench_torque_std_floor,
        "tool_position_noise_std_m": args.tool_position_noise_std_m,
        "dataset": str(args.dataset.resolve()),
        "training_seed": args.seed,
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "best_validation_combined_loss_ema": best_validation,
        "endpoint_loss_weight": args.endpoint_loss_weight,
        "probe_normalized_decode_rmse": float(normalized_rmse.item()),
        "probe_first_action_raw_rmse": first_action_rmse.cpu().tolist(),
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "[SUMMARY] "
        + json.dumps({key: value for key, value in report.items() if key != "history"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
