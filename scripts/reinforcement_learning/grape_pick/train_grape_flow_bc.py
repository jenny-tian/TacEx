#!/usr/bin/env python3
"""Train a GrapePick state/tactile conditional Flow Matching BC policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from grape_flow_policy import (
    GrapeFlowConfig,
    GrapeFlowPolicy,
    save_flow_checkpoint,
)


def trajectory_observation(
    trajectory: np.lib.npyio.NpzFile, *, max_episode_steps: int
) -> np.ndarray:
    step_fraction = np.asarray(trajectory["step"], dtype=np.float32)[:, None]
    step_fraction = step_fraction / float(max_episode_steps)
    result = np.concatenate(
        (
            trajectory["grape_pos_b"],
            trajectory["stem_anchor_pos_b"],
            trajectory["finger_normal_force_n"],
            trajectory["stem_tension_n"][:, None],
            trajectory["stem_attached"][:, None].astype(np.float32),
            step_fraction,
        ),
        axis=-1,
    ).astype(np.float32)
    if result.shape[1] != 11:
        raise ValueError(f"expected 11-D observations, got {result.shape}")
    return result


def action_chunks(action: np.ndarray, horizon: int) -> np.ndarray:
    """Return an edge-padded future action chunk for every trajectory sample."""

    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"expected actions with shape [T, 10], got {action.shape}")
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
            observation = trajectory_observation(
                trajectory, max_episode_steps=max_episode_steps
            )
            action = np.asarray(trajectory["action"], dtype=np.float32)
        if len(observation) != len(action):
            raise ValueError(f"length mismatch in {trajectory_path}")
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
        [np.asarray(item["observation"], dtype=np.float32) for item in episodes]
    )
    chunks = np.concatenate(
        [
            action_chunks(np.asarray(item["action"], dtype=np.float32), horizon)
            for item in episodes
        ]
    )
    return observations, chunks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9073)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must lie in (0, 1)")

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
    observation_std = (
        torch.from_numpy(train_observation.std(axis=0)).float().clamp_min(1.0e-4)
    )
    flat_train_action = np.concatenate(
        [np.asarray(item["action"], dtype=np.float32) for item in train_episodes]
    )
    action_mean = torch.from_numpy(flat_train_action.mean(axis=0)).float()
    action_std = torch.from_numpy(flat_train_action.std(axis=0)).float().clamp_min(1.0e-4)

    train_x = (torch.from_numpy(train_observation).float() - observation_mean) / observation_std
    train_y = (
        torch.from_numpy(train_action).float() - action_mean[None, None, :]
    ) / action_std[None, None, :]
    validation_x = (
        (torch.from_numpy(validation_observation).float() - observation_mean)
        / observation_std
    )
    validation_y = (
        torch.from_numpy(validation_action).float() - action_mean[None, None, :]
    ) / action_std[None, None, :]
    loader = DataLoader(
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

    config = GrapeFlowConfig(
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    model = GrapeFlowPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for observation_batch, action_batch in loader:
            observation_batch = observation_batch.to(device, non_blocking=True)
            action_batch = action_batch.to(device, non_blocking=True)
            loss = model.flow_matching_loss(observation_batch, action_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item()) * len(observation_batch)
            train_count += len(observation_batch)
        scheduler.step()

        model.eval()
        validation_total = 0.0
        validation_samples = 0
        validation_generator = torch.Generator(device=device).manual_seed(
            args.seed + 100_000
        )
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
                loss = model.flow_matching_loss(
                    observation_batch, action_batch, noise=noise, time=time
                )
                validation_total += float(loss.item()) * len(observation_batch)
                validation_samples += len(observation_batch)
        train_loss = train_total / max(train_count, 1)
        validation_loss = validation_total / max(validation_samples, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_flow_loss": train_loss,
                "validation_flow_loss": validation_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[FLOW BC] epoch={epoch + 1:03d}/{args.epochs} "
                f"train={train_loss:.6f} val={validation_loss:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    probe_count = min(4096, len(validation_x))
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
        decoded_raw = decoded * action_std.to(device)[None, None, :] + action_mean.to(
            device
        )[None, None, :]
        target_raw = probe_target * action_std.to(device)[None, None, :] + action_mean.to(
            device
        )[None, None, :]
        first_action_rmse = torch.sqrt(
            torch.mean((decoded_raw[:, 0] - target_raw[:, 0]).square(), dim=0)
        )

    metadata = {
        "dataset": str(args.dataset.resolve()),
        "training_seed": args.seed,
        "train_episode_names": [str(item["name"]) for item in train_episodes],
        "validation_episode_names": [str(item["name"]) for item in validation_episodes],
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "training_objective": "conditional_flow_matching",
        "path_definition": "x_t=(1-t)*native_gaussian_noise+t*normalized_action_chunk",
        "velocity_target": "normalized_action_chunk-native_gaussian_noise",
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
        "policy": "grape_state_tactile_conditional_flow_matching",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": config.__dict__,
        "dataset": str(args.dataset.resolve()),
        "training_seed": args.seed,
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "best_validation_flow_loss": best_validation,
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
