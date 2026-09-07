"""Train the compact tactile-conditioned BC baseline for GrapePick."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
policy_path = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "grape_pick" / "policy.py"
policy_spec = importlib.util.spec_from_file_location("grape_pick_policy", policy_path)
assert policy_spec is not None and policy_spec.loader is not None
policy_module = importlib.util.module_from_spec(policy_spec)
policy_spec.loader.exec_module(policy_module)
GrapeBCPolicy = policy_module.GrapeBCPolicy
trajectory_policy_observation = policy_module.trajectory_policy_observation


def load_episodes(dataset_dir: Path, max_episode_steps: int):
    episodes = []
    for path in sorted(dataset_dir.glob("record_*/trajectory.npz")):
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("success", False):
            continue
        with np.load(path) as trajectory:
            observation = trajectory_policy_observation(
                trajectory, max_episode_steps=max_episode_steps
            )
            action = np.asarray(trajectory["action"], dtype=np.float32)
        episodes.append((path.parent.name, observation, action))
    if len(episodes) < 2:
        raise ValueError(f"Need at least two successful episodes under {dataset_dir}")
    return episodes


def concatenate(episodes):
    return np.concatenate([episode[1] for episode in episodes]), np.concatenate(
        [episode[2] for episode in episodes]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/grape_pick/state_bc.pt"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/grape_pick/train_report.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    episodes = load_episodes(args.dataset, args.max_episode_steps)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    val_count = max(1, int(round(len(episodes) * args.val_fraction)))
    val_indices = set(order[:val_count].tolist())
    train_episodes = [episode for index, episode in enumerate(episodes) if index not in val_indices]
    val_episodes = [episode for index, episode in enumerate(episodes) if index in val_indices]
    train_obs, train_action = concatenate(train_episodes)
    val_obs, val_action = concatenate(val_episodes)

    obs_mean = torch.from_numpy(train_obs.mean(axis=0)).float().to(device)
    obs_std = torch.from_numpy(train_obs.std(axis=0)).float().to(device).clamp_min(1.0e-4)
    action_mean = torch.from_numpy(train_action.mean(axis=0)).float().to(device)
    action_std = torch.from_numpy(train_action.std(axis=0)).float().to(device).clamp_min(1.0e-4)
    train_x = (torch.from_numpy(train_obs).float() - obs_mean.cpu()) / obs_std.cpu()
    train_y = (torch.from_numpy(train_action).float() - action_mean.cpu()) / action_std.cpu()
    val_x = ((torch.from_numpy(val_obs).float().to(device) - obs_mean) / obs_std)
    val_y = ((torch.from_numpy(val_action).float().to(device) - action_mean) / action_std)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = GrapeBCPolicy(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            loss = torch.mean((prediction - batch_y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
            count += len(batch_x)
        model.eval()
        with torch.inference_mode():
            val_loss = float(torch.mean((model(val_x) - val_y) ** 2).item())
        train_loss = total / max(count, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[TRAIN] epoch={epoch + 1:03d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(val_x) * action_std + action_mean
        target = torch.from_numpy(val_action).float().to(device)
        raw_rmse = torch.sqrt(torch.mean((prediction - target) ** 2, dim=0)).cpu().tolist()
    checkpoint = {
        "model_state_dict": best_state,
        "obs_mean": obs_mean.cpu(),
        "obs_std": obs_std.cpu(),
        "action_mean": action_mean.cpu(),
        "action_std": action_std.cpu(),
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "policy": "trajectory_conditioned_tactile_mlp_bc",
        "tactile_inputs": ["left_pad_normal_force_n", "right_pad_normal_force_n"],
        "train_episodes": len(train_episodes),
        "validation_episodes": len(val_episodes),
        "train_samples": len(train_obs),
        "validation_samples": len(val_obs),
        "best_validation_normalized_mse": best_val,
        "validation_action_rmse": raw_rmse,
        "checkpoint": str(args.output.resolve()),
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps({key: value for key, value in report.items() if key != "history"}), flush=True)


if __name__ == "__main__":
    main()
