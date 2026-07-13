from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sim_robot.data.sequence_dataset import list_episodes, split_episode_indices
from sim_robot.policy.flow_matching_policy import load_checkpoint, load_policy


DEFAULT_ACTION_LABELS = [
    "target_x",
    "target_y",
    "target_z",
    "target_rot6d_0",
    "target_rot6d_1",
    "target_rot6d_2",
    "target_rot6d_3",
    "target_rot6d_4",
    "target_rot6d_5",
    "target_width",
]


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path


class LoadedPolicy:
    def __init__(self, checkpoint_path: Path, device: str, use_ema: bool, num_inference_steps: int | None) -> None:
        self.model, self.normalizer, self.checkpoint = load_policy(checkpoint_path, device=device, use_ema=use_ema)
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch_obs = {}
        for key, value in obs.items():
            tensor = torch.from_numpy(value).float()
            if key == "robot0_pos" and tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif key == "robot0_image" and tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
            if key == "robot0_pos":
                tensor = self.normalizer.normalize_tensor("robot0_pos", tensor)
            torch_obs[key] = tensor

        result = self.model.predict_action(torch_obs, num_inference_steps=self.num_inference_steps)
        action_norm = result["action"].detach().cpu().numpy()[0]
        return self.normalizer.unnormalize_numpy("action", action_norm)


def resolve_checkpoint(path: Path, prefer: str) -> Path:
    path = path.expanduser()
    if not path.is_dir():
        return path
    preferred = path / f"{prefer}.pt"
    fallback = path / ("best.pt" if prefer == "latest" else "latest.pt")
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No latest.pt or best.pt found in {path}")


def parse_checkpoint_spec(raw: str, prefer: str) -> CheckpointSpec:
    if "=" in raw:
        label, path_text = raw.split("=", 1)
        path = resolve_checkpoint(Path(path_text), prefer=prefer)
        return CheckpointSpec(label=label, path=path)
    path = resolve_checkpoint(Path(raw), prefer=prefer)
    label = path.parent.name if path.name in {"latest.pt", "best.pt"} else path.stem
    return CheckpointSpec(label=label, path=path)


def read_rows(dataset, indices: np.ndarray) -> np.ndarray:
    if len(indices) > 1 and np.all(np.diff(indices) > 0):
        return dataset[indices]
    return np.stack([dataset[int(i)] for i in indices], axis=0)


def episode_length(demo: h5py.Group, action_key: str, state_key: str, image_key: str) -> int:
    lengths = [
        int(demo["obs"][state_key].shape[0]),
        int(demo["obs"][image_key].shape[0]),
        int(demo["actions"][action_key].shape[0]),
    ]
    attr_key = f"length_{action_key}"
    if attr_key in demo.attrs:
        lengths.append(int(demo.attrs[attr_key]))
    return min(lengths)


def make_obs_from_open_hdf5(
    f: h5py.File,
    demo_index: int,
    frame_index: int,
    policy: LoadedPolicy,
    action_key: str,
    state_key: str,
    image_key: str,
) -> dict[str, np.ndarray]:
    demo = f["data"][f"demo_{demo_index}"]
    length = episode_length(demo, action_key=action_key, state_key=state_key, image_key=image_key)
    state_idx = np.clip(
        np.arange(frame_index - policy.config.n_state_obs_steps + 1, frame_index + 1),
        0,
        length - 1,
    )
    image_idx = np.clip(
        np.arange(frame_index - policy.config.n_image_obs_steps + 1, frame_index + 1),
        0,
        length - 1,
    )
    robot0_pos = read_rows(demo["obs"][state_key], state_idx).astype(np.float32)
    image = read_rows(demo["obs"][image_key], image_idx).astype(np.float32) / 255.0
    return {
        "robot0_pos": robot0_pos,
        "robot0_image": np.transpose(image, (0, 3, 1, 2)).astype(np.float32),
    }


def read_gt_action_chunk(f: h5py.File, demo_index: int, frame_index: int, horizon: int, action_key: str) -> np.ndarray:
    demo = f["data"][f"demo_{demo_index}"]
    length = int(demo["actions"][action_key].shape[0])
    action_idx = np.clip(np.arange(frame_index, frame_index + horizon), 0, length - 1)
    return read_rows(demo["actions"][action_key], action_idx).astype(np.float32)


def parse_dims(raw: str, labels: list[str], action_dim: int) -> list[int]:
    if raw == "all":
        return list(range(action_dim))
    if raw == "xyz":
        return [0, 1, 2]
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lstrip("-").isdigit():
            idx = int(item)
        else:
            if item not in labels:
                raise ValueError(f"Unknown action dim label {item!r}. Available labels: {labels}")
            idx = labels.index(item)
        if idx < 0 or idx >= action_dim:
            raise ValueError(f"Action dim index out of range: {idx}")
        result.append(idx)
    if not result:
        raise ValueError("--dims produced an empty dimension list")
    return result


def select_frame_indices(length: int, start: int, end: int | None, stride: int, max_frames: int | None) -> np.ndarray:
    stop = length if end is None else min(end, length)
    frames = np.arange(max(start, 0), stop, max(stride, 1), dtype=np.int64)
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) == 0:
        raise ValueError(f"No frames selected from length={length}, start={start}, end={end}, stride={stride}")
    return frames


def subplot_grid(num_plots: int) -> tuple[int, int]:
    cols = min(4, num_plots)
    rows = int(np.ceil(num_plots / cols))
    return rows, cols


def plot_first_step(
    output_path: Path,
    frames: np.ndarray,
    gt_first: np.ndarray,
    predictions: dict[str, np.ndarray],
    dims: list[int],
    labels: list[str],
) -> None:
    rows, cols = subplot_grid(len(dims))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for plot_idx, dim in enumerate(dims):
        ax = axes.flat[plot_idx]
        ax.axis("on")
        ax.plot(frames, gt_first[:, dim], color="black", linewidth=2.0, label="gt")
        for label, pred in predictions.items():
            ax.plot(frames, pred[:, dim], linewidth=1.3, label=label)
        ax.set_title(labels[dim] if dim < len(labels) else f"dim_{dim}")
        ax.set_xlabel("frame")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_chunk(
    output_path: Path,
    gt_chunk: np.ndarray,
    predictions: dict[str, np.ndarray],
    dims: list[int],
    labels: list[str],
) -> None:
    steps = np.arange(gt_chunk.shape[0])
    rows, cols = subplot_grid(len(dims))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for plot_idx, dim in enumerate(dims):
        ax = axes.flat[plot_idx]
        ax.axis("on")
        ax.plot(steps, gt_chunk[:, dim], color="black", linewidth=2.0, label="gt")
        for label, pred in predictions.items():
            ax.plot(steps, pred[:, dim], linewidth=1.3, label=label)
        ax.set_title(labels[dim] if dim < len(labels) else f"dim_{dim}")
        ax.set_xlabel("chunk step")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_error(diff: np.ndarray) -> dict[str, float]:
    mse = float(np.mean(np.square(diff)))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def read_logs(log_path: Path) -> list[dict]:
    if not log_path.exists():
        print(f"Warning: no logs.jsonl found at {log_path}; skip loss curve for this checkpoint.")
        return []
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_loss_curves(output_path: Path, checkpoint_specs: list[CheckpointSpec]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    wrote_any = False
    for spec in checkpoint_specs:
        log_dir = spec.path.parent if spec.path.is_file() else spec.path
        rows = read_logs(log_dir / "logs.jsonl")
        if not rows:
            continue
        wrote_any = True
        epochs = np.asarray([row["epoch"] for row in rows], dtype=np.int64)
        train_loss = np.asarray([row["train"]["loss"] for row in rows], dtype=np.float64)
        ax.plot(epochs, train_loss, linewidth=1.5, label=f"{spec.label} train")
        val_values = [np.nan if row.get("val") is None else row["val"]["loss"] for row in rows]
        if not np.all(np.isnan(val_values)):
            ax.plot(epochs, val_values, linewidth=1.5, linestyle="--", label=f"{spec.label} val")
    ax.set_title("Sim flow matching loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if wrote_any:
        fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sim single-camera flow-matching action predictions.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint file/dir, optionally label=path. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-prefer", choices=["latest", "best"], default="latest")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-key", type=str, default=None)
    parser.add_argument("--state-key", type=str, default=None)
    parser.add_argument("--image-key", type=str, default=None)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--demo-index", type=int, action="append", default=None)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=120)
    parser.add_argument("--chunk-frame", type=int, default=None)
    parser.add_argument("--dims", type=str, default="all", help="'all', 'xyz', comma-separated indices, or action labels.")
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    checkpoint_specs = [parse_checkpoint_spec(item, prefer=args.checkpoint_prefer) for item in args.checkpoint]
    first_ckpt = load_checkpoint(checkpoint_specs[0].path, map_location="cpu")
    train_config = first_ckpt.get("train_config", {})
    action_key = str(args.action_key if args.action_key is not None else train_config.get("action_key", "high"))
    state_key = str(args.state_key if args.state_key is not None else train_config.get("state_key", "robot0_pos"))
    image_key = str(args.image_key if args.image_key is not None else train_config.get("image_key", "robot0_image"))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else train_config.get("val_ratio", 0.1))
    seed = int(args.seed if args.seed is not None else train_config.get("seed", 42))
    success_only = bool(args.success_only or train_config.get("success_only", False))

    episodes = list_episodes(
        args.dataset,
        action_key=action_key,
        state_key=state_key,
        image_key=image_key,
        success_only=success_only,
    )
    original_ids = np.asarray([episode.index for episode in episodes], dtype=np.int64)
    train_local, val_local = split_episode_indices(len(original_ids), val_ratio=val_ratio, seed=seed)
    train_ids = np.sort(original_ids[train_local])
    val_ids = np.sort(original_ids[val_local])
    if args.demo_index is not None:
        demo_ids = np.asarray(args.demo_index, dtype=np.int64)
        split_name = "manual"
    else:
        split_ids = train_ids if args.split == "train" else val_ids
        if len(split_ids) == 0:
            raise ValueError(f"{args.split} split is empty. Pass --demo-index or adjust --val-ratio.")
        demo_ids = split_ids[: args.max_episodes]
        split_name = args.split

    policies = {
        spec.label: LoadedPolicy(
            checkpoint_path=spec.path,
            device=args.device,
            use_ema=not args.no_ema,
            num_inference_steps=args.num_inference_steps,
        )
        for spec in checkpoint_specs
    }
    first_policy = next(iter(policies.values()))
    horizon = first_policy.config.n_action_steps
    action_dim = first_policy.config.action_dim
    for label, policy in policies.items():
        if policy.config.n_action_steps != horizon or policy.config.action_dim != action_dim:
            raise ValueError(f"Checkpoint {label} has incompatible action horizon/dim.")

    action_labels = DEFAULT_ACTION_LABELS[:action_dim] if action_dim <= len(DEFAULT_ACTION_LABELS) else [f"action_{i}" for i in range(action_dim)]
    dims = parse_dims(args.dims, action_labels, action_dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(args.output_dir / "loss_curves.png", checkpoint_specs)
    summary_rows = []
    with h5py.File(args.dataset, "r") as f:
        for demo_id_raw in demo_ids:
            demo_id = int(demo_id_raw)
            demo = f["data"][f"demo_{demo_id}"]
            length = episode_length(demo, action_key=action_key, state_key=state_key, image_key=image_key)
            frames = select_frame_indices(
                length=length,
                start=args.start_frame,
                end=args.end_frame,
                stride=args.stride,
                max_frames=args.max_frames_per_episode,
            )
            gt_first = []
            first_predictions = {label: [] for label in policies}
            chunk_predictions_for_plot: dict[str, np.ndarray] = {}
            gt_chunk_for_plot = None
            selected_chunk_frame = int(frames[0] if args.chunk_frame is None else args.chunk_frame)

            iterator = tqdm(frames, desc=f"demo_{demo_id}", leave=False)
            for frame_raw in iterator:
                frame = int(frame_raw)
                gt_chunk = read_gt_action_chunk(f, demo_id, frame, horizon=horizon, action_key=action_key)
                gt_first.append(gt_chunk[0])
                for label, policy in policies.items():
                    obs = make_obs_from_open_hdf5(
                        f,
                        demo_id,
                        frame,
                        policy,
                        action_key=action_key,
                        state_key=state_key,
                        image_key=image_key,
                    )
                    pred_chunk = policy.predict(obs).astype(np.float32)
                    first_predictions[label].append(pred_chunk[0])
                    first_stats = summarize_error(pred_chunk[0] - gt_chunk[0])
                    chunk_stats = summarize_error(pred_chunk - gt_chunk)
                    summary_rows.append(
                        {
                            "demo": demo_id,
                            "frame": frame,
                            "checkpoint": label,
                            "first_mae": first_stats["mae"],
                            "first_mse": first_stats["mse"],
                            "first_rmse": first_stats["rmse"],
                            "chunk_mae": chunk_stats["mae"],
                            "chunk_mse": chunk_stats["mse"],
                            "chunk_rmse": chunk_stats["rmse"],
                        }
                    )
                    if frame == selected_chunk_frame:
                        chunk_predictions_for_plot[label] = pred_chunk
                if frame == selected_chunk_frame:
                    gt_chunk_for_plot = gt_chunk

            gt_first_arr = np.asarray(gt_first, dtype=np.float32)
            pred_first_arrs = {label: np.asarray(values, dtype=np.float32) for label, values in first_predictions.items()}
            plot_first_step(
                output_path=args.output_dir / f"demo_{demo_id}_first_step_actions.png",
                frames=frames,
                gt_first=gt_first_arr,
                predictions=pred_first_arrs,
                dims=dims,
                labels=action_labels,
            )
            if gt_chunk_for_plot is not None:
                plot_chunk(
                    output_path=args.output_dir / f"demo_{demo_id}_chunk_frame_{selected_chunk_frame}.png",
                    gt_chunk=gt_chunk_for_plot,
                    predictions=chunk_predictions_for_plot,
                    dims=dims,
                    labels=action_labels,
                )
            if args.save_arrays:
                np.savez_compressed(
                    args.output_dir / f"demo_{demo_id}_predictions.npz",
                    frames=frames,
                    gt_first=gt_first_arr,
                    **{f"pred_first_{label}": value for label, value in pred_first_arrs.items()},
                )

    if not summary_rows:
        raise RuntimeError("No predictions were evaluated.")

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate = {}
    for label in policies:
        rows = [row for row in summary_rows if row["checkpoint"] == label]
        aggregate[label] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ["first_mae", "first_mse", "first_rmse", "chunk_mae", "chunk_mse", "chunk_rmse"]
        }
    payload = {
        "dataset": str(args.dataset),
        "checkpoints": {spec.label: str(spec.path) for spec in checkpoint_specs},
        "action_key": action_key,
        "state_key": state_key,
        "image_key": image_key,
        "val_ratio": val_ratio,
        "seed": seed,
        "split": split_name,
        "demo_ids": [int(x) for x in demo_ids],
        "dims": [action_labels[i] for i in dims],
        "aggregate": aggregate,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Wrote plots and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()

