#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import train_bc as bc  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "tacex_dinov3_fm_bc" / "best.pt"
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
_PYPLOT: Any | None = None
_PYPLOT_IMPORT_ATTEMPTED = False


def get_pyplot():
    global _PYPLOT, _PYPLOT_IMPORT_ATTEMPTED
    if _PYPLOT_IMPORT_ATTEMPTED:
        return _PYPLOT
    _PYPLOT_IMPORT_ATTEMPTED = True
    try:
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        _PYPLOT = plt
    except Exception as exc:
        print(f"Warning: matplotlib is unavailable ({exc}); using Pillow plot fallback.", file=sys.stderr)
        _PYPLOT = None
    return _PYPLOT


def resolve_checkpoint(path: Path, prefer: str) -> Path:
    path = bc.resolve_path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    preferred = path / f"{prefer}.pt"
    fallback = path / ("latest.pt" if prefer == "best" else "best.pt")
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No best.pt or latest.pt found under {path}")


def checkpoint_label(path: Path) -> str:
    if path.name in {"best.pt", "latest.pt"}:
        return f"{path.parent.name}_{path.stem}"
    return path.stem


def value_as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = bc.parse_csv(value)
        return items if items else list(default)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return list(default)


def optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def make_eval_config(args: argparse.Namespace, ckpt: dict[str, Any]) -> dict[str, Any]:
    train_config = ckpt.get("train_config", {})
    policy_config = ckpt.get("policy_config", {})
    checkpoint_image_keys = value_as_list(policy_config.get("image_keys"), ["rgb", "rgb_third"])
    image_keys = value_as_list(args.image_keys if args.image_keys is not None else train_config.get("image_keys"), checkpoint_image_keys)
    data_root = args.data_root
    if data_root is None:
        data_root = optional_path(train_config.get("data_root")) or (REPO_ROOT / "dataset")

    return {
        "data_root": bc.resolve_path(data_root),
        "data_format": args.data_format if args.data_format is not None else train_config.get("data_format", "auto"),
        "image_keys": image_keys,
        "state_obs_steps": int(policy_config.get("state_obs_steps", train_config.get("state_obs_steps", 1))),
        "image_obs_steps": int(policy_config.get("image_obs_steps", train_config.get("image_obs_steps", 1))),
        "chunk_size": int(policy_config.get("chunk_size", train_config.get("chunk_size", 32))),
        "val_ratio": float(args.val_ratio if args.val_ratio is not None else train_config.get("val_ratio", 0.02)),
        "seed": int(args.split_seed if args.split_seed is not None else train_config.get("seed", 42)),
        "quat_order": str(args.quat_order if args.quat_order is not None else train_config.get("quat_order", "wxyz")),
        "hdf5_action_key": str(
            args.hdf5_action_key if args.hdf5_action_key is not None else train_config.get("hdf5_action_key", "auto")
        ),
        "hdf5_state_key": str(
            args.hdf5_state_key if args.hdf5_state_key is not None else train_config.get("hdf5_state_key", "robot0_pos")
        ),
        "hdf5_image_keys": str(
            args.hdf5_image_keys if args.hdf5_image_keys is not None else train_config.get("hdf5_image_keys", "auto")
        ),
    }


def split_or_manual(
    all_ids: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    split: str,
    manual_ids: list[int] | None,
    max_episodes: int | None,
) -> tuple[np.ndarray, str]:
    if manual_ids is not None:
        manual = np.asarray(manual_ids, dtype=np.int64)
        available = set(int(item) for item in all_ids)
        missing = [int(item) for item in manual if int(item) not in available]
        if missing:
            raise ValueError(f"Requested episodes are not available: {missing}; available={sorted(available)}")
        selected = manual
        split_name = "manual"
    elif split == "all":
        selected = all_ids
        split_name = "all"
    elif split == "train":
        selected = train_ids
        split_name = "train"
    else:
        if len(val_ids):
            selected = val_ids
            split_name = "val"
        else:
            selected = train_ids
            split_name = "train_fallback_from_empty_val"

    if max_episodes is not None and max_episodes > 0:
        selected = selected[:max_episodes]
    if len(selected) == 0:
        raise ValueError(f"No episodes selected for split={split_name}")
    return np.asarray(selected, dtype=np.int64), split_name


def build_eval_dataset(
    config: dict[str, Any],
    normalizer: bc.LinearNormalizer,
    split: str,
    manual_episode_ids: list[int] | None,
    max_episodes: int | None,
) -> tuple[torch.utils.data.Dataset, dict[str, Any]]:
    data_root = config["data_root"]
    requested_format = str(config["data_format"])
    record_image_keys = list(config["image_keys"])
    data_format = bc.infer_data_format(data_root, requested_format, record_image_keys)
    train_local: np.ndarray
    val_local: np.ndarray

    if data_format == "records":
        record_dirs = bc.discover_record_dirs(data_root, record_image_keys)
        if not record_dirs:
            raise ValueError(f"No record episodes with image keys {record_image_keys} found under {data_root}")
        train_local, val_local = bc.split_episode_indices(len(record_dirs), config["val_ratio"], config["seed"])
        numeric_to_local = {bc.numeric_suffix(path.name): idx for idx, path in enumerate(record_dirs)}
        all_ids = np.asarray(sorted(numeric_to_local), dtype=np.int64)
        train_ids = np.asarray([bc.numeric_suffix(record_dirs[int(i)].name) for i in train_local], dtype=np.int64)
        val_ids = np.asarray([bc.numeric_suffix(record_dirs[int(i)].name) for i in val_local], dtype=np.int64)
        selected_ids, split_name = split_or_manual(all_ids, train_ids, val_ids, split, manual_episode_ids, max_episodes)
        selected_local = [numeric_to_local[int(ep_id)] for ep_id in selected_ids]
        selected_dirs = [record_dirs[int(local_idx)] for local_idx in selected_local]
        dataset = bc.RecordsSequenceDataset(
            selected_dirs,
            normalizer,
            record_image_keys,
            config["state_obs_steps"],
            config["image_obs_steps"],
            config["chunk_size"],
            config["quat_order"],
        )
        info = {
            "data_format": data_format,
            "data_root": str(data_root),
            "image_keys": record_image_keys,
            "split": split_name,
            "num_episodes": len(record_dirs),
            "train_episodes": len(train_local),
            "val_episodes": len(val_local),
            "selected_episodes": [int(item) for item in selected_ids],
        }
        return dataset, info

    if data_format == "lerobot":
        df = bc.load_lerobot_table(data_root)
        all_ids = np.sort(df["episode_index"].unique().astype(np.int64))
        train_local, val_local = bc.split_episode_indices(len(all_ids), config["val_ratio"], config["seed"])
        train_ids = all_ids[train_local]
        val_ids = all_ids[val_local]
        selected_ids, split_name = split_or_manual(all_ids, train_ids, val_ids, split, manual_episode_ids, max_episodes)
        dataset = bc.LeRobotSequenceDataset(
            data_root,
            selected_ids,
            normalizer,
            record_image_keys,
            config["state_obs_steps"],
            config["image_obs_steps"],
            config["chunk_size"],
        )
        info = {
            "data_format": data_format,
            "data_root": str(data_root),
            "image_keys": record_image_keys,
            "split": split_name,
            "num_episodes": int(len(all_ids)),
            "train_episodes": int(len(train_ids)),
            "val_episodes": int(len(val_ids)),
            "selected_episodes": [int(item) for item in selected_ids],
        }
        return dataset, info

    if data_format == "hdf5":
        import h5py

        with h5py.File(data_root, "r") as f:
            names = sorted(f["data"].keys(), key=bc.numeric_suffix)
            first = f["data"][names[0]]
            action_key = bc.resolve_hdf5_action_key(first["actions"], config["hdf5_action_key"])
            image_keys = bc.resolve_hdf5_image_keys(first["obs"], config["hdf5_image_keys"])
            state_key = config["hdf5_state_key"]
            if state_key not in first["obs"]:
                raise KeyError(f"HDF5 obs group is missing state key {state_key!r}; available: {list(first['obs'])}")

        train_local, val_local = bc.split_episode_indices(len(names), config["val_ratio"], config["seed"])
        numeric_to_local = {bc.numeric_suffix(name): idx for idx, name in enumerate(names)}
        all_ids = np.asarray(sorted(numeric_to_local), dtype=np.int64)
        train_ids = np.asarray([bc.numeric_suffix(names[int(i)]) for i in train_local], dtype=np.int64)
        val_ids = np.asarray([bc.numeric_suffix(names[int(i)]) for i in val_local], dtype=np.int64)
        selected_ids, split_name = split_or_manual(all_ids, train_ids, val_ids, split, manual_episode_ids, max_episodes)
        selected_local = np.asarray([numeric_to_local[int(ep_id)] for ep_id in selected_ids], dtype=np.int64)
        dataset = bc.HDF5SequenceDataset(
            data_root,
            selected_local,
            normalizer,
            state_key,
            action_key,
            image_keys,
            config["state_obs_steps"],
            config["image_obs_steps"],
            config["chunk_size"],
        )
        info = {
            "data_format": data_format,
            "data_root": str(data_root),
            "image_keys": image_keys,
            "hdf5_state_key": state_key,
            "hdf5_action_key": action_key,
            "split": split_name,
            "num_episodes": len(names),
            "train_episodes": int(len(train_local)),
            "val_episodes": int(len(val_local)),
            "selected_episodes": [int(item) for item in selected_ids],
        }
        return dataset, info

    raise ValueError(f"Unsupported data format: {data_format}")


def episode_from_sample(dataset: torch.utils.data.Dataset, sample: tuple[int, int]) -> tuple[int, str, int]:
    first, t = sample
    if isinstance(dataset, bc.LeRobotSequenceDataset):
        episode_id = int(first)
        return episode_id, f"episode_{episode_id:04d}", int(t)
    episode = dataset.episodes[int(first)]
    return int(episode.index), episode.name, int(t)


def select_sample_indices(
    dataset: torch.utils.data.Dataset,
    start_frame: int,
    end_frame: int | None,
    stride: int,
    max_frames_per_episode: int | None,
) -> tuple[list[int], OrderedDict[int, dict[str, Any]]]:
    selected_indices: list[int] = []
    episode_frames: OrderedDict[int, dict[str, Any]] = OrderedDict()
    stride = max(int(stride), 1)
    start_frame = max(int(start_frame), 0)

    for sample_idx, sample in enumerate(dataset.samples):
        episode_id, episode_name, frame = episode_from_sample(dataset, sample)
        if frame < start_frame:
            continue
        if end_frame is not None and frame >= end_frame:
            continue
        if (frame - start_frame) % stride != 0:
            continue

        entry = episode_frames.setdefault(
            episode_id,
            {
                "episode": episode_id,
                "name": episode_name,
                "frames": [],
                "sample_indices": [],
            },
        )
        if max_frames_per_episode is not None and max_frames_per_episode > 0:
            if len(entry["frames"]) >= max_frames_per_episode:
                continue
        entry["frames"].append(int(frame))
        entry["sample_indices"].append(int(sample_idx))
        selected_indices.append(int(sample_idx))

    if not selected_indices:
        raise ValueError(
            "No frames selected. Check --start-frame/--end-frame/--stride/--max-frames-per-episode."
        )
    return selected_indices, episode_frames


def move_obs_to_device(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in obs.items()}


def make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def unnormalize_action(normalizer: bc.LinearNormalizer, value: torch.Tensor) -> np.ndarray:
    return normalizer.unnormalize_numpy(bc.ACTION_KEY, value.detach().cpu().numpy().astype(np.float32))


def summarize_error(diff: np.ndarray) -> dict[str, float]:
    mse = float(np.mean(np.square(diff)))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def summarize_prediction(pred_chunk: np.ndarray, gt_chunk: np.ndarray) -> dict[str, float]:
    first_diff = pred_chunk[0] - gt_chunk[0]
    chunk_diff = pred_chunk - gt_chunk
    first = summarize_error(first_diff)
    chunk = summarize_error(chunk_diff)
    row = {
        "first_mae": first["mae"],
        "first_mse": first["mse"],
        "first_rmse": first["rmse"],
        "chunk_mae": chunk["mae"],
        "chunk_mse": chunk["mse"],
        "chunk_rmse": chunk["rmse"],
    }
    if pred_chunk.shape[-1] >= 3:
        first_xyz = summarize_error(first_diff[:3])
        chunk_xyz = summarize_error(chunk_diff[..., :3])
        row.update(
            {
                "first_xyz_mae": first_xyz["mae"],
                "first_xyz_mse": first_xyz["mse"],
                "first_xyz_rmse": first_xyz["rmse"],
                "chunk_xyz_mae": chunk_xyz["mae"],
                "chunk_xyz_mse": chunk_xyz["mse"],
                "chunk_xyz_rmse": chunk_xyz["rmse"],
            }
        )
    return row


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    skip = {"episode", "episode_name", "frame"}
    keys = [key for key in rows[0] if key not in skip and isinstance(rows[0][key], (int, float, np.floating))]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def action_labels(action_dim: int) -> list[str]:
    if action_dim <= len(DEFAULT_ACTION_LABELS):
        return DEFAULT_ACTION_LABELS[:action_dim]
    return [f"action_{idx}" for idx in range(action_dim)]


def parse_dims(raw: str, labels: list[str], action_dim: int) -> list[int]:
    aliases = {
        "all": list(range(action_dim)),
        "xyz": [idx for idx in range(min(3, action_dim))],
        "rot6d": [idx for idx in range(3, min(9, action_dim))],
        "width": [action_dim - 1] if action_dim > 0 else [],
    }
    dims: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token in aliases:
            candidates = aliases[token]
        elif token.lstrip("-").isdigit():
            candidates = [int(token)]
        else:
            if token not in labels:
                raise ValueError(f"Unknown action dim {token!r}. Available labels: {labels}")
            candidates = [labels.index(token)]
        for dim in candidates:
            if dim < 0 or dim >= action_dim:
                raise ValueError(f"Action dim index out of range: {dim}")
            if dim not in dims:
                dims.append(dim)
    if not dims:
        raise ValueError("--dims selected no dimensions")
    return dims


def subplot_grid(num_plots: int) -> tuple[int, int]:
    cols = min(4, num_plots)
    rows = int(np.ceil(num_plots / cols))
    return rows, cols


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if abs(hi - lo) < 1.0e-8:
        pad = max(abs(lo) * 0.05, 1.0e-3)
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def draw_line_plot(
    draw: Any,
    area: tuple[int, int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    left, top, right, bottom = area
    x0, x1 = x_limits
    y0, y1 = y_limits
    if len(x) == 0 or abs(x1 - x0) < 1.0e-12 or abs(y1 - y0) < 1.0e-12:
        return
    points = []
    for x_value, y_value in zip(x, y):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        px = left + (float(x_value) - x0) / (x1 - x0) * (right - left)
        py = bottom - (float(y_value) - y0) / (y1 - y0) * (bottom - top)
        points.append((int(round(px)), int(round(py))))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
    elif points:
        px, py = points[0]
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)


def fallback_series_plot(
    output_path: Path,
    x: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    dims: list[int],
    labels: list[str],
    x_label: str,
    title: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cols = min(4, len(dims))
    rows = int(np.ceil(len(dims) / cols))
    cell_w, cell_h = 420, 280
    margin_l, margin_t, margin_r, margin_b = 58, 42, 22, 46
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x_limits = padded_limits(x)
    for plot_idx, dim in enumerate(dims):
        row = plot_idx // cols
        col = plot_idx % cols
        ox, oy = col * cell_w, row * cell_h
        area = (ox + margin_l, oy + margin_t, ox + cell_w - margin_r, oy + cell_h - margin_b)
        y_limits = padded_limits(np.concatenate([gt[:, dim], pred[:, dim]], axis=0))
        draw.rectangle(area, outline=(190, 190, 190), width=1)
        for frac in (0.25, 0.5, 0.75):
            y = int(area[1] + frac * (area[3] - area[1]))
            draw.line((area[0], y, area[2], y), fill=(230, 230, 230), width=1)
        draw_line_plot(draw, area, x, gt[:, dim], x_limits, y_limits, (20, 20, 20), width=3)
        draw_line_plot(draw, area, x, pred[:, dim], x_limits, y_limits, (31, 119, 180), width=2)
        draw.text((ox + 12, oy + 10), labels[dim] if dim < len(labels) else f"dim_{dim}", fill=(20, 20, 20), font=font)
        draw.text((area[0], oy + cell_h - 28), x_label, fill=(80, 80, 80), font=font)
        draw.text((area[0], area[1] - 16), f"{y_limits[1]:.3g}", fill=(100, 100, 100), font=font)
        draw.text((area[0], area[3] + 4), f"{y_limits[0]:.3g}", fill=(100, 100, 100), font=font)
    draw.text((12, image.height - 16), f"{title} | black=dataset blue=prediction", fill=(40, 40, 40), font=font)
    image.save(output_path)


def fallback_xyz_plot(output_path: Path, gt_first: np.ndarray, pred_first: np.ndarray) -> None:
    from PIL import Image, ImageDraw, ImageFont

    projections = [("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
    cell_w, cell_h = 360, 320
    margin_l, margin_t, margin_r, margin_b = 54, 42, 24, 48
    image = Image.new("RGB", (len(projections) * cell_w, cell_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, (name, x_dim, y_dim) in enumerate(projections):
        ox = idx * cell_w
        area = (ox + margin_l, margin_t, ox + cell_w - margin_r, cell_h - margin_b)
        x_limits = padded_limits(np.concatenate([gt_first[:, x_dim], pred_first[:, x_dim]], axis=0))
        y_limits = padded_limits(np.concatenate([gt_first[:, y_dim], pred_first[:, y_dim]], axis=0))
        draw.rectangle(area, outline=(190, 190, 190), width=1)
        draw_line_plot(draw, area, gt_first[:, x_dim], gt_first[:, y_dim], x_limits, y_limits, (20, 20, 20), width=3)
        draw_line_plot(draw, area, pred_first[:, x_dim], pred_first[:, y_dim], x_limits, y_limits, (31, 119, 180), width=2)
        draw.text((ox + 12, 10), f"{name.upper()} projection", fill=(20, 20, 20), font=font)
        draw.text((area[0], area[3] + 8), f"{name[0]}", fill=(80, 80, 80), font=font)
        draw.text((ox + 12, area[1]), f"{name[1]}", fill=(80, 80, 80), font=font)
    draw.text((12, image.height - 16), "XYZ path | black=dataset blue=prediction", fill=(40, 40, 40), font=font)
    image.save(output_path)


def plot_first_step(
    output_path: Path,
    frames: np.ndarray,
    gt_first: np.ndarray,
    pred_first: np.ndarray,
    dims: list[int],
    labels: list[str],
) -> None:
    plt = get_pyplot()
    if plt is None:
        fallback_series_plot(output_path, frames, gt_first, pred_first, dims, labels, "frame", "first-step actions")
        return
    rows, cols = subplot_grid(len(dims))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for plot_idx, dim in enumerate(dims):
        ax = axes.flat[plot_idx]
        ax.axis("on")
        ax.plot(frames, gt_first[:, dim], color="black", linewidth=2.0, label="dataset")
        ax.plot(frames, pred_first[:, dim], color="tab:blue", linewidth=1.5, label="prediction")
        ax.set_title(labels[dim] if dim < len(labels) else f"dim_{dim}")
        ax.set_xlabel("frame")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_xyz_path(output_path: Path, gt_first: np.ndarray, pred_first: np.ndarray) -> None:
    plt = get_pyplot()
    if plt is None:
        fallback_xyz_plot(output_path, gt_first, pred_first)
        return
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_first[:, 0], gt_first[:, 1], gt_first[:, 2], color="black", linewidth=2.0, label="dataset")
    ax.plot(pred_first[:, 0], pred_first[:, 1], pred_first[:, 2], color="tab:blue", linewidth=1.5, label="prediction")
    ax.scatter(gt_first[:1, 0], gt_first[:1, 1], gt_first[:1, 2], color="black", s=20)
    ax.scatter(pred_first[:1, 0], pred_first[:1, 1], pred_first[:1, 2], color="tab:blue", s=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("XYZ first-step action path")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_chunk(
    output_path: Path,
    gt_chunk: np.ndarray,
    pred_chunk: np.ndarray,
    dims: list[int],
    labels: list[str],
) -> None:
    plt = get_pyplot()
    if plt is None:
        fallback_series_plot(output_path, np.arange(gt_chunk.shape[0]), gt_chunk, pred_chunk, dims, labels, "chunk step", "chunk actions")
        return
    steps = np.arange(gt_chunk.shape[0])
    rows, cols = subplot_grid(len(dims))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for plot_idx, dim in enumerate(dims):
        ax = axes.flat[plot_idx]
        ax.axis("on")
        ax.plot(steps, gt_chunk[:, dim], color="black", linewidth=2.0, label="dataset")
        ax.plot(steps, pred_chunk[:, dim], color="tab:blue", linewidth=1.5, label="prediction")
        ax.set_title(labels[dim] if dim < len(labels) else f"dim_{dim}")
        ax.set_xlabel("chunk step")
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prepare_episode_buffers(episode_frames: OrderedDict[int, dict[str, Any]]) -> OrderedDict[int, dict[str, Any]]:
    buffers: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for episode_id, entry in episode_frames.items():
        buffers[episode_id] = {
            "episode": int(episode_id),
            "name": entry["name"],
            "frames": [],
            "gt_first": [],
            "pred_first": [],
            "chunk_frame": None,
            "gt_chunk": None,
            "pred_chunk": None,
        }
    return buffers


def evaluate(
    model: bc.DINOFlowMatchingPolicy,
    normalizer: bc.LinearNormalizer,
    dataset: torch.utils.data.Dataset,
    sample_indices: list[int],
    episode_frames: OrderedDict[int, dict[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_inference_steps: int | None,
    seed: int | None,
) -> tuple[list[dict[str, Any]], OrderedDict[int, dict[str, Any]]]:
    subset = Subset(dataset, sample_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    generator = make_generator(device, seed)
    rows: list[dict[str, Any]] = []
    episode_buffers = prepare_episode_buffers(episode_frames)
    name_by_episode = {episode_id: entry["name"] for episode_id, entry in episode_frames.items()}

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="open-loop eval", leave=False):
            obs = move_obs_to_device(batch["obs"], device)
            result = model.predict_action(obs, generator=generator, num_inference_steps=num_inference_steps)
            pred = unnormalize_action(normalizer, result[bc.ACTION_KEY])
            gt = unnormalize_action(normalizer, batch[bc.ACTION_KEY])
            episodes = batch["episode"].detach().cpu().numpy().astype(np.int64)
            frames = batch["t"].detach().cpu().numpy().astype(np.int64)

            for i in range(pred.shape[0]):
                episode_id = int(episodes[i])
                frame = int(frames[i])
                metrics = summarize_prediction(pred[i], gt[i])
                row = {
                    "episode": episode_id,
                    "episode_name": name_by_episode.get(episode_id, f"episode_{episode_id:04d}"),
                    "frame": frame,
                }
                row.update(metrics)
                rows.append(row)

                buffer = episode_buffers[episode_id]
                buffer["frames"].append(frame)
                buffer["gt_first"].append(gt[i, 0].astype(np.float32))
                buffer["pred_first"].append(pred[i, 0].astype(np.float32))
                if buffer["chunk_frame"] is None:
                    buffer["chunk_frame"] = frame
                    buffer["gt_chunk"] = gt[i].astype(np.float32)
                    buffer["pred_chunk"] = pred[i].astype(np.float32)

    return rows, episode_buffers


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No metrics rows were produced.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    output_dir: Path,
    episode_buffers: OrderedDict[int, dict[str, Any]],
    dims: list[int],
    labels: list[str],
    save_arrays: bool,
) -> None:
    for episode_id, buffer in episode_buffers.items():
        frames = np.asarray(buffer["frames"], dtype=np.int64)
        if len(frames) == 0:
            continue
        gt_first = np.stack(buffer["gt_first"], axis=0)
        pred_first = np.stack(buffer["pred_first"], axis=0)
        stem = f"episode_{episode_id:04d}"
        plot_first_step(output_dir / f"{stem}_first_step_actions.png", frames, gt_first, pred_first, dims, labels)
        if gt_first.shape[-1] >= 3:
            plot_xyz_path(output_dir / f"{stem}_xyz_path.png", gt_first, pred_first)
        if buffer["gt_chunk"] is not None and buffer["pred_chunk"] is not None:
            plot_chunk(
                output_dir / f"{stem}_chunk_frame_{int(buffer['chunk_frame'])}.png",
                buffer["gt_chunk"],
                buffer["pred_chunk"],
                dims,
                labels,
            )
        if save_arrays:
            np.savez_compressed(
                output_dir / f"{stem}_predictions.npz",
                frames=frames,
                gt_first=gt_first,
                pred_first=pred_first,
                chunk_frame=np.asarray(buffer["chunk_frame"], dtype=np.int64),
                gt_chunk=buffer["gt_chunk"],
                pred_chunk=buffer["pred_chunk"],
            )


def grouped_episode_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = f"{int(row['episode']):04d}"
        grouped.setdefault(key, []).append(row)
    return {key: aggregate_metrics(group_rows) for key, group_rows in grouped.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop evaluation for TacEx DINOv3 flow-matching BC checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-prefer", choices=("best", "latest"), default="best")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--data-format", choices=("auto", "records", "lerobot", "hdf5"), default=None)
    parser.add_argument("--image-keys", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--episode-index", type=int, action="append", default=None)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--dims", type=str, default="xyz,width")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default=None)
    parser.add_argument("--hdf5-action-key", type=str, default=None)
    parser.add_argument("--hdf5-state-key", type=str, default=None)
    parser.add_argument("--hdf5-image-keys", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_prefer)
    ckpt = bc.load_checkpoint(checkpoint_path, map_location="cpu")
    config = make_eval_config(args, ckpt)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "bc_open_loop" / checkpoint_label(checkpoint_path)
    output_dir = bc.resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, normalizer, loaded_ckpt = bc.load_policy(checkpoint_path, device=device, use_ema=not args.no_ema)
    config = make_eval_config(args, loaded_ckpt)
    dataset, dataset_info = build_eval_dataset(
        config,
        normalizer,
        split=args.split,
        manual_episode_ids=args.episode_index,
        max_episodes=args.max_episodes,
    )
    sample_indices, episode_frames = select_sample_indices(
        dataset,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        max_frames_per_episode=args.max_frames_per_episode,
    )

    action_dim = int(model.config.action_dim)
    labels = action_labels(action_dim)
    dims = parse_dims(args.dims, labels, action_dim)
    inference_seed = int(args.seed) if args.seed is not None else int(config["seed"])

    rows, episode_buffers = evaluate(
        model=model,
        normalizer=normalizer,
        dataset=dataset,
        sample_indices=sample_indices,
        episode_frames=episode_frames,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        num_inference_steps=args.num_inference_steps,
        seed=inference_seed,
    )
    write_metrics(output_dir / "metrics.csv", rows)
    write_plots(output_dir, episode_buffers, dims, labels, args.save_arrays)

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_type": loaded_ckpt.get("model_type"),
        "epoch": loaded_ckpt.get("epoch"),
        "global_step": loaded_ckpt.get("global_step"),
        "use_ema": not args.no_ema,
        "device": str(device),
        "num_inference_steps": args.num_inference_steps if args.num_inference_steps is not None else model.config.num_inference_steps,
        "seed": inference_seed,
        "dataset": dataset_info,
        "frames": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "stride": args.stride,
            "max_frames_per_episode": args.max_frames_per_episode,
            "num_frames": len(sample_indices),
        },
        "action_dims": [labels[dim] for dim in dims],
        "aggregate": aggregate_metrics(rows),
        "episodes": grouped_episode_aggregate(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(bc.json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(bc.json_ready(summary["aggregate"]), indent=2))
    print(f"Wrote open-loop plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
