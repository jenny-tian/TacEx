#!/usr/bin/env python3
"""Evaluate a saved Diffusion BC on the exact held-out Flow-BC record split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "scripts" / "lerobot" / "src"
DSRL_SRC = REPO_ROOT / "scripts" / "reinforcement_learning" / "dsrl"
for path in (str(DSRL_SRC), str(LEROBOT_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from runtime import disable_optional_transformers_discovery  # noqa: E402

disable_optional_transformers_discovery()

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402

from train_lab_pick_diffusion import matched_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/tacex_lab_pick_slide")
    parser.add_argument(
        "--flow-config",
        type=Path,
        default=REPO_ROOT / "outputs/lab_pick_dinov3_flow_bc200_yaw0/config.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.double().reshape(-1)
    y = y.double().reshape(-1)
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denominator) <= 1.0e-12:
        return None
    return float((x * y).sum() / denominator)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_batches < 0:
        raise ValueError("Invalid batch/worker limit.")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    _, validation_episodes, flow_payload = matched_split(args.flow_config)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    policy = DiffusionPolicy.from_pretrained(str(checkpoint)).to(args.device).eval()
    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        episodes=validation_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    losses: list[float] = []
    diffusion_losses: list[float] = []
    visual_xy_losses: list[float] = []
    visual_xy_predictions: list[torch.Tensor] = []
    visual_xy_targets: list[torch.Tensor] = []
    visual_xy_frame_indices: list[torch.Tensor] = []
    initial_visual_xy_predictions: list[torch.Tensor] = []
    initial_visual_xy_targets: list[torch.Tensor] = []
    examples = 0
    with torch.inference_mode():
        for index, raw_batch in enumerate(loader):
            if args.max_batches and index >= args.max_batches:
                break
            batch = preprocessor(raw_batch)
            loss, loss_dict = policy(batch)
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite validation loss at batch {index}: {value}")
            size = int(batch["action"].shape[0])
            losses.extend([value] * size)
            diffusion_losses.extend([float(loss_dict["diffusion_loss"])] * size)
            if "visual_xy_loss" in loss_dict:
                visual_xy_losses.extend([float(loss_dict["visual_xy_loss"])] * size)
                model_batch = dict(batch)
                model_batch[OBS_IMAGES] = torch.stack(
                    [batch[key] for key in policy.config.image_features], dim=-4
                )
                _, visual_xy = policy.diffusion.prepare_global_conditioning_and_visual_xy(
                    model_batch
                )
                current_index = int(policy.config.n_obs_steps) - 1
                target_xy = batch[ACTION][:, current_index, :2]
                visual_xy_predictions.append(visual_xy.detach().cpu())
                visual_xy_targets.append(target_xy.detach().cpu())
                frame_index = raw_batch.get("frame_index")
                if frame_index is not None:
                    frame_index = torch.as_tensor(frame_index).reshape(-1).cpu()
                    visual_xy_frame_indices.append(frame_index)
                    initial = frame_index == 0
                    if bool(initial.any()):
                        initial_visual_xy_predictions.append(visual_xy.detach().cpu()[initial])
                        initial_visual_xy_targets.append(target_xy.detach().cpu()[initial])
            examples += size

    if not losses:
        raise RuntimeError("Validation loader yielded no batches.")
    visual_metrics: dict[str, Any] = {}
    if visual_xy_predictions:
        prediction = torch.cat(visual_xy_predictions)
        target = torch.cat(visual_xy_targets)
        visual_metrics = {
            "examples": int(prediction.shape[0]),
            "mae": (prediction - target).abs().mean(dim=0).tolist(),
            "pearson_r": [pearson(prediction[:, axis], target[:, axis]) for axis in range(2)],
        }
        if visual_xy_frame_indices:
            frame_indices = torch.cat(visual_xy_frame_indices)
            by_frame_offset: dict[str, Any] = {}
            for frame_offset in range(9):
                selected = frame_indices == frame_offset
                if not bool(selected.any()):
                    continue
                frame_prediction = prediction[selected]
                frame_target = target[selected]
                by_frame_offset[str(frame_offset)] = {
                    "examples": int(selected.sum()),
                    "mae": (frame_prediction - frame_target).abs().mean(dim=0).tolist(),
                    "pearson_r": [
                        pearson(frame_prediction[:, axis], frame_target[:, axis])
                        for axis in range(2)
                    ],
                }
            visual_metrics["by_recorded_frame_offset"] = by_frame_offset
            visual_metrics["runtime_camera_warmup_steps"] = 8
            visual_metrics["runtime_warmup_proxy"] = by_frame_offset.get("8")
        if initial_visual_xy_predictions:
            initial_prediction = torch.cat(initial_visual_xy_predictions)
            initial_target = torch.cat(initial_visual_xy_targets)
            visual_metrics["initial_frames"] = {
                "examples": int(initial_prediction.shape[0]),
                "mae": (initial_prediction - initial_target).abs().mean(dim=0).tolist(),
                "pearson_r": [
                    pearson(initial_prediction[:, axis], initial_target[:, axis])
                    for axis in range(2)
                ],
            }

    payload = {
        "schema_version": 1,
        "metric": "normalized_diffusion_mse",
        "checkpoint": str(checkpoint),
        "model_sha256": sha256(model_path),
        "checkpoint_tree_sha256": tree_sha256(checkpoint),
        "dataset_root": str(dataset_root),
        "flow_config": str(args.flow_config.expanduser().resolve()),
        "flow_config_sha256": sha256(args.flow_config.expanduser().resolve()),
        "validation_records": flow_payload["val_record_names"],
        "validation_episodes": validation_episodes,
        "validation_episode_count": len(validation_episodes),
        "validation_dataset_frames": len(dataset),
        "examples_evaluated": examples,
        "batches_evaluated": math.ceil(examples / args.batch_size),
        "mean_loss": sum(losses) / len(losses),
        "mean_diffusion_loss": sum(diffusion_losses) / len(diffusion_losses),
        "mean_visual_xy_loss": (
            None
            if not visual_xy_losses
            else sum(visual_xy_losses) / len(visual_xy_losses)
        ),
        "min_batch_loss": min(losses),
        "max_batch_loss": max(losses),
        "seed": args.seed,
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "visual_xy_metrics": visual_metrics,
        "config": json.loads((checkpoint / "config.json").read_text(encoding="utf-8")),
    }
    json_write(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
