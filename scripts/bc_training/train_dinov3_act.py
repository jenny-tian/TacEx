#!/usr/bin/env python
"""Train the deterministic DINOv3-ACT behavior-cloning policy.

This is the recommended BC path for the yaw-zero LabPick experiment.  It
filters failed demonstrations, validates the recorded yaw/quaternion contract,
precomputes frozen DINOv3 spatial features, and trains a masked action-chunk
transformer without sampling noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import dinov3_act as act
import train_bc as common


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_VERSION = 1


def record_succeeded(record_dir: Path) -> bool:
    metadata_path = record_dir / "metadata.npz"
    if not metadata_path.exists():
        return False
    with np.load(metadata_path) as metadata:
        return bool(np.asarray(metadata.get("success", False)).reshape(-1)[0])


def record_reset_yaw_degrees(record_dir: Path) -> float:
    with np.load(record_dir / "metadata.npz") as metadata:
        quat = np.asarray(metadata["labware_reset_quat_w"], dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def select_records(args: argparse.Namespace) -> list[Path]:
    root = common.resolve_path(args.data_root)
    records = common.discover_record_dirs(root, common.parse_csv(args.image_keys))
    if args.success_only:
        records = [record for record in records if record_succeeded(record)]
    if args.require_yaw_zero:
        violations = [
            (record.name, record_reset_yaw_degrees(record))
            for record in records
            if abs(record_reset_yaw_degrees(record)) > args.yaw_tolerance_degrees
        ]
        if violations:
            preview = ", ".join(f"{name}={yaw:.3f}deg" for name, yaw in violations[:5])
            raise ValueError(f"Found {len(violations)} non-zero-yaw records: {preview}")
    if args.max_episodes is not None:
        if len(records) < args.max_episodes:
            raise ValueError(f"Requested {args.max_episodes} records, but only {len(records)} passed filtering")
        records = records[: args.max_episodes]
    if not records:
        raise ValueError(f"No usable records found under {root}")
    return records


class ACTRecordsDataset(Dataset):
    """Records dataset with phase, padding masks, and optional DINO cache."""

    def __init__(
        self,
        record_dirs: list[Path],
        normalizer: common.LinearNormalizer,
        image_keys: list[str],
        state_obs_steps: int,
        image_obs_steps: int,
        chunk_size: int,
        quat_order: str,
        feature_cache_dir: Path | None = None,
    ) -> None:
        self.base = common.RecordsSequenceDataset(
            record_dirs=record_dirs,
            normalizer=normalizer,
            image_keys=image_keys,
            state_obs_steps=state_obs_steps,
            image_obs_steps=image_obs_steps,
            chunk_size=chunk_size,
            quat_order=quat_order,
        )
        self.feature_cache_dir = feature_cache_dir
        self._feature_cache: dict[int, dict[str, np.ndarray]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_feature_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.base)

    @property
    def state_dim(self) -> int:
        return self.base.state_dim

    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    @property
    def num_cameras(self) -> int:
        return self.base.num_cameras

    @property
    def image_shape(self) -> tuple[int, ...]:
        return self.base.image_shape

    def _episode_features(self, local_idx: int) -> dict[str, np.ndarray]:
        if self.feature_cache_dir is None:
            raise RuntimeError("Feature cache is not configured")
        if local_idx not in self._feature_cache:
            episode = self.base.episodes[local_idx]
            feature_dir = self.feature_cache_dir / episode.name
            self._feature_cache[local_idx] = {
                key: np.load(feature_dir / f"{key}.npy", mmap_mode="r") for key in self.base.image_keys
            }
        return self._feature_cache[local_idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        local_idx, t = self.base.samples[idx]
        episode = self.base.episodes[local_idx]
        length = episode.length
        valid_steps = min(self.base.chunk_size, length - t)
        action_is_pad = np.arange(self.base.chunk_size) >= valid_steps
        phase = float(t / max(length - 1, 1))

        if self.feature_cache_dir is None:
            sample = self.base[idx]
            sample["obs"]["phase"] = torch.tensor([phase], dtype=torch.float32)
            sample["action_is_pad"] = torch.from_numpy(action_is_pad)
            return sample

        arrays = self.base._episode_arrays(local_idx)
        state_idx = common.clamp_indices(
            length, np.arange(t - self.base.state_obs_steps + 1, t + 1)
        )
        image_idx = common.clamp_indices(
            length, np.arange(t - self.base.image_obs_steps + 1, t + 1)
        )
        action_idx = common.clamp_indices(length, np.arange(t, t + self.base.chunk_size))
        state = self.base.normalizer.normalize_numpy(
            common.STATE_KEY, arrays[common.STATE_KEY][state_idx].astype(np.float32)
        )
        action = self.base.normalizer.normalize_numpy(
            common.ACTION_KEY, arrays[common.ACTION_KEY][action_idx].astype(np.float32)
        )
        cached = self._episode_features(local_idx)
        features = np.stack([cached[key][image_idx] for key in self.base.image_keys], axis=1)
        return {
            "obs": {
                common.STATE_KEY: torch.from_numpy(state),
                "dino_features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
                "phase": torch.tensor([phase], dtype=torch.float32),
            },
            common.ACTION_KEY: torch.from_numpy(action),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "episode_index": torch.tensor(episode.index, dtype=torch.long),
            "frame_index": torch.tensor(t, dtype=torch.long),
        }


def feature_signature(
    dino_path: Path,
    image_keys: list[str],
    feature_grid_size: int,
    dino_hidden_size: int,
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dino_path": str(dino_path.resolve()),
        "image_keys": image_keys,
        "feature_grid_size": int(feature_grid_size),
        "num_spatial_tokens": int(feature_grid_size**2),
        "dino_hidden_size": int(dino_hidden_size),
        "dtype": "float16",
    }


def cache_record_valid(
    cache_dir: Path,
    record_dir: Path,
    signature: dict[str, Any],
    image_keys: list[str],
) -> bool:
    metadata_path = cache_dir / record_dir.name / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if metadata.get("signature") != signature:
        return False
    expected_length = common.record_length(record_dir, image_keys)
    expected_shape = (
        expected_length,
        signature["num_spatial_tokens"],
        signature["dino_hidden_size"],
    )
    for key in image_keys:
        try:
            array = np.load(cache_dir / record_dir.name / f"{key}.npy", mmap_mode="r")
        except (FileNotFoundError, ValueError):
            return False
        if array.shape != expected_shape or array.dtype != np.float16:
            return False
    return True


@torch.inference_mode()
def precompute_dino_features(
    record_dirs: list[Path],
    cache_dir: Path,
    image_keys: list[str],
    dino: common.MinimalDINOv3ViT,
    dino_path: Path,
    feature_grid_size: int,
    batch_size: int,
    device: torch.device,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = feature_signature(dino_path, image_keys, feature_grid_size, dino.config.hidden_size)
    dino = dino.to(device).eval()
    mean = torch.tensor(common.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(common.IMAGENET_STD, device=device).view(1, 3, 1, 1)

    pending = [record for record in record_dirs if not cache_record_valid(cache_dir, record, signature, image_keys)]
    print(f"[CACHE] valid={len(record_dirs) - len(pending)} pending={len(pending)} root={cache_dir}", flush=True)
    for record_number, record_dir in enumerate(pending, start=1):
        output_dir = cache_dir / record_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        length = common.record_length(record_dir, image_keys)
        for key in image_keys:
            images = np.load(record_dir / "aligned" / f"{key}.npy", mmap_mode="r")
            temporary = output_dir / f".{key}.partial.npy"
            output = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float16,
                shape=(length, feature_grid_size**2, dino.config.hidden_size),
            )
            for start in range(0, length, batch_size):
                stop = min(start + batch_size, length)
                # mmap slices are read-only; copy before torch conversion so
                # later in-place normalization cannot alias immutable storage.
                batch_np = np.array(images[start:stop], copy=True)
                batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
                batch = batch.div_(255.0)
                batch = (batch - mean) / std
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                    patches = dino(batch)
                num_patches = patches.shape[1]
                side = int(round(math.sqrt(num_patches)))
                if side * side != num_patches:
                    raise ValueError(f"DINO returned {num_patches} non-square patch tokens")
                pooled = F.adaptive_avg_pool2d(
                    patches.transpose(1, 2).reshape(-1, patches.shape[-1], side, side),
                    output_size=(feature_grid_size, feature_grid_size),
                ).flatten(2).transpose(1, 2)
                output[start:stop] = pooled.float().cpu().numpy().astype(np.float16)
            output.flush()
            del output
            os.replace(temporary, output_dir / f"{key}.npy")
        metadata = {
            "record": str(record_dir.resolve()),
            "length": length,
            "signature": signature,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[CACHE] {record_number}/{len(pending)} {record_dir.name} frames={length}", flush=True)


def move_to_device(batch: Any, device: torch.device) -> Any:
    return common.move_to_device(batch, device)


def run_epoch(
    model: act.DINOv3ACTPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    train: bool,
    amp: bool,
    epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    grad_clip: float,
) -> tuple[dict[str, float], int]:
    model.train(train)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        if train:
            assert optimizer is not None
            lr = common.warmup_cosine_lr(global_step, total_steps, warmup_steps, base_lr)
            common.set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda", dtype=torch.float16):
                losses = model.compute_loss(batch)
                loss = losses["loss"]
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    label = "train" if train else "val"
    print(
        f"epoch={epoch:03d} split={label} batches={batches} "
        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
        flush=True,
    )
    return metrics, global_step


def save_checkpoint(
    path: Path,
    model: act.DINOv3ACTPolicy,
    normalizer: common.LinearNormalizer,
    train_config: dict[str, Any],
    epoch: int,
    global_step: int,
    validation_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_type": act.MODEL_TYPE,
        "model": act.policy_state_dict(model),
        "normalizer": normalizer.state_dict(),
        "policy_config": model.config.to_dict(),
        "train_config": train_config,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_loss": float(validation_loss),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deterministic DINOv3-ACT BC for TacEx LabPick")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--image-keys", type=str, default="rgb,rgb_third")
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-yaw-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yaw-tolerance-degrees", type=float, default=0.1)
    parser.add_argument("--quat-order", choices=("xyzw",), default="xyzw")
    parser.add_argument("--dino-path", type=Path, default=None)
    parser.add_argument("--precompute-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-batch-size", type=int, default=128)
    parser.add_argument("--feature-grid-size", type=int, default=7)
    parser.add_argument("--condition-grid-size", type=int, default=4)

    parser.add_argument("--state-obs-steps", type=int, default=2)
    parser.add_argument("--image-obs-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--normalizer-mode", choices=("limits", "standard"), default="limits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5, help="Save a deployment-only checkpoint every N epochs; 0 disables it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = common.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_keys = common.parse_csv(args.image_keys)
    records = select_records(args)
    train_indices, val_indices = common.split_episode_indices(len(records), args.val_ratio, args.seed)
    train_records = [records[int(index)] for index in train_indices]
    val_records = [records[int(index)] for index in val_indices]
    normalizer = common.compute_records_normalizer(train_records, args.normalizer_mode, args.quat_order)

    dino, dino_path = common.load_dino_for_config(args.dino_path)
    cache_dir: Path | None = None
    if args.precompute_features:
        cache_dir = common.resolve_path(args.feature_cache_dir or (output_dir / "dino_feature_cache"))
        precompute_dino_features(
            records,
            cache_dir,
            image_keys,
            dino,
            dino_path,
            args.feature_grid_size,
            args.cache_batch_size,
            device,
        )
        dino = dino.cpu()
        torch.cuda.empty_cache()

    train_set = ACTRecordsDataset(
        train_records,
        normalizer,
        image_keys,
        args.state_obs_steps,
        args.image_obs_steps,
        args.chunk_size,
        args.quat_order,
        cache_dir,
    )
    val_set = ACTRecordsDataset(
        val_records,
        normalizer,
        image_keys,
        args.state_obs_steps,
        args.image_obs_steps,
        args.chunk_size,
        args.quat_order,
        cache_dir,
    )
    episode_lengths = [common.record_length(record, image_keys) for record in records]
    policy_config = act.DINOv3ACTConfig(
        state_dim=train_set.state_dim,
        action_dim=train_set.action_dim,
        num_cameras=train_set.num_cameras,
        state_obs_steps=args.state_obs_steps,
        image_obs_steps=args.image_obs_steps,
        chunk_size=args.chunk_size,
        feature_grid_size=args.feature_grid_size,
        condition_grid_size=args.condition_grid_size,
        dino_hidden_size=dino.config.hidden_size,
        model_dim=args.model_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        phase_horizon_steps=int(round(float(np.median(episode_lengths)) - 1.0)),
        dino_path=str(dino_path.resolve()),
        image_keys=tuple(image_keys),
    )
    model = act.DINOv3ACTPolicy(policy_config, dino).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    train_config = common.json_ready(vars(args))
    train_config.update(
        {
            "model_type": act.MODEL_TYPE,
            "data_root": str(common.resolve_path(args.data_root)),
            "feature_cache_dir": None if cache_dir is None else str(cache_dir),
            "num_episodes_total": len(records),
            "num_episodes_train": len(train_records),
            "num_episodes_val": len(val_records),
            "num_successful_episodes": sum(record_succeeded(record) for record in records),
            "yaw_min_degrees": min(record_reset_yaw_degrees(record) for record in records),
            "yaw_max_degrees": max(record_reset_yaw_degrees(record) for record in records),
            "train_record_names": [record.name for record in train_records],
            "val_record_names": [record.name for record in val_records],
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "policy_config": policy_config.to_dict(),
            "trainable_parameters": common.count_parameters(model),
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(common.json_ready(train_config), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: train_config[key] for key in (
        "model_type", "num_episodes_total", "num_episodes_train", "num_episodes_val",
        "train_samples", "val_samples", "yaw_min_degrees", "yaw_max_degrees", "trainable_parameters"
    )}, indent=2), flush=True)

    total_steps = max(len(train_loader), 1) * args.epochs
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    log_path = output_dir / "logs.jsonl"
    if log_path.exists():
        log_path.unlink()
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            train=True,
            amp=args.amp,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
            grad_clip=args.grad_clip,
        )
        val_metrics, _ = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            scaler=scaler,
            train=False,
            amp=args.amp,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
            grad_clip=args.grad_clip,
        )
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        validation_loss = val_metrics["loss"]
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            normalizer,
            train_config,
            epoch,
            global_step,
            validation_loss,
            optimizer=optimizer,
        )
        if validation_loss < best_val:
            best_val = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                normalizer,
                train_config,
                epoch,
                global_step,
                validation_loss,
            )
        else:
            epochs_without_improvement += 1
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                normalizer,
                train_config,
                epoch,
                global_step,
                validation_loss,
            )
        print(
            f"epoch={epoch:03d} done seconds={entry['elapsed_sec']:.1f} best_epoch={best_epoch} "
            f"best_val={best_val:.6f}",
            flush=True,
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"[EARLY_STOP] patience={args.early_stop_patience} best_epoch={best_epoch}", flush=True)
            break

    summary = {
        "model_type": act.MODEL_TYPE,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "global_step": global_step,
        "num_episodes_total": len(records),
        "num_episodes_train": len(train_records),
        "num_episodes_val": len(val_records),
        "yaw_degrees": [train_config["yaw_min_degrees"], train_config["yaw_max_degrees"]],
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
