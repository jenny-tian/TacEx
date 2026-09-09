#!/usr/bin/env python3
"""Train a genuine DINOv3-conditioned Flow Matching BC for GrapePick RGB records.

The policy uses two frozen DINOv3 image streams (wrist and third-person), two
steps of robot proprioception plus bilateral pad-normal loads, and predicts an
8-step CAFE action chunk with conditional flow matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
BC_TRAINING_DIR = REPO_ROOT / "scripts" / "bc_training"
if str(BC_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(BC_TRAINING_DIR))

import dinov3_flow as flow  # noqa: E402
import train_bc as common  # noqa: E402


PICK_VARIANT = os.environ.get("TACEX_PICK_VARIANT", "grape").strip().lower()
if PICK_VARIANT not in {"grape", "raspberry"}:
    raise ValueError(
        "TACEX_PICK_VARIANT must be either 'grape' or 'raspberry', "
        f"got {PICK_VARIANT!r}"
    )
if PICK_VARIANT == "raspberry":
    CONTRACT = "raspberry_dinov3_tactile_flow_matching_bc_v1"
    TASK = "TacEx-RaspberryPick-Force-v0"
    BEST_CHECKPOINT_FILENAME = "raspberry_dinov3_flow_bc.pt"
else:
    CONTRACT = "grape_dinov3_tactile_flow_matching_bc_v1"
    TASK = "TacEx-GrapePick-Force-v0"
    BEST_CHECKPOINT_FILENAME = "grape_dinov3_flow_bc.pt"
CACHE_VERSION = 1
RAW_IMAGE_KEYS = ("rgb", "rgb_third")
DEPLOYMENT_IMAGE_KEYS = ("wrist", "third")
STATE_FIELDS = ("robot0_pos[10]", "finger_normal_force_n[2]")


@dataclass(frozen=True)
class Episode:
    path: Path
    name: str
    state: np.ndarray
    action: np.ndarray
    step: np.ndarray
    length: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_state(trajectory: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return deployable proprioception/tactile state without privileged object pose."""

    robot = np.asarray(trajectory["robot0_pos"], dtype=np.float32)
    pad_force = np.asarray(trajectory["finger_normal_force_n"], dtype=np.float32)
    if robot.ndim != 2 or robot.shape[1] != 10:
        raise ValueError(f"robot0_pos must have shape [T,10], got {robot.shape}")
    if pad_force.shape != (len(robot), 2):
        raise ValueError(
            f"finger_normal_force_n must have shape [{len(robot)},2], got {pad_force.shape}"
        )
    return np.concatenate((robot, pad_force), axis=-1).astype(np.float32, copy=False)


def discover_episodes(data_root: Path, max_episodes: int | None) -> list[Episode]:
    episodes: list[Episode] = []
    for record_dir in sorted(data_root.glob("record_*")):
        trajectory_path = record_dir / "trajectory.npz"
        metadata_path = record_dir / "metadata.json"
        if not trajectory_path.is_file() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("task") != TASK or not metadata.get("success", False):
            continue
        if not metadata.get("wrist_rgb_recorded", False) or not metadata.get(
            "scene_rgb_recorded", False
        ):
            continue
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            missing = [key for key in (*RAW_IMAGE_KEYS, "action", "step") if key not in trajectory]
            if missing:
                raise KeyError(f"{trajectory_path} is missing {missing}")
            state = trajectory_state(trajectory)
            action = np.asarray(trajectory["action"], dtype=np.float32)
            step = np.asarray(trajectory["step"], dtype=np.int64)
            image_shapes = {key: trajectory[key].shape for key in RAW_IMAGE_KEYS}
        if action.shape != (len(state), 10):
            raise ValueError(f"{trajectory_path}: action shape {action.shape}")
        if step.shape != (len(state),):
            raise ValueError(f"{trajectory_path}: step shape {step.shape}")
        for key, shape in image_shapes.items():
            if shape != (len(state), 224, 224, 3):
                raise ValueError(f"{trajectory_path}: {key} shape {shape}")
        episodes.append(
            Episode(
                path=trajectory_path,
                name=record_dir.name,
                state=state,
                action=action,
                step=step,
                length=len(state),
            )
        )
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if len(episodes) < 2:
        raise ValueError(
            f"Need at least two successful RGB {PICK_VARIANT.title()}Pick episodes "
            f"under {data_root}; "
            f"found {len(episodes)}"
        )
    return episodes


def compute_normalizer(
    episodes: list[Episode], mode: str
) -> common.LinearNormalizer:
    return common.LinearNormalizer(
        {
            common.STATE_KEY: common.ArrayStats.from_array(
                np.concatenate([episode.state for episode in episodes], axis=0)
            ),
            common.ACTION_KEY: common.ArrayStats.from_array(
                np.concatenate([episode.action for episode in episodes], axis=0)
            ),
        },
        mode=mode,
    )


def cache_signature(
    dino_path: Path, feature_grid_size: int, dino_hidden_size: int
) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "dino_path": str(dino_path.resolve()),
        "image_keys": list(RAW_IMAGE_KEYS),
        "feature_grid_size": int(feature_grid_size),
        "num_spatial_tokens": int(feature_grid_size**2),
        "dino_hidden_size": int(dino_hidden_size),
        "dtype": "float16",
    }


def cache_valid(
    cache_root: Path, episode: Episode, signature: dict[str, Any]
) -> bool:
    record_cache = cache_root / episode.name
    try:
        metadata = json.loads((record_cache / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if metadata.get("signature") != signature or metadata.get("length") != episode.length:
        return False
    shape = (
        episode.length,
        int(signature["num_spatial_tokens"]),
        int(signature["dino_hidden_size"]),
    )
    for key in RAW_IMAGE_KEYS:
        try:
            array = np.load(record_cache / f"{key}.npy", mmap_mode="r")
        except (FileNotFoundError, ValueError):
            return False
        if array.shape != shape or array.dtype != np.float16:
            return False
    return True


@torch.inference_mode()
def precompute_dino_features(
    episodes: list[Episode],
    cache_root: Path,
    dino: common.MinimalDINOv3ViT,
    dino_path: Path,
    feature_grid_size: int,
    batch_size: int,
    device: torch.device,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(dino_path, feature_grid_size, dino.config.hidden_size)
    pending = [episode for episode in episodes if not cache_valid(cache_root, episode, signature)]
    print(
        f"[CACHE] valid={len(episodes) - len(pending)} pending={len(pending)} root={cache_root}",
        flush=True,
    )
    dino = dino.to(device).eval()
    mean = torch.tensor(common.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(common.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    for episode_index, episode in enumerate(pending, start=1):
        record_cache = cache_root / episode.name
        record_cache.mkdir(parents=True, exist_ok=True)
        with np.load(episode.path, allow_pickle=False) as trajectory:
            for key in RAW_IMAGE_KEYS:
                images = np.asarray(trajectory[key], dtype=np.uint8)
                partial = record_cache / f".{key}.partial.npy"
                output = np.lib.format.open_memmap(
                    partial,
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        episode.length,
                        feature_grid_size**2,
                        dino.config.hidden_size,
                    ),
                )
                for start in range(0, episode.length, batch_size):
                    stop = min(start + batch_size, episode.length)
                    image_batch = torch.from_numpy(np.array(images[start:stop], copy=True))
                    image_batch = image_batch.permute(0, 3, 1, 2).to(
                        device=device, dtype=torch.float32
                    )
                    image_batch = image_batch.div_(255.0)
                    image_batch = (image_batch - mean) / std
                    with torch.autocast(
                        device_type=device.type,
                        enabled=device.type == "cuda",
                        dtype=torch.float16,
                    ):
                        patches = dino(image_batch)
                    patch_count = patches.shape[1]
                    side = int(round(math.sqrt(patch_count)))
                    if side * side != patch_count:
                        raise ValueError(f"DINO returned {patch_count} non-square patch tokens")
                    pooled = F.adaptive_avg_pool2d(
                        patches.transpose(1, 2).reshape(-1, patches.shape[-1], side, side),
                        output_size=(feature_grid_size, feature_grid_size),
                    ).flatten(2).transpose(1, 2)
                    output[start:stop] = pooled.float().cpu().numpy().astype(np.float16)
                output.flush()
                del output
                os.replace(partial, record_cache / f"{key}.npy")
        metadata = {
            "record": str(episode.path.parent.resolve()),
            "length": episode.length,
            "signature": signature,
        }
        (record_cache / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[CACHE] {episode_index}/{len(pending)} {episode.name} frames={episode.length}",
            flush=True,
        )


class GrapeDINOSequenceDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        normalizer: common.LinearNormalizer,
        cache_root: Path,
        *,
        state_obs_steps: int,
        image_obs_steps: int,
        chunk_size: int,
        max_episode_steps: int,
    ) -> None:
        self.episodes = episodes
        self.normalizer = normalizer
        self.cache_root = cache_root
        self.state_obs_steps = int(state_obs_steps)
        self.image_obs_steps = int(image_obs_steps)
        self.chunk_size = int(chunk_size)
        self.max_episode_steps = int(max_episode_steps)
        self.samples = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(episodes)
            for frame_index in range(episode.length)
        ]
        self._feature_cache: dict[int, dict[str, np.ndarray]] = {}
        self.state_dim = 12
        self.action_dim = 10
        self.num_cameras = len(RAW_IMAGE_KEYS)

    def __len__(self) -> int:
        return len(self.samples)

    def _features(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index not in self._feature_cache:
            record_cache = self.cache_root / self.episodes[episode_index].name
            self._feature_cache[episode_index] = {
                key: np.load(record_cache / f"{key}.npy", mmap_mode="r")
                for key in RAW_IMAGE_KEYS
            }
        return self._feature_cache[episode_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        state_indices = common.clamp_indices(
            episode.length,
            np.arange(frame_index - self.state_obs_steps + 1, frame_index + 1),
        )
        image_indices = common.clamp_indices(
            episode.length,
            np.arange(frame_index - self.image_obs_steps + 1, frame_index + 1),
        )
        requested_action_indices = np.arange(frame_index, frame_index + self.chunk_size)
        action_is_pad = requested_action_indices >= episode.length
        action_indices = common.clamp_indices(episode.length, requested_action_indices)
        state = self.normalizer.normalize_numpy(
            common.STATE_KEY, episode.state[state_indices]
        )
        action = self.normalizer.normalize_numpy(
            common.ACTION_KEY, episode.action[action_indices]
        )
        cached = self._features(episode_index)
        features = np.stack(
            [cached[key][image_indices] for key in RAW_IMAGE_KEYS], axis=1
        )
        phase = float(episode.step[frame_index]) / float(self.max_episode_steps)
        return {
            "obs": {
                common.STATE_KEY: torch.from_numpy(state),
                "dino_features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
                "phase": torch.tensor([min(max(phase, 0.0), 1.0)], dtype=torch.float32),
            },
            common.ACTION_KEY: torch.from_numpy(action),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }


class LearnedEMA:
    def __init__(self, model: flow.DINOv3FlowPolicy, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone() for key, value in flow.policy_state_dict(model).items()
        }

    @torch.no_grad()
    def step(self, model: flow.DINOv3FlowPolicy) -> None:
        current = flow.policy_state_dict(model)
        for key, averaged in self.shadow.items():
            value = current[key].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                averaged.copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "averaged_model": self.shadow}


def run_epoch(
    model: flow.DINOv3FlowPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    ema: LearnedEMA | None,
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
        batch = common.move_to_device(batch, device)
        if train:
            assert optimizer is not None
            learning_rate = common.warmup_cosine_lr(
                global_step, total_steps, warmup_steps, base_lr
            )
            common.set_lr(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
                dtype=torch.float16,
            ):
                losses = model.compute_loss(batch)
                loss = losses["loss"]
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.step(model)
                global_step += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    metrics = {key: value / max(batches, 1) for key, value in totals.items()}
    split = "train" if train else "val"
    print(
        f"epoch={epoch:03d} split={split} batches={batches} "
        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
        flush=True,
    )
    return metrics, global_step


def save_checkpoint(
    path: Path,
    model: flow.DINOv3FlowPolicy,
    normalizer: common.LinearNormalizer,
    train_config: dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    validation_loss: float,
    ema: LearnedEMA | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
        "contract": CONTRACT,
        "contract_version": 1,
        "task": TASK,
        "model_type": flow.MODEL_TYPE,
        "model": flow.policy_state_dict(model),
        "normalizer": normalizer.state_dict(),
        "policy_config": model.config.to_dict(),
        "train_config": train_config,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_loss": float(validation_loss),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=256)
    parser.add_argument("--max-episode-steps", type=int, default=790)
    parser.add_argument("--dino-path", type=Path, default=None)
    parser.add_argument("--feature-grid-size", type=int, default=7)
    parser.add_argument("--condition-grid-size", type=int, default=4)
    parser.add_argument("--cache-batch-size", type=int, default=128)
    parser.add_argument("--state-obs-steps", type=int, default=2)
    parser.add_argument("--image-obs-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=384)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=384)
    parser.add_argument("--transformer-cond-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--time-embed-scale", type=float, default=1000.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--ode-solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--no-clip-sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--normalizer-mode", choices=("limits", "standard"), default="limits")
    parser.add_argument("--seed", type=int, default=908)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common.seed_everything(args.seed)
    data_root = common.resolve_path(args.data_root)
    output_root = common.resolve_path(args.output_root)
    training_dir = output_root / "training"
    checkpoint_dir = output_root / "checkpoints"
    cache_root = output_root / "dino_feature_cache"
    training_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    episodes = discover_episodes(data_root, args.max_episodes)
    train_indices, validation_indices = common.split_episode_indices(
        len(episodes), args.val_ratio, args.seed
    )
    train_episodes = [episodes[int(index)] for index in train_indices]
    validation_episodes = [episodes[int(index)] for index in validation_indices]
    normalizer = compute_normalizer(train_episodes, args.normalizer_mode)

    dino, dino_path = common.load_dino_for_config(args.dino_path)
    precompute_dino_features(
        episodes,
        cache_root,
        dino,
        dino_path,
        args.feature_grid_size,
        args.cache_batch_size,
        device,
    )
    dino = dino.cpu()
    torch.cuda.empty_cache()

    dataset_kwargs = {
        "normalizer": normalizer,
        "cache_root": cache_root,
        "state_obs_steps": args.state_obs_steps,
        "image_obs_steps": args.image_obs_steps,
        "chunk_size": args.chunk_size,
        "max_episode_steps": args.max_episode_steps,
    }
    train_dataset = GrapeDINOSequenceDataset(train_episodes, **dataset_kwargs)
    validation_dataset = GrapeDINOSequenceDataset(validation_episodes, **dataset_kwargs)
    config = flow.DINOv3FlowConfig(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        num_cameras=train_dataset.num_cameras,
        state_obs_steps=args.state_obs_steps,
        image_obs_steps=args.image_obs_steps,
        chunk_size=args.chunk_size,
        feature_grid_size=args.feature_grid_size,
        condition_grid_size=args.condition_grid_size,
        dino_hidden_size=dino.config.hidden_size,
        cond_dim=args.cond_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dim=args.transformer_dim,
        transformer_cond_layers=args.transformer_cond_layers,
        dropout=args.dropout,
        time_embed_scale=args.time_embed_scale,
        num_inference_steps=args.num_inference_steps,
        ode_solver=args.ode_solver,
        clip_sample=not args.no_clip_sample,
        phase_horizon_steps=args.max_episode_steps,
        dino_path=str(dino_path.resolve()),
        image_keys=DEPLOYMENT_IMAGE_KEYS,
    )
    model = flow.DINOv3FlowPolicy(config, dino).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = LearnedEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_config = common.json_ready(vars(args))
    train_config.update(
        {
            "contract": CONTRACT,
            "task": TASK,
            "model_type": flow.MODEL_TYPE,
            "data_root": str(data_root),
            "dino_path": str(dino_path.resolve()),
            "dino_frozen": True,
            "raw_image_keys": list(RAW_IMAGE_KEYS),
            "deployment_image_keys": list(DEPLOYMENT_IMAGE_KEYS),
            "state_fields": list(STATE_FIELDS),
            "privileged_object_pose_used": False,
            "num_episodes_total": len(episodes),
            "num_episodes_train": len(train_episodes),
            "num_episodes_val": len(validation_episodes),
            "train_record_names": [episode.name for episode in train_episodes],
            "val_record_names": [episode.name for episode in validation_episodes],
            "train_samples": len(train_dataset),
            "val_samples": len(validation_dataset),
            "policy_config": config.to_dict(),
            "trainable_parameters": common.count_parameters(model),
        }
    )
    (training_dir / "config.json").write_text(
        json.dumps(train_config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: train_config[key]
                for key in (
                    "contract",
                    "task",
                    "model_type",
                    "num_episodes_total",
                    "num_episodes_train",
                    "num_episodes_val",
                    "train_samples",
                    "val_samples",
                    "trainable_parameters",
                )
            },
            indent=2,
        ),
        flush=True,
    )

    total_steps = max(len(train_loader), 1) * args.epochs
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    log_path = training_dir / "logs.jsonl"
    if log_path.exists():
        log_path.unlink()
    best_path = checkpoint_dir / BEST_CHECKPOINT_FILENAME
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            train=True,
            amp=args.amp,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
            grad_clip=args.grad_clip,
        )
        validation_metrics, global_step = run_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
            scaler=scaler,
            ema=None,
            train=False,
            amp=args.amp,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            base_lr=args.lr,
            grad_clip=args.grad_clip,
        )
        validation_loss = validation_metrics["loss"]
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": time.time() - started,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            normalizer,
            train_config,
            epoch=epoch,
            global_step=global_step,
            validation_loss=validation_loss,
            ema=ema,
            optimizer=optimizer,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                normalizer,
                train_config,
                epoch=epoch,
                global_step=global_step,
                validation_loss=validation_loss,
                ema=ema,
            )
        else:
            epochs_without_improvement += 1
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model,
                normalizer,
                train_config,
                epoch=epoch,
                global_step=global_step,
                validation_loss=validation_loss,
                ema=ema,
            )
        print(
            f"epoch={epoch:03d} done seconds={entry['elapsed_sec']:.1f} "
            f"best_epoch={best_epoch} best_val={best_validation:.6f}",
            flush=True,
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"[EARLY_STOP] best_epoch={best_epoch}", flush=True)
            break

    summary = {
        "contract": CONTRACT,
        "task": TASK,
        "model_type": flow.MODEL_TYPE,
        "dino_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dino_frozen": True,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "global_step": global_step,
        "num_episodes_total": len(episodes),
        "num_episodes_train": len(train_episodes),
        "num_episodes_val": len(validation_episodes),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
    }
    (training_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
