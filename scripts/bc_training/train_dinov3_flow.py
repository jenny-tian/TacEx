#!/usr/bin/env python
"""Train the DSRL-compatible DINOv3 Flow Matching policy on LabPick records."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import dinov3_flow as flow
import train_bc as common
from train_dinov3_act import (
    ACTRecordsDataset,
    move_to_device,
    precompute_dino_features,
    record_reset_yaw_degrees,
    record_succeeded,
    select_records,
)


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
        batch = move_to_device(batch, device)
        if train:
            assert optimizer is not None
            lr = common.warmup_cosine_lr(global_step, total_steps, warmup_steps, base_lr)
            common.set_lr(optimizer, lr)
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
    label = "train" if train else "val"
    print(
        f"epoch={epoch:03d} split={label} batches={batches} "
        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
        flush=True,
    )
    return metrics, global_step


def save_checkpoint(
    path: Path,
    model: flow.DINOv3FlowPolicy,
    normalizer: common.LinearNormalizer,
    train_config: dict[str, Any],
    epoch: int,
    global_step: int,
    validation_loss: float,
    ema: LearnedEMA | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--image-keys", default="rgb,rgb_third")
    parser.add_argument("--deployment-image-keys", default="robot0_image,robot0_image_third")
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--normalizer-mode", choices=("limits", "standard"), default="limits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = common.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_keys = common.parse_csv(args.image_keys)
    deployment_image_keys = common.parse_csv(args.deployment_image_keys)
    if len(image_keys) != len(deployment_image_keys):
        raise ValueError("--image-keys and --deployment-image-keys must have the same length")
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
    lengths = [common.record_length(record, image_keys) for record in records]
    policy_config = flow.DINOv3FlowConfig(
        state_dim=train_set.state_dim,
        action_dim=train_set.action_dim,
        num_cameras=train_set.num_cameras,
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
        phase_horizon_steps=int(round(float(np.median(lengths)) - 1)),
        dino_path=str(dino_path.resolve()),
        image_keys=tuple(deployment_image_keys),
    )
    model = flow.DINOv3FlowPolicy(policy_config, dino).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    ema = LearnedEMA(model, args.ema_decay) if args.ema_decay > 0 else None
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
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_config = common.json_ready(vars(args))
    train_config.update(
        {
            "model_type": flow.MODEL_TYPE,
            "data_root": str(common.resolve_path(args.data_root)),
            "feature_cache_dir": None if cache_dir is None else str(cache_dir),
            "data_image_keys": image_keys,
            "image_keys": deployment_image_keys,
            "image_shape": list(train_set.image_shape),
            "image_shapes": {key: list(train_set.image_shape) for key in deployment_image_keys},
            "include_phase": True,
            "include_demo_mode": False,
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
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(train_config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: train_config[key] for key in (
                "model_type",
                "num_episodes_total",
                "num_episodes_train",
                "num_episodes_val",
                "train_samples",
                "val_samples",
                "yaw_min_degrees",
                "yaw_max_degrees",
                "trainable_parameters",
            )},
            indent=2,
        ),
        flush=True,
    )

    total_steps = max(len(train_loader), 1) * args.epochs
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    logs = output_dir / "logs.jsonl"
    if logs.exists():
        logs.unlink()
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
        val_metrics, _ = run_epoch(
            model,
            val_loader,
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
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        with logs.open("a", encoding="utf-8") as stream:
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
            ema,
            optimizer,
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
                ema,
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
                ema,
            )
        print(
            f"epoch={epoch:03d} done seconds={entry['elapsed_sec']:.1f} "
            f"best_epoch={best_epoch} best_val={best_val:.6f}",
            flush=True,
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"[EARLY_STOP] best_epoch={best_epoch}", flush=True)
            break
    summary = {
        "model_type": flow.MODEL_TYPE,
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
