from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from sim_robot.common.ema import EMAModel
from sim_robot.data.sequence_dataset import build_datasets
from sim_robot.policy.flow_matching_policy import SimFlowMatchingConfig, SimFlowMatchingPolicy, load_checkpoint


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def run_epoch(
    model: SimFlowMatchingPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMAModel | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int = 0,
    train: bool = True,
    max_steps: int | None = None,
    base_lr: float = 1e-4,
    total_steps: int = 1,
    warmup_steps: int = 0,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    model.train(train)
    sums: dict[str, float] = {}
    n_batches = 0
    desc = f"epoch {epoch:04d} train" if train else f"epoch {epoch:04d} val"
    iterator = tqdm(loader, desc=desc, leave=False)
    for local_step, batch in enumerate(iterator):
        if max_steps is not None and local_step >= max_steps:
            break
        batch = move_to_device(batch, device)
        if train:
            assert optimizer is not None
            lr = warmup_cosine_lr(global_step, total_steps, warmup_steps, base_lr)
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.cuda.amp.autocast():
                    loss_dict = model.compute_loss(batch)
                    loss = loss_dict["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model.compute_loss(batch)
                loss = loss_dict["loss"]
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        if train and ema is not None:
            ema.step(model)
        values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
        n_batches += 1
        iterator.set_postfix(loss=f"{values['loss']:.5f}")
        if train:
            global_step += 1

    mean = {key: value / max(n_batches, 1) for key, value in sums.items()}
    return mean, global_step


def save_checkpoint(
    path: Path,
    model: SimFlowMatchingPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    normalizer_state: dict,
    config: dict,
    ema: EMAModel | None = None,
) -> None:
    checkpoint = {
        "model_type": "sim_single_camera_flow_matching_cross_action",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "normalizer": normalizer_state,
        "policy_config": model.config.to_dict(),
        "train_config": config,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, path)


def _normalizer_center_scale(state: dict, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    stats = state["stats"][key]
    center = torch.as_tensor(stats["mean"], dtype=torch.float32)
    scale = torch.as_tensor(stats["std"], dtype=torch.float32)
    if state.get("mode", "limits") != "standard":
        lower = torch.as_tensor(stats["min"], dtype=torch.float32)
        upper = torch.as_tensor(stats["max"], dtype=torch.float32)
        span = upper - lower
        eps = state.get("range_eps", 1e-4)
        center = torch.where(span < eps, lower, (upper + lower) / 2.0)
        scale = torch.where(span < eps, torch.ones_like(scale), span / 2.0)
    eps = state.get("range_eps", 1e-4)
    return center, torch.where(scale < eps, torch.ones_like(scale), scale)


def initialize_from_checkpoint(
    model: SimFlowMatchingPolicy,
    checkpoint_path: Path,
    normalizer_state: dict | None = None,
    preserve_normalization: bool = False,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source = checkpoint.get("ema", {}).get("averaged_model", checkpoint["model"])
    source_normalizer = checkpoint.get("normalizer")
    target = model.state_dict()
    compatible = {}
    adapted = []
    skipped = []
    for name, value in source.items():
        if name.startswith("obs_encoder.image_projector.") and len(model.config.image_keys) > 1:
            suffix = name.removeprefix("obs_encoder.image_projector.")
            mapped = []
            for camera_index in range(len(model.config.image_keys)):
                target_name = f"obs_encoder.image_projectors.{camera_index}.{suffix}"
                if target_name in target and target[target_name].shape == value.shape:
                    compatible[target_name] = value
                    mapped.append(target_name)
            if mapped:
                adapted.extend(mapped)
                continue
        if name not in target:
            skipped.append(name)
            continue
        if value.shape == target[name].shape:
            compatible[name] = value
            continue
        if (
            name == "obs_encoder.state_projector.0.weight"
            and value.ndim == 2
            and target[name].shape[0] == value.shape[0]
            and target[name].shape[1] == value.shape[1] + 1
        ):
            expanded = target[name].clone()
            expanded[:, : value.shape[1]] = value
            expanded[:, value.shape[1] :] = 0.0
            compatible[name] = expanded
            adapted.append(name)
            continue
        if (
            name == "obs_encoder.state_projector.0.weight"
            and value.ndim == 2
            and target[name].shape[0] == value.shape[0]
            and target[name].shape[1] + 1 == value.shape[1]
        ):
            # Removing the final conditioning feature (for example demonstration_mode)
            # should preserve all weights for the remaining physical state and phase.
            compatible[name] = value[:, : target[name].shape[1]]
            adapted.append(name)
            continue
        if (
            name == "obs_encoder.modality_emb"
            and value.ndim == 3
            and target[name].shape[0] == value.shape[0]
            and target[name].shape[2] == value.shape[2]
            and target[name].shape[1] > value.shape[1]
        ):
            expanded = target[name].clone()
            expanded[:, : value.shape[1]] = value
            expanded[:, value.shape[1] :] = value[:, -1:]
            compatible[name] = expanded
            adapted.append(name)
            continue
        if (
            name == "velocity_net.cond_pos_emb"
            and value.ndim == 3
            and target[name].shape[0] == value.shape[0]
            and target[name].shape[2] == value.shape[2]
            and target[name].shape[1] > value.shape[1]
        ):
            expanded = target[name].clone()
            expanded[:, : value.shape[1]] = value
            extra = target[name].shape[1] - value.shape[1]
            expanded[:, value.shape[1] :] = value[:, -extra:]
            compatible[name] = expanded
            adapted.append(name)
            continue
        skipped.append(name)

    if preserve_normalization:
        if normalizer_state is None or source_normalizer is None:
            raise ValueError("--preserve-init-normalization requires normalizer statistics in both checkpoints")
        old_state_center, old_state_scale = _normalizer_center_scale(source_normalizer, "robot0_pos")
        new_state_center, new_state_scale = _normalizer_center_scale(normalizer_state, "robot0_pos")
        old_action_center, old_action_scale = _normalizer_center_scale(source_normalizer, "action")
        new_action_center, new_action_scale = _normalizer_center_scale(normalizer_state, "action")

        # Map new normalized observations back to source-checkpoint coordinates.
        state_gain = new_state_scale / old_state_scale
        state_offset = (new_state_center - old_state_center) / old_state_scale
        state_weight_name = "obs_encoder.state_projector.0.weight"
        state_bias_name = "obs_encoder.state_projector.0.bias"
        if state_weight_name in compatible and state_bias_name in compatible:
            weight = compatible[state_weight_name]
            compatible[state_weight_name] = weight * state_gain.unsqueeze(0)
            compatible[state_bias_name] = compatible[state_bias_name] + weight @ state_offset
            adapted.extend([state_weight_name, state_bias_name])

        # Preserve physical action coordinates across the changed action limits.
        action_gain = old_action_scale / new_action_scale
        action_offset = (old_action_center - new_action_center) / new_action_scale
        input_weight_name = "velocity_net.input_emb.weight"
        input_bias_name = "velocity_net.input_emb.bias"
        if input_weight_name in compatible and input_bias_name in compatible:
            weight = compatible[input_weight_name]
            compatible[input_weight_name] = weight / action_gain.unsqueeze(0)
            compatible[input_bias_name] = compatible[input_bias_name] - weight @ (action_offset / action_gain)
            adapted.extend([input_weight_name, input_bias_name])

        output_weight_name = "velocity_net.head.weight"
        output_bias_name = "velocity_net.head.bias"
        if output_weight_name in compatible and output_bias_name in compatible:
            compatible[output_weight_name] = compatible[output_weight_name] * action_gain.unsqueeze(1)
            compatible[output_bias_name] = compatible[output_bias_name] * action_gain
            adapted.extend([output_weight_name, output_bias_name])

        visual_weight_name = "visual_xy_head.1.weight"
        visual_bias_name = "visual_xy_head.1.bias"
        if visual_weight_name in compatible and visual_bias_name in compatible:
            compatible[visual_weight_name] = compatible[visual_weight_name] * action_gain[:2].unsqueeze(1)
            compatible[visual_bias_name] = compatible[visual_bias_name] * action_gain[:2] + action_offset[:2]
            adapted.extend([visual_weight_name, visual_bias_name])
    result = model.load_state_dict(compatible, strict=False)
    print(
        f"[INFO] initialized from {checkpoint_path}: loaded={len(compatible)} "
        f"adapted={adapted} skipped={len(skipped)} missing={len(result.missing_keys)}"
    )


def initialize_pretrained_image_backbone(model: SimFlowMatchingPolicy, backbone_name: str) -> None:
    if backbone_name == "none":
        return
    if backbone_name != "resnet50_imagenet1k_v2":
        raise ValueError(f"Unsupported pretrained image backbone: {backbone_name}")

    pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    source = pretrained.state_dict()
    target_module = model.obs_encoder.image_encoder.backbone
    target = target_module.state_dict()
    compatible = {
        name: value for name, value in source.items() if name in target and value.shape == target[name].shape
    }
    result = target_module.load_state_dict(compatible, strict=False)
    print(
        f"[INFO] initialized image backbone from {backbone_name}: loaded={len(compatible)} "
        f"missing={len(result.missing_keys)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-camera sim action policy with flow matching.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-key", type=str, default="high")
    parser.add_argument("--state-key", type=str, default="robot0_pos")
    parser.add_argument("--image-key", type=str, default="robot0_image")
    parser.add_argument(
        "--image-keys",
        type=str,
        default="",
        help="Comma-separated image datasets for multi-camera fusion (for example robot0_image,robot0_image_third).",
    )
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--n-state-obs-steps", type=int, default=2)
    parser.add_argument("--n-image-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalizer-mode", choices=["standard", "gaussian", "limits"], default="limits")
    parser.add_argument("--image-feature-dim", type=int, default=512)
    parser.add_argument(
        "--image-normalization",
        choices=("none", "imagenet"),
        default="none",
        help="Input normalization applied inside the image encoder and stored in the checkpoint.",
    )
    parser.add_argument(
        "--pretrained-image-backbone",
        choices=("none", "resnet50_imagenet1k_v2"),
        default="none",
        help="Reset only the ResNet backbone after loading --init-checkpoint.",
    )
    parser.add_argument("--obs-feature-dim", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-embedding-dim", type=int, default=512)
    parser.add_argument("--transformer-cond-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--ode-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--time-embed-scale", type=float, default=1000.0)
    parser.add_argument("--no-clip-sample", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--include-phase", action="store_true", help="Append normalized episode progress to robot0_pos.")
    parser.add_argument(
        "--include-demo-mode",
        action="store_true",
        help="Append the position-failure/safe/overforce demonstration mode to robot0_pos.",
    )
    parser.add_argument("--safe-close-width-m", type=float, default=0.0065)
    parser.add_argument("--overforce-close-width-m", type=float, default=0.0015)
    parser.add_argument("--safe-sample-weight", type=float, default=1.0)
    parser.add_argument("--overforce-sample-weight", type=float, default=1.0)
    parser.add_argument("--position-failure-sample-weight", type=float, default=1.0)
    parser.add_argument("--close-projection-onset-width-m", type=float, default=0.02)
    parser.add_argument("--overforce-projection-phase", type=float, default=0.30)
    parser.add_argument("--position-failure-offset-m", type=float, default=0.03)
    parser.add_argument("--position-failure-projection-phase", type=float, default=0.30)
    parser.add_argument("--safe-demo-mode-probability", type=float, default=0.50)
    parser.add_argument("--overforce-demo-mode-probability", type=float, default=0.25)
    parser.add_argument("--position-failure-demo-mode-probability", type=float, default=0.25)
    parser.add_argument(
        "--visual-xy-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary image-only loss weight for predicting the first normalized target XY.",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Initialize matching weights from an existing checkpoint.")
    parser.add_argument(
        "--preserve-init-normalization",
        action="store_true",
        help="Remap initialized linear layers so source physical coordinates are preserved under this dataset's normalizer.",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_image_key = args.image_keys if args.image_keys.strip() else args.image_key
    resolved_image_keys = tuple(part.strip() for part in dataset_image_key.split(",") if part.strip())
    model_image_keys = resolved_image_keys if len(resolved_image_keys) > 1 else ("robot0_image",)

    train_set, val_set, normalizer = build_datasets(
        hdf5_path=args.dataset,
        n_state_obs_steps=args.n_state_obs_steps,
        n_image_obs_steps=args.n_image_obs_steps,
        n_action_steps=args.n_action_steps,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalizer_mode=args.normalizer_mode,
        action_key=args.action_key,
        state_key=args.state_key,
        image_key=dataset_image_key,
        success_only=args.success_only,
        cache_images=args.cache_images,
        include_phase=args.include_phase,
        include_demo_mode=args.include_demo_mode,
        overforce_close_width_m=args.overforce_close_width_m,
    )
    train_sampler = WeightedRandomSampler(
        train_set.sample_weights(
            safe_weight=args.safe_sample_weight,
            overforce_weight=args.overforce_sample_weight,
            position_failure_weight=args.position_failure_sample_weight,
        ),
        num_samples=len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    policy_config = SimFlowMatchingConfig(
        robot0_pos_dim=train_set.robot0_pos_dim,
        action_dim=train_set.action_dim,
        n_state_obs_steps=args.n_state_obs_steps,
        n_image_obs_steps=args.n_image_obs_steps,
        image_keys=model_image_keys,
        image_normalization=args.image_normalization,
        n_action_steps=args.n_action_steps,
        image_feature_dim=args.image_feature_dim,
        obs_feature_dim=args.obs_feature_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_embedding_dim=args.transformer_embedding_dim,
        transformer_cond_layers=args.transformer_cond_layers,
        dropout=args.dropout,
        time_embed_scale=args.time_embed_scale,
        num_inference_steps=args.num_inference_steps,
        ode_solver=args.ode_solver,
        clip_sample=not args.no_clip_sample,
        visual_xy_loss_weight=args.visual_xy_loss_weight,
    )
    model = SimFlowMatchingPolicy(policy_config).to(device)
    if args.init_checkpoint is not None:
        initialize_from_checkpoint(
            model,
            args.init_checkpoint.expanduser().resolve(),
            normalizer_state=normalizer.state_dict(),
            preserve_normalization=args.preserve_init_normalization,
        )
    initialize_pretrained_image_backbone(model, args.pretrained_image_backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))
    ema = None if args.no_ema else EMAModel(model, decay=args.ema_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    total_train_steps = max(len(train_loader), 1) * args.epochs
    config = vars(args).copy()
    config.update(
        {
            "model_type": "sim_single_camera_flow_matching_cross_action",
            "policy_config": policy_config.to_dict(),
            "num_parameters": count_parameters(model),
            "train_samples": len(train_set),
            "val_samples": 0 if val_set is None else len(val_set),
            "robot0_pos_dim": train_set.robot0_pos_dim,
            "action_dim": train_set.action_dim,
            "image_shape": train_set.image_shape,
            "image_shapes": train_set.image_shapes,
            "resolved_image_keys": resolved_image_keys,
            "freq_ratio": train_set.freq_ratio,
            "instruction": train_set.instruction,
            "labware": train_set.labware,
            "normalizes_all_action_dims": True,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                key: config[key]
                for key in [
                    "model_type",
                    "policy_config",
                    "num_parameters",
                    "train_samples",
                    "val_samples",
                    "action_key",
                    "normalizes_all_action_dims",
                ]
            },
            indent=2,
        )
    )

    best_val = float("inf")
    global_step = 0
    log_path = args.output_dir / "logs.jsonl"
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler if scaler.is_enabled() else None,
            epoch=epoch,
            train=True,
            max_steps=1 if args.debug else args.max_train_steps,
            base_lr=args.lr,
            total_steps=total_train_steps,
            warmup_steps=args.warmup_steps,
            global_step=global_step,
        )
        val_metrics = None
        eval_model = ema.averaged_model if ema is not None else model
        if val_loader is not None:
            val_metrics, _ = run_epoch(
                eval_model,
                val_loader,
                device,
                epoch=epoch,
                train=False,
                max_steps=1 if args.debug else args.max_val_steps,
            )

        elapsed = time.time() - start_time
        metric = train_metrics["loss"] if val_metrics is None else val_metrics["loss"]
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_sec": elapsed,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        msg = f"epoch {epoch:04d} train_loss={train_metrics['loss']:.6f}"
        if val_metrics is not None:
            msg += f" val_loss={val_metrics['loss']:.6f}"
        msg += f" time={elapsed:.1f}s"
        print(msg)

        save_checkpoint(
            args.output_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            global_step,
            normalizer.state_dict(),
            config,
            ema=ema,
        )
        if metric < best_val:
            best_val = metric
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                normalizer.state_dict(),
                config,
                ema=ema,
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                normalizer.state_dict(),
                config,
                ema=ema,
            )
        if args.debug:
            break
