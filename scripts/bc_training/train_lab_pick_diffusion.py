#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLOW_CONFIG = REPO_ROOT / "outputs/lab_pick_dinov3_flow_bc200_yaw0/config.json"
DEFAULT_LEROBOT_SOURCE = REPO_ROOT / "scripts/lerobot/src"
DEFAULT_DPPO_REVISION = "fa5847a9853aca9e8d5aaa3e2836e025ed8cbf97"
MATCHED_CAMERA_SHAPE = (224, 224)
DIFFUSION_BC_CAMERA_CONTRACT = "matched_full_frame_visual_xy_residual_224x224_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the matched LeRobot Diffusion BC used by the tactile-DPPO experiment."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/tacex_lab_pick_slide")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--action-steps",
        type=int,
        default=31,
        help="Executed actions; LeRobot reserves the first of 32 targets for the older observation.",
    )
    parser.add_argument("--down-dims", type=int, nargs="+", default=[112, 224, 448])
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--noise-scheduler", choices=("DDPM", "DDIM"), default="DDPM")
    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument(
        "--oversample-first-n-frames",
        type=int,
        default=8,
        help="Number of reset/early-approach frames per training episode to oversample.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=16,
        help="Total sampling multiplicity for the selected early frames.",
    )
    parser.add_argument(
        "--state-mask-indices",
        type=int,
        nargs="*",
        default=[0, 1],
        help="Normalized proprioceptive coordinates masked together during training.",
    )
    parser.add_argument(
        "--state-mask-probability",
        type=float,
        default=0.5,
        help="Per-sample probability of masking the selected state coordinates.",
    )
    parser.add_argument("--flow-config", type=Path, default=DEFAULT_FLOW_CONFIG)
    parser.add_argument(
        "--lerobot-source",
        type=Path,
        default=DEFAULT_LEROBOT_SOURCE,
        help="Vendored LeRobot source tree used for both training and Isaac checkpoint loading.",
    )
    parser.add_argument(
        "--use-matched-flow-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train only on the 180 records frozen in the matched Flow BC configuration.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--crop-shape",
        type=int,
        nargs=2,
        default=MATCHED_CAMERA_SHAPE,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Diffusion RGB crop. The matched LabPick contract fixes this to the full "
            "224x224 camera view so reset-time objects near an image edge remain visible."
        ),
    )
    parser.add_argument(
        "--crop-is-random",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must remain false for the matched full-frame camera contract.",
    )
    parser.add_argument(
        "--pretrained-backbone-weights",
        default=None,
        help=(
            "Optional torchvision weight enum, for example "
            "ResNet18_Weights.IMAGENET1K_V1."
        ),
    )
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav"], default="torchcodec")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest checkpoint in --output-dir and write a separate resume manifest.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_index(name: str) -> int:
    prefix = "record_"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise ValueError(f"Invalid matched record name: {name!r}.")
    return int(name[len(prefix) :])


def matched_split(flow_config: Path) -> tuple[list[int], list[int], dict[str, Any]]:
    flow_config = flow_config.expanduser().resolve()
    payload = json.loads(flow_config.read_text(encoding="utf-8"))
    train_names = list(payload.get("train_record_names", []))
    validation_names = list(payload.get("val_record_names", []))
    train = [_record_index(str(name)) for name in train_names]
    validation = [_record_index(str(name)) for name in validation_names]
    if len(train) != 180 or len(validation) != 20:
        raise ValueError(
            f"Matched Flow split must contain 180/20 records, got {len(train)}/{len(validation)}."
        )
    if set(train) & set(validation) or set(train) | set(validation) != set(range(200)):
        raise ValueError("Matched Flow split must be a disjoint partition of record_000000..record_000199.")
    return train, validation, payload


def clean_virtualgl_env(lerobot_source: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("VGL_ISACTIVE", None)
    env.pop("VGL_DISPLAY", None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(lerobot_source)
        if not existing_pythonpath
        else os.pathsep.join((str(lerobot_source), existing_pythonpath))
    )
    return env


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    lerobot_source = args.lerobot_source.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root; missing {info_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    lerobot_train_module = lerobot_source / "lerobot/scripts/lerobot_train.py"
    lerobot_dataset_module = lerobot_source / "lerobot/datasets/lerobot_dataset.py"
    lerobot_factory_module = lerobot_source / "lerobot/datasets/factory.py"
    lerobot_pyproject = lerobot_source.parent / "pyproject.toml"
    if not all(
        path.is_file()
        for path in (
            lerobot_train_module,
            lerobot_dataset_module,
            lerobot_factory_module,
            lerobot_pyproject,
        )
    ):
        raise FileNotFoundError(
            "Invalid vendored LeRobot source; expected "
            f"{lerobot_train_module}, {lerobot_dataset_module}, "
            f"{lerobot_factory_module}, and {lerobot_pyproject}."
        )
    if args.horizon < 1 or args.action_steps < 1:
        raise ValueError("Horizon and action steps must be positive.")
    if args.action_steps > args.horizon - 1:
        raise ValueError(
            "With the matched two-observation input, LeRobot requires action_steps <= horizon - 1."
        )
    crop_shape = tuple(args.crop_shape)
    if crop_shape != MATCHED_CAMERA_SHAPE or args.crop_is_random:
        raise ValueError(
            "Matched Diffusion BC must use the deterministic full 224x224 camera view; "
            f"received crop_shape={crop_shape}, crop_is_random={args.crop_is_random}."
        )
    if args.oversample_first_n_frames < 0 or args.oversample_factor < 1:
        raise ValueError("Invalid early-frame oversampling configuration.")
    if not 0.0 <= args.state_mask_probability <= 1.0:
        raise ValueError("state-mask-probability must lie in [0, 1].")
    if any(index < 0 or index >= 10 for index in args.state_mask_indices):
        raise ValueError("state-mask-indices must refer to the 10-D observation.state vector.")

    train_episodes: list[int] | None = None
    validation_episodes: list[int] = []
    flow_config_payload: dict[str, Any] | None = None
    if args.use_matched_flow_split:
        train_episodes, validation_episodes, flow_config_payload = matched_split(args.flow_config)

    resume_config: Path | None = None
    if args.resume:
        resume_config = output_dir / "checkpoints/last/pretrained_model/train_config.json"
        if not resume_config.is_file():
            raise FileNotFoundError(f"Cannot resume without {resume_config}")
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={resume_config}",
            "--resume=true",
            f"--steps={args.steps}",
            f"--save_freq={args.save_freq}",
            f"--num_workers={args.num_workers}",
            "--wandb.enable=false",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            "--policy.type=diffusion",
            f"--policy.device={args.device}",
            "--policy.push_to_hub=false",
            f"--policy.use_amp={str(args.amp).lower()}",
            f"--policy.use_group_norm={str(args.group_norm).lower()}",
            f"--policy.noise_scheduler_type={args.noise_scheduler}",
            f"--policy.num_train_timesteps={args.num_train_timesteps}",
            f"--policy.num_inference_steps={args.inference_steps}",
            f"--policy.horizon={args.horizon}",
            f"--policy.n_action_steps={args.action_steps}",
            f"--policy.drop_n_last_frames={args.horizon - args.action_steps - 1}",
            f"--policy.crop_shape=[{crop_shape[0]},{crop_shape[1]}]",
            f"--policy.crop_is_random={str(args.crop_is_random).lower()}",
            "--policy.use_visual_xy_residual=true",
            "--policy.visual_xy_camera_index=-1",
            "--policy.visual_xy_head_hidden_dim=256",
            "--policy.visual_xy_loss_weight=2.0",
            "--policy.visual_xy_smooth_l1_beta=0.05",
            f"--oversample_first_n_frames={args.oversample_first_n_frames}",
            f"--oversample_factor={args.oversample_factor}",
            f"--state_mask_indices=[{','.join(str(value) for value in args.state_mask_indices)}]",
            f"--state_mask_probability={args.state_mask_probability}",
            f"--dataset.repo_id={args.repo_id}",
            f"--dataset.root={dataset_root}",
            f"--dataset.video_backend={args.video_backend}",
            f"--output_dir={output_dir}",
            f"--steps={args.steps}",
            f"--batch_size={args.batch_size}",
            f"--num_workers={args.num_workers}",
            f"--save_freq={args.save_freq}",
            f"--policy.down_dims=[{','.join(str(value) for value in args.down_dims)}]",
            f"--seed={args.seed}",
            "--wandb.enable=false",
        ]
        if args.pretrained_backbone_weights is not None:
            command.append(
                f"--policy.pretrained_backbone_weights={args.pretrained_backbone_weights}"
            )
        if train_episodes is not None:
            command.append(
                f"--dataset.episodes=[{','.join(str(value) for value in train_episodes)}]"
            )

    launch_manifest_path = output_dir.with_name(
        output_dir.name + (".resume.launch.json" if args.resume else ".launch.json")
    )
    launch_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "matched_diffusion_bc_for_tactile_dppo",
        "dataset_root": str(dataset_root),
        "dataset_info_sha256": _sha256(info_path),
        "lerobot_source": str(lerobot_source),
        "lerobot_pyproject_sha256": _sha256(lerobot_pyproject),
        "lerobot_train_sha256": _sha256(lerobot_train_module),
        "lerobot_dataset_sha256": _sha256(lerobot_dataset_module),
        "lerobot_factory_sha256": _sha256(lerobot_factory_module),
        "flow_config": str(args.flow_config.expanduser().resolve()),
        "flow_config_sha256": (
            _sha256(args.flow_config.expanduser().resolve())
            if args.use_matched_flow_split
            else None
        ),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "train_records": None if flow_config_payload is None else flow_config_payload["train_record_names"],
        "validation_records": None if flow_config_payload is None else flow_config_payload["val_record_names"],
        "normalization_statistics_scope": (
            "selected_training_episodes_only" if train_episodes is not None else "full_dataset"
        ),
        "scheduler": args.noise_scheduler,
        "num_train_timesteps": args.num_train_timesteps,
        "num_inference_steps": args.inference_steps,
        "horizon": args.horizon,
        "n_obs_steps": 2,
        "n_action_steps": args.action_steps,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "down_dims": list(args.down_dims),
        "amp": args.amp,
        "use_group_norm": args.group_norm,
        "camera_input_shape": list(MATCHED_CAMERA_SHAPE),
        "crop_shape": list(crop_shape),
        "crop_is_random": args.crop_is_random,
        "camera_contract": DIFFUSION_BC_CAMERA_CONTRACT,
        "action_coordinate_contract": "visual_xy_anchor_plus_diffusion_residual_v1",
        "visual_xy_camera": "latest_third_person_image_only",
        "visual_xy_loss": "2.0 * smooth_l1(normalized_current_action_xy, beta=0.05)",
        "oversample_first_n_frames": args.oversample_first_n_frames,
        "oversample_factor": args.oversample_factor,
        "state_mask_indices": list(args.state_mask_indices),
        "state_mask_probability": args.state_mask_probability,
        "observation_identifiability_contract": "early_frame_oversampling_and_xy_modality_dropout_v1",
        "pretrained_backbone_weights": args.pretrained_backbone_weights,
        "resume": args.resume,
        "resume_config": None if resume_config is None else str(resume_config),
        "dppo_reference": "Ren et al., Diffusion Policy Policy Optimization, arXiv:2409.00588",
        "dppo_reference_code_revision": DEFAULT_DPPO_REVISION,
        "command": command,
        "status": "dry_run" if args.dry_run else "started",
    }
    _write_json(launch_manifest_path, launch_manifest)
    print("[COMMAND]", " ".join(command))
    if args.dry_run:
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, env=clean_virtualgl_env(lerobot_source))
    checkpoint = output_dir / "checkpoints" / "last" / "pretrained_model"
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"LeRobot training did not produce {model_path}")
    launch_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "model_sha256": _sha256(model_path),
        }
    )
    _write_json(launch_manifest_path, launch_manifest)


if __name__ == "__main__":
    main()
