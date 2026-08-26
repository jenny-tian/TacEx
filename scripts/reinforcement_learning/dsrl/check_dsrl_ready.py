#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TacEx LabPick prerequisites for Diffusion/DSRL training.")
    parser.add_argument("--records", type=Path, default=None, help="Directory containing record_* episodes.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Converted LeRobot dataset root.")
    parser.add_argument("--policy", type=Path, default=None, help="Trained LeRobot Diffusion checkpoint directory.")
    parser.add_argument("--min-success", type=int, default=50)
    return parser.parse_args()


def inspect_records(root: Path, min_success: int) -> list[str]:
    errors: list[str] = []
    record_dirs = sorted(path for path in root.glob("record_*") if path.is_dir())
    successes = 0
    compatible = 0
    for record_dir in record_dirs:
        metadata_path = record_dir / "metadata.npz"
        aligned = record_dir / "aligned"
        if aligned.is_dir() and all((aligned / name).is_file() for name in ("rgb.npy", "action.npy", "xyz.npy", "quat.npy", "width.npy")):
            compatible += 1
        if metadata_path.is_file():
            with np.load(metadata_path) as metadata:
                successes += int(bool(metadata["success"])) if "success" in metadata else 0

    print(f"[INFO] records={len(record_dirs)} compatible_aligned={compatible} successful={successes}")
    if compatible < min_success:
        errors.append(f"Need at least {min_success} new-format aligned episodes; found {compatible}.")
    if successes < min_success:
        errors.append(f"Need at least {min_success} successful episodes; found {successes}.")
    return errors


def inspect_dataset(root: Path) -> list[str]:
    errors: list[str] = []
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return [f"Missing LeRobot metadata: {info_path}"]
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    required = {"action", "observation.state", "observation.images.rgb"}
    missing = sorted(required - set(features))
    print(
        f"[INFO] dataset episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
        f"fps={info.get('fps')} features={sorted(features)}"
    )
    if missing:
        errors.append(f"LeRobot dataset is missing required features: {missing}")
    return errors


def inspect_policy(root: Path) -> list[str]:
    errors: list[str] = []
    config_path = root / "config.json"
    if not config_path.is_file():
        return [f"Missing policy config: {config_path}"]
    config = json.loads(config_path.read_text())
    policy_type = config.get("type")
    scheduler = config.get("noise_scheduler_type")
    print(
        f"[INFO] policy type={policy_type!r} scheduler={scheduler!r} "
        f"horizon={config.get('horizon')} action_steps={config.get('n_action_steps')}"
    )
    if policy_type != "diffusion":
        errors.append(f"Expected a diffusion policy, found {policy_type!r}.")
    if scheduler != "DDIM":
        errors.append(f"DSRL requires noise_scheduler_type='DDIM', found {scheduler!r}.")
    for filename in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (root / filename).is_file():
            errors.append(f"Missing checkpoint asset: {root / filename}")
    return errors


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    if args.records is not None:
        errors.extend(inspect_records(args.records.expanduser().resolve(), args.min_success))
    if args.dataset_root is not None:
        errors.extend(inspect_dataset(args.dataset_root.expanduser().resolve()))
    if args.policy is not None:
        errors.extend(inspect_policy(args.policy.expanduser().resolve()))
    if args.records is None and args.dataset_root is None and args.policy is None:
        raise SystemExit("Provide at least one of --records, --dataset-root, or --policy.")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("[PASS] Requested DSRL prerequisites are ready.")


if __name__ == "__main__":
    main()
