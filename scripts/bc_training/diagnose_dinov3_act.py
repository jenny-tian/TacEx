#!/usr/bin/env python
"""Measure early-frame XY localization error for a trained DINOv3-ACT policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import dinov3_act as act
import train_bc as common


def summarize(errors: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(errors.size),
        "mean_m": float(errors.mean()),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.quantile(errors, 0.90)),
        "max_m": float(errors.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, normalizer, checkpoint = act.load_policy(args.checkpoint, device=args.device)
    config = checkpoint["train_config"]
    data_root = Path(config["data_root"])
    cache_root = Path(config["feature_cache_dir"])
    train_names = set(config["train_record_names"])
    names = [*config["train_record_names"], *config["val_record_names"]]
    image_keys = list(model.config.image_keys)

    states: list[np.ndarray] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    phases: list[float] = []
    rows: list[dict[str, Any]] = []
    for name in names:
        record = data_root / name
        aligned = record / "aligned"
        actions = np.load(aligned / "action.npy", mmap_mode="r")
        frame_index = min(max(args.frame_index, 0), len(actions) - 1)
        state_indices = common.clamp_indices(
            len(actions),
            np.arange(frame_index - model.config.state_obs_steps + 1, frame_index + 1),
        )
        image_indices = common.clamp_indices(
            len(actions),
            np.arange(frame_index - model.config.image_obs_steps + 1, frame_index + 1),
        )
        state = common.record_state_from_parts(
            np.load(aligned / "xyz.npy", mmap_mode="r"),
            np.load(aligned / "quat.npy", mmap_mode="r"),
            np.load(aligned / "width.npy", mmap_mode="r"),
            quat_order="xyzw",
        )[state_indices]
        state = normalizer.normalize_numpy(common.STATE_KEY, state)
        camera_features = np.stack(
            [np.asarray(np.load(cache_root / name / f"{key}.npy", mmap_mode="r")[image_indices], dtype=np.float32)
             for key in image_keys],
            axis=1,
        )
        states.append(state)
        features.append(camera_features)
        targets.append(np.asarray(actions[frame_index], dtype=np.float32))
        phases.append(frame_index / max(len(actions) - 1, 1))
        rows.append(
            {"record": name, "split": "train" if name in train_names else "val", "frame_index": frame_index}
        )

    device = next(model.parameters()).device
    predictions: list[np.ndarray] = []
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, len(names), batch_size):
            stop = min(start + batch_size, len(names))
            obs = {
                common.STATE_KEY: torch.from_numpy(np.stack(states[start:stop])).to(device),
                "dino_features": torch.from_numpy(np.stack(features[start:stop])).to(device),
                "phase": torch.tensor(phases[start:stop], dtype=torch.float32, device=device).unsqueeze(1),
            }
            normalized = model.predict_action(obs)[common.ACTION_KEY][:, 0].cpu().numpy()
            predictions.append(normalizer.unnormalize_numpy(common.ACTION_KEY, normalized))
    prediction = np.concatenate(predictions, axis=0)
    target = np.stack(targets)
    errors = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=1)
    for row, pred, truth, error in zip(rows, prediction, target, errors, strict=True):
        row.update(
            {
                "predicted_xy_m": pred[:2].tolist(),
                "target_xy_m": truth[:2].tolist(),
                "xy_error_m": float(error),
            }
        )

    train_mask = np.asarray([row["split"] == "train" for row in rows])
    result = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "requested_frame_index": args.frame_index,
        "metric": "Euclidean error between predicted and demonstrated action XY at the requested frame",
        "all": summarize(errors),
        "train": summarize(errors[train_mask]),
        "validation": summarize(errors[~train_mask]),
        "records": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
