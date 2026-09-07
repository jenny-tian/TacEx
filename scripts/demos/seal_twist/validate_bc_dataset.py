"""Validate SealTwist BC trajectories and summarize their physical ranges."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
dataset_module_path = (
    ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "seal_twist" / "dataset.py"
)
spec = importlib.util.spec_from_file_location("seal_twist_dataset", dataset_module_path)
assert spec is not None and spec.loader is not None
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
validate_episode = dataset_module.validate_episode


def _stats(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("exp_report/bottle/dataset"))
    parser.add_argument(
        "--report", type=Path, default=Path("exp_report/bottle/dataset_validation.json")
    )
    args = parser.parse_args()

    records = sorted(path for path in args.dataset.glob("record_*") if path.is_dir())
    if not records:
        raise ValueError(f"No record_* directories under {args.dataset}")
    metadata = [validate_episode(path) for path in records]
    success_count = sum(bool(item["success"]) for item in metadata)
    report = {
        "dataset": str(args.dataset.resolve()),
        "episodes": len(metadata),
        "successes": success_count,
        "success_rate": success_count / len(metadata),
        "schema_versions": sorted({int(item["schema_version"]) for item in metadata}),
        "peak_seal_torque_nm": _stats(item["peak_seal_torque_nm"] for item in metadata),
        "peak_left_finger_force_n": _stats(
            item["peak_left_finger_force_n"] for item in metadata
        ),
        "peak_right_finger_force_n": _stats(
            item["peak_right_finger_force_n"] for item in metadata
        ),
        "final_alignment_error_m": _stats(
            item["final_alignment_error_m"] for item in metadata
        ),
        "expert_target_torque_nm": _stats(
            item["expert_target_torque_nm"] for item in metadata
        ),
        "expert_grasp_width_m": _stats(item["expert_grasp_width_m"] for item in metadata),
        "all_successes_respect_torque_force_and_alignment_limits": all(
            bool(item["success"])
            and item["torque_window_nm"][0]
            <= item["peak_seal_torque_nm"]
            <= item["torque_window_nm"][1]
            and item["final_alignment_error_m"] <= item["alignment_tolerance_m"]
            for item in metadata
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
