"""Validate every GrapePick trajectory and summarize its force distribution."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
dataset_module_path = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "grape_pick" / "dataset.py"
spec = importlib.util.spec_from_file_location("grape_pick_dataset", dataset_module_path)
assert spec is not None and spec.loader is not None
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
validate_episode = dataset_module.validate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    records = sorted(path for path in args.dataset.glob("record_*") if path.is_dir())
    if not records:
        raise ValueError(f"No record_* directories under {args.dataset}")
    metadata = [validate_episode(path) for path in records]
    success_count = sum(bool(item["success"]) for item in metadata)
    peak_grip = np.asarray([item["peak_grip_force_n"] for item in metadata], dtype=np.float64)
    peak_stem = np.asarray([item["peak_stem_tension_n"] for item in metadata], dtype=np.float64)
    report = {
        "dataset": str(args.dataset.resolve()),
        "episodes": len(metadata),
        "successes": success_count,
        "success_rate": success_count / len(metadata),
        "peak_grip_force_n": {
            "min": float(peak_grip.min()),
            "mean": float(peak_grip.mean()),
            "max": float(peak_grip.max()),
        },
        "peak_stem_tension_n": {
            "min": float(peak_stem.min()),
            "mean": float(peak_stem.mean()),
            "max": float(peak_stem.max()),
        },
        "schema_versions": sorted({int(item["schema_version"]) for item in metadata}),
        "all_successes_respect_episode_grip_limit": all(
            not item["success"]
            or item["peak_grip_force_n"]
            <= item.get("episode_damage_threshold_n", item["grip_limit_n"]) + 1.0e-4
            for item in metadata
        ),
    }
    report_path = args.report or args.dataset / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
