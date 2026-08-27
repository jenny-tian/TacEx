#!/usr/bin/env python
"""Merge completed CAFE record shards using validated directory symlinks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def record_index(path: Path) -> int:
    suffix = path.name.removeprefix("record_")
    return int(suffix) if suffix.isdigit() else -1


def record_contract(record: Path, yaw_tolerance_degrees: float) -> dict[str, object]:
    metadata_path = record / "metadata.npz"
    with np.load(metadata_path) as metadata:
        success = bool(np.asarray(metadata["success"]).reshape(-1)[0])
        quat = np.asarray(metadata["labware_reset_quat_w"], dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    aligned = record / "aligned"
    required = ("action.npy", "xyz.npy", "quat.npy", "width.npy", "rgb.npy", "rgb_third.npy")
    missing = [name for name in required if not (aligned / name).is_file()]
    if not success:
        raise ValueError(f"Record is not successful: {record}")
    if abs(yaw) > yaw_tolerance_degrees:
        raise ValueError(f"Record yaw is {yaw:.6f} degrees, expected zero: {record}")
    if missing:
        raise ValueError(f"Record is missing aligned files {missing}: {record}")
    lengths = {name: len(np.load(aligned / name, mmap_mode="r")) for name in required}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) <= 0:
        raise ValueError(f"Record stream lengths differ: {record}: {lengths}")
    return {"source": str(record.resolve()), "success": True, "yaw_degrees": yaw, "length": next(iter(lengths.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge successful yaw-zero CAFE record shards")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--target-records", type=int, required=True)
    parser.add_argument("--yaw-tolerance-degrees", type=float, default=0.1)
    args = parser.parse_args()
    if args.target_records < 1:
        parser.error("--target-records must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        (path for path in output.glob("record_*") if path.is_dir() and record_index(path) >= 0),
        key=record_index,
    )
    if len(existing) > args.target_records:
        raise ValueError(f"Output already has {len(existing)} records, exceeding target {args.target_records}")
    seen = {str(path.resolve()) for path in existing}
    candidates: list[Path] = []
    for source in args.source:
        source = source.expanduser().resolve()
        candidates.extend(
            sorted(
                (path for path in source.glob("record_*") if path.is_dir() and record_index(path) >= 0),
                key=record_index,
            )
        )

    next_index = max((record_index(path) for path in existing), default=-1) + 1
    added: list[Path] = []
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if len(existing) + len(added) >= args.target_records:
            break
        if resolved in seen:
            continue
        record_contract(candidate, args.yaw_tolerance_degrees)
        destination = output / f"record_{next_index:06d}"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.symlink(resolved, destination, target_is_directory=True)
        added.append(destination)
        seen.add(resolved)
        next_index += 1

    records = sorted(
        (path for path in output.glob("record_*") if path.is_dir() and record_index(path) >= 0),
        key=record_index,
    )
    if len(records) != args.target_records:
        raise ValueError(
            f"Only assembled {len(records)}/{args.target_records} records; "
            f"existing={len(existing)}, added={len(added)}, candidates={len(candidates)}"
        )
    contracts = [record_contract(record, args.yaw_tolerance_degrees) for record in records]
    manifest = {
        "target_records": args.target_records,
        "num_records": len(records),
        "num_added": len(added),
        "all_successful": all(bool(item["success"]) for item in contracts),
        "yaw_min_degrees": min(float(item["yaw_degrees"]) for item in contracts),
        "yaw_max_degrees": max(float(item["yaw_degrees"]) for item in contracts),
        "lengths": sorted({int(item["length"]) for item in contracts}),
        "records": [
            {"record": record.name, **contract} for record, contract in zip(records, contracts, strict=True)
        ],
    }
    manifest_path = output / "bc200_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
