from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _record_index(path: Path) -> int:
    suffix = path.name.removeprefix("record_")
    if not suffix.isdigit():
        raise ValueError(f"Invalid record directory name: {path.name}")
    return int(suffix)


def discover_records(input_root: Path) -> list[Path]:
    records = [
        path
        for path in input_root.glob("record_*")
        if path.is_dir() and (path / "aligned" / "action.npy").is_file()
    ]
    return sorted(records, key=_record_index)


def load_success(record_dir: Path) -> bool:
    metadata_path = record_dir / "metadata.npz"
    if not metadata_path.exists():
        return True
    with np.load(metadata_path, allow_pickle=True) as metadata:
        return bool(np.asarray(metadata.get("success", True)).item())


def quat_to_rot6d(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    if order == "xyzw":
        x, y, z, w = np.moveaxis(q, -1, 0)
    elif order == "wxyz":
        w, x, y, z = np.moveaxis(q, -1, 0)
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")

    norm = np.maximum(np.sqrt(w * w + x * x + y * y + z * z), 1.0e-8)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = np.empty((len(q), 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation[:, :, :2].reshape(len(q), 6)


def record_state(xyz: np.ndarray, quat: np.ndarray, width: np.ndarray, quat_order: str) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    width = np.asarray(width, dtype=np.float32).reshape(len(xyz), -1)
    if xyz.shape != (len(xyz), 3):
        raise ValueError(f"Expected xyz shape (T, 3), received {xyz.shape}.")
    if quat.shape != (len(xyz), 4):
        raise ValueError(f"Expected quat shape (T, 4), received {quat.shape}.")
    if width.shape != (len(xyz), 1):
        raise ValueError(f"Expected width shape (T, 1), received {width.shape}.")
    return np.concatenate((xyz, quat_to_rot6d(quat, quat_order), width), axis=-1).astype(np.float32)


def _metadata(record_dir: Path) -> dict[str, np.ndarray]:
    metadata_path = record_dir / "metadata.npz"
    if not metadata_path.exists():
        return {}
    with np.load(metadata_path, allow_pickle=True) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _metadata_text(metadata: dict[str, np.ndarray], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    value = np.asarray(value)
    if value.ndim != 0:
        return None
    return str(value.item())


def _resolve_action_alignment(metadata: dict[str, np.ndarray], requested: str) -> str:
    if requested != "auto":
        return requested
    # Legacy records contain post-action observations. New records explicitly
    # declare pre_action timing and can use same-index causal pairs.
    timing = _metadata_text(metadata, "observation_timing")
    return "same-index" if timing == "pre_action" else "next-action"



def _compression_kwargs(compression: str) -> dict:
    if compression == "none":
        return {}
    return {"compression": compression, "shuffle": True}


def convert_records(
    input_root: str | Path,
    output_path: str | Path,
    *,
    success_only: bool = True,
    max_episodes: int | None = None,
    quat_order: str = "xyzw",
    include_third_camera: bool = False,
    compression: str = "none",
    overwrite: bool = False,
    labware: str = "slide",
    instruction: str = "pick up the transparent labware",
    action_alignment: str = "auto",
) -> int:
    if action_alignment not in {"auto", "same-index", "next-action"}:
        raise ValueError("action_alignment must be auto, same-index, or next-action")
    input_root = Path(input_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    records = discover_records(input_root)
    if success_only:
        records = [record for record in records if load_success(record)]
    if max_episodes is not None:
        records = records[:max_episodes]
    if not records:
        raise ValueError(f"No compatible records found under {input_root}.")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = _compression_kwargs(compression)

    with h5py.File(output_path, "w") as h5:
        h5.attrs["num_demos"] = len(records)
        h5.attrs["include_images"] = True
        h5.attrs["freq_ratio"] = 1
        h5.attrs["fps"] = 60
        h5.attrs["labware"] = labware
        h5.attrs["instruction"] = instruction
        h5.attrs["high_freq_obs_keys"] = "robot0_pos,robot0_force"
        h5.attrs["low_freq_obs_keys"] = "robot0_image"
        h5.attrs["high_freq_action_key"] = "high"
        h5.attrs["low_freq_action_key"] = "low"
        data_group = h5.create_group("data")

        for demo_index, record_dir in enumerate(records):
            aligned = record_dir / "aligned"
            xyz = np.load(aligned / "xyz.npy", mmap_mode="r")
            quat = np.load(aligned / "quat.npy", mmap_mode="r")
            width = np.load(aligned / "width.npy", mmap_mode="r")
            force = np.load(aligned / "ft.npy", mmap_mode="r")
            image = np.load(aligned / "rgb.npy", mmap_mode="r")
            action = np.load(aligned / "action.npy", mmap_mode="r")
            metadata = _metadata(record_dir)
            arrays = {
                "xyz": xyz,
                "quat": quat,
                "width": width,
                "force": force,
                "image": image,
                "action": action,
            }
            third_image = None
            if include_third_camera:
                third_image = np.load(aligned / "rgb_third.npy", mmap_mode="r")
                arrays["third_image"] = third_image
            lengths = {name: len(value) for name, value in arrays.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Aligned array lengths differ in {record_dir}: {lengths}")
            length = len(action)
            if length < 1:
                raise ValueError(f"Empty aligned episode: {record_dir}")
            alignment = _resolve_action_alignment(metadata, action_alignment)
            if alignment == "next-action":
                if length < 2:
                    raise ValueError(f"Cannot shift a one-frame episode: {record_dir}")
                xyz = xyz[:-1]
                quat = quat[:-1]
                width = width[:-1]
                force = force[:-1]
                image = image[:-1]
                action = action[1:]
                if third_image is not None:
                    third_image = third_image[:-1]
                length -= 1
            state = record_state(xyz, quat, width, quat_order)
            if action.shape != (length, 10):
                raise ValueError(f"Expected action shape ({length}, 10) in {record_dir}, received {action.shape}.")
            if image.shape[1:] != (224, 224, 3):
                raise ValueError(f"Expected wrist image shape (T, 224, 224, 3), received {image.shape}.")

            demo = data_group.create_group(f"demo_{demo_index}")
            actions = demo.create_group("actions")
            actions.create_dataset("high", data=action, dtype=np.float32, **dataset_kwargs)
            actions.create_dataset("low", data=action, dtype=np.float32, **dataset_kwargs)
            observations = demo.create_group("obs")
            observations.create_dataset("robot0_pos", data=state, dtype=np.float32, **dataset_kwargs)
            observations.create_dataset("robot0_force", data=force, dtype=np.float32, **dataset_kwargs)
            observations.create_dataset("robot0_image", data=image, dtype=np.uint8, **dataset_kwargs)
            if third_image is not None:
                observations.create_dataset("robot0_image_third", data=third_image, dtype=np.uint8, **dataset_kwargs)

            demo.attrs["success"] = load_success(record_dir)
            demo.attrs["length_high"] = length
            demo.attrs["length_low"] = length
            demo.attrs["freq_ratio"] = 1
            demo.attrs["fps"] = 60
            demo.attrs["source_record"] = record_dir.name
            demo.attrs["action_alignment"] = alignment
            demo.attrs["observation_timing"] = _metadata_text(metadata, "observation_timing") or "legacy_post_action"
            demo.attrs["instruction"] = instruction
            if "labware_reset_pos_w" in metadata:
                demo.attrs["labware_reset_pos_w"] = metadata["labware_reset_pos_w"].astype(np.float32)
            if "labware_reset_quat_w" in metadata:
                demo.attrs["labware_reset_quat_w"] = metadata["labware_reset_quat_w"].astype(np.float32)

            print(
                f"[INFO] converted {demo_index + 1}/{len(records)} {record_dir.name} "
                f"frames={length} success={bool(demo.attrs['success'])}",
                flush=True,
            )

    print(f"[SUMMARY] wrote {len(records)} episodes to {output_path}", flush=True)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert aligned LabPick record directories to Flow Matching HDF5.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--include-third-camera", action="store_true")
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--labware", type=str, default="slide")
    parser.add_argument("--instruction", type=str, default="pick up the transparent labware")
    parser.add_argument(
        "--action-alignment",
        choices=("auto", "same-index", "next-action"),
        default="auto",
        help="Pair observations with the same action, or shift legacy post-action records to the next action.",
    )
    args = parser.parse_args()
    convert_records(
        args.input,
        args.output,
        success_only=args.success_only,
        max_episodes=args.max_episodes,
        quat_order=args.quat_order,
        include_third_camera=args.include_third_camera,
        compression=args.compression,
        overwrite=args.overwrite,
        labware=args.labware,
        instruction=args.instruction,
        action_alignment=args.action_alignment,
    )


if __name__ == "__main__":
    main()
