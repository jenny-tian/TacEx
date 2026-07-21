from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "scripts" / "lerobot" / "src"
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))


ACTION_KEY = "action"
STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.rgb"
THIRD_IMAGE_KEY = "observation.images.rgb_third"
DEFAULT_TASK = "lab pick"
DEFAULT_FPS = 30
DEFAULT_HDF5_THIRD_IMAGE_KEYS = (
    "rgb_third",
    "robot0_image_third",
    "robot0_image_third_person",
    "robot0_third_image",
    "third_person_image",
)


def import_h5py():
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "HDF5 input requires a working h5py installation. The current Python environment failed "
            f"to import h5py: {exc}"
        ) from exc
    return h5py


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "LeRobot conversion requires a working lerobot environment. In this repo, the known-good "
            "environment is likely /home/limx/anaconda3/envs/lerobot/bin/python. The current Python "
            f"failed to import LeRobotDataset: {exc}"
        ) from exc
    return LeRobotDataset


@dataclass(frozen=True)
class EpisodeSummary:
    name: str
    length: int
    task: str


@dataclass(frozen=True)
class DatasetSpec:
    fps: int
    task: str
    action_dim: int
    state_dim: int
    image_shapes: dict[str, tuple[int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TacEx CAFE HDF5 files or ForceCapture-style record directories to LeRobot v3 datasets."
    )
    parser.add_argument("--input", type=Path, required=True, help="HDF5 file or directory containing record_* episodes.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output LeRobot dataset root.")
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot repo id stored in metadata.")
    parser.add_argument("--fps", type=int, default=None, help="Dataset FPS. Defaults to metadata/timestamps or 30.")
    parser.add_argument("--task", type=str, default=None, help="Task text stored in every frame.")
    parser.add_argument("--success-only", action="store_true", help="Skip episodes marked success=False.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output root before writing.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert at most this many episodes.")
    parser.add_argument("--no-videos", action="store_true", help="Store image frames in parquet instead of mp4 videos.")
    parser.add_argument("--state-key", type=str, default="robot0_pos", help="HDF5 obs key used as observation.state.")
    parser.add_argument("--image-key", type=str, default="robot0_image", help="HDF5 obs key used as RGB image.")
    parser.add_argument(
        "--third-image-key",
        type=str,
        default="auto",
        help="Optional HDF5 obs key for the third-person camera. 'auto' discovers common names.",
    )
    parser.add_argument("--no-third-camera", action="store_true", help="Do not include the third-person camera.")
    parser.add_argument(
        "--action-key",
        type=str,
        default="auto",
        help="HDF5 action key. 'auto' prefers actions/low, then actions/high.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Quaternion order for record directory aligned/quat.npy.",
    )
    return parser.parse_args()


def as_resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def default_output_root(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}_lerobot")
    return input_path.with_name(f"{input_path.name}_lerobot")


def prepare_output_root(input_path: Path, output_root: Path, overwrite: bool) -> None:
    overlaps_input = (
        output_root == input_path
        or output_root in input_path.parents
        or (input_path.is_dir() and input_path in output_root.parents)
    )
    if overlaps_input:
        raise ValueError(f"Refusing to use an output root that overlaps the input path: {output_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    return value


def attr_str(attrs: h5py.AttributeManager | dict[str, Any], key: str, default: str | None = None) -> str | None:
    if key not in attrs:
        return default
    value = decode_attr(attrs[key])
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def attr_int(attrs: h5py.AttributeManager, keys: tuple[str, ...], default: int | None = None) -> int | None:
    for key in keys:
        if key in attrs:
            value = decode_attr(attrs[key])
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            return int(round(float(value)))
    return default


def numeric_suffix(name: str) -> int:
    tail = name.rsplit("_", maxsplit=1)[-1]
    return int(tail) if tail.isdigit() else 0


def is_hdf5_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in {".hdf5", ".h5", ".hdf"}:
        return True
    try:
        return bool(import_h5py().is_hdf5(path))
    except OSError:
        return False


def discover_record_dirs(root: Path) -> list[Path]:
    if root.is_dir() and (root / "aligned").is_dir():
        return [root]
    record_dirs = [path for path in root.glob("record_*") if path.is_dir() and (path / "aligned").is_dir()]
    return sorted(record_dirs, key=lambda path: numeric_suffix(path.name))


def infer_input_kind(input_path: Path) -> str:
    if is_hdf5_file(input_path):
        return "hdf5"
    if input_path.is_dir() and discover_record_dirs(input_path):
        return "record"
    raise ValueError(f"Input must be a HDF5 file or a directory containing record_* folders: {input_path}")


def ensure_1d_float32(value: Any, expected_dim: int | None = None, name: str = "array") -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if expected_dim is not None and array.shape != (expected_dim,):
        raise ValueError(f"{name} has shape {array.shape}; expected ({expected_dim},).")
    return array


def normalize_image(value: Any, name: str = "image") -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} has shape {image.shape}; expected a 3D RGB image.")
    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"{name} has shape {image.shape}; expected HWC or CHW RGB image.")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        min_value = float(np.nanmin(image)) if image.size else 0.0
        if min_value < 0.0:
            raise ValueError(f"{name} contains negative pixels; cannot convert to uint8.")
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.integer):
        return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    raise ValueError(f"{name} has unsupported dtype {image.dtype}.")


def quat_to_rot6d(quat: np.ndarray, order: str) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    if order == "xyzw":
        x, y, z, w = np.moveaxis(q, -1, 0)
    else:
        w, x, y, z = np.moveaxis(q, -1, 0)

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.maximum(norm, 1.0e-8)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - z * w)
    rot[:, 0, 2] = 2.0 * (x * z + y * w)
    rot[:, 1, 0] = 2.0 * (x * y + z * w)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - x * w)
    rot[:, 2, 0] = 2.0 * (x * z - y * w)
    rot[:, 2, 1] = 2.0 * (y * z + x * w)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot[:, :, :2].reshape(q.shape[0], 6)


def record_state_from_parts(xyz: np.ndarray, quat: np.ndarray, width: np.ndarray, quat_order: str) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    width = np.asarray(width, dtype=np.float32).reshape(len(xyz), -1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"record xyz has shape {xyz.shape}; expected (T, 3).")
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"record quat has shape {quat.shape}; expected (T, 4).")
    if width.shape[1] != 1:
        raise ValueError(f"record width has shape {width.shape}; expected (T, 1).")
    if not (len(xyz) == len(quat) == len(width)):
        raise ValueError(f"record state component lengths differ: xyz={len(xyz)}, quat={len(quat)}, width={len(width)}.")
    return np.concatenate((xyz, quat_to_rot6d(quat, quat_order), width), axis=-1).astype(np.float32, copy=False)


def infer_fps_from_timestamps(record_dirs: list[Path], fallback: int = DEFAULT_FPS) -> int:
    fps_values = []
    for record_dir in record_dirs:
        timestamps_path = record_dir / "aligned" / "timestamps.npy"
        if not timestamps_path.exists():
            continue
        timestamps = np.load(timestamps_path, mmap_mode="r")
        if len(timestamps) < 2:
            continue
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            fps_values.append(1.0 / float(np.median(diffs)))
    if not fps_values:
        print(f"[WARN] Could not infer record FPS from timestamps; using {fallback}.")
        return fallback
    fps = int(round(float(np.median(fps_values))))
    if fps <= 0:
        raise ValueError(f"Inferred invalid FPS from timestamps: {fps}")
    return fps


def hdf5_episode_names(h5: h5py.File, success_only: bool) -> list[str]:
    if "data" not in h5:
        raise ValueError("HDF5 file is missing required group /data.")
    names = sorted(h5["data"].keys(), key=numeric_suffix)
    selected = []
    for name in names:
        demo = h5["data"][name]
        if success_only and not bool(demo.attrs.get("success", True)):
            continue
        selected.append(name)
    return selected


def hdf5_task(h5: h5py.File, demo: h5py.Group, explicit_task: str | None) -> str:
    if explicit_task:
        return explicit_task
    return attr_str(demo.attrs, "instruction", attr_str(h5.attrs, "instruction", DEFAULT_TASK)) or DEFAULT_TASK


def hdf5_fps(h5: h5py.File, first_demo: h5py.Group, explicit_fps: int | None) -> int:
    if explicit_fps is not None:
        return explicit_fps
    keys = ("fps", "low_fps", "camera_fps", "video_fps", "control_fps", "hz")
    fps = attr_int(h5.attrs, keys)
    if fps is None:
        fps = attr_int(first_demo.attrs, keys)
    return fps if fps is not None and fps > 0 else DEFAULT_FPS


def resolve_hdf5_action_key(actions: h5py.Group, requested: str) -> str:
    if requested != "auto":
        if requested not in actions:
            raise KeyError(f"HDF5 actions group is missing requested key {requested!r}. Available: {list(actions)}")
        return requested
    if "low" in actions:
        return "low"
    if "high" in actions:
        return "high"
    if len(actions) == 1:
        return next(iter(actions.keys()))
    raise KeyError(f"Could not auto-select action key. Available action keys: {list(actions)}")


def resolve_hdf5_third_image_key(obs: h5py.Group, requested: str, include_third: bool) -> str | None:
    if not include_third:
        return None
    if requested == "":
        return None
    if requested != "auto":
        if requested not in obs:
            raise KeyError(f"HDF5 obs group is missing requested third image key {requested!r}. Available: {list(obs)}")
        return requested
    for candidate in DEFAULT_HDF5_THIRD_IMAGE_KEYS:
        if candidate in obs:
            return candidate
    return None


def effective_action_length(demo: h5py.Group, action_key: str) -> int:
    dataset_len = int(demo["actions"][action_key].shape[0])
    attr_len = attr_int(demo.attrs, (f"length_{action_key}",), dataset_len)
    return min(dataset_len, attr_len or dataset_len)


def effective_state_length(demo: h5py.Group, state_key: str, image_len: int) -> int:
    dataset_len = int(demo["obs"][state_key].shape[0])
    if dataset_len == image_len:
        return dataset_len
    attr_len = attr_int(demo.attrs, ("length_high", f"length_{state_key}"), dataset_len)
    return min(dataset_len, attr_len or dataset_len)


def effective_image_length(demo: h5py.Group, image_key: str) -> int:
    dataset_len = int(demo["obs"][image_key].shape[0])
    attr_len = attr_int(demo.attrs, ("length_low", f"length_{image_key}"), dataset_len)
    return min(dataset_len, attr_len or dataset_len)


def select_source_index(frame_index: int, target_len: int, source_len: int, freq_ratio: int) -> int:
    if source_len <= 0:
        raise ValueError("Source sequence has no frames.")
    if source_len < target_len:
        raise ValueError(f"Source sequence is shorter than image sequence: source={source_len}, image={target_len}.")
    if source_len == target_len:
        return frame_index
    if freq_ratio > 1:
        idx = frame_index * freq_ratio
        if idx < source_len:
            return idx
    if target_len <= 1:
        return 0
    return min(int(round(frame_index * (source_len - 1) / (target_len - 1))), source_len - 1)


def inspect_hdf5_spec(path: Path, args: argparse.Namespace) -> DatasetSpec:
    h5py = import_h5py()
    with h5py.File(path, "r") as h5:
        names = hdf5_episode_names(h5, args.success_only)
        if args.max_episodes is not None:
            names = names[: args.max_episodes]
        if not names:
            raise ValueError("No HDF5 episodes selected.")

        first_demo = h5["data"][names[0]]
        obs = first_demo["obs"]
        actions = first_demo["actions"]
        if args.state_key not in obs:
            raise KeyError(f"HDF5 demo {names[0]} is missing obs/{args.state_key}. Available: {list(obs)}")
        if args.image_key not in obs:
            raise KeyError(f"HDF5 demo {names[0]} is missing obs/{args.image_key}. Available: {list(obs)}")
        action_key = resolve_hdf5_action_key(actions, args.action_key)
        third_image_key = resolve_hdf5_third_image_key(obs, args.third_image_key, not args.no_third_camera)

        image0 = normalize_image(obs[args.image_key][0], f"{names[0]}/obs/{args.image_key}[0]")
        image_shapes = {IMAGE_KEY: tuple(int(v) for v in image0.shape)}
        if third_image_key is not None:
            third_image0 = normalize_image(obs[third_image_key][0], f"{names[0]}/obs/{third_image_key}[0]")
            image_shapes[THIRD_IMAGE_KEY] = tuple(int(v) for v in third_image0.shape)
        state0 = ensure_1d_float32(obs[args.state_key][0], name=f"{names[0]}/obs/{args.state_key}[0]")
        action0 = ensure_1d_float32(actions[action_key][0], name=f"{names[0]}/actions/{action_key}[0]")
        return DatasetSpec(
            fps=hdf5_fps(h5, first_demo, args.fps),
            task=hdf5_task(h5, first_demo, args.task),
            action_dim=action0.shape[0],
            state_dim=state0.shape[0],
            image_shapes=image_shapes,
        )


def inspect_record_spec(record_dirs: list[Path], args: argparse.Namespace) -> DatasetSpec:
    selected_dirs = record_dirs[: args.max_episodes] if args.max_episodes is not None else record_dirs
    if not selected_dirs:
        raise ValueError("No record episodes selected.")
    first = selected_dirs[0]
    aligned = first / "aligned"
    image0 = normalize_image(np.load(aligned / "rgb.npy", mmap_mode="r")[0], f"{first.name}/aligned/rgb.npy[0]")
    image_shapes = {IMAGE_KEY: tuple(int(v) for v in image0.shape)}
    third_image_path = aligned / "rgb_third.npy"
    if not args.no_third_camera and third_image_path.exists():
        third_image0 = normalize_image(np.load(third_image_path, mmap_mode="r")[0], f"{first.name}/aligned/rgb_third.npy[0]")
        image_shapes[THIRD_IMAGE_KEY] = tuple(int(v) for v in third_image0.shape)
    action0 = ensure_1d_float32(np.load(aligned / "action.npy", mmap_mode="r")[0], name=f"{first.name}/aligned/action.npy[0]")
    state = record_state_from_parts(
        np.load(aligned / "xyz.npy", mmap_mode="r"),
        np.load(aligned / "quat.npy", mmap_mode="r"),
        np.load(aligned / "width.npy", mmap_mode="r"),
        args.quat_order,
    )
    fps = args.fps if args.fps is not None else infer_fps_from_timestamps(selected_dirs)
    return DatasetSpec(
        fps=fps,
        task=args.task or DEFAULT_TASK,
        action_dim=action0.shape[0],
        state_dim=state.shape[1],
        image_shapes=image_shapes,
    )


def feature_names(prefix: str, dim: int) -> list[str]:
    if dim == 10:
        return [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"] + [f"{prefix}_rot6d_{i}" for i in range(6)] + [
            f"{prefix}_width"
        ]
    return [f"{prefix}_{i}" for i in range(dim)]


def make_features(spec: DatasetSpec, use_videos: bool) -> dict[str, dict[str, Any]]:
    features = {
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (spec.action_dim,),
            "names": feature_names("action", spec.action_dim),
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (spec.state_dim,),
            "names": feature_names("state", spec.state_dim),
        },
    }
    for image_key, image_shape in spec.image_shapes.items():
        features[image_key] = {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    return features


def hdf5_frames(path: Path, args: argparse.Namespace, spec: DatasetSpec) -> Iterator[EpisodeSummary | dict[str, Any]]:
    h5py = import_h5py()
    with h5py.File(path, "r") as h5:
        names = hdf5_episode_names(h5, args.success_only)
        if args.max_episodes is not None:
            names = names[: args.max_episodes]
        for name in names:
            demo = h5["data"][name]
            obs = demo["obs"]
            actions = demo["actions"]
            if args.state_key not in obs:
                raise KeyError(f"HDF5 demo {name} is missing obs/{args.state_key}.")
            if args.image_key not in obs:
                raise KeyError(f"HDF5 demo {name} is missing obs/{args.image_key}.")
            action_key = resolve_hdf5_action_key(actions, args.action_key)
            third_image_key = resolve_hdf5_third_image_key(obs, args.third_image_key, THIRD_IMAGE_KEY in spec.image_shapes)
            image_len = effective_image_length(demo, args.image_key)
            third_image_len = effective_image_length(demo, third_image_key) if third_image_key is not None else 0
            state_len = effective_state_length(demo, args.state_key, image_len)
            action_len = effective_action_length(demo, action_key)
            freq_ratio = attr_int(demo.attrs, ("freq_ratio",), attr_int(h5.attrs, ("freq_ratio",), 1)) or 1
            task = hdf5_task(h5, demo, args.task)
            yield EpisodeSummary(name=name, length=image_len, task=task)
            for i in range(image_len):
                state_idx = select_source_index(i, image_len, state_len, freq_ratio)
                action_idx = select_source_index(i, image_len, action_len, freq_ratio)
                state = ensure_1d_float32(
                    obs[args.state_key][state_idx], expected_dim=spec.state_dim, name=f"{name}/obs/{args.state_key}[{state_idx}]"
                )
                action = ensure_1d_float32(
                    actions[action_key][action_idx],
                    expected_dim=spec.action_dim,
                    name=f"{name}/actions/{action_key}[{action_idx}]",
                )
                image = normalize_image(obs[args.image_key][i], f"{name}/obs/{args.image_key}[{i}]")
                if image.shape != spec.image_shapes[IMAGE_KEY]:
                    raise ValueError(
                        f"{name}/obs/{args.image_key}[{i}] shape {image.shape}; expected {spec.image_shapes[IMAGE_KEY]}."
                    )
                frame = {
                    ACTION_KEY: action,
                    STATE_KEY: state,
                    IMAGE_KEY: image,
                    "task": task,
                }
                if third_image_key is not None:
                    third_idx = select_source_index(i, image_len, third_image_len, 1)
                    third_image = normalize_image(
                        obs[third_image_key][third_idx], f"{name}/obs/{third_image_key}[{third_idx}]"
                    )
                    if third_image.shape != spec.image_shapes[THIRD_IMAGE_KEY]:
                        raise ValueError(
                            f"{name}/obs/{third_image_key}[{third_idx}] shape {third_image.shape}; "
                            f"expected {spec.image_shapes[THIRD_IMAGE_KEY]}."
                        )
                    frame[THIRD_IMAGE_KEY] = third_image
                yield frame


def load_record_success(record_dir: Path) -> bool:
    metadata_path = record_dir / "metadata.npz"
    if not metadata_path.exists():
        return True
    with np.load(metadata_path, allow_pickle=True) as metadata:
        if "success" not in metadata:
            return True
        return bool(np.asarray(metadata["success"]).item())


def record_frames(record_dirs: list[Path], args: argparse.Namespace, spec: DatasetSpec) -> Iterator[EpisodeSummary | dict[str, Any]]:
    selected = record_dirs[: args.max_episodes] if args.max_episodes is not None else record_dirs
    for record_dir in selected:
        if args.success_only and not load_record_success(record_dir):
            continue
        aligned = record_dir / "aligned"
        rgb = np.load(aligned / "rgb.npy", mmap_mode="r")
        rgb_third = (
            np.load(aligned / "rgb_third.npy", mmap_mode="r") if THIRD_IMAGE_KEY in spec.image_shapes else None
        )
        action = np.load(aligned / "action.npy", mmap_mode="r")
        state = record_state_from_parts(
            np.load(aligned / "xyz.npy", mmap_mode="r"),
            np.load(aligned / "quat.npy", mmap_mode="r"),
            np.load(aligned / "width.npy", mmap_mode="r"),
            args.quat_order,
        )
        length = min(len(rgb), len(action), len(state), len(rgb_third) if rgb_third is not None else len(rgb))
        if length <= 0:
            raise ValueError(f"Record episode has no aligned frames: {record_dir}")
        lengths_match = len(rgb) == len(action) == len(state) and (rgb_third is None or len(rgb_third) == len(rgb))
        if not lengths_match:
            raise ValueError(
                f"Record episode lengths differ in {record_dir}: "
                f"rgb={len(rgb)}, rgb_third={0 if rgb_third is None else len(rgb_third)}, "
                f"action={len(action)}, state={len(state)}."
            )
        task = args.task or DEFAULT_TASK
        yield EpisodeSummary(name=record_dir.name, length=length, task=task)
        for i in range(length):
            frame_action = ensure_1d_float32(action[i], expected_dim=spec.action_dim, name=f"{record_dir.name}/action[{i}]")
            frame_state = ensure_1d_float32(state[i], expected_dim=spec.state_dim, name=f"{record_dir.name}/state[{i}]")
            frame_image = normalize_image(rgb[i], f"{record_dir.name}/rgb[{i}]")
            if frame_image.shape != spec.image_shapes[IMAGE_KEY]:
                raise ValueError(f"{record_dir.name}/rgb[{i}] shape {frame_image.shape}; expected {spec.image_shapes[IMAGE_KEY]}.")
            frame = {
                ACTION_KEY: frame_action,
                STATE_KEY: frame_state,
                IMAGE_KEY: frame_image,
                "task": task,
            }
            if rgb_third is not None:
                frame_third_image = normalize_image(rgb_third[i], f"{record_dir.name}/rgb_third[{i}]")
                if frame_third_image.shape != spec.image_shapes[THIRD_IMAGE_KEY]:
                    raise ValueError(
                        f"{record_dir.name}/rgb_third[{i}] shape {frame_third_image.shape}; "
                        f"expected {spec.image_shapes[THIRD_IMAGE_KEY]}."
                    )
                frame[THIRD_IMAGE_KEY] = frame_third_image
            yield frame


def convert(input_path: Path, output_root: Path, repo_id: str, args: argparse.Namespace) -> None:
    kind = infer_input_kind(input_path)
    if kind == "hdf5":
        spec = inspect_hdf5_spec(input_path, args)
        frame_stream = hdf5_frames(input_path, args, spec)
    else:
        record_dirs = discover_record_dirs(input_path)
        spec = inspect_record_spec(record_dirs, args)
        frame_stream = record_frames(record_dirs, args, spec)

    print(
        "[INFO] Creating LeRobot dataset "
        f"repo_id={repo_id!r}, root={output_root}, input_kind={kind}, fps={spec.fps}, "
        f"action_dim={spec.action_dim}, state_dim={spec.state_dim}, image_shapes={spec.image_shapes}"
    )
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=spec.fps,
        robot_type="tacex",
        features=make_features(spec, use_videos=not args.no_videos),
        use_videos=not args.no_videos,
    )

    episode_count = 0
    frame_count = 0
    current_episode: EpisodeSummary | None = None
    try:
        for item in frame_stream:
            if isinstance(item, EpisodeSummary):
                if current_episode is not None:
                    dataset.save_episode()
                    episode_count += 1
                    print(f"[INFO] Saved {current_episode.name}: {current_episode.length} frames")
                current_episode = item
                print(f"[INFO] Converting {item.name}: {item.length} frames, task={item.task!r}")
                continue
            dataset.add_frame(item)
            frame_count += 1

        if current_episode is not None:
            dataset.save_episode()
            episode_count += 1
            print(f"[INFO] Saved {current_episode.name}: {current_episode.length} frames")
    finally:
        dataset.finalize()

    if episode_count == 0:
        raise ValueError("No episodes were converted.")
    print(f"[DONE] Wrote {episode_count} episodes / {frame_count} frames to {output_root}")


def main() -> None:
    args = parse_args()
    input_path = as_resolved(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    output_root = as_resolved(args.output_root) if args.output_root is not None else as_resolved(default_output_root(input_path))
    prepare_output_root(input_path, output_root, args.overwrite)
    repo_id = args.repo_id or f"local/{output_root.name}"

    convert(input_path=input_path, output_root=output_root, repo_id=repo_id, args=args)


if __name__ == "__main__":
    main()
"""
python scripts/bc_training/create_lerobot_dataset.py \
  --input /home/limx/github_repo/TacEx/dataset \
  --output-root /home/limx/github_repo/TacEx/dataset_lerobot \
  --repo-id tacex/lab_pick \
  --overwrite
"""
