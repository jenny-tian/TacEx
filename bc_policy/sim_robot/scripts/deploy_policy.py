from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


def get_observation() -> dict[str, np.ndarray]:
    """Simulation integration point for online inference.

    Expected return format:
        {
            "robot0_pos": np.ndarray,   # shape (10,), raw simulator state
            "robot0_image": np.ndarray, # HWC uint8 or CHW/HWC float image
        }
    """

    raise NotImplementedError("Connect the simulator observation adapter here.")


class HDF5ReplaySource:
    def __init__(
        self,
        path: Path,
        demo_index: int,
        action_key: str,
        state_key: str,
        image_key: str,
        start_frame: int,
        stride: int,
        max_steps: int | None,
    ) -> None:
        import h5py

        self.file = h5py.File(path, "r")
        self.demo = self.file["data"][f"demo_{int(demo_index)}"]
        self.state_key = state_key
        self.image_key = image_key
        length = min(
            int(self.demo["obs"][state_key].shape[0]),
            int(self.demo["obs"][image_key].shape[0]),
            int(self.demo["actions"][action_key].shape[0]),
        )
        frames = np.arange(max(0, start_frame), length, max(1, stride), dtype=np.int64)
        if max_steps is not None:
            frames = frames[:max_steps]
        if len(frames) == 0:
            raise ValueError(f"No replay frames selected from demo_{demo_index}, length={length}")
        self.frames = frames
        self.cursor = 0

    def close(self) -> None:
        self.file.close()

    def next_observation(self) -> tuple[int, dict[str, np.ndarray]]:
        if self.cursor >= len(self.frames):
            raise StopIteration
        frame = int(self.frames[self.cursor])
        self.cursor += 1
        obs = self.demo["obs"]
        return frame, {
            "robot0_pos": obs[self.state_key][frame].astype(np.float32),
            "robot0_image": obs[self.image_key][frame],
        }


def print_action_chunk(step: int, frame: int | None, action_chunk: np.ndarray, elapsed_ms: float, precision: int, as_json: bool) -> None:
    if as_json:
        payload = {
            "step": step,
            "frame": frame,
            "inference_ms": elapsed_ms,
            "action_chunk": action_chunk.astype(float).tolist(),
        }
        print(json.dumps(payload))
        return

    frame_text = "" if frame is None else f" frame={frame}"
    print(f"\n[sim deploy step={step}{frame_text}] inference_ms={elapsed_ms:.2f}")
    from sim_robot.deployment.policy_runner import format_action_chunk

    print(format_action_chunk(action_chunk, precision=precision))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sim policy inference and print every predicted action chunk.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loop-hz", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-resize-images", action="store_true")
    parser.add_argument("--replay-hdf5", type=Path, default=None)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--action-key", type=str, default="high")
    parser.add_argument("--state-key", type=str, default="robot0_pos")
    parser.add_argument("--image-key", type=str, default="robot0_image")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    from sim_robot.deployment.policy_runner import SimActionChunkPolicyRunner

    runner = SimActionChunkPolicyRunner(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_ema=not args.no_ema,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        resize_images=not args.no_resize_images,
    )

    source = None
    if args.replay_hdf5 is not None:
        source = HDF5ReplaySource(
            path=args.replay_hdf5,
            demo_index=args.demo_index,
            action_key=args.action_key,
            state_key=args.state_key,
            image_key=args.image_key,
            start_frame=args.start_frame,
            stride=args.stride,
            max_steps=args.max_steps,
        )

    period = 0.0 if args.loop_hz <= 0 else 1.0 / args.loop_hz
    step = 0
    try:
        while args.max_steps is None or step < args.max_steps:
            loop_start = time.perf_counter()
            frame = None
            if source is None:
                obs = get_observation()
            else:
                frame, obs = source.next_observation()

            infer_start = time.perf_counter()
            action_chunk = runner.predict_action_chunk(obs)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            print_action_chunk(step, frame, action_chunk, infer_ms, args.print_precision, args.json)
            step += 1

            if period > 0:
                sleep_time = period - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except StopIteration:
        print("Replay finished.")
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()

