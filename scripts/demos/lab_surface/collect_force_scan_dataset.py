"""Collect tactile force-scan demonstrations for TacEx-LabSurface-ForceScan-v0.

Each episode is saved as one compressed NPZ file.  The expert is deliberately
simple: it follows the left-to-right scan path and closes a proportional force
loop using the simulated contact sensor.  Tactile streams are recorded but are
not used by the expert, so they remain valid inputs for later BC/DSRL training.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _reexec_with_torch_cuda_library_path() -> None:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_dir / "site-packages"
    required = [site_packages / "torch" / "lib", site_packages / "nvidia" / "cudnn" / "lib"]
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    missing = [str(p) for p in required if p.is_dir() and str(p) not in current]
    if missing:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current])
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


_reexec_with_torch_cuda_library_path()


def _bootstrap_source_paths() -> None:
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return
    package_dir = Path(spec.origin).resolve().parent
    source_root = package_dir / "source"
    for package_name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
        package_source = source_root / package_name
        if (package_source / package_name).is_dir() and str(package_source) not in sys.path:
            sys.path.insert(0, str(package_source))


_bootstrap_source_paths()

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--record_dir", type=Path, default=Path("datasets/lab_surface_force_scan"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_episode_steps", type=int, default=0, help="0 uses the environment episode length")
parser.add_argument("--episode_length_s", type=float, default=40.0, help="Collection horizon for full-board approach, reposition and scan")
parser.add_argument("--action_position_scale_m", type=float, default=0.001, help="Collection-only Cartesian increment scale")
parser.add_argument("--force_kp", type=float, default=0.04, help="Normalized proportional force-loop gain")
parser.add_argument("--save_tactile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--save_visual", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--save_scene_preview", action="store_true", help="Save a rendered RGB preview in record_dir")
parser.add_argument("--record_video", action="store_true", help="Record the selected episode as an MP4")
parser.add_argument("--video_episode", type=int, default=0, help="Zero-based episode index to record")
parser.add_argument("--video_fps", type=float, default=30.0)
parser.add_argument("--video_stride", type=int, default=4, help="Append one video frame every N simulation steps")
parser.add_argument("--approach_fraction", type=float, default=0.65, help="Fraction of steps used to move above scan start")
parser.add_argument("--xy_kp", type=float, default=0.15, help="Proportional gain for lateral target tracking")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video_episode < 0:
    parser.error("--video_episode must be non-negative")
if args.video_fps <= 0.0:
    parser.error("--video_fps must be positive")
if args.video_stride <= 0:
    parser.error("--video_stride must be positive")
if not 0.1 <= args.approach_fraction < 0.9:
    parser.error("--approach_fraction must be in [0.1, 0.9)")
if not 0.01 <= args.xy_kp <= 1.0:
    parser.error("--xy_kp must be in [0.01, 1.0]")
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import tacex_tasks  # noqa: F401  (registers the task)
from tacex_tasks.lab_surface.lab_surface_env_cfg import LabSurfaceForceScanEnvCfg


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _surface_label(xy: np.ndarray, raised_centers: np.ndarray, groove_x: np.ndarray) -> tuple[int, int]:
    """Return (label, defect_kind): 0 flat, 1 raised, 2 groove."""
    if raised_centers.size:
        distances = np.linalg.norm(raised_centers - xy[None, :], axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) < 0.012:
            return 1, 0
    if groove_x.size and float(np.min(np.abs(groove_x - xy[0]))) < 0.004:
        return 2, 1
    return 0, -1


def _annotate_video_frame(frame: np.ndarray, *, step: int, force_n: float, target_force_n: float) -> np.ndarray:
    import cv2

    annotated = np.ascontiguousarray(frame[..., :3].copy())
    cv2.rectangle(annotated, (6, 6), (188, 38), (20, 20, 20), thickness=-1)
    cv2.putText(
        annotated,
        f"step {step:04d}  F={force_n:.2f}/{target_force_n:.1f} N",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def main() -> None:
    cfg = LabSurfaceForceScanEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.episode_length_s
    cfg.action_position_scale_m = args.action_position_scale_m
    # Demonstration collection must observe and correct a brief force peak;
    # the training/evaluation environment keeps the 4 N safety termination.
    cfg.terminate_on_overforce = False
    # Keep the nominal downward probe orientation for the well-conditioned
    # absolute pose IK used by the Panda asset.  The lateral target is still
    # absolute and Y is held at the measured contact line.
    cfg.hold_current_orientation_for_collection = False
    cfg.use_resolved_rate_position_servo = False
    # Keep the physics controller active during collection.  The Cartesian
    # target is smoothed below so rigid step features do not create impulses.
    cfg.expert_kinematic_control = False
    # Randomized board poses are supported by the environment; collect the
    # baseline fixed-pose dataset until the pose-aware controller is validated.
    cfg.randomize_board_pose = False
    if args.device:
        cfg.sim.device = args.device
    env = gym.make("TacEx-LabSurface-ForceScan-v0", cfg=cfg, render_mode="rgb_array")
    unwrapped = env.unwrapped
    max_steps = args.max_episode_steps or int(unwrapped.max_episode_length)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    groove_x = np.asarray([cfg.board_center[0] + (i - 1.5) * 0.048 for i in range(4)], dtype=np.float32)
    totals = {"episodes": 0, "timeouts": 0, "overforce": 0}
    preview_saved = False
    recorded_video_path: Path | None = None

    for episode in range(args.num_episodes):
        env.reset(seed=args.seed + episode)
        scan_start_local = np.asarray([cfg.scan_start_xy[0], cfg.scan_start_xy[1]], dtype=np.float32)
        scan_end_local = np.asarray([cfg.scan_end_xy[0], cfg.scan_end_xy[1]], dtype=np.float32)
        video_writer = None
        if args.record_video and episode == args.video_episode:
            import imageio.v2 as imageio

            recorded_video_path = args.record_dir / f"episode_{episode:05d}.mp4"
            video_writer = imageio.get_writer(
                recorded_video_path,
                fps=args.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        centers = _as_numpy(unwrapped.defect_centers[0]).astype(np.float32)
        kinds = _as_numpy(unwrapped.defect_kind[0]).astype(np.int64)
        raised_centers = centers[kinds == 0]
        records: dict[str, list[np.ndarray | float | int]] = {
            "tool_pos": [], "tool_pos_local": [], "tool_quat": [], "scan_target_xy": [], "contact_force_n": [], "force_filtered_n": [],
            "tactile_depth": [], "action": [], "reward": [], "surface_label": [], "surface_kind": [],
        }
        tactile_records: dict[str, list[np.ndarray]] = {"left_tactile_rgb": [], "right_tactile_rgb": [], "left_height_map": [], "right_height_map": []}
        visual_records: dict[str, list[np.ndarray]] = {"scene_rgb": [], "scene_depth": []}
        phase = "approach"
        scan_start_step = 0
        settle_count = 0
        contact_stable_count = 0
        force_target_z = None
        surface_command = None

        for step in range(max_steps):
            pos_world, _ = unwrapped._compute_frame_pose()
            pos_local = unwrapped.board_world_to_local(pos_world)
            pos_np = _as_numpy(pos_local[0]).astype(np.float32)
            if phase == "scan":
                scan_elapsed_s = (step - scan_start_step) * float(cfg.sim.dt) * float(cfg.decimation)
                scan_distance = min(
                    scan_elapsed_s * cfg.scan_speed_m_s,
                    float(scan_end_local[0] - scan_start_local[0]),
                )
                progress = scan_distance / max(float(scan_end_local[0] - scan_start_local[0]), 1.0e-6)
            else:
                progress = 0.0
            desired_x = float(scan_start_local[0] + (scan_end_local[0] - scan_start_local[0]) * progress) if phase == "scan" else float(scan_start_local[0])
            desired_y = float(scan_start_local[1])
            metrics = unwrapped.record_metrics()
            force_raw = float(metrics["contact_force_n"])
            force = float(metrics["force_filtered_n"])
            visual = unwrapped.visual_record() if args.save_visual or args.save_scene_preview or video_writer else {}
            if 0.5 <= force <= cfg.target_force_n + 0.8:
                contact_stable_count += 1
            else:
                contact_stable_count = 0
            # Keep the clearance target available to all state transitions.
            board_clearance_z = cfg.board_top_z + cfg.probe_contact_offset_m + 0.035
            # The rigid contact backend can report a one-frame impulse rather
            # than a gradual 3 N ramp.  Switch to the lateral position servo
            # on the first confirmed contact, otherwise pose IK can drift in Y
            # during the impulse and never satisfy the old stable-count gate.
            xy_error = np.asarray([desired_x - pos_np[0], desired_y - pos_np[1]], dtype=np.float32)
            xy_action = np.clip(xy_error / 0.02, -1.0, 1.0) if phase != "scan" else np.asarray([0.65, 0.0], dtype=np.float32)
            # First move to a fixed clearance and let the high-PD arm settle.
            # Then descend with a small absolute target increment.  This avoids
            # accumulating velocity from repeatedly changing a 1 mm target,
            # which previously caused a 12 mm single-step collision.
            if phase == "approach":
                z_action = np.clip((board_clearance_z - pos_np[2]) / cfg.action_position_scale_m, -1.0, 1.0)
                if abs(float(pos_np[2]) - board_clearance_z) < 0.003:
                    settle_count += 1
                else:
                    settle_count = 0
                if settle_count >= 15:
                    phase = "descend"
                    descent_target_z = float(pos_np[2])
            elif phase == "descend":
                z_action = 0.0
                descent_target_z = max(
                    # 3 N at the virtual 1000 N/m contact stiffness requires
                    # 3 mm penetration below the probe's zero-force height.
                    float(cfg.board_top_z + cfg.probe_contact_offset_m - cfg.target_force_n / cfg.virtual_contact_stiffness_n_per_m),
                    float(descent_target_z) - 0.00008,
                )
                if force_raw > 0.05:
                    phase = "scan"
                    scan_start_step = step
                    surface_now = float(
                        _as_numpy(
                            unwrapped.board_local_surface_height(
                                torch.as_tensor([[desired_x, desired_y]], device=unwrapped.device)
                            )[0]
                        )
                    )
                    preview_x = min(desired_x + cfg.surface_preview_distance_m, float(scan_end_local[0]))
                    surface_ahead = float(
                        _as_numpy(
                            unwrapped.board_local_surface_height(
                                torch.as_tensor([[preview_x, desired_y]], device=unwrapped.device)
                            )[0]
                        )
                    )
                    # Follow the surface that is about to arrive at the probe.  Using
                    # the upcoming height also anticipates a groove instead of only
                    # pre-lifting for bumps.
                    force_target_z = surface_ahead + cfg.probe_contact_offset_m - cfg.target_force_n / cfg.virtual_contact_stiffness_n_per_m
            else:
                # Positive z is upward; below-target force moves down.
                z_action = np.clip(-args.force_kp * (cfg.target_force_n - force), -0.25, 0.25)
            if phase == "approach":
                target_x, target_y = float(scan_start_local[0]), float(scan_start_local[1])
                target_z = board_clearance_z
            elif phase == "descend":
                target_x, target_y = float(scan_start_local[0]), float(scan_start_local[1])
                target_z = float(descent_target_z)
            else:
                # Use the exact local surface height as the feed-forward
                # normal target, then apply only a small filtered-force trim.
                surface_now = float(
                    _as_numpy(
                        unwrapped.board_local_surface_height(
                            torch.as_tensor([[desired_x, desired_y]], device=unwrapped.device)
                        )[0]
                    )
                )
                preview_x = min(desired_x + cfg.surface_preview_distance_m, float(scan_end_local[0]))
                surface_ahead = float(
                    _as_numpy(
                        unwrapped.board_local_surface_height(
                            torch.as_tensor([[preview_x, desired_y]], device=unwrapped.device)
                        )[0]
                    )
                )
                # Symmetric preview: move toward the upcoming surface height for both
                # positive (bump) and negative (groove) height changes.  Limit the
                # commanded height change per physics step so the rigid step in the
                # geometry is not converted into a collision impulse.
                if surface_command is None:
                    surface_command = surface_now
                surface_command += float(np.clip(surface_ahead - surface_command, -0.00008, 0.00008))
                surface = surface_command
                force_trim = float(np.clip((force - cfg.target_force_n) / cfg.virtual_contact_stiffness_n_per_m, -0.004, 0.004))
                # If measured force is below target, lower the probe; if it is above,
                # raise it.  The previous sign inverted this feedback.
                desired_z = surface + cfg.probe_contact_offset_m - cfg.target_force_n / cfg.virtual_contact_stiffness_n_per_m + force_trim
                # Move from the measured pose toward the force target in small
                # increments. This keeps the normal command continuous at a
                # groove/bump edge and prevents a single-step penetration jump.
                target_z = float(pos_np[2] + np.clip(desired_z - float(pos_np[2]), -0.0005, 0.0005))
                target_x = desired_x
                # Independent local-Y correction: amplify the lateral error
                # without changing the X feed-forward or normal force loop.
                target_y = desired_y - 0.5 * float(pos_np[1] - desired_y)
            target_local = torch.as_tensor([target_x, target_y, target_z], device=unwrapped.device, dtype=torch.float32).view(1, 3)
            target_pos = unwrapped.board_local_to_world(target_local)
            # Use absolute-pose IK to approach, then the board-frame
            # resolved-rate servo for the long scan.  The latter keeps the
            # lateral rail stable while the local target supplies the normal
            # force feed-forward height.
            unwrapped.set_external_target(target_pos, use_resolved_rate=(phase != "approach"))
            action_np = np.asarray([xy_action[0], xy_action[1], z_action, 0.0], dtype=np.float32)
            action_np[:2] = np.clip(action_np[:2], -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(torch.as_tensor(action_np, device=unwrapped.device).view(1, 4))
            done = bool(_as_numpy(terminated)[0] or _as_numpy(truncated)[0])
            # DirectRLEnv resets a terminated environment inside ``step``.
            # Keep the pre-reset sample as the final record; otherwise the
            # next episode's initial pose contaminates force and trajectory
            # statistics for this episode.
            metrics_after = metrics if done else unwrapped.record_metrics()

            if args.save_scene_preview and not preview_saved:
                frame = visual.get("scene_rgb")
                if frame is not None and frame.size and float(np.asarray(frame).std()) > 1.0:
                    from PIL import Image

                    Image.fromarray(np.asarray(frame)[..., :3]).save(args.record_dir / "scene_preview.png")
                    preview_saved = True

            if video_writer is not None and step % args.video_stride == 0:
                frame = visual.get("scene_rgb")
                if frame is not None and frame.size and float(np.asarray(frame).std()) > 1.0:
                    video_writer.append_data(
                        _annotate_video_frame(
                            np.asarray(frame),
                            step=step,
                            force_n=float(metrics_after["contact_force_n"]),
                            target_force_n=cfg.target_force_n,
                        )
                    )
            tool_world = torch.as_tensor(metrics_after["tool_pos"], device=unwrapped.device).view(1, 3)
            tool_local = _as_numpy(unwrapped.board_world_to_local(tool_world)[0])
            label, kind = _surface_label(tool_local[:2], raised_centers, groove_x)
            for key in ("tool_pos", "tool_quat", "scan_target_xy"):
                records[key].append(np.asarray(metrics_after[key], dtype=np.float32))
            records["tool_pos_local"].append(tool_local.astype(np.float32))
            records["contact_force_n"].append(float(metrics_after["contact_force_n"]))
            records["force_filtered_n"].append(float(metrics_after["force_filtered_n"]))
            records["tactile_depth"].append(float(metrics_after["tactile_depth"]))
            records["action"].append(action_np)
            records["reward"].append(float(_as_numpy(reward)[0]))
            records["surface_label"].append(label)
            records["surface_kind"].append(kind)
            if args.save_tactile:
                tactile = unwrapped.tactile_record()
                for key, value in tactile.items():
                    if key in tactile_records:
                        tactile_records[key].append(np.asarray(value))
            if args.save_visual:
                for key, value in visual.items():
                    if key in visual_records:
                        visual_records[key].append(np.asarray(value))
            if done:
                totals["timeouts"] += int(bool(_as_numpy(truncated)[0]))
                totals["overforce"] += int(bool(_as_numpy(terminated)[0]))
                break

        if video_writer is not None:
            video_writer.close()

        payload = {key: np.asarray(value) for key, value in records.items()}
        payload["defect_centers_xy"] = centers
        payload["defect_kind"] = kinds
        payload["groove_x"] = groove_x
        payload["episode_index"] = np.asarray(episode, dtype=np.int32)
        payload["board_translation"] = _as_numpy(unwrapped.board_translation[0]).astype(np.float32)
        payload["board_quat"] = _as_numpy(unwrapped.board_quat[0]).astype(np.float32)
        if args.save_tactile:
            payload.update({key: np.asarray(value) for key, value in tactile_records.items() if value})
        if args.save_visual:
            payload.update({key: np.asarray(value) for key, value in visual_records.items() if value})
        np.savez_compressed(args.record_dir / f"episode_{episode:05d}.npz", **payload)
        totals["episodes"] += 1

    env.close()
    print(f"Saved {totals['episodes']} episodes to {args.record_dir}")
    print(f"terminal overforce={totals['overforce']} timeout={totals['timeouts']}")
    if recorded_video_path is not None:
        print(f"video={recorded_video_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
