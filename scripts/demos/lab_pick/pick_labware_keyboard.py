from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard-control labware grasping with a Franka/GelSight gripper.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--labware",
    choices=("slide", "coverslip", "cup"),
    default="slide",
    help="Object to spawn and pick: microscope slide, cover slip, or simplified glass cup proxy.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Simulation duration in seconds. Use 0 to run until closed.")
parser.add_argument("--save_camera_images", action="store_true", help="Save RGB images from both cameras every second.")
parser.add_argument("--save_tactile_images", action="store_true", help="Save left/right GelSight tactile RGB images.")
parser.add_argument("--record_video", action="store_true", help="Record camera video to an mp4 file.")
parser.add_argument(
    "--video_camera",
    choices=("third", "wrist", "viewer", "tactile_left", "tactile_right"),
    default="third",
    help="Camera source for mp4.",
)
parser.add_argument("--video_every_n_steps", type=int, default=4, help="Record one video frame every N sim steps.")
parser.add_argument("--video_fps", type=int, default=30, help="Output video FPS.")
parser.add_argument("--print_state_interval", type=int, default=60, help="Print robot/object state every N sim steps.")
parser.add_argument("--pos_sensitivity", type=float, default=0.0015, help="Per-step Cartesian motion while a key is held.")
parser.add_argument("--rot_sensitivity", type=float, default=0.02, help="Per-step rotation command while a key is held.")
parser.add_argument("--open_width", type=float, default=0.04, help="Finger joint target for an open gripper.")
parser.add_argument("--close_width", type=float, default=0.0, help="Finger joint target for a closed gripper.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio

from isaaclab.devices import Se3Keyboard

from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg


def run_simulator(env: LabPickEnv):
    print(f"[INFO] Starting keyboard labware pick demo: labware={env.labware_name}, envs={env.num_envs}")
    env.reset()
    env.reset_keyboard_target(open_width=args_cli.open_width)

    should_reset = False
    forced_gripper_closed: bool | None = None

    def request_reset():
        nonlocal should_reset
        should_reset = True

    def force_open_gripper():
        nonlocal forced_gripper_closed
        forced_gripper_closed = False
        print("[INFO] Gripper command: open")

    def force_close_gripper():
        nonlocal forced_gripper_closed
        forced_gripper_closed = True
        print("[INFO] Gripper command: close")

    teleop_interface = Se3Keyboard(pos_sensitivity=args_cli.pos_sensitivity, rot_sensitivity=args_cli.rot_sensitivity)
    teleop_interface.add_callback("R", request_reset)
    teleop_interface.add_callback("O", force_open_gripper)
    teleop_interface.add_callback("P", force_close_gripper)
    teleop_interface.reset()
    print(teleop_interface)
    print("[INFO] Extra keys: O open gripper, P close gripper, R reset scene, L clear current keyboard command.")
    print("[INFO] Click the Isaac Sim viewport first so keyboard events are captured.")

    max_steps = int(args_cli.duration / env.physics_dt) if args_cli.duration > 0.0 else None
    output_dir = Path.cwd() / "logs" / "lab_pick" / env.labware_name
    save_interval = max(1, int(1.0 / env.physics_dt))
    video_writer = None
    if args_cli.record_video:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{args_cli.video_camera}_camera.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=args_cli.video_fps, macro_block_size=1)
        print(f"[INFO] Recording {args_cli.video_camera} camera video to: {video_path}")

    try:
        while simulation_app.is_running() and (max_steps is None or int(env.step_count[0].item()) < max_steps):
            if should_reset:
                env.reset()
                env.reset_keyboard_target(open_width=args_cli.open_width)
                teleop_interface.reset()
                should_reset = False
                forced_gripper_closed = None

            delta_pose, close_gripper = teleop_interface.advance()
            if forced_gripper_closed is not None:
                close_gripper = forced_gripper_closed
            env.command_keyboard(
                delta_pose,
                close_gripper,
                open_width=args_cli.open_width,
                close_width=args_cli.close_width,
            )
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env.sim.render()

            step = int(env.step_count[0].item())
            if args_cli.print_state_interval > 0 and step % args_cli.print_state_interval == 0:
                env.print_state()

            if args_cli.save_camera_images and step % save_interval == 0:
                env.save_camera_images(output_dir)

            if args_cli.save_tactile_images and step % save_interval == 0:
                env.save_tactile_images(output_dir)

            if video_writer is not None and step % args_cli.video_every_n_steps == 0:
                video_writer.append_data(env.get_video_frame(args_cli.video_camera))
    finally:
        if video_writer is not None:
            video_writer.close()

    final_height = env.labware.data.root_pos_w[:, 2]
    lifted = final_height - env.initial_object_height
    print(f"[RESULT] object_lift_delta_z={lifted[0].item():.4f} m, lifted={lifted[0].item() > 0.03}")
    env.close()


def main():
    env_cfg = LabPickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.labware_name = args_cli.labware
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = LabPickEnv(env_cfg, render_mode="rgb_array")
    print("[INFO] Setup complete.")
    run_simulator(env)


if __name__ == "__main__":
    main()
    simulation_app.close()
