from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect CAFE-compatible BC demonstrations for LabPick.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=100)
parser.add_argument("--labware", choices=("slide", "coverslip", "cup"), default="slide")
parser.add_argument("--dataset_file", type=str, default="/home/tjx/TacEx/datasets/lab_pick_slide_bc.hdf5")
parser.add_argument("--success_only", action="store_true")
parser.add_argument("--max_episode_steps", type=int, default=960)
parser.add_argument("--freq_ratio", type=int, default=3)
parser.add_argument("--include_marker", action="store_true")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tacex_tasks.lab_pick.bc_dataset import CafeHdf5Writer
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg


def main():
    env_cfg = LabPickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.labware_name = args_cli.labware
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = LabPickEnv(env_cfg, render_mode="rgb_array")
    writer = CafeHdf5Writer(args_cli.dataset_file, freq_ratio=args_cli.freq_ratio, include_marker=args_cli.include_marker)
    recorded = 0

    try:
        while simulation_app.is_running() and recorded < args_cli.num_demos:
            env.reset()
            for _ in range(args_cli.max_episode_steps):
                env.command_pick_state_machine()
                obs = env.get_cafe_observation()
                action = env.get_cafe_action()
                writer.append_high_step(obs, action)
                if int(env.step_count[0].item()) % args_cli.freq_ratio == 0:
                    writer.append_low_step(env.get_cafe_image(), action)

                env._pre_physics_step(None)
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                env.sim.render()

                terminated, time_out = env._get_dones()
                done = bool((terminated | time_out)[0].item())

                lift_delta = env.labware.data.root_pos_w[:, 2] - env.initial_object_height
                success = bool((lift_delta[0] > env.cfg.success_lift_height).item())
                if done or success:
                    exported = writer.flush_episode(
                        success=success,
                        labware_reset_pos_w=env.labware_reset_pos_w,
                        labware_reset_quat_w=env.labware_reset_quat_w,
                        success_only=args_cli.success_only,
                    )
                    if exported:
                        recorded += 1
                        print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success={success}")
                    break
            else:
                exported = writer.flush_episode(
                    success=False,
                    labware_reset_pos_w=env.labware_reset_pos_w,
                    labware_reset_quat_w=env.labware_reset_quat_w,
                    success_only=args_cli.success_only,
                )
                if exported:
                    recorded += 1
                    print(f"[INFO] recorded_demo={recorded}/{args_cli.num_demos} success=False")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
