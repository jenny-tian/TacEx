from pathlib import Path
import importlib.util

import torch


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "grape_pick"


def read(name: str) -> str:
    return (TASK_ROOT / name).read_text(encoding="utf-8")


def load_physics_module():
    spec = importlib.util.spec_from_file_location("grape_pick_physics", TASK_ROOT / "physics.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tether_is_tension_only_and_reaches_three_newtons():
    compute_stem_tether_force = load_physics_module().compute_stem_tether_force

    anchor = torch.zeros((1, 3))
    position = torch.tensor([[0.0, 0.0, 0.12]])
    velocity = torch.zeros((1, 3))
    force, tension, extension = compute_stem_tether_force(
        position,
        velocity,
        anchor,
        rest_length_m=0.10,
        stiffness_n_per_m=150.0,
        damping_n_s_per_m=0.0,
        max_tension_n=8.0,
    )
    torch.testing.assert_close(extension, torch.tensor([0.02]))
    torch.testing.assert_close(tension, torch.tensor([3.0]), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(force, torch.tensor([[0.0, 0.0, -3.0]]), atol=1.0e-5, rtol=1.0e-5)

    compressed_force, compressed_tension, _ = compute_stem_tether_force(
        torch.tensor([[0.0, 0.0, 0.08]]),
        velocity,
        anchor,
        rest_length_m=0.10,
        stiffness_n_per_m=150.0,
        damping_n_s_per_m=0.0,
        max_tension_n=8.0,
    )
    assert compressed_tension.item() == 0.0
    assert torch.count_nonzero(compressed_force).item() == 0


def test_threshold_requires_consecutive_dwell_steps():
    update_threshold_dwell = load_physics_module().update_threshold_dwell

    counter = torch.zeros(1, dtype=torch.long)
    for expected in (1, 2):
        counter, triggered = update_threshold_dwell(counter, torch.tensor([3.1]), 3.0, required_steps=3)
        assert counter.item() == expected
        assert not triggered.item()
    counter, triggered = update_threshold_dwell(counter, torch.tensor([3.1]), 3.0, required_steps=3)
    assert triggered.item()
    counter, triggered = update_threshold_dwell(counter, torch.tensor([2.9]), 3.0, required_steps=3)
    assert counter.item() == 0
    assert not triggered.item()

    counters = torch.zeros(2, dtype=torch.long)
    counters, triggered = update_threshold_dwell(
        counters,
        torch.tensor([3.2, 3.2]),
        torch.tensor([3.0, 3.4]),
        required_steps=1,
    )
    assert triggered.tolist() == [True, False]


def test_task_uses_real_pad_contacts_without_virtual_force_fallback():
    root_package = (TASK_ROOT.parent / "__init__.py").read_text(encoding="utf-8")
    cfg = read("grape_pick_env_cfg.py")
    env = read("grape_pick_env.py")
    package = read("__init__.py")

    assert "from . import grape_pick as _grape_pick" in root_package
    assert 'id="TacEx-GrapePick-Force-v0"' in package
    assert "detach_force_threshold_n: float = 3.0" in cfg
    assert "damage_grip_force_threshold_n: float = 5.0" in cfg
    assert "damage_force_randomization_range_n: tuple[float, float] = (3.40, 5.00)" in cfg
    assert "detach_force_randomization_range_n: tuple[float, float] = (2.80, 3.40)" in cfg
    assert 'prim_path="/World/envs/env_.*/Robot/gelpad_left"' in cfg
    assert 'prim_path="/World/envs/env_.*/Robot/gelpad_right"' in cfg
    assert 'filter_prim_paths_expr=["/World/envs/env_.*/target_grape"]' in cfg
    assert "def get_contact_force_metrics(self)" in env
    assert "sensor.data.force_matrix_w" in env
    assert "left_normal_force_n" in env
    assert "right_normal_force_n" in env
    assert "there is no tactile fallback" in env
    assert "_estimate_contact_forces_from_tactile" not in env
    assert "virtual_contact" not in env
    assert "self.grape.set_external_force_and_torque" in env
    assert '"stem_tension_n"' in env
    assert '"crushed"' in env
    assert '"slipped"' in env
    assert '"no_detach"' in env


def test_horizontal_vine_randomization_and_obstacle_guard_are_configured():
    cfg = read("grape_pick_env_cfg.py")
    env = read("grape_pick_env.py")
    dataset = read("dataset.py")

    assert "randomize_grape_pose: bool = True" in cfg
    assert "grape_position_randomization_m: tuple[float, float, float] = (0.020, 0.020, 0.035)" in cfg
    assert "stem_anchor_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.090)" in cfg
    assert 'prim_path="/World/envs/env_.*/vine_trunk"' in cfg
    assert 'prim_path="/World/envs/env_.*/target_lateral_branch"' in cfg
    assert 'spawn=_static_cylinder(0.014, 0.86, WOOD_BROWN, axis="Z")' in cfg
    assert 'spawn=_static_cylinder(0.007, lateral_branch_length_m, STEM_GREEN, axis="X")' in cfg
    assert "obstacle_contact_threshold_n: float = 0.25" in cfg
    for obstacle in ("vine_trunk", "target_lateral_branch", "trellis_left", "trellis_right"):
        assert f'/World/envs/env_.*/{obstacle}' in cfg

    assert "(0.0, 0.731689, 0.0, 0.681639)" in env
    assert "center_target_b[approach_mask, 0] -= self.cfg.expert_approach_distance_m" in env
    assert "self.pull_direction_b[pull_mask]" in env
    for body in (
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
        "gelsight_mini_case_left",
        "gelsight_mini_case_right",
        "gelpad_left",
        "gelpad_right",
    ):
        assert f'"{body}"' in env
    assert "self.collided_obstacle |= obstacle_contact_force_n >= self.cfg.obstacle_contact_threshold_n" in env
    assert "& ~self.collided_obstacle" in env

    assert "SCHEMA_VERSION = 2" in dataset
    assert '"obstacle_contact_force_n"' in dataset
    assert '"sampled_grape_position_offset_m"' in dataset
    assert "Successful episode contains a robot-to-vine/trellis collision" in dataset
