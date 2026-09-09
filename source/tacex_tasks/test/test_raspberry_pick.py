from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "raspberry_pick"


def read(name: str) -> str:
    return (TASK_ROOT / name).read_text(encoding="utf-8")


def test_raspberry_task_is_registered_and_imported():
    package = read("__init__.py")
    root_package = (TASK_ROOT.parent / "__init__.py").read_text(encoding="utf-8")
    assert 'id="TacEx-RaspberryPick-Force-v0"' in package
    assert "from . import raspberry_pick as _raspberry_pick" in root_package


def test_raspberry_scene_has_real_force_limits_and_layered_grove():
    cfg = read("raspberry_pick_env_cfg.py")
    assert 'prim_path="/World/envs/env_.*/target_raspberry"' in cfg
    assert "detach_force_threshold_n: float = 1.40" in cfg
    assert "damage_grip_force_threshold_n: float = 2.80" in cfg
    assert 'filter_prim_paths_expr=["/World/envs/env_.*/target_raspberry"]' in cfg
    assert "_background_assets()" in cfg
    assert "orchard_cane_" in cfg
    assert "background_raspberries = _background_raspberry_specs()" in cfg
    assert "foliage_mass_" not in cfg
    assert "_static_sphere" not in cfg
    assert "leaf_visuals = _leaf_specs()" in cfg
    assert (TASK_ROOT / "assets" / "raspberry_grove_cc0.jpg").is_file()


def test_target_visual_uses_drupelets_and_calyx():
    env = read("raspberry_pick_env.py")
    assert "_spawn_target_raspberry_visual" in env
    assert "drupelet_" in env
    assert "RaspberryDrupelet" in env
    assert "calyx_leaf_" in env
    assert "_spawn_serrated_leaf" in env
    assert "_spawn_background_raspberries" in env
    assert "_spawn_textured_backdrops" in env
    assert "self.raspberry = self.grape" in env


def test_collection_script_records_real_trajectory_and_video():
    script = (
        ROOT / "scripts" / "demos" / "raspberry_pick" / "collect_dataset_video.py"
    ).read_text(encoding="utf-8")
    assert "RaspberryTrajectoryBuffer" in script
    assert "env.command_expert()" in script
    assert "buffer.append(" in script
    assert "imageio.get_writer" in script
    assert 'env.get_camera_rgb("third")' in script
    assert 'env.get_camera_rgb("wrist")' in script
    assert "paper_first_video_path" in script
    assert "paper_third_video_path" in script
    assert "pad grip" in script
    assert "pedicle" in script


def test_raspberry_bc_entrypoints_use_shared_dinov3_flow_pipeline():
    collection = (
        ROOT / "scripts" / "demos" / "raspberry_pick" / "collect_bc_dataset.py"
    ).read_text(encoding="utf-8")
    training = (
        ROOT
        / "scripts"
        / "reinforcement_learning"
        / "raspberry_pick"
        / "train_raspberry_dinov3_flow_bc.py"
    ).read_text(encoding="utf-8")
    evaluation = (
        ROOT
        / "scripts"
        / "reinforcement_learning"
        / "raspberry_pick"
        / "eval_raspberry_dinov3_flow_bc.py"
    ).read_text(encoding="utf-8")
    for script in (collection, training, evaluation):
        assert 'os.environ["TACEX_PICK_VARIANT"] = "raspberry"' in script
        assert "runpy.run_path" in script
    assert "collect_dataset.py" in collection
    assert "train_grape_dinov3_flow_bc.py" in training
    assert "eval_grape_dinov3_flow_bc.py" in evaluation


def test_shared_bc_pipeline_declares_raspberry_contract_and_physical_ranges():
    collector = (
        ROOT / "scripts" / "demos" / "grape_pick" / "collect_dataset.py"
    ).read_text(encoding="utf-8")
    trainer = (
        ROOT
        / "scripts"
        / "reinforcement_learning"
        / "grape_pick"
        / "train_grape_dinov3_flow_bc.py"
    ).read_text(encoding="utf-8")
    evaluator = (
        ROOT
        / "scripts"
        / "reinforcement_learning"
        / "grape_pick"
        / "eval_grape_dinov3_flow_bc.py"
    ).read_text(encoding="utf-8")
    assert "RaspberryTrajectoryBuffer as GrapeTrajectoryBuffer" in collector
    assert 'TASK_ID = "TacEx-RaspberryPick-Force-v0"' in collector
    assert 'CONTRACT = "raspberry_dinov3_tactile_flow_matching_bc_v1"' in trainer
    assert 'BEST_CHECKPOINT_FILENAME = "raspberry_dinov3_flow_bc.pt"' in trainer
    assert 'TASK = "TacEx-RaspberryPick-Force-v0"' in evaluator
    assert 'OBJECT_NAME = "raspberry"' in evaluator
    assert "default=(2.4, 3.1) if IS_RASPBERRY" in evaluator
    assert "default=(1.2, 1.6) if IS_RASPBERRY" in evaluator
    assert 'choices=("zero", "gaussian")' in evaluator
    assert '"checkpoint_weights": "ema" if args.use_ema else "raw_best_epoch"' in evaluator
