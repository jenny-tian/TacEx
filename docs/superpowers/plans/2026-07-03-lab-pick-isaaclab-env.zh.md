# Lab Pick IsaacLab 环境实施计划

> **给 agentic workers：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项执行本计划。步骤使用 checkbox（`- [ ]`）追踪进度。

**目标：** 把当前 `scripts/demos/lab_pick` 原型整理成可复用、可注册、可测试的 IsaacLab `DirectRLEnv`，补齐 terminate 函数、reset 函数、载玻片位置随机化、BC 数据采集脚本和 IsaacLab 启动脚本。

**架构：** 将环境配置和环境逻辑从 demo 脚本移入 `tacex_tasks.lab_pick` 包。通过 `gym.register` 注册 IsaacLab task id，demo 脚本只负责 CLI、`AppLauncher`、录像、手动/自动控制和数据采集入口。终止逻辑 `_get_dones()`、重置逻辑 `_reset_idx()`、reset 随机化、BC 观测/动作导出全部放在环境模块中，保证采集数据和后续训练使用同一个环境定义。

**技术栈：** IsaacLab `DirectRLEnv`、Gymnasium 注册、TacEx GelSight 传感器、Franka/GelSight rigid gripper、PyTorch tensor、pytest、现有 Conda 环境 `/home/tjx/miniforge3/envs/env_isaaclab`。

---

## 已知上下文和假设

- 当前会话无法直接读取对话 `019f1b90-d72e-7271-a285-4f8121f5bd06` 的历史内容，本计划基于本机 `/home/tjx/TacEx` 当前代码状态编写。
- 现有相关原型：
  - `scripts/demos/lab_pick/pick_labware.py`
  - `scripts/demos/lab_pick/pick_labware_keyboard.py`
- 需要保留的能力：
  - labware 类型：`slide`、`coverslip`、`cup`
  - Franka + GelSight gripper
  - wrist camera、third-person camera
  - 左右 GelSight tactile 输出
  - 自动 pick 状态机
  - 键盘遥操作
  - 可选保存相机图片、触觉图片、mp4 视频
- BC 训练需要的不只是 terminate/reset。还必须有：
  - reset 时载玻片初始位置随机化
  - 明确的 policy observation
  - 明确的 expert action 表达
  - 与 `SilkyFinish/ForceCapture-CAFE` 一致的多频 HDF5 数据集导出
  - 成功/失败 episode 过滤
  - 采集命令和 smoke test
- ForceCapture-CAFE 数据格式要求：
  - root：`/data/demo_i`
  - 高频 action：`/data/demo_i/actions/high`，shape `(N_high, 10)`，float32
  - 低频 action：`/data/demo_i/actions/low`，shape `(N_low, 10)`，float32
  - 高频低维 obs：`/data/demo_i/obs/robot0_pos`，shape `(N_high, 10)`，float32
  - 高频力/触觉 obs：`/data/demo_i/obs/robot0_force`，shape `(N_high, 4)`，float32
  - 低频图像 obs：`/data/demo_i/obs/robot0_image`，shape `(N_low, 224, 224, 3)`，uint8
  - 可选高频 marker obs：`/data/demo_i/obs/robot0_marker2d`，shape `(N_high, 728)`，float32
  - demo attrs：`length_high`、`length_low`、`freq_ratio`
  - file attrs：`num_demos`、`include_images`、`freq_ratio`、`high_freq_obs_keys`、`low_freq_obs_keys`、`high_freq_action_key`、`low_freq_action_key`
  - `robot0_pos` 和 action 的 10 维含义为 `xyz(3) + rot6d(6) + gripper_width(1)`。
  - `robot0_force` 的最后一维是 CAFE 训练代码里的 `C_t` 标签；仿真中先用 `[left_touch, right_touch, mean_touch, contact_tag]` 作为兼容代理特征。
- IsaacLab `DirectRLEnv` 中 terminate 函数应实现为 `_get_dones()`，reset 函数应实现为 `_reset_idx()`。

## 文件结构

- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`
  - 定义 `LabPickEnvCfg`、`LabPickSlideEnvCfg`、`LabPickCoverslipEnvCfg`、`LabPickCupEnvCfg`
  - 放置 sim、scene、robot、labware、camera、GelSight、termination threshold 配置
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`
  - 定义 `LabPickEnv`
  - 实现 `_get_dones()`、`_reset_idx()`、状态机控制、键盘控制、相机/触觉保存工具
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py`
  - 封装 ForceCapture-CAFE 兼容 HDF5 writer，保存多频 observation/action、success、初始随机化参数和 episode metadata
- 新建：`scripts/demos/lab_pick/collect_bc_dataset.py`
  - 自动状态机批量采集 BC 数据；支持随机 seed、demo 数量、输出路径、只保存成功 episode
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/__init__.py`
  - 注册 `TacEx-LabPick-Slide-Direct-v0`
  - 注册 `TacEx-LabPick-Coverslip-Direct-v0`
  - 注册 `TacEx-LabPick-Cup-Direct-v0`
- 修改：`scripts/demos/lab_pick/pick_labware.py`
  - 改成自动 pick 的薄启动脚本
- 修改：`scripts/demos/lab_pick/pick_labware_keyboard.py`
  - 改成键盘遥操作的薄启动脚本
- 新建：`source/tacex_tasks/test/test_lab_pick_static.py`
  - 快速静态测试，检查注册、配置、terminate、reset、脚本结构

---

### Task 1：先写静态测试

**文件：**
- 新建：`source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] **Step 1：写失败测试**

创建 `source/tacex_tasks/test/test_lab_pick_static.py`：

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "source" / "tacex_tasks" / "tacex_tasks" / "lab_pick"
SCRIPT_ROOT = ROOT / "scripts" / "demos" / "lab_pick"


def read(path: Path) -> str:
    return path.read_text()


def test_lab_pick_package_registers_three_labware_tasks():
    source = read(TASK_ROOT / "__init__.py")
    assert 'id="TacEx-LabPick-Slide-Direct-v0"' in source
    assert 'id="TacEx-LabPick-Coverslip-Direct-v0"' in source
    assert 'id="TacEx-LabPick-Cup-Direct-v0"' in source
    assert 'entry_point=f"{__name__}.lab_pick_env:LabPickEnv"' in source
    assert '"env_cfg_entry_point": LabPickSlideEnvCfg' in source
    assert '"env_cfg_entry_point": LabPickCoverslipEnvCfg' in source
    assert '"env_cfg_entry_point": LabPickCupEnvCfg' in source


def test_lab_pick_cfg_defines_scene_assets_and_termination_thresholds():
    source = read(TASK_ROOT / "lab_pick_env_cfg.py")
    assert "class LabPickEnvCfg(DirectRLEnvCfg):" in source
    assert "class LabPickSlideEnvCfg(LabPickEnvCfg):" in source
    assert "class LabPickCoverslipEnvCfg(LabPickEnvCfg):" in source
    assert "class LabPickCupEnvCfg(LabPickEnvCfg):" in source
    assert 'labware_name = "slide"' in source
    assert 'labware_name = "coverslip"' in source
    assert 'labware_name = "cup"' in source
    assert "terminate_object_drop_height: float = 0.010" in source
    assert "terminate_object_xy_distance: float = 0.30" in source
    assert "success_lift_height: float = 0.030" in source
    assert "FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG" in source
    assert "GelSightMiniCfg" in source
    assert "TiledCameraCfg" in source


def test_lab_pick_env_implements_dones_and_reset():
    source = read(TASK_ROOT / "lab_pick_env.py")
    assert "class LabPickEnv(DirectRLEnv):" in source
    assert "def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:" in source
    assert "terminated = object_dropped | object_too_far | ee_outside_workspace" in source
    assert "time_out = self.episode_length_buf >= self.max_episode_length - 1" in source
    assert "return terminated, time_out" in source
    assert "def _reset_idx(self, env_ids: torch.Tensor | None):" in source
    assert "self.labware.write_root_state_to_sim(root_state, env_ids=env_ids)" in source
    assert "self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)" in source
    assert "self.gsmini_left.reset(env_ids=env_ids)" in source
    assert "self.gsmini_right.reset(env_ids=env_ids)" in source
    assert "self.step_count[env_ids] = 0" in source


def test_launch_scripts_are_thin_and_import_shared_env():
    scripted = read(SCRIPT_ROOT / "pick_labware.py")
    keyboard = read(SCRIPT_ROOT / "pick_labware_keyboard.py")
    for source in (scripted, keyboard):
        assert "from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv" in source
        assert "from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg" in source
        assert "class LabPickEnv(" not in source
        assert "class LabPickEnvCfg(" not in source
        assert "AppLauncher.add_app_launcher_args(parser)" in source
```

- [ ] **Step 2：运行测试，确认失败原因正确**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：失败，原因是 `source/tacex_tasks/tacex_tasks/lab_pick/` 相关文件尚未创建。

- [ ] **Step 3：提交失败测试**

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "test: add lab pick environment checks"
```

---

### Task 2：创建 IsaacLab 环境配置

**文件：**
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`

- [ ] **Step 1：抽出 `LabPickEnvCfg`**

从 `scripts/demos/lab_pick/pick_labware.py` / `pick_labware_keyboard.py` 中迁移配置，保留已有场景、物体、相机和 GelSight 配置。

`LabPickEnvCfg` 必须包含：

```python
@configclass
class LabPickEnvCfg(DirectRLEnvCfg):
    labware_name: str = "slide"
    terminate_object_drop_height: float = 0.010
    terminate_object_xy_distance: float = 0.30
    terminate_ee_workspace_margin: float = 0.05
    success_lift_height: float = 0.030
    tactile_threshold_mm: float = 0.0
    randomize_labware_position: bool = True
    labware_pos_randomization_xy: tuple[float, float] = (0.020, 0.010)
    labware_yaw_randomization: float = 0.20

    decimation = 1
    episode_length_s = 8.0
    action_space = 0
    observation_space = 1
    state_space = 0
```

配置模块还必须包含三个具体 task cfg：

```python
@configclass
class LabPickSlideEnvCfg(LabPickEnvCfg):
    labware_name = "slide"


@configclass
class LabPickCoverslipEnvCfg(LabPickEnvCfg):
    labware_name = "coverslip"


@configclass
class LabPickCupEnvCfg(LabPickEnvCfg):
    labware_name = "cup"
```

保留原型中的这些配置对象：

```python
viewer: ViewerCfg = ViewerCfg(...)
sim: SimulationCfg = SimulationCfg(...)
scene: InteractiveSceneCfg = InteractiveSceneCfg(...)
ground = AssetBaseCfg(...)
light = AssetBaseCfg(...)
plate = RigidObjectCfg(...)
slide = RigidObjectCfg(...)
labware_support = RigidObjectCfg(...)
coverslip = RigidObjectCfg(...)
cup = RigidObjectCfg(...)
robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(...)
wrist_camera = TiledCameraCfg(...)
third_person_camera = TiledCameraCfg(...)
gsmini_left = GelSightMiniCfg(...)
gsmini_right = gsmini_left.replace(...)
ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
```

- [ ] **Step 2：运行静态测试**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：配置相关断言通过，注册和环境实现相关断言仍失败。

- [ ] **Step 3：提交配置**

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py
git commit -m "feat: add lab pick IsaacLab config"
```

---

### Task 3：实现环境、terminate、reset 和 reset 随机化

**文件：**
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`

- [ ] **Step 1：创建 `LabPickEnv`**

以 `scripts/demos/lab_pick/pick_labware_keyboard.py` 为主迁移环境逻辑，因为它已经使用 pose IK，并包含 `reset_keyboard_target()` 和键盘控制辅助函数。

构造函数使用 `cfg.labware_name` 选择物体，不再额外传入 `labware` 参数：

```python
class LabPickEnv(DirectRLEnv):
    cfg: LabPickEnvCfg

    def __init__(self, cfg: LabPickEnvCfg, render_mode: str | None = None, **kwargs):
        self.labware_name = cfg.labware_name
        self.labware_cfg = getattr(cfg, cfg.labware_name)
        super().__init__(cfg, render_mode, **kwargs)
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.gripper_width = torch.full((self.num_envs, len(self._finger_joint_ids)), 0.04, device=self.device)
        self.initial_object_height = self.labware.data.root_pos_w[:, 2].clone()
        self.initial_object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
        self.has_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_target_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_target_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_target_quat_b[:, 0] = 1.0
        self.nominal_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.nominal_ee_quat_b[:, 0] = 1.0
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.workspace_min_b = torch.tensor([0.25, -0.35, 0.015], device=self.device)
        self.workspace_max_b = torch.tensor([0.78, 0.35, 0.50], device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
```

- [ ] **Step 2：实现 terminate 函数 `_get_dones()`**

终止条件：
- labware 掉落
- labware 相对初始位置横向偏移过远
- end-effector 超出工作空间
- IsaacLab episode timeout

代码：

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    object_pos_b = self.labware.data.root_pos_w - self._robot.data.root_link_pos_w
    object_drop_delta = self.labware.data.root_pos_w[:, 2] - self.initial_object_height
    object_dropped = object_drop_delta < -self.cfg.terminate_object_drop_height

    object_xy_delta = object_pos_b[:, :2] - self.initial_object_pos_b[:, :2]
    object_too_far = torch.linalg.norm(object_xy_delta, dim=1) > self.cfg.terminate_object_xy_distance

    ee_pos_b, _ = self._compute_frame_pose()
    workspace_min = self.workspace_min_b - self.cfg.terminate_ee_workspace_margin
    workspace_max = self.workspace_max_b + self.cfg.terminate_ee_workspace_margin
    ee_outside_workspace = torch.any((ee_pos_b < workspace_min) | (ee_pos_b > workspace_max), dim=1)

    terminated = object_dropped | object_too_far | ee_outside_workspace
    time_out = self.episode_length_buf >= self.max_episode_length - 1
    return terminated, time_out
```

- [ ] **Step 3：实现 reset 函数 `_reset_idx()`**

重置内容：
- labware root pose 和 velocity
- labware 初始位置随机化
- robot joint position 和 velocity
- IK command / gripper command
- tactile 接触状态
- 初始物体高度和相对位置
- step counter
- 左右 GelSight sensor

代码：

```python
def _reset_idx(self, env_ids: torch.Tensor | None):
    super()._reset_idx(env_ids)
    if env_ids is None:
        env_ids = self._robot._ALL_INDICES

    root_state = self.labware.data.default_root_state[env_ids].clone()
    root_state[:, :3] += self.scene.env_origins[env_ids]
    root_state[:, 7:] = 0.0

    if self.cfg.randomize_labware_position:
        xy_range = torch.tensor(self.cfg.labware_pos_randomization_xy, device=self.device)
        xy_noise = (2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0) * xy_range
        root_state[:, 0:2] += xy_noise

        yaw_range = self.cfg.labware_yaw_randomization
        yaw = (2.0 * torch.rand((len(env_ids),), device=self.device) - 1.0) * yaw_range
        yaw_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            yaw,
        )
        root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, root_state[:, 3:7])

    self.labware.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    self.has_touched[env_ids] = False
    self.initial_object_height[env_ids] = root_state[:, 2]
    self.initial_object_pos_b[env_ids] = root_state[:, :3] - self._robot.data.root_link_pos_w[env_ids]
    self.step_count[env_ids] = 0
    self.gripper_width[env_ids] = 0.04

    _, ee_quat_b = self._compute_frame_pose()
    self.nominal_ee_quat_b[env_ids] = ee_quat_b[env_ids]
    self.reset_keyboard_target(env_ids)

    self.gsmini_left.reset(env_ids=env_ids)
    self.gsmini_right.reset(env_ids=env_ids)
```

- [ ] **Step 4：保存随机化状态供数据集记录**

在 `__init__()` 中增加：

```python
self.labware_reset_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
self.labware_reset_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
self.labware_reset_quat_w[:, 0] = 1.0
```

在 `_reset_idx()` 写入 labware root state 后增加：

```python
self.labware_reset_pos_w[env_ids] = root_state[:, :3]
self.labware_reset_quat_w[env_ids] = root_state[:, 3:7]
```

这样 HDF5 数据集中能记录每条 demonstration 的初始载玻片位姿，后续排查泛化失败时可以按初始分布筛选。

- [ ] **Step 5：调整 `_apply_action()` 和 step 计数**

`step_count` 使用 per-env tensor：

```python
def _apply_action(self):
    ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
    joint_pos = self._robot.data.joint_pos[:, :]

    if torch.linalg.norm(ee_pos_curr_b) > 0.0:
        jacobian = self._compute_frame_jacobian()
        joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
    else:
        joint_pos_des = joint_pos.clone()

    joint_pos_des[:, self._finger_joint_ids] = self.gripper_width
    self._robot.set_joint_position_target(joint_pos_des)
    self.step_count += 1
```

日志、保存文件名、录像间隔中原来使用 `self.step_count` 的地方，改成 `int(self.step_count[0].item())`。

- [ ] **Step 6：运行测试**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：环境相关断言通过，注册和脚本相关断言仍失败。

- [ ] **Step 7：提交环境实现**

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py
git commit -m "feat: implement lab pick lifecycle and reset randomization"
```

---

### Task 4：注册 IsaacLab 任务

**文件：**
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/__init__.py`

- [ ] **Step 1：注册三个 task id**

创建 `source/tacex_tasks/tacex_tasks/lab_pick/__init__.py`：

```python
import gymnasium as gym

from .lab_pick_env_cfg import LabPickCoverslipEnvCfg, LabPickCupEnvCfg, LabPickSlideEnvCfg


gym.register(
    id="TacEx-LabPick-Slide-Direct-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabPickSlideEnvCfg},
)

gym.register(
    id="TacEx-LabPick-Coverslip-Direct-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabPickCoverslipEnvCfg},
)

gym.register(
    id="TacEx-LabPick-Cup-Direct-v0",
    entry_point=f"{__name__}.lab_pick_env:LabPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": LabPickCupEnvCfg},
)
```

- [ ] **Step 2：运行测试**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：只剩脚本相关断言失败。

- [ ] **Step 3：提交注册**

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/tacex_tasks/lab_pick/__init__.py
git commit -m "feat: register lab pick tasks"
```

---

### Task 5：补齐 CAFE 兼容 observation 和 expert action 接口

**文件：**
- 修改：`source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py`
- 修改：`source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py`
- 修改：`source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] **Step 1：扩展静态测试**

在 `test_lab_pick_env_implements_dones_and_reset()` 后增加：

```python
def test_lab_pick_env_exposes_bc_observation_and_action():
    source = read(TASK_ROOT / "lab_pick_env.py")
    assert "def get_cafe_observation(self) -> dict[str, torch.Tensor]:" in source
    assert "def get_cafe_action(self) -> torch.Tensor:" in source
    assert "def _quat_to_rot6d(" in source
    assert '"robot0_pos"' in source
    assert '"robot0_force"' in source
    assert '"robot0_image"' in source
    assert "xyz(3) + rot6d(6) + gripper_width(1)" in source
```

- [ ] **Step 2：实现 quaternion 到 rot6d**

ForceCapture-CAFE 的 `convert_hdf5.py` 会把 quaternion 转成 6D rotation，再拼成 10 维 action/pose。IsaacLab 采集脚本也必须使用同样表达。

在 `LabPickEnv` 中新增：

```python
def _quat_to_rot6d(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
    # IsaacLab 使用 wxyz，scipy 版本转换脚本使用 xyzw；这里直接用 IsaacLab math 得到 rotation matrix。
    rot_mat = math_utils.matrix_from_quat(quat_wxyz)
    return rot_mat[:, :, :2].reshape(quat_wxyz.shape[0], 6)
```

- [ ] **Step 3：实现 CAFE observation**

在 `LabPickEnv` 中新增：

```python
def get_cafe_observation(self) -> dict[str, torch.Tensor]:
    tool_pos_b, tool_quat_b = self._compute_frame_pose()
    tool_rot6d_b = self._quat_to_rot6d(tool_quat_b)
    left_touch, right_touch = self.tactile_contact_depths()
    mean_touch = 0.5 * (left_touch + right_touch)
    contact_tag = ((left_touch > self.cfg.tactile_threshold_mm) | (right_touch > self.cfg.tactile_threshold_mm)).float()

    return {
        # CAFE: xyz(3) + rot6d(6) + gripper_width(1)
        "robot0_pos": torch.cat((tool_pos_b, tool_rot6d_b, self.gripper_width[:, :1]), dim=-1).detach().clone(),
        # CAFE force shape [4], last dim is contact/tag C_t.
        "robot0_force": torch.stack((left_touch, right_touch, mean_touch, contact_tag), dim=-1).detach().clone(),
    }
```

- [ ] **Step 4：实现 CAFE expert action**

CAFE action 也是 10 维：`xyz(3) + rot6d(6) + gripper_width(1)`。这里使用状态机产生的下一步目标 pose 作为 expert action。

```python
def get_cafe_action(self) -> torch.Tensor:
    target_rot6d_b = self._quat_to_rot6d(self.last_target_quat_b)
    return torch.cat(
        (
            self.last_target_pos_b,
            target_rot6d_b,
            self.gripper_width[:, :1],
        ),
        dim=-1,
    ).detach().clone()
```

此时 action 维度为 10：`target_pos_b(3) + target_rot6d_b(6) + gripper_width(1)`。

- [ ] **Step 5：实现低频图像 observation**

CAFE loader 读取 HDF5 中的图像时期待 HWC `uint8`，并在 dataloader 内部转成 CHW float。因此写入 HDF5 时保存 `(224, 224, 3)`。

在 `LabPickEnv` 中新增：

```python
def get_cafe_image(self) -> torch.Tensor:
    rgb = self.third_person_camera.data.output["rgb"][:, :, :, :3]
    rgb = rgb.permute(0, 3, 1, 2).float()
    rgb = torch.nn.functional.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    rgb = rgb.permute(0, 2, 3, 1).clamp(0, 255).byte()
    return rgb.detach().clone()
```

- [ ] **Step 6：调整 observation/action space**

在 `LabPickEnvCfg` 中明确：

```python
action_space = 10
observation_space = 14
state_space = 0
```

`_get_observations()` 继续返回 DirectRLEnv 需要的 `policy`，但内容改成 CAFE 高频低维 observation：

```python
def _get_observations(self) -> dict:
    obs = self.get_cafe_observation()
    policy = torch.cat((obs["robot0_pos"], obs["robot0_force"]), dim=-1)
    return {"policy": policy}
```

- [ ] **Step 7：运行静态测试**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：CAFE observation/action 相关断言通过。

- [ ] **Step 8：提交 CAFE 接口**

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env.py source/tacex_tasks/tacex_tasks/lab_pick/lab_pick_env_cfg.py source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "feat: expose lab pick CAFE observations and actions"
```

---

### Task 6：实现 ForceCapture-CAFE 兼容 HDF5 数据采集

**文件：**
- 新建：`source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py`
- 新建：`scripts/demos/lab_pick/collect_bc_dataset.py`
- 修改：`source/tacex_tasks/test/test_lab_pick_static.py`

- [ ] **Step 1：添加静态测试**

在 `source/tacex_tasks/test/test_lab_pick_static.py` 中新增：

```python
def test_lab_pick_bc_dataset_writer_and_collection_script_exist():
    writer_source = read(TASK_ROOT / "bc_dataset.py")
    script_source = read(SCRIPT_ROOT / "collect_bc_dataset.py")
    assert "class CafeHdf5Writer:" in writer_source
    assert "def append_high_step(" in writer_source
    assert "def append_low_step(" in writer_source
    assert "def flush_episode(" in writer_source
    assert "h5py.File" in writer_source
    assert 'demo_group.create_group("actions")' in writer_source
    assert 'actions_group.create_dataset("high"' in writer_source
    assert 'actions_group.create_dataset("low"' in writer_source
    assert 'obs_group.create_dataset("robot0_pos"' in writer_source
    assert 'obs_group.create_dataset("robot0_force"' in writer_source
    assert 'obs_group.create_dataset("robot0_image"' in writer_source
    assert 'f.attrs["freq_ratio"] = self.freq_ratio' in writer_source
    assert "env.get_cafe_observation()" in script_source
    assert "env.get_cafe_action()" in script_source
    assert "env.get_cafe_image()" in script_source
    assert "--num_demos" in script_source
    assert "--dataset_file" in script_source
    assert "--success_only" in script_source
```

- [ ] **Step 2：实现 HDF5 writer**

创建 `source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py`。写出的 HDF5 必须匹配 ForceCapture-CAFE：

```python
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


class CafeHdf5Writer:
    def __init__(self, dataset_file: str | Path, freq_ratio: int = 3, include_marker: bool = False):
        self.dataset_file = Path(dataset_file)
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)
        self.freq_ratio = freq_ratio
        self.include_marker = include_marker
        self.episode_index = 0
        self.high_pos: list[np.ndarray] = []
        self.high_force: list[np.ndarray] = []
        self.high_action: list[np.ndarray] = []
        self.low_image: list[np.ndarray] = []
        self.low_action: list[np.ndarray] = []
        self.high_marker2d: list[np.ndarray] = []

    def append_high_step(self, obs: dict[str, torch.Tensor], action: torch.Tensor):
        self.high_pos.append(obs["robot0_pos"][0].detach().cpu().numpy().astype(np.float32))
        self.high_force.append(obs["robot0_force"][0].detach().cpu().numpy().astype(np.float32))
        self.high_action.append(action[0].detach().cpu().numpy().astype(np.float32))
        if self.include_marker:
            self.high_marker2d.append(np.zeros((728,), dtype=np.float32))

    def append_low_step(self, image: torch.Tensor, action: torch.Tensor):
        self.low_image.append(image[0].detach().cpu().numpy().astype(np.uint8))
        self.low_action.append(action[0].detach().cpu().numpy().astype(np.float32))

    def flush_episode(
        self,
        *,
        success: bool,
        labware_reset_pos_w: torch.Tensor,
        labware_reset_quat_w: torch.Tensor,
        success_only: bool,
    ):
        if success_only and not success:
            self.clear_episode()
            return False
        if not self.high_action or not self.low_action:
            return False

        n_low = len(self.low_action)
        n_high = min(len(self.high_action), n_low * self.freq_ratio)
        n_low = n_high // self.freq_ratio
        n_high = n_low * self.freq_ratio
        if n_high == 0 or n_low == 0:
            self.clear_episode()
            return False

        high_pos = self.high_pos[:n_high]
        high_force = self.high_force[:n_high]
        high_action = self.high_action[:n_high]
        low_image = self.low_image[:n_low]
        low_action = self.low_action[:n_low]
        high_marker2d = self.high_marker2d[:n_high]

        with h5py.File(self.dataset_file, "a") as h5:
            h5.attrs["num_demos"] = max(int(h5.attrs.get("num_demos", 0)), self.episode_index + 1)
            h5.attrs["include_images"] = True
            h5.attrs["freq_ratio"] = self.freq_ratio
            h5.attrs["high_freq_obs_keys"] = "robot0_pos,robot0_force,robot0_marker2d" if self.include_marker else "robot0_pos,robot0_force"
            h5.attrs["low_freq_obs_keys"] = "robot0_image"
            h5.attrs["high_freq_action_key"] = "high"
            h5.attrs["low_freq_action_key"] = "low"

            data_group = h5.require_group("data")
            demo = data_group.create_group(f"demo_{self.episode_index}")
            actions_group = demo.create_group("actions")
            actions_group.create_dataset("high", data=np.stack(high_action, axis=0), dtype=np.float32)
            actions_group.create_dataset("low", data=np.stack(low_action, axis=0), dtype=np.float32)

            obs_group = demo.create_group("obs")
            obs_group.create_dataset("robot0_pos", data=np.stack(high_pos, axis=0), dtype=np.float32)
            obs_group.create_dataset("robot0_force", data=np.stack(high_force, axis=0), dtype=np.float32)
            obs_group.create_dataset("robot0_image", data=np.stack(low_image, axis=0), dtype=np.uint8)
            if self.include_marker:
                obs_group.create_dataset("robot0_marker2d", data=np.stack(high_marker2d, axis=0), dtype=np.float32)

            demo.attrs["length_high"] = n_high
            demo.attrs["length_low"] = n_low
            demo.attrs["freq_ratio"] = self.freq_ratio
            demo.attrs["success"] = bool(success)
            demo.attrs["labware_reset_pos_w"] = labware_reset_pos_w[0].detach().cpu().numpy()
            demo.attrs["labware_reset_quat_w"] = labware_reset_quat_w[0].detach().cpu().numpy()

        self.episode_index += 1
        self.clear_episode()
        return True

    def clear_episode(self):
        self.high_pos.clear()
        self.high_force.clear()
        self.high_action.clear()
        self.low_image.clear()
        self.low_action.clear()
        self.high_marker2d.clear()
```

- [ ] **Step 3：实现自动采集脚本**

创建 `scripts/demos/lab_pick/collect_bc_dataset.py`。脚本结构沿用 `pick_labware.py` 的 `AppLauncher`，但循环中记录 BC 数据：

```python
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect BC demonstrations for LabPick.")
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
                writer.flush_episode(
                    success=False,
                    labware_reset_pos_w=env.labware_reset_pos_w,
                    labware_reset_quat_w=env.labware_reset_quat_w,
                    success_only=args_cli.success_only,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 4：运行静态测试**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：HDF5 writer 和采集脚本断言通过。

- [ ] **Step 5：运行一条 headless 数据采集 smoke test**

```bash
cd /home/tjx/TacEx
env -u DISPLAY PYTHONUNBUFFERED=1 /home/tjx/miniforge3/envs/env_isaaclab/bin/python scripts/demos/lab_pick/collect_bc_dataset.py --labware slide --num_envs 1 --num_demos 2 --dataset_file /home/tjx/TacEx/datasets/lab_pick_slide_bc_smoke.hdf5 --success_only --headless
```

预期：

```text
[INFO] recorded_demo=1/2 success=True
[INFO] recorded_demo=2/2 success=True
```

并生成：

```text
/home/tjx/TacEx/datasets/lab_pick_slide_bc_smoke.hdf5
```

- [ ] **Step 6：检查 CAFE HDF5 schema**

运行：

```bash
cd /home/tjx/TacEx
/home/tjx/miniforge3/envs/env_isaaclab/bin/python -c "import h5py; f=h5py.File('/home/tjx/TacEx/datasets/lab_pick_slide_bc_smoke.hdf5','r'); d=f['data']['demo_0']; assert d['actions']['high'].shape[1]==10; assert d['actions']['low'].shape[1]==10; assert d['obs']['robot0_pos'].shape[1]==10; assert d['obs']['robot0_force'].shape[1]==4; assert d['obs']['robot0_image'].shape[1:]==(224,224,3); assert d.attrs['length_high']==d.attrs['length_low']*f.attrs['freq_ratio']; print('CAFE schema OK')"
```

预期：

```text
CAFE schema OK
```

- [ ] **Step 7：提交数据采集功能**

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/tacex_tasks/lab_pick/bc_dataset.py scripts/demos/lab_pick/collect_bc_dataset.py source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "feat: collect lab pick BC datasets"
```

---

### Task 7：改造 IsaacLab 启动脚本

**文件：**
- 修改：`scripts/demos/lab_pick/pick_labware.py`
- 修改：`scripts/demos/lab_pick/pick_labware_keyboard.py`

- [ ] **Step 1：改造自动 pick 脚本**

保留 `AppLauncher` 位置，删除本地 `LabPickEnvCfg` 和 `LabPickEnv` 类定义，改为导入共享环境：

```python
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg
```

保留 CLI 参数：

```text
--num_envs
--labware
--duration
--save_camera_images
--save_tactile_images
--record_video
--video_camera
--video_every_n_steps
--video_fps
--print_state_interval
```

`main()` 使用：

```python
def main():
    env_cfg = LabPickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.labware_name = args_cli.labware
    env_cfg.episode_length_s = args_cli.duration
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = LabPickEnv(env_cfg, render_mode="rgb_array")
    print("[INFO] Setup complete.")
    run_simulator(env)
```

- [ ] **Step 2：改造键盘遥操作脚本**

同样导入共享环境：

```python
from tacex_tasks.lab_pick.lab_pick_env import LabPickEnv
from tacex_tasks.lab_pick.lab_pick_env_cfg import LabPickEnvCfg
```

保留键盘相关参数：

```text
--pos_sensitivity
--rot_sensitivity
--open_width
--close_width
```

环境方法不要直接读取脚本全局 `args_cli`。将方法签名改成：

```python
def reset_keyboard_target(self, env_ids: torch.Tensor | None = None, open_width: float = 0.04):
    ...
    self.gripper_width[env_ids] = open_width
```

以及：

```python
def command_keyboard(self, delta_pose, close_gripper: bool, open_width: float = 0.04, close_width: float = 0.0):
    ...
    self.gripper_width[:] = close_width if close_gripper else open_width
```

脚本调用：

```python
env.reset_keyboard_target(open_width=args_cli.open_width)
env.command_keyboard(
    delta_pose,
    close_gripper,
    open_width=args_cli.open_width,
    close_width=args_cli.close_width,
)
```

- [ ] **Step 3：运行静态测试**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest source/tacex_tasks/test/test_lab_pick_static.py -q
```

预期：全部通过。

- [ ] **Step 4：提交脚本改造**

```bash
cd /home/tjx/TacEx
git add scripts/demos/lab_pick/pick_labware.py scripts/demos/lab_pick/pick_labware_keyboard.py source/tacex_tasks/test/test_lab_pick_static.py source/tacex_tasks/tacex_tasks/lab_pick
git commit -m "refactor: use shared lab pick environment in launchers"
```

---

### Task 8：运行 IsaacLab smoke test

**文件：**
- 正常情况下不改文件；如果 smoke test 暴露真实 bug，再修对应实现。

- [ ] **Step 1：确认环境已注册**

```bash
cd /home/tjx/TacEx
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python scripts/reinforcement_learning/list_envs.py | rg "TacEx-LabPick"
```

预期输出包含：

```text
TacEx-LabPick-Slide-Direct-v0
TacEx-LabPick-Coverslip-Direct-v0
TacEx-LabPick-Cup-Direct-v0
```

- [ ] **Step 2：运行 slide 自动 pick headless 测试**

```bash
cd /home/tjx/TacEx
env -u DISPLAY PYTHONUNBUFFERED=1 /home/tjx/miniforge3/envs/env_isaaclab/bin/python scripts/demos/lab_pick/pick_labware.py --labware slide --num_envs 1 --duration 4 --record_video --video_camera third --headless --print_state_interval 120
```

预期输出包含：

```text
[INFO] Setup complete.
[INFO] Starting labware pick demo: labware=slide, envs=1
[RESULT] object_lift_delta_z=... m, lifted=...
```

预期生成文件：

```text
/home/tjx/TacEx/logs/lab_pick/slide/third_camera.mp4
```

- [ ] **Step 3：测试 coverslip 和 cup**

```bash
cd /home/tjx/TacEx
env -u DISPLAY PYTHONUNBUFFERED=1 /home/tjx/miniforge3/envs/env_isaaclab/bin/python scripts/demos/lab_pick/pick_labware.py --labware coverslip --num_envs 1 --duration 2 --headless --print_state_interval 120
env -u DISPLAY PYTHONUNBUFFERED=1 /home/tjx/miniforge3/envs/env_isaaclab/bin/python scripts/demos/lab_pick/pick_labware.py --labware cup --num_envs 1 --duration 2 --headless --print_state_interval 120
```

预期：两个命令都无 Python exception，并打印 `[RESULT] object_lift_delta_z=`。

- [ ] **Step 4：运行 TacEx 环境测试过滤 LabPick**

```bash
cd /home/tjx/TacEx/source/tacex_tasks/test
PYTHONPATH=/home/tjx/TacEx/source/tacex:/home/tjx/TacEx/source/tacex_assets:/home/tjx/TacEx/source/tacex_tasks /home/tjx/miniforge3/envs/env_isaaclab/bin/python -m pytest test_environments.py -q -k "LabPick"
```

预期：LabPick 环境可以创建、reset、step，不出现无效 observation、reward 或 reset crash。

- [ ] **Step 5：如有 smoke test 修复，提交**

如果 smoke test 需要修实现：

```bash
cd /home/tjx/TacEx
git add source/tacex_tasks/tacex_tasks/lab_pick scripts/demos/lab_pick source/tacex_tasks/test/test_lab_pick_static.py
git commit -m "fix: stabilize lab pick smoke tests"
```

如果没有修复，不创建空提交。

---

## 自检清单

- 需求覆盖：
  - IsaacLab 环境配置：Task 2
  - terminate 函数：Task 3 的 `_get_dones()`
  - reset 函数：Task 3 的 `_reset_idx()`
  - 载玻片位置随机化：Task 3 的 `_reset_idx()`
  - BC observation/action：Task 5
  - BC HDF5 数据采集：Task 6
  - IsaacLab 启动脚本：Task 7
  - 注册和验收：Task 4、Task 8
- 类型一致性：
  - `LabPickEnvCfg.labware_name` 是选择 `slide` / `coverslip` / `cup` 的唯一入口
  - `LabPickEnv.__init__()` 不再接受额外 `labware` 参数
  - `step_count` 是 per-env tensor，日志和文件名使用 `step_count[0].item()`
- 执行原则：
  - 先测试，再实现
  - 小步提交
  - 不改动无关 dirty worktree 内容
