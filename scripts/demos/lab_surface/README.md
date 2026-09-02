# 恒力表面扫描场景

任务 ID：`TacEx-LabSurface-ForceScan-v0`

该任务控制 Panda 末端探头从左向右扫描刚性板面，并在平面、凸起和凹槽上保持 `3 N` 法向接触力。场景用于采集视觉、触觉和力控示范数据，后续可用于 BC 或 DSRL 训练。

## 场景与控制参数

| 参数 | 当前值 |
| --- | ---: |
| 板面尺寸 | `240 x 160 mm` |
| 扫描范围 | `x=0.41 m` 到 `x=0.63 m` |
| 有效扫描长度 | `220 mm` |
| 扫描速度 | `10 mm/s` |
| 目标法向力 | `3.0 N` |
| 目标容差 | `±0.3 N` |
| 环境过力阈值 | `4.0 N` |
| 凸起高度 | `2 mm` |
| 凹槽深度 | `1 mm` |
| 表面前视距离 | `1.5 mm` |
| 仿真频率 | `120 Hz` |

深灰色结构为底板，浅灰色区域为正常表面，蓝色 V 形连续斜坡为凹槽，红色球冠为随机位置凸起。缺陷边缘采用连续曲面或斜坡，避免刚性竖直台阶引起非物理冲击；颜色使用不透明材质，便于在 Isaac Sim 视图和视频中识别。

采集器使用以下流程：

1. 移动到扫描起点上方并稳定。
2. 低速下降，检测首次接触。
3. 沿板面 X 方向扫描，同时保持 Y 方向和探头姿态。
4. 经过接触稳定阶段后才开始记录有效扫描。
5. 使用前方表面高度进行双向前馈：遇到凸起提前抬升，遇到凹槽提前下降。
6. 根据低通滤波后的力误差调整 Z 目标，并限制每步高度变化以减小刚性接触冲击。

固定板面采集时 `randomize_board_pose=False`。环境支持板面平移和倾斜随机化，但当前基线采集器尚未将其作为默认配置。

## 环境准备

以下命令假设仓库位于 `/home/tjx/TacEx`，Isaac Lab 位于 `/home/tjx/IsaacLab-2.3.2`，Python 环境为 `/home/tjx/miniforge3/envs/env_isaaclab`。

在仓库根目录执行：

```bash
CONDA_PREFIX=/home/tjx/miniforge3/envs/env_isaaclab \
PATH=/home/tjx/miniforge3/envs/env_isaaclab/bin:$PATH \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/IsaacLab-2.3.2/isaaclab.sh -p \
  scripts/demos/lab_surface/collect_force_scan_dataset.py --help
```

## 采集数据

采集 100 条完整示范：

```bash
CONDA_PREFIX=/home/tjx/miniforge3/envs/env_isaaclab \
PATH=/home/tjx/miniforge3/envs/env_isaaclab/bin:$PATH \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/IsaacLab-2.3.2/isaaclab.sh -p \
  scripts/demos/lab_surface/collect_force_scan_dataset.py \
  --num_episodes 100 \
  --episode_length_s 40 \
  --record_dir datasets/lab_surface_force_scan \
  --headless
```

默认保存外部相机和 GelSight 数据。仅验证控制轨迹、减少磁盘占用时，可增加 `--no-save_visual --no-save_tactile`。加入 `--save_scene_preview` 会在输出目录生成 `scene_preview.png`，用于检查场景颜色和相机视角。

## 录制视频

录制第 1 条轨迹：

```bash
CONDA_PREFIX=/home/tjx/miniforge3/envs/env_isaaclab \
PATH=/home/tjx/miniforge3/envs/env_isaaclab/bin:$PATH \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/IsaacLab-2.3.2/isaaclab.sh -p \
  scripts/demos/lab_surface/collect_force_scan_dataset.py \
  --num_episodes 1 \
  --episode_length_s 40 \
  --record_dir datasets/lab_surface_force_scan_preview \
  --record_video --video_episode 0 --video_fps 30 --video_stride 4 \
  --headless
```

输出视频为 `episode_00000.mp4`。画面来自固定左前上方场景相机，并叠加当前接触力和目标力。

## 数据格式

每个回合保存为 `episode_XXXXX.npz`，主要字段如下：

| 字段 | 内容 |
| --- | --- |
| `tool_pos` | 探头在机器人基座坐标系中的位置 |
| `tool_pos_local` | 探头在板面局部坐标系中的位置 |
| `tool_quat` | 探头姿态四元数 |
| `scan_target_xy` | 当前扫描目标位置 |
| `contact_force_n` | 用于环境判定的接触力 |
| `force_filtered_n` | 低通滤波后的控制力 |
| `tactile_depth` | GelSight 最大压入深度 |
| `action` | 采集器动作 `[dx, dy, dz, dyaw]` |
| `reward` | 当前环境奖励 |
| `surface_label` | `0=平面`、`1=凸起`、`2=凹槽` |
| `surface_kind` | 缺陷类别，非缺陷区域为 `-1` |
| `phase` | `approach`、`descend`、`contact_settle` 或 `scan` |
| `defect_centers_xy` | 随机凸起中心位置 |
| `groove_x` | 四条固定凹槽的 X 坐标 |
| `board_translation` | 板面平移随机化参数 |
| `board_quat` | 板面旋转随机化参数 |
| `scene_rgb`, `scene_depth` | 外部相机 RGB 与深度 |
| `left_tactile_rgb`, `right_tactile_rgb` | 左右 GelSight RGB |
| `left_height_map`, `right_height_map` | 左右 GelSight 高度图 |

## 过力判定

环境训练和评估时，接触力超过 `4 N` 会终止回合。数据采集脚本会设置 `terminate_on_overforce=False`，以便记录完整扫描以及瞬态过力样本。因此采集输出中的 `terminal overforce` 通常为零，不能据此判断轨迹是否曾超过 `4 N`。

接触力由有效范围内的 PhysX 接触读数和连续几何弹簧估计共同得到。超过目标控制带的单帧刚性碰撞冲量会回退到连续估计，降低边缘数值冲量对控制的影响。

## 完整验证

无图像、无触觉保存的 40 秒验证：

```bash
CONDA_PREFIX=/home/tjx/miniforge3/envs/env_isaaclab \
PATH=/home/tjx/miniforge3/envs/env_isaaclab/bin:$PATH \
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
/home/tjx/IsaacLab-2.3.2/isaaclab.sh -p \
  scripts/demos/lab_surface/collect_force_scan_dataset.py \
  --num_episodes 1 --episode_length_s 40 \
  --record_dir artifacts/lab_surface_validation \
  --no-save_visual --no-save_tactile --headless
```

当前固定板面单回合验证结果：

| 指标 | 结果 |
| --- | ---: |
| 扫描段平均力 | `2.997 N` |
| 力跟踪 RMSE | `0.107 N` |
| 原始力 5–95 百分位 | `2.855–3.190 N` |
| `3±0.3 N` 占比 | `96.95%` |
| `3±0.5 N` 占比 | `99.45%` |
| 失触率 | `0%` |
| 超过 `4 N` 占比 | `0%` |
| 原始力峰值 | `3.316 N` |
| 滤波力峰值 | `3.310 N` |
| 横向 RMSE | `0.003 mm` |

该版本在 30 秒单回合中完整覆盖约 `220 mm` 扫描长度和四条凹槽。平面、凸起和凹槽的扫描力均保持在目标附近，未出现 `>4 N` 峰值或失触。后续训练或控制器比较仍应同时报告 RMSE、容差带占比、失触率和峰值，不能只报告平均力。
