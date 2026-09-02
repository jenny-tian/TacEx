# 恒力表面扫描场景

任务 ID：`TacEx-LabSurface-ForceScan-v0`

场景中深灰色底板上铺有浅灰色平板区；蓝色窄条是低于平板顶面的凹槽，红色方块是随机位置的凸起。两种缺陷均使用不透明材质，便于 Isaac Sim 视图和录制视频中区分。

启动 Isaac Lab 后采集 100 条示范：

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/IsaacLab-2.3.2/isaaclab.sh -p scripts/demos/lab_surface/collect_force_scan_dataset.py \
  --num_episodes 100 --record_dir datasets/lab_surface_force_scan --headless
```

加入 `--save_scene_preview` 会在数据目录写出 `scene_preview.png`，用于检查场景颜色和相机视角。

录制第 1 条采集轨迹的视频：

```bash
PYTHONPATH=source/tacex:source/tacex_assets:source/tacex_tasks \
  /home/tjx/IsaacLab-2.3.2/isaaclab.sh -p scripts/demos/lab_surface/collect_force_scan_dataset.py \
  --num_episodes 1 --episode_length_s 40 --action_position_scale_m 0.001 \
  --record_dir datasets/lab_surface_force_scan_preview \
  --record_video --video_episode 0 --video_fps 30 --video_stride 4 --headless
```

视频文件名为 `episode_00000.mp4`，画面来自 BC 数据中的固定左前上方场景相机，并叠加当前接触力与目标力；该角度同时保留末端探头和整个扫描面。

每个 `episode_XXXXX.npz` 包含末端位姿、扫描目标、接触力、触觉深度、动作、奖励、表面标签以及凸起中心；默认同时保存固定场景相机的 `scene_rgb`/`scene_depth` 和 GelSight RGB/height map。关闭外部视觉或触觉图像可分别加 `--no-save_visual`、`--no-save_tactile`。

采集器沿 `x=0.41 m` 到 `x=0.63 m` 扫描，覆盖板面内侧约 220 mm，目标法向力为 `3 N`。采集控制器先固定高度稳定，再用低速下降探测接触；接触后先回到左边界并稳定，再对力信号低通滤波，用累计法向目标进行导纳调节，同时保持 `y` 和探头姿态。数据中的 `contact_force_n` 是原始峰值诊断，`force_filtered_n` 是控制和训练使用的滤波力。力超过 `4 N` 时环境终止该回合，避免把过力样本继续写入后续轨迹。
