#!/usr/bin/env python
"""Generate a compact Markdown/JSON comparison report for BC100 vs DINOv3-BC200."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def two_sided_fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a,b],[c,d]], without scipy."""
    row1 = a + b
    row2 = c + d
    successes = a + c
    total = row1 + row2

    def probability(x: int) -> float:
        if not (max(0, successes - row2) <= x <= min(row1, successes)):
            return 0.0
        return math.comb(successes, x) * math.comb(total - successes, row1 - x) / math.comb(total, row1)

    observed = probability(a)
    return min(
        1.0,
        sum(
            probability(x)
            for x in range(max(0, successes - row2), min(row1, successes) + 1)
            if probability(x) <= observed + 1.0e-15
        ),
    )


def two_sided_exact_mcnemar(new_only: int, baseline_only: int) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes."""
    discordant = new_only + baseline_only
    if discordant == 0:
        return 1.0
    smaller = min(new_only, baseline_only)
    lower_tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--new-eval", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--offline-diagnostic", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    baseline = load_json(args.baseline_eval)
    new = load_json(args.new_eval)
    training = load_json(args.training_summary)
    config = load_json(args.training_config)
    diagnostic = load_json(args.offline_diagnostic) if args.offline_diagnostic else None
    diagnostic_summary = (
        None if diagnostic is None else {key: value for key, value in diagnostic.items() if key != "records"}
    )
    checkpoint_path = Path(training["best_checkpoint"])
    b_success, b_trials = int(baseline["successes"]), int(baseline["num_trials"])
    n_success, n_trials = int(new["successes"]), int(new["num_trials"])
    b_rate, n_rate = b_success / b_trials, n_success / n_trials
    b_ci = wilson_interval(b_success, b_trials)
    n_ci = wilson_interval(n_success, n_trials)
    p_value = two_sided_fisher_exact(n_success, n_trials - n_success, b_success, b_trials - b_success)
    baseline_by_seed = {int(row["seed"]): bool(row["success"]) for row in baseline.get("results", [])}
    new_by_seed = {int(row["seed"]): bool(row["success"]) for row in new.get("results", [])}
    paired_seeds = sorted(set(baseline_by_seed) & set(new_by_seed))
    both_success = sum(baseline_by_seed[seed] and new_by_seed[seed] for seed in paired_seeds)
    new_only = sum((not baseline_by_seed[seed]) and new_by_seed[seed] for seed in paired_seeds)
    baseline_only = sum(baseline_by_seed[seed] and (not new_by_seed[seed]) for seed in paired_seeds)
    both_failure = len(paired_seeds) - both_success - new_only - baseline_only
    mcnemar_p = two_sided_exact_mcnemar(new_only, baseline_only) if paired_seeds else None
    increased = n_rate > b_rate
    delta = n_rate - b_rate
    xy_range = new.get("labware_random_xy_m", [None, None])
    xy_text = (
        f"x=±{float(xy_range[0]):.2f} m, y=±{float(xy_range[1]):.2f} m"
        if len(xy_range) == 2 and all(value is not None for value in xy_range)
        else str(xy_range)
    )

    result = {
        "baseline": {"successes": b_success, "trials": b_trials, "success_rate": b_rate, "wilson_95_ci": b_ci},
        "dinov3_bc200": {"successes": n_success, "trials": n_trials, "success_rate": n_rate, "wilson_95_ci": n_ci},
        "absolute_change": delta,
        "relative_change": None if b_rate == 0 else delta / b_rate,
        "success_rate_increased": increased,
        "fisher_exact_two_sided_p": p_value,
        "paired_comparison": {
            "num_paired_seeds": len(paired_seeds),
            "both_success": both_success,
            "dinov3_only_success": new_only,
            "baseline_only_success": baseline_only,
            "both_failure": both_failure,
            "exact_mcnemar_two_sided_p": mcnemar_p,
        },
        "yaw_degrees": new.get("labware_random_yaw_degrees"),
        "xy_randomization_m": new.get("labware_random_xy_m"),
        "training": training,
        "offline_diagnostic": diagnostic_summary,
    }
    conclusion = "上升" if increased else "未上升"
    significance = "达到" if p_value < 0.05 else "未达到"
    paired_sentence = ""
    if mcnemar_p is not None:
        paired_significance = "达到" if mcnemar_p < 0.05 else "未达到"
        paired_sentence = (
            f"同 seed 配对结果为新模型独有成功 {new_only} 次、旧模型独有成功 {baseline_only} 次，"
            f"exact McNemar p={mcnemar_p:.4g}（{paired_significance} 0.05 显著性水平）。"
        )
    diagnostic_lines: list[str] = []
    if diagnostic is not None:
        frame_index = diagnostic.get("requested_frame_index", 0)
        diagnostic_lines = [
            "## 离线视觉定位诊断",
            "",
            f"在记录第 {frame_index} 帧（与评估的相机预热步数对应）上，200 条轨迹的 action-XY "
            f"欧氏误差均值为 {diagnostic['all']['mean_m'] * 1000:.2f} mm、P90 为 "
            f"{diagnostic['all']['p90_m'] * 1000:.2f} mm；20 条 validation 轨迹均值为 "
            f"{diagnostic['validation']['mean_m'] * 1000:.2f} mm。该指标只验证开环定位，不能替代闭环成功率。",
            "",
        ]
    lines = [
        "# TacEx DINOv3 BC200 实验报告",
        "",
        f"结论：在同一 yaw=0 闭环评估协议下，新 DINOv3-BC200 成功率{conclusion}。"
        f"成功率从 {b_success}/{b_trials}（{b_rate:.1%}）变为 {n_success}/{n_trials}（{n_rate:.1%}），"
        f"绝对变化 {delta:+.1%}；Fisher 精确检验 p={p_value:.4g}，差异{significance} 0.05 显著性水平。",
        paired_sentence,
        "",
        "## 结果",
        "",
        "| 模型 | 训练轨迹 | 成功/试验 | 成功率 | 95% Wilson CI |",
        "|---|---:|---:|---:|---:|",
        f"| 旧 Flow Matching BC100 | 100 | {b_success}/{b_trials} | {b_rate:.1%} | {b_ci[0]:.1%}–{b_ci[1]:.1%} |",
        f"| 新 DINOv3-ACT BC200 | {config['num_episodes_total']} | {n_success}/{n_trials} | {n_rate:.1%} | {n_ci[0]:.1%}–{n_ci[1]:.1%} |",
        "",
        "同 seed 配对列联表：",
        "",
        "| 两者成功 | 仅新模型成功 | 仅旧模型成功 | 两者失败 |",
        "|---:|---:|---:|---:|",
        f"| {both_success} | {new_only} | {baseline_only} | {both_failure} |",
        "",
        f"新模型接触率为 {int(new.get('touched', 0))}/{n_trials}（{float(new.get('touch_rate', 0.0)):.1%}），"
        f"破损率为 {int(new.get('broken', 0))}/{n_trials}（{float(new.get('broken_rate', 0.0)):.1%}）；"
        f"失败原因计数为 `{json.dumps(new.get('failure_counts', {}), ensure_ascii=False)}`。",
        "",
        "## 实验约束",
        "",
        f"- 物体 yaw 随机范围：{new.get('labware_random_yaw_degrees', 0.0)}°（不做 yaw 泛化）。",
        f"- XY 随机范围：{xy_text}。",
        f"- 训练集：{config['num_episodes_total']} 条成功轨迹；训练/验证划分 "
        f"{config['num_episodes_train']}/{config['num_episodes_val']}。",
        f"- 数据集路径：`{config['data_root']}`；审计到的成功轨迹 "
        f"{config['num_successful_episodes']}/{config['num_episodes_total']}，yaw 范围 "
        f"{config['yaw_min_degrees']:.1f}°–{config['yaw_max_degrees']:.1f}°。",
        f"- 最优 checkpoint：epoch {training['best_epoch']}，验证损失 {training['best_validation_loss']:.6f}。",
        f"- 评估：120 Hz 物理仿真，每个策略动作重复 {new.get('action_repeat')} 次，"
        f"每次重规划执行 {new.get('chunk_execute_steps')} 个动作；玻璃破损阈值 "
        f"{new.get('break_force_threshold_n')} N。",
        "",
        "## 新模型",
        "",
        "- 冻结的 `facebook/dinov3-vits16-pretrain-lvd1689m` 双相机视觉编码器；视觉定位头使用第三视角 7×7 空间 token，Transformer 条件分支使用双相机 4×4 token。",
        "- 无状态第三视角 XY 定位头直接覆盖 action chunk 的 XY，避免后半段 proprioception 捷径掩盖初始目标定位误差。",
        "- 确定性 ACT 风格 action-chunk decoder，消除 Flow Matching 采样噪声。",
        "- 显式轨迹 phase token、终点 padding mask，以及偏重位置/夹爪维度的 Smooth-L1 损失。",
        "- 数据读取固定为采集端实际使用的 `xyzw`，并在训练前强制检查所有 reset yaw 为 0。",
        f"- 仿真器 action-yaw 对齐覆盖：`{new.get('rl_align_cafe_action_yaw', False)}`（正式对比关闭，不使用 yaw oracle）。",
        f"- 可训练参数：{config['trainable_parameters']:,}；checkpoint SHA256：`{sha256(checkpoint_path)}`。",
        "",
        *diagnostic_lines,
        "## 解读边界",
        "",
        "本对比回答的是“新模型 + 200 条成功轨迹”的整体方案是否优于当前 BC100。模型结构、"
        "失败轨迹过滤和数据量同时发生变化，因此不能把增益单独归因于 DINOv3 或轨迹数。"
        "若要估计纯数据规模效应，应再训练同结构的 DINOv3-ACT BC100，并使用多个训练种子复验。",
        "",
        "## 产物",
        "",
        f"- 新 checkpoint：`{training['best_checkpoint']}`",
        f"- 训练日志：`{args.training_summary.expanduser().resolve().parent / 'logs.jsonl'}`",
        f"- BC100 评估：`{args.baseline_eval.expanduser().resolve()}`",
        f"- BC200 评估：`{args.new_eval.expanduser().resolve()}`",
    ]
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_output = (args.json_output or output_path.with_suffix(".json")).expanduser().resolve()
    json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output_path} and {json_output}")


if __name__ == "__main__":
    main()
