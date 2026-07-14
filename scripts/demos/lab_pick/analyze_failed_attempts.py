from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


FAILURE_TYPES = {
    "no_contact",
    "slip",
    "over_force",
    "bad_alignment",
    "object_drop",
    "workspace_error",
    "timeout_or_no_success",
    "unknown",
}
SUGGESTED_ACTIONS = {
    "increase_grip",
    "reduce_grip",
    "adjust_xy",
    "lower_approach_height",
    "slow_closing",
    "keep_current",
    "unknown",
}


parser = argparse.ArgumentParser(description="Analyze LabPick failed attempts with a VLM.")
parser.add_argument("--record_dir", type=str, required=True, help="Directory containing failed_attempts/attempt_xxxxxx.")
parser.add_argument("--model", type=str, default=os.environ.get("LAB_PICK_VLM_MODEL", "gpt-4.1-mini"))
parser.add_argument("--api_base", type=str, default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
parser.add_argument("--api_mode", choices=("responses", "chat_completions"), default="responses")
parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
parser.add_argument("--api_timeout", type=float, default=60.0)
parser.add_argument("--break_force_threshold_n", type=float, default=6.0)
parser.add_argument("--skip_existing", action="store_true")
parser.add_argument("--latest_only", action="store_true")
parser.add_argument("--max_attempts", type=int, default=0, help="0 means analyze all attempts.")
parser.add_argument("--frame", choices=("auto", "failure", "last"), default="auto")
parser.add_argument("--dry_run", action="store_true", help="Use deterministic local heuristics instead of calling the VLM API.")
args_cli = parser.parse_args()


def _attempt_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def _parse_info_file(info_path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    if not info_path.is_file():
        return info
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def _frame_prefix(attempt_dir: Path) -> str:
    if args_cli.frame == "failure":
        return "failure_frame"
    if args_cli.frame == "last":
        return "last_frame"
    if (attempt_dir / "failure_frame_ft.npy").is_file():
        return "failure_frame"
    return "last_frame"


def _load_ft(attempt_dir: Path, frame_prefix: str) -> np.ndarray:
    ft_path = attempt_dir / f"{frame_prefix}_ft.npy"
    if not ft_path.is_file():
        raise FileNotFoundError(f"Missing {ft_path}")
    return np.load(ft_path).astype(np.float32).reshape(6)


def _image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported image format for VLM input: {path}")


def _ensure_image_path(attempt_dir: Path, frame_prefix: str) -> Path:
    for suffix in ("png", "jpg", "jpeg", "webp"):
        name = f"{frame_prefix}_rgb.{suffix}"
        image_path = attempt_dir / name
        if image_path.is_file():
            return image_path

    rgb_npy = attempt_dir / f"{frame_prefix}_rgb.npy"
    if not rgb_npy.is_file():
        raise FileNotFoundError(f"Missing {frame_prefix} image file in {attempt_dir}")

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"{attempt_dir} has only {frame_prefix}_rgb.npy; install Pillow or save PNG during collection.") from exc

    rgb = np.load(rgb_npy).astype(np.uint8)
    image_path = attempt_dir / f"{frame_prefix}_rgb.png"
    Image.fromarray(rgb[:, :, :3]).save(image_path)
    return image_path


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{_image_mime(image_path)};base64,{encoded}"


def _safe_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _attempt_context(attempt_dir: Path, break_force_threshold_n: float) -> dict[str, Any]:
    frame_prefix = _frame_prefix(attempt_dir)
    info = _parse_info_file(attempt_dir / f"{frame_prefix}_info.txt")
    if not info and frame_prefix != "last_frame":
        info = _parse_info_file(attempt_dir / "last_frame_info.txt")
    ft = _load_ft(attempt_dir, frame_prefix)
    force_norm = float(np.linalg.norm(ft[:3]))
    torque_norm = float(np.linalg.norm(ft[3:]))
    frame_step_key = f"{frame_prefix}_step"
    return {
        "attempt_id": attempt_dir.name,
        "attempt_dir": str(attempt_dir),
        "analysis_frame": frame_prefix,
        "script_failure_reason": info.get("failure_reason", "unknown"),
        "frame_step": int(_safe_float(info.get(frame_step_key), _safe_float(info.get("last_step"), -1))),
        "first_failure_step": int(_safe_float(info.get("first_failure_step"), -1)),
        "timestamp": _safe_float(info.get("timestamp")),
        "ft": [float(v) for v in ft.tolist()],
        "force_norm_n": force_norm,
        "torque_norm_nm": torque_norm,
        "break_force_threshold_n": _safe_float(info.get("break_force_threshold_n"), break_force_threshold_n),
    }


def _analysis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "failure_type": {"type": "string", "enum": sorted(FAILURE_TYPES)},
            "visual_reason": {"type": "string"},
            "force_reason": {"type": "string"},
            "combined_reason": {"type": "string"},
            "observed_force_n": {"type": "number"},
            "break_threshold_n": {"type": "number"},
            "suggested_force_range_n": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "suggested_action": {"type": "string", "enum": sorted(SUGGESTED_ACTIONS)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "failure_type",
            "visual_reason",
            "force_reason",
            "combined_reason",
            "observed_force_n",
            "break_threshold_n",
            "suggested_force_range_n",
            "suggested_action",
            "confidence",
        ],
    }


def _build_prompt(context: dict[str, Any]) -> str:
    safe_upper = 0.75 * float(context["break_force_threshold_n"])
    soft_upper = 0.85 * float(context["break_force_threshold_n"])
    return (
        "你是机器人触觉抓取失败分析助手。请分析 Franka 夹爪在 IsaacLab 中抓取载玻片的失败原因。\n"
        "你会看到失败轨迹中用于诊断的一帧 RGB 图像，以及该帧 fingertip/contact sensor 的 6D 力/力矩。\n"
        "默认优先使用第一次触发失败判定的 failure_frame；如果旧数据没有 failure_frame，才使用 last_frame。\n"
        "FT 顺序为 [Fx, Fy, Fz, Tx, Ty, Tz]，力单位 N，力矩单位 N*m。\n\n"
        f"当前分析帧: {context['analysis_frame']}\n"
        f"脚本记录的失败原因: {context['script_failure_reason']}\n"
        f"frame_step: {context['frame_step']}\n"
        f"first_failure_step: {context['first_failure_step']}\n"
        f"timestamp: {context['timestamp']:.6f}\n"
        f"FT: {context['ft']}\n"
        f"force_norm_n: {context['force_norm_n']:.6f}\n"
        f"torque_norm_nm: {context['torque_norm_nm']:.6f}\n"
        f"break_force_threshold_n: {context['break_force_threshold_n']:.6f}\n\n"
        "力判断参考：\n"
        "- force_norm < 1.0N 通常表示接触不足或没有夹住。\n"
        f"- 1.0N 到 {safe_upper:.2f}N 通常较安全。\n"
        f"- {safe_upper:.2f}N 到 {soft_upper:.2f}N 接近风险区，需要谨慎。\n"
        f"- 超过 {context['break_force_threshold_n']:.2f}N 判定为破碎失败。\n\n"
        "请结合图像和力信息判断失败类型，并给出下一次采集建议。"
        "解释字段请用简洁中文；枚举字段必须使用 schema 中的英文枚举值。只输出 JSON。"
    )


def _extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    raise RuntimeError(f"Could not find text in VLM response keys={sorted(response.keys())}")


def _post_json(*, api_key: str, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VLM API request failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(raw)


def _call_responses_vlm(*, api_key: str, image_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    response = _post_json(
        api_key=api_key,
        url=f"{args_cli.api_base.rstrip('/')}/responses",
        timeout=args_cli.api_timeout,
        payload={
        "model": args_cli.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _build_prompt(context)},
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "labpick_failure_analysis",
                "schema": _analysis_schema(),
                "strict": True,
            }
        },
        },
    )
    return json.loads(_extract_response_text(response))


def _call_chat_completions_vlm(*, api_key: str, image_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    response = _post_json(
        api_key=api_key,
        url=f"{args_cli.api_base.rstrip('/')}/chat/completions",
        timeout=args_cli.api_timeout,
        payload={
            "model": args_cli.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(context)},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "labpick_failure_analysis",
                    "schema": _analysis_schema(),
                    "strict": True,
                },
            },
        },
    )
    return json.loads(response["choices"][0]["message"]["content"])


def _call_vlm(*, api_key: str, image_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    if args_cli.api_mode == "chat_completions":
        return _call_chat_completions_vlm(api_key=api_key, image_path=image_path, context=context)
    return _call_responses_vlm(api_key=api_key, image_path=image_path, context=context)


def _heuristic_analysis(context: dict[str, Any]) -> dict[str, Any]:
    reason = str(context["script_failure_reason"])
    force_norm = float(context["force_norm_n"])
    threshold = float(context["break_force_threshold_n"])
    safe_upper = min(4.5, 0.75 * threshold)

    failure_type = "unknown"
    suggested_action = "unknown"
    visual_reason = "dry_run 未调用 VLM，无法从图像判断。"
    force_reason = f"最后一帧 force_norm 为 {force_norm:.3f}N。"
    combined_reason = "根据脚本失败原因和 FT 做本地启发式判断。"

    if "break_force" in reason or force_norm > threshold:
        failure_type = "over_force"
        suggested_action = "reduce_grip"
        combined_reason = "接触力超过或接近破碎阈值，需要减小闭合量或放慢闭合。"
    elif "object_drop" in reason:
        failure_type = "object_drop"
        suggested_action = "increase_grip" if force_norm < 1.0 else "adjust_xy"
        combined_reason = "脚本检测到物体掉落，需要检查接触位置和夹持力。"
    elif "object_xy_distance" in reason:
        failure_type = "bad_alignment"
        suggested_action = "adjust_xy"
        combined_reason = "物体相对初始位置偏移过大，优先调整 approach XY。"
    elif "ee_workspace" in reason:
        failure_type = "workspace_error"
        suggested_action = "adjust_xy"
        combined_reason = "末端执行器越界，需限制轨迹目标或工作空间。"
    elif force_norm < 1.0:
        failure_type = "no_contact"
        suggested_action = "lower_approach_height"
        combined_reason = "最后一帧接触力很小，可能未接触或夹持不足。"
    elif force_norm > safe_upper:
        failure_type = "over_force"
        suggested_action = "slow_closing"
        combined_reason = "最后一帧接触力处于风险区，建议降低闭合速度或闭合量。"
    else:
        failure_type = "timeout_or_no_success"
        suggested_action = "adjust_xy"
        combined_reason = "接触力在安全范围内但未成功，可能是位姿或轨迹问题。"

    return {
        "failure_type": failure_type,
        "visual_reason": visual_reason,
        "force_reason": force_reason,
        "combined_reason": combined_reason,
        "observed_force_n": force_norm,
        "break_threshold_n": threshold,
        "suggested_force_range_n": [1.0, safe_upper],
        "suggested_action": suggested_action,
        "confidence": 0.35,
    }


def _normalize_analysis(analysis: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    threshold = float(context["break_force_threshold_n"])
    safe_upper = min(0.85 * threshold, threshold - 1.0e-6)
    force_range = analysis.get("suggested_force_range_n", [1.0, 0.75 * threshold])
    if not isinstance(force_range, list) or len(force_range) != 2:
        force_range = [1.0, 0.75 * threshold]
    lo = max(0.0, float(force_range[0]))
    hi = max(lo, min(float(force_range[1]), safe_upper))

    failure_type = str(analysis.get("failure_type", "unknown"))
    suggested_action = str(analysis.get("suggested_action", "unknown"))
    return {
        "failure_type": failure_type if failure_type in FAILURE_TYPES else "unknown",
        "visual_reason": str(analysis.get("visual_reason", "")),
        "force_reason": str(analysis.get("force_reason", "")),
        "combined_reason": str(analysis.get("combined_reason", "")),
        "observed_force_n": float(analysis.get("observed_force_n", context["force_norm_n"])),
        "break_threshold_n": float(analysis.get("break_threshold_n", threshold)),
        "suggested_force_range_n": [lo, hi],
        "suggested_action": suggested_action if suggested_action in SUGGESTED_ACTIONS else "unknown",
        "confidence": min(1.0, max(0.0, float(analysis.get("confidence", 0.0)))),
    }


def _write_text_report(path: Path, context: dict[str, Any], analysis: dict[str, Any], image_path: Path):
    force_range = analysis["suggested_force_range_n"]
    text = (
        f"Attempt: {context['attempt_id']}\n"
        f"Analysis frame: {context['analysis_frame']}\n"
        f"Image: {image_path}\n"
        f"Script failure reason: {context['script_failure_reason']}\n"
        f"Failure type: {analysis['failure_type']}\n"
        f"Visual reason: {analysis['visual_reason']}\n"
        f"Force reason: {analysis['force_reason']}\n"
        f"Combined reason: {analysis['combined_reason']}\n"
        f"Observed force: {analysis['observed_force_n']:.6f} N\n"
        f"Break threshold: {analysis['break_threshold_n']:.6f} N\n"
        f"Suggested force range: {force_range[0]:.6f} N - {force_range[1]:.6f} N\n"
        f"Suggested action: {analysis['suggested_action']}\n"
        f"Confidence: {analysis['confidence']:.3f}\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_summary(failed_root: Path, rows: list[dict[str, Any]]):
    json_path = failed_root / "failure_summary.json"
    csv_path = failed_root / "failure_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "attempt_id",
        "analysis_frame",
        "failure_type",
        "script_failure_reason",
        "force_norm_n",
        "torque_norm_nm",
        "suggested_force_min_n",
        "suggested_force_max_n",
        "suggested_action",
        "confidence",
        "analysis_file",
        "image_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote summary: {json_path}")
    print(f"[INFO] wrote summary: {csv_path}")


def _select_attempts(failed_root: Path) -> list[Path]:
    attempts = sorted((p for p in failed_root.glob("attempt_*") if p.is_dir()), key=_attempt_sort_key)
    if args_cli.latest_only and attempts:
        attempts = [attempts[-1]]
    if args_cli.max_attempts > 0:
        attempts = attempts[: args_cli.max_attempts]
    return attempts


def main():
    record_dir = Path(args_cli.record_dir).expanduser().resolve()
    failed_root = record_dir / "failed_attempts"
    if not failed_root.is_dir():
        raise FileNotFoundError(f"Missing failed attempts directory: {failed_root}")

    attempts = _select_attempts(failed_root)
    if not attempts:
        print(f"[WARN] no attempt directories found under {failed_root}")
        return

    api_key = os.environ.get(args_cli.api_key_env, "")
    if not args_cli.dry_run and not api_key:
        raise RuntimeError(f"Set {args_cli.api_key_env} or use --dry_run.")

    rows: list[dict[str, Any]] = []
    for attempt_dir in attempts:
        output_json = attempt_dir / "vlm_failure_analysis.json"
        output_txt = attempt_dir / "vlm_failure_analysis.txt"
        if args_cli.skip_existing and output_json.is_file():
            print(f"[INFO] skip existing {attempt_dir}")
            analysis = json.loads(output_json.read_text(encoding="utf-8"))
        else:
            context = _attempt_context(attempt_dir, args_cli.break_force_threshold_n)
            image_path = _ensure_image_path(attempt_dir, context["analysis_frame"])
            if args_cli.dry_run:
                raw_analysis = _heuristic_analysis(context)
            else:
                raw_analysis = _call_vlm(api_key=api_key, image_path=image_path, context=context)
            analysis = _normalize_analysis(raw_analysis, context)
            output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_text_report(output_txt, context, analysis, image_path)
            print(f"[INFO] analyzed {attempt_dir} failure_type={analysis['failure_type']}")

        context = _attempt_context(attempt_dir, args_cli.break_force_threshold_n)
        image_path = _ensure_image_path(attempt_dir, context["analysis_frame"])
        force_range = analysis["suggested_force_range_n"]
        rows.append(
            {
                "attempt_id": attempt_dir.name,
                "analysis_frame": context["analysis_frame"],
                "failure_type": analysis["failure_type"],
                "script_failure_reason": context["script_failure_reason"],
                "force_norm_n": f"{context['force_norm_n']:.6f}",
                "torque_norm_nm": f"{context['torque_norm_nm']:.6f}",
                "suggested_force_min_n": f"{float(force_range[0]):.6f}",
                "suggested_force_max_n": f"{float(force_range[1]):.6f}",
                "suggested_action": analysis["suggested_action"],
                "confidence": f"{float(analysis['confidence']):.6f}",
                "analysis_file": str(output_json),
                "image_file": str(image_path),
            }
        )

    _write_summary(failed_root, rows)


if __name__ == "__main__":
    main()
