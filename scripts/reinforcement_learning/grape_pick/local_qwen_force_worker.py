#!/usr/bin/env python3
"""Persistent local Qwen3-VL worker for episodic GrapePick force inference.

The process reads one JSON request per line on stdin and writes one JSON
response per line on stdout.  Loading the VLM once keeps online adaptation
practical while Isaac Lab runs in a separate Python environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        raise ValueError("response contains no JSON object")
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def build_prompt(request: dict[str, Any]) -> str:
    payload = {
        "task": "horizontal robotic grape grasp followed by downward stem detachment",
        "force_definition": (
            "grip_force_n is max(left_pad_normal_force_n, right_pad_normal_force_n); "
            "stem_tension_n is a different force created by the pull"
        ),
        "known_global_bounds": request["physical_force_bounds_n"],
        "minimum_randomized_damage_force_n": request["minimum_damage_force_n"],
        "current_controller_range_n": request["current_range_n"],
        "visual_input_type": request.get("visual_type", "unknown"),
        "just_completed_episode": request["episode"],
        "recent_history": request.get("recent_history", []),
    }
    return (
        "你是机器人葡萄采摘的视觉-力觉顾问。输入图像是刚结束 episode 的真实诊断图"
        "（蓝线为夹持力、橙线为茎秆张力、绿带为当前目标范围、红线为损伤下界），"
        "结合图像、接触力轨迹摘要、结果和历史，为下一个 episode 推断闭环夹持力范围。\n"
        "规则：夹持力过高会压坏葡萄；夹持力只负责稳定摩擦，不能通过增大夹持力代替向下拉力；"
        "未接触、轨迹或未拉断等失败不能自动解释为夹持力不足。成功时可围绕实际稳定力小幅收缩；"
        "压坏时降低范围；明确滑移且实测已跟踪当前范围时才提高。建议应保守、逐 episode 小步变化，"
        "并严格位于 known_global_bounds 内。必须先做数值一致性检查：只有 outcome=crushed 或"
        "peak_grip_force_n 大于等于 minimum_randomized_damage_force_n 时，才允许判为 excessive_force；"
        "峰值仅超过当前目标带不等于超过损伤阈值。若 no_detach 且仍保持接触、茎秆张力未达到"
        "脱离条件，应判为 trajectory 而不是盲目降低夹持力。不要臆造未提供的传感器值。\n"
        "只输出一个 JSON 对象，不要 Markdown："
        '{"force_range_n":[LOW,HIGH],"failure_cause":"success|excessive_force|slip|force_tracking|trajectory|no_contact|unknown",'
        '"force_assessment":"too_low|safe|too_high|not_force_related|unknown",'
        '"rationale":"简短依据","confidence":0.0}\n\n'
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    ).eval()
    print(
        json.dumps(
            {
                "ready": True,
                "advisor": "Qwen3-VL-4B-Instruct",
                "model_path": str(model_path),
                "advisor_is_real_vlm": True,
            }
        ),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                print(json.dumps({"shutdown": True}), flush=True)
                return
            prompt = build_prompt(request)
            image_path = Path(request["image_path"]).expanduser().resolve()
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            completion = generated[:, inputs["input_ids"].shape[1] :]
            raw_response = processor.batch_decode(
                completion, skip_special_tokens=True
            )[0].strip()
            parsed = extract_json_object(raw_response)
            force_range = parsed.get("force_range_n")
            if not isinstance(force_range, list) or len(force_range) != 2:
                raise ValueError("force_range_n must contain [low, high]")
            low, high = map(float, force_range)
            if not (0.0 <= low < high):
                raise ValueError("force_range_n must satisfy 0 <= low < high")
            response = {
                "ok": True,
                "advisor": "Qwen3-VL-4B-Instruct",
                "advisor_is_real_vlm": True,
                "request_episode_index": int(request["episode"]["episode_index"]),
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed": parsed,
                "force_range_n": [low, high],
            }
        except Exception as exc:
            response = {
                "ok": False,
                "advisor": "Qwen3-VL-4B-Instruct",
                "advisor_is_real_vlm": True,
                "raw_response": locals().get("raw_response", ""),
                "prompt": locals().get("prompt", ""),
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
