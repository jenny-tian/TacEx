"""VLM prompting, structured response parsing, and provider adapters."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

from .contracts import EpisodeFeedback, ForceRange, VLMRecommendation


FAILURE_CAUSES = (
    "success",
    "insufficient_force",
    "slip",
    "excessive_force",
    "bad_alignment",
    "trajectory_error",
    "force_tracking_error",
    "no_contact",
    "unknown",
)
FORCE_ASSESSMENTS = ("too_low", "safe", "too_high", "not_force_related", "unknown")


class VLMAdvisor(Protocol):
    """One-call-per-episode interface used by :class:`EpisodeForceAdaptationLoop`."""

    def recommend(
        self,
        *,
        current_range_n: ForceRange,
        episode: EpisodeFeedback,
        history: Sequence[EpisodeFeedback],
        image_paths: Sequence[str | Path] = (),
    ) -> VLMRecommendation: ...


def force_recommendation_schema() -> dict[str, Any]:
    """Return the strict response schema shared by supported HTTP APIs."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_contact_force_range_n": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "failure_cause": {"type": "string", "enum": list(FAILURE_CAUSES)},
            "force_assessment": {"type": "string", "enum": list(FORCE_ASSESSMENTS)},
            "rationale": {"type": "string"},
            "next_experiment": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "target_contact_force_range_n",
            "failure_cause",
            "force_assessment",
            "rationale",
            "next_experiment",
            "confidence",
        ],
    }


def _history_payload(
    history: Sequence[EpisodeFeedback], *, recent_limit: int = 12
) -> dict[str, Any]:
    failures = Counter(item.failure_reason for item in history if not item.success)
    return {
        "completed_episodes": len(history),
        "successes": sum(int(item.success) for item in history),
        "failure_reason_counts": dict(sorted(failures.items())),
        "recent_episodes": [item.to_dict() for item in history[-recent_limit:]],
    }


def build_force_advisor_prompt(
    *,
    current_range_n: ForceRange,
    episode: EpisodeFeedback,
    history: Sequence[EpisodeFeedback],
    physical_range_n: ForceRange,
    break_force_threshold_n: float,
) -> str:
    """Build an evidence-first prompt for episode-level force-range selection."""

    payload = {
        "task": "Franka parallel-jaw grasp and lift of one fragile glass microscope slide",
        "force_definition": (
            "target_contact_force_range_n is the normal load of one fingertip. "
            "mean_contact_force_n is the mean of the two fingertip magnitudes; "
            "peak_contact_force_n is the larger fingertip peak used for break safety."
        ),
        "physical_allowed_range_n": physical_range_n.as_list(),
        "break_force_threshold_per_finger_n": float(break_force_threshold_n),
        "current_estimate_n": current_range_n.as_list(),
        "just_completed_episode": episode.to_dict(),
        "history_before_this_episode": _history_payload(history),
    }
    return (
        "你是脆性物体机器人抓取的 VLM 力学顾问。每个 episode 结束后只做一次判断。\n"
        "目标是根据图像、触觉/特权接触力和历史结果，估计下一 episode 的单指法向目标接触力范围。\n"
        "推理规则：\n"
        "1) 先判断失败是否由力导致。姿态偏差、未接触、轨迹或工作空间错误不能被误判为抓力不足。\n"
        "2) 只有实际平均力已达到当前目标范围、仍发生滑移/掉落时，才可提高下界；若实测力未达到目标下界，"
        "这是力跟踪/接触质量问题，不能据此上调任务目标力。破碎或过力时必须降低上界；成功时围绕稳定实测力收缩。\n"
        "3) 历史证据优先于单帧外观；建议必须落在 physical_allowed_range_n 内并低于破碎阈值。\n"
        "4) 做小步、可验证的更新；证据不足时保持当前范围。不要臆造传感器读数。\n"
        "5) 输出只包含 schema 所要求的 JSON。范围必须严格递增，单位 N。\n\n"
        "实验上下文：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _data_url(path: str | Path) -> str:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"VLM image does not exist: {image_path}")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_responses_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    if not chunks:
        raise ValueError("VLM response did not contain output text.")
    return "\n".join(chunks)


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        stripped = fenced.group(1)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object.")
    return value


def _parse_recommendation(
    raw: dict[str, Any], *, physical_range_n: ForceRange
) -> VLMRecommendation:
    proposed = ForceRange.from_pair(
        raw.get("target_contact_force_range_n"), name="VLM force range"
    )
    if (
        proposed.minimum_n < physical_range_n.minimum_n
        or proposed.maximum_n > physical_range_n.maximum_n
    ):
        raise ValueError(
            f"VLM range {proposed.as_list()} is outside physical bounds {physical_range_n.as_list()}."
        )
    failure_cause = str(raw.get("failure_cause", "unknown"))
    force_assessment = str(raw.get("force_assessment", "unknown"))
    if failure_cause not in FAILURE_CAUSES:
        raise ValueError(f"Unknown failure_cause {failure_cause!r}.")
    if force_assessment not in FORCE_ASSESSMENTS:
        raise ValueError(f"Unknown force_assessment {force_assessment!r}.")
    return VLMRecommendation(
        target_range_n=proposed,
        failure_cause=failure_cause,
        force_assessment=force_assessment,
        rationale=str(raw.get("rationale", "")),
        next_experiment=str(raw.get("next_experiment", "")),
        confidence=float(raw.get("confidence", 0.0)),
    )


class OpenAICompatibleVLMAdvisor:
    """Structured multimodal advisor using Responses or Chat Completions HTTP APIs."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        physical_range_n: ForceRange,
        break_force_threshold_n: float,
        api_base: str = "https://api.openai.com/v1",
        api_mode: str = "responses",
        timeout_s: float = 90.0,
    ) -> None:
        if not model.strip() or not api_key.strip():
            raise ValueError("model and api_key must be non-empty.")
        if api_mode not in {"responses", "chat_completions"}:
            raise ValueError("api_mode must be 'responses' or 'chat_completions'.")
        self.model = model
        self.api_key = api_key
        self.physical_range_n = physical_range_n
        self.break_force_threshold_n = float(break_force_threshold_n)
        self.api_base = api_base.rstrip("/")
        self.api_mode = api_mode
        self.timeout_s = float(timeout_s)
        self.call_count = 0
        self.last_prompt = ""
        self.last_raw_response: dict[str, Any] | None = None

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.api_base}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"VLM request failed with HTTP {exc.code}: {detail}"
            ) from exc

    def recommend(
        self,
        *,
        current_range_n: ForceRange,
        episode: EpisodeFeedback,
        history: Sequence[EpisodeFeedback],
        image_paths: Sequence[str | Path] = (),
    ) -> VLMRecommendation:
        prompt = build_force_advisor_prompt(
            current_range_n=current_range_n,
            episode=episode,
            history=history,
            physical_range_n=self.physical_range_n,
            break_force_threshold_n=self.break_force_threshold_n,
        )
        self.call_count += 1
        self.last_prompt = prompt
        if self.api_mode == "responses":
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            content.extend(
                {"type": "input_image", "image_url": _data_url(path)}
                for path in image_paths
            )
            raw_response = self._post(
                "responses",
                {
                    "model": self.model,
                    "input": [{"role": "user", "content": content}],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "labpick_target_contact_force",
                            "schema": force_recommendation_schema(),
                            "strict": True,
                        }
                    },
                },
            )
            text = _extract_responses_text(raw_response)
        else:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": _data_url(path)}}
                for path in image_paths
            )
            raw_response = self._post(
                "chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "labpick_target_contact_force",
                            "schema": force_recommendation_schema(),
                            "strict": True,
                        },
                    },
                },
            )
            text = str(raw_response["choices"][0]["message"]["content"])
        self.last_raw_response = raw_response
        return _parse_recommendation(
            _parse_json_text(text), physical_range_n=self.physical_range_n
        )


class DeterministicVLMAdvisor:
    """Auditable offline stand-in for integration tests when no VLM endpoint is available.

    This class follows the same structured contract and is deliberately labelled
    as a deterministic stand-in. It must not be reported as a real VLM call.
    """

    def __init__(
        self,
        *,
        physical_range_n: ForceRange,
        break_force_threshold_n: float | None = None,
        step_n: float = 0.35,
    ) -> None:
        if step_n <= 0.0:
            raise ValueError("step_n must be positive.")
        self.physical_range_n = physical_range_n
        self.break_force_threshold_n = (
            physical_range_n.maximum_n
            if break_force_threshold_n is None
            else float(break_force_threshold_n)
        )
        if self.break_force_threshold_n <= 0.0:
            raise ValueError("break_force_threshold_n must be positive.")
        self.step_n = float(step_n)
        self.call_count = 0
        self.last_prompt = ""

    def recommend(
        self,
        *,
        current_range_n: ForceRange,
        episode: EpisodeFeedback,
        history: Sequence[EpisodeFeedback],
        image_paths: Sequence[str | Path] = (),
    ) -> VLMRecommendation:
        del image_paths
        self.call_count += 1
        self.last_prompt = build_force_advisor_prompt(
            current_range_n=current_range_n,
            episode=episode,
            history=history,
            physical_range_n=self.physical_range_n,
            break_force_threshold_n=self.break_force_threshold_n,
        )
        reason = episode.failure_reason.lower()
        center = current_range_n.center_n
        width = current_range_n.width_n
        cause = "trajectory_error"
        assessment = "not_force_related"
        rationale = "失败证据未指向抓取力，保持当前范围。"
        next_center = center

        if episode.success:
            cause = "success"
            assessment = "safe"
            observed = (
                episode.mean_contact_force_n
                if episode.mean_contact_force_n > 0.0
                else center
            )
            next_center = 0.65 * observed + 0.35 * center
            width *= 0.75
            rationale = "抓取成功，围绕稳定接触力收缩搜索范围。"
        elif any(token in reason for token in ("broken", "over_force", "excessive")):
            cause = "excessive_force"
            assessment = "too_high"
            next_center = center - self.step_n
            width *= 0.8
            rationale = "单指峰值力触发破碎/过力约束，降低目标力。"
        elif any(
            token in reason for token in ("slip", "dropped", "insufficient", "too_low")
        ):
            cause = (
                "slip"
                if "slip" in reason or "dropped" in reason
                else "insufficient_force"
            )
            assessment = "too_low"
            next_center = center + self.step_n
            width *= 0.8
            rationale = "已接触但滑移或掉落，逐步提高目标力。"
        elif "force_tracking" in reason:
            cause = "force_tracking_error"
            rationale = (
                "实测力未达到已设目标，属于执行/接触质量问题，不能上调任务目标力。"
            )
        elif "no_contact" in reason or episode.contact_fraction <= 0.0:
            cause = "no_contact"
            rationale = "没有形成接触，不能把失败归因于目标力，保持当前范围。"
        elif "alignment" in reason or "too_far" in reason:
            cause = "bad_alignment"
            rationale = "接触位姿偏差是主因，保持当前力范围。"

        half = max(0.10, 0.5 * width)
        center_low = self.physical_range_n.minimum_n + half
        center_high = self.physical_range_n.maximum_n - half
        if center_low > center_high:
            half = 0.5 * self.physical_range_n.width_n
            center_low = center_high = self.physical_range_n.center_n
        next_center = min(center_high, max(center_low, next_center))
        target_range = ForceRange(next_center - half, next_center + half)
        return VLMRecommendation(
            target_range_n=target_range,
            failure_cause=cause,
            force_assessment=assessment,
            rationale=rationale,
            next_experiment="保持位姿策略不变，仅验证新的目标力范围。",
            confidence=(
                0.75
                if cause in {"success", "excessive_force", "slip", "insufficient_force"}
                else 0.45
            ),
        )


__all__ = [
    "DeterministicVLMAdvisor",
    "OpenAICompatibleVLMAdvisor",
    "VLMAdvisor",
    "build_force_advisor_prompt",
    "force_recommendation_schema",
]
