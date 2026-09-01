"""Episode transaction boundary for VLM calls, estimator updates, and logging."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .advisor import VLMAdvisor
from .contracts import EpisodeFeedback, ForceDecision, ForceRange
from .estimator import ConvergentForceEstimator


class EpisodeForceAdaptationLoop:
    """Guarantee one advisor call and one force update per completed episode."""

    def __init__(
        self,
        *,
        advisor: VLMAdvisor,
        estimator: ConvergentForceEstimator,
        log_path: str | Path | None = None,
    ) -> None:
        self.advisor = advisor
        self.estimator = estimator
        self.history: list[EpisodeFeedback] = []
        self._completed_episode_ids: set[int] = set()
        self.log_path = (
            None if log_path is None else Path(log_path).expanduser().resolve()
        )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                raise FileExistsError(
                    f"Refusing to append to existing VLM episode log: {self.log_path}"
                )

    @property
    def current_range_n(self) -> ForceRange:
        return self.estimator.current_range_n

    def complete_episode(
        self,
        episode: EpisodeFeedback,
        *,
        image_paths: Sequence[str | Path] = (),
    ) -> ForceDecision:
        if episode.episode_index in self._completed_episode_ids:
            raise RuntimeError(
                f"Episode {episode.episode_index} has already updated the VLM force estimate."
            )
        history_before = tuple(self.history)
        recommendation = self.advisor.recommend(
            current_range_n=self.current_range_n,
            episode=episode,
            history=history_before,
            image_paths=image_paths,
        )
        decision = self.estimator.update(recommendation, episode)
        self.history.append(episode)
        self._completed_episode_ids.add(episode.episode_index)
        if self.log_path is not None:
            prompt = str(getattr(self.advisor, "last_prompt", ""))
            row: dict[str, Any] = {
                "schema_version": 1,
                "advisor_call_index": len(self.history),
                "episode": episode.to_dict(),
                "history_size_before_call": len(history_before),
                "vlm_recommendation": recommendation.to_dict(),
                "decision": decision.to_dict(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "image_paths": [str(Path(path)) for path in image_paths],
            }
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_range_n": self.current_range_n.as_list(),
            "completed_episode_ids": sorted(self._completed_episode_ids),
            "history": [item.to_dict() for item in self.history],
            "estimator": self.estimator.state_dict(),
        }


__all__ = ["EpisodeForceAdaptationLoop"]
