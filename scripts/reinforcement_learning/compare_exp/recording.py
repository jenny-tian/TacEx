"""Durable online transition/episode recording shared by comparison baselines."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from online_dsrl_metrics import any_flag, classify_terminal, extract_step_metrics, scalar


class OnlineEpisodeRecorder(gym.Wrapper):
    """Record every outer interaction and every terminal outcome as JSONL."""

    def __init__(
        self,
        env: gym.Env,
        *,
        output_dir: str | Path,
        mode: str,
        experiment_seed: int,
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interaction_path = self.output_dir / "interactions.jsonl"
        self.episode_path = self.output_dir / "episodes.jsonl"
        self.trajectory_dir = self.output_dir / "trajectories"
        self.trajectory_dir.mkdir(exist_ok=True)
        for path in (self.interaction_path, self.episode_path):
            if path.exists():
                raise FileExistsError(f"Refusing to append to existing log: {path}")
        self._interaction_stream = self.interaction_path.open("x", encoding="utf-8", buffering=1)
        self._episode_stream = self.episode_path.open("x", encoding="utf-8", buffering=1)
        self.mode = str(mode)
        self.experiment_seed = int(experiment_seed)
        self.completed_episodes = 0
        self.total_outer_interactions = 0
        self.results: list[dict[str, Any]] = []
        self._pending_episode: dict[str, Any] | None = None
        self._active_reset_seed: int | None = None
        self._trajectory_stream: Any | None = None
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._episode_return = 0.0
        self._episode_outer_interactions = 0
        self._episode_physics_steps = 0
        self._episode_peak_force_n = 0.0
        self._episode_max_lift_m = 0.0
        self._episode_min_grasp_distance_m = float("inf")
        self._episode_flags = {
            "object_broken": False,
            "object_dropped": False,
            "object_too_far": False,
            "ee_outside_workspace": False,
            "success": False,
            "timeout": False,
        }
        self._terminal_tactile_actor: list[float] | None = None

    @staticmethod
    def _write(stream: Any, row: dict[str, Any], *, sync: bool = False) -> None:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if sync:
            stream.flush()
            os.fsync(stream.fileno())

    def reset(self, **kwargs: Any):
        if self._pending_episode is not None:
            raise RuntimeError("Commit the pending terminal episode before reset.")
        seed = kwargs.get("seed")
        self._active_reset_seed = None if seed is None else int(seed)
        self._reset_episode_state()
        self._open_trajectory()
        return self.env.reset(**kwargs)

    def _open_trajectory(self) -> None:
        if self._trajectory_stream is not None and not self._trajectory_stream.closed:
            raise RuntimeError("Previous residual trajectory stream is still open.")
        path = self.trajectory_dir / f"episode_{self.completed_episodes:03d}.jsonl.gz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite trajectory: {path}")
        self._trajectory_stream = gzip.open(path, "xt", encoding="utf-8")

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_outer_interactions += 1
        self._episode_outer_interactions += 1
        physics_steps = int(
            info.get(
                "dppo/physics_steps_executed",
                info.get(
                    "dsrl/physics_steps_executed",
                    info.get("clean_residual/action_repeat", 1),
                ),
            )
        )
        self._episode_physics_steps += physics_steps
        reward_value = scalar(reward)
        self._episode_return += reward_value
        metrics = extract_step_metrics(info)
        for name, value in metrics["flags"].items():
            self._episode_flags[name] |= bool(value)
        self._episode_peak_force_n = max(
            self._episode_peak_force_n, metrics["contact_force_n"]
        )
        self._episode_max_lift_m = max(
            self._episode_max_lift_m, metrics["lift_m"]
        )
        if metrics["grasp_distance_m"] > 0.0:
            self._episode_min_grasp_distance_m = min(
                self._episode_min_grasp_distance_m,
                metrics["grasp_distance_m"],
            )
        terminal = any_flag(terminated) or any_flag(truncated)
        tactile_actor = (
            torch.as_tensor(info["tactile_actor"])
            .detach()
            .cpu()
            .float()
            .reshape(-1)
            .tolist()
        )
        if len(tactile_actor) != 5:
            raise ValueError(f"Expected 5-D tactile_actor in info, received {len(tactile_actor)}.")
        status, success, failure_reason = classify_terminal(
            self._episode_flags,
            terminated=any_flag(terminated),
            truncated=any_flag(truncated),
        )
        row = {
            "schema_version": 1,
            "mode": self.mode,
            "interaction_index": self.total_outer_interactions,
            "episode_index": self.completed_episodes,
            "episode_interaction": self._episode_outer_interactions,
            "terminal": terminal,
            "success": success if terminal else None,
            "failure_reason": failure_reason if terminal else None,
            "action": torch.as_tensor(action).detach().cpu().reshape(-1).tolist(),
            "reward": reward_value,
            "episode_return_so_far": self._episode_return,
            "physics_steps": physics_steps,
            "episode_physics_steps_so_far": self._episode_physics_steps,
            "metrics": metrics,
            "tactile_actor": tactile_actor,
        }
        self._write(self._interaction_stream, row)
        if self._trajectory_stream is None:
            raise RuntimeError("Residual trajectory stream is unavailable before a step.")
        self._write(self._trajectory_stream, row, sync=terminal)
        if terminal:
            self._trajectory_stream.close()
        if terminal:
            self._terminal_tactile_actor = tactile_actor
            self._pending_episode = {
                "status": status,
                "success": bool(success),
                "terminal_reason": "success" if success else str(failure_reason),
            }
        return observation, reward, terminated, truncated, info

    def complete_pending_episode(self, *, dsrl_updates_completed: int) -> dict[str, Any]:
        if self._pending_episode is None:
            raise RuntimeError("No terminal episode is waiting to be committed.")
        result = {
            "schema_version": 1,
            "episode_index": self.completed_episodes,
            "experiment_seed": self.experiment_seed,
            "reset_seed": self._active_reset_seed,
            "mode": self.mode,
            **self._pending_episode,
            "diagnosed_failure_reason": self._pending_episode["terminal_reason"],
            "episode_return": self._episode_return,
            "outer_interactions": self._episode_outer_interactions,
            "physics_steps": self._episode_physics_steps,
            "ending_outer_interaction": self.total_outer_interactions,
            "gradient_updates_completed": int(dsrl_updates_completed),
            "peak_contact_force_n": self._episode_peak_force_n,
            "terminal_tactile_actor": self._terminal_tactile_actor,
            "max_lift_m": self._episode_max_lift_m,
            "min_grasp_distance_m": (
                None
                if self._episode_min_grasp_distance_m == float("inf")
                else self._episode_min_grasp_distance_m
            ),
            "trajectory": str(
                self.trajectory_dir / f"episode_{self.completed_episodes:03d}.jsonl.gz"
            ),
        }
        self._interaction_stream.flush()
        os.fsync(self._interaction_stream.fileno())
        self._write(self._episode_stream, result, sync=True)
        self.results.append(result)
        self.completed_episodes += 1
        self._pending_episode = None
        partial = self.output_dir / "results.partial.json"
        temporary = partial.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"completed_episodes": self.completed_episodes, "results": self.results},
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(partial)
        return result

    def begin_auto_reset_episode(self) -> None:
        if self._pending_episode is not None:
            raise RuntimeError("Commit the pending terminal episode first.")
        self._active_reset_seed = None
        self._reset_episode_state()
        self._open_trajectory()

    def close(self):
        streams = [self._interaction_stream, self._episode_stream]
        if self._trajectory_stream is not None:
            streams.append(self._trajectory_stream)
        for stream in streams:
            if not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        return super().close()


__all__ = ["OnlineEpisodeRecorder"]
