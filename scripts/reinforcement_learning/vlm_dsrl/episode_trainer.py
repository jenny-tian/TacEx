"""SKRL trainer that stops on an exact completed-episode budget."""

from __future__ import annotations

from typing import Any

import torch

from skrl.trainers.torch import SequentialTrainer


class EpisodeLimitedSequentialTrainer(SequentialTrainer):
    """Run online updates after every transition until ``num_episodes`` finish."""

    def __init__(
        self,
        *,
        episode_env: Any,
        num_episodes: int | None,
        max_interactions: int,
        stop_at_interaction_budget: bool = False,
        **kwargs: Any,
    ) -> None:
        if num_episodes is not None and num_episodes < 1:
            raise ValueError("num_episodes must be positive or None.")
        if max_interactions < 1:
            raise ValueError("Episode and interaction budgets are invalid.")
        if not stop_at_interaction_budget and num_episodes is None:
            raise ValueError("An episode target is required unless interaction-limited.")
        self.episode_env = episode_env
        self.num_episodes = None if num_episodes is None else int(num_episodes)
        self.max_interactions = int(max_interactions)
        self.stop_at_interaction_budget = bool(stop_at_interaction_budget)
        self.interactions_completed = 0
        self.gradient_updates_completed = 0
        super().__init__(**kwargs)
        if self.num_simultaneous_agents != 1 or self.env.num_envs != 1:
            raise ValueError("Episode-limited training currently requires one agent/env.")

    def train(self) -> None:
        self.agents.enable_training_mode(True)
        observations, _ = self.env.reset()
        states = self.env.state()

        for timestep in range(self.max_interactions):
            self.agents.pre_interaction(
                timestep=timestep, timesteps=self.max_interactions
            )
            with torch.no_grad():
                actions, _ = self.agents.act(
                    observations,
                    states,
                    timestep=timestep,
                    timesteps=self.max_interactions,
                )
                (
                    next_observations,
                    rewards,
                    terminated,
                    truncated,
                    infos,
                ) = self.env.step(actions)
                next_states = self.env.state()
                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.max_interactions,
                )
                if self.cfg.environment_info in infos:
                    for key, value in infos[self.cfg.environment_info].items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            self.agents.track_data(
                                key if "/" in key else f"Info / {key}", value.item()
                            )

            self.agents.post_interaction(
                timestep=timestep, timesteps=self.max_interactions
            )
            self.interactions_completed = timestep + 1
            optimizer_updates = getattr(
                self.agents, "optimizer_updates_completed", None
            )
            if optimizer_updates is not None:
                self.gradient_updates_completed = int(optimizer_updates)
            elif timestep >= int(self.agents.cfg.learning_starts):
                gradient_steps = getattr(self.agents.cfg, "gradient_steps", None)
                if gradient_steps is not None:
                    self.gradient_updates_completed += int(gradient_steps)
                else:
                    rollouts = int(getattr(self.agents.cfg, "rollouts", 0))
                    if rollouts > 0 and not (timestep + 1) % rollouts:
                        self.gradient_updates_completed += int(
                            self.agents.cfg.learning_epochs
                        ) * int(self.agents.cfg.mini_batches)

            should_reset = bool((terminated | truncated).any().item())
            if should_reset:
                result = self.episode_env.complete_pending_episode(
                    dsrl_updates_completed=self.gradient_updates_completed
                )
                print(
                    "[EPISODE] "
                    f"mode={self.episode_env.mode} "
                    f"episode={result['episode_index'] + 1}/"
                    f"{self.num_episodes if self.num_episodes is not None else '?'} "
                    f"success={result['success']} "
                    f"reason={result['terminal_reason']}/"
                    f"{result['diagnosed_failure_reason']} "
                    f"force_peak={result['peak_contact_force_n']:.3f}N",
                    flush=True,
                )
                if (
                    self.num_episodes is not None
                    and self.episode_env.completed_episodes >= self.num_episodes
                ):
                    return
                with torch.no_grad():
                    self.episode_env.begin_auto_reset_episode()
                    observations = next_observations
                    states = next_states
            else:
                observations = next_observations
                states = next_states

        if self.stop_at_interaction_budget:
            return
        raise RuntimeError(
            f"Reached {self.max_interactions} outer interactions with only "
            f"{self.episode_env.completed_episodes}/{self.num_episodes} episodes."
        )


__all__ = ["EpisodeLimitedSequentialTrainer"]
