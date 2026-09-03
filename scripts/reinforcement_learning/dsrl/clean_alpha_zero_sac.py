"""A minimal alpha-zero SAC update used by the clean residual baseline.

This class intentionally keeps the replay, stochastic policy, twin critics,
target critics and Polyak updates from SAC while removing every entropy term
from the optimization objective. It is an ablation named SAC for continuity
with the project, not the maximum-entropy SAC objective from the paper.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.sac import SAC, SAC_CFG


class CleanAlphaZeroSAC(SAC):
    """One-step twin-Q SAC with a base-anchor penalty and no entropy objective."""

    def __init__(
        self,
        *,
        cfg: SAC_CFG | dict,
        action_l2_weight: float = 0.0,
        **kwargs,
    ) -> None:
        resolved_cfg = SAC_CFG(**cfg) if isinstance(cfg, dict) else cfg
        if resolved_cfg.learn_entropy:
            raise ValueError("CleanAlphaZeroSAC requires learn_entropy=False.")
        if float(resolved_cfg.initial_entropy_value) != 0.0:
            raise ValueError("CleanAlphaZeroSAC requires initial_entropy_value=0.0.")
        if int(resolved_cfg.random_timesteps) != 0:
            raise ValueError(
                "CleanAlphaZeroSAC requires random_timesteps=0 so the same "
                "stochastic policy collects every transition."
            )
        if not math.isfinite(action_l2_weight) or action_l2_weight < 0.0:
            raise ValueError("action_l2_weight must be finite and non-negative.")
        self.action_l2_weight = float(action_l2_weight)
        super().__init__(cfg=resolved_cfg, **kwargs)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Run one-step alpha-zero SAC updates.

        The targets and losses are deliberately written out here so there is
        no hidden ``0 * log_prob`` term:

        ``y = r + gamma * (1 - terminated) * min(Q1_target, Q2_target)``

        ``L_policy = -mean(min(Q1(s, a_pi), Q2(s, a_pi)))``
        """

        for _ in range(self.cfg.gradient_steps):
            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
            ) = self.memory.sample(
                names=self._tensors_names,
                batch_size=self.cfg.batch_size,
            )[0]

            with torch.autocast(
                device_type=self._device_type,
                enabled=self.cfg.mixed_precision,
            ):
                inputs = {
                    "observations": self._observation_preprocessor(
                        sampled_observations, train=True
                    ),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(
                        sampled_next_observations, train=True
                    ),
                    "states": self._state_preprocessor(
                        sampled_next_states, train=True
                    ),
                }

                with torch.no_grad():
                    next_actions, _ = self.policy.act(next_inputs, role="policy")
                    target_q1, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_actions},
                        role="target_critic_1",
                    )
                    target_q2, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_actions},
                        role="target_critic_2",
                    )
                    target_q = torch.minimum(target_q1, target_q2)
                    target_values = sampled_rewards + (
                        self.cfg.discount_factor
                        * sampled_terminated.logical_not()
                        * target_q
                    )

                critic_1_values, _ = self.critic_1.act(
                    {**inputs, "taken_actions": sampled_actions},
                    role="critic_1",
                )
                critic_2_values, _ = self.critic_2.act(
                    {**inputs, "taken_actions": sampled_actions},
                    role="critic_2",
                )
                critic_loss = 0.5 * (
                    F.mse_loss(critic_1_values, target_values)
                    + F.mse_loss(critic_2_values, target_values)
                )

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.critic_1.parameters(), self.critic_2.parameters()
                    ),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            with torch.autocast(
                device_type=self._device_type,
                enabled=self.cfg.mixed_precision,
            ):
                policy_actions, policy_outputs = self.policy.act(inputs, role="policy")
                policy_q1, _ = self.critic_1.act(
                    {**inputs, "taken_actions": policy_actions}, role="critic_1"
                )
                policy_q2, _ = self.critic_2.act(
                    {**inputs, "taken_actions": policy_actions}, role="critic_2"
                )
                q_policy_loss = -torch.minimum(policy_q1, policy_q2).mean()
                action_l2_loss = policy_actions.square().mean()
                policy_loss = q_policy_loss + self.action_l2_weight * action_l2_loss

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if config.torch.is_distributed:
                self.policy.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.grad_norm_clip
                )
            self.scaler.step(self.policy_optimizer)

            self.scaler.update()
            self.target_critic_1.update_parameters(
                self.critic_1, polyak=self.cfg.polyak
            )
            self.target_critic_2.update_parameters(
                self.critic_2, polyak=self.cfg.polyak
            )

            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            if self.write_interval > 0 and timestep % self.write_interval == 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Policy Q loss", q_policy_loss.item())
                self.track_data("Loss / Policy action L2", action_l2_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())
                self.track_data("Q-network / Q1 (max)", policy_q1.max().item())
                self.track_data("Q-network / Q1 (min)", policy_q1.min().item())
                self.track_data("Q-network / Q1 (mean)", policy_q1.mean().item())
                self.track_data("Q-network / Q2 (max)", policy_q2.max().item())
                self.track_data("Q-network / Q2 (min)", policy_q2.min().item())
                self.track_data("Q-network / Q2 (mean)", policy_q2.mean().item())
                self.track_data("Target / Target (max)", target_values.max().item())
                self.track_data("Target / Target (min)", target_values.min().item())
                self.track_data("Target / Target (mean)", target_values.mean().item())
                self.track_data(
                    "Action / Sampled residual RMS",
                    policy_actions.square().mean().sqrt().item(),
                )
                self.track_data(
                    "Action / Deterministic residual RMS",
                    policy_outputs["mean_actions"].square().mean().sqrt().item(),
                )
                self.track_data(
                    "Policy / Log std mean",
                    policy_outputs["log_std"].mean().item(),
                )
                if self.policy_scheduler:
                    self.track_data(
                        "Learning / Policy learning rate",
                        self.policy_scheduler.get_last_lr()[0],
                    )
                if self.critic_scheduler:
                    self.track_data(
                        "Learning / Critic learning rate",
                        self.critic_scheduler.get_last_lr()[0],
                    )
