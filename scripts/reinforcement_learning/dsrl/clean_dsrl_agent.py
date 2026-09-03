"""SAC update used by the clean Flow-noise DSRL path.

This matches the reference DSRL objective: the actor loss contains entropy,
the temperature is learned, and the critic target omits entropy by default.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.sac import SAC, SAC_CFG


class CleanDSRLSAC(SAC):
    """Twin-Q DSRL-SAC with an optional entropy-free target backup."""

    def __init__(
        self,
        *,
        cfg: SAC_CFG | dict,
        backup_entropy: bool = False,
        minimum_entropy_value: float | None = 1.0e-3,
        action_l2_weight: float = 0.0,
        max_gradient_updates: int | None = None,
        **kwargs,
    ) -> None:
        resolved_cfg = SAC_CFG(**cfg) if isinstance(cfg, dict) else cfg
        if not resolved_cfg.learn_entropy:
            raise ValueError("CleanDSRLSAC requires learn_entropy=True.")
        if float(resolved_cfg.initial_entropy_value) <= 0.0:
            raise ValueError("initial_entropy_value must be positive.")
        if int(resolved_cfg.random_timesteps) != 0:
            raise ValueError("CleanDSRLSAC samples its actor from timestep zero.")
        if minimum_entropy_value is not None:
            if not math.isfinite(minimum_entropy_value) or minimum_entropy_value <= 0.0:
                raise ValueError("minimum_entropy_value must be positive or None.")
        self.backup_entropy = bool(backup_entropy)
        self.minimum_entropy_value = minimum_entropy_value
        if not math.isfinite(action_l2_weight) or action_l2_weight < 0.0:
            raise ValueError("action_l2_weight must be finite and non-negative.")
        self.action_l2_weight = float(action_l2_weight)
        if max_gradient_updates is not None:
            if isinstance(max_gradient_updates, bool) or not isinstance(max_gradient_updates, int):
                raise TypeError("max_gradient_updates must be an integer or None.")
            if max_gradient_updates < 1:
                raise ValueError("max_gradient_updates must be positive or None.")
        self.max_gradient_updates = max_gradient_updates
        self.optimizer_updates_completed = 0
        super().__init__(cfg=resolved_cfg, **kwargs)

    def update(self, *, timestep: int, timesteps: int) -> None:
        del timesteps
        for _ in range(self.cfg.gradient_steps):
            if (
                self.max_gradient_updates is not None
                and self.optimizer_updates_completed >= self.max_gradient_updates
            ):
                return
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
                    next_actions, next_outputs = self.policy.act(
                        next_inputs, role="policy"
                    )
                    next_q1, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_actions},
                        role="target_critic_1",
                    )
                    next_q2, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_actions},
                        role="target_critic_2",
                    )
                    next_q = torch.minimum(next_q1, next_q2)
                    if self.backup_entropy:
                        next_q = (
                            next_q
                            - self._entropy_coefficient * next_outputs["log_prob"]
                        )
                    target_values = sampled_rewards + (
                        self.cfg.discount_factor
                        * sampled_terminated.logical_not()
                        * next_q
                    )

                critic_q1, _ = self.critic_1.act(
                    {**inputs, "taken_actions": sampled_actions}, role="critic_1"
                )
                critic_q2, _ = self.critic_2.act(
                    {**inputs, "taken_actions": sampled_actions}, role="critic_2"
                )
                critic_loss = 0.5 * (
                    F.mse_loss(critic_q1, target_values)
                    + F.mse_loss(critic_q2, target_values)
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
                actions, outputs = self.policy.act(inputs, role="policy")
                log_prob = outputs["log_prob"]
                policy_q1, _ = self.critic_1.act(
                    {**inputs, "taken_actions": actions}, role="critic_1"
                )
                policy_q2, _ = self.critic_2.act(
                    {**inputs, "taken_actions": actions}, role="critic_2"
                )
                q_policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.minimum(policy_q1, policy_q2)
                ).mean()
                action_l2_loss = actions.square().mean()
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

            with torch.autocast(
                device_type=self._device_type,
                enabled=self.cfg.mixed_precision,
            ):
                entropy_loss = -(
                    self.log_entropy_coefficient
                    * (log_prob + self._target_entropy).detach()
                ).mean()
            self.entropy_optimizer.zero_grad()
            self.scaler.scale(entropy_loss).backward()
            self.scaler.step(self.entropy_optimizer)
            if self.minimum_entropy_value is not None:
                with torch.no_grad():
                    self.log_entropy_coefficient.clamp_(
                        min=math.log(self.minimum_entropy_value)
                    )
            self._entropy_coefficient = torch.exp(
                self.log_entropy_coefficient.detach()
            )

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
            self.optimizer_updates_completed += 1

            if self.write_interval > 0 and timestep % self.write_interval == 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Policy Q-entropy loss", q_policy_loss.item())
                self.track_data("Loss / Policy action L2", action_l2_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())
                self.track_data("Loss / Entropy loss", entropy_loss.item())
                self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())
                self.track_data("Q-network / Q1 (mean)", policy_q1.mean().item())
                self.track_data("Q-network / Q2 (mean)", policy_q2.mean().item())
                self.track_data("Target / Target (mean)", target_values.mean().item())
                self.track_data(
                    "DSRL / Sampled noise RMS",
                    actions.square().mean().sqrt().item(),
                )
                self.track_data(
                    "DSRL / Deterministic noise RMS",
                    outputs["mean_actions"].square().mean().sqrt().item(),
                )
                self.track_data("DSRL / Log std mean", outputs["log_std"].mean().item())


__all__ = ["CleanDSRLSAC"]
