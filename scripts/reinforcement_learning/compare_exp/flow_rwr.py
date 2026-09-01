"""Direct Flow-policy fine-tuning by success-filtered reward-weighted regression.

The baseline deliberately updates the pretrained Flow velocity field itself.
Successful online chunks are treated as weighted self-imitation targets; failed
episodes receive zero weight. This is the binary-reward special case of RWR
(also commonly called self-imitation learning), and is intentionally distinct
from DSRL, which keeps the Flow model frozen and learns an initial-noise policy.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hybrid_wrapper import VLMDSRLLabPickWrapper


@dataclass
class FlowRWRSample:
    state: torch.Tensor
    features: torch.Tensor
    phase: torch.Tensor
    actions: torch.Tensor
    action_is_pad: torch.Tensor


class FlowRWRFineTuner:
    """Success-filtered replay and direct gradient updates of a Flow model."""

    def __init__(
        self,
        runner: Any,
        *,
        learning_rate: float = 1.0e-5,
        weight_decay: float = 1.0e-4,
        batch_size: int = 8,
        gradient_steps_per_success: int = 4,
        replay_capacity: int = 512,
        grad_clip: float = 1.0,
        seed: int = 0,
    ) -> None:
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("Flow-RWR optimizer parameters are invalid.")
        if min(batch_size, gradient_steps_per_success, replay_capacity) < 1:
            raise ValueError("Flow-RWR batch, update, and replay sizes must be positive.")
        if grad_clip <= 0.0:
            raise ValueError("Flow-RWR grad_clip must be positive.")
        if not hasattr(runner, "model") or not hasattr(runner, "build_model_obs"):
            raise TypeError("Flow-RWR requires the DINOv3 Flow runner.")
        self.runner = runner
        self.model = runner.model
        self.batch_size = int(batch_size)
        self.gradient_steps_per_success = int(gradient_steps_per_success)
        self.replay_capacity = int(replay_capacity)
        self.grad_clip = float(grad_clip)
        self.rng = random.Random(int(seed))
        self.replay: list[FlowRWRSample] = []
        self.pending: list[FlowRWRSample] = []
        self.gradient_updates = 0
        self.successful_episodes_added = 0
        self.last_losses: dict[str, float] = {}
        self.state_key: str | None = None

        # Updating the velocity field is a direct policy update. Conditioning
        # encoders and the frozen DINO backbone remain fixed to reduce drift in
        # the deliberately small online-data regime.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        parameters = list(self.model.velocity_net.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=(0.9, 0.95),
        )

    @torch.no_grad()
    def capture_observation(self) -> dict[str, torch.Tensor]:
        obs = self.runner.build_model_obs()
        state_key = next(
            (key for key in ("robot0_pos", "state", "observation.state") if key in obs),
            None,
        )
        if state_key is None:
            raise KeyError(
                "Flow-RWR could not identify the normalized state key; "
                f"available keys are {sorted(obs)}."
            )
        if self.state_key is not None and state_key != self.state_key:
            raise RuntimeError(
                f"Flow runner state key changed from {self.state_key!r} to {state_key!r}."
            )
        self.state_key = state_key
        features = self.model.extract_dino_features(obs["images"])
        return {
            "state": obs[state_key].detach().cpu(),
            "features": features.detach().to(device="cpu", dtype=torch.float16),
            "phase": obs["phase"].detach().cpu(),
        }

    @torch.no_grad()
    def add_pending_chunk(
        self,
        observation: dict[str, torch.Tensor],
        physical_actions: torch.Tensor,
        *,
        executed_actions: int,
    ) -> None:
        horizon = int(self.runner.config.chunk_size)
        if physical_actions.shape != (horizon, int(self.runner.config.action_dim)):
            raise ValueError(
                "Unexpected Flow action chunk shape: "
                f"{tuple(physical_actions.shape)}."
            )
        if not 1 <= int(executed_actions) <= horizon:
            raise ValueError("executed_actions must lie inside the Flow horizon.")
        normalized = self.runner.normalizer.normalize_tensor(
            "action", physical_actions.detach()
        ).clamp(-1.0, 1.0)
        padding = torch.arange(horizon, device=normalized.device) >= int(executed_actions)
        self.pending.append(
            FlowRWRSample(
                state=observation["state"].clone(),
                features=observation["features"].clone(),
                phase=observation["phase"].clone(),
                actions=normalized.cpu(),
                action_is_pad=padding.cpu(),
            )
        )

    def _batch(self) -> dict[str, Any]:
        if self.state_key is None:
            raise RuntimeError("Capture at least one Flow-RWR observation before sampling replay.")
        size = min(self.batch_size, len(self.replay))
        selected = self.rng.sample(self.replay, size)
        device = self.runner.device
        return {
            "obs": {
                self.state_key: torch.cat([item.state for item in selected]).to(device),
                "dino_features": torch.cat([item.features for item in selected])
                .to(device=device, dtype=torch.float32),
                "phase": torch.cat([item.phase for item in selected]).to(device),
            },
            "action": torch.stack([item.actions for item in selected]).to(device),
            "action_is_pad": torch.stack(
                [item.action_is_pad for item in selected]
            ).to(device),
        }

    def complete_episode(self, *, success: bool) -> int:
        """Commit successful chunks, then optimize; return cumulative updates."""

        if success:
            self.replay.extend(self.pending)
            if len(self.replay) > self.replay_capacity:
                del self.replay[: len(self.replay) - self.replay_capacity]
            self.successful_episodes_added += 1
        self.pending.clear()
        if not success or not self.replay:
            return self.gradient_updates

        self.model.train()
        try:
            for _ in range(self.gradient_steps_per_success):
                losses = self.model.compute_loss(self._batch())
                loss = losses["loss"]
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.velocity_net.parameters(), self.grad_clip
                )
                self.optimizer.step()
                self.gradient_updates += 1
                self.last_losses = {
                    name: float(value.detach().cpu())
                    for name, value in losses.items()
                }
        finally:
            self.model.eval()
        return self.gradient_updates

    def save_checkpoint(self, path: str | Path, *, source_checkpoint: str | Path) -> Path:
        """Write a regular DINOv3 Flow checkpoint loadable by the standard runner."""

        from dinov3_flow import policy_state_dict

        source = Path(source_checkpoint).expanduser().resolve()
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        original = self.runner.checkpoint
        payload = {
            key: copy.deepcopy(value)
            for key, value in original.items()
            if key not in {"model", "ema", "optimizer"}
        }
        state = {
            key: value.detach().cpu()
            for key, value in policy_state_dict(self.model).items()
        }
        payload["model"] = state
        if "ema" in original:
            payload["ema"] = {
                key: copy.deepcopy(value)
                for key, value in original["ema"].items()
                if key != "averaged_model"
            }
            payload["ema"]["averaged_model"] = state
        payload["online_finetuning"] = {
            "algorithm": "success_filtered_reward_weighted_regression",
            "directly_updated_module": "velocity_net",
            "source_checkpoint": str(source),
            "gradient_updates": self.gradient_updates,
            "successful_episodes_added": self.successful_episodes_added,
            "replay_samples": len(self.replay),
            "last_losses": self.last_losses,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return target


class FlowRWRLabPickWrapper(VLMDSRLLabPickWrapper):
    """Native Flow execution plus episode-boundary direct Flow-RWR updates."""

    def __init__(self, *args: Any, fine_tuner_kwargs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, mode="flow_rwr", adaptation=None, **kwargs)
        self.fine_tuner = FlowRWRFineTuner(
            self.adapter.runner,
            **({} if fine_tuner_kwargs is None else fine_tuner_kwargs),
        )
        self._flow_rwr_updates_completed = 0

    def step_bc(self):
        self._ensure_flow_observation_ready()
        observation = self.fine_tuner.capture_observation()
        result = super().step_bc()
        if self.last_decoded_action_chunk is None:
            raise RuntimeError("Flow-RWR did not receive a decoded action chunk.")
        info = result[-1]
        self.fine_tuner.add_pending_chunk(
            observation,
            self.last_decoded_action_chunk,
            executed_actions=int(info["clean_dsrl/action_steps_executed"]),
        )
        return result

    def complete_pending_episode(self, *, dsrl_updates_completed: int = 0):
        if self._pending_episode is None:
            raise RuntimeError("No terminal Flow-RWR episode is pending.")
        self._flow_rwr_updates_completed = self.fine_tuner.complete_episode(
            success=bool(self._pending_episode["success"])
        )
        return super().complete_pending_episode(
            dsrl_updates_completed=dsrl_updates_completed
        )

    def _extra_episode_result(self) -> dict[str, Any]:
        return {
            "flow_rwr_updates_completed": self._flow_rwr_updates_completed,
            "flow_rwr_replay_samples": len(self.fine_tuner.replay),
        }

    def discard_pending_training_samples(self) -> None:
        self.fine_tuner.pending.clear()


def write_flow_rwr_metadata(path: str | Path, fine_tuner: FlowRWRFineTuner) -> None:
    target = Path(path)
    target.write_text(
        json.dumps(
            {
                "algorithm": "success_filtered_reward_weighted_regression",
                "direct_policy_update": True,
                "updated_module": "velocity_net",
                "gradient_updates": fine_tuner.gradient_updates,
                "successful_episodes_added": fine_tuner.successful_episodes_added,
                "replay_samples": len(fine_tuner.replay),
                "last_losses": fine_tuner.last_losses,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FlowRWRFineTuner",
    "FlowRWRLabPickWrapper",
    "FlowRWRSample",
    "write_flow_rwr_metadata",
]
