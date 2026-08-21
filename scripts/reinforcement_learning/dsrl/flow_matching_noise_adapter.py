from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim_robot.deployment.policy_runner import SimActionChunkPolicyRunner


class FlowMatchingNoiseAdapter:
    """Expose a frozen sim_robot Flow Matching prior as a DSRL action space."""

    def __init__(self, runner: SimActionChunkPolicyRunner) -> None:
        self.runner = runner
        self.horizon = int(runner.config.n_action_steps)
        self.action_dim = int(runner.config.action_dim)
        self.noise_dim = self.horizon * self.action_dim
        self.n_action_steps = self.horizon

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        num_inference_steps: int = 20,
        visual_xy_lock_phase: float | None = 0.30,
        use_visual_xy_override: bool = True,
        seed: int | None = None,
    ) -> "FlowMatchingNoiseAdapter":
        runner = SimActionChunkPolicyRunner(
            checkpoint_path=checkpoint,
            device=device,
            use_ema=True,
            num_inference_steps=num_inference_steps,
            seed=seed,
            visual_xy_lock_phase=visual_xy_lock_phase,
            use_visual_xy_override=use_visual_xy_override,
        )
        return cls(runner)

    @property
    def noise_shape(self) -> tuple[int, int]:
        return self.horizon, self.action_dim

    @property
    def observation_dim(self) -> int:
        return int(self.runner.model.obs_encoder.global_cond_dim)

    @property
    def is_ready(self) -> bool:
        return self.runner.is_ready()

    def reshape_noise(self, flat_noise: torch.Tensor) -> torch.Tensor:
        if flat_noise.shape[-1] != self.noise_dim:
            raise ValueError(
                f"Expected flattened DSRL action dimension {self.noise_dim}, "
                f"received {flat_noise.shape[-1]}."
            )
        return flat_noise.reshape(*flat_noise.shape[:-1], self.horizon, self.action_dim)

    def update(self, observation: dict[str, Any]) -> None:
        self.runner.update(observation)

    def normalize_proprioception(
        self,
        physical_proprioception: torch.Tensor,
        *,
        phase: float,
    ) -> torch.Tensor:
        """Normalize the current 10-D CAFE state with the BC statistics."""

        if (
            physical_proprioception.ndim != 2
            or physical_proprioception.shape[-1] != 10
        ):
            raise ValueError(
                "Expected physical proprioception [B, 10], got "
                f"{tuple(physical_proprioception.shape)}."
            )
        extras: list[float] = []
        if bool(getattr(self.runner, "include_phase", False)):
            extras.append(float(np.clip(phase, 0.0, 1.0)))
        if bool(getattr(self.runner, "include_demo_mode", False)):
            mode = getattr(self.runner, "demonstration_mode", "safe")
            extras.append(
                {"position_failure": -1.0, "safe": 0.0, "overforce": 1.0}[mode]
            )
        model_state = physical_proprioception
        if extras:
            conditioning = physical_proprioception.new_tensor(extras).expand(
                physical_proprioception.shape[0], -1
            )
            model_state = torch.cat((physical_proprioception, conditioning), dim=-1)
        if model_state.shape[-1] != int(self.runner.config.robot0_pos_dim):
            raise ValueError(
                "Flow checkpoint state mismatch: prepared "
                f"{model_state.shape[-1]} values, expected "
                f"{self.runner.config.robot0_pos_dim}."
            )
        normalized = self.runner.normalizer.normalize_tensor("robot0_pos", model_state)
        return normalized[:, :10]

    @torch.inference_mode()
    def encode_observation(self) -> torch.Tensor:
        return self.runner.encode_observation()

    @torch.inference_mode()
    def decode(self, flat_noise: torch.Tensor | None = None) -> torch.Tensor:
        noise: np.ndarray | torch.Tensor | None = None
        if flat_noise is not None:
            noise = self.reshape_noise(flat_noise)
        action_chunk = self.runner.predict_action_chunk(initial_noise=noise)
        return torch.as_tensor(action_chunk, dtype=torch.float32, device=self.runner.device)

    @torch.inference_mode()
    def decode_with_normalized(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one native BC chunk in physical and normalized coordinates."""

        physical = self.decode(None)
        normalized = self.runner.normalizer.normalize_tensor("action", physical)
        return physical, normalized

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Convert one or more normalized 10-D CAFE actions to physical units."""

        if normalized_action.ndim != 2 or normalized_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected normalized actions [N, {self.action_dim}], got "
                f"{tuple(normalized_action.shape)}."
            )
        return self.runner.normalizer.unnormalize_tensor("action", normalized_action)

    def reset(self) -> None:
        self.runner.reset()
