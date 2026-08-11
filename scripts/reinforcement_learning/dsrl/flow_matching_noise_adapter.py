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
    ) -> "FlowMatchingNoiseAdapter":
        runner = SimActionChunkPolicyRunner(
            checkpoint_path=checkpoint,
            device=device,
            use_ema=True,
            num_inference_steps=num_inference_steps,
            visual_xy_lock_phase=visual_xy_lock_phase,
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

    def reset(self) -> None:
        self.runner.reset()
