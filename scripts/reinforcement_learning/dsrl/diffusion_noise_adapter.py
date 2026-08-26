from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


class DiffusionNoiseAdapter:
    """Expose a frozen LeRobot Diffusion Policy's initial noise as an RL action.

    DSRL-SAC predicts a flattened tensor with shape ``horizon * action_dim``.
    This adapter reshapes it into the diffusion prior, runs deterministic DDIM
    denoising, and returns the physical action chunk. Observation and action
    normalization remain in LeRobot's saved pre/post processors.
    """

    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
    ) -> None:
        if policy.config.noise_scheduler_type != "DDIM":
            raise ValueError(
                "DSRL requires a deterministic DDIM diffusion policy; "
                f"received {policy.config.noise_scheduler_type!r}."
            )
        if policy.config.action_feature is None:
            raise ValueError("Diffusion policy has no action feature.")

        self.policy = policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.horizon = int(policy.config.horizon)
        self.action_dim = int(policy.config.action_feature.shape[0])
        self.noise_dim = self.horizon * self.action_dim

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path, device: str = "cuda") -> "DiffusionNoiseAdapter":
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        policy = DiffusionPolicy.from_pretrained(checkpoint).to(device)
        device_override = {"device": device}
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        return cls(policy, preprocessor=preprocessor, postprocessor=postprocessor)

    @property
    def noise_shape(self) -> tuple[int, int]:
        return self.horizon, self.action_dim

    def reshape_noise(self, flat_noise: torch.Tensor) -> torch.Tensor:
        if flat_noise.shape[-1] != self.noise_dim:
            raise ValueError(
                f"Expected flattened DSRL action dimension {self.noise_dim}, "
                f"received {flat_noise.shape[-1]}."
            )
        return flat_noise.reshape(*flat_noise.shape[:-1], self.horizon, self.action_dim)

    @torch.inference_mode()
    def decode_processed(self, processed_batch: dict[str, torch.Tensor], flat_noise: torch.Tensor) -> torch.Tensor:
        noise = self.reshape_noise(flat_noise).to(next(self.policy.parameters()).device)
        return self.policy.diffusion.generate_actions(processed_batch, noise=noise)

    @torch.inference_mode()
    def decode(self, raw_batch: dict[str, Any], flat_noise: torch.Tensor) -> torch.Tensor:
        """Decode the first physical action while maintaining policy history queues."""
        if self.preprocessor is None:
            raise RuntimeError("decode(raw_batch, ...) requires a saved LeRobot preprocessor.")
        processed_batch = self.preprocessor(raw_batch)
        action = self.policy.select_action(processed_batch, noise=self.reshape_noise(flat_noise))
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        return action

    def reset(self) -> None:
        self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()
