from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim_robot.common.normalizer import LinearNormalizer
from sim_robot.model.obs_encoder import SingleCameraObsEncoder
from sim_robot.model.transformer import CrossAttentionTransformer


@dataclass
class SimFlowMatchingConfig:
    robot0_pos_dim: int = 10
    action_dim: int = 10
    n_state_obs_steps: int = 2
    n_image_obs_steps: int = 2
    n_action_steps: int = 32
    image_feature_dim: int = 1024
    obs_feature_dim: int = 1024
    transformer_layers: int = 16
    transformer_heads: int = 16
    transformer_embedding_dim: int = 1024
    transformer_cond_layers: int = 2
    dropout: float = 0.1
    time_embed_scale: float = 1000.0
    num_inference_steps: int = 100
    ode_solver: str = "euler"
    clip_sample: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimFlowMatchingConfig":
        return cls(**data)


class SimFlowMatchingPolicy(nn.Module):
    def __init__(self, config: SimFlowMatchingConfig) -> None:
        super().__init__()
        if config.transformer_embedding_dim % config.transformer_heads != 0:
            raise ValueError("transformer_embedding_dim must be divisible by transformer_heads")
        if config.ode_solver not in {"euler", "heun"}:
            raise ValueError("ode_solver must be 'euler' or 'heun'")
        self.config = config
        self.obs_encoder = SingleCameraObsEncoder(
            robot0_pos_dim=config.robot0_pos_dim,
            n_state_obs_steps=config.n_state_obs_steps,
            n_image_obs_steps=config.n_image_obs_steps,
            image_feature_dim=config.image_feature_dim,
            step_feature_dim=config.obs_feature_dim,
            dropout=config.dropout,
        )
        self.velocity_net = CrossAttentionTransformer(
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            horizon=config.n_action_steps,
            cond_dim=config.obs_feature_dim,
            n_cond_tokens=self.obs_encoder.total_cond_tokens,
            n_layer=config.transformer_layers,
            n_head=config.transformer_heads,
            n_emb=config.transformer_embedding_dim,
            p_drop_emb=config.dropout,
            p_drop_attn=config.dropout,
            n_cond_layers=config.transformer_cond_layers,
        )

    def _model_time(self, t: torch.Tensor) -> torch.Tensor:
        return t * float(self.config.time_embed_scale)

    def _model_forward(self, sample: torch.Tensor, t: torch.Tensor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cond_tokens, _ = self.obs_encoder(obs)
        return self.velocity_net(sample, self._model_time(t), cond_tokens=cond_tokens)

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        obs = batch["obs"]
        action = batch["action"]
        batch_size = action.shape[0]
        x0 = torch.randn_like(action)
        x1 = action
        t = torch.rand(batch_size, device=action.device, dtype=action.dtype)
        t_view = t.view(batch_size, *([1] * (action.ndim - 1)))
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        pred_velocity = self._model_forward(xt, t, obs)
        loss = F.mse_loss(pred_velocity, target_velocity)
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(
        self,
        obs: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        obs = {key: value.to(device) for key, value in obs.items()}
        batch_size = obs["robot0_pos"].shape[0]
        steps = self.config.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError("num_inference_steps must be >= 1")

        action = torch.randn(
            batch_size,
            self.config.n_action_steps,
            self.config.action_dim,
            device=device,
            generator=generator,
        )
        dt = 1.0 / float(steps)
        for i in range(steps):
            t0 = torch.full((batch_size,), i / float(steps), device=device, dtype=action.dtype)
            v0 = self._model_forward(action, t0, obs)
            if self.config.ode_solver == "heun" and i < steps - 1:
                proposal = action + dt * v0
                t1 = torch.full((batch_size,), (i + 1) / float(steps), device=device, dtype=action.dtype)
                v1 = self._model_forward(proposal, t1, obs)
                action = action + 0.5 * dt * (v0 + v1)
            else:
                action = action + dt * v0
            if self.config.clip_sample:
                action = action.clamp(-1.0, 1.0)
        return {
            "action": action,
            "action_pred": action,
        }


def load_checkpoint(path: str | Path, map_location=None) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except NotImplementedError as exc:
        message = str(exc)
        if "PosixPath" not in message and "WindowsPath" not in message:
            raise
        import os
        import pathlib

        if os.name == "nt":
            original = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            finally:
                pathlib.PosixPath = original

        original = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        finally:
            pathlib.WindowsPath = original


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> tuple[SimFlowMatchingPolicy, LinearNormalizer, dict]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = SimFlowMatchingConfig.from_dict(ckpt["policy_config"])
    model = SimFlowMatchingPolicy(config).to(device)
    if use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"]["averaged_model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    return model, normalizer, ckpt

