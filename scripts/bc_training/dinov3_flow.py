#!/usr/bin/env python
"""DINOv3-conditioned Flow Matching policy with a DSRL-compatible noise interface."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_bc as common


MODEL_TYPE = "tacex_dinov3_flow_matching_bc_v2"


@dataclass
class DINOv3FlowConfig:
    state_dim: int = 10
    action_dim: int = 10
    num_cameras: int = 2
    state_obs_steps: int = 2
    image_obs_steps: int = 2
    chunk_size: int = 32
    feature_grid_size: int = 7
    condition_grid_size: int = 4
    dino_hidden_size: int = 384
    cond_dim: int = 384
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_dim: int = 384
    transformer_cond_layers: int = 2
    dropout: float = 0.1
    time_embed_scale: float = 1000.0
    num_inference_steps: int = 20
    ode_solver: str = "euler"
    clip_sample: bool = True
    phase_horizon_steps: int = 384
    dino_path: str = ""
    image_keys: tuple[str, ...] = ("robot0_image", "robot0_image_third")

    @property
    def robot0_pos_dim(self) -> int:
        return self.state_dim

    @property
    def n_state_obs_steps(self) -> int:
        return self.state_obs_steps

    @property
    def n_image_obs_steps(self) -> int:
        return self.image_obs_steps

    @property
    def n_action_steps(self) -> int:
        return self.chunk_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DINOv3FlowConfig":
        values = dict(data)
        if isinstance(values.get("image_keys"), list):
            values["image_keys"] = tuple(values["image_keys"])
        return cls(**values)


class DINOv3FlowPolicy(nn.Module):
    """Frozen DINOv3 observation encoder plus a conditional Flow velocity field."""

    def __init__(self, config: DINOv3FlowConfig, dino: common.MinimalDINOv3ViT) -> None:
        super().__init__()
        if config.transformer_dim % config.transformer_heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if not 1 <= config.condition_grid_size <= config.feature_grid_size:
            raise ValueError("condition_grid_size must be in [1, feature_grid_size]")
        if config.ode_solver not in {"euler", "heun"}:
            raise ValueError("ode_solver must be 'euler' or 'heun'")
        self.config = config
        self.dino = dino
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)
        self.dino.eval()

        dim = config.cond_dim
        num_condition_tokens = config.condition_grid_size**2
        self.vision_projector = nn.Sequential(
            nn.LayerNorm(config.dino_hidden_size),
            nn.Linear(config.dino_hidden_size, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.state_projector = nn.Sequential(
            nn.Linear(config.state_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.phase_projector = nn.Sequential(
            nn.Linear(7, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.state_step_embedding = nn.Parameter(torch.randn(1, config.state_obs_steps, dim) * 0.02)
        self.image_step_embedding = nn.Parameter(torch.randn(1, config.image_obs_steps, 1, 1, dim) * 0.02)
        self.camera_embedding = nn.Parameter(torch.randn(1, 1, config.num_cameras, 1, dim) * 0.02)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, 1, num_condition_tokens, dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, dim) * 0.02)

        visual_input_dim = config.feature_grid_size**2 * config.dino_hidden_size
        self.visual_xy_head = nn.Sequential(
            nn.LayerNorm(visual_input_dim),
            nn.Linear(visual_input_dim, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, 2),
        )
        num_condition_inputs = (
            1
            + config.state_obs_steps
            + config.image_obs_steps * config.num_cameras * num_condition_tokens
        )
        self.velocity_net = common.CrossAttentionTransformer(
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            horizon=config.chunk_size,
            cond_dim=dim,
            n_cond_tokens=num_condition_inputs,
            n_layer=config.transformer_layers,
            n_head=config.transformer_heads,
            n_emb=config.transformer_dim,
            dropout=config.dropout,
            n_cond_layers=config.transformer_cond_layers,
        )
        self.dsrl_observation_norm = nn.LayerNorm(dim)

        mean = torch.tensor(common.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(common.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        action_weights = torch.tensor([3.0, 3.0, 2.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 2.0])
        if config.action_dim != action_weights.numel():
            action_weights = torch.ones(config.action_dim)
        self.register_buffer("action_weights", action_weights.view(1, 1, -1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    @property
    def num_spatial_tokens(self) -> int:
        return self.config.feature_grid_size**2

    @staticmethod
    def _phase_features(phase: torch.Tensor) -> torch.Tensor:
        phase = phase.reshape(-1, 1).clamp(0.0, 1.0)
        return torch.cat(
            (
                phase,
                torch.sin(math.pi * phase),
                torch.cos(math.pi * phase),
                torch.sin(2 * math.pi * phase),
                torch.cos(2 * math.pi * phase),
                torch.sin(4 * math.pi * phase),
                torch.cos(4 * math.pi * phase),
            ),
            dim=-1,
        )

    def _spatial_pool(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        side = int(round(math.sqrt(patch_tokens.shape[1])))
        if side * side != patch_tokens.shape[1]:
            raise ValueError("DINO patch tokens must form a square grid")
        grid = patch_tokens.transpose(1, 2).reshape(-1, patch_tokens.shape[-1], side, side)
        pooled = F.adaptive_avg_pool2d(
            grid, output_size=(self.config.feature_grid_size, self.config.feature_grid_size)
        )
        return pooled.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def extract_dino_features(self, images: torch.Tensor) -> torch.Tensor:
        batch, steps, cameras, channels, height, width = images.shape
        flat = images.reshape(batch * steps * cameras, channels, height, width)
        flat = (flat - self.image_mean.to(dtype=flat.dtype)) / self.image_std.to(dtype=flat.dtype)
        features = self._spatial_pool(self.dino(flat))
        return features.reshape(
            batch, steps, cameras, self.num_spatial_tokens, self.config.dino_hidden_size
        )

    def _condition_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.condition_grid_size == self.config.feature_grid_size:
            return features
        batch, steps, cameras, _, hidden = features.shape
        grid = self.config.feature_grid_size
        output = self.config.condition_grid_size
        flat = features.reshape(-1, grid, grid, hidden).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(flat, output_size=(output, output))
        return pooled.permute(0, 2, 3, 1).reshape(batch, steps, cameras, output**2, hidden)

    def encode_observation(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state = obs[common.STATE_KEY]
        if "dino_features" in obs:
            features = obs["dino_features"].float()
        elif "images" in obs:
            features = self.extract_dino_features(obs["images"])
        else:
            raise KeyError("Observation must contain images or dino_features")
        batch, image_steps, cameras, tokens, hidden = features.shape
        expected = (
            self.config.image_obs_steps,
            self.config.num_cameras,
            self.num_spatial_tokens,
            self.config.dino_hidden_size,
        )
        if (image_steps, cameras, tokens, hidden) != expected:
            raise ValueError(f"DINO feature shape mismatch: got {tuple(features.shape)}, expected Bx{expected}")

        state_tokens = self.state_projector(state)
        state_tokens = state_tokens + self.state_step_embedding[:, : state.shape[1]]
        state_tokens = state_tokens + self.modality_embedding[:, 0:1]

        visual_xy = self.visual_xy_head(features[:, -1, -1].reshape(batch, -1))
        condition_features = self._condition_features(features)
        condition_tokens = condition_features.shape[3]
        vision_tokens = self.vision_projector(condition_features)
        vision_tokens = (
            vision_tokens
            + self.image_step_embedding[:, :image_steps]
            + self.camera_embedding[:, :, :cameras]
            + self.spatial_embedding[:, :, :, :condition_tokens]
            + self.modality_embedding[:, 1].view(1, 1, 1, 1, -1)
        ).reshape(batch, -1, self.config.cond_dim)
        phase = obs["phase"].to(dtype=state.dtype)
        phase_token = self.phase_projector(self._phase_features(phase)).unsqueeze(1)
        phase_token = phase_token + self.modality_embedding[:, 2:3]
        return torch.cat((phase_token, state_tokens, vision_tokens), dim=1), visual_xy

    def dsrl_observation(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        return self.dsrl_observation_norm(cond_tokens).mean(dim=1)

    def _velocity(self, sample: torch.Tensor, time: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(
            sample,
            time * float(self.config.time_embed_scale),
            cond_tokens=cond_tokens,
        )

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        target = batch[common.ACTION_KEY]
        cond_tokens, visual_xy = self.encode_observation(batch["obs"])
        batch_size = target.shape[0]
        noise = torch.randn_like(target)
        time = torch.rand(batch_size, device=target.device, dtype=target.dtype)
        time_view = time.view(batch_size, 1, 1)
        sample = (1.0 - time_view) * noise + time_view * target
        target_velocity = target - noise
        prediction = self._velocity(sample, time, cond_tokens)

        valid = batch["action_is_pad"].logical_not().to(dtype=prediction.dtype).unsqueeze(-1)
        step = torch.arange(self.config.chunk_size, device=target.device, dtype=target.dtype)
        time_weights = torch.pow(torch.tensor(0.98, device=target.device), step).view(1, -1, 1)
        weights = valid * time_weights * self.action_weights.to(dtype=target.dtype)
        flow_loss = ((prediction - target_velocity).square() * weights).sum() / weights.sum().clamp_min(1.0)
        visual_xy_loss = F.smooth_l1_loss(visual_xy, target[:, 0, :2], beta=0.05)
        loss = flow_loss + 2.0 * visual_xy_loss
        return {"loss": loss, "flow_loss": flow_loss, "visual_xy_loss": visual_xy_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        obs: dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        obs = {key: value.to(device) for key, value in obs.items()}
        batch_size = obs[common.STATE_KEY].shape[0]
        steps = self.config.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError("num_inference_steps must be positive")
        shape = (batch_size, self.config.chunk_size, self.config.action_dim)
        if initial_noise is None:
            action = torch.randn(shape, device=device, generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial_noise must have shape {shape}, got {tuple(initial_noise.shape)}")
            action = initial_noise.to(device=device, dtype=obs[common.STATE_KEY].dtype).clone()
        cond_tokens, visual_xy = self.encode_observation(obs)
        dt = 1.0 / steps
        for index in range(steps):
            t0 = torch.full((batch_size,), index / steps, device=device, dtype=action.dtype)
            v0 = self._velocity(action, t0, cond_tokens)
            if self.config.ode_solver == "heun" and index < steps - 1:
                proposal = action + dt * v0
                t1 = torch.full((batch_size,), (index + 1) / steps, device=device, dtype=action.dtype)
                v1 = self._velocity(proposal, t1, cond_tokens)
                action = action + 0.5 * dt * (v0 + v1)
            else:
                action = action + dt * v0
            if self.config.clip_sample:
                action = action.clamp(-1.0, 1.0)
        return {
            common.ACTION_KEY: action,
            "action_pred": action,
            "visual_xy": visual_xy,
            "dsrl_observation": self.dsrl_observation(cond_tokens),
        }


def policy_state_dict(model: DINOv3FlowPolicy) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if not key.startswith("dino.")}


def load_policy_state_dict(model: DINOv3FlowPolicy, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("dino.")]
    if missing or unexpected:
        raise RuntimeError(f"Policy state mismatch: missing={missing}, unexpected={unexpected}")


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> tuple[DINOv3FlowPolicy, common.LinearNormalizer, dict[str, Any]]:
    target = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    checkpoint = common.load_checkpoint(checkpoint_path, map_location=target)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Expected {MODEL_TYPE}, got {checkpoint.get('model_type')}")
    config = DINOv3FlowConfig.from_dict(checkpoint["policy_config"])
    dino, _ = common.load_dino_for_config(Path(config.dino_path) if config.dino_path else None)
    model = DINOv3FlowPolicy(config, dino).to(target)
    state = checkpoint["ema"]["averaged_model"] if use_ema and "ema" in checkpoint else checkpoint["model"]
    load_policy_state_dict(model, state)
    model.eval()
    normalizer = common.LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])
    normalizer.stats["robot0_pos"] = normalizer.stats[common.STATE_KEY]
    return model, normalizer, checkpoint


class DINOv3FlowRunner:
    """Online runner implementing the interface consumed by FlowMatchingNoiseAdapter."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        use_ema: bool = True,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        visual_xy_lock_phase: float | None = 0.30,
        use_visual_xy_override: bool = True,
        **_: Any,
    ) -> None:
        self.model, self.normalizer, self.checkpoint = load_policy(
            checkpoint_path, device=device, use_ema=use_ema
        )
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        self.num_inference_steps = num_inference_steps
        self.image_keys = tuple(self.config.image_keys)
        self.include_phase = True
        self.include_demo_mode = False
        self.current_phase = 0.0
        self.visual_xy_lock_phase = visual_xy_lock_phase
        self.use_visual_xy_override = bool(use_visual_xy_override)
        self.locked_visual_xy: np.ndarray | None = None
        self.state_history: deque[np.ndarray] = deque(maxlen=self.config.state_obs_steps)
        self.image_history: deque[np.ndarray] = deque(maxlen=self.config.image_obs_steps)
        self.generator: torch.Generator | None = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(seed))

    @property
    def observation_dim(self) -> int:
        return self.config.cond_dim

    def reset(self, **_: Any) -> None:
        self.state_history.clear()
        self.image_history.clear()
        self.current_phase = 0.0
        self.locked_visual_xy = None

    def update(
        self,
        observation: dict[str, Any] | np.ndarray,
        images: dict[str, np.ndarray] | None = None,
        phase: float | None = None,
    ) -> None:
        if isinstance(observation, dict):
            state = np.asarray(observation["robot0_pos"], dtype=np.float32)
            cameras = [common.normalize_image_array(observation[key], key) for key in self.image_keys]
            phase_value = observation.get("phase", phase)
        else:
            if images is None:
                raise ValueError("images must be provided with an array state")
            state = np.asarray(observation, dtype=np.float32)
            cameras = [common.normalize_image_array(images[key], key) for key in self.image_keys]
            phase_value = phase
        state = state.reshape(-1)
        if state.shape != (self.config.state_dim,):
            raise ValueError(f"state must have shape ({self.config.state_dim},), got {state.shape}")
        camera_stack = np.stack(cameras, axis=0)
        if not self.state_history:
            for _ in range(self.config.state_obs_steps):
                self.state_history.append(state.copy())
            for _ in range(self.config.image_obs_steps):
                self.image_history.append(camera_stack.copy())
        else:
            self.state_history.append(state.copy())
            self.image_history.append(camera_stack.copy())
        if phase_value is not None:
            self.current_phase = float(np.clip(phase_value, 0.0, 1.0))

    def is_ready(self) -> bool:
        return (
            len(self.state_history) == self.config.state_obs_steps
            and len(self.image_history) == self.config.image_obs_steps
        )

    def build_model_obs(self) -> dict[str, torch.Tensor]:
        if not self.is_ready():
            raise RuntimeError("Observation history is not ready")
        state = self.normalizer.normalize_numpy(common.STATE_KEY, np.stack(self.state_history))
        images = np.stack(self.image_history)
        images_t = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float().div_(255.0)
        return {
            common.STATE_KEY: torch.from_numpy(state).unsqueeze(0).to(self.device),
            "images": images_t.unsqueeze(0).to(self.device),
            "phase": torch.tensor([[self.current_phase]], dtype=torch.float32, device=self.device),
        }

    @torch.inference_mode()
    def encode_observation(self) -> torch.Tensor:
        cond_tokens, _ = self.model.encode_observation(self.build_model_obs())
        return self.model.dsrl_observation(cond_tokens)

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        initial_noise: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        noise = None
        if initial_noise is not None:
            noise = torch.as_tensor(initial_noise, dtype=torch.float32, device=self.device)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
        result = self.model.predict_action(
            self.build_model_obs(),
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            initial_noise=noise,
        )
        normalized = result[common.ACTION_KEY][0].cpu().numpy()
        if self.use_visual_xy_override:
            visual_xy = result["visual_xy"][0].cpu().numpy()
            should_lock = (
                self.visual_xy_lock_phase is not None
                and self.current_phase >= self.visual_xy_lock_phase
            )
            if should_lock and self.locked_visual_xy is None:
                self.locked_visual_xy = visual_xy.copy()
            selected = self.locked_visual_xy if self.locked_visual_xy is not None else visual_xy
            normalized = normalized.copy()
            normalized[:, :2] = selected
        return self.normalizer.unnormalize_numpy(common.ACTION_KEY, normalized)

