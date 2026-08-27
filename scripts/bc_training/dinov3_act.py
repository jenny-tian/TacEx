#!/usr/bin/env python
"""Deterministic DINOv3 action-chunking policy used by TacEx LabPick BC.

The existing BC implementation in :mod:`train_bc` is flow-matching based.  It
is useful for multi-modal data, but the scripted yaw-zero LabPick data are
essentially unimodal.  This module therefore uses an ACT-style deterministic
decoder, an explicit trajectory-progress token, and masked action chunks.  The
frozen DINOv3 encoder can consume either RGB images or precomputed spatial
features during training.
"""

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


MODEL_TYPE = "tacex_dinov3_act_bc_v2"


@dataclass
class DINOv3ACTConfig:
    state_dim: int = 10
    action_dim: int = 10
    num_cameras: int = 2
    state_obs_steps: int = 2
    image_obs_steps: int = 2
    chunk_size: int = 32
    feature_grid_size: int = 7
    condition_grid_size: int = 4
    dino_hidden_size: int = 384
    model_dim: int = 384
    encoder_layers: int = 4
    decoder_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    phase_horizon_steps: int = 384
    dino_path: str = ""
    image_keys: tuple[str, ...] = ("rgb", "rgb_third")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DINOv3ACTConfig":
        values = dict(data)
        if isinstance(values.get("image_keys"), list):
            values["image_keys"] = tuple(values["image_keys"])
        return cls(**values)


class DINOv3ACTPolicy(nn.Module):
    """DINOv3-conditioned deterministic action-chunk transformer."""

    def __init__(self, config: DINOv3ACTConfig, dino: common.MinimalDINOv3ViT) -> None:
        super().__init__()
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if config.feature_grid_size < 1:
            raise ValueError("feature_grid_size must be positive")
        if not 1 <= config.condition_grid_size <= config.feature_grid_size:
            raise ValueError("condition_grid_size must be in [1, feature_grid_size]")
        self.config = config
        self.dino = dino
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)
        self.dino.eval()

        dim = config.model_dim
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
        # The third-person camera has a fixed extrinsic pose, so flattening its
        # ordered spatial tokens gives a direct, non-privileged visual
        # localization path.  Keeping this head state-free prevents the model
        # from explaining late-trajectory XY using robot proprioception while
        # failing to localize the object in the initial frame.
        visual_input_dim = config.feature_grid_size**2 * config.dino_hidden_size
        self.visual_xy_head = nn.Sequential(
            nn.LayerNorm(visual_input_dim),
            nn.Linear(visual_input_dim, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, 2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.num_heads,
            dim_feedforward=4 * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.num_heads,
            dim_feedforward=4 * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(dim),
        )
        self.action_queries = nn.Parameter(torch.randn(1, config.chunk_size, dim) * 0.02)
        self.action_step_embedding = nn.Parameter(torch.randn(1, config.chunk_size, dim) * 0.02)
        self.action_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.action_dim))

        mean = torch.tensor(common.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(common.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        # Cartesian position and gripper width drive task success.  The six
        # rotation values are retained for interface compatibility but receive
        # less weight because yaw is deliberately fixed in this experiment.
        action_weights = torch.tensor([3.0, 3.0, 2.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 2.0])
        if config.action_dim != len(action_weights):
            action_weights = torch.ones(config.action_dim)
        self.register_buffer("action_weights", action_weights.view(1, 1, -1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    @property
    def num_spatial_tokens(self) -> int:
        return self.config.feature_grid_size**2

    @property
    def num_condition_tokens(self) -> int:
        return self.config.condition_grid_size**2

    def spatial_pool(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        num_patches = patch_tokens.shape[1]
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise ValueError(f"DINO patch token count {num_patches} is not a square grid")
        features = patch_tokens.transpose(1, 2).reshape(-1, patch_tokens.shape[-1], side, side)
        features = F.adaptive_avg_pool2d(
            features, output_size=(self.config.feature_grid_size, self.config.feature_grid_size)
        )
        return features.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def extract_dino_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return BxTxCxNxD spatial DINO features for normalized [0, 1] RGB."""
        batch_size, image_steps, num_cameras, channels, height, width = images.shape
        flat = images.reshape(batch_size * image_steps * num_cameras, channels, height, width)
        flat = (flat - self.image_mean.to(dtype=flat.dtype)) / self.image_std.to(dtype=flat.dtype)
        patches = self.dino(flat)
        pooled = self.spatial_pool(patches)
        return pooled.reshape(
            batch_size,
            image_steps,
            num_cameras,
            self.num_spatial_tokens,
            self.config.dino_hidden_size,
        )

    @staticmethod
    def _phase_features(phase: torch.Tensor) -> torch.Tensor:
        phase = phase.reshape(-1, 1).clamp(0.0, 1.0)
        return torch.cat(
            (
                phase,
                torch.sin(math.pi * phase),
                torch.cos(math.pi * phase),
                torch.sin(2.0 * math.pi * phase),
                torch.cos(2.0 * math.pi * phase),
                torch.sin(4.0 * math.pi * phase),
                torch.cos(4.0 * math.pi * phase),
            ),
            dim=-1,
        )

    def _condition_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.condition_grid_size == self.config.feature_grid_size:
            return features
        batch_size, image_steps, num_cameras, _, hidden = features.shape
        grid = self.config.feature_grid_size
        condition_grid = self.config.condition_grid_size
        flat = features.reshape(-1, grid, grid, hidden).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(flat, output_size=(condition_grid, condition_grid))
        return pooled.permute(0, 2, 3, 1).reshape(
            batch_size, image_steps, num_cameras, condition_grid**2, hidden
        )

    def encode_observation(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state = obs[common.STATE_KEY]
        if "dino_features" in obs:
            dino_features = obs["dino_features"].float()
        elif "images" in obs:
            dino_features = self.extract_dino_features(obs["images"])
        else:
            raise KeyError("Observation must contain 'images' or 'dino_features'")

        batch_size, image_steps, num_cameras, num_tokens, hidden = dino_features.shape
        expected = (
            self.config.image_obs_steps,
            self.config.num_cameras,
            self.num_spatial_tokens,
            self.config.dino_hidden_size,
        )
        if (image_steps, num_cameras, num_tokens, hidden) != expected:
            raise ValueError(
                "DINO feature shape mismatch: "
                f"got {tuple(dino_features.shape)}, expected (B, {', '.join(map(str, expected))})"
            )

        state_tokens = self.state_projector(state)
        state_tokens = state_tokens + self.state_step_embedding[:, : state.shape[1]]
        state_tokens = state_tokens + self.modality_embedding[:, 0:1]

        # The latest third-person frame is always the final camera in the
        # canonical rgb,rgb_third ordering used by this experiment.
        visual_features = dino_features[:, -1, -1].reshape(batch_size, -1)
        visual_xy = self.visual_xy_head(visual_features)

        condition_features = self._condition_features(dino_features)
        condition_tokens = condition_features.shape[3]
        vision_tokens = self.vision_projector(condition_features)
        vision_tokens = (
            vision_tokens
            + self.image_step_embedding[:, :image_steps]
            + self.camera_embedding[:, :, :num_cameras]
            + self.spatial_embedding[:, :, :, :condition_tokens]
            + self.modality_embedding[:, 1].view(1, 1, 1, 1, -1)
        )
        vision_tokens = vision_tokens.reshape(batch_size, -1, self.config.model_dim)

        phase = obs["phase"].to(dtype=state.dtype)
        phase_token = self.phase_projector(self._phase_features(phase)).unsqueeze(1)
        phase_token = phase_token + self.modality_embedding[:, 2:3]
        memory = self.encoder(torch.cat((phase_token, state_tokens, vision_tokens), dim=1))
        return memory, visual_xy

    def _forward_with_visual_xy(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        memory, visual_xy = self.encode_observation(obs)
        queries = self.action_queries + self.action_step_embedding
        queries = queries.expand(memory.shape[0], -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        raw_action = self.action_head(decoded)
        action = torch.cat(
            (visual_xy.unsqueeze(1).expand(-1, self.config.chunk_size, -1), raw_action[..., 2:]), dim=-1
        )
        return action, visual_xy

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward_with_visual_xy(obs)[0]

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        target = batch[common.ACTION_KEY]
        prediction, visual_xy = self._forward_with_visual_xy(batch["obs"])
        valid = batch["action_is_pad"].logical_not().to(dtype=prediction.dtype).unsqueeze(-1)

        per_element = F.smooth_l1_loss(prediction, target, beta=0.05, reduction="none")
        time = torch.arange(self.config.chunk_size, device=prediction.device, dtype=prediction.dtype)
        time_weights = torch.pow(torch.tensor(0.98, device=prediction.device), time).view(1, -1, 1)
        weights = valid * time_weights * self.action_weights.to(dtype=prediction.dtype)
        chunk_loss = (per_element * weights).sum() / weights.sum().clamp_min(1.0)
        first_error = F.smooth_l1_loss(prediction[:, 0], target[:, 0], beta=0.05, reduction="none")
        first_weights = self.action_weights[:, 0].to(dtype=prediction.dtype)
        first_loss = (first_error * first_weights).sum() / (first_weights.sum() * prediction.shape[0])
        visual_xy_loss = F.smooth_l1_loss(visual_xy, target[:, 0, :2], beta=0.05)
        loss = chunk_loss + first_loss + 2.0 * visual_xy_loss
        return {
            "loss": loss,
            "chunk_loss": chunk_loss,
            "first_action_loss": first_loss,
            "visual_xy_loss": visual_xy_loss,
        }

    @torch.inference_mode()
    def predict_action(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        action = self(obs).clamp(-1.0, 1.0)
        return {common.ACTION_KEY: action, "action_pred": action}


def policy_state_dict(model: DINOv3ACTPolicy) -> dict[str, torch.Tensor]:
    """Save only learned policy weights; DINOv3 is reloaded from its snapshot."""
    return {key: value for key, value in model.state_dict().items() if not key.startswith("dino.")}


def load_policy_state_dict(model: DINOv3ACTPolicy, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_dino_missing = [key for key in missing if not key.startswith("dino.")]
    if non_dino_missing or unexpected:
        raise RuntimeError(f"Policy state mismatch: missing={non_dino_missing}, unexpected={unexpected}")


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> tuple[DINOv3ACTPolicy, common.LinearNormalizer, dict[str, Any]]:
    target_device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    checkpoint = common.load_checkpoint(checkpoint_path, map_location=target_device)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(
            f"Unsupported checkpoint model_type={checkpoint.get('model_type')!r}; expected {MODEL_TYPE!r}"
        )
    config = DINOv3ACTConfig.from_dict(checkpoint["policy_config"])
    dino, _ = common.load_dino_for_config(Path(config.dino_path) if config.dino_path else None)
    model = DINOv3ACTPolicy(config, dino).to(target_device)
    load_policy_state_dict(model, checkpoint["model"])
    model.eval()
    normalizer = common.LinearNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])
    return model, normalizer, checkpoint


class DINOv3ACTRunner:
    """Stateful image/state history adapter for simulation deployment."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda") -> None:
        self.model, self.normalizer, self.checkpoint = load_policy(checkpoint_path, device=device)
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        self.state_history: deque[np.ndarray] = deque(maxlen=self.config.state_obs_steps)
        self.image_history: deque[np.ndarray] = deque(maxlen=self.config.image_obs_steps)
        self.phase = 0.0

    def reset(self) -> None:
        self.state_history.clear()
        self.image_history.clear()
        self.phase = 0.0

    def update(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray,
        phase: float,
    ) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.config.state_dim,):
            raise ValueError(f"state must have shape ({self.config.state_dim},), got {state.shape}")
        if isinstance(images, dict):
            cameras = [common.normalize_image_array(images[key], key) for key in self.config.image_keys]
        elif isinstance(images, list):
            cameras = [common.normalize_image_array(image, f"image_{i}") for i, image in enumerate(images)]
        else:
            array = np.asarray(images)
            cameras = (
                [common.normalize_image_array(array[i], f"image_{i}") for i in range(array.shape[0])]
                if array.ndim == 4
                else [common.normalize_image_array(array, "image")]
            )
        if len(cameras) != self.config.num_cameras:
            raise ValueError(f"Expected {self.config.num_cameras} cameras, got {len(cameras)}")
        camera_stack = np.stack(cameras, axis=0)

        if not self.state_history:
            for _ in range(self.config.state_obs_steps):
                self.state_history.append(state.copy())
            for _ in range(self.config.image_obs_steps):
                self.image_history.append(camera_stack.copy())
        else:
            self.state_history.append(state.copy())
            self.image_history.append(camera_stack.copy())
        self.phase = float(np.clip(phase, 0.0, 1.0))

    def build_model_obs(self) -> dict[str, torch.Tensor]:
        if len(self.state_history) != self.config.state_obs_steps:
            raise RuntimeError("Observation history is not ready; call update() first")
        state = np.stack(self.state_history).astype(np.float32)
        state = self.normalizer.normalize_numpy(common.STATE_KEY, state)
        images = np.stack(self.image_history)
        images_t = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float().div_(255.0)
        return {
            common.STATE_KEY: torch.from_numpy(state).unsqueeze(0).to(self.device),
            "images": images_t.unsqueeze(0).to(self.device),
            "phase": torch.tensor([[self.phase]], dtype=torch.float32, device=self.device),
        }

    @torch.inference_mode()
    def predict_action_chunk(self) -> np.ndarray:
        action = self.model.predict_action(self.build_model_obs())[common.ACTION_KEY]
        return self.normalizer.unnormalize_numpy(common.ACTION_KEY, action[0].cpu().numpy())
