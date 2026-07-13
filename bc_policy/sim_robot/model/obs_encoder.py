from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = max(1, num_channels // 16)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResNet50ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 1024, use_group_norm: bool = True) -> None:
        super().__init__()
        norm_layer = _group_norm if use_group_norm else None
        self.backbone = resnet50(weights=None, norm_layer=norm_layer)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.Mish(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(image))


class SingleCameraObsEncoder(nn.Module):
    """Encode robot state and one RGB camera stream as condition tokens."""

    def __init__(
        self,
        robot0_pos_dim: int,
        n_state_obs_steps: int,
        n_image_obs_steps: int,
        image_feature_dim: int = 1024,
        step_feature_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.robot0_pos_dim = robot0_pos_dim
        self.n_state_obs_steps = int(n_state_obs_steps)
        self.n_image_obs_steps = int(n_image_obs_steps)
        self.step_feature_dim = int(step_feature_dim)

        self.state_projector = nn.Sequential(
            nn.Linear(robot0_pos_dim, step_feature_dim),
            nn.LayerNorm(step_feature_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(step_feature_dim, step_feature_dim),
            nn.LayerNorm(step_feature_dim),
            nn.Mish(),
        )
        self.image_encoder = ResNet50ImageEncoder(output_dim=image_feature_dim)
        self.image_projector = nn.Sequential(
            nn.Linear(image_feature_dim, step_feature_dim),
            nn.LayerNorm(step_feature_dim),
            nn.Mish(),
        )

        self.modality_emb = nn.Parameter(torch.zeros(1, 2, step_feature_dim))
        nn.init.normal_(self.modality_emb, mean=0.0, std=0.02)

        self.total_cond_tokens = self.n_state_obs_steps + self.n_image_obs_steps
        self.global_cond_dim = step_feature_dim * self.total_cond_tokens

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        robot0_pos = obs["robot0_pos"]
        image = obs["robot0_image"]
        batch_size = robot0_pos.shape[0]

        state_tokens = self.state_projector(robot0_pos) + self.modality_emb[:, 0:1]

        _, image_steps, image_c, image_h, image_w = image.shape
        image_feat = self.image_encoder(image.reshape(batch_size * image_steps, image_c, image_h, image_w))
        image_tokens = self.image_projector(image_feat.reshape(batch_size, image_steps, -1)) + self.modality_emb[:, 1:2]

        cond_tokens = torch.cat([state_tokens, image_tokens], dim=1)
        global_cond = cond_tokens.reshape(batch_size, -1)
        return cond_tokens, global_cond

