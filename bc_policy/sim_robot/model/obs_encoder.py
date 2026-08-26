from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = max(1, num_channels // 16)
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResNet50ImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1024,
        use_group_norm: bool = True,
        image_normalization: str = "none",
    ) -> None:
        super().__init__()
        if image_normalization not in {"none", "imagenet"}:
            raise ValueError("image_normalization must be 'none' or 'imagenet'")
        self.image_normalization = image_normalization
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
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
        if self.image_normalization == "imagenet":
            image = (image - self.image_mean) / self.image_std
        return self.projector(self.backbone(image))


class SingleCameraObsEncoder(nn.Module):
    """Encode robot state and one or more RGB camera streams as condition tokens."""

    def __init__(
        self,
        robot0_pos_dim: int,
        n_state_obs_steps: int,
        n_image_obs_steps: int,
        image_feature_dim: int = 1024,
        step_feature_dim: int = 1024,
        dropout: float = 0.0,
        image_keys: tuple[str, ...] = ("robot0_image",),
        image_normalization: str = "none",
    ) -> None:
        super().__init__()
        self.robot0_pos_dim = robot0_pos_dim
        self.n_state_obs_steps = int(n_state_obs_steps)
        self.n_image_obs_steps = int(n_image_obs_steps)
        self.step_feature_dim = int(step_feature_dim)
        self.image_keys = tuple(image_keys)
        if not self.image_keys:
            raise ValueError("At least one image key is required.")

        self.state_projector = nn.Sequential(
            nn.Linear(robot0_pos_dim, step_feature_dim),
            nn.LayerNorm(step_feature_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(step_feature_dim, step_feature_dim),
            nn.LayerNorm(step_feature_dim),
            nn.Mish(),
        )
        # Share the expensive visual backbone across cameras, while retaining
        # camera-specific projection heads for wrist/third-person domains.
        self.image_encoder = ResNet50ImageEncoder(
            output_dim=image_feature_dim,
            image_normalization=image_normalization,
        )
        if len(self.image_keys) == 1:
            self.image_projector = nn.Sequential(
                nn.Linear(image_feature_dim, step_feature_dim),
                nn.LayerNorm(step_feature_dim),
                nn.Mish(),
            )
        else:
            self.image_projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(image_feature_dim, step_feature_dim),
                        nn.LayerNorm(step_feature_dim),
                        nn.Mish(),
                    )
                    for _ in self.image_keys
                ]
            )

        self.modality_emb = nn.Parameter(torch.zeros(1, 1 + len(self.image_keys), step_feature_dim))
        nn.init.normal_(self.modality_emb, mean=0.0, std=0.02)

        self.total_cond_tokens = self.n_state_obs_steps + self.n_image_obs_steps * len(self.image_keys)
        self.global_cond_dim = step_feature_dim * self.total_cond_tokens

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        robot0_pos = obs["robot0_pos"]
        batch_size = robot0_pos.shape[0]

        state_tokens = self.state_projector(robot0_pos) + self.modality_emb[:, 0:1]
        image_tokens_per_camera = []
        for camera_index, image_key in enumerate(self.image_keys):
            image = obs[image_key]
            _, image_steps, image_c, image_h, image_w = image.shape
            image_feat = self.image_encoder(
                image.reshape(batch_size * image_steps, image_c, image_h, image_w)
            )
            image_feat = image_feat.reshape(batch_size, image_steps, -1)
            projector = self.image_projector if len(self.image_keys) == 1 else self.image_projectors[camera_index]
            image_tokens_per_camera.append(
                projector(image_feat) + self.modality_emb[:, camera_index + 1 : camera_index + 2]
            )

        cond_tokens = torch.cat([state_tokens, *image_tokens_per_camera], dim=1)
        global_cond = cond_tokens.reshape(batch_size, -1)
        return cond_tokens, global_cond

