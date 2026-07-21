#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("dinoact")
@dataclass
class DINOACTConfig(ACTConfig):
    """ACT with a DINOv3 ViT backbone (Hugging Face `transformers`), instead of torchvision ResNet.

    Install: `pip install 'lerobot[transformers-dep]'`.

    Configure image normalization in the dataset / policy stats to match the DINOv3 checkpoint (typically
    ImageNet mean and std); the model does not apply an extra ImageNet normalization inside the backbone.
    """

    dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dinov3_attn_implementation: str = "sdpa"
    # If True, resize inputs to `dinov3_image_size` (square) before the ViT (bicubic).
    dinov3_interpolate_images: bool = True
    # Target H=W when interpolating; if None, uses `image_size` from the HF model config when available,
    # otherwise 224. Must be divisible by the model `patch_size` when interpolating.
    dinov3_image_size: int | None = 224
    # If True, split action dimensions into non-rot and rot6d groups and use:
    #   recon_loss = l1_nonrot + lambda_rot * chordal_rot
    # where chordal_rot = ||R_pred - R_gt||_F after converting rot6d to rotation matrices.
    use_rot6d_chordal_loss: bool = False
    # If True, unnormalize rot6d dims back to dataset space before rot6d->matrix and Chordal loss.
    # This preserves rotation geometry even when ACTION normalization is enabled.
    rot6d_chordal_use_unnormalized: bool = True
    lambda_rot: float = 1.0

    def __post_init__(self):
        # Skip ACTConfig.__post_init__ (ResNet-only `vision_backbone` check); keep PreTrainedConfig setup.
        PreTrainedConfig.__post_init__(self)

        if not self.dinov3_model_id or not str(self.dinov3_model_id).strip():
            raise ValueError("`dinov3_model_id` must be a non-empty Hugging Face model id or path.")

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        if self.dinov3_interpolate_images and self.dinov3_image_size is not None and self.dinov3_image_size < 1:
            raise ValueError("`dinov3_image_size` must be positive when set.")
        if self.lambda_rot < 0:
            raise ValueError("`lambda_rot` must be non-negative.")
