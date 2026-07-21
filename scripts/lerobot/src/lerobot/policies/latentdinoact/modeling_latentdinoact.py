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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.latentdinoact.configuration_latentdinoact import LATENTDINOACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class DINOv3ACTBackbone(nn.Module):
    """DINOv3 vision encoder returning a `(B, C, H, W)` feature map compatible with ACT."""

    def __init__(
        self,
        model_id: str,
        attn_implementation: str,
        interpolate_images: bool,
        image_size: int | None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "LATENTDINOACT requires the `transformers` package. Install with: pip install 'lerobot[transformers-dep]'"
            ) from e

        self.model = AutoModel.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
        )
        self.patch_size = int(self.model.config.patch_size)
        self.embed_dim = int(self.model.config.hidden_size)
        self.interpolate_images = interpolate_images

        if interpolate_images:
            if image_size is not None:
                if image_size % self.patch_size != 0:
                    raise ValueError(
                        f"`dinov3_image_size` ({image_size}) must be divisible by patch_size ({self.patch_size})."
                    )
                self._target_hw = (image_size, image_size)
            else:
                cfg_sz = getattr(self.model.config, "image_size", None)
                if cfg_sz is None:
                    default_sz = 224
                    self._target_hw = (default_sz, default_sz)
                elif isinstance(cfg_sz, int):
                    if cfg_sz % self.patch_size != 0:
                        raise ValueError(
                            f"Model image_size ({cfg_sz}) must be divisible by patch_size ({self.patch_size})."
                        )
                    self._target_hw = (cfg_sz, cfg_sz)
                else:
                    h, w = int(cfg_sz[0]), int(cfg_sz[1])
                    self._target_hw = (h, w)
            if self._target_hw[0] % self.patch_size != 0 or self._target_hw[1] % self.patch_size != 0:
                raise ValueError(
                    f"Target spatial size {self._target_hw} must be divisible by patch_size {self.patch_size}."
                )
        else:
            self._target_hw = None

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor]:
        x = pixel_values
        _, _, h, w = x.shape

        if self.interpolate_images:
            assert self._target_hw is not None
            th, tw = self._target_hw
            if h != th or w != tw:
                x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)
                h, w = th, tw
        elif h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Image spatial dims {(h, w)} must be divisible by patch_size={self.patch_size}, "
                "or enable `dinov3_interpolate_images`."
            )

        gh, gw = h // self.patch_size, w // self.patch_size
        outputs = self.model(pixel_values=x)
        last_hidden = outputs.last_hidden_state

        num_registers = int(getattr(self.model.config, "num_register_tokens", 0))
        patch_tokens = last_hidden[:, 1 + num_registers :, :]

        expected_n = gh * gw
        if patch_tokens.shape[1] != expected_n:
            raise RuntimeError(
                f"DINOv3 produced {patch_tokens.shape[1]} patch tokens but expected {expected_n} for grid {(gh, gw)}."
            )

        b = x.shape[0]
        feat = patch_tokens.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        return {"feature_map": feat}


class LATENTDINOACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = LATENTDINOACTConfig
    name = "latentdinoact"

    def __init__(
        self,
        config: LATENTDINOACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._dataset_stats = kwargs.get("dataset_stats")

        self.model = LATENTDINOACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    @staticmethod
    def _rotation_6d_to_matrix(d6: Tensor) -> Tensor:
        """Convert 6D rotation representation to rotation matrices."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    @staticmethod
    def _split_action_for_rot6d_loss(action: Tensor) -> tuple[Tensor, Tensor]:
        """Split action into non-rotation dims and rot6d groups.

        Supports layouts:
        - N groups of [pos(3), rot6d(6)]
        - N groups of [pos(3), rot6d(6)] + optional tail non-rotation dims
        """
        action_dim = action.shape[-1]
        main_dim = action_dim - (action_dim % 9)
        if main_dim == 0:
            raise ValueError(
                f"Action dimension {action_dim} is incompatible with [pos(3)+rot6d(6)] groups. "
                "Expected at least one full 9D [pos(3)+rot6d(6)] group."
            )

        num_groups = main_dim // 9
        leading_shape = action.shape[:-1]
        main = action[..., :main_dim].reshape(*leading_shape, num_groups, 9)

        nonrot_main = main[..., :3].reshape(*leading_shape, num_groups * 3)
        if main_dim < action_dim:
            nonrot = torch.cat([nonrot_main, action[..., main_dim:]], dim=-1)
        else:
            nonrot = nonrot_main

        rot6d = main[..., 3:9]
        return nonrot, rot6d

    @staticmethod
    def _get_rot6d_flat_indices(action_dim: int) -> list[int]:
        """Get flattened indices of rot6d components in [pos(3), rot6d(6)] groups."""
        main_dim = action_dim - (action_dim % 9)
        if main_dim == 0:
            raise ValueError(
                f"Action dimension {action_dim} is incompatible with [pos(3)+rot6d(6)] groups. "
                "Expected at least one full 9D [pos(3)+rot6d(6)] group."
            )

        indices: list[int] = []
        num_groups = main_dim // 9
        for group_idx in range(num_groups):
            start = group_idx * 9 + 3
            indices.extend(range(start, start + 6))
        return indices

    def _get_action_stats(self) -> dict[str, Tensor] | None:
        if not isinstance(self._dataset_stats, dict):
            return None
        action_key = ACTION if ACTION in self._dataset_stats else next(iter(self.config.output_features.keys()), None)
        if action_key is None:
            return None
        stats = self._dataset_stats.get(action_key)
        return stats if isinstance(stats, dict) else None

    def _unnormalize_action_dims(self, action: Tensor, indices: list[int]) -> Tensor:
        """Unnormalize selected action dims according to ACTION normalization mapping."""
        action_stats = self._get_action_stats()
        norm_mode = self.config.normalization_mapping.get("ACTION", NormalizationMode.IDENTITY)
        if norm_mode != NormalizationMode.IDENTITY and action_stats is None:
            raise ValueError(
                "ACTION normalization is enabled but dataset stats are missing; "
                "cannot unnormalize rot6d dims for Chordal loss."
            )
        if norm_mode == NormalizationMode.IDENTITY:
            return action[..., indices]

        idx = torch.tensor(indices, device=action.device, dtype=torch.long)
        selected = action.index_select(dim=-1, index=idx)

        def _select_stat(name: str) -> Tensor | None:
            stat = action_stats.get(name)
            if stat is None:
                return None
            if isinstance(stat, Tensor):
                stat_t = stat.to(device=action.device, dtype=action.dtype)
            else:
                stat_t = torch.as_tensor(stat, device=action.device, dtype=action.dtype)
            return stat_t.index_select(dim=-1, index=idx)

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = _select_stat("mean")
            std = _select_stat("std")
            if mean is None or std is None:
                raise ValueError(
                    "Missing ACTION stats for MEAN_STD normalization. Required keys: 'mean' and 'std'."
                )
            return selected * std + mean

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = _select_stat("min")
            max_val = _select_stat("max")
            if min_val is None or max_val is None:
                raise ValueError(
                    "Missing ACTION stats for MIN_MAX normalization. Required keys: 'min' and 'max'."
                )
            return (selected + 1) * (max_val - min_val) / 2 + min_val

        if norm_mode == NormalizationMode.QUANTILES:
            q01 = _select_stat("q01")
            q99 = _select_stat("q99")
            if q01 is None or q99 is None:
                raise ValueError(
                    "Missing ACTION stats for QUANTILES normalization. Required keys: 'q01' and 'q99'."
                )
            return (selected + 1) * (q99 - q01) / 2 + q01

        if norm_mode == NormalizationMode.QUANTILE10:
            q10 = _select_stat("q10")
            q90 = _select_stat("q90")
            if q10 is None or q90 is None:
                raise ValueError(
                    "Missing ACTION stats for QUANTILE10 normalization. Required keys: 'q10' and 'q90'."
                )
            return (selected + 1) * (q90 - q10) / 2 + q10

        return selected

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'none'.")

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat), rtc_postfix_loss_mask = self.model(batch)

        action_mask = (~batch["action_is_pad"]).to(dtype=actions_hat.dtype).unsqueeze(-1)
        if rtc_postfix_loss_mask is not None:
            action_mask = action_mask * rtc_postfix_loss_mask.to(dtype=actions_hat.dtype).unsqueeze(-1)
        loss_dict: dict[str, float] = {}

        if self.config.use_rot6d_chordal_loss:
            target_nonrot, target_rot6d = self._split_action_for_rot6d_loss(batch[ACTION])
            pred_nonrot, pred_rot6d = self._split_action_for_rot6d_loss(actions_hat)

            nonrot_l1 = F.l1_loss(target_nonrot, pred_nonrot, reduction="none") * action_mask
            nonrot_l1_per_sample = nonrot_l1.mean(dim=(1, 2))
            nonrot_l1_loss = nonrot_l1_per_sample.mean()

            if self.config.rot6d_chordal_use_unnormalized:
                rot_indices = self._get_rot6d_flat_indices(actions_hat.shape[-1])
                target_rot6d = self._unnormalize_action_dims(batch[ACTION], rot_indices).reshape(
                    *batch[ACTION].shape[:-1], -1, 6
                )
                pred_rot6d = self._unnormalize_action_dims(actions_hat, rot_indices).reshape(
                    *actions_hat.shape[:-1], -1, 6
                )

            target_rot_m = self._rotation_6d_to_matrix(target_rot6d)
            pred_rot_m = self._rotation_6d_to_matrix(pred_rot6d)
            rot_chordal = torch.linalg.norm(pred_rot_m - target_rot_m, ord="fro", dim=(-2, -1))
            rot_chordal = rot_chordal * action_mask.squeeze(-1).unsqueeze(-1)
            rot_chordal_per_sample = rot_chordal.mean(dim=(1, 2))
            rot_chordal_loss = rot_chordal_per_sample.mean()

            recon_per_sample = nonrot_l1_per_sample + self.config.lambda_rot * rot_chordal_per_sample
            recon_loss = recon_per_sample.mean()
            loss_dict["l1_nonrot_loss"] = nonrot_l1_loss.item()
            loss_dict["chordal_rot_loss"] = rot_chordal_loss.item()
            loss_dict["recon_loss"] = recon_loss.item()
        else:
            l1_loss = F.l1_loss(batch[ACTION], actions_hat, reduction="none") * action_mask
            recon_per_sample = l1_loss.mean(dim=(1, 2))
            recon_loss = recon_per_sample.mean()
            loss_dict["l1_loss"] = recon_loss.item()

        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            kld_per_sample = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1)
            mean_kld = kld_per_sample.mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss_per_sample = recon_per_sample + kld_per_sample * self.config.kl_weight
        else:
            loss_per_sample = recon_per_sample

        loss = loss_per_sample.mean()
        loss_dict["loss"] = loss.item()

        if reduction == "none":
            return loss_per_sample, loss_dict
        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class LATENTDINOACT(nn.Module):
    """Action Chunking Transformer with DINOv3 backbone: underlying network for LATENTDINOACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: LATENTDINOACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction (DINOv3 ViT → spatial feature map).
        if self.config.image_features:
            self.backbone = DINOv3ACTBackbone(
                model_id=config.dinov3_model_id,
                attn_implementation=config.dinov3_attn_implementation,
                interpolate_images=config.dinov3_interpolate_images,
                image_size=config.dinov3_image_size,
            )
            backbone_out_channels = self.backbone.embed_dim

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_out_channels, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Training-time RTC: project ground-truth action prefix tokens into the encoder feature space.
        # See https://arxiv.org/abs/2512.05964 for the underlying idea (adapted to ACT, which is
        # non-diffusion, by simply feeding the prefix as additional encoder tokens and masking the
        # loss on the corresponding postfix positions).
        if self.config.training_time_rtc:
            max_prefix = min(self.config.training_time_rtc_maxnum_actions, self.config.chunk_size - 1)
            if max_prefix <= 0:
                raise ValueError(
                    "`training_time_rtc_maxnum_actions` must be positive and strictly less than `chunk_size`."
                )
            self._rtc_max_prefix = max_prefix
            self.encoder_prefix_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _obs_images_list(self, batch: dict[str, Tensor]) -> list[Tensor]:
        """Match `LATENTDINOACTPolicy.predict_action_chunk`: list under `OBS_IMAGES` or per-feature keys."""
        if OBS_IMAGES in batch:
            return batch[OBS_IMAGES]
        if self.config.image_features:
            return [batch[name] for name in self.config.image_features]
        return []

    def _batch_ref_tensor(self, batch: dict[str, Tensor]) -> Tensor:
        """Return a batch tensor used to infer batch size and device."""
        if OBS_IMAGES in batch:
            return batch[OBS_IMAGES][0]
        if OBS_ENV_STATE in batch:
            return batch[OBS_ENV_STATE]
        if self.config.robot_state_feature and OBS_STATE in batch:
            return batch[OBS_STATE]
        if self.config.image_features:
            first = next(iter(self.config.image_features))
            return batch[first]
        raise KeyError(
            "Cannot infer batch size: expected OBS_IMAGES, observation.environment_state, "
            "observation.state, or configured image keys in batch."
        )

    def _build_encoder_inputs(
        self,
        batch: dict[str, Tensor],
        latent_sample: Tensor,
        *,
        include_rtc: bool,
    ) -> tuple[list[Tensor], list[Tensor], Tensor | None, Tensor | None]:
        """Build encoder token lists, optional RTC padding mask, and optional postfix loss mask."""
        ref = self._batch_ref_tensor(batch)
        batch_size = ref.shape[0]

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in self._obs_images_list(batch):
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        rtc_postfix_loss_mask: Tensor | None = None
        encoder_key_padding_mask: Tensor | None = None
        prefix_actions: Tensor | None = None
        delay: Tensor | None = None
        max_d: int = 0

        if include_rtc and self.config.training_time_rtc:
            max_d = self._rtc_max_prefix
            if self.training and ACTION in batch:
                device = batch[ACTION].device
                delay = torch.randint(0, max_d + 1, (batch_size,), device=device)
                prefix_actions = batch[ACTION][:, :max_d]
            elif (not self.training) and "action_prefix" in batch:
                raw_prefix = batch["action_prefix"]
                device = raw_prefix.device
                if raw_prefix.shape[1] >= max_d:
                    prefix_actions = raw_prefix[:, :max_d].to(dtype=torch.float32)
                else:
                    pad = torch.zeros(
                        batch_size,
                        max_d - int(raw_prefix.shape[1]),
                        int(raw_prefix.shape[-1]),
                        device=device,
                        dtype=raw_prefix.dtype,
                    )
                    prefix_actions = torch.cat([raw_prefix, pad], dim=1).to(dtype=torch.float32)
                if "prefix_delay" in batch:
                    d_tensor = batch["prefix_delay"].to(device=device, dtype=torch.long)
                    if d_tensor.dim() == 0:
                        d_tensor = d_tensor.expand(batch_size)
                    delay = torch.clamp(d_tensor, min=0, max=max_d)
                else:
                    k_in = min(int(raw_prefix.shape[1]), max_d)
                    delay = torch.full((batch_size,), k_in, dtype=torch.long, device=device)

        if prefix_actions is not None:
            assert delay is not None
            n_non_prefix = len(encoder_in_tokens)
            prefix_embed = self.encoder_prefix_action_input_proj(prefix_actions)
            prefix_embed = prefix_embed.transpose(0, 1)
            prefix_pos = self.decoder_pos_embed.weight[:max_d].unsqueeze(1)

            encoder_in_tokens.extend(list(prefix_embed))
            encoder_in_pos_embed.extend(list(prefix_pos))

            positions = torch.arange(max_d, device=prefix_actions.device)
            prefix_validity = positions[None, :] < delay[:, None]
            prefix_padding = ~prefix_validity
            non_prefix_padding = torch.zeros(
                (batch_size, n_non_prefix), dtype=torch.bool, device=prefix_actions.device
            )
            encoder_key_padding_mask = torch.cat([non_prefix_padding, prefix_padding], dim=1)

            if self.training:
                chunk_positions = torch.arange(self.config.chunk_size, device=prefix_actions.device)
                rtc_postfix_loss_mask = chunk_positions[None, :] >= delay[:, None]

        return encoder_in_tokens, encoder_in_pos_embed, encoder_key_padding_mask, rtc_postfix_loss_mask

    def _encode_decode(
        self,
        batch: dict[str, Tensor],
        latent_sample: Tensor,
        *,
        include_rtc: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Run transformer encoder + decoder; returns (B, chunk_size, dim_model) and optional RTC loss mask."""
        ref = self._batch_ref_tensor(batch)
        batch_size = ref.shape[0]

        encoder_in_tokens, encoder_in_pos_embed, encoder_key_padding_mask, rtc_postfix_loss_mask = (
            self._build_encoder_inputs(batch, latent_sample, include_rtc=include_rtc)
        )

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(
            encoder_in_tokens,
            pos_embed=encoder_in_pos_embed,
            key_padding_mask=encoder_key_padding_mask,
        )
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        return decoder_out, rtc_postfix_loss_mask

    def inference_latent(self, batch: dict[str, Tensor]) -> Tensor:
        """Encoder + decoder on observations with zero VAE latent; returns decoder hidden states.

        Returns:
            (B, chunk_size, dim_model) tensor for RL latent pipelines or ``decode_from_latent``.
        """
        ref = self._batch_ref_tensor(batch)
        batch_size = ref.shape[0]
        latent_sample = torch.zeros(
            [batch_size, self.config.latent_dim],
            dtype=torch.float32,
            device=ref.device,
        )
        decoder_out, _ = self._encode_decode(
            batch,
            latent_sample,
            include_rtc=self.config.training_time_rtc,
        )

        action = self.action_head(decoder_out)

        return action

    def decode_from_latent(self, batch: dict[str, Tensor], latent: Tensor) -> Tensor:
        """Map decoder hidden states to actions via the action head.

        Args:
            latent: (B, chunk_size, dim_model) output from ``inference_latent`` or the transformer decoder.
        """
        return latent

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None], Tensor | None]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        Returns:
            (B, chunk_size, action_dim) batch of action sequences.
            Tuple containing the latent PDF's parameters (mean, log(σ²)).
            Optional (B, chunk_size) RTC postfix mask for the reconstruction loss.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            ref = self._batch_ref_tensor(batch)
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim],
                dtype=torch.float32,
                device=ref.device,
            )

        decoder_out, rtc_postfix_loss_mask = self._encode_decode(
            batch,
            latent_sample,
            include_rtc=self.config.training_time_rtc,
        )
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2), rtc_postfix_loss_mask

class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
