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
from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.dinoactrtc.configuration_dinoactrtc import DINOACTRTCConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def _get_rot6d_indices_from_pose9(last_dim: int) -> list[int]:
    """Return flattened rot6d indices for repeated [pos(3), rot6d(6)] groups."""
    main_dim = last_dim - (last_dim % 9)
    indices: list[int] = []
    for group_start in range(0, main_dim, 9):
        indices.extend(range(group_start + 3, group_start + 9))
    return indices


class DINOACTRTCRot6DNormalizerProcessorStep(NormalizerProcessorStep):
    """Normalize DINOACTRTC tensors while keeping rot6d pose components untouched."""

    _registry_name = None

    def _apply_transform(
        self, tensor: torch.Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> torch.Tensor:
        transformed = super()._apply_transform(tensor, key, feature_type, inverse=inverse)
        if (
            feature_type not in (FeatureType.STATE, FeatureType.ACTION)
            or self.norm_map.get(feature_type, NormalizationMode.IDENTITY) != NormalizationMode.MEAN_STD
            or tensor.ndim == 0
        ):
            return transformed

        rot6d_indices = _get_rot6d_indices_from_pose9(tensor.shape[-1])
        if not rot6d_indices:
            return transformed

        transformed = transformed.clone()
        transformed[..., rot6d_indices] = tensor[..., rot6d_indices].to(
            device=transformed.device, dtype=transformed.dtype
        )
        return transformed


class DINOACTRTCRot6DUnnormalizerProcessorStep(UnnormalizerProcessorStep):
    """Unnormalize DINOACTRTC tensors while keeping rot6d pose components untouched."""

    _registry_name = None

    def _apply_transform(
        self, tensor: torch.Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> torch.Tensor:
        transformed = super()._apply_transform(tensor, key, feature_type, inverse=inverse)
        if (
            feature_type not in (FeatureType.STATE, FeatureType.ACTION)
            or self.norm_map.get(feature_type, NormalizationMode.IDENTITY) != NormalizationMode.MEAN_STD
            or tensor.ndim == 0
        ):
            return transformed

        rot6d_indices = _get_rot6d_indices_from_pose9(tensor.shape[-1])
        if not rot6d_indices:
            return transformed

        transformed = transformed.clone()
        transformed[..., rot6d_indices] = tensor[..., rot6d_indices].to(
            device=transformed.device, dtype=transformed.dtype
        )
        return transformed


def make_dinoactrtc_pre_post_processors(
    config: DINOACTRTCConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the DINOACTRTC policy.

    Image normalization follows `config.normalization_mapping` and dataset stats (same pipeline as ACT).
    For DINOv3 pretrained weights, configure visual stats to match the checkpoint (typically ImageNet mean/std).
    """

    normalizer_cls = (
        NormalizerProcessorStep
        if config.standardize_rot6d
        else DINOACTRTCRot6DNormalizerProcessorStep
    )
    unnormalizer_cls = (
        UnnormalizerProcessorStep
        if config.standardize_rot6d
        else DINOACTRTCRot6DUnnormalizerProcessorStep
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        normalizer_cls(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        unnormalizer_cls(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
