from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class VisualPoseProbe(nn.Module):
    """Small probe mapping frozen BC visual features to relative object pose.

    Output is normalized relative position (3), relative rotation 6D (6),
    and a confidence score (1).  It is intentionally separate from the BC
    action heads so DSRL receives an interpretable visual state without
    changing or retraining the BC policy.
    """

    def __init__(self, input_dim: int = 522, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = 10
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.net(features)
        # Confidence is a bounded probability; pose values remain normalized
        # so the probe has a stable scale in the SAC observation.
        return torch.cat((output[..., :9], torch.sigmoid(output[..., 9:])), dim=-1)

    @classmethod
    def load(cls, checkpoint: str | Path, *, device: str | torch.device = "cuda") -> "VisualPoseProbe":
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = cls(input_dim=int(payload["input_dim"]), hidden_dim=int(payload.get("hidden_dim", 256)))
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        return model
