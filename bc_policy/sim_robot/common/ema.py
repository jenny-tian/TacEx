from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.averaged_model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.averaged_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        for averaged, current in zip(self.averaged_model.parameters(), model.parameters()):
            averaged.mul_(self.decay).add_(current.detach(), alpha=1.0 - self.decay)
        for averaged, current in zip(self.averaged_model.buffers(), model.buffers()):
            averaged.copy_(current)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "averaged_model": self.averaged_model.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.averaged_model.load_state_dict(state["averaged_model"])

