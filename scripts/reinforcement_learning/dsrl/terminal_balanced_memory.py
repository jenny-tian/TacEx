from __future__ import annotations

import torch
from skrl.memories.torch import RandomMemory


class TerminalBalancedRandomMemory(RandomMemory):
    """Random replay with a minimum fraction of terminal transitions per batch."""

    def __init__(self, *args, terminal_fraction: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= terminal_fraction <= 1.0:
            raise ValueError("terminal_fraction must be in [0, 1].")
        self.terminal_fraction = float(terminal_fraction)

    def sample(
        self,
        names: list[str],
        *,
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> list[list[torch.Tensor]]:
        if self.terminal_fraction <= 0.0 or sequence_length != 1 or "terminated" not in self.tensors:
            return super().sample(
                names,
                batch_size=batch_size,
                mini_batches=mini_batches,
                sequence_length=sequence_length,
            )

        size = len(self)
        terminal_mask = self.tensors_view["terminated"][:size].bool().reshape(size, -1).any(dim=-1)
        terminal_indexes = torch.nonzero(terminal_mask, as_tuple=False).flatten()
        if terminal_indexes.numel() == 0:
            return super().sample(names, batch_size=batch_size, mini_batches=mini_batches)

        terminal_count = min(batch_size, max(1, round(batch_size * self.terminal_fraction)))
        terminal_draw = terminal_indexes[
            torch.randint(terminal_indexes.numel(), (terminal_count,), device=terminal_indexes.device)
        ]
        random_draw = torch.randint(
            size,
            (batch_size - terminal_count,),
            device=terminal_indexes.device,
        )
        indexes = torch.cat((terminal_draw, random_draw))
        indexes = indexes[torch.randperm(indexes.numel(), device=indexes.device)]
        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
