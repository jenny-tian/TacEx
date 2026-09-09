from __future__ import annotations

import json
from pathlib import Path

from tacex_tasks.grape_pick.dataset import GrapeTrajectoryBuffer, validate_episode


class RaspberryTrajectoryBuffer(GrapeTrajectoryBuffer):
    """Grape-compatible trajectory schema with raspberry-specific metadata."""

    def save_episode(self, env_index: int, output_dir: str | Path, **kwargs) -> Path:
        output_path = super().save_episode(env_index, output_dir, **kwargs)
        metadata_path = output_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "task": "TacEx-RaspberryPick-Force-v0",
                "instruction": (
                    "approach the ripe red raspberry, grip it below the damage "
                    "threshold, and pull until the pedicle detaches"
                ),
                "object": "raspberry",
                "contact_force_source": (
                    "PhysX ContactSensor filtered to target_raspberry"
                ),
                "raspberry_position_randomization_m": metadata[
                    "grape_position_randomization_m"
                ],
                "sampled_raspberry_position_offset_m": metadata[
                    "sampled_grape_position_offset_m"
                ],
                "raspberry_reset_pos_w": metadata["grape_reset_pos_w"],
                "raspberry_reset_quat_w": metadata["grape_reset_quat_w"],
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return output_path


__all__ = ["RaspberryTrajectoryBuffer", "validate_episode"]

