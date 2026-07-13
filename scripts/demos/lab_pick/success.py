from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StableLiftSuccessTracker:
    lift_height_m: float = 0.2
    hold_steps: int = 60
    max_object_gripper_distance_m: float = 0.08
    stable_steps: int = 0

    def update(self, lift_delta_m: float, has_touched: bool, object_gripper_distance_m: float) -> bool:
        stable = (
            has_touched
            and lift_delta_m >= self.lift_height_m
            and object_gripper_distance_m <= self.max_object_gripper_distance_m
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        return self.stable_steps >= self.hold_steps
