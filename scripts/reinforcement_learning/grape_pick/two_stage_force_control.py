"""Policy-agnostic two-stage force overlay for berry picking.

The base policy remains responsible for reaching, alignment, and orientation.
After bilateral contact, this overlay first regulates pad-normal grip force and
then regulates pedicle tension by changing only the Cartesian target component
parallel to the known pull direction.  The force targets may be supplied per
environment, which permits a simulator-ground-truth oracle ablation without
exposing those values to the base policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grape_force_control import (
    ForceControlDiagnostics,
    ForceControllerConfig,
    ForceRange,
    GrapeForceController,
)


APPROACH = 0
GRIP_STABILIZE = 1
TENSION_PULL = 2
RETAIN = 3
STAGE_NAMES = ("approach", "grip_stabilize", "tension_pull", "retain")


@dataclass(frozen=True)
class ForceBalanceAlignmentConfig:
    """Bounded tool-centering response to unilateral pad-normal force."""

    contact_force_threshold_n: float = 0.05
    alignment_velocity_m_s: float = 0.030
    maximum_alignment_offset_m: float = 0.012

    def __post_init__(self) -> None:
        if self.contact_force_threshold_n <= 0.0:
            raise ValueError("contact force threshold must be positive")
        if self.alignment_velocity_m_s <= 0.0:
            raise ValueError("alignment velocity must be positive")
        if self.maximum_alignment_offset_m <= 0.0:
            raise ValueError("maximum alignment offset must be positive")


@dataclass(frozen=True)
class ForceBalanceAlignmentDiagnostics:
    active: torch.Tensor
    normalized_force_imbalance: torch.Tensor
    alignment_offset_m: torch.Tensor


class ForceBalanceAligner:
    """Move the tool toward the contacted finger until both pads engage.

    The closing axis points from the left pad to the right pad.  If only the
    left pad contacts the fruit, the tool center must move toward the left pad
    (negative closing-axis direction), and vice versa.  The offset is retained
    after bilateral contact so the subsequent grip/tension stages remain
    centered on the acquired object.
    """

    def __init__(self, config: ForceBalanceAlignmentConfig | None = None) -> None:
        self.config = ForceBalanceAlignmentConfig() if config is None else config
        self._offset_m: torch.Tensor | None = None
        self._search_direction: torch.Tensor | None = None

    def reset(self) -> None:
        self._offset_m = None
        self._search_direction = None

    def control(
        self,
        policy_action: torch.Tensor,
        *,
        left_force_n: torch.Tensor,
        right_force_n: torch.Tensor,
        bilateral_contact: torch.Tensor,
        closing_axis_b: torch.Tensor,
        dt_s: float,
    ) -> tuple[torch.Tensor, ForceBalanceAlignmentDiagnostics]:
        if policy_action.ndim != 2 or policy_action.shape[-1] != 10:
            raise ValueError("policy_action must have shape [B, 10]")
        batch = len(policy_action)
        left = torch.as_tensor(
            left_force_n, dtype=policy_action.dtype, device=policy_action.device
        ).reshape(-1)
        right = torch.as_tensor(
            right_force_n, dtype=policy_action.dtype, device=policy_action.device
        ).reshape(-1)
        bilateral = torch.as_tensor(
            bilateral_contact, dtype=torch.bool, device=policy_action.device
        ).reshape(-1)
        axis = torch.as_tensor(
            closing_axis_b, dtype=policy_action.dtype, device=policy_action.device
        )
        if len(left) != batch or len(right) != batch or len(bilateral) != batch:
            raise ValueError("force/contact batch does not match action batch")
        if axis.shape != (batch, 3):
            raise ValueError("closing_axis_b must have shape [B, 3]")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        if bool((axis_norm <= 1.0e-8).any()):
            raise ValueError("closing_axis_b must be non-zero")
        axis = axis / axis_norm
        if self._offset_m is None or len(self._offset_m) != batch:
            self._offset_m = torch.zeros(
                batch, dtype=policy_action.dtype, device=policy_action.device
            )
            self._search_direction = torch.zeros_like(self._offset_m)
        assert self._search_direction is not None

        threshold = float(self.config.contact_force_threshold_n)
        left_only = (left >= threshold) & (right < threshold) & ~bilateral
        right_only = (right >= threshold) & (left < threshold) & ~bilateral
        unilateral = left_only | right_only
        imbalance = (right - left) / torch.maximum(
            torch.maximum(left, right), torch.full_like(left, threshold)
        )
        detected_direction = torch.sign(imbalance)
        self._search_direction = torch.where(
            unilateral, detected_direction, self._search_direction
        )
        self._search_direction = torch.where(
            bilateral, torch.zeros_like(self._search_direction), self._search_direction
        )
        limit = float(self.config.maximum_alignment_offset_m)
        reverse = (
            ((self._offset_m <= -limit) & (self._search_direction < 0.0))
            | ((self._offset_m >= limit) & (self._search_direction > 0.0))
        ) & ~bilateral
        self._search_direction = torch.where(
            reverse, -self._search_direction, self._search_direction
        )
        active = (self._search_direction != 0.0) & ~bilateral
        velocity = torch.where(
            active,
            self._search_direction * float(self.config.alignment_velocity_m_s),
            torch.zeros_like(imbalance),
        )
        self._offset_m = torch.clamp(
            self._offset_m + velocity * float(dt_s),
            min=-limit,
            max=limit,
        )
        executed = policy_action.clone()
        executed[:, :3] += axis * self._offset_m[:, None]
        return executed, ForceBalanceAlignmentDiagnostics(
            active=active.detach().clone(),
            normalized_force_imbalance=imbalance.detach().clone(),
            alignment_offset_m=self._offset_m.detach().clone(),
        )


@dataclass(frozen=True)
class TwoStageForceControllerConfig:
    """Configuration for the grip-then-tension hybrid controller."""

    contact_confirm_steps: int = 3
    grip_stable_steps: int = 6
    tension_filter_alpha: float = 0.35
    tension_kp_velocity_m_s_per_n: float = 0.012
    maximum_pull_velocity_m_s: float = 0.010
    maximum_return_velocity_m_s: float = 0.004
    maximum_pull_offset_m: float = 0.060

    def __post_init__(self) -> None:
        if self.contact_confirm_steps < 1:
            raise ValueError("contact_confirm_steps must be positive")
        if self.grip_stable_steps < 1:
            raise ValueError("grip_stable_steps must be positive")
        if not 0.0 < self.tension_filter_alpha <= 1.0:
            raise ValueError("tension_filter_alpha must lie in (0, 1]")
        if self.tension_kp_velocity_m_s_per_n <= 0.0:
            raise ValueError("tension gain must be positive")
        if self.maximum_pull_velocity_m_s <= 0.0:
            raise ValueError("maximum pull velocity must be positive")
        if self.maximum_return_velocity_m_s < 0.0:
            raise ValueError("maximum return velocity must be non-negative")
        if self.maximum_pull_offset_m <= 0.0:
            raise ValueError("maximum pull offset must be positive")


@dataclass(frozen=True)
class TwoStageForceDiagnostics:
    stage: torch.Tensor
    grip: ForceControlDiagnostics
    filtered_tension_n: torch.Tensor
    target_tension_n: torch.Tensor
    tension_error_n: torch.Tensor
    pull_velocity_m_s: torch.Tensor
    pull_offset_m: torch.Tensor


class TwoStageBerryForceController:
    """Coordinate normal grip force with pull-axis pedicle tension.

    Stage 1 starts on bilateral contact.  Translation along the pull axis is
    held while the gripper force servo establishes a stable grip.  Stage 2
    then applies a bounded admittance correction along the pull direction while
    the gripper loop remains active.  Once detachment is observed, the last
    pull position and grip are retained long enough for the task success dwell.
    """

    def __init__(
        self,
        grip_target_range: ForceRange,
        config: TwoStageForceControllerConfig | None = None,
        *,
        grip_controller_config: ForceControllerConfig | None = None,
    ) -> None:
        self.config = TwoStageForceControllerConfig() if config is None else config
        self.grip_controller = GrapeForceController(
            grip_target_range, grip_controller_config
        )
        self.reset()

    @property
    def grip_target_range(self) -> ForceRange:
        return self.grip_controller.target_range

    def set_grip_target_range(self, target_range: ForceRange) -> None:
        self.grip_controller.set_target_range(target_range)

    def reset(self) -> None:
        self.grip_controller.reset()
        self._stage: torch.Tensor | None = None
        self._contact_confirm_count: torch.Tensor | None = None
        self._grip_stable_count: torch.Tensor | None = None
        self._pull_anchor: torch.Tensor | None = None
        self._pull_offset: torch.Tensor | None = None
        self._filtered_tension: torch.Tensor | None = None

    @staticmethod
    def _batch_vector(
        value: float | torch.Tensor,
        *,
        like: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(-1)
        if result.numel() == 1:
            result = result.expand(len(like))
        if len(result) != len(like):
            raise ValueError(f"{name} batch does not match action batch")
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} must be finite")
        return result

    def _initialize(self, action: torch.Tensor, tension: torch.Tensor) -> None:
        batch = len(action)
        self._stage = torch.full(
            (batch,), APPROACH, dtype=torch.long, device=action.device
        )
        self._contact_confirm_count = torch.zeros(
            batch, dtype=torch.long, device=action.device
        )
        self._grip_stable_count = torch.zeros(
            batch, dtype=torch.long, device=action.device
        )
        self._pull_anchor = action[:, :3].detach().clone()
        self._pull_offset = torch.zeros(batch, dtype=action.dtype, device=action.device)
        self._filtered_tension = tension.detach().clone()

    def control(
        self,
        policy_action: torch.Tensor,
        *,
        grip_force_n: torch.Tensor,
        bilateral_contact: torch.Tensor,
        pedicle_tension_n: torch.Tensor,
        stem_attached: torch.Tensor,
        pull_direction: torch.Tensor,
        target_tension_n: float | torch.Tensor,
        dt_s: float,
        grip_safety_limit_n: float | torch.Tensor,
    ) -> tuple[torch.Tensor, TwoStageForceDiagnostics]:
        if policy_action.ndim != 2 or policy_action.shape[-1] != 10:
            raise ValueError("policy_action must have shape [B, 10]")
        batch = len(policy_action)
        bilateral = torch.as_tensor(
            bilateral_contact, dtype=torch.bool, device=policy_action.device
        ).reshape(-1)
        attached = torch.as_tensor(
            stem_attached, dtype=torch.bool, device=policy_action.device
        ).reshape(-1)
        tension = self._batch_vector(
            pedicle_tension_n, like=policy_action, name="pedicle_tension_n"
        )
        grip_force = self._batch_vector(
            grip_force_n, like=policy_action, name="grip_force_n"
        )
        tension_target = self._batch_vector(
            target_tension_n, like=policy_action, name="target_tension_n"
        )
        direction = torch.as_tensor(
            pull_direction, dtype=policy_action.dtype, device=policy_action.device
        )
        if direction.shape != (batch, 3):
            raise ValueError("pull_direction must have shape [B, 3]")
        direction_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if bool((direction_norm <= 1.0e-8).any()):
            raise ValueError("pull_direction must be non-zero")
        direction = direction / direction_norm
        if len(bilateral) != batch or len(attached) != batch:
            raise ValueError("contact/attachment batch does not match action batch")
        if bool((tension_target <= 0.0).any()):
            raise ValueError("target_tension_n must be positive")

        if self._stage is None or len(self._stage) != batch:
            self._initialize(policy_action, tension)
        assert self._contact_confirm_count is not None
        contact_candidate = bilateral & (
            grip_force >= self.grip_controller.config.contact_on_force_n
        )
        self._contact_confirm_count = torch.where(
            contact_candidate,
            torch.clamp(
                self._contact_confirm_count + 1,
                max=self.config.contact_confirm_steps,
            ),
            torch.zeros_like(self._contact_confirm_count),
        )
        confirmed_bilateral = contact_candidate & (
            self._contact_confirm_count >= self.config.contact_confirm_steps
        )

        executed, grip_diagnostics = self.grip_controller.control(
            policy_action,
            contact_force_n=grip_force,
            bilateral_contact=confirmed_bilateral,
            dt_s=dt_s,
            safety_force_limit_n=grip_safety_limit_n,
        )

        assert self._stage is not None
        assert self._contact_confirm_count is not None
        assert self._grip_stable_count is not None
        assert self._pull_anchor is not None
        assert self._pull_offset is not None
        assert self._filtered_tension is not None

        just_contacted = grip_diagnostics.active & (self._stage == APPROACH)
        self._pull_anchor = torch.where(
            just_contacted[:, None], policy_action[:, :3], self._pull_anchor
        )
        self._pull_offset = torch.where(
            just_contacted, torch.zeros_like(self._pull_offset), self._pull_offset
        )
        self._filtered_tension = torch.where(
            just_contacted, tension, self._filtered_tension
        )
        self._stage = torch.where(
            just_contacted,
            torch.full_like(self._stage, GRIP_STABILIZE),
            self._stage,
        )

        # A brief contact spike must not permanently seize the pull axis.  Let
        # the frozen policy approach/reacquire after the grip servo's contact
        # hysteresis has expired, and restart any later pull from zero offset.
        contact_lost = (
            (self._stage >= GRIP_STABILIZE)
            & (self._stage <= TENSION_PULL)
            & attached
            & ~grip_diagnostics.active
        )
        self._stage = torch.where(
            contact_lost, torch.full_like(self._stage, APPROACH), self._stage
        )
        self._grip_stable_count = torch.where(
            contact_lost,
            torch.zeros_like(self._grip_stable_count),
            self._grip_stable_count,
        )
        self._pull_offset = torch.where(
            contact_lost, torch.zeros_like(self._pull_offset), self._pull_offset
        )

        in_grip_band = (
            grip_diagnostics.active
            & confirmed_bilateral
            & ~grip_diagnostics.safety_override
            & (grip_diagnostics.filtered_force_n >= self.grip_target_range.minimum_n)
            & (grip_diagnostics.filtered_force_n <= self.grip_target_range.maximum_n)
        )
        stabilizing = self._stage == GRIP_STABILIZE
        self._grip_stable_count = torch.where(
            stabilizing & in_grip_band,
            self._grip_stable_count + 1,
            torch.where(stabilizing, torch.zeros_like(self._grip_stable_count), self._grip_stable_count),
        )
        begin_pull = stabilizing & (
            self._grip_stable_count >= self.config.grip_stable_steps
        )
        self._stage = torch.where(
            begin_pull, torch.full_like(self._stage, TENSION_PULL), self._stage
        )
        self._stage = torch.where(
            ~attached & (self._stage >= GRIP_STABILIZE),
            torch.full_like(self._stage, RETAIN),
            self._stage,
        )

        alpha = self.config.tension_filter_alpha
        self._filtered_tension = torch.where(
            self._stage == TENSION_PULL,
            alpha * tension + (1.0 - alpha) * self._filtered_tension,
            tension,
        )
        tension_error = tension_target - self._filtered_tension
        pull_velocity = torch.clamp(
            self.config.tension_kp_velocity_m_s_per_n * tension_error,
            min=-self.config.maximum_return_velocity_m_s,
            max=self.config.maximum_pull_velocity_m_s,
        )
        pull_enabled = (
            (self._stage == TENSION_PULL)
            & attached
            & confirmed_bilateral
            & ~grip_diagnostics.safety_override
        )
        pull_velocity = torch.where(
            pull_enabled, pull_velocity, torch.zeros_like(pull_velocity)
        )
        self._pull_offset = torch.clamp(
            self._pull_offset + pull_velocity * float(dt_s),
            min=0.0,
            max=self.config.maximum_pull_offset_m,
        )

        force_controlled = (self._stage >= GRIP_STABILIZE)[:, None]
        relative = policy_action[:, :3] - self._pull_anchor
        policy_along_pull = (relative * direction).sum(dim=-1)
        controlled_position = policy_action[:, :3] + direction * (
            self._pull_offset - policy_along_pull
        )[:, None]
        executed[:, :3] = torch.where(
            force_controlled, controlled_position, executed[:, :3]
        )

        diagnostics = TwoStageForceDiagnostics(
            stage=self._stage.detach().clone(),
            grip=grip_diagnostics,
            filtered_tension_n=self._filtered_tension.detach().clone(),
            target_tension_n=tension_target.detach().clone(),
            tension_error_n=tension_error.detach().clone(),
            pull_velocity_m_s=pull_velocity.detach().clone(),
            pull_offset_m=self._pull_offset.detach().clone(),
        )
        return executed, diagnostics


__all__ = [
    "APPROACH",
    "ForceBalanceAligner",
    "ForceBalanceAlignmentConfig",
    "ForceBalanceAlignmentDiagnostics",
    "GRIP_STABILIZE",
    "RETAIN",
    "STAGE_NAMES",
    "TENSION_PULL",
    "TwoStageBerryForceController",
    "TwoStageForceControllerConfig",
    "TwoStageForceDiagnostics",
]
