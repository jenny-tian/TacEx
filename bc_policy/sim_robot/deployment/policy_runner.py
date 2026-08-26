from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sim_robot.policy.flow_matching_policy import load_policy


ACTION_LABELS = [
    "target_x",
    "target_y",
    "target_z",
    "target_rot6d_0",
    "target_rot6d_1",
    "target_rot6d_2",
    "target_rot6d_3",
    "target_rot6d_4",
    "target_rot6d_5",
    "target_width",
]


@dataclass(frozen=True)
class OnlineObservation:
    robot0_pos: np.ndarray
    robot0_image: np.ndarray
    phase: float | None = None


class SimActionChunkPolicyRunner:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        use_ema: bool = True,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        resize_images: bool = True,
        visual_xy_lock_phase: float | None = 0.30,
        use_visual_xy_override: bool = True,
        use_visual_rotation_override: bool = True,
        demonstration_mode: str = "safe",
        force_demo_mode_projection: bool = False,
    ) -> None:
        self.model, self.normalizer, self.checkpoint = load_policy(
            checkpoint_path, device=device, use_ema=use_ema
        )
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        self.num_inference_steps = num_inference_steps
        self.resize_images = resize_images
        self.visual_xy_lock_phase = visual_xy_lock_phase
        self.use_visual_xy_override = bool(use_visual_xy_override)
        self.use_visual_rotation_override = bool(use_visual_rotation_override)
        self.current_phase: float | None = None
        self.locked_visual_xy: np.ndarray | None = None
        # Cached image-derived geometric estimates from the auxiliary heads.
        # These are predictions from camera observations, never simulator state.
        self.last_visual_xy: torch.Tensor | None = None
        self.last_visual_rotation: torch.Tensor | None = None

        train_config = self.checkpoint.get("train_config", {})
        self.image_keys = tuple(self.config.image_keys)
        image_shapes = train_config.get("image_shapes", {})
        self.expected_image_hws = {
            key: self._read_expected_hw(image_shapes.get(key, train_config.get("image_shape")))
            for key in self.image_keys
        }
        self.include_phase = bool(train_config.get("include_phase", False))
        self.include_demo_mode = bool(train_config.get("include_demo_mode", False))
        self.uses_demonstration_mode = self.include_demo_mode or bool(force_demo_mode_projection)
        # Keep evaluation model-driven by default. Projection is an explicit legacy/debug option.
        self.project_demo_mode_width = bool(force_demo_mode_projection)
        self.demonstration_mode = self._validate_demonstration_mode(demonstration_mode)
        self.safe_close_width_m = float(train_config.get("safe_close_width_m", 0.0065))
        self.overforce_close_width_m = float(train_config.get("overforce_close_width_m", 0.0055))
        self.close_projection_onset_width_m = float(
            train_config.get("close_projection_onset_width_m", 0.02)
        )
        self.overforce_projection_phase = float(train_config.get("overforce_projection_phase", 0.30))
        self.position_failure_offset_m = float(train_config.get("position_failure_offset_m", 0.03))
        self.position_failure_projection_phase = float(
            train_config.get("position_failure_projection_phase", 0.30)
        )
        self.demo_mode_names = ("position_failure", "safe", "overforce")
        self.demo_mode_probabilities = np.asarray(
            [
                train_config.get("position_failure_demo_mode_probability", 0.25),
                train_config.get("safe_demo_mode_probability", 0.50),
                train_config.get("overforce_demo_mode_probability", 0.25),
            ],
            dtype=np.float64,
        )
        if np.any(self.demo_mode_probabilities < 0.0) or self.demo_mode_probabilities.sum() <= 0.0:
            raise ValueError("Demonstration mode probabilities must be non-negative with positive sum.")
        self.demo_mode_probabilities /= self.demo_mode_probabilities.sum()

        self.state_history: deque[np.ndarray] = deque(maxlen=self.config.n_state_obs_steps)
        self.image_history: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=self.config.n_image_obs_steps) for key in self.image_keys
        }

        self.generator = None
        if seed is not None:
            try:
                self.generator = torch.Generator(device=self.device)
            except RuntimeError:
                self.generator = torch.Generator()
            self.generator.manual_seed(int(seed))

    @staticmethod
    def _read_expected_hw(shape: Any) -> tuple[int, int] | None:
        if shape is None:
            return None
        values = list(shape)
        if len(values) < 2:
            return None
        return int(values[0]), int(values[1])

    @staticmethod
    def _copy_frame(frame: np.ndarray) -> np.ndarray:
        return np.asarray(frame).copy()

    @staticmethod
    def _validate_demonstration_mode(mode: str) -> str:
        if mode not in {"safe", "overforce", "position_failure"}:
            raise ValueError(
                "demonstration_mode must be 'safe', 'overforce', or 'position_failure'"
            )
        return mode

    def reset(self, demonstration_mode: str | None = None) -> None:
        if demonstration_mode is not None:
            self.demonstration_mode = self._validate_demonstration_mode(demonstration_mode)
        elif getattr(self, "include_demo_mode", False):
            if self.generator is None:
                sample = float(np.random.random())
            else:
                sample = float(torch.rand((), device=self.device, generator=self.generator).item())
            mode_index = int(
                np.searchsorted(np.cumsum(self.demo_mode_probabilities), sample, side="right")
            )
            self.demonstration_mode = self.demo_mode_names[min(mode_index, 2)]
        self.state_history.clear()
        for history in self.image_history.values():
            history.clear()
        self.current_phase = None
        self.locked_visual_xy = None
        self.last_visual_xy = None
        self.last_visual_rotation = None

    def update(self, obs: OnlineObservation | dict[str, Any]) -> None:
        if isinstance(obs, dict):
            phase = None if obs.get("phase") is None else float(obs["phase"])
            self.current_phase = phase
            state = self._prepare_state(np.asarray(obs["robot0_pos"]), phase)
            images = {
                key: self._prepare_image(np.asarray(obs[key]), image_key=key)
                for key in self.image_keys
            }
        else:
            if len(self.image_keys) != 1:
                raise ValueError("Multi-camera policy observations must be provided as a dictionary.")
            self.current_phase = obs.phase
            state = self._prepare_state(obs.robot0_pos, obs.phase)
            images = {
                self.image_keys[0]: self._prepare_image(
                    obs.robot0_image, image_key=self.image_keys[0]
                )
            }

        if not self.state_history:
            for _ in range(self.config.n_state_obs_steps):
                self.state_history.append(self._copy_frame(state))
            for key, image in images.items():
                for _ in range(self.config.n_image_obs_steps):
                    self.image_history[key].append(self._copy_frame(image))
            return

        self.state_history.append(state)
        for key, image in images.items():
            self.image_history[key].append(image)

    def _prepare_state(self, state: np.ndarray, phase: float | None = None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        include_demo_mode = bool(getattr(self, "include_demo_mode", False))
        base_dim = self.config.robot0_pos_dim - int(self.include_phase) - int(include_demo_mode)
        if state.shape[0] == base_dim:
            extras = []
            if self.include_phase:
                if phase is None:
                    raise ValueError(
                        "Phase-conditioned policy requires observation field 'phase' in [0, 1]."
                    )
                extras.append(float(np.clip(phase, 0.0, 1.0)))
            if include_demo_mode:
                mode = getattr(self, "demonstration_mode", "safe")
                extras.append({"position_failure": -1.0, "safe": 0.0, "overforce": 1.0}[mode])
            if extras:
                state = np.concatenate((state, np.asarray(extras, dtype=np.float32)))
        if state.shape[0] != self.config.robot0_pos_dim:
            raise ValueError(
                f"robot0_pos must have shape ({self.config.robot0_pos_dim},), got {state.shape}"
            )
        return state

    def _prepare_image(self, image: np.ndarray, image_key: str = "robot0_image") -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"{image_key} must be HWC or CHW image, got shape {image.shape}")

        if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        elif image.shape[-1] != 3:
            raise ValueError(f"{image_key} must have 1, 3, or 4 channels, got shape {image.shape}")

        expected_image_hw = self.expected_image_hws.get(image_key)
        if expected_image_hw is not None and tuple(image.shape[:2]) != expected_image_hw:
            if not self.resize_images:
                raise ValueError(
                    f"{image_key} expected HxW={expected_image_hw}, got {tuple(image.shape[:2])}"
                )
            image = self._resize_hwc(image, expected_image_hw)

        image = image.astype(np.float32)
        if image.size and float(np.nanmax(image)) > 1.5:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)
        return np.transpose(image, (2, 0, 1)).astype(np.float32)

    @staticmethod
    def _resize_hwc(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        if image.dtype == np.uint8:
            uint8_image = image
        else:
            float_image = image.astype(np.float32)
            if float_image.size and float(np.nanmax(float_image)) <= 1.5:
                float_image = float_image * 255.0
            uint8_image = np.clip(float_image, 0.0, 255.0).astype(np.uint8)
        pil = Image.fromarray(uint8_image)
        pil = pil.resize((int(hw[1]), int(hw[0])), Image.BILINEAR)
        return np.asarray(pil)

    def is_ready(self) -> bool:
        return len(self.state_history) == self.config.n_state_obs_steps and all(
            len(history) == self.config.n_image_obs_steps for history in self.image_history.values()
        )

    def build_model_obs(self) -> dict[str, torch.Tensor]:
        if not self.is_ready():
            raise RuntimeError("Observation history is not ready. Call update() first.")
        state = np.stack(list(self.state_history), axis=0).astype(np.float32)

        robot0_pos = torch.from_numpy(state).unsqueeze(0).to(self.device)
        robot0_pos = self.normalizer.normalize_tensor("robot0_pos", robot0_pos)
        model_obs = {"robot0_pos": robot0_pos}
        for key, history in self.image_history.items():
            image = np.stack(list(history), axis=0).astype(np.float32)
            model_obs[key] = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return model_obs

    @torch.inference_mode()
    def encode_observation(self) -> torch.Tensor:
        """Return the frozen BC encoder's exact flattened conditioning tokens."""

        _, global_cond = self.model.obs_encoder(self.build_model_obs())
        return global_cond

    @torch.inference_mode()
    def visual_probe_observation(self) -> torch.Tensor:
        """Return frozen-BC image features plus normalized robot state.

        The object pose probe consumes only observations available to the BC
        (camera pixels and robot proprioception).  Simulator object state is
        deliberately not included in this runtime tensor.
        """
        model_obs = self.build_model_obs()
        cond_tokens, _ = self.model.obs_encoder(model_obs)
        state_tokens = int(self.config.n_state_obs_steps)
        image_tokens = cond_tokens[:, state_tokens:]
        pooled_image = image_tokens.mean(dim=1)
        current_state = model_obs["robot0_pos"][:, -1]
        return torch.cat((pooled_image, current_state), dim=-1)

    @torch.inference_mode()
    def visual_pose_estimate(self) -> torch.Tensor:
        """Return the frozen BC's image-derived XY and 6D rotation estimate."""

        batch_size = 1
        if self.last_visual_xy is None:
            xy = torch.zeros((batch_size, 2), device=self.device, dtype=torch.float32)
        else:
            xy = self.last_visual_xy.to(self.device, dtype=torch.float32).reshape(batch_size, 2)
        if self.last_visual_rotation is None:
            rotation = torch.zeros((batch_size, 6), device=self.device, dtype=torch.float32)
        else:
            rotation = self.last_visual_rotation.to(self.device, dtype=torch.float32).reshape(batch_size, 6)
        return torch.cat((xy, rotation), dim=-1)

    def _apply_demo_mode_width(self, action: np.ndarray) -> np.ndarray:
        if not getattr(self, "project_demo_mode_width", False):
            return action
        action = action.copy()
        closing = action[:, -1] < self.close_projection_onset_width_m
        if self.demonstration_mode == "position_failure":
            if (
                getattr(self, "current_phase", None) is not None
                and self.current_phase >= self.position_failure_projection_phase
            ):
                action[:, 1] += self.position_failure_offset_m
            action[closing, -1] = np.maximum(action[closing, -1], self.safe_close_width_m)
            return action
        if self.demonstration_mode == "overforce":
            if (
                getattr(self, "current_phase", None) is not None
                and self.current_phase >= self.overforce_projection_phase
            ):
                action[:, -1] = np.minimum(action[:, -1], self.overforce_close_width_m)
        else:
            action[closing, -1] = np.maximum(action[closing, -1], self.safe_close_width_m)
        return action

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        obs: OnlineObservation | dict[str, Any] | None = None,
        initial_noise: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        if obs is not None:
            self.update(obs)
        model_obs = self.build_model_obs()
        if initial_noise is not None:
            initial_noise = torch.as_tensor(initial_noise, dtype=torch.float32, device=self.device)
            if initial_noise.ndim == 2:
                initial_noise = initial_noise.unsqueeze(0)
        result = self.model.predict_action(
            model_obs,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
            initial_noise=initial_noise,
        )
        action_norm = result["action"].detach().cpu().numpy()[0]
        visual_xy = result.get("visual_xy")
        self.last_visual_xy = None if visual_xy is None else visual_xy.detach().clone()
        if visual_xy is not None and getattr(self, "use_visual_xy_override", True):
            current_xy = visual_xy.detach().cpu().numpy()[0]
            should_lock = (
                self.visual_xy_lock_phase is not None
                and self.current_phase is not None
                and self.current_phase >= self.visual_xy_lock_phase
            )
            if should_lock and self.locked_visual_xy is None:
                self.locked_visual_xy = current_xy.copy()
            selected_xy = self.locked_visual_xy if self.locked_visual_xy is not None else current_xy
            action_norm = action_norm.copy()
            action_norm[:, :2] = selected_xy
        visual_rotation = result.get("visual_rotation")
        self.last_visual_rotation = (
            None if visual_rotation is None else visual_rotation.detach().clone()
        )
        if visual_rotation is not None and self.use_visual_rotation_override:
            action_norm = action_norm.copy()
            action_norm[:, 3:9] = visual_rotation.detach().cpu().numpy()[0]
        action = self.normalizer.unnormalize_numpy("action", action_norm)
        return self._apply_demo_mode_width(action)


def format_action_chunk(action_chunk: np.ndarray, precision: int = 5) -> str:
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    labels = ACTION_LABELS[: action_chunk.shape[-1]]
    return "\n".join(
        [
            f"action_chunk shape={tuple(action_chunk.shape)}",
            "columns: " + ", ".join(labels),
            np.array2string(action_chunk, precision=precision, suppress_small=False),
        ]
    )
