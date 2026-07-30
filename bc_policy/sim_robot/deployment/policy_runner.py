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
    ) -> None:
        self.model, self.normalizer, self.checkpoint = load_policy(checkpoint_path, device=device, use_ema=use_ema)
        self.config = self.model.config
        self.device = next(self.model.parameters()).device
        self.num_inference_steps = num_inference_steps
        self.resize_images = resize_images
        self.visual_xy_lock_phase = visual_xy_lock_phase
        self.current_phase: float | None = None
        self.locked_visual_xy: np.ndarray | None = None

        train_config = self.checkpoint.get("train_config", {})
        self.image_keys = tuple(self.config.image_keys)
        image_shapes = train_config.get("image_shapes", {})
        self.expected_image_hws = {
            key: self._read_expected_hw(image_shapes.get(key, train_config.get("image_shape")))
            for key in self.image_keys
        }
        self.include_phase = bool(train_config.get("include_phase", False))

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

    def reset(self) -> None:
        self.state_history.clear()
        for history in self.image_history.values():
            history.clear()
        self.current_phase = None
        self.locked_visual_xy = None

    def update(self, obs: OnlineObservation | dict[str, Any]) -> None:
        if isinstance(obs, dict):
            phase = None if obs.get("phase") is None else float(obs["phase"])
            self.current_phase = phase
            state = self._prepare_state(np.asarray(obs["robot0_pos"]), phase)
            images = {
                key: self._prepare_image(np.asarray(obs[key]), image_key=key) for key in self.image_keys
            }
        else:
            if len(self.image_keys) != 1:
                raise ValueError("Multi-camera policy observations must be provided as a dictionary.")
            self.current_phase = obs.phase
            state = self._prepare_state(obs.robot0_pos, obs.phase)
            images = {self.image_keys[0]: self._prepare_image(obs.robot0_image, image_key=self.image_keys[0])}

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
        if self.include_phase and state.shape[0] == self.config.robot0_pos_dim - 1:
            if phase is None:
                raise ValueError("Phase-conditioned policy requires observation field 'phase' in [0, 1].")
            state = np.concatenate((state, np.asarray([np.clip(phase, 0.0, 1.0)], dtype=np.float32)))
        if state.shape[0] != self.config.robot0_pos_dim:
            raise ValueError(f"robot0_pos must have shape ({self.config.robot0_pos_dim},), got {state.shape}")
        return state

    def _prepare_image(self, image: np.ndarray, image_key: str = "robot0_image") -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"robot0_image must be HWC or CHW image, got shape {image.shape}")

        if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        elif image.shape[-1] != 3:
            raise ValueError(f"robot0_image must have 1, 3, or 4 channels, got shape {image.shape}")

        expected_image_hw = self.expected_image_hws.get(image_key)
        if expected_image_hw is not None and tuple(image.shape[:2]) != expected_image_hw:
            if not self.resize_images:
                raise ValueError(f"{image_key} expected HxW={expected_image_hw}, got {tuple(image.shape[:2])}")
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
        if visual_xy is not None:
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
        return self.normalizer.unnormalize_numpy("action", action_norm)


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

