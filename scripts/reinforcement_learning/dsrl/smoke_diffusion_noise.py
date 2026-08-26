#!/usr/bin/env python3
from __future__ import annotations

import argparse

from runtime import disable_optional_transformers_discovery, reexec_without_virtualgl


reexec_without_virtualgl()
disable_optional_transformers_discovery()

import torch

from diffusion_noise_adapter import DiffusionNoiseAdapter
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the deterministic DDIM noise interface used by DSRL.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    cfg = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))},
        device=args.device,
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        n_groups=8,
        noise_scheduler_type="DDIM",
        num_train_timesteps=4,
        num_inference_steps=2,
        crop_shape=None,
    )
    policy = DiffusionPolicy(cfg).to(args.device)
    adapter = DiffusionNoiseAdapter(policy)
    batch = {
        "observation.state": torch.zeros(2, 2, 10, device=args.device),
        "observation.environment_state": torch.zeros(2, 2, 1, device=args.device),
    }
    flat_noise = torch.randn(2, adapter.noise_dim, device=args.device)
    first = adapter.decode_processed(batch, flat_noise)
    second = adapter.decode_processed(batch, flat_noise)

    if first.shape != (2, 2, 10):
        raise AssertionError(f"Unexpected action chunk shape: {tuple(first.shape)}")
    if not torch.allclose(first, second):
        raise AssertionError("DDIM did not produce deterministic actions from identical noise.")
    if not torch.isfinite(first).all():
        raise AssertionError("Diffusion output contains NaN or Inf.")

    device_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device
    print(
        "[PASS] DSRL diffusion-noise smoke test "
        f"device={device_name!r} torch={torch.__version__} noise_shape={adapter.noise_shape} "
        f"action_chunk_shape={tuple(first.shape)}"
    )


if __name__ == "__main__":
    main()
