from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.reinforcement_learning.dppo.tactile_dppo import DPPORollout, TactileDPPO
from scripts.reinforcement_learning.dppo.tactile_dppo_wrapper import TactileDPPOLabPickWrapper
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel


class _FakeResidual(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 4))


class _FakeUnet(nn.Module):
    def __init__(self, global_dim: int, timestep_dim: int, action_dim: int) -> None:
        super().__init__()
        self.down_modules = nn.ModuleList(
            [nn.ModuleList([_FakeResidual(global_dim + timestep_dim)])]
        )
        self.projection = nn.Linear(global_dim, action_dim, bias=False)

    def forward(self, sample, timestep, global_cond=None):
        del timestep
        return 0.05 * sample + self.projection(global_cond).unsqueeze(1)


class _FakeScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            prediction_type="epsilon",
            thresholding=False,
            clip_sample=True,
            clip_sample_range=1.0,
        )
        self.alphas_cumprod = torch.linspace(0.99, 0.30, 10)
        self.timesteps = torch.tensor([9, 6, 3, 0])

    def set_timesteps(self, count, device=None):
        assert count == 4
        self.timesteps = torch.tensor([9, 6, 3, 0], device=device)


class _FakeDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = _FakeUnet(global_dim=6, timestep_dim=4, action_dim=2)
        self.noise_scheduler = _FakeScheduler()
        self.num_inference_steps = 4

    @staticmethod
    def visual_residual_to_action(residual, visual_xy):
        action = residual.clone()
        action[..., :2] += visual_xy.unsqueeze(1)
        return action


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            noise_scheduler_type="DDPM",
            diffusion_step_embed_dim=4,
            horizon=8,
            action_feature=SimpleNamespace(shape=(2,)),
            n_obs_steps=2,
            n_action_steps=4,
            use_visual_xy_residual=True,
        )
        self.diffusion = _FakeDiffusion()


def _model() -> TactileDPPO:
    torch.manual_seed(2)
    return TactileDPPO(
        _FakePolicy(),
        fine_tune_denoising_steps=2,
        update_epochs=2,
        minibatches=2,
        value_hidden_dims=(16, 16),
    )


def test_zero_tactile_adapter_preserves_pretrained_condition() -> None:
    model = _model()
    condition = torch.randn(3, 6)
    tactile = torch.randn(3, 5)
    assert torch.equal(model.conditioned(condition, tactile), condition)


def test_sampling_and_logprob_std_floors_are_separate() -> None:
    model = _model()
    assert model.min_sampling_denoising_std == pytest.approx(1.0e-3)
    assert model.min_logprob_denoising_std == pytest.approx(0.1)
    with pytest.raises(ValueError, match="sampling std floor"):
        TactileDPPO(
            _FakePolicy(),
            min_sampling_denoising_std=0.2,
            min_logprob_denoising_std=0.1,
        )


def test_inference_schedule_can_be_frozen_independently_of_bc_config() -> None:
    model = TactileDPPO(_FakePolicy(), num_inference_steps=4, fine_tune_denoising_steps=2)
    assert model.num_inference_steps == 4
    with pytest.raises(ValueError, match="num_inference_steps"):
        TactileDPPO(_FakePolicy(), num_inference_steps=11)


def test_frozen_reverse_steps_ignore_the_trainable_tactile_adapter() -> None:
    model = _model()
    condition = torch.randn(2, 6)
    sample = torch.randn(2, 8, 2)
    timestep = torch.full((2,), 9, dtype=torch.long)
    tactile_a = torch.zeros(2, 5)
    tactile_b = torch.ones(2, 5)
    with torch.no_grad():
        model.tactile_adapter.weight.fill_(0.25)
        mean_a, std_a = model.transition_mean_std(
            sample, timestep, condition, tactile_a, use_base_policy=True
        )
        mean_b, std_b = model.transition_mean_std(
            sample, timestep, condition, tactile_b, use_base_policy=True
        )
    assert torch.equal(mean_a, mean_b)
    assert torch.equal(std_a, std_b)


def test_unchanged_denoising_policy_has_exact_unit_importance_ratio() -> None:
    model = _model()
    condition = torch.randn(2, 6)
    tactile = torch.randn(2, 5)
    visual_xy = condition[:, -2:].clone()
    sample = model.sample(condition, tactile, visual_xy)
    recomputed = []
    for denoising_index in range(model.fine_tune_denoising_steps):
        log_prob, _ = model.transition_log_prob(
            sample["chain_previous"][:, denoising_index],
            sample["chain_next"][:, denoising_index],
            sample["timesteps"][:, denoising_index],
            condition,
            tactile,
        )
        recomputed.append(log_prob)
    new_log_prob = torch.stack(recomputed, dim=1)
    ratio = (new_log_prob - sample["old_log_probs"]).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1.0e-6, rtol=1.0e-6)


def test_rollout_update_changes_real_denoising_parameters_and_tactile_response() -> None:
    model = _model()
    rollout = DPPORollout(capacity=4)
    initial_actor = model.actor_ft.projection.weight.detach().clone()
    for index in range(4):
        condition = torch.randn(1, 6)
        tactile = torch.tensor([[0.2 * index, 0.1, 1.0, 0.0, float(index > 0)]])
        visual_xy = condition[:, -2:].clone()
        sample = model.sample(condition, tactile, visual_xy)
        rollout.add(
            global_condition=condition,
            tactile=tactile,
            chain_previous=sample["chain_previous"],
            chain_next=sample["chain_next"],
            timesteps=sample["timesteps"],
            old_log_probs=sample["old_log_probs"],
            value=sample["value"],
            reward=float(index),
            done=index == 3,
        )
    diagnostics = model.update(rollout, bootstrap_value=0.0)
    assert diagnostics.optimizer_steps == 4
    assert diagnostics.denoising_transitions == 8
    assert not torch.equal(model.actor_ft.projection.weight.detach(), initial_actor)
    assert torch.count_nonzero(model.tactile_adapter.weight.detach()) > 0

    condition = torch.randn(1, 6)
    zero = torch.zeros(1, 5)
    contact = torch.tensor([[1.0, 0.5, 1.0, 1.0, 1.0]])
    conditioned_zero = model.conditioned(condition, zero)
    conditioned_contact = model.conditioned(condition, contact)
    assert not torch.allclose(conditioned_zero, conditioned_contact)


def test_visual_residual_translation_is_an_exact_inverse() -> None:
    action = torch.randn(3, 8, 10) * 0.2
    visual_xy = torch.randn(3, 2) * 0.2
    residual = DiffusionModel.action_to_visual_residual(action, visual_xy)
    reconstructed = DiffusionModel.visual_residual_to_action(residual, visual_xy)
    assert torch.equal(reconstructed[..., 2:], action[..., 2:])
    assert torch.allclose(reconstructed[..., :2], action[..., :2], atol=1.0e-7, rtol=0.0)


def test_dppo_chain_and_likelihood_remain_in_visual_residual_coordinates() -> None:
    model = _model()
    condition = torch.zeros(1, 6)
    visual_xy = torch.tensor([[0.2, -0.3]])
    condition[:, -2:] = visual_xy
    tactile = torch.zeros(1, 5)
    sample = model.sample(condition, tactile, visual_xy)
    final_residual = sample["chain_next"][:, -1, 1:5]
    assert torch.equal(sample["residual_action"], final_residual)
    expected_action = final_residual.clone()
    expected_action[..., :2] += visual_xy.unsqueeze(1)
    assert torch.allclose(sample["action"], expected_action.clamp(-1.0, 1.0))


def test_visual_xy_head_consumes_only_latest_selected_camera_features() -> None:
    model = DiffusionModel.__new__(DiffusionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_features={"wrist": object(), "third": object()},
        visual_xy_camera_index=-1,
    )
    model.rgb_feature_dim = 3
    model.visual_xy_head = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.visual_xy_head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    features = torch.randn(2, 2, 6)
    predicted = model._visual_xy_from_image_features(features)
    assert torch.equal(predicted, features[:, -1, 3:5])
    altered = features.clone()
    altered[:, :, :3] += 1000.0
    altered[:, 0, 3:] -= 1000.0
    assert torch.equal(model._visual_xy_from_image_features(altered), predicted)


def test_reset_clears_visual_xy_lock() -> None:
    wrapper = TactileDPPOLabPickWrapper.__new__(TactileDPPOLabPickWrapper)
    wrapper._policy_steps = 123
    wrapper._locked_visual_xy = torch.ones(1, 2)
    wrapper._reset_visual_xy_state()
    assert wrapper._policy_steps == 0
    assert wrapper._locked_visual_xy is None


def test_legacy_or_missing_checkpoint_contract_is_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="Incompatible DPPO checkpoint contract"):
        model.load_checkpoint_payload({"contract_version": "single_latent_flow_ppo"})


def test_diffusion_bc_camera_contract_keeps_the_full_reset_view() -> None:
    launcher = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "bc_training"
        / "train_lab_pick_diffusion.py"
    ).read_text(encoding="utf-8")
    wrapper = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "reinforcement_learning"
        / "dppo"
        / "tactile_dppo_wrapper.py"
    ).read_text(encoding="utf-8")
    assert 'MATCHED_CAMERA_SHAPE = (224, 224)' in launcher
    assert 'f"--policy.crop_shape=[{crop_shape[0]},{crop_shape[1]}]"' in launcher
    assert 'f"--policy.crop_is_random={str(args.crop_is_random).lower()}"' in launcher
    assert (
        'DIFFUSION_BC_CAMERA_CONTRACT = "matched_full_frame_visual_xy_residual_224x224_v2"'
        in wrapper
    )


def test_diffusion_bc_launcher_forces_visual_identifiability() -> None:
    launcher = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "bc_training"
        / "train_lab_pick_diffusion.py"
    ).read_text(encoding="utf-8")
    assert 'default=8' in launcher
    assert 'default=16' in launcher
    assert 'default=[0, 1]' in launcher
    assert 'default=0.5' in launcher
    assert '"observation_identifiability_contract"' in launcher
    assert '"--policy.use_visual_xy_residual=true"' in launcher
    assert '"action_coordinate_contract"' in launcher
