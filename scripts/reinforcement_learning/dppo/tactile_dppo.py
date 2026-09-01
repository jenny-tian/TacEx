"""Genuine denoising-chain PPO for a frozen LeRobot Diffusion Policy.

The implementation follows the diffusion-MDP construction used by Ren et al.
(arXiv:2409.00588): every trainable reverse-diffusion transition is stored,
assigned a Gaussian transition likelihood, and optimized with a clipped PPO
ratio.  It intentionally does not expose or optimize one initial latent as a
surrogate action.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal


TACTILE_DPPO_CONTRACT_VERSION = "tactile_dppo_visual_xy_residual_transition_v3"


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf.")


@dataclass
class DPPODiagnostics:
    optimizer_steps: int = 0
    denoising_transitions: int = 0
    actor_loss: float = 0.0
    value_loss: float = 0.0
    approx_kl: float = 0.0
    ratio_mean: float = 1.0
    clip_fraction: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class DPPORollout:
    """Fixed-size on-policy buffer retaining the actual reverse-diffusion chain."""

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("DPPO rollout capacity must be positive.")
        self.capacity = int(capacity)
        self.clear()

    def clear(self) -> None:
        self.global_conditions: list[Tensor] = []
        self.tactile: list[Tensor] = []
        self.chain_previous: list[Tensor] = []
        self.chain_next: list[Tensor] = []
        self.timesteps: list[Tensor] = []
        self.old_log_probs: list[Tensor] = []
        self.values: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.dones: list[Tensor] = []

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def full(self) -> bool:
        return len(self) >= self.capacity

    def add(
        self,
        *,
        global_condition: Tensor,
        tactile: Tensor,
        chain_previous: Tensor,
        chain_next: Tensor,
        timesteps: Tensor,
        old_log_probs: Tensor,
        value: Tensor,
        reward: Tensor | float,
        done: Tensor | bool,
    ) -> None:
        if self.full:
            raise RuntimeError("DPPO rollout is already full.")
        transition_count = int(timesteps.shape[-1])
        expected_chain = (transition_count, *chain_previous.shape[-2:])
        if tuple(chain_previous.shape[-3:]) != expected_chain:
            raise ValueError(
                f"chain_previous must end in {expected_chain}, got {tuple(chain_previous.shape)}."
            )
        if chain_next.shape != chain_previous.shape:
            raise ValueError("Previous and next diffusion chains must have identical shapes.")
        if old_log_probs.shape[-1] != transition_count:
            raise ValueError("One old log probability is required per denoising transition.")
        fields = (
            ("global_condition", global_condition, self.global_conditions),
            ("tactile", tactile, self.tactile),
            ("chain_previous", chain_previous, self.chain_previous),
            ("chain_next", chain_next, self.chain_next),
            ("timesteps", timesteps, self.timesteps),
            ("old_log_probs", old_log_probs, self.old_log_probs),
            ("value", value, self.values),
            ("reward", torch.as_tensor(reward), self.rewards),
            ("done", torch.as_tensor(done), self.dones),
        )
        for name, tensor, destination in fields:
            tensor = torch.as_tensor(tensor).detach().cpu()
            _finite(name, tensor.float())
            destination.append(tensor)

    def tensors(self, device: torch.device | str) -> dict[str, Tensor]:
        if not self:
            raise RuntimeError("Cannot materialize an empty DPPO rollout.")
        return {
            "global_condition": torch.cat(self.global_conditions, dim=0).to(device),
            "tactile": torch.cat(self.tactile, dim=0).to(device),
            "chain_previous": torch.cat(self.chain_previous, dim=0).to(device),
            "chain_next": torch.cat(self.chain_next, dim=0).to(device),
            "timesteps": torch.cat(self.timesteps, dim=0).long().to(device),
            "old_log_probs": torch.cat(self.old_log_probs, dim=0).to(device),
            "values": torch.cat(self.values, dim=0).reshape(-1).to(device),
            "rewards": torch.stack(self.rewards, dim=0).reshape(-1).float().to(device),
            "dones": torch.stack(self.dones, dim=0).reshape(-1).float().to(device),
        }


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 512, 512)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, global_condition: Tensor, tactile: Tensor) -> Tensor:
        return self.net(torch.cat((global_condition, tactile), dim=-1)).squeeze(-1)


class TactileDPPO(nn.Module):
    """Freeze visual/base diffusion modules and PPO-tune the last denoising steps."""

    def __init__(
        self,
        pretrained_policy: nn.Module,
        *,
        tactile_dim: int = 5,
        fine_tune_denoising_steps: int = 5,
        num_inference_steps: int | None = None,
        min_sampling_denoising_std: float = 1.0e-3,
        min_logprob_denoising_std: float = 0.1,
        learning_rate: float = 3.0e-5,
        clip_range: float = 0.2,
        value_clip: float = 0.2,
        grad_clip: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 8,
        minibatches: int = 4,
        value_loss_coefficient: float = 1.0,
        entropy_coefficient: float = 0.0,
        value_hidden_dims: tuple[int, ...] = (512, 512, 512),
    ) -> None:
        super().__init__()
        diffusion = getattr(pretrained_policy, "diffusion", None)
        if diffusion is None or not hasattr(diffusion, "unet"):
            raise TypeError("pretrained_policy must be a LeRobot DiffusionPolicy.")
        config = pretrained_policy.config
        if config.noise_scheduler_type != "DDPM":
            raise ValueError("Tactile DPPO requires a stochastic DDPM checkpoint.")
        if not bool(getattr(config, "use_visual_xy_residual", False)):
            raise ValueError(
                "Tactile DPPO requires a visual-x/y residual Diffusion BC checkpoint."
            )
        for name, value in (
            ("tactile_dim", tactile_dim),
            ("fine_tune_denoising_steps", fine_tune_denoising_steps),
            ("update_epochs", update_epochs),
            ("minibatches", minibatches),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if (
            min_sampling_denoising_std <= 0.0
            or min_logprob_denoising_std <= 0.0
            or learning_rate <= 0.0
            or grad_clip <= 0.0
        ):
            raise ValueError("DPPO standard deviations, learning rate, and grad clip must be positive.")
        if min_sampling_denoising_std > min_logprob_denoising_std:
            raise ValueError(
                "DPPO sampling std floor cannot exceed the log-probability std floor."
            )
        if not 0.0 < clip_range < 1.0:
            raise ValueError("clip_range must be in (0, 1).")

        self.base_policy = pretrained_policy.eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.actor_ft = copy.deepcopy(diffusion.unet)
        for parameter in self.actor_ft.parameters():
            parameter.requires_grad_(True)
        self.actor_ft.train()
        self.scheduler = diffusion.noise_scheduler
        self.num_inference_steps = int(
            diffusion.num_inference_steps
            if num_inference_steps is None
            else num_inference_steps
        )
        maximum_inference_steps = int(
            getattr(
                self.scheduler.config,
                "num_train_timesteps",
                len(self.scheduler.alphas_cumprod),
            )
        )
        if not 1 <= self.num_inference_steps <= maximum_inference_steps:
            raise ValueError(
                "num_inference_steps must be in "
                f"[1, {maximum_inference_steps}], got {self.num_inference_steps}."
            )
        if fine_tune_denoising_steps > self.num_inference_steps:
            raise ValueError("fine_tune_denoising_steps exceeds the inference schedule.")

        first_residual = self.actor_ft.down_modules[0][0]
        cond_dim = int(first_residual.cond_encoder[1].in_features)
        self.global_condition_dim = cond_dim - int(config.diffusion_step_embed_dim)
        if self.global_condition_dim < 1:
            raise ValueError("Could not infer a valid Diffusion Policy conditioning dimension.")
        self.tactile_dim = int(tactile_dim)
        self.tactile_adapter = nn.Linear(self.tactile_dim, self.global_condition_dim, bias=False)
        nn.init.zeros_(self.tactile_adapter.weight)
        self.value = ValueNetwork(
            self.global_condition_dim + self.tactile_dim,
            hidden_dims=value_hidden_dims,
        )

        self.fine_tune_denoising_steps = int(fine_tune_denoising_steps)
        # DPPO's reference implementation deliberately separates the noise
        # injected into the sampled reverse chain from the wider numerical
        # floor used to evaluate transition log probabilities.  Keeping the
        # sampling floor near native DDPM inference preserves the pretrained
        # absolute-action policy before any online update.
        self.min_sampling_denoising_std = float(min_sampling_denoising_std)
        self.min_logprob_denoising_std = float(min_logprob_denoising_std)
        self.learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.value_clip = float(value_clip)
        self.grad_clip = float(grad_clip)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.update_epochs = int(update_epochs)
        self.minibatches = int(minibatches)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.optimizer = torch.optim.Adam(
            [
                *self.actor_ft.parameters(),
                *self.tactile_adapter.parameters(),
                *self.value.parameters(),
            ],
            lr=self.learning_rate,
        )
        self.optimizer_steps = 0

    @property
    def device(self) -> torch.device:
        return next(self.actor_ft.parameters()).device

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_observation(
        self, processed_history: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Return the detached denoiser condition and image-only x/y anchor."""
        self.base_policy.eval()
        with torch.no_grad():
            condition, visual_xy = (
                self.base_policy.diffusion.prepare_global_conditioning_and_visual_xy(
                    processed_history
                )
            )
        condition = condition.detach()
        visual_xy = visual_xy.detach()
        if condition.ndim != 2 or condition.shape[-1] != self.global_condition_dim:
            raise ValueError(
                f"Expected global condition [B,{self.global_condition_dim}], got {tuple(condition.shape)}."
            )
        _finite("global condition", condition)
        if visual_xy.ndim != 2 or tuple(visual_xy.shape) != (condition.shape[0], 2):
            raise ValueError(f"Expected visual x/y [B,2], got {tuple(visual_xy.shape)}.")
        _finite("visual x/y", visual_xy)
        if not torch.equal(condition[:, -2:], visual_xy):
            raise RuntimeError("Visual x/y must be the final two global-condition coordinates.")
        return condition, visual_xy

    def condition_with_visual_xy(self, global_condition: Tensor, visual_xy: Tensor) -> Tensor:
        """Replace the condition anchor when the deployment-time lock is active."""
        if global_condition.ndim != 2 or global_condition.shape[-1] != self.global_condition_dim:
            raise ValueError("Invalid DPPO global condition shape.")
        if visual_xy.ndim != 2 or visual_xy.shape != (global_condition.shape[0], 2):
            raise ValueError("Invalid DPPO visual x/y shape.")
        condition = global_condition.clone()
        condition[:, -2:] = visual_xy.to(condition)
        return condition

    def conditioned(self, global_condition: Tensor, tactile: Tensor) -> Tensor:
        if global_condition.ndim != 2 or global_condition.shape[-1] != self.global_condition_dim:
            raise ValueError("Invalid DPPO global condition shape.")
        if tactile.ndim != 2 or tactile.shape[-1] != self.tactile_dim:
            raise ValueError("Invalid DPPO tactile shape.")
        if global_condition.shape[0] != tactile.shape[0]:
            raise ValueError("Condition and tactile batch sizes differ.")
        _finite("tactile", tactile)
        return global_condition + self.tactile_adapter(tactile.to(global_condition))

    def _schedule(self) -> Tensor:
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        return self.scheduler.timesteps

    def _previous_timesteps(self, timesteps: Tensor) -> Tensor:
        schedule = self.scheduler.timesteps.to(timesteps.device)
        previous = torch.full_like(timesteps, -1)
        for index, timestep in enumerate(schedule):
            mask = timesteps == timestep
            if bool(mask.any()):
                value = schedule[index + 1] if index + 1 < len(schedule) else -1
                previous[mask] = value
        if bool((previous == -1).logical_and(timesteps != schedule[-1]).any()):
            raise ValueError("Encountered a timestep outside the active DDPM inference schedule.")
        return previous

    def transition_mean_std(
        self,
        sample: Tensor,
        timesteps: Tensor,
        global_condition: Tensor,
        tactile: Tensor,
        *,
        use_base_policy: bool = False,
        minimum_std: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute q(x_prev | x_t, condition) for arbitrary scheduled timesteps."""
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(sample.shape[0])
        timesteps = timesteps.long().to(sample.device)
        actor = self.base_policy.diffusion.unet if use_base_policy else self.actor_ft
        # Keep the early reverse steps genuinely frozen. The trainable tactile
        # adapter conditions only the fine-tuned steps whose likelihoods are
        # retained and optimized by PPO.
        condition = (
            global_condition
            if use_base_policy
            else self.conditioned(global_condition, tactile)
        )
        model_output = actor(sample, timesteps, global_cond=condition)

        previous = self._previous_timesteps(timesteps)
        alphas = self.scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_t = alphas[timesteps]
        alpha_prev = torch.where(
            previous >= 0,
            alphas[previous.clamp_min(0)],
            torch.ones_like(alpha_t),
        )
        while alpha_t.ndim < sample.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_prev = alpha_prev.unsqueeze(-1)
        beta_t = 1.0 - alpha_t
        beta_prev = 1.0 - alpha_prev
        current_alpha = alpha_t / alpha_prev
        current_beta = 1.0 - current_alpha

        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            pred_original = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        elif prediction_type == "sample":
            pred_original = model_output
        else:
            raise ValueError(f"Unsupported DDPM prediction type: {prediction_type}")
        if self.scheduler.config.thresholding:
            pred_original = self.scheduler._threshold_sample(pred_original)
        elif self.scheduler.config.clip_sample:
            limit = float(self.scheduler.config.clip_sample_range)
            pred_original = pred_original.clamp(-limit, limit)

        mean = (
            alpha_prev.sqrt() * current_beta / beta_t * pred_original
            + current_alpha.sqrt() * beta_prev / beta_t * sample
        )
        variance = (beta_prev / beta_t * current_beta).clamp_min(1.0e-20)
        floor = self.min_logprob_denoising_std if minimum_std is None else float(minimum_std)
        std = variance.sqrt().clamp_min(floor)
        _finite("DDPM transition mean", mean)
        _finite("DDPM transition std", std)
        return mean, std

    def transition_log_prob(
        self,
        sample: Tensor,
        next_sample: Tensor,
        timesteps: Tensor,
        global_condition: Tensor,
        tactile: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mean, std = self.transition_mean_std(
            sample,
            timesteps,
            global_condition,
            tactile,
        )
        distribution = Normal(mean, std)
        elementwise = distribution.log_prob(next_sample)
        # DPPO averages over the action horizon and dimensions before forming
        # the likelihood ratio, matching the reference implementation.
        log_prob = elementwise.mean(dim=(-1, -2))
        entropy = distribution.entropy().mean(dim=(-1, -2))
        _finite("DDPM transition log probability", log_prob)
        return log_prob, entropy

    @torch.no_grad()
    def sample(
        self,
        global_condition: Tensor,
        tactile: Tensor,
        visual_xy: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Sample an action and retain each trainable reverse-diffusion transition."""
        self.base_policy.eval()
        batch_size = global_condition.shape[0]
        if visual_xy.ndim != 2 or visual_xy.shape != (batch_size, 2):
            raise ValueError("visual_xy must have shape [B,2].")
        if not torch.equal(global_condition[:, -2:], visual_xy.to(global_condition)):
            raise ValueError("The execution anchor must match the DPPO global condition.")
        condition = self.conditioned(global_condition, tactile)
        config = self.base_policy.config
        current = torch.randn(
            (batch_size, int(config.horizon), int(config.action_feature.shape[0])),
            device=self.device,
            dtype=global_condition.dtype,
            generator=generator,
        )
        schedule = self._schedule()
        fine_start = len(schedule) - self.fine_tune_denoising_steps
        previous_chain: list[Tensor] = []
        next_chain: list[Tensor] = []
        used_timesteps: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        denoising_steps = 0
        for index, timestep in enumerate(schedule):
            timestep_batch = torch.full(
                (batch_size,), int(timestep.item()), device=self.device, dtype=torch.long
            )
            use_base = index < fine_start
            mean, std = self.transition_mean_std(
                current,
                timestep_batch,
                global_condition,
                tactile,
                use_base_policy=use_base,
                minimum_std=self.min_sampling_denoising_std,
            )
            noise = torch.randn(
                current.shape,
                device=current.device,
                dtype=current.dtype,
                generator=generator,
            )
            following = mean + std * noise
            denoising_steps += 1
            if not use_base:
                previous_chain.append(current)
                next_chain.append(following)
                used_timesteps.append(timestep_batch)
                logprob_std = std.clamp_min(self.min_logprob_denoising_std)
                old_log_probs.append(
                    Normal(mean, logprob_std).log_prob(following).mean(dim=(-1, -2))
                )
            current = following

        start = int(config.n_obs_steps) - 1
        end = start + int(config.n_action_steps)
        residual_action = current[:, start:end]
        action = self.base_policy.diffusion.visual_residual_to_action(
            residual_action, visual_xy.to(residual_action)
        ).clamp(-1.0, 1.0)
        value = self.value(global_condition, tactile)
        result = {
            "action": action,
            "residual_action": residual_action,
            "chain_previous": torch.stack(previous_chain, dim=1),
            "chain_next": torch.stack(next_chain, dim=1),
            "timesteps": torch.stack(used_timesteps, dim=1),
            "old_log_probs": torch.stack(old_log_probs, dim=1),
            "value": value,
            "denoising_steps": torch.tensor(denoising_steps, device=self.device),
        }
        for name, value_tensor in result.items():
            _finite(name, value_tensor.float())
        return result

    def predict_value(self, global_condition: Tensor, tactile: Tensor) -> Tensor:
        return self.value(global_condition, tactile)

    def _advantages(
        self,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        bootstrap_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        for index in reversed(range(len(rewards))):
            next_value = bootstrap_value.reshape(()) if index == len(rewards) - 1 else values[index + 1]
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + self.gamma * next_value * nonterminal - values[index]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[index] = last_gae
        return advantages, advantages + values

    def update(self, rollout: DPPORollout, bootstrap_value: Tensor | float) -> DPPODiagnostics:
        """Run clipped PPO over environment × denoising transitions."""
        data = rollout.tensors(self.device)
        advantages, returns = self._advantages(
            data["rewards"],
            data["dones"],
            data["values"],
            torch.as_tensor(bootstrap_value, device=self.device, dtype=torch.float32),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
        rollout_count, denoising_count = data["timesteps"].shape
        environment_indices = torch.arange(rollout_count, device=self.device).repeat_interleave(
            denoising_count
        )
        denoising_indices = torch.arange(denoising_count, device=self.device).repeat(rollout_count)
        total = rollout_count * denoising_count
        minibatch_size = math.ceil(total / self.minibatches)
        metrics: list[dict[str, float]] = []

        for _ in range(self.update_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                selection = permutation[start : start + minibatch_size]
                env_index = environment_indices[selection]
                denoise_index = denoising_indices[selection]
                previous = data["chain_previous"][env_index, denoise_index]
                following = data["chain_next"][env_index, denoise_index]
                timestep = data["timesteps"][env_index, denoise_index]
                old_log_prob = data["old_log_probs"][env_index, denoise_index]
                new_log_prob, entropy = self.transition_log_prob(
                    previous,
                    following,
                    timestep,
                    data["global_condition"][env_index],
                    data["tactile"][env_index],
                )
                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()
                advantage = advantages[env_index]
                unclipped = -advantage * ratio
                clipped = -advantage * ratio.clamp(
                    1.0 - self.clip_range, 1.0 + self.clip_range
                )
                actor_loss = torch.maximum(unclipped, clipped).mean()

                new_value = self.value(
                    data["global_condition"][env_index], data["tactile"][env_index]
                )
                old_value = data["values"][env_index]
                value_clipped = old_value + (new_value - old_value).clamp(
                    -self.value_clip, self.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    (new_value - returns[env_index]).square(),
                    (value_clipped - returns[env_index]).square(),
                ).mean()
                loss = (
                    actor_loss
                    + self.value_loss_coefficient * value_loss
                    - self.entropy_coefficient * entropy.mean()
                )
                _finite("DPPO loss", loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        *self.actor_ft.parameters(),
                        *self.tactile_adapter.parameters(),
                        *self.value.parameters(),
                    ],
                    self.grad_clip,
                )
                self.optimizer.step()
                self.optimizer_steps += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                metrics.append(
                    {
                        "actor_loss": float(actor_loss.detach()),
                        "value_loss": float(value_loss.detach()),
                        "approx_kl": float(approx_kl.detach()),
                        "ratio_mean": float(ratio.mean().detach()),
                        "clip_fraction": float(clip_fraction.detach()),
                        "entropy": float(entropy.mean().detach()),
                        "grad_norm": float(torch.as_tensor(grad_norm).detach()),
                    }
                )

        rollout.clear()
        if not metrics:
            return DPPODiagnostics()
        return DPPODiagnostics(
            optimizer_steps=len(metrics),
            denoising_transitions=total,
            **{
                key: sum(row[key] for row in metrics) / len(metrics)
                for key in (
                    "actor_loss",
                    "value_loss",
                    "approx_kl",
                    "ratio_mean",
                    "clip_fraction",
                    "entropy",
                    "grad_norm",
                )
            },
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "contract_version": TACTILE_DPPO_CONTRACT_VERSION,
            "actor_ft": self.actor_ft.state_dict(),
            "tactile_adapter": self.tactile_adapter.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "config": {
                "tactile_dim": self.tactile_dim,
                "global_condition_dim": self.global_condition_dim,
                "action_coordinates": "visual_xy_anchor_plus_diffusion_residual_v1",
                "fine_tune_denoising_steps": self.fine_tune_denoising_steps,
                "num_inference_steps": self.num_inference_steps,
                "min_sampling_denoising_std": self.min_sampling_denoising_std,
                "min_logprob_denoising_std": self.min_logprob_denoising_std,
                "learning_rate": self.learning_rate,
                "clip_range": self.clip_range,
                "value_clip": self.value_clip,
                "grad_clip": self.grad_clip,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "update_epochs": self.update_epochs,
                "minibatches": self.minibatches,
            },
        }

    def load_checkpoint_payload(self, payload: dict[str, Any], *, load_optimizer: bool = True) -> None:
        received = payload.get("contract_version")
        if received != TACTILE_DPPO_CONTRACT_VERSION:
            raise ValueError(
                f"Incompatible DPPO checkpoint contract {received!r}; "
                f"expected {TACTILE_DPPO_CONTRACT_VERSION!r}."
            )
        self.actor_ft.load_state_dict(payload["actor_ft"], strict=True)
        self.tactile_adapter.load_state_dict(payload["tactile_adapter"], strict=True)
        self.value.load_state_dict(payload["value"], strict=True)
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.optimizer_steps = int(payload.get("optimizer_steps", 0))


__all__ = [
    "TACTILE_DPPO_CONTRACT_VERSION",
    "DPPODiagnostics",
    "DPPORollout",
    "TactileDPPO",
]
