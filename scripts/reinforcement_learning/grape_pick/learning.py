"""Small tactile residual learners used by the GrapePick comparison.

The policies operate on three bounded residual coordinates (tool X, tool Z and
gripper width).  The frozen behavior-cloning policy remains responsible for
the full 10-D CAFE action and wrist orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def exploration_scale_schedule(
    start: float,
    end: float,
    step: int,
    total_steps: int,
    schedule: str,
) -> float:
    """Return a bounded external multiplier for SAC behavior-policy noise."""

    start = float(start)
    end = float(end)
    step = int(step)
    total_steps = int(total_steps)
    if start < 0.0 or end < 0.0:
        raise ValueError("Exploration scales must be non-negative.")
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}.")
    if schedule not in ("constant", "linear", "cosine", "exponential"):
        raise ValueError(f"Unknown exploration schedule {schedule!r}.")
    if schedule == "constant" or total_steps == 1:
        return start

    progress = step / float(total_steps - 1)
    if schedule == "linear":
        return start + progress * (end - start)
    if schedule == "cosine":
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end + (start - end) * weight
    if start <= 0.0 or end <= 0.0:
        raise ValueError("Exponential exploration decay requires positive endpoints.")
    return start * math.exp(math.log(end / start) * progress)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        initial_log_std: float = -2.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.network = mlp(observation_dim, 2 * action_dim, hidden_dim=hidden_dim)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[action_dim:] = float(initial_log_std)

    def distribution(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(observation).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 1.0)

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(observation)
        std = log_std.exp()
        raw = mean if deterministic else mean + std * torch.randn_like(mean)
        action = torch.tanh(raw)
        normal_log_prob = -0.5 * (
            ((raw - mean) / std).square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        correction = torch.log(1.0 - action.square() + 1.0e-6)
        return action, (normal_log_prob - correction).sum(dim=-1)

    def log_prob(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        clipped = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        raw = torch.atanh(clipped)
        mean, log_std = self.distribution(observation)
        std = log_std.exp()
        normal_log_prob = -0.5 * (
            ((raw - mean) / std).square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        correction = torch.log(1.0 - clipped.square() + 1.0e-6)
        return (normal_log_prob - correction).sum(dim=-1)

    def deterministic_action_with_detached_features(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        """Actor mean whose auxiliary gradients are confined to the output head."""

        features = self.network[:-1](observation).detach()
        mean, _ = self.network[-1](features).chunk(2, dim=-1)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = int(action_dim)
        self.context_dim = 0
        self.network = mlp(observation_dim + action_dim, 1, hidden_dim=hidden_dim)
        self.context_network: nn.Sequential | None = None

    def enable_context(self, context_dim: int, hidden_dim: int = 128) -> None:
        """Add a zero-output context adapter without perturbing the base Q path."""

        self.context_dim = int(context_dim)
        if self.context_dim <= 0:
            return
        self.context_network = mlp(
            self.context_dim,
            1,
            hidden_dim=hidden_dim,
        )
        nn.init.zeros_(self.context_network[-1].weight)
        nn.init.zeros_(self.context_network[-1].bias)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        expected_dim = self.action_dim + self.context_dim
        if action.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected critic action dim {expected_dim}, got {action.shape[-1]}."
            )
        base_action = action[..., : self.action_dim]
        value = self.network(torch.cat((observation, base_action), dim=-1))
        if self.context_network is not None:
            # The adapter represents the task-level value shift induced by the
            # VLM action context. Keeping it independent of the residual action
            # preserves the base critic's action ranking and its actor gradient.
            value = value + self.context_network(action[..., self.action_dim :])
        return value.squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = mlp(observation_dim, 1, hidden_dim=hidden_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


@dataclass
class SACBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor
    action_context: torch.Tensor | None = None
    next_action_context: torch.Tensor | None = None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        *,
        action_context_dim: int = 0,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.action_context_dim = int(action_context_dim)
        self.observation = torch.empty((capacity, observation_dim), dtype=torch.float32)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32)
        self.reward = torch.empty(capacity, dtype=torch.float32)
        self.next_observation = torch.empty((capacity, observation_dim), dtype=torch.float32)
        self.done = torch.empty(capacity, dtype=torch.float32)
        self.action_context = (
            torch.empty((capacity, self.action_context_dim), dtype=torch.float32)
            if self.action_context_dim > 0
            else None
        )
        self.next_action_context = (
            torch.empty((capacity, self.action_context_dim), dtype=torch.float32)
            if self.action_context_dim > 0
            else None
        )
        self.size = 0
        self.position = 0

    def add(self, batch: SACBatch) -> int:
        tensors = (
            batch.observation.detach().cpu(),
            batch.action.detach().cpu(),
            batch.reward.detach().cpu(),
            batch.next_observation.detach().cpu(),
            batch.done.detach().cpu(),
        )
        action_context = (
            batch.action_context.detach().cpu()
            if batch.action_context is not None
            else None
        )
        next_action_context = (
            batch.next_action_context.detach().cpu()
            if batch.next_action_context is not None
            else None
        )
        count = len(tensors[0])
        if count >= self.capacity:
            tensors = tuple(value[-self.capacity :] for value in tensors)
            if action_context is not None:
                action_context = action_context[-self.capacity :]
            if next_action_context is not None:
                next_action_context = next_action_context[-self.capacity :]
            count = self.capacity
        indices = (torch.arange(count) + self.position) % self.capacity
        self.observation[indices] = tensors[0]
        self.action[indices] = tensors[1]
        self.reward[indices] = tensors[2]
        self.next_observation[indices] = tensors[3]
        self.done[indices] = tensors[4]
        if self.action_context_dim > 0:
            if action_context is None or next_action_context is None:
                raise ValueError("Action-conditioned replay requires current and next action context.")
            assert self.action_context is not None
            assert self.next_action_context is not None
            if (
                action_context.shape != (count, self.action_context_dim)
                or next_action_context.shape != (count, self.action_context_dim)
            ):
                raise ValueError(
                    f"Expected action contexts with shape ({count}, "
                    f"{self.action_context_dim}); got {tuple(action_context.shape)} and "
                    f"{tuple(next_action_context.shape)}."
                )
            self.action_context[indices] = action_context
            self.next_action_context[indices] = next_action_context
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)
        return int(indices[-1].item())

    def add_reward(self, index: int, delta: float) -> None:
        """Add delayed credit to one still-resident replay transition."""

        index = int(index)
        if not 0 <= index < self.capacity:
            raise IndexError(f"replay index {index} is outside capacity {self.capacity}")
        if self.size < self.capacity and index >= self.size:
            raise IndexError(f"replay index {index} has not been initialized")
        self.reward[index] += float(delta)

    def sample(self, batch_size: int) -> SACBatch:
        indices = torch.randint(self.size, (batch_size,))
        action_context = (
            self.action_context[indices].to(self.device)
            if self.action_context is not None
            else None
        )
        next_action_context = (
            self.next_action_context[indices].to(self.device)
            if self.next_action_context is not None
            else None
        )
        return SACBatch(
            self.observation[indices].to(self.device),
            self.action[indices].to(self.device),
            self.reward[indices].to(self.device),
            self.next_observation[indices].to(self.device),
            self.done[indices].to(self.device),
            action_context,
            next_action_context,
        )

    def state_dict(self) -> dict:
        """Serialize only initialized replay slots for resumable online training."""

        state = {
            "capacity": self.capacity,
            "action_context_dim": self.action_context_dim,
            "size": self.size,
            "position": self.position,
            "observation": self.observation[: self.size].clone(),
            "action": self.action[: self.size].clone(),
            "reward": self.reward[: self.size].clone(),
            "next_observation": self.next_observation[: self.size].clone(),
            "done": self.done[: self.size].clone(),
        }
        if self.action_context is not None:
            assert self.next_action_context is not None
            state["action_context"] = self.action_context[: self.size].clone()
            state["next_action_context"] = self.next_action_context[: self.size].clone()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore replay storage while validating its observation/action contract."""

        size = int(state["size"])
        position = int(state["position"])
        if size < 0 or size > self.capacity:
            raise ValueError(f"replay size {size} exceeds capacity {self.capacity}")
        if not 0 <= position < self.capacity:
            raise ValueError(f"invalid replay position {position}")
        if int(state.get("action_context_dim", 0)) != self.action_context_dim:
            raise ValueError("replay action-context dimension mismatch")
        tensors = {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "done": self.done,
        }
        for name, destination in tensors.items():
            source = torch.as_tensor(state[name]).cpu()
            if source.shape != destination[:size].shape:
                raise ValueError(
                    f"replay {name} shape mismatch: expected "
                    f"{tuple(destination[:size].shape)}, got {tuple(source.shape)}"
                )
            destination[:size].copy_(source)
        if self.action_context_dim > 0:
            assert self.action_context is not None and self.next_action_context is not None
            for name, destination in (
                ("action_context", self.action_context),
                ("next_action_context", self.next_action_context),
            ):
                source = torch.as_tensor(state[name]).cpu()
                if source.shape != destination[:size].shape:
                    raise ValueError(
                        f"replay {name} shape mismatch: expected "
                        f"{tuple(destination[:size].shape)}, got {tuple(source.shape)}"
                    )
                destination[:size].copy_(source)
        self.size = size
        self.position = position


class SACResidualAgent:
    """Soft actor-critic over bounded residual actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        device: torch.device,
        *,
        learning_rate: float = 3.0e-4,
        gamma: float = 0.995,
        tau: float = 0.01,
        residual_l2_weight: float = 0.5,
        initial_log_std: float = -2.0,
        initial_alpha: float = 0.05,
        target_entropy: float | None = None,
        critic_action_context_dim: int = 0,
        critic_context_mode: str = "concat",
        executed_action_mask: tuple[float, ...] | None = None,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.residual_l2_weight = float(residual_l2_weight)
        self.action_dim = int(action_dim)
        self.critic_action_context_dim = int(critic_action_context_dim)
        if critic_context_mode not in ("concat", "bias_adapter"):
            raise ValueError(f"Unknown critic context mode {critic_context_mode!r}.")
        self.critic_context_mode = critic_context_mode
        if executed_action_mask is None:
            executed_action_mask = (1.0,) * self.action_dim
        if len(executed_action_mask) != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} executed-action mask values, "
                f"got {len(executed_action_mask)}."
            )
        self.executed_action_mask = torch.tensor(
            executed_action_mask, dtype=torch.float32, device=device
        )
        self.actor = SquashedGaussianActor(
            observation_dim,
            action_dim,
            initial_log_std=initial_log_std,
        ).to(device)
        use_context_adapter = (
            self.critic_action_context_dim > 0
            and self.critic_context_mode == "bias_adapter"
        )
        critic_action_dim = (
            action_dim
            if use_context_adapter
            else action_dim + self.critic_action_context_dim
        )
        self.q1 = QNetwork(observation_dim, critic_action_dim).to(device)
        self.q2 = QNetwork(observation_dim, critic_action_dim).to(device)
        self.target_q1 = QNetwork(observation_dim, critic_action_dim).to(device)
        self.target_q2 = QNetwork(observation_dim, critic_action_dim).to(device)
        if use_context_adapter:
            cuda_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(17_291)
                for network in (self.q1, self.q2, self.target_q1, self.target_q2):
                    network.enable_context(self.critic_action_context_dim)
                    network.to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=learning_rate
        )
        self.log_alpha = torch.tensor(
            math.log(float(initial_alpha)), device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = (
            -float(action_dim) if target_entropy is None else float(target_entropy)
        )
        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _critic_action(
        self,
        actor_action: torch.Tensor,
        action_context: torch.Tensor | None,
    ) -> torch.Tensor:
        executed_action = actor_action * self.executed_action_mask
        if self.critic_action_context_dim == 0:
            if action_context is not None and action_context.shape[-1] != 0:
                raise ValueError("This SAC critic was created without action context.")
            return executed_action
        if action_context is None:
            raise ValueError("SAC critic action context is required but missing.")
        if action_context.shape[-1] != self.critic_action_context_dim:
            raise ValueError(
                f"Expected critic action context dim {self.critic_action_context_dim}, "
                f"got {action_context.shape[-1]}."
            )
        return torch.cat((executed_action, action_context), dim=-1)

    @torch.inference_mode()
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool,
        *,
        exploration_scale: float = 1.0,
    ) -> torch.Tensor:
        if deterministic:
            action, _ = self.actor.sample(observation, deterministic=True)
            return action
        mean, log_std = self.actor.distribution(observation)
        raw = mean + float(exploration_scale) * log_std.exp() * torch.randn_like(mean)
        return torch.tanh(raw)

    def update(
        self,
        batch: SACBatch,
        *,
        action_prior: torch.Tensor | None = None,
        action_prior_mask: torch.Tensor | None = None,
        action_prior_weight: float = 0.0,
    ) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch.next_observation)
            next_critic_action = self._critic_action(
                next_action, batch.next_action_context
            )
            target_q = torch.minimum(
                self.target_q1(batch.next_observation, next_critic_action),
                self.target_q2(batch.next_observation, next_critic_action),
            ) - self.alpha.detach() * next_log_prob
            target = batch.reward + self.gamma * (1.0 - batch.done) * target_q

        replay_critic_action = self._critic_action(batch.action, batch.action_context)
        q1 = self.q1(batch.observation, replay_critic_action)
        q2 = self.q2(batch.observation, replay_critic_action)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_([*self.q1.parameters(), *self.q2.parameters()], 5.0)
        self.q_optimizer.step()

        sampled_action, log_prob = self.actor.sample(batch.observation)
        sampled_critic_action = self._critic_action(
            sampled_action, batch.action_context
        )
        actor_loss = (
            self.alpha.detach() * log_prob
            - torch.minimum(
                self.q1(batch.observation, sampled_critic_action),
                self.q2(batch.observation, sampled_critic_action),
            )
        ).mean() + self.residual_l2_weight * sampled_action.square().mean()
        prior_loss = torch.zeros((), device=self.device)
        if action_prior is not None and float(action_prior_weight) > 0.0:
            if action_prior.shape != sampled_action.shape:
                raise ValueError(
                    f"Expected action prior shape {tuple(sampled_action.shape)}, "
                    f"got {tuple(action_prior.shape)}."
                )
            if action_prior_mask is None:
                action_prior_mask = torch.ones_like(action_prior)
            if action_prior_mask.shape != sampled_action.shape:
                raise ValueError(
                    f"Expected action prior mask shape {tuple(sampled_action.shape)}, "
                    f"got {tuple(action_prior_mask.shape)}."
                )
            deterministic_action = self.actor.deterministic_action_with_detached_features(
                batch.observation
            )
            squared_error = (deterministic_action - action_prior.detach()).square()
            active_weight = action_prior_mask.detach().sum().clamp_min(1.0)
            prior_loss = (squared_error * action_prior_mask.detach()).sum() / active_weight
            actor_loss = actor_loss + float(action_prior_weight) * prior_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_parameter.lerp_(parameter, self.tau)
            for target_parameter, parameter in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        self.updates += 1
        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "action_prior_loss": float(prior_loss.item()),
        }

    def checkpoint(self) -> dict:
        return {
            "algorithm": "sac",
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "updates": self.updates,
            "critic_action_context_dim": self.critic_action_context_dim,
            "critic_context_mode": self.critic_context_mode,
            "executed_action_mask": self.executed_action_mask.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict) -> None:
        """Restore networks and, when available, optimizer state for continuation."""

        if checkpoint.get("algorithm") != "sac":
            raise ValueError("SAC checkpoint algorithm mismatch")
        if int(checkpoint.get("critic_action_context_dim", 0)) != self.critic_action_context_dim:
            raise ValueError("SAC critic action-context dimension mismatch")
        if str(checkpoint.get("critic_context_mode", "concat")) != self.critic_context_mode:
            raise ValueError("SAC critic context mode mismatch")
        saved_mask = torch.as_tensor(checkpoint["executed_action_mask"]).to(self.device)
        if not torch.equal(saved_mask, self.executed_action_mask):
            raise ValueError("SAC executed-action mask mismatch")
        self.actor.load_state_dict(checkpoint["actor"], strict=True)
        self.q1.load_state_dict(checkpoint["q1"], strict=True)
        self.q2.load_state_dict(checkpoint["q2"], strict=True)
        self.target_q1.load_state_dict(checkpoint["target_q1"], strict=True)
        self.target_q2.load_state_dict(checkpoint["target_q2"], strict=True)
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(checkpoint["log_alpha"]).to(self.device))
        self.updates = int(checkpoint["updates"])
        optimizer_pairs = (
            ("actor_optimizer", self.actor_optimizer),
            ("q_optimizer", self.q_optimizer),
            ("alpha_optimizer", self.alpha_optimizer),
        )
        for name, optimizer in optimizer_pairs:
            if name not in checkpoint:
                continue
            optimizer.load_state_dict(checkpoint[name])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)


class SafePullBanditAgent:
    """Low-data residual Q-learner over a bounded one-dimensional pull action.

    The force supervisor remains responsible for grip safety.  This learner
    selects one additional pull residual per episode and updates its action
    values only from observed episodic returns.  Action zero is always present
    and is the exact no-RL force-guard policy.
    """

    def __init__(
        self,
        device: torch.device,
        actions: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20),
    ):
        if not actions or actions[0] != 0.0:
            raise ValueError("Safe pull action set must begin with the zero-residual baseline.")
        if any(action < 0.0 or action > 1.0 for action in actions):
            raise ValueError("Safe pull residuals must lie in [0, 1].")
        self.device = device
        self.actions = torch.tensor(actions, dtype=torch.float32, device=device)
        self.q_values = torch.zeros_like(self.actions)
        self.counts = torch.zeros_like(self.actions, dtype=torch.long)
        self.updates = 0

    @torch.inference_mode()
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool,
        *,
        exploration_scale: float = 1.0,
    ) -> torch.Tensor:
        batch_size = len(observation)
        best_index = int(torch.argmax(self.q_values).item())
        indices = torch.full(
            (batch_size,), best_index, dtype=torch.long, device=self.device
        )
        if not deterministic and exploration_scale > 0.0:
            explore = torch.rand(batch_size, device=self.device) < float(exploration_scale)
            random_indices = torch.randint(
                len(self.actions), (batch_size,), device=self.device
            )
            indices = torch.where(explore, random_indices, indices)
        return self.actions[indices].unsqueeze(-1)

    @torch.no_grad()
    def update(self, action: torch.Tensor, episode_return: torch.Tensor) -> dict[str, float]:
        flat_action = action.reshape(-1).to(self.device)
        flat_return = episode_return.reshape(-1).to(self.device)
        distances = (flat_action[:, None] - self.actions[None, :]).abs()
        indices = distances.argmin(dim=1)
        for index in range(len(self.actions)):
            selected = indices == index
            count = int(selected.sum().item())
            if count == 0:
                continue
            batch_mean = flat_return[selected].mean()
            old_count = int(self.counts[index].item())
            new_count = old_count + count
            self.q_values[index] = (
                self.q_values[index] * old_count + batch_mean * count
            ) / new_count
            self.counts[index] = new_count
        self.updates += 1
        best_index = int(torch.argmax(self.q_values).item())
        return {
            "best_pull_residual": float(self.actions[best_index].item()),
            "best_q_value": float(self.q_values[best_index].item()),
            "sampled_actions": float(len(flat_action)),
        }

    def checkpoint(self) -> dict:
        return {
            "algorithm": "safe_pull_bandit_q_learning",
            "actions": self.actions.detach().cpu(),
            "q_values": self.q_values.detach().cpu(),
            "counts": self.counts.detach().cpu(),
            "updates": self.updates,
        }

    def load_checkpoint(self, checkpoint: dict) -> None:
        self.actions = checkpoint["actions"].to(self.device).float()
        self.q_values = checkpoint["q_values"].to(self.device).float()
        self.counts = checkpoint["counts"].to(self.device).long()
        self.updates = int(checkpoint.get("updates", 0))


class AWRAgent:
    """Advantage-weighted regression on collected tactile residual actions."""

    def __init__(self, observation_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.actor = SquashedGaussianActor(observation_dim, action_dim).to(device)
        self.value = ValueNetwork(observation_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3.0e-4)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3.0e-4)
        self.updates = 0

    @torch.inference_mode()
    def act(self, observation: torch.Tensor, deterministic: bool) -> torch.Tensor:
        action, _ = self.actor.sample(observation, deterministic=deterministic)
        return action

    def update(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        returns: torch.Tensor,
        *,
        epochs: int = 8,
        batch_size: int = 1024,
        beta: float = 1.0,
    ) -> dict[str, float]:
        observation = observation.to(self.device)
        action = action.to(self.device)
        returns = returns.to(self.device)
        last_actor = 0.0
        last_value = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(observation), device=self.device)
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                obs_b = observation[index]
                action_b = action[index]
                returns_b = returns[index]
                value = self.value(obs_b)
                value_loss = F.mse_loss(value, returns_b)
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                self.value_optimizer.step()

                with torch.no_grad():
                    advantage = returns_b - self.value(obs_b)
                    weights = torch.exp(advantage / beta).clamp(max=20.0)
                actor_loss = -(weights * self.actor.log_prob(obs_b, action_b)).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
                self.actor_optimizer.step()
                last_actor = float(actor_loss.item())
                last_value = float(value_loss.item())
                self.updates += 1
        return {"actor_loss": last_actor, "value_loss": last_value}

    def checkpoint(self) -> dict:
        return {
            "algorithm": "awr",
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "updates": self.updates,
        }


class DiffusionResidualPolicy(nn.Module):
    """Short stochastic denoising chain for DPPO residual actions."""

    def __init__(self, observation_dim: int, action_dim: int, denoising_steps: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.denoising_steps = denoising_steps
        self.denoiser = mlp(observation_dim + action_dim + 1, action_dim, hidden_dim=128)
        nn.init.zeros_(self.denoiser[-1].weight)
        nn.init.zeros_(self.denoiser[-1].bias)
        self.log_std = nn.Parameter(torch.tensor(-2.3))

    def transition_mean(self, observation: torch.Tensor, state: torch.Tensor, step: int) -> torch.Tensor:
        time = torch.full(
            (len(observation), 1),
            float(step + 1) / float(self.denoising_steps),
            dtype=observation.dtype,
            device=observation.device,
        )
        prediction = torch.tanh(self.denoiser(torch.cat((observation, state, time), dim=-1)))
        mix = 1.0 / float(step + 1)
        return (1.0 - mix) * state + mix * prediction

    def sample(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.zeros(
            (len(observation), self.action_dim), device=observation.device
        ) if deterministic else torch.randn(
            (len(observation), self.action_dim), device=observation.device
        )
        trace = [state]
        log_prob = torch.zeros(len(observation), device=observation.device)
        std = self.log_std.clamp(-4.0, -0.7).exp()
        for step in reversed(range(self.denoising_steps)):
            mean = self.transition_mean(observation, state, step)
            next_state = mean if deterministic else mean + std * torch.randn_like(mean)
            log_prob += (
                -0.5
                * (((next_state - mean) / std).square() + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
            ).sum(dim=-1)
            state = next_state
            trace.append(state)
        return torch.tanh(state), log_prob, torch.stack(trace, dim=1)

    def trace_log_prob(self, observation: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        state = trace[:, 0]
        log_prob = torch.zeros(len(observation), device=observation.device)
        std = self.log_std.clamp(-4.0, -0.7).exp()
        for trace_index, step in enumerate(reversed(range(self.denoising_steps))):
            mean = self.transition_mean(observation, state, step)
            next_state = trace[:, trace_index + 1]
            log_prob += (
                -0.5
                * (((next_state - mean) / std).square() + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
            ).sum(dim=-1)
            state = next_state
        return log_prob


class DiffusionPPOAgent:
    def __init__(self, observation_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.policy = DiffusionResidualPolicy(observation_dim, action_dim).to(device)
        self.value = ValueNetwork(observation_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [*self.policy.parameters(), *self.value.parameters()], lr=1.0e-4
        )
        self.updates = 0

    @torch.inference_mode()
    def act(
        self, observation: torch.Tensor, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.sample(observation, deterministic=deterministic)

    def update(
        self,
        observation: torch.Tensor,
        trace: torch.Tensor,
        old_log_prob: torch.Tensor,
        returns: torch.Tensor,
        advantage: torch.Tensor,
        *,
        epochs: int = 4,
        batch_size: int = 1024,
        ratio_clip: float = 0.2,
    ) -> dict[str, float]:
        observation = observation.to(self.device)
        trace = trace.to(self.device)
        old_log_prob = old_log_prob.to(self.device)
        returns = returns.to(self.device)
        advantage = advantage.to(self.device)
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1.0e-5)
        last_loss = 0.0
        last_kl = 0.0
        for _ in range(epochs):
            order = torch.randperm(len(observation), device=self.device)
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                new_log_prob = self.policy.trace_log_prob(observation[index], trace[index])
                log_ratio = (new_log_prob - old_log_prob[index]).clamp(-8.0, 8.0)
                ratio = log_ratio.exp()
                unclipped = ratio * advantage[index]
                clipped = ratio.clamp(1.0 - ratio_clip, 1.0 + ratio_clip) * advantage[index]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(self.value(observation[index]), returns[index])
                entropy_bonus = self.policy.log_std.clamp(-4.0, -0.7)
                loss = policy_loss + 0.5 * value_loss - 1.0e-3 * entropy_bonus
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [*self.policy.parameters(), *self.value.parameters()], 0.5
                )
                self.optimizer.step()
                last_loss = float(loss.item())
                last_kl = float((-log_ratio).mean().item())
                self.updates += 1
        return {"loss": last_loss, "approx_kl": last_kl}

    def checkpoint(self) -> dict:
        return {
            "algorithm": "diffusion_ppo",
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "denoising_steps": self.policy.denoising_steps,
            "updates": self.updates,
        }


def discounted_returns(
    reward: torch.Tensor, done: torch.Tensor, valid: torch.Tensor, gamma: float = 0.995
) -> torch.Tensor:
    """Compute masked returns for tensors shaped ``[time, env]``."""

    result = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for step in reversed(range(reward.shape[0])):
        running = reward[step] + gamma * running * (1.0 - done[step])
        running = torch.where(valid[step], running, torch.zeros_like(running))
        result[step] = running
    return result
