# Clean LabPick DSRL-SAC

This path implements diffusion-policy reinforcement learning for the
repository's frozen Flow Matching policy. It follows the low-dimensional noise
action and SAC objective in `/home/limx/vlarl/src/dsrl`, while reusing the task,
camera, reward, and checkpoint conventions of the clean residual-RL path.

## Absolute-noise v2 contract

- Frozen policy: Flow Matching BC with a `32 x 10` initial-noise tensor.
- Actor observation: the frozen Flow encoder's 3072-D `global_cond` feature.
- Actor action: a tanh-squashed Gaussian `learned_noise_steps x 10` absolute
  noise tensor in `[-1, 1]`. It is not a delta and is never combined with a
  Gaussian base-noise template.
- The default Actor learns one noise row and repeats its last row over all 32
  decoder steps (`1 x 10 -> repeat_last -> 32 x 10`). No additional noise
  scaling is applied.
- The mean head has zero weights and bias, and the initial `log_std` is exactly
  zero. Therefore every initial observation has zero mean and the initial
  deterministic action is exactly zero.
- Critic input: Flow feature 3072 + privileged simulator state 19 + the same
  low-dimensional absolute noise action used by the decoder.
- Privileged state: normalized proprioception 10 + relative object position 3
  + object Rot6D 6. It is never supplied to the Actor.
- One outer transition: decode a complete Flow chunk and execute its configured
  prefix. Rewards within the prefix are discounted with `chunk_discount`; the
  SAC bootstrap discount is `chunk_discount ** chunk_execute_steps`.
- Time limits are terminal for bootstrapping. No wrapper- or agent-side reward
  shaping is added.

The persistent policy contract version is 2. Checkpoints produced by the old
residual-noise implementation do not contain this marker and are rejected
instead of being interpreted with the new action semantics.

## SAC objective

The online policy samples a low-dimensional absolute noise action `z`. With
twin target critics, the default target and losses are:

```text
y = R_chunk + gamma_outer * (1 - terminal) * min(Q1_target(s', z'), Q2_target(s', z'))
L_critic = 0.5 * (MSE(Q1, y) + MSE(Q2, y))
L_actor = mean(alpha * log pi(z | o) - min(Q1(s, z), Q2(s, z)))
L_alpha = -mean(log(alpha) * (log pi(z | o) + target_entropy))
```

This matches the reference's entropy-bearing Actor loss, learned temperature,
and entropy-free critic backup. `--backup_entropy` enables the conventional SAC
entropy term in the target as an explicit alternative.

## Train

```bash
TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/train_lab_pick_clean_dsrl_sac.py \
  --bc_policy outputs/lab_pick_flow_bc100_scratch_rawclose_safe70_overforce24_pos6/best.pt \
  --learned_noise_steps 1 \
  --chunk_execute_steps 32 \
  --actor_lr 3e-6 --critic_lr 3e-5 --alpha_lr 3e-5 \
  --timesteps 500 --seed 42
```

The dedicated task is `TacEx-LabPick-Slide-Clean-DSRL-SAC-v0`. Each run writes
`params/clean_dsrl_sac.yaml` with the v2 action contract, effective dimensions,
discounts, objective switches, and BC checkpoint hash. Run names include
`absolute` to distinguish them from legacy residual-noise experiments.

## Evaluate

Native BC, all-zero absolute noise, and a deterministic v2 checkpoint can be
evaluated with the same randomized seeds:

```bash
TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/eval_lab_pick_clean_dsrl_sac.py \
  --bc_policy /path/to/best.pt \
  --mode native_bc \
  --chunk_execute_steps 32 --num_trials 20 --seed 42 \
  --output logs/clean_dsrl_experiments/native_bc.json

TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/eval_lab_pick_clean_dsrl_sac.py \
  --bc_policy /path/to/best.pt \
  --mode zero_noise \
  --chunk_execute_steps 32 --num_trials 20 --seed 42 \
  --output logs/clean_dsrl_experiments/zero_absolute_noise.json

TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/eval_lab_pick_clean_dsrl_sac.py \
  --bc_policy /path/to/best.pt --checkpoint /path/to/agent_200.pt \
  --mode deterministic \
  --chunk_execute_steps 32 --num_trials 20 --seed 42 \
  --output logs/clean_dsrl_experiments/dsrl_absolute_step200.json
```

`native_bc` keeps the Flow policy's native Gaussian-noise behavior.
`zero_noise` instead decodes from a full `32 x 10` tensor of zero absolute
noise. Use the same seed interval for every condition.
