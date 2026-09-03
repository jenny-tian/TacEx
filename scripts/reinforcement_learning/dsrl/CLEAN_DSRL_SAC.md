# Base-anchored LabPick DSRL-SAC

This path fine-tunes the frozen Flow Matching policy through a bounded
correction to its native initial-noise sample.

## v4 contract

- Frozen policy prior: independent Gaussian `32 x 10` Flow noise.
- Actor input: frozen Flow encoder feature plus the causal 5-D tactile vector.
- Actor output: a tanh-bounded `learned_noise_steps x 10` correction. The last
  learned row is repeated over the remaining horizon and multiplied by
  `noise_residual_scale` (default `0.25`).
- Decoder noise: `native_gaussian + scale * learned_correction`.
- The correction policy starts at zero mean with `initial_log_std=-2`, avoiding
  a large random departure from the frozen policy before replay is populated.
- A zero action, or `noise_residual_scale=0`, is bitwise identical to the
  native frozen-BC noise path for the same generator state.
- Critic input: actor observation, privileged 19-D simulator state, and the
  low-dimensional correction. Privileged state is never supplied to the actor.
- One outer transition decodes and executes one 32-action prefix. The bootstrap
  discount is `chunk_discount ** chunk_execute_steps`.
- The SAC actor objective includes a configurable action-L2 base anchor
  (`action_l2_weight`, default `10.0`) in addition to entropy and twin-Q terms.
- Online optimization is capped at 500 gradient updates by default. The
  low-data critic otherwise drifts late in a 100-episode run and pulls the
  correction away from the validated base neighborhood.

The persistent policy contract version is 4. Older absolute-noise checkpoints
are rejected because their action has different semantics.

## Train

```bash
TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/train_lab_pick_clean_dsrl_sac.py \
  --bc_policy outputs/lab_pick_dinov3_flow_bc200_yaw0/best.pt \
  --learned_noise_steps 1 --noise_residual_scale 0.25 \
  --action_l2_weight 10.0 --max_gradient_updates 500 \
  --chunk_execute_steps 32 \
  --actor_lr 3e-5 --critic_lr 3e-5 --alpha_lr 3e-5 \
  --timesteps 500 --seed 42
```

## Evaluate

Use `native_bc` and `zero_noise` on the same seed interval to verify the
fallback invariant; under v4, `zero_noise` means zero correction and must match
`native_bc` exactly.
