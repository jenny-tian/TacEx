# Clean LabPick residual SAC

This is the small, independent baseline intended for code review. It keeps the
name SAC for continuity with the project, but deliberately uses an alpha-zero
objective: the policy is stochastic and reparameterized, while entropy is
absent from both losses and there is no temperature optimizer.

## One fixed contract

- Frozen policy: Flow Matching BC.
- BC prediction horizon: 32 actions; execute only the first 10, then replan.
- One RL step: one BC action held for two 120 Hz physics steps.
- Rotation: reset-time simulator ground-truth yaw. BC Rot6D is not executed and
  the Actor cannot change rotation.
- Actor observation (29D): normalized proprioception 10 + relative object
  position 3 + live object Rot6D 6 + current normalized BC action 10.
- Actor action (4D): post-tanh raw residual for x, y, z, and gripper width.
- Execution: add `0.15 * residual` to normalized BC indices `[0, 1, 2, 9]`,
  leave Rot6D unchanged, then unnormalize once.
- Critic input (33D): privileged state 19 + complete BC action 10 + raw
  residual 4.
- Reward: the existing dense LabPick reward (reach/lift progress, contact,
  success/failure, timeout, and force terms) without any wrapper- or
  agent-side shaping. Timeouts are terminal for bootstrapping.

The frozen BC retains its deployed visual-XY override and locks visual XY
after phase `0.30`; both choices are pinned and recorded in the run metadata.

There is one tanh-squashed Gaussian policy from the first transition onward.
There is no random-action phase or separate warm-up noise distribution.

## Update equations

For one replay transition, the online policy samples

```text
r' ~ tanh(N(mu(o'), sigma(o')))
```

and the one-step twin-Q target is

```text
y = R + gamma * (1 - terminated) * min(Q1_target(s', BC', r'),
                                             Q2_target(s', BC', r'))
```

The losses are

```text
L_critic = 0.5 * (MSE(Q1, y) + MSE(Q2, y))
L_actor  = -mean(min(Q1(s, BC, r_pi), Q2(s, BC, r_pi)))
```

Only the two target Critics are Polyak averaged. There is no target Actor,
entropy backup, entropy Actor term, learned alpha, H-step return, extra
agent-side potential target, residual trust-region loss, staged Actor delay,
strict-zero episode mixture, or external exploration noise.

Exploration therefore comes from the Actor's learned Gaussian only. The v1
initialization uses `log_std=-3` (raw standard deviation about `0.05`). The
29D observation deliberately omits the Flow history and chunk offset, so this
baseline is compact but not a fully Markov representation of the frozen BC.

## Train

First materialize the Flow checkpoint (including its Git LFS object), then run:

```bash
OMNI_KIT_ACCEPT_EULA=yes \
TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/train_lab_pick_clean_residual_sac.py \
  --bc_policy outputs/lab_pick_flow_bc100_scratch_rawclose_safe70_overforce24_pos6/best.pt \
  --residual_scale 0.15 \
  --timesteps 50000 \
  --seed 42
```

The dedicated task is
`TacEx-LabPick-Slide-Clean-Residual-SAC-v0`. Its config is
`source/tacex_tasks/tacex_tasks/lab_pick/agents/skrl_clean_residual_sac_cfg.yaml`.
Runtime logs include `params/clean_residual_sac.yaml`, which records the exact
action, target, and entropy contracts.
