# Force-constrained GrapePick results

The upright-vine, horizontal-grasp task was evaluated with Isaac Sim 4.5,
Isaac Lab 2.1.x, and the dual-GelSight Panda rigid-pad asset. Grip force always
comes from the two physical pad-to-grape `ContactSensor`s.

## Tactile BC dataset and model

- Dataset: `datasets/grape_pick_horizontal_tactile_bc_256`.
- Schema-v2 demonstrations: 256/256 successful.
- Training/validation split: 205/51 complete episodes.
- BC inputs include the instantaneous left and right pad-normal forces.
- Best validation normalized MSE: `1.0906e-4`.
- Peak expert grip force: 3.518–3.533 N (mean 3.519 N).
- Peak expert stem tension: 3.052–3.137 N (mean 3.075 N).
- Checkpoint: `exp_report/compare_exp_100_with_tactile/grape_pick/checkpoints/tactile_bc.pt`.
- Checkpoint SHA-256: `7ff984c4b66ecc195cfde2afd0614848e9b89f518cfeec71743d68e22ef3fa08`.

The closed-loop evaluation randomizes the grape damage limit over 3.4–5.0 N
and the stem-detachment force over 2.8–3.4 N. On three paired, unseen 100-episode
sets, the base BC obtains 199/300 = **66.3%**. Its failures include 29 crushed
grapes and 36 stems that remain attached, so the calibrated baseline exhibits
both required failure modes.

## Formal tactile-policy comparison

Every method uses the same frozen tactile BC and the same 11-D observation,
including left/right pad-normal loads at indices 6 and 7. Each RL method uses
100 training episodes and a separate 40-episode validation set for checkpoint
selection. Final evaluation uses paired seeds 50000–50099, 60000–60099, and
70000–70099.

| Method | Success | Rate | Main failures |
|---|---:|---:|---|
| Tactile BC | 199/300 | 66.3% | crushed 29, no-detach 36, post-detach timeout 36 |
| Residual RL | 296/300 | 98.7% | 4 total |
| AWR | 245/300 | 81.7% | 55 total |
| DPPO | 209/300 | 69.7% | 91 total |
| DSRL | 287/300 | 95.7% | 13 total |
| **VLM+DSRL** | **299/300** | **99.7%** | no-detach 1 |

Residual RL, DSRL, and VLM+DSRL all strictly outperform the base policy.
VLM+DSRL is also strictly best overall.

The VLM result uses a real local `Qwen3-VL-4B-Instruct` inference artifact,
not a hard-coded label. It recommends a 2.8–3.3 N grip interval. The tactile
safety layer operates with a 3.2 N upper control target, rate-limits target
motion near contact, and obtains a 3.208 N maximum grip load over the 300
formal episodes. Tough stems are handled by extra downward pull rather than
extra squeeze.

## Whole-arm collision audit

An intermediate audit exposed that the original task-level collision flag only
covered the gripper: `panda_link5` could hit `trellis_left` without being
reported. The final scene keeps physical trellis colliders, moves the visual
post behind the elbow corridor, and monitors all eight Panda links plus the
seven hand/finger/GelSight bodies. The maximum robot-to-vine/trellis contact
load is 0.000 N for every method in all 1,800 formal evaluation episodes.

Machine-readable evidence, individual episode outcomes, checkpoints, VLM
prompt/response provenance, and acceptance checks are under
`exp_report/compare_exp_100_with_tactile/grape_pick/formal_v2/`.
