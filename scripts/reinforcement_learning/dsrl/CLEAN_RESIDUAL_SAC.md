# Base-preserving LabPick residual SAC

This baseline adds a small learned correction to selected coordinates of the
frozen Flow Matching policy while retaining an exact frozen-base fallback.

## v3 contract

- The frozen BC supplies a native Gaussian-noise `32 x 10` action chunk and all
  32 actions are consumed before replanning.
- The BC's Rot6D is executed unchanged; simulator/oracle yaw is disabled.
- Actor observation: normalized proprioception 10 + relative object position 3
  + object Rot6D 6 + current normalized BC action 10 + causal tactile input 5.
- Actor action: post-tanh residual for normalized `(x, y, z, width)`.
- Execution: `BC + residual_scale * residual` at indices `[0, 1, 2, 9]`.
  `residual_scale` may be zero and then every physical action is exactly BC.
- The validated deployment scale is `0.01`. Corrections are disabled before
  first tactile contact, preserving the visual approach phase.
- The alpha-zero twin-Q actor loss includes an action-L2 base-anchor term
  (default weight `1.0`) to prevent the observed residual saturation.
- Time limits are terminal for bootstrapping; there is no target actor or
  entropy term.

The persistent policy contract version is 3. Older checkpoints are rejected.

## Train

```bash
TACEX_ISAAC_PYTHON=/path/to/isaaclab/python \
python scripts/reinforcement_learning/dsrl/train_lab_pick_clean_residual_sac.py \
  --bc_policy outputs/lab_pick_dinov3_flow_bc200_yaw0/best.pt \
  --residual_scale 0.01 --residual_contact_gate \
  --action_l2_weight 1.0 --timesteps 50000 --seed 42
```
