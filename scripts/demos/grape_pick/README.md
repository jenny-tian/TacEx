# Force-Constrained Grape Picking

`TacEx-GrapePick-Force-v0` uses the Panda dual-GelSight gripper to detach one
target grape from a vineyard trellis. The grape and both gel pads are physical
colliders. Grip force is the larger of the left/right pad-normal loads read by
PhysX `ContactSensor`s filtered to the target grape; it is never inferred from
visual indentation.

The target fruit hangs below a lateral branch on an upright vine. The expert
keeps the gripper horizontal, approaches from the robot side, closes under
force feedback, and pulls downward along the pedicel. At every reset, the
fruit, pedicel anchor, lateral branch, and upright vine move together by up to
`±(0.020, 0.020, 0.035) m` in XYZ. This preserves the hanging geometry while
providing spatial variation for data collection.

The upright vine, lateral branch, and both trellis posts have physical
colliders. Fifteen contact sensors cover all eight Panda arm links plus the
hand, fingers, GelSight cases, and gel pads; a contact load of 0.25 N or more
is latched as an obstacle collision and cannot be counted as a successful
demonstration. The thin rendered pedicel is visual only because its attachment
is represented by the explicit
tension-only force model.

The nominal stem detaches at 3 N and the nominal damage ceiling is 5 N. For
base-policy evaluation, per-episode thresholds are randomized to model fruit
and stem variability:

- stem detachment: 2.8–3.4 N;
- grape damage: 3.4–5.0 N;
- evaluation horizon: 790 physics steps (6.58 s);
- stable post-detachment hold: 5 steps.

This calibrated distribution intentionally yields successful picks, over-force
damage, and insufficient-pull failures instead of an unrealistically perfect
baseline.

The BC observation explicitly includes the instantaneous left and right
pad-normal contact forces. It is therefore a tactile-conditioned policy even
when optical GelSight RGB recording is disabled.

## Commands

```bash
# One randomized, force-safe expert trajectory and third-person preview
# (optical GelSight disabled)
python scripts/demos/grape_pick/run_grape_pick.py \
  --headless --device cpu --disable_gelsight --seed 42 \
  --record_video --render_width 960 --render_height 540 \
  --video_stride 4 \
  --video_path artifacts/grape_pick/grape_horizontal_preview.mp4 \
  --summary_path artifacts/grape_pick/grape_horizontal_preview.json

# One schema-v2 trajectory with synchronized wrist and third-person RGB
python scripts/demos/grape_pick/collect_dataset.py \
  --headless --device cpu --num_episodes 1 --num_envs 1 --seed 42 \
  --record_stride 4 --max_episode_steps 900 --max_batches 1 --save_rgb \
  --output datasets/grape_pick_horizontal_preview

# Parallel state/contact dataset collection
python scripts/demos/grape_pick/collect_dataset.py \
  --headless --device cpu --num_episodes 256 --num_envs 32 \
  --output datasets/grape_pick_horizontal_tactile_bc_256

# Dataset audit
python scripts/demos/grape_pick/validate_dataset.py \
  --dataset datasets/grape_pick_horizontal_tactile_bc_256

# Train and independently evaluate the trajectory-conditioned MLP baseline
python scripts/demos/grape_pick/train_state_bc.py \
  --dataset datasets/grape_pick_horizontal_tactile_bc_256 \
  --output exp_report/compare_exp_100_with_tactile/grape_pick/checkpoints/tactile_bc.pt
python scripts/demos/grape_pick/eval_state_bc.py \
  --headless --device cpu \
  --checkpoint exp_report/compare_exp_100_with_tactile/grape_pick/checkpoints/tactile_bc.pt \
  --episodes 200 --seed 70000
```

The formal RL comparison and machine-audited summary can be reproduced with
`scripts/reinforcement_learning/grape_pick/run_tactile_comparison.py` and
`summarize_tactile_comparison.py`; see [RESULTS.md](RESULTS.md) for the protocol
and measured results.

```bash
python scripts/reinforcement_learning/grape_pick/run_tactile_comparison.py \
  --headless --device cpu --learner-device cuda \
  --bc-checkpoint exp_report/compare_exp_100_with_tactile/grape_pick/checkpoints/tactile_bc.pt \
  --vlm-force-range exp_report/compare_exp_100_with_tactile/grape_pick/vlm_force_range.json \
  --output-root exp_report/compare_exp_100_with_tactile/grape_pick/formal_v2 \
  --train-episodes 100 --validation-episodes 40 --eval-episodes 100 \
  --eval-seeds 50000 60000 70000 --num-envs 20

python scripts/reinforcement_learning/grape_pick/summarize_tactile_comparison.py \
  --output-root exp_report/compare_exp_100_with_tactile/grape_pick/formal_v2
```

Schema v2 records the instantaneous robot-to-vine/trellis contact force and
stores the configured and sampled fruit-position offsets in `metadata.json`.
The validator rejects any successful trajectory whose recorded obstacle
contact reaches 0.25 N.

The installed Isaac Lab 2.1.0 build currently stalls while initializing the
repository's optical `GelSightSensor`. Therefore the verified preview dataset
and video use the physical pad contact sensors without synthetic tactile RGB.
The code keeps `--save_gelsight` as an explicit opt-in for installations where
that renderer is compatible; it never silently substitutes a virtual force.
