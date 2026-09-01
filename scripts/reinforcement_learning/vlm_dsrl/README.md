# Online VLM + DSRL LabPick experiment

This package combines the repository's clean Flow-noise DSRL implementation
with episode-level VLM force-range adaptation.

The newer comparison driver in `../compare_exp/run_comparison.py` adds a
strict train/evaluation split, residual SAC, and a direct Flow-RWR baseline.
Its outputs and protocol are documented in `exp_report/compare_exp/README.md`.

- DSRL is updated after every outer interaction once the replay warm-up ends.
- The advisor is called exactly once after each completed episode and after the
  terminal DSRL update.
- Force feedback is disabled in free space. During contact it can replace only
  CAFE action index 9 (gripper width); XYZ and Rot6D remain policy-controlled.
- Every physical step is stored in a compressed trajectory, including the
  policy/executed actions, tactile force, reward, terminal flags, and object
  pose diagnostics.
- The `base` control is the frozen original Flow BC policy: native `step_bc()`
  inference with no SAC/DSRL updates, advisor calls, or contact-force control.
- The `joint_bilateral` ablation is identical to `joint` except that first
  force-controller activation requires both tactile sensors and both per-finger
  forces to cross their contact thresholds. Emergency over-force opening stays
  independent of this gate.

Run the full comparison (5 methods x 3 thresholds x 50 episodes):

```bash
export TACEX_ISAAC_PYTHON=/home/limx/anaconda3/envs/env_isaaclab/bin/python
python scripts/reinforcement_learning/vlm_dsrl/run_matrix.py
```

For a break threshold below the default 3.25 N target-force ceiling, pass a
lower physical and initial range explicitly. For example, the 2 N comparison
keeps a 0.25 N safety margin below the break threshold:

```bash
python scripts/reinforcement_learning/vlm_dsrl/run_matrix.py \
  --thresholds 2 3.5 4 4.5 \
  --physical-force-range-n 0.25 1.75 \
  --initial-force-range-n 1.0 1.5
```

Completed higher-threshold runs are skipped with the default `--resume`; the
explicit low-force range is applied to the new 2 N runs. Each run's
`run_metadata.json` remains the authoritative record of its force contract.

The default advisor is a deterministic offline substitute. To use a real
OpenAI-compatible multimodal model, set `OPENAI_API_KEY` and pass
`--advisor openai`. Do not label substitute runs as real VLM inference.

Record one MP4 per episode from a physical-step camera stream with
`run_experiment.py --record-videos --video-camera third`. The sampling interval
and playback rate are controlled by `--video-every-n-physics-steps` and
`--video-fps`; each `results.json` episode links to its finalized video.

For the frozen 3.5 N joint run and its 10-video diagnostic reproduction, rebuild
the video manifest and failure analysis with:

```bash
python scripts/reinforcement_learning/vlm_dsrl/analyze_joint_diagnostics.py
```

Generate the report after all runs complete:

```bash
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/vlm_dsrl/generate_report.py \
  --input-root exp_report/vlm_with_dsrl
```
