# Episode-level VLM force adaptation

This package evaluates a frozen DINOv3 Flow BC policy with one force-range
advisor transaction after every episode and a 120 Hz tactile force controller
during contact. The controller clones the 10-D policy action and replaces only
CAFE index 9 (gripper width); XYZ and Rot6D remain unchanged.

Run with the deterministic offline protocol substitute:

```bash
export TACEX_ISAAC_PYTHON=/home/limx/anaconda3/envs/env_isaaclab/bin/python
python scripts/vlm_force/eval_lab_pick_vlm_force.py \
  --checkpoint outputs/lab_pick_dinov3_flow_bc200_yaw0/best.pt \
  --output-dir exp_report/vlm_inference_res/my_run \
  --num-trials 20 --seed 42 --device cuda:0
```

Run a real OpenAI-compatible multimodal advisor by setting `OPENAI_API_KEY` and
adding `--advisor openai --vlm-model MODEL`. Use `--api-mode responses` (default)
or `--api-mode chat_completions`. Add `--save-advisor-images` if images should
also be retained locally. The client validates a strict JSON response containing
`target_contact_force_range_n`.

For the policy-only ablation, add `--no-force-control`. The advisor still logs a
recommendation, but it cannot affect execution. Generate the checked report with:

```bash
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/vlm_force/generate_vlm_force_report.py
```

The formal report explicitly distinguishes real VLM calls from the deterministic
offline substitute; do not describe the latter as VLM inference evidence.

The HTTP payloads follow the official OpenAI documentation for
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
the [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create),
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
