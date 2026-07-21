# DINOACT — ACT with DINOv3 backbone

ACT ([Learning Fine-Grained Bimanual Manipulation...](https://huggingface.co/papers/2304.13705)) with a **DINOv3** vision encoder from Hugging Face `transformers` instead of torchvision ResNet.

## Dependencies

DINOv3 loading requires Transformers (not installed in the core `lerobot` package):

```bash
pip install 'lerobot[transformers-dep]'
```

## Image normalization

Pretrained DINOv3 expects inputs in the same value range and normalization as documented for your checkpoint (typically **ImageNet mean and standard deviation** on RGB). Configure visual features in your dataset metadata / policy stats accordingly. The backbone performs optional **spatial resize** (`dinov3_interpolate_images`, `dinov3_image_size`) only; it does **not** apply an extra ImageNet normalization inside the model.

## Training

Use policy type **`dinoact`** (see `DINOACTConfig` in `configuration_dinoact.py`). Main knobs: `dinov3_model_id`, `dinov3_attn_implementation`, `dinov3_interpolate_images`, `dinov3_image_size`.

### Optional rot6d Chordal reconstruction loss

If your action layout contains repeated `[pos(3), rot6d(6)]` groups (e.g. body anchors) and optional tail non-rotation dimensions, you can switch reconstruction loss from pure L1 to:

- non-rotation dimensions: L1
- rotation dimensions: Chordal distance `||R_pred - R_gt||_F` after `rot6d -> rotation matrix`
- combined reconstruction: `recon_loss = l1_nonrot + lambda_rot * chordal_rot`

Config:

```yaml
policy:
  type: dinoact
  use_rot6d_chordal_loss: true
  lambda_rot: 1.0
```

## Paper (original ACT)

https://tonyzhaozh.github.io/aloha

```bibtex
@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```
