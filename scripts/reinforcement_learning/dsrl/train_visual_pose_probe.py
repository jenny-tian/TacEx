"""Train an independent image-to-relative-pose probe on frozen BC features."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "bc_policy"))
from sim_robot.policy.flow_matching_policy import load_policy  # noqa: E402

from visual_pose_probe import VisualPoseProbe  # noqa: E402


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1.0e-8)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    raw = np.asarray(rot6d, dtype=np.float32).reshape(3, 2)
    first = raw[:, 0] / max(float(np.linalg.norm(raw[:, 0])), 1.0e-8)
    second = raw[:, 1] - first * float(np.dot(first, raw[:, 1]))
    second /= max(float(np.linalg.norm(second)), 1.0e-8)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)[:, :2].reshape(6)


class PoseFrameDataset(Dataset):
    def __init__(self, path: str | Path, demo_names: list[str], max_frames: int = 128) -> None:
        self.path = str(path)
        self.demo_names = list(demo_names)
        self.max_frames = int(max_frames)
        self.samples: list[tuple[str, int]] = []
        with h5py.File(self.path, "r") as h5:
            for name in self.demo_names:
                demo = h5["data"][name]
                if not bool(demo.attrs.get("success", True)):
                    continue
                length = int(demo["obs"]["robot0_pos"].shape[0])
                # Early frames are before contact/object motion and therefore
                # match the reset-pose supervision available in this dataset.
                for t in range(min(length, self.max_frames)):
                    self.samples.append((name, t))
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        name, t = self.samples[index]
        demo = self._file()["data"][name]
        obs = demo["obs"]
        if t == 0:
            wrist = np.asarray(obs["robot0_image"][0:1], dtype=np.float32)
            third = np.asarray(obs["robot0_image_third"][0:1], dtype=np.float32)
            wrist = np.repeat(wrist, 2, axis=0)
            third = np.repeat(third, 2, axis=0)
        else:
            wrist = np.asarray(obs["robot0_image"][t - 1 : t + 1], dtype=np.float32)
            third = np.asarray(obs["robot0_image_third"][t - 1 : t + 1], dtype=np.float32)
        wrist = wrist / 255.0
        third = third / 255.0
        wrist = torch.from_numpy(wrist).permute(0, 3, 1, 2).contiguous()
        third = torch.from_numpy(third).permute(0, 3, 1, 2).contiguous()
        state_raw = np.asarray(obs["robot0_pos"][t], dtype=np.float32)
        state = torch.from_numpy(state_raw)
        object_pos = np.asarray(demo.attrs["labware_reset_pos_w"], dtype=np.float32)
        object_quat = np.asarray(demo.attrs["labware_reset_quat_w"], dtype=np.float32)
        tool_rot = rot6d_to_matrix(state_raw[3:9])
        relative_pos = (object_pos - state_raw[:3]) / 0.25
        relative_rot = matrix_to_rot6d(tool_rot.T @ quat_wxyz_to_matrix(object_quat))
        # Reset-pose labels are most reliable before contact. Keep later
        # frames as low-confidence examples so the probe learns uncertainty
        # instead of fitting a stale simulator pose as ground truth.
        confidence = float(np.exp(-max(t - 48, 0) / 72.0))
        target = np.concatenate((relative_pos, relative_rot, np.asarray([confidence], dtype=np.float32)))
        return {
            "wrist": wrist,
            "third": third,
            "state": state,
            "phase": torch.tensor(float(t) / 383.0, dtype=torch.float32),
            "prev_phase": torch.tensor(float(max(t - 1, 0)) / 383.0, dtype=torch.float32),
            "target": torch.from_numpy(target.astype(np.float32)),
        }


def make_features(model, normalizer, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    state = batch["state"].to(device)
    # BC checkpoints with phase conditioning expect the extra scalar.
    phase = torch.stack((batch["prev_phase"], batch["phase"]), dim=1).to(device).unsqueeze(-1)
    state_history = state[:, None, :].expand(-1, 2, -1)
    state_history = torch.cat((state_history, phase), dim=-1)
    state_history = normalizer.normalize_tensor("robot0_pos", state_history)
    obs = {
        "robot0_pos": state_history,
        "robot0_image": batch["wrist"].to(device),
        "robot0_image_third": batch["third"].to(device),
    }
    tokens, _ = model.obs_encoder(obs)
    return torch.cat((tokens[:, model.config.n_state_obs_steps :].mean(dim=1), state_history[:, -1]), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", required=True)
    parser.add_argument("--dataset", default="datasets/lab_pick_slide_flow_matching_causal_third_200.hdf5")
    parser.add_argument("--output", default="outputs/lab_pick_visual_pose_probe/best.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, checkpoint = load_policy(args.bc_checkpoint, device=str(device), use_ema=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with h5py.File(args.dataset, "r") as h5:
        names = sorted(h5["data"].keys())
    split = max(1, int(0.1 * len(names)))
    train_names, valid_names = names[split:], names[:split]
    train_set = PoseFrameDataset(args.dataset, train_names, args.max_frames)
    valid_set = PoseFrameDataset(args.dataset, valid_names, args.max_frames)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    input_dim = int(model.config.obs_feature_dim + model.config.robot0_pos_dim)
    probe = VisualPoseProbe(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        probe.train(); train_loss = 0.0
        for batch in train_loader:
            with torch.no_grad():
                features = make_features(model, normalizer, batch, device)
            prediction = probe(features)
            target = batch["target"].to(device)
            confidence = target[..., 9].detach()
            pose_error = torch.nn.functional.smooth_l1_loss(
                prediction[..., :9], target[..., :9], reduction="none"
            ).mean(dim=-1)
            loss = (pose_error * confidence).sum() / confidence.sum().clamp_min(1.0)
            loss = loss + 0.25 * torch.nn.functional.binary_cross_entropy(prediction[..., 9], target[..., 9])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss += float(loss.item()) * target.shape[0]
        probe.eval(); valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                prediction = probe(make_features(model, normalizer, batch, device))
                target = batch["target"].to(device)
                confidence = target[..., 9]
                pose_error = torch.nn.functional.smooth_l1_loss(
                    prediction[..., :9], target[..., :9], reduction="none"
                ).mean(dim=-1)
                loss = (pose_error * confidence).sum() / confidence.sum().clamp_min(1.0)
                loss = loss + 0.25 * torch.nn.functional.binary_cross_entropy(prediction[..., 9], target[..., 9])
                valid_loss += float(loss.item()) * target.shape[0]
        train_loss /= max(1, len(train_set)); valid_loss /= max(1, len(valid_set))
        print(f"[probe] epoch={epoch + 1}/{args.epochs} train={train_loss:.5f} valid={valid_loss:.5f}", flush=True)
        if valid_loss < best:
            best = valid_loss
            output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": probe.state_dict(), "input_dim": input_dim, "hidden_dim": 256,
                        "bc_checkpoint": str(Path(args.bc_checkpoint).resolve()),
                        "dataset": str(Path(args.dataset).resolve()), "max_frames": args.max_frames,
                        "valid_loss": valid_loss}, output)
    print(f"[probe] saved {args.output} best_valid={best:.5f}")


if __name__ == "__main__":
    main()
