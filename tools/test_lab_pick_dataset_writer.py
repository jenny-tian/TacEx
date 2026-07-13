import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np


DATASET_PATH = Path(__file__).resolve().parents[1] / "scripts" / "demos" / "lab_pick" / "dataset_writer.py"
SPEC = spec_from_file_location("lab_pick_dataset_writer", DATASET_PATH)
dataset_writer = module_from_spec(SPEC)
SPEC.loader.exec_module(dataset_writer)


class _TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def __getitem__(self, index):
        return _TensorLike(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeEnv:
    labware_reset_pos_w = _TensorLike([[0.52, 0.0, 0.02]])
    labware_reset_quat_w = _TensorLike([[1.0, 0.0, 0.0, 0.0]])


def _append_episode(writer, value):
    writer.action.append(np.ones(10, dtype=np.float32) * value)
    writer.robot0_pos.append(np.ones(10, dtype=np.float32) * (value + 10))
    writer.robot0_force.append(np.ones(6, dtype=np.float32) * (value + 20))
    writer.low_action.append(np.ones(10, dtype=np.float32) * (value + 30))
    writer.robot0_image.append(np.zeros((224, 224, 3), dtype=np.uint8))


class TestLabPickDatasetWriter(unittest.TestCase):
    def test_writes_forcecapture_cafe_multifrequency_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = Path(tmpdir) / "demo.hdf5"
            writer = dataset_writer.LabPickHdf5Writer(dataset_file, labware="slide", instruction="pick", task_id=3)
            for index in range(3):
                writer.action.append(np.ones(10, dtype=np.float32) * index)
                writer.robot0_pos.append(np.ones(10, dtype=np.float32) * (index + 10))
                writer.robot0_force.append(np.ones(6, dtype=np.float32) * (index + 20))
            writer.low_action.append(np.ones(10, dtype=np.float32) * 30)
            writer.robot0_image.append(np.zeros((224, 224, 3), dtype=np.uint8))

            writer.write_episode(_FakeEnv(), success=True)

            with h5py.File(dataset_file, "r") as h5:
                demo = h5["data/demo_0"]
                self.assertEqual(demo["actions/high"].shape, (3, 10))
                self.assertEqual(demo["actions/low"].shape, (1, 10))
                self.assertEqual(demo["obs/robot0_pos"].shape, (3, 10))
                self.assertEqual(demo["obs/robot0_force"].shape, (3, 6))
                self.assertEqual(demo["obs/robot0_image"].shape, (1, 224, 224, 3))
                self.assertEqual(demo.attrs["length_high"], 3)
                self.assertEqual(demo.attrs["length_low"], 1)
                self.assertEqual(demo.attrs["freq_ratio"], 3)
                self.assertEqual(h5.attrs["high_freq_action_key"], "high")
                self.assertEqual(h5.attrs["low_freq_action_key"], "low")
                self.assertEqual(h5.attrs["high_freq_obs_keys"], "robot0_pos,robot0_force")
                self.assertEqual(h5.attrs["low_freq_obs_keys"], "robot0_image")
                self.assertEqual(h5.attrs["freq_ratio"], 3)

    def test_writes_cafe_raw_record_with_real_force_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record_dir = Path(tmpdir) / "record_000000"
            writer = dataset_writer.CafeRecordWriter(record_dir)
            for index in range(2):
                writer.append_sample(
                    timestamp=0.1 * index,
                    sample={
                        "xyz": np.array([0.5, 0.0, 0.1], dtype=np.float32),
                        "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        "width": np.array([0.01], dtype=np.float32),
                        "ft": np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32) * (index + 1),
                        "marker2d": np.zeros((14, 26, 2), dtype=np.float32),
                        "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                        "action": np.ones(10, dtype=np.float32) * index,
                    },
                )

            wrote = writer.flush_episode(
                success=True,
                labware_reset_pos_w=np.array([0.52, 0.0, 0.02], dtype=np.float32),
                labware_reset_quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                force_source="isaac_physx_contact_sensor",
            )

            self.assertTrue(wrote)
            aligned_ft = np.load(record_dir / "aligned_60Hz" / "ft.npy")
            sensor_ft = np.load(record_dir / "ftsensor" / "ft.npy")
            marker_flat = np.load(record_dir / "xense" / "marker2d_flatten.npy")
            self.assertEqual(aligned_ft.shape, (2, 6))
            self.assertEqual(sensor_ft.shape, (2, 6))
            self.assertEqual(marker_flat.shape, (2, 728))
            self.assertGreater(float(np.linalg.norm(aligned_ft)), 0.0)
            metadata = np.load(record_dir / "metadata.npz")
            self.assertEqual(str(metadata["force_source"]), "isaac_physx_contact_sensor")

    def test_demo_numbering_starts_at_zero_and_num_demos_matches_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = Path(tmpdir) / "demo.hdf5"
            writer = dataset_writer.LabPickHdf5Writer(dataset_file, labware="slide", instruction="pick", task_id=3)

            _append_episode(writer, 1)
            writer.write_episode(_FakeEnv(), success=True)
            _append_episode(writer, 2)
            writer.write_episode(_FakeEnv(), success=True)

            with h5py.File(dataset_file, "r") as h5:
                self.assertEqual(sorted(h5["data"].keys()), ["demo_0", "demo_1"])
                self.assertEqual(h5.attrs["num_demos"], 2)


if __name__ == "__main__":
    unittest.main()
