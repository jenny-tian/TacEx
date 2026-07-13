import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SAMPLING_PATH = Path(__file__).resolve().parents[1] / "scripts" / "demos" / "lab_pick" / "sampling.py"
SPEC = spec_from_file_location("lab_pick_sampling", SAMPLING_PATH)
sampling = module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)


class TestLabPickSampling(unittest.TestCase):
    def test_defaults_sample_dataset_at_20hz_from_120hz_sim(self):
        self.assertEqual(sampling.dataset_sample_interval_steps(), 6)

    def test_rejects_non_positive_sample_interval(self):
        with self.assertRaises(ValueError):
            sampling.dataset_sample_interval_steps(sample_interval_s=0.0)

    def test_collects_until_target_when_max_attempts_is_unlimited(self):
        self.assertTrue(sampling.should_continue_collection(recorded=49, target_demos=50, attempts=300, max_attempts=None))
        self.assertFalse(sampling.should_continue_collection(recorded=50, target_demos=50, attempts=300, max_attempts=None))

    def test_optional_max_attempts_limit_only_applies_when_set(self):
        self.assertTrue(sampling.should_continue_collection(recorded=10, target_demos=50, attempts=299, max_attempts=300))
        self.assertFalse(sampling.should_continue_collection(recorded=10, target_demos=50, attempts=300, max_attempts=300))


if __name__ == "__main__":
    unittest.main()
