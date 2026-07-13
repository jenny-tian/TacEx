import unittest
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SUCCESS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "demos" / "lab_pick" / "success.py"
SPEC = spec_from_file_location("lab_pick_success", SUCCESS_PATH)
success = module_from_spec(SPEC)
sys.modules[SPEC.name] = success
SPEC.loader.exec_module(success)


class TestLabPickSuccess(unittest.TestCase):
    def test_requires_touch_before_counting_stable_lift(self):
        tracker = success.StableLiftSuccessTracker(lift_height_m=0.2, hold_steps=2)

        self.assertFalse(tracker.update(lift_delta_m=0.21, has_touched=False, object_gripper_distance_m=0.03))
        self.assertEqual(tracker.stable_steps, 0)

    def test_succeeds_after_stable_lift_hold(self):
        tracker = success.StableLiftSuccessTracker(lift_height_m=0.2, hold_steps=2)

        self.assertFalse(tracker.update(lift_delta_m=0.21, has_touched=True, object_gripper_distance_m=0.03))
        self.assertTrue(tracker.update(lift_delta_m=0.22, has_touched=True, object_gripper_distance_m=0.03))

    def test_resets_when_lift_drops_or_object_moves_away(self):
        tracker = success.StableLiftSuccessTracker(lift_height_m=0.2, hold_steps=2, max_object_gripper_distance_m=0.08)

        self.assertFalse(tracker.update(lift_delta_m=0.21, has_touched=True, object_gripper_distance_m=0.03))
        self.assertFalse(tracker.update(lift_delta_m=0.19, has_touched=True, object_gripper_distance_m=0.03))
        self.assertEqual(tracker.stable_steps, 0)

        self.assertFalse(tracker.update(lift_delta_m=0.21, has_touched=True, object_gripper_distance_m=0.09))
        self.assertEqual(tracker.stable_steps, 0)


if __name__ == "__main__":
    unittest.main()
