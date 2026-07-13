import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


VISUALS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "demos" / "lab_pick" / "visuals.py"
SPEC = spec_from_file_location("lab_pick_visuals", VISUALS_PATH)
visuals = module_from_spec(SPEC)
SPEC.loader.exec_module(visuals)


class TestLabPickVisuals(unittest.TestCase):
    def test_slide_visual_material_is_visible_in_previews(self):
        self.assertEqual(visuals.SLIDE_VISUAL_DIFFUSE_COLOR, (0.25, 0.75, 1.0))
        self.assertEqual(visuals.SLIDE_VISUAL_OPACITY, 0.65)
        self.assertEqual(visuals.SLIDE_VISUAL_ROUGHNESS, 0.18)


if __name__ == "__main__":
    unittest.main()
