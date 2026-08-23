import unittest

import numpy as np
from PIL import Image

from vlmrobust.config import ConditionRun
from vlmrobust.distortions import apply_distortion, stable_seed


class DistortionTests(unittest.TestCase):
    def setUp(self):
        values = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        self.image = Image.fromarray(values, mode="RGB")

    def _condition(self, family, parameter, value, repeat=0):
        return ConditionRun(family=family, parameter=parameter, value=value, repeat=repeat)

    def test_stable_seed_is_deterministic_and_repeat_sensitive(self):
        self.assertEqual(stable_seed(42, 7, "crop", 0.8, 0), stable_seed(42, 7, "crop", 0.8, 0))
        self.assertNotEqual(stable_seed(42, 7, "crop", 0.8, 0), stable_seed(42, 7, "crop", 0.8, 1))

    def test_noise_is_reproducible_for_same_image_and_repeat(self):
        condition = self._condition("gaussian_noise", "sigma", 0.2)
        first = np.asarray(apply_distortion(self.image, condition, image_id=9, base_seed=42))
        second = np.asarray(apply_distortion(self.image, condition, image_id=9, base_seed=42))
        np.testing.assert_array_equal(first, second)

    def test_crop_repeat_changes_pixels_but_preserves_shape(self):
        first = apply_distortion(self.image, self._condition("crop", "ratio", 0.6, 0), 9, 42)
        second = apply_distortion(self.image, self._condition("crop", "ratio", 0.6, 1), 9, 42)
        self.assertEqual(self.image.size, first.size)
        self.assertEqual("RGB", first.mode)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_deterministic_transforms_return_rgb(self):
        conditions = [
            self._condition("jpeg", "quality", 40),
            self._condition("brightness", "factor", 0.6),
            self._condition("style", "operation", "grayscale"),
            self._condition("style", "operation", "invert"),
        ]
        for condition in conditions:
            with self.subTest(condition=condition.slug):
                first = apply_distortion(self.image, condition, 9, 42)
                second = apply_distortion(self.image, condition, 9, 999)
                self.assertEqual("RGB", first.mode)
                np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


if __name__ == "__main__":
    unittest.main()
