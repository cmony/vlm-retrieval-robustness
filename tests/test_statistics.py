import unittest

import numpy as np

from vlmrobust.statistics import aggregate_by_image, holm_adjust, paired_cluster_bootstrap


class StatisticsTests(unittest.TestCase):
    def test_aggregates_caption_values_within_image(self):
        values = np.array([1.0, 3.0, 2.0, 4.0])
        mapping = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(aggregate_by_image(values, mapping, 2), [2.0, 3.0])

    def test_zero_effect_has_unit_p_value(self):
        clean = np.array([0.1, 0.2, 0.3, 0.4])
        result = paired_cluster_bootstrap(clean, clean, repetitions=500, seed=3, confidence=0.95)
        self.assertEqual(0.0, result.estimate)
        self.assertEqual(1.0, result.p_value)

    def test_positive_effect_confidence_interval_is_positive(self):
        clean = np.zeros(8)
        distorted = np.arange(1.0, 9.0)
        first = paired_cluster_bootstrap(clean, distorted, repetitions=1000, seed=7, confidence=0.95)
        second = paired_cluster_bootstrap(clean, distorted, repetitions=1000, seed=7, confidence=0.95)
        self.assertGreater(first.ci_low, 0.0)
        self.assertEqual(first, second)

    def test_holm_adjustment(self):
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
