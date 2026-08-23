import math
import unittest

import numpy as np

from scripts.rococo_metrics import retrieval_stats, success_to_failure


class RoCOCOMetricsTest(unittest.TestCase):
    def test_success_to_failure_uses_clean_success_denominator(self):
        clean = np.array([1, 1, 2, 5, 1])
        attacked = np.array([2, 1, 1, 9, 7])

        got = success_to_failure(clean, attacked, k=1)

        self.assertEqual(got["clean_success_count"], 3)
        self.assertEqual(got["success_to_failure_count"], 2)
        self.assertAlmostEqual(got["success_to_failure_rate"], 2 / 3)

    def test_success_to_failure_nan_when_no_clean_success(self):
        clean = np.array([2, 3, 4])
        attacked = np.array([1, 2, 3])

        got = success_to_failure(clean, attacked, k=1)

        self.assertEqual(got["clean_success_count"], 0)
        self.assertEqual(got["success_to_failure_count"], 0)
        self.assertTrue(math.isnan(got["success_to_failure_rate"]))

    def test_retrieval_stats_separates_wrong_and_injected_top1(self):
        best_rank = np.array([1, 2, 5, 11])
        top1_is_gt = np.array([True, False, False, False])
        top1_index = np.array([0, 7, 11, 13])

        got = retrieval_stats(
            best_rank,
            top1_is_gt,
            top1_index=top1_index,
            injected_start=10,
        )

        self.assertAlmostEqual(got["r@1"], 0.25)
        self.assertAlmostEqual(got["r@5"], 0.75)
        self.assertAlmostEqual(got["r@10"], 0.75)
        self.assertAlmostEqual(got["wrong_top1_rate"], 0.75)
        self.assertAlmostEqual(got["injected_top1_rate"], 0.5)
        self.assertLessEqual(got["injected_top1_rate"], got["wrong_top1_rate"])


if __name__ == "__main__":
    unittest.main()
