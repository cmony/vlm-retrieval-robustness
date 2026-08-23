import tempfile
import unittest
from pathlib import Path

import numpy as np

from vlmrobust.data import CocoIndex, CocoRecord
from vlmrobust.retrieval import RankResult
from vlmrobust.runner import atomic_save_rank_result, dataset_fingerprint, is_valid_checkpoint, summarize_workload


class RunnerTests(unittest.TestCase):
    def test_atomic_checkpoint_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rank.npz"
            result = RankResult(np.array([1, 2]), np.array([2, 1, 1, 2]), np.array([0, 0, 1, 1]))
            metadata = {"model": "clip_b32", "condition": "clean", "image_count": 2, "caption_count": 4}
            atomic_save_rank_result(path, result, metadata)
            self.assertTrue(is_valid_checkpoint(path, metadata))
            self.assertFalse(is_valid_checkpoint(path, {**metadata, "image_count": 3}))

    def test_corrupt_checkpoint_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.npz"
            path.write_bytes(b"not a zip")
            self.assertFalse(is_valid_checkpoint(path, {"model": "x"}))

    def test_workload_count_matches_design(self):
        summary = summarize_workload(image_count=5000, caption_count=25000, condition_run_count=35, model_count=4)
        self.assertEqual(700000, summary["image_model_evaluations"])
        self.assertEqual(100000, summary["text_model_evaluations"])

    def test_dataset_fingerprint_changes_with_image_or_caption(self):
        mapping = np.array([0], dtype=np.int64)
        first = CocoIndex((CocoRecord(1, Path("one.jpg"), ("caption",)),), ("caption",), mapping)
        changed_image = CocoIndex((CocoRecord(2, Path("two.jpg"), ("caption",)),), ("caption",), mapping)
        changed_caption = CocoIndex((CocoRecord(1, Path("one.jpg"), ("different",)),), ("different",), mapping)
        self.assertNotEqual(dataset_fingerprint(first), dataset_fingerprint(changed_image))
        self.assertNotEqual(dataset_fingerprint(first), dataset_fingerprint(changed_caption))


if __name__ == "__main__":
    unittest.main()
