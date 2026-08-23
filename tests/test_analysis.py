import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vlmrobust.analysis import _failure_transition, _mean_repeat_summary, analyze_output
from vlmrobust.config import build_condition_runs
from vlmrobust.retrieval import RankResult, summarize_ranks
from vlmrobust.runner import PROTOCOL_VERSION, atomic_save_rank_result


class AnalysisIntegrationTests(unittest.TestCase):
    def test_text_to_image_transition_is_query_weighted(self):
        mapping = np.array([0, 0, 0, 1])
        clean = RankResult(np.array([1, 1]), np.array([1, 1, 1, 1]), mapping)
        distorted = RankResult(np.array([1, 1]), np.array([1, 1, 1, 2]), mapping)
        transition = _failure_transition(clean, [distorted], "text_to_image", 1)
        self.assertEqual(4, transition.clean_success_count)
        self.assertEqual(1.0, transition.success_to_failure_count)
        self.assertEqual(0.0, transition.failure_to_success_count)
        self.assertEqual(0.25, transition.success_to_failure_rate)

    def test_documents_estimator_divergence(self):
        mapping = np.array([0, 0, 1, 1])
        repeats = [
            RankResult(np.array([1, 2]), np.array([1, 1, 2, 2]), mapping),
            RankResult(np.array([2, 2]), np.array([2, 2, 2, 2]), mapping),
            RankResult(np.array([1, 1]), np.array([1, 1, 1, 1]), mapping),
        ]
        metric_first = _mean_repeat_summary(repeats, "image_to_text")
        rank_first = summarize_ranks(
            np.mean([result.image_to_text for result in repeats], axis=0)
        )
        self.assertEqual(0.5, metric_first["r1"])
        self.assertEqual(0.0, rank_first["r1"])

    def test_stochastic_aggregate_is_metric_first(self):
        config = {
            "models": [{"name": "toy", "type": "clip", "checkpoint": "toy/checkpoint"}],
            "statistics": {"bootstrap_repetitions": 100, "confidence": 0.95, "seed": 7},
            "distortions": {
                "gaussian_noise": {"parameter": "sigma", "values": [0.2], "repetitions": 3}
            },
        }
        mapping = np.array([0, 0, 1, 1])
        clean = RankResult(np.array([1, 2]), np.array([1, 1, 2, 2]), mapping)
        repeats = [
            RankResult(np.array([1, 2]), np.array([1, 1, 2, 2]), mapping),
            RankResult(np.array([2, 2]), np.array([2, 2, 2, 2]), mapping),
            RankResult(np.array([1, 1]), np.array([1, 1, 1, 1]), mapping),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (root / "dataset_index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["dataset_fingerprint", "image_index", "image_id", "file_name", "caption_index", "caption_id", "caption"],
                )
                writer.writeheader()
                for image_index in range(2):
                    for caption_index in range(2):
                        writer.writerow({"dataset_fingerprint": "toy-data", "image_index": image_index, "image_id": 100 + image_index, "file_name": f"{image_index}.jpg", "caption_index": caption_index, "caption_id": image_index * 2 + caption_index, "caption": "toy"})
            for condition, result in zip(build_condition_runs(config), [clean, *repeats]):
                atomic_save_rank_result(root / "ranks" / "toy" / f"{condition.slug}.npz", result, {"model": "toy", "checkpoint": "toy/checkpoint", "condition": condition.slug, "semantic_condition": condition.semantic_slug, "image_count": 2, "caption_count": 4, "dataset_fingerprint": "toy-data", "protocol_version": PROTOCOL_VERSION, "software_fingerprint": "toy-software"})
            paths = analyze_output(root, config)
            with paths["condition_aggregate"].open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            noise_i2t = next(row for row in rows if row["semantic_condition"] == "gaussian_noise__sigma-0p2" and row["direction"] == "image_to_text")
            self.assertAlmostEqual(0.5, float(noise_i2t["r1"]), places=12)
            for direction, query_count in (("image_to_text", 2), ("text_to_image", 4)):
                clean_row = next(row for row in rows if row["semantic_condition"] == "clean" and row["direction"] == direction)
                distorted_row = next(row for row in rows if row["semantic_condition"] == "gaussian_noise__sigma-0p2" and row["direction"] == direction)
                for cutoff in (1, 5, 10):
                    delta = float(distorted_row[f"r{cutoff}"]) - float(clean_row[f"r{cutoff}"])
                    success_to_failure = float(distorted_row[f"success_to_failure_count_r{cutoff}"])
                    failure_to_success = float(distorted_row[f"failure_to_success_count_r{cutoff}"])
                    self.assertAlmostEqual(delta, (failure_to_success - success_to_failure) / query_count, places=12)
                    self.assertAlmostEqual(
                        float(distorted_row[f"failure_transition_r{cutoff}"]),
                        success_to_failure / float(distorted_row[f"clean_success_count_r{cutoff}"]),
                        places=12,
                    )

    def test_writes_analysis_to_separate_directory(self):
        config = {
            "models": [{"name": "toy", "type": "clip", "checkpoint": "toy/checkpoint"}],
            "statistics": {"bootstrap_repetitions": 100, "confidence": 0.95, "seed": 7},
            "distortions": {"style": {"parameter": "operation", "values": ["invert"], "repetitions": 1}},
        }
        mapping = np.array([0, 0, 1, 1])
        clean = RankResult(np.array([1, 1]), np.array([1, 1, 1, 1]), mapping)
        distorted = RankResult(np.array([2, 1]), np.array([2, 2, 1, 1]), mapping)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corrected = root / "corrected"
            with (root / "dataset_index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset_fingerprint", "image_index", "image_id", "file_name", "caption_index", "caption_id", "caption"])
                writer.writeheader()
                for image_index in range(2):
                    for caption_index in range(2):
                        writer.writerow({"dataset_fingerprint": "toy-data", "image_index": image_index, "image_id": 100 + image_index, "file_name": f"{image_index}.jpg", "caption_index": caption_index, "caption_id": image_index * 2 + caption_index, "caption": "toy"})
            for condition, result in zip(build_condition_runs(config), (clean, distorted)):
                atomic_save_rank_result(root / "ranks" / "toy" / f"{condition.slug}.npz", result, {"model": "toy", "checkpoint": "toy/checkpoint", "condition": condition.slug, "semantic_condition": condition.semantic_slug, "image_count": 2, "caption_count": 4, "dataset_fingerprint": "toy-data", "protocol_version": PROTOCOL_VERSION, "software_fingerprint": "toy-software"})
            paths = analyze_output(root, config, analysis_output_dir=corrected)
            self.assertTrue(all(path.parent == corrected.resolve() for path in paths.values()))
            self.assertFalse((root / "condition_aggregate.csv").exists())

    def test_writes_all_three_analysis_tables_from_rank_checkpoints(self):
        config = {
            "models": [{"name": "toy", "type": "clip", "checkpoint": "toy/checkpoint"}],
            "statistics": {"bootstrap_repetitions": 100, "confidence": 0.95, "seed": 7},
            "distortions": {
                "style": {"parameter": "operation", "values": ["invert"], "repetitions": 1}
            },
        }
        mapping = np.array([0, 0, 1, 1])
        clean = RankResult(np.array([1, 1]), np.array([1, 1, 1, 1]), mapping)
        distorted = RankResult(np.array([2, 1]), np.array([2, 2, 1, 1]), mapping)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (root / "dataset_index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["dataset_fingerprint", "image_index", "image_id", "file_name", "caption_index", "caption_id", "caption"],
                )
                writer.writeheader()
                for image_index in range(2):
                    for caption_index in range(2):
                        writer.writerow(
                            {
                                "dataset_fingerprint": "toy-data",
                                "image_index": image_index,
                                "image_id": 100 + image_index,
                                "file_name": f"{image_index}.jpg",
                                "caption_index": caption_index,
                                "caption_id": image_index * 2 + caption_index,
                                "caption": "toy",
                            }
                        )
            for condition, result in zip(build_condition_runs(config), (clean, distorted)):
                metadata = {
                    "model": "toy",
                    "checkpoint": "toy/checkpoint",
                    "condition": condition.slug,
                    "semantic_condition": condition.semantic_slug,
                    "image_count": 2,
                    "caption_count": 4,
                    "dataset_fingerprint": "toy-data",
                    "protocol_version": PROTOCOL_VERSION,
                    "software_fingerprint": "toy-software",
                }
                atomic_save_rank_result(root / "ranks" / "toy" / f"{condition.slug}.npz", result, metadata)
            paths = analyze_output(root, config)
            self.assertEqual(
                {"condition_metrics", "condition_aggregate", "paired_bootstrap", "per_image_rank_shift"},
                set(paths),
            )
            for path in paths.values():
                self.assertTrue(path.is_file())
            with paths["condition_aggregate"].open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            invert_i2t = next(
                row for row in rows
                if row["semantic_condition"] == "style__operation-invert" and row["direction"] == "image_to_text"
            )
            self.assertEqual("0.5", invert_i2t["failure_transition_r1"])
            with paths["paired_bootstrap"].open(encoding="utf-8") as handle:
                bootstrap_rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["p_value_holm"] for row in bootstrap_rows))
            clean_i2t = next(row for row in rows if row["semantic_condition"] == "clean" and row["direction"] == "image_to_text")
            delta = float(invert_i2t["r1"]) - float(clean_i2t["r1"])
            sf = float(invert_i2t["success_to_failure_count_r1"])
            fs = float(invert_i2t["failure_to_success_count_r1"])
            self.assertAlmostEqual(delta, (fs - sf) / 2, places=12)


if __name__ == "__main__":
    unittest.main()
