import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vlmrobust.config import build_condition_runs
from vlmrobust.contrasts import (
    build_pairwise_contrasts,
    load_i2t_family_effects,
    validate_family_effects_against_aggregate,
)
from vlmrobust.retrieval import RankResult
from vlmrobust.runner import PROTOCOL_VERSION, atomic_save_rank_result
from vlmrobust.statistics import holm_adjust


class ContrastTests(unittest.TestCase):
    def test_positive_difference_means_first_model_loses_less(self):
        effects = {
            "a": np.array([-0.1, -0.2, -0.1, -0.2]),
            "b": np.array([-0.3, -0.4, -0.3, -0.4]),
        }
        rows = build_pairwise_contrasts(
            {"Noise": effects}, repetitions=500, seed=7, confidence=0.95
        )
        self.assertEqual(1, len(rows))
        self.assertAlmostEqual(0.2, rows[0]["difference"], places=12)

    def test_four_models_and_five_families_produce_thirty_rows(self):
        pattern = np.array([-0.30, -0.16, -0.11, -0.04, 0.02, 0.05, -0.07, 0.01])
        model_effects = {}
        for family_index, family in enumerate(("Noise", "JPEG", "Crop", "Brightness", "Style")):
            family_effects = {}
            for model_index, model in enumerate(("a", "b", "c", "d")):
                jitter = np.roll(pattern, model_index) * (0.01 + 0.002 * family_index)
                family_effects[model] = pattern + 0.03 * model_index + jitter
            model_effects[family] = family_effects
        rows = build_pairwise_contrasts(model_effects, 100, 3, 0.95)
        self.assertEqual(30, len(rows))
        self.assertEqual({30}, {row["holm_family_size"] for row in rows})
        adjusted = holm_adjust(np.array([row["p_value"] for row in rows]))
        np.testing.assert_allclose(adjusted, np.array([row["p_value_holm"] for row in rows]))

    def test_loads_metric_first_family_effects_and_validates_aggregate(self):
        config = {
            "models": [
                {"name": "a", "type": "clip", "checkpoint": "a/checkpoint"},
                {"name": "b", "type": "clip", "checkpoint": "b/checkpoint"},
            ],
            "statistics": {"bootstrap_repetitions": 100, "confidence": 0.95, "seed": 7},
            "distortions": {
                "gaussian_noise": {"parameter": "sigma", "values": [0.2], "repetitions": 2},
                "jpeg": {"parameter": "quality", "values": [80], "repetitions": 1},
                "crop": {"parameter": "ratio", "values": [0.9], "repetitions": 2},
                "brightness": {"parameter": "factor", "values": [1.2], "repetitions": 1},
                "style": {"parameter": "operation", "values": ["invert"], "repetitions": 1},
            },
        }
        mapping = np.array([0, 0, 1, 1])
        clean = {
            "a": RankResult(np.array([1, 2]), np.array([1, 1, 2, 2]), mapping),
            "b": RankResult(np.array([1, 1]), np.array([1, 1, 1, 1]), mapping),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for model in config["models"]:
                name = model["name"]
                for condition in build_condition_runs(config):
                    if condition.family == "clean":
                        result = clean[name]
                    elif condition.family == "gaussian_noise" and name == "a":
                        ranks = np.array([1, 2]) if condition.repeat == 0 else np.array([2, 2])
                        result = RankResult(ranks, np.array([1, 1, 2, 2]), mapping)
                    elif condition.family == "gaussian_noise" and name == "b":
                        ranks = np.array([1, 2]) if condition.repeat == 0 else np.array([1, 1])
                        result = RankResult(ranks, np.array([1, 1, 1, 1]), mapping)
                    else:
                        result = clean[name]
                    atomic_save_rank_result(
                        root / "ranks" / name / f"{condition.slug}.npz",
                        result,
                        {
                            "model": name,
                            "checkpoint": model["checkpoint"],
                            "condition": condition.slug,
                            "semantic_condition": condition.semantic_slug,
                            "image_count": 2,
                            "caption_count": 4,
                            "dataset_fingerprint": "toy-data",
                            "protocol_version": PROTOCOL_VERSION,
                            "software_fingerprint": "toy-software",
                        },
                    )
            effects = load_i2t_family_effects(root, config)
            self.assertAlmostEqual(-0.25, float(np.mean(effects["Noise"]["a"])), places=12)
            self.assertAlmostEqual(-0.25, float(np.mean(effects["Noise"]["b"])), places=12)
            aggregate = root / "condition_aggregate.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["model", "direction", "semantic_condition", "family", "r1"],
                )
                writer.writeheader()
                for model_name, clean_r1 in (("a", 0.5), ("b", 1.0)):
                    writer.writerow(
                        {
                            "model": model_name,
                            "direction": "image_to_text",
                            "semantic_condition": "clean",
                            "family": "clean",
                            "r1": clean_r1,
                        }
                    )
                    for family, label in (
                        ("gaussian_noise", "Noise"),
                        ("jpeg", "JPEG"),
                        ("crop", "Crop"),
                        ("brightness", "Brightness"),
                        ("style", "Style"),
                    ):
                        delta = -0.25 if label == "Noise" else 0.0
                        writer.writerow(
                            {
                                "model": model_name,
                                "direction": "image_to_text",
                                "semantic_condition": f"{family}__toy",
                                "family": family,
                                "r1": clean_r1 + delta,
                            }
                        )
            validate_family_effects_against_aggregate(effects, aggregate)


if __name__ == "__main__":
    unittest.main()
