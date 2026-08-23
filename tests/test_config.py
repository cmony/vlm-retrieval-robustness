import tempfile
import unittest
from pathlib import Path

import yaml

from vlmrobust.config import build_condition_runs, load_config


class ConfigTests(unittest.TestCase):
    def _config_file(self, limit=5000):
        payload = {
            "experiment": {"seed": 42, "output_dir": "outputs/test"},
            "data": {
                "image_dir": "data/coco/val2017",
                "captions_json": "data/coco/annotations/captions_val2017.json",
                "limit": limit,
                "captions_per_image": 5,
            },
            "models": [{"name": "clip_b32", "type": "clip", "checkpoint": "openai/clip-vit-base-patch32"}],
            "runtime": {"device": "cuda", "precision": "bf16", "image_batch_size": 256, "text_batch_size": 1024, "similarity_chunk_size": 512, "loader_workers": 8},
            "statistics": {"bootstrap_repetitions": 2000, "confidence": 0.95, "seed": 2026},
            "distortions": {
                "gaussian_noise": {"parameter": "sigma", "values": [0.05, 0.1, 0.2, 0.3], "repetitions": 3},
                "jpeg": {"parameter": "quality", "values": [80, 60, 40, 20], "repetitions": 1},
                "crop": {"parameter": "ratio", "values": [0.9, 0.8, 0.7, 0.6], "repetitions": 3},
                "brightness": {"parameter": "factor", "values": [0.6, 0.8, 1.2, 1.4], "repetitions": 1},
                "style": {"parameter": "operation", "values": ["grayscale", "invert"], "repetitions": 1},
            },
        }
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_expands_exactly_35_unique_condition_runs(self):
        config = load_config(self._config_file())
        runs = build_condition_runs(config)
        self.assertEqual(35, len(runs))
        self.assertEqual(35, len({run.slug for run in runs}))
        self.assertEqual("clean", runs[0].family)
        noise = [run for run in runs if run.family == "gaussian_noise"]
        jpeg = [run for run in runs if run.family == "jpeg"]
        self.assertEqual(12, len(noise))
        self.assertEqual(4, len(jpeg))

    def test_rejects_invalid_image_limit(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            load_config(self._config_file(limit=0))


if __name__ == "__main__":
    unittest.main()
