from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vlmrobust.config import ConditionRun, load_config
from vlmrobust.data import CocoRecord
from vlmrobust.models import load_model_adapter, release_model
from vlmrobust.runner import prepare_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and API-smoke-test the configured VLM checkpoints")
    parser.add_argument("--config", default="configs/a100_full.yaml")
    parser.add_argument("--models", nargs="+", help="Test only listed configured model names")
    args = parser.parse_args()
    config = prepare_config(load_config(args.config), model_names=args.models)
    runtime = config["runtime"]
    with tempfile.TemporaryDirectory() as temp:
        image_path = Path(temp) / "red-square.png"
        Image.new("RGB", (224, 224), (220, 30, 30)).save(image_path)
        record = CocoRecord(1, image_path, ("a red square",), (1,))
        for model_spec in config["models"]:
            name = model_spec["name"]
            print(f"Testing {name}: {model_spec['checkpoint']}")
            adapter = load_model_adapter(model_spec, runtime["device"], runtime["precision"])
            try:
                text = adapter.encode_texts(record.captions, batch_size=1)
                image = adapter.encode_images(
                    (record,), ConditionRun("clean", "none", "clean", 0),
                    batch_size=1, loader_workers=0, base_seed=int(config["experiment"]["seed"]),
                )
                if text.ndim != 2 or image.ndim != 2 or text.shape != image.shape:
                    raise RuntimeError(f"{name} produced incompatible feature shapes: {image.shape} vs {text.shape}")
                if not np.all(np.isfinite(text)) or not np.all(np.isfinite(image)):
                    raise RuntimeError(f"{name} produced non-finite features")
                cosine = float((image @ text.T).item())
                print(f"  OK projected dimension={image.shape[1]}, cosine={cosine:.6f}")
            finally:
                release_model(adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
