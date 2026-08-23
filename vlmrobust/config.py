from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ConditionRun:
    family: str
    parameter: str
    value: Any
    repeat: int = 0

    @property
    def slug(self) -> str:
        if self.family == "clean":
            return "clean"
        value = str(self.value).replace("-", "m").replace(".", "p").replace("/", "-")
        return f"{self.family}__{self.parameter}-{value}__repeat-{self.repeat}"

    @property
    def semantic_slug(self) -> str:
        if self.family == "clean":
            return "clean"
        value = str(self.value).replace("-", "m").replace(".", "p").replace("/", "-")
        return f"{self.family}__{self.parameter}-{value}"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    _validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required_sections = ("experiment", "data", "models", "runtime", "statistics", "distortions")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"missing configuration sections: {', '.join(missing)}")
    data = config["data"]
    _positive_int(data.get("limit"), "data.limit")
    _positive_int(data.get("captions_per_image"), "data.captions_per_image")
    if data["limit"] > 5000:
        raise ValueError("data.limit cannot exceed the 5,000-image COCO val2017 split")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("models must be a non-empty list")
    names = [model.get("name") for model in config["models"]]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("model names must be present and unique")
    for key in ("image_batch_size", "text_batch_size", "similarity_chunk_size"):
        _positive_int(config["runtime"].get(key), f"runtime.{key}")
    _positive_int(config["statistics"].get("bootstrap_repetitions"), "statistics.bootstrap_repetitions")
    confidence = config["statistics"].get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 < float(confidence) < 1.0:
        raise ValueError("statistics.confidence must be between zero and one")
    for family, spec in config["distortions"].items():
        if not isinstance(spec.get("values"), list) or not spec["values"]:
            raise ValueError(f"distortions.{family}.values must be a non-empty list")
        _positive_int(spec.get("repetitions"), f"distortions.{family}.repetitions")
        if not spec.get("parameter"):
            raise ValueError(f"distortions.{family}.parameter is required")


def build_condition_runs(config: dict[str, Any]) -> list[ConditionRun]:
    runs = [ConditionRun("clean", "none", "clean", 0)]
    for family, spec in config["distortions"].items():
        for value in spec["values"]:
            for repeat in range(spec["repetitions"]):
                runs.append(ConditionRun(family, spec["parameter"], value, repeat))
    slugs = [run.slug for run in runs]
    if len(slugs) != len(set(slugs)):
        raise ValueError("condition run slugs are not unique")
    return runs
