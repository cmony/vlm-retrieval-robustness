from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConditionRun, build_condition_runs
from .runner import PROTOCOL_VERSION, load_rank_result
from .statistics import holm_adjust, paired_cluster_bootstrap


FAMILY_LABELS = {
    "gaussian_noise": "Noise",
    "jpeg": "JPEG",
    "crop": "Crop",
    "brightness": "Brightness",
    "style": "Style",
}
FAMILY_ORDER = tuple(FAMILY_LABELS.values())


def _checkpoint_path(rank_root: Path, model_name: str, condition: ConditionRun) -> Path:
    return rank_root / "ranks" / model_name / f"{condition.slug}.npz"


def _check_metadata(
    path: Path,
    metadata: dict[str, Any],
    model_name: str,
    condition: ConditionRun,
    dataset_fingerprint: str | None,
) -> str:
    if metadata.get("model") != model_name:
        raise ValueError(f"checkpoint model mismatch: {path}")
    if metadata.get("condition") != condition.slug:
        raise ValueError(f"checkpoint condition mismatch: {path}")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"checkpoint protocol mismatch: {path}")
    fingerprint = metadata.get("dataset_fingerprint")
    if not fingerprint:
        raise ValueError(f"missing dataset fingerprint: {path}")
    if dataset_fingerprint is not None and fingerprint != dataset_fingerprint:
        raise ValueError(f"dataset fingerprint mismatch: {path}")
    return str(fingerprint)


def load_i2t_family_effects(rank_root: str | Path, config: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    root = Path(rank_root).expanduser().resolve()
    semantic_groups: dict[str, list[ConditionRun]] = defaultdict(list)
    for condition in build_condition_runs(config):
        if condition.family != "clean":
            semantic_groups[condition.semantic_slug].append(condition)

    output: dict[str, dict[str, np.ndarray]] = {label: {} for label in FAMILY_ORDER}
    reference_fingerprint: str | None = None
    reference_mapping: np.ndarray | None = None
    reference_image_count: int | None = None
    reference_software: str | None = None

    clean_condition = ConditionRun("clean", "none", "clean", 0)
    for model in config["models"]:
        model_name = model["name"]
        clean_path = _checkpoint_path(root, model_name, clean_condition)
        if not clean_path.is_file():
            raise FileNotFoundError(f"missing rank checkpoint: {clean_path}")
        clean, clean_metadata = load_rank_result(clean_path)
        reference_fingerprint = _check_metadata(
            clean_path, clean_metadata, model_name, clean_condition, reference_fingerprint
        )
        software = clean_metadata.get("software_fingerprint")
        if reference_software is None:
            reference_software = software
        elif software != reference_software:
            raise ValueError("selected model checkpoints were produced by different software revisions")
        if reference_mapping is None:
            reference_mapping = clean.caption_image_index.copy()
            reference_image_count = len(clean.image_to_text)
        else:
            np.testing.assert_array_equal(clean.caption_image_index, reference_mapping)
            if len(clean.image_to_text) != reference_image_count:
                raise ValueError(f"image count mismatch for {model_name}")

        clean_success = (clean.image_to_text <= 1).astype(np.float64)
        effects_by_family: dict[str, list[np.ndarray]] = defaultdict(list)
        for group in semantic_groups.values():
            repeat_success: list[np.ndarray] = []
            for condition in group:
                path = _checkpoint_path(root, model_name, condition)
                if not path.is_file():
                    raise FileNotFoundError(f"missing rank checkpoint: {path}")
                result, metadata = load_rank_result(path)
                _check_metadata(path, metadata, model_name, condition, reference_fingerprint)
                if metadata.get("software_fingerprint") != reference_software:
                    raise ValueError(f"software fingerprint mismatch: {path}")
                np.testing.assert_array_equal(result.caption_image_index, reference_mapping)
                if len(result.image_to_text) != reference_image_count:
                    raise ValueError(f"image count mismatch: {path}")
                repeat_success.append((result.image_to_text <= 1).astype(np.float64))
            condition_effect = np.mean(repeat_success, axis=0) - clean_success
            effects_by_family[FAMILY_LABELS[group[0].family]].append(condition_effect)
        for family in FAMILY_ORDER:
            if family not in effects_by_family:
                raise ValueError(f"no effects loaded for family {family}")
            output[family][model_name] = np.mean(effects_by_family[family], axis=0)
    return output


def build_pairwise_contrasts(
    model_effects: dict[str, dict[str, np.ndarray]],
    repetitions: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, effects in model_effects.items():
        model_names = list(effects)
        if len(model_names) < 2:
            raise ValueError(f"at least two models are required for {family}")
        lengths = {len(np.asarray(effects[name])) for name in model_names}
        if len(lengths) != 1:
            raise ValueError(f"unaligned per-image effects in {family}")
        for first_model, second_model in combinations(model_names, 2):
            first = np.asarray(effects[first_model], dtype=np.float64)
            second = np.asarray(effects[second_model], dtype=np.float64)
            result = paired_cluster_bootstrap(
                second,
                first,
                repetitions=repetitions,
                seed=seed,
                confidence=confidence,
            )
            rows.append(
                {
                    "family": family,
                    "first_model": first_model,
                    "second_model": second_model,
                    "difference": result.estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "p_value": result.p_value,
                    "bootstrap_repetitions": int(repetitions),
                    "seed": int(seed),
                    "confidence": float(confidence),
                    "holm_family_size": 0,
                }
            )
    adjusted = holm_adjust(np.asarray([row["p_value"] for row in rows], dtype=np.float64))
    family_size = len(rows)
    for row, value in zip(rows, adjusted):
        row["p_value_holm"] = float(value)
        row["holm_family_size"] = family_size
    return rows


def validate_family_effects_against_aggregate(
    model_effects: dict[str, dict[str, np.ndarray]],
    aggregate_path: str | Path,
    tolerance: float = 1e-9,
) -> None:
    path = Path(aggregate_path).expanduser().resolve()
    expected: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["direction"] == "image_to_text" and row["semantic_condition"] != "clean":
                expected[(row["model"], FAMILY_LABELS[row["family"]])].append(float(row["r1"]))
            elif row["direction"] == "image_to_text" and row["semantic_condition"] == "clean":
                expected[(row["model"], "clean")].append(float(row["r1"]))
    clean_by_model = {
        model: values[0]
        for (model, family), values in expected.items()
        if family == "clean" and len(values) == 1
    }
    for family, effects in model_effects.items():
        for model_name, values in effects.items():
            if model_name not in clean_by_model:
                raise ValueError(f"missing clean aggregate row for {model_name}")
            condition_values = expected.get((model_name, family))
            if not condition_values:
                raise ValueError(f"missing aggregate rows for {model_name} {family}")
            aggregate_delta = float(np.mean(condition_values) - clean_by_model[model_name])
            effect_delta = float(np.mean(values))
            if abs(aggregate_delta - effect_delta) > tolerance:
                raise ValueError(
                    f"family effect mismatch for {model_name} {family}: "
                    f"{effect_delta} vs {aggregate_delta}"
                )
