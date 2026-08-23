from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ConditionRun, build_condition_runs
from .runner import PROTOCOL_VERSION, load_rank_result
from .retrieval import RankResult, summarize_ranks
from .statistics import aggregate_by_image, holm_adjust, paired_cluster_bootstrap


SUMMARY_KEYS = ("r1", "r5", "r10", "mrr", "mean_rank", "median_rank")
INFERENCE_METRICS = ("r1", "r5", "r10", "mrr", "rank")


@dataclass(frozen=True)
class TransitionSummary:
    clean_success_count: int
    success_to_failure_count: float
    failure_to_success_count: float
    success_to_failure_rate: float


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _direction_ranks(result: RankResult, direction: str) -> np.ndarray:
    if direction == "image_to_text":
        return result.image_to_text
    if direction == "text_to_image":
        return result.text_to_image
    raise ValueError(f"unknown retrieval direction: {direction}")


def _mean_repeat_summary(results: list[RankResult], direction: str) -> dict[str, float]:
    """Average retrieval metrics across repeats, without averaging ranks first."""
    if not results:
        raise ValueError("at least one repeat is required")
    summaries = [summarize_ranks(_direction_ranks(result, direction)) for result in results]
    return {
        key: float(np.mean([summary[key] for summary in summaries]))
        for key in SUMMARY_KEYS
    }


def _per_image_values(result: RankResult, direction: str, metric: str) -> np.ndarray:
    values = _raw_query_values(result, direction, metric)
    if direction == "image_to_text":
        return values
    return aggregate_by_image(values, result.caption_image_index, len(result.image_to_text))


def _raw_query_values(result: RankResult, direction: str, metric: str) -> np.ndarray:
    ranks = _direction_ranks(result, direction).astype(np.float64)
    if metric == "r1":
        values = (ranks <= 1).astype(np.float64)
    elif metric == "r5":
        values = (ranks <= 5).astype(np.float64)
    elif metric == "r10":
        values = (ranks <= 10).astype(np.float64)
    elif metric == "mrr":
        values = 1.0 / ranks
    elif metric == "rank":
        values = ranks
    else:
        raise ValueError(f"unknown metric: {metric}")
    return values


def _failure_transition(
    clean: RankResult, distorted: list[RankResult], direction: str, cutoff: int
) -> TransitionSummary:
    metric = f"r{cutoff}"
    clean_success = _raw_query_values(clean, direction, metric)
    distorted_success = np.mean(
        [_raw_query_values(result, direction, metric) for result in distorted], axis=0
    )
    eligible = clean_success > 0.5
    eligible_count = int(eligible.sum())
    success_to_failure = float(np.sum(clean_success * (1.0 - distorted_success)))
    failure_to_success = float(np.sum((1.0 - clean_success) * distorted_success))
    rate = success_to_failure / eligible_count if eligible_count else float("nan")
    return TransitionSummary(eligible_count, success_to_failure, failure_to_success, rate)


def _image_ids_from_index(root: Path, image_count: int) -> list[int]:
    index_path = root / "dataset_index.csv"
    if not index_path.is_file():
        return list(range(image_count))
    by_position: dict[int, int] = {}
    with index_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_position[int(row["image_index"])] = int(row["image_id"])
    if set(by_position) != set(range(image_count)):
        raise ValueError("dataset_index.csv does not match rank checkpoint image count")
    return [by_position[index] for index in range(image_count)]


def _checkpoint_path(output_dir: Path, model_name: str, condition: ConditionRun) -> Path:
    return output_dir / "ranks" / model_name / f"{condition.slug}.npz"


def _dataset_index_fingerprint(root: Path) -> str:
    path = root / "dataset_index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset index: {path}")
    with path.open(encoding="utf-8") as handle:
        first = next(csv.DictReader(handle), None)
    if not first or not first.get("dataset_fingerprint"):
        raise ValueError("dataset_index.csv has no dataset fingerprint")
    return first["dataset_fingerprint"]


def discover_complete_models(output_dir: str | Path, config: dict[str, Any]) -> list[str]:
    root = Path(output_dir).expanduser().resolve()
    fingerprint = _dataset_index_fingerprint(root)
    conditions = build_condition_runs(config)
    complete: list[str] = []
    reference_software: str | None = None
    for model in config["models"]:
        consistent = True
        model_software: str | None = None
        for condition in conditions:
            path = _checkpoint_path(root, model["name"], condition)
            if not path.is_file():
                consistent = False
                break
            try:
                _, metadata = load_rank_result(path)
            except (OSError, ValueError, KeyError):
                consistent = False
                break
            if (
                metadata.get("model") != model["name"]
                or metadata.get("condition") != condition.slug
                or metadata.get("dataset_fingerprint") != fingerprint
                or metadata.get("protocol_version") != PROTOCOL_VERSION
            ):
                consistent = False
                break
            software = metadata.get("software_fingerprint")
            model_software = software if model_software is None else model_software
            if software != model_software:
                consistent = False
                break
        if consistent and (reference_software is None or model_software == reference_software):
            complete.append(model["name"])
            reference_software = model_software
    return complete


def analyze_output(
    output_dir: str | Path,
    config: dict[str, Any],
    model_names: Iterable[str] | None = None,
    analysis_output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    report_root = (
        Path(analysis_output_dir).expanduser().resolve()
        if analysis_output_dir is not None
        else root
    )
    dataset_fingerprint = _dataset_index_fingerprint(root)
    conditions = build_condition_runs(config)
    selected = set(model_names) if model_names else {model["name"] for model in config["models"]}
    model_specs = [model for model in config["models"] if model["name"] in selected]
    if not model_specs:
        raise ValueError("no configured models selected for analysis")
    condition_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    bootstrap_config = config["statistics"]
    analysis_software_fingerprint: str | None = None
    for model_spec in model_specs:
        model_name = model_spec["name"]
        loaded: dict[str, RankResult] = {}
        for condition in conditions:
            path = _checkpoint_path(root, model_name, condition)
            if not path.is_file():
                raise FileNotFoundError(f"missing rank checkpoint: {path}")
            result, metadata = load_rank_result(path)
            if (
                metadata.get("model") != model_name
                or metadata.get("condition") != condition.slug
                or metadata.get("dataset_fingerprint") != dataset_fingerprint
                or metadata.get("protocol_version") != PROTOCOL_VERSION
            ):
                raise ValueError(f"checkpoint provenance does not match this analysis: {path}")
            software = metadata.get("software_fingerprint")
            if analysis_software_fingerprint is None:
                analysis_software_fingerprint = software
            elif software != analysis_software_fingerprint:
                raise ValueError("selected model checkpoints were produced by different software revisions")
            loaded[condition.slug] = result
            for direction in ("image_to_text", "text_to_image"):
                summary = summarize_ranks(_direction_ranks(result, direction))
                condition_rows.append(
                    {
                        "model": model_name,
                        "checkpoint": model_spec["checkpoint"],
                        "direction": direction,
                        "condition": condition.slug,
                        "semantic_condition": condition.semantic_slug,
                        "family": condition.family,
                        "parameter": condition.parameter,
                        "value": condition.value,
                        "repeat": condition.repeat,
                        **summary,
                    }
                )
        semantic_groups: dict[str, list[ConditionRun]] = defaultdict(list)
        for condition in conditions:
            semantic_groups[condition.semantic_slug].append(condition)
        clean_result = loaded["clean"]
        image_ids = _image_ids_from_index(root, len(clean_result.image_to_text))
        for semantic_slug, group in semantic_groups.items():
            for direction in ("image_to_text", "text_to_image"):
                rows = [
                    row
                    for row in condition_rows
                    if row["model"] == model_name
                    and row["direction"] == direction
                    and row["semantic_condition"] == semantic_slug
                ]
                aggregate = {
                    "model": model_name,
                    "checkpoint": model_spec["checkpoint"],
                    "direction": direction,
                    "semantic_condition": semantic_slug,
                    "family": group[0].family,
                    "parameter": group[0].parameter,
                    "value": group[0].value,
                    "repetitions": len(group),
                }
                repeat_results = [loaded[condition.slug] for condition in group]
                aggregate.update(_mean_repeat_summary(repeat_results, direction))
                for cutoff in (1, 5, 10):
                    transition = _failure_transition(
                        clean_result, [loaded[condition.slug] for condition in group], direction, cutoff
                    )
                    aggregate[f"clean_success_count_r{cutoff}"] = transition.clean_success_count
                    aggregate[f"success_to_failure_count_r{cutoff}"] = transition.success_to_failure_count
                    aggregate[f"failure_to_success_count_r{cutoff}"] = transition.failure_to_success_count
                    aggregate[f"failure_transition_r{cutoff}"] = transition.success_to_failure_rate
                aggregate_rows.append(aggregate)
            if semantic_slug == "clean":
                continue
            for direction in ("image_to_text", "text_to_image"):
                clean_rank = _per_image_values(clean_result, direction, "rank")
                distorted_rank = np.mean(
                    [_per_image_values(loaded[condition.slug], direction, "rank") for condition in group], axis=0
                )
                for image_index, image_id in enumerate(image_ids):
                    per_image_rows.append(
                        {
                            "model": model_name,
                            "direction": direction,
                            "semantic_condition": semantic_slug,
                            "family": group[0].family,
                            "parameter": group[0].parameter,
                            "value": group[0].value,
                            "repetitions": len(group),
                            "image_index": image_index,
                            "image_id": image_id,
                            "clean_rank": float(clean_rank[image_index]),
                            "distorted_mean_rank": float(distorted_rank[image_index]),
                            "rank_shift": float(distorted_rank[image_index] - clean_rank[image_index]),
                        }
                    )
                for metric in INFERENCE_METRICS:
                    clean_values = _per_image_values(clean_result, direction, metric)
                    distorted_values = np.mean(
                        [_per_image_values(loaded[condition.slug], direction, metric) for condition in group],
                        axis=0,
                    )
                    result = paired_cluster_bootstrap(
                        clean_values,
                        distorted_values,
                        int(bootstrap_config["bootstrap_repetitions"]),
                        int(bootstrap_config["seed"]),
                        float(bootstrap_config["confidence"]),
                    )
                    bootstrap_rows.append(
                        {
                            "model": model_name,
                            "direction": direction,
                            "semantic_condition": semantic_slug,
                            "family": group[0].family,
                            "parameter": group[0].parameter,
                            "value": group[0].value,
                            "metric": metric,
                            "estimate_distorted_minus_clean": result.estimate,
                            "ci_low": result.ci_low,
                            "ci_high": result.ci_high,
                            "p_value": result.p_value,
                        }
                    )
    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(bootstrap_rows):
        grouped_indices[(row["direction"], row["metric"])].append(index)
    for indices in grouped_indices.values():
        adjusted = holm_adjust(np.asarray([bootstrap_rows[index]["p_value"] for index in indices]))
        for index, value in zip(indices, adjusted):
            bootstrap_rows[index]["p_value_holm"] = float(value)
    condition_path = report_root / "condition_metrics.csv"
    aggregate_path = report_root / "condition_aggregate.csv"
    bootstrap_path = report_root / "paired_bootstrap.csv"
    per_image_path = report_root / "per_image_rank_shift.csv"
    condition_fields = [
        "model", "checkpoint", "direction", "condition", "semantic_condition", "family",
        "parameter", "value", "repeat", *SUMMARY_KEYS,
    ]
    aggregate_fields = [
        "model", "checkpoint", "direction", "semantic_condition", "family", "parameter", "value",
        "repetitions", *SUMMARY_KEYS,
        "clean_success_count_r1", "success_to_failure_count_r1", "failure_to_success_count_r1", "failure_transition_r1",
        "clean_success_count_r5", "success_to_failure_count_r5", "failure_to_success_count_r5", "failure_transition_r5",
        "clean_success_count_r10", "success_to_failure_count_r10", "failure_to_success_count_r10", "failure_transition_r10",
    ]
    bootstrap_fields = [
        "model", "direction", "semantic_condition", "family", "parameter", "value", "metric",
        "estimate_distorted_minus_clean", "ci_low", "ci_high", "p_value", "p_value_holm",
    ]
    per_image_fields = [
        "model", "direction", "semantic_condition", "family", "parameter", "value", "repetitions",
        "image_index", "image_id", "clean_rank", "distorted_mean_rank", "rank_shift",
    ]
    _write_csv(condition_path, condition_rows, condition_fields)
    _write_csv(aggregate_path, aggregate_rows, aggregate_fields)
    _write_csv(bootstrap_path, bootstrap_rows, bootstrap_fields)
    _write_csv(per_image_path, per_image_rows, per_image_fields)
    return {
        "condition_metrics": condition_path,
        "condition_aggregate": aggregate_path,
        "paired_bootstrap": bootstrap_path,
        "per_image_rank_shift": per_image_path,
    }
