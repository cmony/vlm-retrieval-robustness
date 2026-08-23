from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float


def aggregate_by_image(values: np.ndarray, image_index: np.ndarray, image_count: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    mapping = np.asarray(image_index, dtype=np.int64)
    if data.ndim != 1 or mapping.ndim != 1 or len(data) != len(mapping):
        raise ValueError("values and image_index must be aligned one-dimensional arrays")
    if image_count <= 0 or len(data) == 0:
        raise ValueError("image_count and values must be non-empty")
    if mapping.min() < 0 or mapping.max() >= image_count:
        raise ValueError("image_index contains an out-of-range value")
    counts = np.bincount(mapping, minlength=image_count)
    if np.any(counts == 0):
        raise ValueError("every image must have at least one value")
    sums = np.bincount(mapping, weights=data, minlength=image_count)
    return sums / counts


def paired_cluster_bootstrap(
    clean: np.ndarray,
    distorted: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float,
) -> BootstrapResult:
    clean_values = np.asarray(clean, dtype=np.float64)
    distorted_values = np.asarray(distorted, dtype=np.float64)
    if clean_values.shape != distorted_values.shape or clean_values.ndim != 1 or len(clean_values) == 0:
        raise ValueError("clean and distorted must be aligned non-empty one-dimensional arrays")
    if not np.all(np.isfinite(clean_values)) or not np.all(np.isfinite(distorted_values)):
        raise ValueError("bootstrap inputs must be finite")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    differences = distorted_values - clean_values
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 256):
        stop = min(start + 256, repetitions)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    if np.all(differences == 0.0):
        p_value = 1.0
    else:
        lower_tail = (np.count_nonzero(bootstrap_means <= 0.0) + 1.0) / (repetitions + 1.0)
        upper_tail = (np.count_nonzero(bootstrap_means >= 0.0) + 1.0) / (repetitions + 1.0)
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return BootstrapResult(float(differences.mean()), float(ci_low), float(ci_high), float(p_value))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be a finite one-dimensional array in [0, 1]")
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, sorted_values * np.arange(len(values), 0, -1, dtype=np.float64))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted
