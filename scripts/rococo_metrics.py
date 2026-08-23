"""Metric helpers for the RoCOCO gallery-injection reanalysis.

The important distinction is denominator discipline:

* retrieval rates are query-level unconditional means;
* success-to-failure is conditional on clean R@k success;
* wrong top-1 and injected top-1 are reported separately.
"""

from __future__ import annotations

import numpy as np


def retrieval_stats(
    best_rank: np.ndarray,
    top1_is_gt: np.ndarray,
    top1_index: np.ndarray | None = None,
    injected_start: int | None = None,
) -> dict:
    """Summarise retrieval ranks and top-1 diagnostics.

    Parameters
    ----------
    best_rank:
        One-based rank of the best ground-truth item for each query.
    top1_is_gt:
        Whether the top-ranked item is a ground-truth item.
    top1_index:
        Optional zero-based gallery index of the top-ranked item.
    injected_start:
        First gallery index belonging to injected adversarial items.
    """

    stats = {
        "r@1": float((best_rank <= 1).mean()),
        "r@5": float((best_rank <= 5).mean()),
        "r@10": float((best_rank <= 10).mean()),
        "mrr": float((1.0 / best_rank).mean()),
        "median_rank": float(np.median(best_rank)),
        "wrong_top1_rate": float((~top1_is_gt).mean()),
    }
    if top1_index is not None and injected_start is not None:
        stats["injected_top1_rate"] = float((top1_index >= injected_start).mean())
    return stats


def success_to_failure(clean_rank: np.ndarray, attacked_rank: np.ndarray, k: int = 1) -> dict:
    """Conditional clean-success-to-attacked-failure transition rate."""

    eligible = clean_rank <= k
    sf = eligible & (attacked_rank > k)
    count = int(eligible.sum())
    return {
        "clean_success_count": count,
        "success_to_failure_count": int(sf.sum()),
        "success_to_failure_rate": float(sf.sum() / count) if count else float("nan"),
    }
