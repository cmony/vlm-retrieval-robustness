from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vlmrobust.config import load_config
from vlmrobust.contrasts import (
    build_pairwise_contrasts,
    load_i2t_family_effects,
    validate_family_effects_against_aggregate,
)


FIELDNAMES = [
    "family",
    "first_model",
    "second_model",
    "difference",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_holm",
    "bootstrap_repetitions",
    "seed",
    "confidence",
    "holm_family_size",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild image-to-text R@1 checkpoint contrasts.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--rank-root", required=True, help="Root containing ranks/<model>/*.npz.")
    parser.add_argument("--output", required=True, help="Destination CSV path.")
    parser.add_argument(
        "--condition-aggregate",
        help="Optional corrected condition_aggregate.csv used to verify family means before writing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stats = config["statistics"]
    repetitions = int(stats["bootstrap_repetitions"])
    seed = int(stats["seed"])
    confidence = float(stats["confidence"])

    effects = load_i2t_family_effects(args.rank_root, config)
    if args.condition_aggregate:
        validate_family_effects_against_aggregate(effects, args.condition_aggregate)
    rows = build_pairwise_contrasts(effects, repetitions, seed, confidence)
    if len(rows) != 30:
        raise ValueError(f"expected 30 checkpoint contrasts, got {len(rows)}")
    if {row["holm_family_size"] for row in rows} != {30}:
        raise ValueError("Holm correction must be applied over exactly 30 contrasts")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
