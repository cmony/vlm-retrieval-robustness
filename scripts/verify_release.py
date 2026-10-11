from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    natural = ROOT / "results" / "natural"
    aggregate = read_csv(natural / "condition_aggregate.csv")
    metrics = read_csv(natural / "condition_metrics.csv")
    bootstrap = read_csv(natural / "paired_bootstrap.csv")
    contrasts = read_csv(natural / "checkpoint_contrasts_i2t_r1.csv")
    rococo = read_csv(ROOT / "results" / "rococo" / "rococo_results.csv")

    require(len(aggregate) == 152, f"expected 152 aggregate rows, got {len(aggregate)}")
    require(len(metrics) == 280, f"expected 280 metric rows, got {len(metrics)}")
    require(len(bootstrap) == 720, f"expected 720 bootstrap rows, got {len(bootstrap)}")
    require(len(contrasts) == 30, f"expected 30 contrasts, got {len(contrasts)}")
    require({int(r["holm_family_size"]) for r in contrasts} == {30}, "Holm family must be 30")
    significant = sum(float(r["p_value_holm"]) <= 0.05 for r in contrasts)
    require(significant == 11, f"expected 11 significant contrasts, got {significant}")
    require(len(rococo) == 20, f"expected 20 RoCOCO rows, got {len(rococo)}")

    clean = {
        (row["model"], row["direction"]): row
        for row in aggregate
        if row["family"] == "clean"
    }
    transition_identities = 0
    for row in aggregate:
        n_queries = 5000 if row["direction"] == "image_to_text" else 25000
        clean_row = clean[(row["model"], row["direction"])]
        for cutoff in (1, 5, 10):
            clean_success = float(row[f"clean_success_count_r{cutoff}"])
            success_to_failure = float(row[f"success_to_failure_count_r{cutoff}"])
            failure_to_success = float(row[f"failure_to_success_count_r{cutoff}"])
            expected_delta = float(row[f"r{cutoff}"]) - float(clean_row[f"r{cutoff}"])
            transition_delta = (failure_to_success - success_to_failure) / n_queries
            require(
                math.isclose(expected_delta, transition_delta, rel_tol=0.0, abs_tol=1e-10),
                f"transition identity mismatch: {row['model']}/{row['semantic_condition']}/R@{cutoff}",
            )
            expected_rate = 0.0 if clean_success == 0 else success_to_failure / clean_success
            require(
                math.isclose(
                    expected_rate,
                    float(row[f"failure_transition_r{cutoff}"]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ),
                f"conditional rate mismatch: {row['model']}/{row['semantic_condition']}/R@{cutoff}",
            )
            transition_identities += 1

    for row in rococo:
        expected = float(row["success_to_failure_count_r1"]) / float(row["clean_success_count_r1"])
        require(
            math.isclose(expected, float(row["stf_r1"]), rel_tol=0.0, abs_tol=1e-12),
            f"conditional transition mismatch: {row['model']}/{row['perturbation']}",
        )

    rank_files = list((ROOT / "artifacts" / "ranks").glob("*/*.npz"))
    require(len(rank_files) == 140, f"expected 140 rank checkpoints, got {len(rank_files)}")

    per_image = natural / "per_image_rank_shift.csv.gz"
    with gzip.open(per_image, "rt", encoding="utf-8", newline="") as handle:
        per_image_rows = sum(1 for _ in handle) - 1
    require(per_image_rows == 720000, f"expected 720000 per-image rows, got {per_image_rows}")

    figures = list((ROOT / "paper" / "figures").glob("figure*_revised.pdf"))
    require(len(figures) == 5, f"expected five paper figures, got {len(figures)}")

    # Resubmission analyses (scripts/revision/)
    revision = ROOT / "results" / "revision"
    exposure = read_csv(revision / "exposure_audit.csv")
    gap_contrasts = read_csv(revision / "exposure_gap_contrasts.csv")
    transitions = read_csv(revision / "transition_rates_family.csv")
    relevance = read_csv(ROOT / "results" / "rococo" / "rococo_relevance_crosstab.csv")
    selfpair = read_csv(ROOT / "results" / "rococo" / "rococo_selfpair_audit.csv")
    require(len(exposure) == 8, f"expected 8 exposure rows, got {len(exposure)}")
    require(len(gap_contrasts) == 6, f"expected 6 exposure contrasts, got {len(gap_contrasts)}")
    require(len(transitions) == 40, f"expected 40 transition rows, got {len(transitions)}")
    require(len(relevance) == 24, f"expected 24 relevance rows, got {len(relevance)}")
    require(len(selfpair) == 12, f"expected 12 self-pair audit rows, got {len(selfpair)}")
    relevance_identities = 0
    for row in relevance:
        f = {k: float(v) for k, v in row.items() if k not in ("model", "perturbation", "direction", "queries")}
        exact = -(f["S_to_own"] + f["S_to_other_inj"] + f["S_to_other"])
        aware = f["F_to_own"] - f["S_to_other_inj"] - f["S_to_other"]
        require(abs(exact - f["delta_exact"]) < 1e-3, f"exact identity failed: {row}")
        require(abs(aware - f["delta_source_aware"]) < 1e-3, f"source-aware identity failed: {row}")
        require(f["F_to_orig"] == 0.0, f"gallery enlargement cannot recover original items: {row}")
        relevance_identities += 3
    for row in selfpair:
        if row["model"] != "BLIP":
            continue
        require(row["identical_embeddings"] == row["identical_and_self_paired"] == row["self_paired_images"],
                f"BLIP identical embeddings should coincide with self-paired images: {row}")

    print("RELEASE GATE PASSED")
    print(f"aggregate={len(aggregate)}, metrics={len(metrics)}, bootstrap={len(bootstrap)}")
    print(f"contrasts={len(contrasts)}, significant={significant}, rococo={len(rococo)}")
    print(
        f"transition_identities={transition_identities}, rank_checkpoints={len(rank_files)}, "
        f"per_image={per_image_rows}, figures={len(figures)}"
    )
    print(
        f"exposure={len(exposure)}, exposure_contrasts={len(gap_contrasts)}, transitions={len(transitions)}, "
        f"relevance={len(relevance)}, relevance_identities={relevance_identities}, selfpair={len(selfpair)}"
    )


if __name__ == "__main__":
    main()
