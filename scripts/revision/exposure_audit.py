"""Table 5: BLIP fine-tuning exposure audit in both retrieval directions.

Images in the Karpathy training split (used for BLIP retrieval fine-tuning)
are "exposed"; the rest are "unexposed". Queries are restricted to each subset
(images for image-to-text, captions of those images for text-to-image) while
the gallery stays the full 5,000 images / 25,000 captions.

Retention = equal-weight mean over the five families of the family-average
distorted R@1 (metric-first averaging over repeats), divided by clean R@1.

Bootstrap: B=5000 resamples drawn separately within the exposed and unexposed
images; the five captions of an image stay together. The same draws are used
for every checkpoint, so BLIP-minus-control gaps are paired.

Outputs: results/revision/exposure_audit.csv, results/revision/exposure_gap_contrasts.csv
"""
from __future__ import annotations

import numpy as np

from _common import FAMILY, LABEL, MODELS, OUT, conditions, image_ids, karpathy_ids, load_ranks, write_csv

B = 5000
SEED = 2026


def subset_r1(model: str, condition: str, direction: str, mask: np.ndarray) -> float:
    i2t, t2i, owner = load_ranks(model, condition)
    return float((i2t[mask] == 1).mean()) if direction == "i2t" else float((t2i[mask[owner]] == 1).mean())


def retention(model: str, direction: str, mask: np.ndarray) -> tuple[float, float]:
    clean = subset_r1(model, "clean", direction, mask)
    families: dict[str, list[float]] = {}
    for (family, _param), slugs in conditions(model).items():
        value = np.mean([subset_r1(model, s, direction, mask) for s in slugs])
        families.setdefault(FAMILY[family], []).append(value)
    distorted = np.mean([np.mean(v) for v in families.values()])
    return clean, distorted / clean


def per_image_clean_success(model: str, direction: str) -> np.ndarray:
    i2t, t2i, owner = load_ranks(model, "clean")
    if direction == "i2t":
        return (i2t == 1).astype(float)
    hits = np.zeros(len(i2t))
    np.add.at(hits, owner, (t2i == 1))
    counts = np.bincount(owner, minlength=len(i2t))
    return hits / counts


def main() -> None:
    ids = image_ids()
    train = karpathy_ids("train")
    exposed = np.array([i in train for i in ids])
    unexposed = ~exposed
    full = np.ones_like(exposed)
    print(f"exposed={exposed.sum()} unexposed={unexposed.sum()}")

    rng = np.random.default_rng(SEED)
    E, U = np.where(exposed)[0], np.where(unexposed)[0]
    draw_e = rng.integers(0, len(E), (B, len(E)))
    draw_u = rng.integers(0, len(U), (B, len(U)))

    rows, contrast_rows = [], []
    for direction in ["i2t", "t2i"]:
        boot_gap, point_gap = {}, {}
        for model in MODELS:
            clean_e = subset_r1(model, "clean", direction, exposed)
            clean_u, ret_u = retention(model, direction, unexposed)
            _clean_f, ret_f = retention(model, direction, full)
            s = per_image_clean_success(model, direction)
            boot_gap[model] = s[E][draw_e].mean(1) - s[U][draw_u].mean(1)
            point_gap[model] = clean_e - clean_u
            lo, hi = np.percentile(boot_gap[model], [2.5, 97.5])
            rows.append([direction, LABEL[model], round(clean_e, 4), round(clean_u, 4),
                         round(clean_e - clean_u, 4), round(lo, 4), round(hi, 4),
                         round(ret_u, 4), round(ret_f, 4)])
        for control in MODELS[1:]:
            diff = boot_gap["blip_base"] - boot_gap[control]
            point = point_gap["blip_base"] - point_gap[control]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
            contrast_rows.append([direction, f"BLIP - {LABEL[control]}", round(point, 4),
                                  round(lo, 4), round(hi, 4), max(p, 2 / (B + 1))])
    write_csv(OUT / "exposure_audit.csv",
              ["direction", "model", "clean_r1_exposed", "clean_r1_unexposed", "gap",
               "gap_ci_low", "gap_ci_high", "retention_unexposed", "retention_full"], rows)
    write_csv(OUT / "exposure_gap_contrasts.csv",
              ["direction", "contrast", "gap_difference", "ci_low", "ci_high", "p_two_sided_bound"],
              contrast_rows)


if __name__ == "__main__":
    main()
