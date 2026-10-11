"""Table 1 (query movement) and Table 2 F->S column.

For every checkpoint, family, and direction, success-to-failure (S->F) and
failure-to-success (F->S) transitions at R@1 relative to clean are computed per
repetition, averaged over repetitions (metric-first), then averaged over the
settings of the family. Conditional rates divide by the number of clean
successes (S->F) or clean failures (F->S); counts are reported unrounded.

Output: results/revision/transition_rates_family.csv
"""
from __future__ import annotations

import numpy as np

from _common import FAMILY, FAMILY_ORDER, LABEL, MODELS, OUT, conditions, load_ranks, write_csv


def main() -> None:
    rows = []
    for direction in ["i2t", "t2i"]:
        for model in MODELS:
            i0, t0, _ = load_ranks(model, "clean")
            clean = (i0 if direction == "i2t" else t0) == 1
            n = clean.size
            family_values: dict[str, list[np.ndarray]] = {}
            for (family, _param), slugs in conditions(model).items():
                per_rep = []
                for slug in slugs:
                    i, t, _ = load_ranks(model, slug)
                    now = (i if direction == "i2t" else t) == 1
                    s2f, f2s = (clean & ~now).sum(), (~clean & now).sum()
                    per_rep.append([s2f / clean.sum(), f2s / (~clean).sum(), s2f, f2s])
                family_values.setdefault(FAMILY[family], []).append(np.mean(per_rep, 0))
            for family in FAMILY_ORDER:
                v = np.mean(family_values[family], 0)
                rows.append([direction, LABEL[model], family, round(v[0], 4), round(v[1], 4),
                             round(v[2], 2), round(v[3], 2), round((v[3] - v[2]) / n, 4),
                             round((v[2] + v[3]) / n, 4)])
    write_csv(OUT / "transition_rates_family.csv",
              ["direction", "model", "family", "cond_s2f_r1", "cond_f2s_r1", "count_s2f",
               "count_f2s", "delta_r1", "fraction_changing_outcome"], rows)


if __name__ == "__main__":
    main()
