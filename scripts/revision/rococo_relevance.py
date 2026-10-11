"""Table 4: relevance-policy decomposition of the adapted RoCOCO stress test.

Requires the RoCOCO embedding cache written by scripts/rococo_only.py
(``<rococo_root>/cache/*.npy``) and its annotations (``<rococo_root>/annotations``).

For every query the top-ranked item is classified as
  orig      the original ground truth,
  own       an injected candidate derived from the query's own ground truth
            (a substituted version of one of the query image's captions, or
            the mixed copy whose 0.9-weighted component is the query's image),
  other_inj another injected candidate,
  other     another original item,
and cross-tabulated against the clean outcome (S = clean success, F = clean
failure). Exact-item R@1 counts only ``orig``; source-aware R@1 also counts
``own``. Exact score ties between an original and an injected item are broken
in favour of the original.

The mixing script pairs images inside shuffled batches of 64 with
torch.randperm, which can return the identity; such self-paired images give a
mixed copy identical to the original. The partner map is reconstructed with the
same generator calls, and every image-mixing row is also reported with the
affected queries excluded.

Outputs: results/rococo/rococo_relevance_crosstab.csv,
         results/rococo/rococo_selfpair_audit.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import LABEL, MODELS, REPO_ROOT, write_csv

PERTURBATIONS = ["same_concept", "diff_concept", "rand_voca", "danger"]
SEEDS = [1, 10, 100]
CATS = ["orig", "own", "other_inj", "other"]


def partners(n: int, seed: int, batch_size: int = 64) -> np.ndarray:
    """Replay the generator calls of rococo_only.build_mixed_images."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=gen).tolist()
    part = np.empty(n, dtype=int)
    for start in range(0, n, batch_size):
        idx = shuffled[start:start + batch_size]
        perm = torch.randperm(len(idx), generator=gen).tolist()
        for local, i in enumerate(idx):
            part[i] = idx[perm[local]]
    return part


def crosstab(clean_ok: np.ndarray, cat: np.ndarray) -> dict:
    n = len(cat)
    out = {"clean_r1": clean_ok.mean()}
    for state, mask in [("S", clean_ok), ("F", ~clean_ok)]:
        for ci, name in enumerate(CATS):
            out[f"{state}_to_{name}"] = ((cat == ci) & mask).sum() / n
    out["exact_r1"] = (cat == 0).mean()
    out["source_aware_r1"] = ((cat == 0) | (cat == 1)).mean()
    out["delta_exact"] = out["exact_r1"] - out["clean_r1"]
    out["delta_source_aware"] = out["source_aware_r1"] - out["clean_r1"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rococo-root", default="rococo_run", type=Path)
    args = ap.parse_args()
    cache, ann = args.rococo_root / "cache", args.rococo_root / "annotations"
    karpathy = json.load(open(ann / "coco_karpathy_test.json"))
    n = len(karpathy)
    owner = np.concatenate([[i] * len(r["caption"]) for i, r in enumerate(karpathy)])
    adv_owner = np.repeat(np.arange(n), 5)
    part = {s: partners(n, s) for s in SEEDS}

    rows, audit = [], []
    keys = ["clean_r1", "exact_r1", "source_aware_r1", "delta_exact", "delta_source_aware"] + \
           [f"{s}_to_{c}" for s in "SF" for c in CATS]
    for model in MODELS:
        img = np.load(cache / f"{model}_img.npy").astype(np.float32)
        gt = np.load(cache / f"{model}_gt.npy").astype(np.float32)

        # caption injection, image-to-text
        s_gt = img @ gt.T
        top_gt = s_gt.argmax(1)
        clean_ok = owner[top_gt] == np.arange(n)
        for pert in PERTURBATIONS:
            adv = np.load(cache / f"{model}_{pert}.npy").astype(np.float32)
            s_adv = img @ adv.T
            injected = s_adv.max(1) > s_gt.max(1)
            ta = s_adv.argmax(1)
            cat = np.where(~injected, np.where(clean_ok, 0, 3),
                           np.where(adv_owner[ta] == np.arange(n), 1, 2))
            r = crosstab(clean_ok, cat)
            rows.append([LABEL[model], pert, "image_to_text", "all"] + [round(r[k], 4) for k in keys])

        # image mixing, text-to-image, pooled over seeds
        s_img = gt @ img.T
        c_top = s_img.argmax(1)
        c_ok = c_top == owner
        cats, oks, selfq = [], [], []
        for seed in SEEDS:
            mix = np.load(cache / f"{model}_mix_mix_0.9_{seed}.npy").astype(np.float32)
            s_mix = gt @ mix.T
            injected = s_mix.max(1) > s_img.max(1)
            t = np.where(injected, s_mix.argmax(1), c_top)
            cats.append(np.where(~injected, np.where(t == owner, 0, 3), np.where(t == owner, 1, 2)))
            oks.append(c_ok)
            is_self = part[seed] == np.arange(n)
            selfq.append(is_self[owner])
            identical = np.abs(img - mix).max(1) == 0
            cos = (img * mix).sum(1) / (np.linalg.norm(img, axis=1) * np.linalg.norm(mix, axis=1))
            audit.append([LABEL[model], seed, int(is_self.sum()), int(identical.sum()),
                          int((identical & is_self).sum()), round(float(np.median(cos[~is_self])), 4),
                          round(float(np.median(cos[is_self])), 4)])
        cat, ok, sq = np.concatenate(cats), np.concatenate(oks), np.concatenate(selfq)
        for subset, m in [("all", np.ones_like(sq)), ("excluding_self_paired", ~sq)]:
            r = crosstab(ok[m], cat[m])
            rows.append([LABEL[model], "image_mixing", "text_to_image", subset] + [round(r[k], 4) for k in keys])

    out_dir = REPO_ROOT / "results" / "rococo"
    write_csv(out_dir / "rococo_relevance_crosstab.csv",
              ["model", "perturbation", "direction", "queries"] + keys, rows)
    write_csv(out_dir / "rococo_selfpair_audit.csv",
              ["model", "seed", "self_paired_images", "identical_embeddings",
               "identical_and_self_paired", "median_cos_original_vs_mix_non_self",
               "median_cos_original_vs_mix_self"], audit)


if __name__ == "__main__":
    main()
