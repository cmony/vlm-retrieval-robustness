"""Shared paths and loaders for the resubmission analyses (CPU only).

All natural-distortion analyses read the released rank checkpoints in
``artifacts/ranks/<model>/<condition>.npz``; each file stores the 1-based rank
of the ground-truth item for every image query (``image_to_text``) and every
caption query (``text_to_image``), plus ``caption_image_index``.
"""
from __future__ import annotations

import csv
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RANKS = REPO_ROOT / "artifacts" / "ranks"
DATASET_INDEX = REPO_ROOT / "artifacts" / "dataset_index.csv"
OUT = REPO_ROOT / "results" / "revision"
KARPATHY_CACHE = REPO_ROOT / "cache" / "karpathy"
ROCOCO_RAW = "https://raw.githubusercontent.com/pseulki/rococo/main/annotations/"

MODELS = ["blip_base", "siglip_base", "clip_vit_b16", "clip_vit_b32"]
LABEL = {"blip_base": "BLIP", "siglip_base": "SigLIP",
         "clip_vit_b16": "CLIP ViT-B/16", "clip_vit_b32": "CLIP ViT-B/32"}
FAMILY = {"gaussian_noise": "Noise", "jpeg": "JPEG", "crop": "Crop",
          "brightness": "Brightness", "style": "Style"}
FAMILY_ORDER = ["Noise", "JPEG", "Crop", "Brightness", "Style"]


def load_ranks(model: str, condition: str):
    z = np.load(RANKS / model / f"{condition}.npz")
    return z["image_to_text"], z["text_to_image"], z["caption_image_index"]


def conditions(model: str) -> dict[tuple[str, str], list[str]]:
    """Map (family, parameter) -> list of repetition slugs (excluding clean)."""
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted((RANKS / model).glob("*.npz")):
        slug = path.stem
        if slug == "clean":
            continue
        family, parameter, _repeat = slug.split("__")
        out[(family, parameter)].append(slug)
    return out


def image_ids() -> np.ndarray:
    ids: dict[int, int] = {}
    with open(DATASET_INDEX, newline="") as handle:
        for row in csv.DictReader(handle):
            ids[int(row["image_index"])] = int(row["image_id"])
    return np.array([ids[i] for i in range(len(ids))])


def karpathy_ids(split: str) -> set[int]:
    """COCO image ids of a Karpathy split, downloaded once into cache/karpathy/."""
    KARPATHY_CACHE.mkdir(parents=True, exist_ok=True)
    path = KARPATHY_CACHE / f"coco_karpathy_{split}.json"
    if not path.exists():
        urllib.request.urlretrieve(ROCOCO_RAW + path.name, path)
    records = json.load(open(path))
    return {int(re.search(r"(\d+)\.jpg", r["image"]).group(1)) for r in records}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(rows)} rows)")
