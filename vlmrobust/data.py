from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CocoRecord:
    image_id: int
    image_path: Path
    captions: tuple[str, ...]
    caption_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CocoIndex:
    records: tuple[CocoRecord, ...]
    captions: tuple[str, ...]
    caption_image_index: np.ndarray


def load_coco_index(
    image_dir: str | Path,
    captions_json: str | Path,
    limit: int,
    sample_seed: int,
    captions_per_image: int = 5,
) -> CocoIndex:
    image_root = Path(image_dir).expanduser().resolve()
    annotation_path = Path(captions_json).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO captions JSON not found: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    image_rows = sorted(payload.get("images", []), key=lambda row: int(row["id"]))
    if limit <= 0 or limit > len(image_rows):
        raise ValueError(f"limit must be between 1 and {len(image_rows)}")
    if limit < len(image_rows):
        rng = np.random.default_rng(sample_seed)
        selected_positions = np.sort(rng.permutation(len(image_rows))[:limit])
        image_rows = [image_rows[int(position)] for position in selected_positions]
    selected_ids = {int(row["id"]) for row in image_rows}
    grouped: dict[int, list[tuple[int, str]]] = {image_id: [] for image_id in selected_ids}
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in grouped:
            grouped[image_id].append((int(annotation["id"]), str(annotation["caption"]).strip()))
    records: list[CocoRecord] = []
    flat_captions: list[str] = []
    caption_image_index: list[int] = []
    for record_index, row in enumerate(image_rows):
        image_id = int(row["id"])
        image_path = image_root / str(row["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image file missing for image {image_id}: {image_path}")
        ordered_pairs = tuple(sorted(grouped[image_id], key=lambda item: item[0]))
        if len(ordered_pairs) < captions_per_image:
            raise ValueError(
                f"image {image_id} has {len(ordered_pairs)} captions; expected at least {captions_per_image}"
            )
        selected_pairs = ordered_pairs[:captions_per_image]
        ordered = tuple(text for _, text in selected_pairs)
        caption_ids = tuple(annotation_id for annotation_id, _ in selected_pairs)
        if any(not text for text in ordered):
            raise ValueError(f"image {image_id} contains an empty caption")
        records.append(CocoRecord(image_id, image_path, ordered, caption_ids))
        flat_captions.extend(ordered)
        caption_image_index.extend([record_index] * len(ordered))
    return CocoIndex(
        records=tuple(records),
        captions=tuple(flat_captions),
        caption_image_index=np.asarray(caption_image_index, dtype=np.int64),
    )
