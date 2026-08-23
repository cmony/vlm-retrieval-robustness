from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankResult:
    image_to_text: np.ndarray
    text_to_image: np.ndarray
    caption_image_index: np.ndarray


def _normalize_numpy(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings must not contain zero vectors")
    return array / norms


def compute_bidirectional_ranks(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    caption_image_index: np.ndarray,
    chunk_size: int,
    device: str = "cpu",
) -> RankResult:
    images = _normalize_numpy(image_embeddings)
    texts = _normalize_numpy(text_embeddings)
    mapping = np.asarray(caption_image_index, dtype=np.int64)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(mapping) != len(texts) or mapping.ndim != 1:
        raise ValueError("caption_image_index must contain one entry per text embedding")
    if len(images) == 0 or len(texts) == 0:
        raise ValueError("image and text embeddings must not be empty")
    if mapping.min() < 0 or mapping.max() >= len(images):
        raise ValueError("caption_image_index contains an image index outside the embedding range")
    if images.shape[1] != texts.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    if device.startswith("cuda"):
        image_ranks, text_ranks = _compute_torch(images, texts, mapping, chunk_size, device)
    else:
        image_ranks, text_ranks = _compute_numpy(images, texts, mapping, chunk_size)
    return RankResult(image_ranks, text_ranks, mapping.copy())


def _compute_numpy(
    images: np.ndarray,
    texts: np.ndarray,
    mapping: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    positives = [np.flatnonzero(mapping == image_index) for image_index in range(len(images))]
    if any(len(indices) == 0 for indices in positives):
        raise ValueError("every image must have at least one caption")
    image_ranks = np.empty(len(images), dtype=np.int32)
    for start in range(0, len(images), chunk_size):
        stop = min(start + chunk_size, len(images))
        scores = images[start:stop] @ texts.T
        positive_scores = np.asarray(
            [scores[row, positives[start + row]].max() for row in range(stop - start)],
            dtype=np.float32,
        )
        image_ranks[start:stop] = 1 + np.sum(scores > positive_scores[:, None], axis=1)
    text_ranks = np.empty(len(texts), dtype=np.int32)
    for start in range(0, len(texts), chunk_size):
        stop = min(start + chunk_size, len(texts))
        scores = texts[start:stop] @ images.T
        target = scores[np.arange(stop - start), mapping[start:stop]]
        text_ranks[start:stop] = 1 + np.sum(scores > target[:, None], axis=1)
    return image_ranks, text_ranks


def _compute_torch(
    images: np.ndarray,
    texts: np.ndarray,
    mapping: np.ndarray,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CUDA rank computation") from exc
    image_tensor = torch.from_numpy(images).to(device=device)
    text_tensor = torch.from_numpy(texts).to(device=device)
    mapping_tensor = torch.from_numpy(mapping).to(device=device)
    positive_lists = [np.flatnonzero(mapping == index) for index in range(len(images))]
    if any(len(indices) == 0 for indices in positive_lists):
        raise ValueError("every image must have at least one caption")
    image_ranks = np.empty(len(images), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(images), chunk_size):
            stop = min(start + chunk_size, len(images))
            scores = image_tensor[start:stop] @ text_tensor.T
            positive_scores = torch.stack(
                [scores[row, torch.as_tensor(positive_lists[start + row], device=device)].max() for row in range(stop - start)]
            )
            image_ranks[start:stop] = (
                1 + (scores > positive_scores[:, None]).sum(dim=1)
            ).to(dtype=torch.int32).cpu().numpy()
    text_ranks = np.empty(len(texts), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(texts), chunk_size):
            stop = min(start + chunk_size, len(texts))
            scores = text_tensor[start:stop] @ image_tensor.T
            rows = torch.arange(stop - start, device=device)
            target = scores[rows, mapping_tensor[start:stop]]
            text_ranks[start:stop] = (
                1 + (scores > target[:, None]).sum(dim=1)
            ).to(dtype=torch.int32).cpu().numpy()
    del image_tensor, text_tensor, mapping_tensor
    return image_ranks, text_ranks


def summarize_ranks(ranks: np.ndarray) -> dict[str, float]:
    values = np.asarray(ranks)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 1):
        raise ValueError("ranks must be a non-empty one-dimensional array of positive integers")
    return {
        "r1": float(np.mean(values <= 1)),
        "r5": float(np.mean(values <= 5)),
        "r10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values.astype(np.float64))),
        "mean_rank": float(np.mean(values)),
        "median_rank": float(np.median(values)),
    }
