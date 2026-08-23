from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from .config import ConditionRun


def stable_seed(base_seed: int, image_id: int, family: str, value: Any, repeat: int) -> int:
    payload = f"{base_seed}|{image_id}|{family}|{value}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def apply_distortion(
    image: Image.Image,
    condition: ConditionRun,
    image_id: int,
    base_seed: int,
) -> Image.Image:
    source = image.convert("RGB")
    if condition.family == "clean":
        return source.copy()
    rng = np.random.default_rng(
        stable_seed(base_seed, image_id, condition.family, condition.value, condition.repeat)
    )
    if condition.family == "gaussian_noise":
        return _gaussian_noise(source, float(condition.value), rng)
    if condition.family == "jpeg":
        return _jpeg(source, int(condition.value))
    if condition.family == "crop":
        return _random_crop(source, float(condition.value), rng)
    if condition.family == "brightness":
        return ImageEnhance.Brightness(source).enhance(float(condition.value)).convert("RGB")
    if condition.family == "style":
        return _style(source, str(condition.value))
    raise ValueError(f"unsupported distortion family: {condition.family}")


def _gaussian_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    if not 0.0 <= sigma <= 1.0:
        raise ValueError("Gaussian noise sigma must be between 0 and 1")
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = np.clip(array + rng.normal(0.0, sigma, array.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8), mode="RGB")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _random_crop(image: Image.Image, ratio: float, rng: np.random.Generator) -> Image.Image:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("crop ratio must be in (0, 1]")
    width, height = image.size
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left = int(rng.integers(0, width - crop_width + 1))
    top = int(rng.integers(0, height - crop_height + 1))
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), resample=Image.Resampling.BICUBIC).convert("RGB")


def _style(image: Image.Image, operation: str) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    if operation == "grayscale":
        gray = 0.2989 * array[:, :, 0] + 0.5870 * array[:, :, 1] + 0.1140 * array[:, :, 2]
        array = np.stack((gray, gray, gray), axis=2)
    elif operation == "invert":
        array = 255.0 - array
    else:
        raise ValueError(f"unsupported style operation: {operation}")
    return Image.fromarray(np.clip(array, 0.0, 255.0).astype(np.uint8), mode="RGB")
