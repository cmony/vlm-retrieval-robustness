from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .config import ConditionRun
from .data import CocoRecord
from .distortions import apply_distortion


class HuggingFaceAdapter:
    text_padding: bool | str = True

    def __init__(self, model: Any, processor: Any, device: str, precision: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.precision = precision

    def _autocast(self):
        if not self.device.startswith("cuda") or self.precision == "fp32":
            return nullcontext()
        import torch

        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        return {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}

    def _feature_tensor(self, output: Any):
        import torch

        if isinstance(output, torch.Tensor):
            return output
        for name in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, name, None)
            if isinstance(value, torch.Tensor):
                return value
        last_hidden = getattr(output, "last_hidden_state", None)
        if isinstance(last_hidden, torch.Tensor):
            return last_hidden[:, 0]
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            value = output[0]
            return value[:, 0] if value.ndim == 3 else value
        raise TypeError(f"unsupported feature output type: {type(output)!r}")

    def _normalize(self, tensor: Any) -> np.ndarray:
        import torch.nn.functional as functional

        normalized = functional.normalize(tensor.float(), dim=-1)
        return normalized.detach().cpu().numpy().astype(np.float32, copy=False)

    def _encode_text_batch(self, inputs: dict[str, Any]):
        return self.model.get_text_features(**inputs)

    def _encode_image_batch(self, inputs: dict[str, Any]):
        return self.model.get_image_features(**inputs)

    def encode_texts(self, captions: Sequence[str], batch_size: int) -> np.ndarray:
        import torch

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(captions), batch_size):
                inputs = self.processor(
                    text=list(captions[start : start + batch_size]),
                    padding=self.text_padding,
                    truncation=True,
                    return_tensors="pt",
                )
                with self._autocast():
                    output = self._encode_text_batch(self._move_inputs(inputs))
                batches.append(self._normalize(self._feature_tensor(output)))
        return np.concatenate(batches, axis=0)

    def encode_images(
        self,
        records: Sequence[CocoRecord],
        condition: ConditionRun,
        batch_size: int,
        loader_workers: int,
        base_seed: int,
    ) -> np.ndarray:
        import torch

        batches: list[np.ndarray] = []
        executor = ThreadPoolExecutor(max_workers=loader_workers) if loader_workers > 0 else None
        try:
            with torch.inference_mode():
                for start in range(0, len(records), batch_size):
                    batch_records = records[start : start + batch_size]
                    if executor is None:
                        images = [self._load_image(record, condition, base_seed) for record in batch_records]
                    else:
                        images = list(
                            executor.map(
                                lambda record: self._load_image(record, condition, base_seed),
                                batch_records,
                            )
                        )
                    inputs = self.processor(images=images, return_tensors="pt")
                    with self._autocast():
                        output = self._encode_image_batch(self._move_inputs(inputs))
                    batches.append(self._normalize(self._feature_tensor(output)))
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        return np.concatenate(batches, axis=0)

    @staticmethod
    def _load_image(record: CocoRecord, condition: ConditionRun, base_seed: int) -> Image.Image:
        with Image.open(Path(record.image_path)) as handle:
            source = handle.convert("RGB")
            return apply_distortion(source, condition, record.image_id, base_seed)


class BlipRetrievalAdapter(HuggingFaceAdapter):
    """Expose BLIP retrieval projection features without invoking the pairwise ITM head."""

    def _encode_text_batch(self, inputs: dict[str, Any]):
        output = self.model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        )
        return self.model.text_proj(output.last_hidden_state[:, 0, :])

    def _encode_image_batch(self, inputs: dict[str, Any]):
        output = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
        return self.model.vision_proj(output.last_hidden_state[:, 0, :])


class SiglipAdapter(HuggingFaceAdapter):
    # SigLIP was trained with fixed-length padding; Hugging Face recommends preserving it.
    text_padding = "max_length"


def load_model_adapter(model_spec: dict[str, Any], device: str, precision: str) -> HuggingFaceAdapter:
    try:
        import torch
        from transformers import AutoProcessor, BlipForImageTextRetrieval, CLIPModel, SiglipModel
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Transformers are required for model inference. Install requirements.txt first."
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is CUDA but PyTorch cannot access a CUDA GPU")
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16, or fp32")
    if device.startswith("cuda") and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU/PyTorch build does not support BF16; select fp16")
    model_type = model_spec["type"]
    checkpoint = model_spec["checkpoint"]
    classes = {"clip": CLIPModel, "blip": BlipForImageTextRetrieval, "siglip": SiglipModel}
    if model_type not in classes:
        raise ValueError(f"unsupported model type: {model_type}")
    dtype = None
    if device.startswith("cuda"):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    kwargs = {"dtype": dtype} if dtype is not None else {}
    processor_kwargs: dict[str, Any] = {}
    if model_spec.get("revision"):
        kwargs["revision"] = model_spec["revision"]
        processor_kwargs["revision"] = model_spec["revision"]
    processor = AutoProcessor.from_pretrained(checkpoint, **processor_kwargs)
    model = classes[model_type].from_pretrained(checkpoint, **kwargs).to(device)
    model.eval()
    adapter_classes = {"blip": BlipRetrievalAdapter, "siglip": SiglipAdapter}
    adapter_class = adapter_classes.get(model_type, HuggingFaceAdapter)
    return adapter_class(model, processor, device, precision)


def release_model(adapter: HuggingFaceAdapter) -> None:
    try:
        import torch

        del adapter.model
        del adapter.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return
