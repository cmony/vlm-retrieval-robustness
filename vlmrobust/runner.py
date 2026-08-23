from __future__ import annotations

import json
import hashlib
import csv
import logging
import os
import platform
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import __version__
from .config import build_condition_runs
from .data import CocoIndex, load_coco_index
from .models import load_model_adapter, release_model
from .retrieval import RankResult, compute_bidirectional_ranks


LOGGER = logging.getLogger("vlmrobust")
PROTOCOL_VERSION = "coco-retrieval-robustness-2026-08-09-v1"


def atomic_save_rank_result(path: str | Path, result: RankResult, metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".npz", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                image_to_text=np.asarray(result.image_to_text, dtype=np.int32),
                text_to_image=np.asarray(result.text_to_image, dtype=np.int32),
                caption_image_index=np.asarray(result.caption_image_index, dtype=np.int32),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_rank_result(path: str | Path) -> tuple[RankResult, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        result = RankResult(
            payload["image_to_text"].astype(np.int32),
            payload["text_to_image"].astype(np.int32),
            payload["caption_image_index"].astype(np.int64),
        )
        metadata = json.loads(str(payload["metadata"].item()))
    return result, metadata


def is_valid_checkpoint(path: str | Path, expected: dict[str, Any]) -> bool:
    try:
        result, metadata = load_rank_result(path)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        image_count = int(metadata["image_count"])
        caption_count = int(metadata["caption_count"])
        return (
            result.image_to_text.ndim == 1
            and result.text_to_image.ndim == 1
            and result.caption_image_index.ndim == 1
            and len(result.image_to_text) == image_count
            and len(result.text_to_image) == caption_count
            and len(result.caption_image_index) == caption_count
            and np.all((result.image_to_text >= 1) & (result.image_to_text <= caption_count))
            and np.all((result.text_to_image >= 1) & (result.text_to_image <= image_count))
            and np.all((result.caption_image_index >= 0) & (result.caption_image_index < image_count))
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def summarize_workload(
    image_count: int, caption_count: int, condition_run_count: int, model_count: int
) -> dict[str, int]:
    return {
        "images": image_count,
        "captions": caption_count,
        "condition_runs_per_model": condition_run_count,
        "models": model_count,
        "image_model_evaluations": image_count * condition_run_count * model_count,
        "text_model_evaluations": caption_count * model_count,
    }


def dataset_fingerprint(index: CocoIndex) -> str:
    digest = hashlib.sha256()
    for record in index.records:
        digest.update(str(record.image_id).encode("ascii"))
        digest.update(b"\0")
        for caption_id, caption in zip(record.caption_ids or range(len(record.captions)), record.captions):
            digest.update(str(caption_id).encode("ascii"))
            digest.update(b"\0")
            digest.update(caption.encode("utf-8"))
            digest.update(b"\0")
        if record.image_path.is_file():
            with record.image_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            digest.update(str(record.image_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_config(
    config: dict[str, Any],
    limit: int | None = None,
    model_names: list[str] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(config)
    if limit is not None:
        if limit <= 0 or limit > 5000:
            raise ValueError("--limit must be between 1 and 5000")
        resolved["data"]["limit"] = limit
        if output_dir is None:
            original = Path(resolved["experiment"]["output_dir"])
            resolved["experiment"]["output_dir"] = str(original.with_name(f"{original.name}_n{limit}"))
    if model_names:
        requested = set(model_names)
        available = {model["name"] for model in resolved["models"]}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"unknown model names: {', '.join(unknown)}")
        resolved["models"] = [model for model in resolved["models"] if model["name"] in requested]
    if output_dir is not None:
        resolved["experiment"]["output_dir"] = output_dir
    return resolved


def load_index_for_config(config: dict[str, Any]) -> CocoIndex:
    data = config["data"]
    return load_coco_index(
        data["image_dir"],
        data["captions_json"],
        limit=int(data["limit"]),
        sample_seed=int(config["experiment"]["seed"]),
        captions_per_image=int(data["captions_per_image"]),
    )


def _atomic_save_embeddings(path: Path, embeddings: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                embeddings=np.asarray(embeddings, dtype=np.float32),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_cached_embeddings(path: Path, expected: dict[str, Any]) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            embeddings = payload["embeddings"].astype(np.float32)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        if embeddings.ndim != 2 or len(embeddings) != int(expected["caption_count"]):
            return None
        return embeddings
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _environment_manifest(config: dict[str, Any], workload: dict[str, int]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "package_version": __version__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "workload": workload,
        "models": config["models"],
    }
    for package in ("numpy", "PIL", "yaml", "torch", "transformers"):
        try:
            module = __import__(package)
            manifest[f"version_{package}"] = getattr(module, "__version__", "unknown")
        except ImportError:
            manifest[f"version_{package}"] = "not-installed"
    try:
        import torch

        manifest["cuda_runtime"] = torch.version.cuda
        manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        manifest["cuda_runtime"] = None
        manifest["gpu"] = None
    return manifest


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def _write_dataset_index(path: Path, index: CocoIndex, fingerprint: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_fingerprint", "image_index", "image_id", "file_name",
                "caption_index", "caption_id", "caption",
            ],
        )
        writer.writeheader()
        for image_index, record in enumerate(index.records):
            caption_ids = record.caption_ids or tuple(range(len(record.captions)))
            for caption_index, (caption_id, caption) in enumerate(zip(caption_ids, record.captions)):
                writer.writerow(
                    {
                        "dataset_fingerprint": fingerprint,
                        "image_index": image_index,
                        "image_id": record.image_id,
                        "file_name": record.image_path.name,
                        "caption_index": caption_index,
                        "caption_id": caption_id,
                        "caption": caption,
                    }
                )


def run_experiment(config: dict[str, Any], resume: bool = True) -> Path:
    output_dir = Path(config["experiment"]["output_dir"]).expanduser().resolve()
    _configure_logging(output_dir)
    index = load_index_for_config(config)
    data_fingerprint = dataset_fingerprint(index)
    conditions = build_condition_runs(config)
    workload = summarize_workload(len(index.records), len(index.captions), len(conditions), len(config["models"]))
    manifest = _environment_manifest(config, workload)
    manifest["dataset_fingerprint"] = data_fingerprint
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["source_fingerprint"] = source_fingerprint()
    manifest["resolved_models"] = []
    software_fingerprint = hashlib.sha256(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "package_version", "protocol_version", "source_fingerprint", "version_numpy",
                    "version_PIL", "version_torch", "version_transformers",
                )
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest["software_fingerprint"] = software_fingerprint
    _write_dataset_index(output_dir / "dataset_index.csv", index, data_fingerprint)
    resolved_text = yaml.safe_dump(
        {key: value for key, value in config.items() if not key.startswith("_")}, sort_keys=False
    )
    canonical_config = output_dir / "resolved_config.yaml"
    if not canonical_config.exists():
        canonical_config.write_text(resolved_text, encoding="utf-8")
        resolved_config_path = canonical_config
    elif canonical_config.read_text(encoding="utf-8") == resolved_text:
        resolved_config_path = canonical_config
    else:
        config_history = output_dir / "resolved_configs"
        config_history.mkdir(parents=True, exist_ok=True)
        run_stamp = manifest["started_at_utc"].replace(":", "-").replace("+", "_")
        resolved_config_path = config_history / f"run-{run_stamp}.yaml"
        resolved_config_path.write_text(resolved_text, encoding="utf-8")
    manifest["resolved_config_file"] = str(resolved_config_path.relative_to(output_dir))
    manifest_path = output_dir / "run_manifest.json"
    prior_runs: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_runs = list(prior.get("runs", [])) if isinstance(prior, dict) else []
            if isinstance(prior, dict) and "started_at_utc" in prior:
                prior_runs.append(prior)
        except (OSError, ValueError, json.JSONDecodeError):
            prior_runs = []
    manifest_document = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_fingerprint": data_fingerprint,
        "runs": [*prior_runs, manifest],
    }
    manifest_path.write_text(json.dumps(manifest_document, indent=2), encoding="utf-8")
    runtime = config["runtime"]
    for model_spec in config["models"]:
        model_name = model_spec["name"]
        LOGGER.info("Loading model %s (%s)", model_name, model_spec["checkpoint"])
        adapter = load_model_adapter(model_spec, runtime["device"], runtime["precision"])
        try:
            resolved_revision = getattr(adapter.model.config, "_commit_hash", None) or model_spec.get("revision", "main")
            manifest["resolved_models"].append(
                {"name": model_name, "checkpoint": model_spec["checkpoint"], "resolved_revision": resolved_revision}
            )
            manifest_path.write_text(json.dumps(manifest_document, indent=2), encoding="utf-8")
            text_metadata = {
                "model": model_name,
                "checkpoint": model_spec["checkpoint"],
                "resolved_revision": resolved_revision,
                "model_type": model_spec["type"],
                "precision": runtime["precision"],
                "protocol_version": PROTOCOL_VERSION,
                "software_fingerprint": software_fingerprint,
                "caption_count": len(index.captions),
                "dataset_fingerprint": data_fingerprint,
            }
            text_path = output_dir / "embeddings" / model_name / "text_embeddings.npz"
            text_embeddings = _load_cached_embeddings(text_path, text_metadata) if resume else None
            if text_embeddings is None:
                LOGGER.info("Encoding %d captions for %s", len(index.captions), model_name)
                text_embeddings = adapter.encode_texts(index.captions, int(runtime["text_batch_size"]))
                _atomic_save_embeddings(text_path, text_embeddings, text_metadata)
            for run_number, condition in enumerate(conditions, start=1):
                expected = {
                    "model": model_name,
                    "checkpoint": model_spec["checkpoint"],
                    "resolved_revision": resolved_revision,
                    "model_type": model_spec["type"],
                    "precision": runtime["precision"],
                    "protocol_version": PROTOCOL_VERSION,
                    "software_fingerprint": software_fingerprint,
                    "experiment_seed": int(config["experiment"]["seed"]),
                    "condition": condition.slug,
                    "semantic_condition": condition.semantic_slug,
                    "family": condition.family,
                    "parameter": condition.parameter,
                    "value": condition.value,
                    "repeat": condition.repeat,
                    "image_count": len(index.records),
                    "caption_count": len(index.captions),
                    "dataset_fingerprint": data_fingerprint,
                }
                rank_path = output_dir / "ranks" / model_name / f"{condition.slug}.npz"
                if resume and is_valid_checkpoint(rank_path, expected):
                    LOGGER.info("[%d/%d] Resume skip: %s / %s", run_number, len(conditions), model_name, condition.slug)
                    continue
                started = time.perf_counter()
                LOGGER.info("[%d/%d] Encoding images: %s / %s", run_number, len(conditions), model_name, condition.slug)
                image_embeddings = adapter.encode_images(
                    index.records,
                    condition,
                    int(runtime["image_batch_size"]),
                    int(runtime.get("loader_workers", 0)),
                    int(config["experiment"]["seed"]),
                )
                ranks = compute_bidirectional_ranks(
                    image_embeddings,
                    text_embeddings,
                    index.caption_image_index,
                    int(runtime["similarity_chunk_size"]),
                    runtime["device"],
                )
                expected["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                atomic_save_rank_result(rank_path, ranks, expected)
                LOGGER.info("Saved %s in %.1f s", rank_path.name, expected["elapsed_seconds"])
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory. Reduce runtime.image_batch_size first; if ranking failed, "
                    "reduce runtime.similarity_chunk_size. Existing condition checkpoints remain reusable."
                ) from exc
            raise
        finally:
            release_model(adapter)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest_document, indent=2), encoding="utf-8")
    return output_dir
