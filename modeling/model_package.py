from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from config.experiment_config import ExperimentConfig
from config.train_config import load_train_config
from core_inference.allocators.signal_predictor_allocator import (
    SignalPredictorAllocator,
)
from data.object_store import B2ObjectStore


MODEL_PACKAGE_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1
ALLOCATOR_CONFIG_VERSION = 1
PACKAGE_FILE_NAMES = (
    "model.pt",
    "config.yaml",
    "allocator_config.json",
    "metadata.json",
)
COMPLETION_FILE_NAME = "COMPLETED"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetReference:
    object_store_uri: str
    version_id: str | None
    size: int
    etag: str | None
    fingerprint: str
    query: dict[str, Any]


@dataclass(frozen=True)
class PublishedModelPackage:
    object_store_uri: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedModelPackage:
    allocator: SignalPredictorAllocator
    config: ExperimentConfig
    metadata: dict[str, Any]


def build_dataset_reference(
    object_store: B2ObjectStore,
    object_key: str,
    query: Mapping[str, Any],
) -> DatasetReference:
    object_metadata = object_store.metadata(object_key)
    if not object_metadata.version_id:
        raise ValueError(
            "Dataset object has no version ID; immutable provenance is required"
        )
    identity = {
        "object_store_uri": object_metadata.uri,
        "version_id": object_metadata.version_id,
        "size": object_metadata.size,
        "query": copy.deepcopy(dict(query)),
    }
    return DatasetReference(
        object_store_uri=object_metadata.uri,
        version_id=object_metadata.version_id,
        size=object_metadata.size,
        etag=object_metadata.etag,
        fingerprint=_fingerprint(identity),
        query=identity["query"],
    )


def publish_model_package(
    *,
    model: torch.nn.Module,
    train_config_path: str | Path,
    allocator_params: Mapping[str, Any],
    dataset_reference: DatasetReference,
    final_metrics: Mapping[str, Any],
    wandb_run_id: str,
    object_store: B2ObjectStore,
) -> PublishedModelPackage:
    package_key = f"models/{wandb_run_id}"
    object_store_uri = object_store.uri_for_key(package_key)

    for file_name in (*PACKAGE_FILE_NAMES, COMPLETION_FILE_NAME):
        key = f"{package_key}/{file_name}"
        if object_store.exists(key):
            raise FileExistsError(f"Immutable model package already exists: {key}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        package_directory = Path(temporary_directory)
        _write_checkpoint(model, package_directory / "model.pt")
        shutil.copyfile(train_config_path, package_directory / "config.yaml")
        _write_json(
            package_directory / "allocator_config.json",
            _allocator_config(allocator_params),
        )

        payload_hashes = {
            file_name: _file_identity(package_directory / file_name)
            for file_name in PACKAGE_FILE_NAMES
            if file_name != "metadata.json"
        }
        metadata = {
            "model_package_format_version": MODEL_PACKAGE_FORMAT_VERSION,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model_format": "pytorch_state_dict",
            "wandb_run_id": wandb_run_id,
            "object_store_uri": object_store_uri,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "pytorch_version": torch.__version__,
            "dataset": asdict(dataset_reference),
            "final_metrics": _json_compatible(dict(final_metrics)),
            "files": payload_hashes,
        }
        _write_json(package_directory / "metadata.json", metadata)

        completion = {
            "model_package_format_version": MODEL_PACKAGE_FORMAT_VERSION,
            "files": {
                file_name: _file_identity(package_directory / file_name)
                for file_name in PACKAGE_FILE_NAMES
            },
        }
        _write_json(package_directory / COMPLETION_FILE_NAME, completion)

        for file_name in PACKAGE_FILE_NAMES:
            object_store.upload_file(
                f"{package_key}/{file_name}",
                package_directory / file_name,
            )
        object_store.upload_file(
            f"{package_key}/{COMPLETION_FILE_NAME}",
            package_directory / COMPLETION_FILE_NAME,
        )

    return PublishedModelPackage(
        object_store_uri=object_store_uri,
        metadata=metadata,
    )


def download_model_package(
    *,
    object_store: B2ObjectStore,
    object_store_uri: str,
    destination: str | Path,
) -> Path:
    package_key = object_store.key_from_uri(object_store_uri)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    completion_path = destination / COMPLETION_FILE_NAME
    object_store.download_file(
        f"{package_key}/{COMPLETION_FILE_NAME}",
        completion_path,
    )
    completion = _read_json(completion_path)
    if completion.get("model_package_format_version") != MODEL_PACKAGE_FORMAT_VERSION:
        raise ValueError("Unsupported model package format version")

    file_entries = completion.get("files")
    if not isinstance(file_entries, dict) or set(file_entries) != set(PACKAGE_FILE_NAMES):
        raise ValueError("Model package completion manifest has unexpected files")

    for file_name in PACKAGE_FILE_NAMES:
        object_store.download_file(
            f"{package_key}/{file_name}",
            destination / file_name,
        )
        expected = file_entries[file_name]
        actual = _file_identity(destination / file_name)
        if actual != expected:
            raise ValueError(f"Integrity check failed for model package file: {file_name}")

    return destination


def load_model_package(
    package_directory: str | Path,
    device: torch.device,
) -> LoadedModelPackage:
    package_directory = Path(package_directory)
    config = load_train_config(package_directory / "config.yaml")
    checkpoint = torch.load(
        package_directory / "model.pt",
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version")

    model = config.model_config.model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    allocator_config = _read_json(package_directory / "allocator_config.json")
    if allocator_config.get("allocator_config_version") != ALLOCATOR_CONFIG_VERSION:
        raise ValueError("Unsupported allocator configuration version")
    if allocator_config.get("allocator_type") != "signal_predictor_allocator":
        raise ValueError("Unsupported allocator type")

    allocator = SignalPredictorAllocator(
        signal_predictor=model,
        trade_asset_count=int(allocator_config["trade_asset_count"]),
        select_from_n_best=int(allocator_config["select_from_n_best"]),
        confidence_threshold=float(allocator_config["confidence_threshold"]),
        allow_short_positions=config.rl_config.allow_short_positions,
    )
    allocator.to(device).eval()

    return LoadedModelPackage(
        allocator=allocator,
        config=config,
        metadata=_read_json(package_directory / "metadata.json"),
    )


def _write_checkpoint(model: torch.nn.Module, path: Path) -> None:
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    clean_state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in module.state_dict().items()
    }
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": copy.deepcopy(clean_state_dict),
        },
        path,
    )


def _allocator_config(allocator_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "allocator_config_version": ALLOCATOR_CONFIG_VERSION,
        "allocator_type": "signal_predictor_allocator",
        "trade_asset_count": int(allocator_params["trade_asset_count"]),
        "select_from_n_best": int(allocator_params["select_from_n_best"]),
        "confidence_threshold": float(allocator_params["confidence_threshold"]),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file_object:
        for chunk in iter(lambda: file_object.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def _fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _json_compatible(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_compatible(dict(value)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
