from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import wandb
from wandb.errors import CommError

from config.experiment_config import ExperimentConfig
from config.settings import Settings, WandbSettings
from data.object_store import B2ObjectStore
from modeling.model_package import (
    DatasetReference,
    LoadedModelPackage,
    PublishedModelPackage,
    download_model_package,
    load_model_package,
    publish_model_package,
)


MODEL_ARTIFACT_NAME = "portfolio-allocation-model"


@dataclass(frozen=True)
class PublishedWandbModel:
    source_artifact_name: str
    object_store_uri: str
    package_metadata: dict[str, Any]


@dataclass(frozen=True)
class ResolvedProductionModel:
    resolved_artifact_name: str
    package: LoadedModelPackage


def start_training_run(
    config: ExperimentConfig,
    settings: WandbSettings,
) -> wandb.Run:
    if config.raw_config is None:
        raise ValueError("Training config has no serializable source manifest")

    _configure_wandb_auth(settings)
    init_args: dict[str, Any] = {
        "project": settings.project,
        "job_type": "training",
        "group": config.observability_config.experiment_name,
        "config": config.raw_config,
    }
    run = wandb.init(**init_args)
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("validation/*", step_metric="epoch")
    return run


def log_epoch_metrics(
    run: wandb.Run,
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    run.log(
        {
            "epoch": epoch,
            **{
                _metric_name(name): _scalar(value)
                for name, value in metrics.items()
            },
        }
    )


def log_test_sequences(
    run: wandb.Run,
    *,
    realized_returns: Any,
    allocations: Any,
    asset_names: Sequence[str],
    confidences: Any | None = None,
) -> None:
    returns = _as_numpy(realized_returns).reshape(-1)
    allocation_values = _as_numpy(allocations)
    if allocation_values.ndim != 2:
        raise ValueError("Test allocations must have shape (step, asset)")
    if allocation_values.shape[0] != returns.shape[0]:
        raise ValueError("Test returns and allocations must have the same step count")
    if allocation_values.shape[1] != len(asset_names):
        raise ValueError("Asset names must match the allocation asset dimension")

    confidence_values = None
    if confidences is not None:
        confidence_values = _as_numpy(confidences).reshape(-1)
        if confidence_values.shape[0] != returns.shape[0]:
            raise ValueError(
                "Test confidences and returns must have the same step count"
            )

    cumulative_wealth = np.cumprod(1.0 + returns) - 1.0
    run.define_metric("test/step")
    run.define_metric("test/realized_return", step_metric="test/step")
    run.define_metric("test/cumulative_wealth", step_metric="test/step")
    run.define_metric("test/confidence", step_metric="test/step")
    run.define_metric("test/allocation/*", step_metric="test/step")

    for step, realized_return in enumerate(returns):
        metrics = {
            "test/step": step,
            "test/realized_return": float(realized_return),
            "test/cumulative_wealth": float(cumulative_wealth[step]),
            **{
                f"test/allocation/{asset_name}": float(allocation)
                for asset_name, allocation in zip(
                    asset_names,
                    allocation_values[step],
                    strict=True,
                )
            },
        }
        if confidence_values is not None:
            metrics["test/confidence"] = float(confidence_values[step])
        run.log(metrics)


def publish_training_result(
    *,
    run: wandb.Run,
    model: torch.nn.Module,
    train_config_path: str | Path,
    allocator_params: Mapping[str, Any],
    dataset_reference: DatasetReference,
    final_metrics: Mapping[str, Any],
    settings: Settings,
) -> PublishedWandbModel:
    object_store = B2ObjectStore.from_settings(settings.b2, key_prefix="")
    _configure_wandb_s3(settings)

    try:
        package = publish_model_package(
            model=model,
            train_config_path=train_config_path,
            allocator_params=allocator_params,
            dataset_reference=dataset_reference,
            final_metrics=final_metrics,
            wandb_run_id=run.id,
            object_store=object_store,
        )
        _update_summary(run, allocator_params, final_metrics, dataset_reference)
        logged_artifact = _log_model_artifact(
            run,
            package,
            dataset_reference,
            final_metrics,
        )
        logged_artifact.wait()
        source_artifact_name = logged_artifact.qualified_name
        run.finish()
    except Exception:
        run.finish(exit_code=1)
        raise

    return PublishedWandbModel(
        source_artifact_name=source_artifact_name,
        object_store_uri=package.object_store_uri,
        package_metadata=package.metadata,
    )


def link_model_artifact_to_registry(
    source_artifact_name: str,
    settings: WandbSettings,
) -> str:
    _configure_wandb_auth(settings)
    artifact = wandb.Api().artifact(source_artifact_name)
    try:
        linked_artifact = artifact.link(
            target_path=settings.registry_collection_path,
        )
    except CommError as error:
        if "cannot create registry" in str(error).lower():
            raise RuntimeError(
                f"Create the W&B Registry '{settings.registry}' in the "
                "W&B UI before linking model artifacts"
            ) from error
        raise
    return linked_artifact.qualified_name


def promote_registry_artifact_to_production(
    linked_artifact_name: str,
    settings: WandbSettings,
) -> str:
    _configure_wandb_auth(settings)
    artifact = wandb.Api().artifact(linked_artifact_name)
    artifact.aliases = list(dict.fromkeys([*artifact.aliases, "production"]))
    artifact.save()
    return settings.production_artifact_path


def load_production_model(
    *,
    settings: Settings,
    device: torch.device,
) -> ResolvedProductionModel:
    _configure_wandb_auth(settings.wandb)
    artifact = wandb.Api().artifact(settings.wandb.production_artifact_path)
    object_store_uri = artifact.metadata.get("object_store_uri")
    if not isinstance(object_store_uri, str) or not object_store_uri:
        raise ValueError("Production artifact has no object_store_uri metadata")

    object_store = B2ObjectStore.from_settings(settings.b2, key_prefix="")
    with tempfile.TemporaryDirectory() as temporary_directory:
        package_directory = download_model_package(
            object_store=object_store,
            object_store_uri=object_store_uri,
            destination=temporary_directory,
        )
        package = load_model_package(package_directory, device)

    return ResolvedProductionModel(
        resolved_artifact_name=artifact.qualified_name,
        package=package,
    )


def _log_model_artifact(
    run: wandb.Run,
    package: PublishedModelPackage,
    dataset_reference: DatasetReference,
    final_metrics: Mapping[str, Any],
) -> wandb.Artifact:
    package_metadata = package.metadata
    artifact = wandb.Artifact(
        name=MODEL_ARTIFACT_NAME,
        type="model",
        metadata={
            "object_store_uri": package.object_store_uri,
            "model_package_format_version": package_metadata[
                "model_package_format_version"
            ],
            "checkpoint_format_version": package_metadata[
                "checkpoint_format_version"
            ],
            "git_commit": package_metadata["git_commit"],
            "pytorch_version": package_metadata["pytorch_version"],
            "dataset_fingerprint": dataset_reference.fingerprint,
            "dataset_object_store_uri": dataset_reference.object_store_uri,
            "dataset_version_id": dataset_reference.version_id,
            "final_metrics": {
                key: _scalar(value)
                for key, value in final_metrics.items()
            },
        },
    )
    artifact.add_reference(package.object_store_uri, checksum=True)
    return run.log_artifact(artifact)


def _update_summary(
    run: wandb.Run,
    allocator_params: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    dataset_reference: DatasetReference,
) -> None:
    for name, value in final_metrics.items():
        run.summary[f"evaluation/{name}"] = _scalar(value)
    for name, value in allocator_params.items():
        run.summary[f"allocator/{name}"] = _scalar(value)
    run.summary["dataset/fingerprint"] = dataset_reference.fingerprint
    run.summary["dataset/object_store_uri"] = dataset_reference.object_store_uri
    run.summary["dataset/version_id"] = dataset_reference.version_id


def _configure_wandb_s3(settings: Settings) -> None:
    os.environ["AWS_S3_ENDPOINT_URL"] = settings.b2.endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = settings.b2.access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.b2.secret_access_key
    os.environ["AWS_REGION"] = settings.b2.region


def _configure_wandb_auth(settings: WandbSettings) -> None:
    os.environ["WANDB_API_KEY"] = settings.api_key


def _metric_name(name: str) -> str:
    if name.startswith("train_"):
        return f"train/{name.removeprefix('train_')}"
    if name.startswith("val_"):
        return f"validation/{name.removeprefix('val_')}"
    return name.replace("_", "/", 1)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
