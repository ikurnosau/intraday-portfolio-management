"""Load declarative training configuration from YAML."""

from __future__ import annotations

import copy
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import yaml
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.experiment_config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ObservabilityConfig,
    RLConfig,
    TrainConfig,
)
from data.processed.features_polars import (
    CloseOpenReturn,
    Column,
    HighLowRange,
    LocationInRange,
    LogReturn,
    LogVolumeChange,
    MovingAverageSlope,
    RealizedVolatility,
    RelativeSpread as RelativeSpreadFeature,
    RollingVWAPDistance,
    TimeOfDayCyclic,
    VolatilitySlope,
    VWAPDistance,
)
from data.processed.indicators_polars import EMA, RSI, SMA
from data.processed.missing_values_handling import ContinuousForwardFillPolars
from data.processed.normalization import MinMaxNormalizerOverWindow
from data.processed.statistics import (
    FutureReturn,
    RelativeSpread as RelativeSpreadStatistic,
    RollingVolatility,
)
from data.processed.targets import (
    FutureMeanReturnClassification,
    ReturnOverHorizon,
    TripleClassification,
)
from data.raw.retrievers.alpaca_markets_retriever import AlpacaMarketsRetriever
from modeling.loss import PositionReturnLoss, RiskAdjustedPositionReturnLoss
from modeling.metrics import MeanReturn, rmse_regression
from modeling.models.tsa_allocator import TSAllocator
from modeling.models.tsa_classifier import TemporalSpatial


# Increment SCHEMA_VERSION when the YAML structure or its interpretation
# changes. Increment CONFIG_REVISION whenever the behavior of any registered
# feature, statistic, target, normalizer, model, loss, or metric changes.
# These values are maintained manually; model artifacts should also record the
# Git commit SHA to identify the exact source code used during training.
SCHEMA_VERSION = 7
CONFIG_REVISION = 15
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "train_config.yaml"

T = TypeVar("T")
Factory = Callable[..., T]


def _rmse_metric(**_: Any) -> Callable[..., float]:
    return rmse_regression


FEATURES: dict[str, Factory[Any]] = {
    "log_return": LogReturn,
    "high_low_range": HighLowRange,
    "close_open_return": CloseOpenReturn,
    "log_volume_change": LogVolumeChange,
    "sma": SMA,
    "ema": EMA,
    "rsi": RSI,
    "realized_volatility": RealizedVolatility,
    "rolling_vwap_distance": RollingVWAPDistance,
    "vwap_distance": VWAPDistance,
    "location_in_range": LocationInRange,
    "time_of_day_cyclic": TimeOfDayCyclic,
    "moving_average_slope": MovingAverageSlope,
    "volatility_slope": VolatilitySlope,
    "column": Column,
    "relative_spread": RelativeSpreadFeature,
}

STATISTICS: dict[str, Factory[Any]] = {
    "future_return": FutureReturn,
    "rolling_volatility": RollingVolatility,
    "relative_spread": RelativeSpreadStatistic,
}

TARGETS: dict[str, Factory[Any]] = {
    "future_mean_return_classification": FutureMeanReturnClassification,
    "triple_classification": TripleClassification,
    "return_over_horizon": ReturnOverHorizon,
}

NORMALIZERS: dict[str, Factory[Any]] = {
    "min_max_over_window": MinMaxNormalizerOverWindow,
}

MISSING_VALUE_HANDLERS: dict[str, Factory[Any]] = {
    "continuous_forward_fill_polars": ContinuousForwardFillPolars,
}

RETRIEVERS: dict[str, Factory[Any]] = {
    "alpaca_markets": AlpacaMarketsRetriever,
}

MODELS: dict[str, Factory[torch.nn.Module]] = {
    "temporal_spatial": TemporalSpatial,
    "temporal_spatial_allocator": TSAllocator,
}

LOSSES: dict[str, Factory[Any]] = {
    "mse": torch.nn.MSELoss,
    "risk_adjusted_position_return": RiskAdjustedPositionReturnLoss,
}

METRICS: dict[str, Factory[Any]] = {
    "rmse_regression": _rmse_metric,
    "position_return": PositionReturnLoss,
    "mean_return": MeanReturn,
}

OPTIMIZERS: dict[str, Factory[torch.optim.Optimizer]] = {
    "adamw": torch.optim.AdamW,
}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration value: {path}.{key}")
    return mapping[key]


def _build(
    spec: Any,
    registry: dict[str, Factory[T]],
    path: str,
    extra_params: dict[str, Any] | None = None,
) -> T:
    spec = _mapping(spec, path)
    name = _required(spec, "name", path)
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown component '{name}' at {path}; available: {available}"
        )
    params = copy.deepcopy(spec.get("params", {}))
    if not isinstance(params, dict):
        raise ValueError(f"{path}.params must be a mapping")
    if extra_params:
        for key, value in extra_params.items():
            params.setdefault(key, value)
    try:
        return registry[name](**params)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {path} ({name}): {exc}") from exc


def _timeframe(spec: Any) -> TimeFrame:
    spec = _mapping(spec, "data.frequency")
    amount = int(_required(spec, "amount", "data.frequency"))
    unit_name = str(_required(spec, "unit", "data.frequency"))
    try:
        unit = getattr(TimeFrameUnit, unit_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown timeframe unit: {unit_name}") from exc
    return TimeFrame(amount=amount, unit=unit)


def _datetime(value: Any, path: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path} is not a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{path} must include a timezone offset")
    return parsed


def _named_components(
    specs: Any,
    registry: dict[str, Factory[Any]],
    path: str,
) -> dict[str, Any]:
    if not isinstance(specs, list):
        raise ValueError(f"{path} must be a list to preserve feature order")
    result: dict[str, Any] = {}
    for index, item in enumerate(specs):
        item = _mapping(item, f"{path}[{index}]")
        output_name = _required(item, "output", f"{path}[{index}]")
        if output_name in result:
            raise ValueError(f"Duplicate output '{output_name}' at {path}")
        result[output_name] = _build(item, registry, f"{path}[{index}]")
    return result


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
            and not (
                "name" in value
                and value.get("name") != merged[key].get("name")
            )
        ):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path, ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    with path.open(encoding="utf-8") as config_file:
        raw = _mapping(yaml.safe_load(config_file), "config")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError("config.extends must be a path string")
    parent_path = (path.parent / parent).resolve()
    return _merge(
        _load_yaml(parent_path, (*ancestors, path)),
        raw,
    )


def load_train_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ExperimentConfig:
    """Load, validate, and instantiate a training configuration."""

    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml(config_path)

    schema_version = _required(raw, "schema_version", "config")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_version}; "
            f"this code supports {SCHEMA_VERSION}"
        )

    revision = _required(raw, "config_revision", "config")
    if revision != CONFIG_REVISION:
        warnings.warn(
            f"Config revision {revision} does not match code revision "
            f"{CONFIG_REVISION}",
            RuntimeWarning,
            stacklevel=2,
        )

    data = _mapping(_required(raw, "data", "config"), "data")
    frequency = _timeframe(_required(data, "frequency", "data"))
    symbols = _required(data, "symbols", "data")
    if not isinstance(symbols, list) or not all(
        isinstance(symbol, str) for symbol in symbols
    ):
        raise ValueError("data.symbols must be a list of strings")

    features = _named_components(
        _required(data, "features", "data"),
        FEATURES,
        "data.features",
    )
    statistics = _named_components(
        _required(data, "statistics", "data"),
        STATISTICS,
        "data.statistics",
    )

    data_config = DataConfig(
        retriever=_build(
            _required(data, "retriever", "data"),
            RETRIEVERS,
            "data.retriever",
            {"timeframe": frequency},
        ),
        symbol_or_symbols=symbols,
        frequency=frequency,
        start=_datetime(_required(data, "start", "data"), "data.start"),
        end=_datetime(_required(data, "end", "data"), "data.end"),
        train_set_last_date=_datetime(
            _required(data, "train_set_last_date", "data"),
            "data.train_set_last_date",
        ),
        val_set_last_date=_datetime(
            _required(data, "val_set_last_date", "data"),
            "data.val_set_last_date",
        ),
        features_polars=features,
        statistics=statistics,
        target=_build(
            _required(data, "target", "data"), TARGETS, "data.target"
        ),
        normalizer=_build(
            _required(data, "normalizer", "data"),
            NORMALIZERS,
            "data.normalizer",
        ),
        missing_values_handler_polars=_build(
            _required(data, "missing_values_handler", "data"),
            MISSING_VALUE_HANDLERS,
            "data.missing_values_handler",
            {"frequency": str(frequency)},
        ),
        in_seq_len=int(_required(data, "in_seq_len", "data")),
        horizon=int(_required(data, "horizon", "data")),
        multi_asset_prediction=bool(
            _required(data, "multi_asset_prediction", "data")
        ),
        validator=None,
    )

    model = _mapping(_required(raw, "model", "config"), "model")
    model_instance = _build(
        model,
        MODELS,
        "model",
        {
            "input_dim": len(features),
            "num_assets": len(symbols),
        },
    )
    model_config = ModelConfig(model=model_instance)

    train = _mapping(_required(raw, "train", "config"), "train")
    optimizer = _build(
        _required(train, "optimizer", "train"),
        OPTIMIZERS,
        "train.optimizer",
        {"params": model_instance.parameters()},
    )
    metrics = _named_components(
        _required(train, "metrics", "train"),
        METRICS,
        "train.metrics",
    )
    train_config = TrainConfig(
        loss_fn=_build(
            _required(train, "loss", "train"), LOSSES, "train.loss"
        ),
        optimizer=optimizer,
        scheduler=copy.deepcopy(
            _required(train, "scheduler", "train")
        ),
        num_epochs=int(_required(train, "num_epochs", "train")),
        early_stopping_patience=int(
            _required(train, "early_stopping_patience", "train")
        ),
        device=torch.device(_required(train, "device", "train")),
        cudnn_benchmark=bool(
            _required(train, "cudnn_benchmark", "train")
        ),
        metrics=metrics,
        batch_size=int(_required(train, "batch_size", "train")),
        shuffle=bool(_required(train, "shuffle", "train")),
        num_workers=int(_required(train, "num_workers", "train")),
        prefetch_factor=int(_required(train, "prefetch_factor", "train")),
        pin_memory=bool(_required(train, "pin_memory", "train")),
        persistent_workers=bool(
            _required(train, "persistent_workers", "train")
        ),
        drop_last=bool(_required(train, "drop_last", "train")),
        save_path=str(_required(train, "save_path", "train")),
    )

    rl = _mapping(_required(raw, "rl", "config"), "rl")
    rl_config = RLConfig(
        trajectory_length=int(
            _required(rl, "trajectory_length", "rl")
        ),
        fee=float(_required(rl, "fee", "rl")),
        spread_multiplier=float(
            _required(rl, "spread_multiplier", "rl")
        ),
        trade_asset_count=int(
            _required(rl, "trade_asset_count", "rl")
        ),
        allow_short_positions=bool(
            _required(rl, "allow_short_positions", "rl")
        ),
    )

    observability = _mapping(
        _required(raw, "observability", "config"),
        "observability",
    )
    observability_config = ObservabilityConfig(
        experiment_name=str(
            _required(
                observability,
                "experiment_name",
                "observability",
            )
        )
    )

    return ExperimentConfig(
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        rl_config=rl_config,
        observability_config=observability_config,
        raw_config=raw,
    )
