from datetime import datetime, time
from typing import Callable
from dataclasses import dataclass
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from alpaca.data.timeframe import TimeFrame
from typing import Any
from core_data_prep.validations import Validator
from modeling.rl.actors.base_actor import BaseActor

@dataclass
class DataConfig: 
    retriever: Any
    symbol_or_symbols: str | list[str]
    frequency: TimeFrame

    start: datetime
    end: datetime
    train_set_last_date: datetime 
    val_set_last_date: datetime

    features_polars: dict[str, Callable]
    statistics: dict[str, Callable]
    target: Callable
    normalizer: Callable
    missing_values_handler_polars: Callable
    in_seq_len: int
    horizon: int

    multi_asset_prediction: bool

    validator: Validator

@dataclass
class ModelConfig: 
    model: torch.nn.Module
    registered_model_name: str


@dataclass
class TrainConfig:
    # --- optimisation ------------------------------------------------------
    loss_fn: torch.nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler | dict  # allow passing configuration dicts
    num_epochs: int
    early_stopping_patience: int

    # --- hardware / runtime ------------------------------------------------
    device: torch.device
    cudnn_benchmark: bool = False  # enable cuDNN autotuner for fixed shapes

    # --- metric callbacks ---------------------------------------------------
    metrics: dict[str, Callable] | None = None

    # --- DataLoader parameters ---------------------------------------------
    batch_size: int
    shuffle: bool 
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool

    # --- misc --------------------------------------------------------------
    save_path: str


@dataclass
class RLConfig:
    trajectory_length: int
    fee: float
    spread_multiplier: float
    trade_asset_count: int


@dataclass
class ObservabilityConfig: 
    experiment_name: str


@dataclass
class ExperimentConfig: 
    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig    
    rl_config: RLConfig
    observability_config: ObservabilityConfig
