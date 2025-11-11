from datetime import datetime, time
from typing import Callable
from dataclasses import dataclass
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from alpaca.data.timeframe import TimeFrame

@dataclass
class DataConfig: 
    symbol_or_symbols: str | list[str]
    frequency: TimeFrame

    start: datetime
    end: datetime
    train_set_last_date: datetime 
    val_set_last_date: datetime

    features: dict[str, Callable]
    statistics: dict[str, Callable]
    target: Callable
    normalizer: Callable
    missing_values_handler: Callable
    in_seq_len: int
    horizon: int

    multi_asset_prediction: bool


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
    batch_size: int | None = None           # fallback to DataConfig.batch_size
    shuffle: bool | None = None             # fallback to DataConfig.shuffle
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False

    # --- misc --------------------------------------------------------------
    save_path: str | None = None


@dataclass
class ObservabilityConfig: 
    experiment_name: str


@dataclass
class ExperimentConfig: 
    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig    
    observability_config: ObservabilityConfig
