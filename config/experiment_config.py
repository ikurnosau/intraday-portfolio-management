from datetime import datetime, time
from typing import Callable
from dataclasses import dataclass
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

@dataclass
class DataConfig: 
    symbol_or_symbols: str | list[str]
    start: datetime
    end: datetime

    features: dict[str, Callable]
    target: Callable
    normalizer: Callable
    missing_values_handler: Callable
    in_seq_len: int
    train_set_last_date: datetime
    multi_asset_prediction: bool

    batch_size: int
    shuffle: bool

    cutoff_time: datetime.time = None  # time-of-day cutoff for features


@dataclass
class ModelConfig: 
    model: torch.nn.Module
    registered_model_name: str


@dataclass
class TrainConfig: 
    loss_fn: torch.nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler
    num_epochs: int
    device: torch.device
    metrics: dict[str, Callable]
    save_path: str


@dataclass
class ObservabilityConfig: 
    experiment_name: str


@dataclass
class ExperimentConfig: 
    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig    
    observability_config: ObservabilityConfig
