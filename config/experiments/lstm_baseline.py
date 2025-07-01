import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from datetime import datetime, timezone, time
import torch

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification
from data.processed.normalization import MinMaxNormalizer
from data.processed.missing_values_handling import ForwardFillFlatBars
from modeling.models.tsa_classifier import TemporalSpatialClassifier
from modeling.models.lstm import LSTMClassifier
from modeling.metrics import accuracy_multi_asset, rmse_multi_asset, accuracy, rmse

data_config = DataConfig(
    symbol_or_symbols=['AAPL'],
    start=datetime(2024, 6, 1),
    end=datetime(2025, 6, 1),

    features={
        "close": lambda data: data['close'],
        "volume": lambda data: data['volume'],
        # "log_volume_ret": LogVolumeReturn(),
        "return": lambda data: data['close'].pct_change()
    },
    target=Balanced3ClassClassification(base_feature='close'),
    normalizer=MinMaxNormalizer(),
    missing_values_handler=ForwardFillFlatBars(),
    train_set_last_date=datetime(2025, 5, 1, tzinfo=timezone.utc), 
    in_seq_len=60,
    multi_asset_prediction=False,

    batch_size=32,
    shuffle=True,

    cutoff_time=time(hour=14, minute=10),

)

NUM_ASSETS = len(data_config.symbol_or_symbols) - 1

model_config=ModelConfig(
    model=LSTMClassifier(
        input_dim=3,
        n_class=3, 
        hidden_dim=64,
        num_layers=2, 
        bidirectional=True,
        layer_norm=True,
        dropout=0.2
    ),
    # model=MLPClassifier(
    #     input_dim=37,
    #     n_class=3,
    #     hidden_dims=[128, 64],
    #     dropout=0.0, 
    #     activation=torch.nn.ReLU(inplace=True),
    #     batch_norm=False
    # ),
    # model=TemporalSpatialClassifier(
    #     input_dim=3,
    #     hidden_dim=128,
    #     n_class=3,
    #     lstm_layers=2,
    #     bidirectional=True,
    #     num_heads=8,
    #     dropout=0.2,
    #     use_spatial_attention=True,
    #     num_assets=NUM_ASSETS,
    #     asset_embed_dim=16
    # ),
    registered_model_name="LSTM Default"
)

cur_optimizer = torch.optim.AdamW(
    model_config.model.parameters(), 
    lr=1e-3,  # This will be overridden by OneCycleLR
    weight_decay=1e-10,
    amsgrad=True)

train_config=TrainConfig(
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=cur_optimizer,
    scheduler=torch.optim.lr_scheduler.StepLR(
        optimizer=cur_optimizer,
        step_size=3,   # epochs between LR drops (≈ 3 drops in 10-epoch run)
        gamma=0.5      # multiply LR by this factor at each step
    ),
    metrics={ "accuracy": accuracy, "rmse": rmse },
    num_epochs=10,  # Slightly longer tail for stability
    device=torch.device("cuda"),
    save_path=""
)

observability_config = ObservabilityConfig(
    experiment_name="Temporal Dependency Extraction 1 asset"
)

config = ExperimentConfig(
    data_config=data_config,
    model_config=model_config,
    train_config=train_config,
    observability_config=observability_config
)