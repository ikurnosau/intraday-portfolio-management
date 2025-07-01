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
from modeling.metrics import accuracy_multi_asset, rmse_multi_asset

data_config = DataConfig(
    symbol_or_symbols=Constants.Data.MOST_LIQUID_TECH_STOCKS,
    # start=datetime(2025, 1, 1), 
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
    multi_asset_prediction=True,

    batch_size=32,
    shuffle=True,

    cutoff_time=time(hour=14, minute=10),

)

NUM_ASSETS = len(data_config.symbol_or_symbols) - 1

model_config=ModelConfig(
    # model=LSTMClassifier(
    #     input_dim=3,
    #     n_class=3, 
    #     hidden_dim=64,
    #     num_layers=1, 
    #     bidirectional=True,
    #     dropout=0.0
    # ),
    # model=MLPClassifier(
    #     input_dim=37,
    #     n_class=3,
    #     hidden_dims=[128, 64],
    #     dropout=0.0, 
    #     activation=torch.nn.ReLU(inplace=True),
    #     batch_norm=False
    # ),
    model=TemporalSpatialClassifier(
        input_dim=3,
        hidden_dim=128,
        n_class=3,
        lstm_layers=2,
        bidirectional=True,
        num_heads=8,
        dropout=0.2,
        use_spatial_attention=True,
        num_assets=NUM_ASSETS,
        asset_embed_dim=16
    ),
    registered_model_name="TSA With Spatial Attention"
)

cur_optimizer = torch.optim.AdamW(
    model_config.model.parameters(), 
    lr=1e-3,  # This will be overridden by OneCycleLR
    weight_decay=1e-10,
    amsgrad=True)

train_config=TrainConfig(
    loss_fn=lambda outputs, targets: torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)),
        targets.view(-1)
    ),
    optimizer=cur_optimizer,
    scheduler={
        'type': 'OneCycleLR',
        'max_lr': 5e-3,        # Milder peak LR (empirically worked better)
        'pct_start': 0.1,      # 10 % warm-up — quick ramp then long decay
        'div_factor': 25,      # Start LR = max_lr / 25  (2e-4)
        'final_div_factor': 1e3, # End LR  = max_lr / 1 000 (5e-6) for steadier tail
        'anneal_strategy': 'cos',
        'three_phase': False,
        'cycle_momentum': False  # Disable momentum cycling when using AdamW
    },
    metrics={"accuracy": accuracy_multi_asset, "rmse": rmse_multi_asset},
    num_epochs=10,  # Slightly longer tail for stability
    device=torch.device("cuda"),
    save_path=""
)

observability_config = ObservabilityConfig(
    experiment_name="TSA With Spatial Attention"
)

config = ExperimentConfig(
    data_config=data_config,
    model_config=model_config,
    train_config=train_config,
    observability_config=observability_config
)