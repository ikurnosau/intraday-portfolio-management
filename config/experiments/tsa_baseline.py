import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from datetime import datetime, timezone, time
import torch

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification, BinaryClassification
from data.processed.normalization import ZScoreOverWindowNormalizer, ZScoreNormalizer, MinMaxNormalizer
from data.processed.missing_values_handling import ForwardFillFlatBars
from modeling.models.mlp import MLPClassifier
from modeling.models.lstm import LSTMClassifier
from modeling.models.tsa_classifier import TemporalSpatialClassifier
from modeling.metrics import accuracy, rmse, accuracy_multi_asset, rmse_multi_asset

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
        hidden_dim=64,
        n_class=3,
        lstm_layers=1,
        bidirectional=False,
        num_heads=4,
        dropout=0.1,
        use_spatial_attention=True
    ),
    registered_model_name="TSA With Spatial Attention"
)

cur_optimizer = torch.optim.AdamW(
    model_config.model.parameters(), 
    lr=1e-3,
    weight_decay=1e-10,
    amsgrad=True)

train_config=TrainConfig(
    loss_fn=lambda outputs, targets: torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)),  # Reshape to [batch_size * num_assets, num_classes]
        targets.view(-1)  # Reshape to [batch_size * num_assets]
    ),
    optimizer=cur_optimizer,
    scheduler = torch.optim.lr_scheduler.StepLR(
        cur_optimizer, 
        step_size=10, 
        gamma=0.5),
    metrics={ "accuracy": accuracy_multi_asset, "rmse": rmse_multi_asset },
    num_epochs=40,
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