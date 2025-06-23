import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from datetime import datetime, timezone
import torch

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification, BinaryClassification
from data.processed.normalization import ZScoreOverWindowNormalizer, ZScoreNormalizer, MinMaxNormalizer
from data.processed.missing_values_handling import DummyMissingValuesHandler, ForwardFillFlatBars
from modeling.models.mlp import MLPClassifier, MLPClassifierScaled
from modeling.models.lstm import LSTMClassifier
from modeling.metrics import accuracy, rmse

data_config = DataConfig(
    symbol_or_symbols=Constants.Data.MOST_LIQUID_TECH_STOCKS, 
    # start=datetime(2025, 1, 1), 
    start=datetime(2024, 6, 1),
    end=datetime(2025, 6, 1),

    features={
        "close": lambda data: data['close'],
        "volume": lambda data: data['volume'],
    },
    target=Balanced3ClassClassification(base_feature='close'),
    normalizer=MinMaxNormalizer(),
    missing_values_handler=ForwardFillFlatBars(),
    train_set_last_date=datetime(2025, 5, 1, tzinfo=timezone.utc), 
    in_seq_len=50,
    multi_asset_prediction=False,

    batch_size=256,
    shuffle=False
)

model_config=ModelConfig(
    model=LSTMClassifier(
        input_dim=2,
        n_class=3, 
        hidden_dim=256,
        num_layers=3, 
        bidirectional=True,
        dropout=0.1
    ),
    # model=MLPClassifier(
    #     input_dim=37,
    #     n_class=3,
    #     hidden_dims=[128, 64]
    # ),
    registered_model_name="LSTM Large Dataset"
)

cur_optimizer = torch.optim.AdamW(
    model_config.model.parameters(), 
    lr=1e-3,
    weight_decay=1e-5,
    amsgrad=True)

train_config=TrainConfig(
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=cur_optimizer,
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau (
        cur_optimizer, 
        factor=0.5,
        patience=10,
        verbose=True),
    metrics={ "accuracy": accuracy, "rmse": rmse },
    num_epochs=50,
    device=torch.device("cuda"),
    save_path=""
)

observability_config = ObservabilityConfig(
    experiment_name="Cur Experiment"
)

config = ExperimentConfig(
    data_config=data_config,
    model_config=model_config,
    train_config=train_config,
    observability_config=observability_config
)