import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import torch
import numpy as np

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification, Balanced5ClassClassification, BinaryClassification, MeanReturnSignClassification, FutureMeanReturnClassification
from data.processed.normalization import MinMaxNormalizer, ZScoreOverWindowNormalizer, MinMaxNormalizerOverWindow
from data.processed.missing_values_handling import ForwardFillFlatBars, DummyMissingValuesHandler
from modeling.models.tsa_classifier import TemporalSpatial
from modeling.models.lstm import LSTMClassifier
from modeling.models.mlp import MLP
from modeling.models.tcn import TCN
from modeling.metrics import accuracy_multi_asset, accuracy, rmse_regression


frequency = TimeFrame(amount=1, unit=TimeFrameUnit.Minute)

data_config = DataConfig(
    symbol_or_symbols=Constants.Data.LOWEST_VOL_TO_SPREAD_MAY_JUNE,
    frequency=frequency,

    # start=datetime(2024, 6, 1, tzinfo=timezone.utc),
    # end=datetime(2025, 6, 1, tzinfo=timezone.utc),
    # train_set_last_date=datetime(2025, 4, 1, tzinfo=timezone.utc), 
    # val_set_last_date=datetime(2025, 5, 1, tzinfo=timezone.utc),

    start=datetime(2024, 9, 1, tzinfo=Constants.Data.EASTERN_TZ),
    end=datetime(2025, 10, 1, tzinfo=Constants.Data.EASTERN_TZ),
    train_set_last_date=datetime(2025, 7, 1, tzinfo=Constants.Data.EASTERN_TZ), 
    val_set_last_date=datetime(2025, 8, 1, tzinfo=Constants.Data.EASTERN_TZ),

    features={
        # --- Raw micro-price & volume dynamics ------------------------------------------------------
        "log_ret": lambda df: np.log(df['close'] / df['close'].shift(1)),
        "hl_range": lambda df: (df['high'] - df['low']) / df['close'],
        "close_open": lambda df: (df['close'] - df['open']) / df['open'],
        "vol_delta": lambda df: np.log(df['volume'] / (df['volume'].shift(1) + 1e-8) + 1e-8),

        # --- Momentum & trend -----------------------------------------------------------------------
        "EMA_fast": EMA(3),              # fast EMA (≈ 3-min)
        "EMA_slow": EMA(30),            # slow EMA adjusted for 60-bar window
        "RSI2": RSI(2),
        "RSI6": RSI(6),
        # Optionally uncomment to add a slow oscillator now that the window is 60
        # "RSI12": RSI(12),

        # --- Volatility ----------------------------------------------------------------------------
        "realvol20": lambda df: df['close'].pct_change().rolling(20).std().astype(np.float32),

        # --- Microstructure & order-flow -----------------------------------------------------------
        "VWAP_dist": lambda df: (df['close'] - VWAP()(df)) / df['close'],
        "loc_in_range": lambda df: (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8),

        # --- Time-of-day cyclic encodings -----------------------------------------------------------
        "tod_sin": lambda df: np.sin(2 * np.pi * (df['date'].dt.hour * 60 + df['date'].dt.minute) / (6.5 * 60)),
        "tod_cos": lambda df: np.cos(2 * np.pi * (df['date'].dt.hour * 60 + df['date'].dt.minute) / (6.5 * 60)),

        # --- Derived features ----------------------------------------------------------------------
        "ema_slope": lambda df: EMA(3)(df) - EMA(15)(df),     # or ratio
        "vol_slope": lambda df: df['close'].pct_change()
                             .rolling(10).std()
                             / (df['close'].pct_change().rolling(20).std() + 1e-8)
    },
    target=FutureMeanReturnClassification(base_feature='close', horizon=1),
    normalizer=MinMaxNormalizerOverWindow(window=60, fit_feature=None),
    missing_values_handler=ForwardFillFlatBars(frequency=str(frequency)),

    in_seq_len=60,
    multi_asset_prediction=True,

    cutoff_time=datetime.combine(
        datetime.today(), 
        Constants.Data.REGULAR_TRADING_HOURS_START + timedelta(minutes=30)\
    ).time(),
)


model_config = ModelConfig(
    model=TemporalSpatial(
        input_dim=len(data_config.features),
        output_dim=1,  # regression
        hidden_dim=128,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.2,
        num_heads=4,
        use_spatial_attention=True,
        num_assets=len(data_config.symbol_or_symbols),
        asset_embed_dim=16,
    ),
    # model=TCN(
    #     in_channels=len(data_config.features),
    #     hidden_channels=128,
    #     kernel_size=2,
    #     num_layers=6,
    #     output_dim=1,
    #     use_layer_norm=True,
    #     dropout=0.2,
    # ),
    registered_model_name="TemporalSpatial Regressor",
)

cur_optimizer = torch.optim.AdamW(
    model_config.model.parameters(),
    lr=1e-3,
    weight_decay=1e-10,
    amsgrad=True,
)

train_config = TrainConfig(
    loss_fn=torch.nn.MSELoss(),
    optimizer=cur_optimizer,
    scheduler={
        "type": "OneCycleLR",
        "max_lr": 3e-3,        # peak learning rate
        "pct_start": 0.1,       # 10 % warm-up
        "div_factor": 25,       # initial LR = max_lr / 25
        "final_div_factor": 1e3,# final LR = max_lr / 1000
        "anneal_strategy": "cos",
        "cycle_momentum": False,
    },
    metrics={"rmse": rmse_regression},
    num_epochs=20,
    early_stopping_patience=10,

    device=torch.device("cuda"),
    cudnn_benchmark=True,

    batch_size=128,
    shuffle=True,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    
    save_path="",
)

observability_config = ObservabilityConfig(
    experiment_name="Return Regression MLP"
)

config = ExperimentConfig(
    data_config=data_config,
    model_config=model_config,
    train_config=train_config,
    observability_config=observability_config
)