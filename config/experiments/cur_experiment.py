import sys
import os

from sympy.functions.elementary.piecewise import false
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import torch
import numpy as np
import polars as pl
import math
from data.processed.indicators_polars import EMA as EMA_pl, RSI as RSI_pl, VWAP as VWAP_pl

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification, Balanced5ClassClassification, BinaryClassification, MeanReturnSignClassification, FutureMeanReturnClassification, TripleClassification, ReturnOverHorizon
from data.processed.normalization import MinMaxNormalizer, ZScoreOverWindowNormalizer, MinMaxNormalizerOverWindow
from data.processed.missing_values_handling import ForwardFillFlatBars, DummyMissingValuesHandler
from core_data_prep.core_data_prep import ContinuousForwardFill, ContinuousForwardFillPolars
from modeling.models.tsa_classifier import TemporalSpatial
from modeling.models.lstm import LSTMClassifier
from modeling.models.mlp import MLP
from modeling.models.tcn import TCN
from modeling.models.tsa_allocator import TSAllocator
from modeling.models.tcn import TCNPredictor
from modeling.models.tst import TimeSeriesTransformer
from modeling.loss import PositionReturnLoss, position_return_loss_with_entropy
from modeling.metrics import accuracy_multi_asset, accuracy, rmse_regression


frequency = TimeFrame(amount=1, unit=TimeFrameUnit.Day)
horizon = 30
target = TripleClassification(horizon=horizon, base_feature='close')

data_config = DataConfig(
    symbol_or_symbols=Constants.Data.LOWEST_VOL_TO_SPREAD_MAY_JUNE,
    frequency=frequency,

    # start=datetime(2024, 6, 1, tzinfo=timezone.utc),
    # end=datetime(2025, 6, 1, tzinfo=timezone.utc),
    # train_set_last_date=datetime(2025, 4, 1, tzinfo=timezone.utc),
    # val_set_last_date=datetime(2025, 5, 1, tzinfo=timezone.utc),

    start=datetime(1970, 1, 2, tzinfo=Constants.Data.EASTERN_TZ),
    end=datetime(2019, 1, 1, tzinfo=Constants.Data.EASTERN_TZ),
    train_set_last_date=datetime(2012, 1, 1, tzinfo=Constants.Data.EASTERN_TZ), 
    val_set_last_date=datetime(2014, 1, 1, tzinfo=Constants.Data.EASTERN_TZ),

    features={
        # --- Raw micro-price & volume dynamics ------------------------------------------------------
        "log_ret": lambda df: np.log((df['close'] + 1e-8) / (df['close'].shift(1) + 1e-8)).fillna(0.0),
        "hl_range": lambda df: (df['high'] - df['low']) / (df['close'] + 1e-8),
        "close_open": lambda df: (df['close'] - df['open']) / (df['open'] + 1e-8),
        "vol_delta": lambda df: np.log((df['volume'] + 1e-8) / (df['volume'].shift(1) + 1e-8)).fillna(0.0),

        # --- Momentum & trend -----------------------------------------------------------------------
        "EMA_fast": EMA(3),              # fast EMA (≈ 3-min)
        "EMA_slow": EMA(30),            # slow EMA adjusted for 60-bar window
        "RSI2": RSI(2),
        "RSI6": RSI(6),
        # Optionally uncomment to add a slow oscillator now that the window is 60
        # "RSI12": RSI(12),

        # --- Volatility ----------------------------------------------------------------------------
        "realvol20": lambda df: (df['close'] + 1e-8).pct_change().rolling(20).std().astype(np.float32).fillna(0.0),

        # --- Microstructure & order-flow -----------------------------------------------------------
        "VWAP_dist": lambda df: (df['close'] - VWAP()(df)) / (df['close'] + 1e-8),
        "loc_in_range": lambda df: (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8),

        # --- Time-of-day cyclic encodings -----------------------------------------------------------
        "tod_sin": lambda df: np.sin(2 * np.pi * (df['date'].dt.hour * 60 + df['date'].dt.minute) / (6.5 * 60)),
        "tod_cos": lambda df: np.cos(2 * np.pi * (df['date'].dt.hour * 60 + df['date'].dt.minute) / (6.5 * 60)),

        # --- Derived features ----------------------------------------------------------------------
        "ema_slope": lambda df: EMA(3)(df) - EMA(15)(df),     # or ratio
        "vol_slope": lambda df: ((df['close'] + 1e-8).pct_change()
                             .rolling(10).std()
                             / ((df['close'] + 1e-8).pct_change().rolling(20).std() + 1e-8)).fillna(0.0),

        "is_missing": lambda df: df['is_missing'],
    },

    features_polars={
        # --- Raw micro-price & volume dynamics ------------------------------------------------------
        "log_ret": lambda lf: (((pl.col("close") + 1e-8) / (pl.col("close").shift(1) + 1e-8)).log()).fill_null(0.0),
        "hl_range": lambda lf: (pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-8),
        "close_open": lambda lf: (pl.col("close") - pl.col("open")) / (pl.col("open") + 1e-8),
        "vol_delta": lambda lf: (((pl.col("volume") + 1e-8) / (pl.col("volume").shift(1) + 1e-8)).log()).fill_null(0.0),

        # --- Momentum & trend -----------------------------------------------------------------------
        "EMA_fast": EMA_pl(3),              # fast EMA (≈ 3-min)
        "EMA_slow": EMA_pl(30),            # slow EMA adjusted for 60-bar window
        "RSI2": RSI_pl(2),
        "RSI6": RSI_pl(6),
        # Optionally uncomment to add a slow oscillator now that the window is 60
        # "RSI12": RSI_pl(12),

        # --- Volatility ----------------------------------------------------------------------------
        "realvol20": lambda lf: (pl.col("close") + 1e-8).pct_change().rolling_std(window_size=20).fill_null(0.0),

        # --- Microstructure & order-flow -----------------------------------------------------------
        "VWAP_dist": lambda lf: (pl.col("close") - VWAP_pl()(lf)) / (pl.col("close") + 1e-8),
        "loc_in_range": lambda lf: (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8),

        # --- Time-of-day cyclic encodings -----------------------------------------------------------
        "tod_sin": lambda lf: (((pl.col("date").dt.hour() * 60 + pl.col("date").dt.minute()).cast(pl.Float32) * (2 * math.pi) / (6.5 * 60)).sin()) - 0.19212672,
        "tod_cos": lambda lf: (((pl.col("date").dt.hour() * 60 + pl.col("date").dt.minute()).cast(pl.Float32) * (2 * math.pi) / (6.5 * 60)).cos()) + 0.018629849,

        # --- Derived features ----------------------------------------------------------------------
        "ema_slope": lambda lf: EMA_pl(3)(lf) - EMA_pl(15)(lf),     # or ratio
        "vol_slope": lambda lf: (
            (pl.col("close") + 1e-8).pct_change().rolling_std(window_size=10) / (
                (pl.col("close") + 1e-8).pct_change().rolling_std(window_size=20) + 1e-8
            )
        ).fill_null(0.0),

        "is_missing": lambda lf: pl.col("is_missing"),
    },

    statistics={
        "next_return": lambda df: (df['close'].shift(-horizon) / df['close'] - 1.0).fillna(0.0).astype(np.float32),
        "volatility": lambda df: df[getattr(target, 'base_feature', 'close')].pct_change().astype(np.float32)\
            .rolling(window=10).std().fillna(0.0).astype(np.float32),
        "spread": lambda df: (df['ask_price'] - df['bid_price']) / (df['ask_price'] + 1e-8),
    },
    
    target=target,
    normalizer=MinMaxNormalizerOverWindow(window=60, fit_feature=None),
    missing_values_handler=ContinuousForwardFill(frequency=str(frequency)),
    missing_values_handler_polars=ContinuousForwardFillPolars(frequency=str(frequency)),

    in_seq_len=80,
    horizon=horizon,
    multi_asset_prediction=True,
)


model_config = ModelConfig(
    model=TimeSeriesTransformer(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        seq_len=data_config.in_seq_len,
        feat_dim=len(data_config.features),
        dropout=0.2,
    ),
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

    batch_size=16,
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