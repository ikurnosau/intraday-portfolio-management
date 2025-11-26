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

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig, RLConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification, Balanced5ClassClassification, BinaryClassification, MeanReturnSignClassification, FutureMeanReturnClassification, TripleClassification, ReturnOverHorizon
from data.processed.normalization import MinMaxNormalizer, ZScoreOverWindowNormalizer, MinMaxNormalizerOverWindow
from data.processed.missing_values_handling import ForwardFillFlatBars, DummyMissingValuesHandler
from data.processed.missing_values_handling import ContinuousForwardFillPolars
from data.raw.retrievers.stooq_retriever import StooqRetriever
from data.raw.retrievers.alpaca_markets_retriever import AlpacaMarketsRetriever
from modeling.models.tsa_classifier import TemporalSpatial
from modeling.models.lstm import LSTMClassifier
from modeling.models.mlp import MLP
from modeling.models.tcn import TCN
from modeling.models.tsa_allocator import TSAllocator
from modeling.models.tcn import TCNPredictor
from modeling.models.tst import TimeSeriesTransformer
from modeling.loss import PositionReturnLoss, position_return_loss_with_entropy, RiskAdjustedPositionReturnLoss
from modeling.metrics import accuracy_multi_asset, accuracy, rmse_regression, MeanReturn
from core_data_prep.validations import Validator


frequency = TimeFrame(amount=1, unit=TimeFrameUnit.Day)
horizon = 30
target = ReturnOverHorizon(horizon=horizon, base_feature='close')

data_config = DataConfig(
    retriever=StooqRetriever(download_from_gdrive=False),

    symbol_or_symbols=Constants.Data.DJIA,
    frequency=frequency,

    # start=datetime(2024, 6, 1, tzinfo=timezone.utc),
    # end=datetime(2025, 6, 1, tzinfo=timezone.utc),
    # train_set_last_date=datetime(2025, 4, 1, tzinfo=timezone.utc),
    # val_set_last_date=datetime(2025, 5, 1, tzinfo=timezone.utc),

    start=datetime(1970, 1, 2, tzinfo=Constants.Data.EASTERN_TZ),
    end=datetime(2019, 1, 1, tzinfo=Constants.Data.EASTERN_TZ),
    train_set_last_date=datetime(2012, 1, 1, tzinfo=Constants.Data.EASTERN_TZ), 
    val_set_last_date=datetime(2000, 1, 1, tzinfo=Constants.Data.EASTERN_TZ),

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
    missing_values_handler_polars=ContinuousForwardFillPolars(frequency=str(frequency)),

    in_seq_len=30,
    horizon=horizon,
    multi_asset_prediction=True,

    validator=None, #Validator(),
)


model_config = ModelConfig(
    model=TSAllocator(
        input_dim=len(data_config.features_polars),
        output_dim=1,  # regression
        hidden_dim=64,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.2,
        num_heads=2,
        use_spatial_attention=True,
        num_assets=len(data_config.symbol_or_symbols),
        asset_embed_dim=8,
    ),
    registered_model_name="TemporalSpatial Regressor",
)

cur_optimizer = torch.optim.AdamW(
    model_config.model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    amsgrad=True,
)

train_config = TrainConfig(
    loss_fn=PositionReturnLoss(fee=0.001),
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
    metrics={"log_return": PositionReturnLoss(fee=0.001), "mean_return": MeanReturn(fee=0.001)},
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

rl_config = RLConfig(
    trajectory_length=12,
    fee=0.001,
    spread_multiplier=0.0,
    trade_asset_count=1,
)

observability_config = ObservabilityConfig(
    experiment_name="Return Regression MLP"
)

config = ExperimentConfig(
    data_config=data_config,
    model_config=model_config,
    train_config=train_config,
    rl_config=rl_config,
    observability_config=observability_config
)