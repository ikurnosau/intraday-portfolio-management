import numpy as np
import pandas as pd
import torch
from datetime import timedelta, datetime, timezone, time
from torch.utils.data import TensorDataset, DataLoader
import pytest

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, ObservabilityConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification
from data.processed.normalization import MinMaxNormalizer
from data.processed.missing_values_handling import ForwardFillFlatBars
from data.processed.dataset_creation import DatasetCreator
from modeling.models.mlp import MLPClassifier
from modeling.models.lstm import LSTMClassifier
from modeling.metrics import accuracy, rmse
from modeling.trainer import Trainer

# --------------------------------------------------------------------------
# Test-specific configuration that mirrors cur_experiment.py
# --------------------------------------------------------------------------

TRAIN_SET_LAST_DATE = datetime(2025, 5, 1, tzinfo=timezone.utc)

test_data_config = DataConfig(
    symbol_or_symbols=['SYN1'],  # synthetic data symbol
    start=TRAIN_SET_LAST_DATE - timedelta(days=1),  # one day before split
    end=TRAIN_SET_LAST_DATE + timedelta(days=1),    # one day after split
    features={
        "open": lambda data: data['open'],
        "high": lambda data: data['high'],
        "low": lambda data: data['low'],
        "close": lambda data: data['close'],
        "volume": lambda data: data['volume'],
        "return": lambda data: data['close'].pct_change(),
        "OBV": OBV(),
        "RSI6": RSI(6),
        "RSI12": RSI(12),
        "EMA3": EMA(3),
        "EMA6": EMA(6),
        "EMA12": EMA(12),
        "ATR14": ATR(14),
        "MFI": MFI(14),
        "ADX14": ADX(14),
        "ADX20": ADX(20),
        "MOM1": MOM(1),
        "MOM3": MOM(3),
        "CCI12": CCI(12),
        "CCI20": CCI(20),
        "ROCR12": ROCR(12),
        "MACD": MACD(),
        "WILLR": WILLR(10),
        "TRIX": TRIX(20),
        "BB_LOW": BollingerBand(BollingerBand.BBType.LOWER),
        "BB_UP": BollingerBand(BollingerBand.BBType.UPPER),
        "EMA_26": EMA(26, base_feature="close"),
        "VWAP": VWAP(high_feature='high', low_feature='low', close_feature='close'),
        "ATR_28": ATR(28, high_feature='high', low_feature='low', close_feature='close'),
        "FRL_0": FRL(FRL.FIB_RATIOS[0], high_feature='high', low_feature='low', close_feature='close'),
        "FRL_1": FRL(FRL.FIB_RATIOS[1], high_feature='high', low_feature='low', close_feature='close'),
        "FRL_2": FRL(FRL.FIB_RATIOS[2], high_feature='high', low_feature='low', close_feature='close'),
        "FRL_3": FRL(FRL.FIB_RATIOS[3], high_feature='high', low_feature='low', close_feature='close'),
        "FRL_4": FRL(FRL.FIB_RATIOS[4], high_feature='high', low_feature='low', close_feature='close'),
        "RSI_28": RSI(24),
        "Oscillator_K": Oscillator(Oscillator.LineType.K),
        "Oscillator_D": Oscillator(Oscillator.LineType.D),
    },
    target=Balanced3ClassClassification(base_feature='close'),
    normalizer=MinMaxNormalizer(),
    missing_values_handler=ForwardFillFlatBars(),
    train_set_last_date=TRAIN_SET_LAST_DATE,
    in_seq_len=1,
    multi_asset_prediction=False,
    batch_size=32,
    shuffle=False,
    cutoff_time=time(hour=14, minute=10),
)

test_model_config = ModelConfig(
    model=MLPClassifier(
        input_dim=37,  # matches number of features
        n_class=3,
        hidden_dims=[128, 64],
        dropout=0.0,
        activation=torch.nn.ReLU(inplace=True),
        batch_norm=False
    ),
    registered_model_name="Test MLP"
)

test_train_config = TrainConfig(
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=None,  # will be set in the test
    scheduler=None,  # not needed for quick test
    metrics={"accuracy": accuracy, "rmse": rmse},
    num_epochs=10,
    device=torch.device("cpu"),
    save_path=""
)

test_config = ExperimentConfig(
    data_config=test_data_config,
    model_config=test_model_config,
    train_config=test_train_config,
    observability_config=ObservabilityConfig(experiment_name="Test Experiment")
)

# --------------------------------------------------------------------------
# Synthetic OHLCV generator that embeds an easy-to-learn pattern
# --------------------------------------------------------------------------

ROWS = 20000  # ~ 33 hours and 20 minutes, covers two trading sessions
START = (TRAIN_SET_LAST_DATE - timedelta(days=10)).replace(hour=13, minute=30)  # previous day's market open 13:30 UTC


def make_pattern_ohlcv(n=ROWS):
    """
    Price pattern requires looking at 3 consecutive prices to predict the next one:
    If we see [1.0, 2.0, 3.0] → next is 4.0 (big up)
    If we see [4.0, 2.0, 3.0] → next is 1.0 (big down)
    """
    base = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0], dtype=np.float32)

    close = np.tile(base, n // 6 + 1)[:n]
    idx = pd.date_range(START, periods=n, freq="1min", tz=TRAIN_SET_LAST_DATE.tzinfo)

    df = pd.DataFrame(
        {
            "date": idx,
            "open":  close,
            "high":  close,
            "low":   close,
            "close": close,
            "volume": 1.0,
        }
    )
    return df


# --------------------------------------------------------------------------
# Test configurations
# --------------------------------------------------------------------------

def create_mlp_full_features_config():
    """Configuration for MLP with full feature set"""
    return ExperimentConfig(
        data_config=DataConfig(
            symbol_or_symbols=['SYN1'],  # synthetic data symbol
            start=TRAIN_SET_LAST_DATE - timedelta(days=1),  # one day before split
            end=TRAIN_SET_LAST_DATE + timedelta(days=1),    # one day after split
            features={
                "open": lambda data: data['open'],
                "high": lambda data: data['high'],
                "low": lambda data: data['low'],
                "close": lambda data: data['close'],
                "volume": lambda data: data['volume'],
                "return": lambda data: data['close'].pct_change(),
                "OBV": OBV(),
                "RSI6": RSI(6),
                "RSI12": RSI(12),
                "EMA3": EMA(3),
                "EMA6": EMA(6),
                "EMA12": EMA(12),
                "ATR14": ATR(14),
                "MFI": MFI(14),
                "ADX14": ADX(14),
                "ADX20": ADX(20),
                "MOM1": MOM(1),
                "MOM3": MOM(3),
                "CCI12": CCI(12),
                "CCI20": CCI(20),
                "ROCR12": ROCR(12),
                "MACD": MACD(),
                "WILLR": WILLR(10),
                "TRIX": TRIX(20),
                "BB_LOW": BollingerBand(BollingerBand.BBType.LOWER),
                "BB_UP": BollingerBand(BollingerBand.BBType.UPPER),
                "EMA_26": EMA(26, base_feature="close"),
                "VWAP": VWAP(high_feature='high', low_feature='low', close_feature='close'),
                "ATR_28": ATR(28, high_feature='high', low_feature='low', close_feature='close'),
                "FRL_0": FRL(FRL.FIB_RATIOS[0], high_feature='high', low_feature='low', close_feature='close'),
                "FRL_1": FRL(FRL.FIB_RATIOS[1], high_feature='high', low_feature='low', close_feature='close'),
                "FRL_2": FRL(FRL.FIB_RATIOS[2], high_feature='high', low_feature='low', close_feature='close'),
                "FRL_3": FRL(FRL.FIB_RATIOS[3], high_feature='high', low_feature='low', close_feature='close'),
                "FRL_4": FRL(FRL.FIB_RATIOS[4], high_feature='high', low_feature='low', close_feature='close'),
                "RSI_28": RSI(24),
                "Oscillator_K": Oscillator(Oscillator.LineType.K),
                "Oscillator_D": Oscillator(Oscillator.LineType.D),
            },
            target=Balanced3ClassClassification(base_feature='close'),
            normalizer=MinMaxNormalizer(),
            missing_values_handler=ForwardFillFlatBars(),
            train_set_last_date=TRAIN_SET_LAST_DATE,
            in_seq_len=1,
            multi_asset_prediction=False,
            batch_size=32,
            shuffle=False,
            cutoff_time=time(hour=14, minute=10),
        ),
        model_config=ModelConfig(
            model=MLPClassifier(
                input_dim=37,  # matches number of features
                n_class=3,
                hidden_dims=[128, 64],
                dropout=0.0,
                activation=torch.nn.ReLU(inplace=True),
                batch_norm=False
            ),
            registered_model_name="Test MLP Full Features"
        ),
        train_config=TrainConfig(
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=None,  # will be set in the test
            scheduler=None,  # not needed for quick test
            metrics={"accuracy": accuracy, "rmse": rmse},
            num_epochs=10,
            device=torch.device("cpu"),
            save_path=""
        ),
        observability_config=ObservabilityConfig(experiment_name="Test MLP Full Features")
    )

def create_lstm_minimal_features_config():
    """Configuration for LSTM with just close and volume"""
    return ExperimentConfig(
        data_config=DataConfig(
            symbol_or_symbols=['SYN1'],
            start=TRAIN_SET_LAST_DATE - timedelta(days=1),
            end=TRAIN_SET_LAST_DATE + timedelta(days=1),
            features={
                "close": lambda data: data['close'],
                "volume": lambda data: data['volume'],
            },
            target=Balanced3ClassClassification(base_feature='close'),
            normalizer=MinMaxNormalizer(),
            missing_values_handler=ForwardFillFlatBars(),
            train_set_last_date=TRAIN_SET_LAST_DATE,
            in_seq_len=3,  # Use sequence of 3 minutes for LSTM
            multi_asset_prediction=False,
            batch_size=32,
            shuffle=False,
            cutoff_time=time(hour=14, minute=10),
        ),
        model_config=ModelConfig(
            model=LSTMClassifier(
                input_dim=2,  # close + volume
                n_class=3,
                hidden_dim=32,
                num_layers=2,
                bidirectional=True,
                dropout=0.0
            ),
            registered_model_name="Test LSTM Minimal Features"
        ),
        train_config=TrainConfig(
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=None,
            scheduler=None,
            metrics={"accuracy": accuracy, "rmse": rmse},
            num_epochs=10,
            device=torch.device("cpu"),
            save_path=""
        ),
        observability_config=ObservabilityConfig(experiment_name="Test LSTM Minimal Features")
    )

# --------------------------------------------------------------------------
# End-to-end tests
# --------------------------------------------------------------------------

def run_pipeline_test(config: ExperimentConfig):
    """Common test logic for both MLP and LSTM configurations"""
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. Build DatasetCreator from the test's DataConfig
    dcfg = config.data_config
    creator = DatasetCreator(
        features=dcfg.features,
        target=dcfg.target,
        normalizer=dcfg.normalizer,
        missing_values_handler=dcfg.missing_values_handler,
        in_seq_len=dcfg.in_seq_len,
        train_set_last_date=dcfg.train_set_last_date,
        multi_asset_prediction=dcfg.multi_asset_prediction,
        cutoff_time=dcfg.cutoff_time,
    )

    # 2. Synthetic raw data (single symbol to keep training fast)
    raw = { "SYN1": make_pattern_ohlcv() }

    Xtr, ytr, Xval, yval = creator.create_dataset_numpy(raw)

    # PyTorch tensors
    Xtr_t, Xval_t = map(torch.from_numpy, (Xtr, Xval))
    ytr_t = torch.from_numpy(ytr).long()
    yval_t = torch.from_numpy(yval).long()

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(Xval_t, yval_t), batch_size=256, shuffle=False
    )

    # 3. Re-use the model, loss from test config
    mcfg = config.model_config
    tcfg = config.train_config

    # Create fresh optimizer for the test
    optimiser = torch.optim.Adam(
        mcfg.model.parameters(), lr=1e-2
    )

    trainer = Trainer(
        model=mcfg.model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=tcfg.loss_fn,
        optimizer=optimiser,
        scheduler=None,
        metrics=tcfg.metrics,
        num_epochs=10,          # few epochs are enough for the pattern
        device=torch.device("cpu"),
    )

    _, history = trainer.train()
    final_acc = history["val_accuracy"][-1]

    # 4. Threshold: random guess = 1/3 ; we expect > 0.9 on this simple pattern
    assert final_acc > 0.9

def test_pipeline_mlp_full_features():
    """Test MLP with full feature set"""
    run_pipeline_test(create_mlp_full_features_config())

def test_pipeline_lstm_minimal_features():
    """Test LSTM with just close and volume features"""
    run_pipeline_test(create_lstm_minimal_features_config())