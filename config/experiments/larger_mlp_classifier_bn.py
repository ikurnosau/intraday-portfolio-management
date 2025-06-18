import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from datetime import datetime, timezone
import torch

from config.experiment_config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig
from config.constants import Constants
from data.processed.indicators import *
from data.processed.targets import Balanced3ClassClassification, BinaryClassification
from data.processed.normalization import ZScoreOverWindowNormalizer, ZScoreNormalizer, MinMaxNormalizer
from data.processed.missing_values_handling import DummyMissingValuesHandler
from modeling.models.mlp import MLPClassifier, MLPClassifierScaled
from modeling.metrics import accuracy, rmse

data_config = DataConfig(
    symbol_or_symbols='AAPL', 
    start=datetime(2025, 1, 1), 
    end=datetime(2025, 6, 1),

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
    missing_values_handler=DummyMissingValuesHandler(),
    train_set_last_date=datetime(2025, 5, 1, tzinfo=timezone.utc), 
    in_seq_len=1,
    flatten_sequence=False,

    batch_size=32,
    shuffle=False
)

model_config=ModelConfig(
    model=MLPClassifierScaled(
        input_dim=37,
        n_class=3,
        hidden_dims=[512, 256, 128, 64],
    )
)

cur_optimizer = torch.optim.Adam(
    model_config.model.parameters(), 
    lr=1e-4)

train_config=TrainConfig(
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=cur_optimizer,
    scheduler = torch.optim.lr_scheduler.StepLR(
        cur_optimizer, 
        step_size=5, 
        gamma=0.5),
    metrics={ "accuracy": accuracy, "rmse": rmse },
    num_epochs=30,
    device=torch.device("cuda"),
    save_path=None
)

config = ExperimentConfig(
    data_config=data_config,
    model_config=model_config,
    train_config=train_config
)