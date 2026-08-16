"""Serializable statistics used by training and inference pipelines."""

import numpy as np
import pandas as pd


class FutureReturn:
    def __init__(
        self,
        horizon: int,
        feature: str = "close",
        fill_value: float = 0.0,
    ):
        self.horizon = horizon
        self.feature = feature
        self.fill_value = fill_value

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return (
            data[self.feature].shift(-self.horizon) / data[self.feature] - 1.0
        ).fillna(self.fill_value).astype(np.float32)


class RollingVolatility:
    def __init__(
        self,
        window: int,
        feature: str = "close",
        fill_value: float = 0.0,
    ):
        self.window = window
        self.feature = feature
        self.fill_value = fill_value

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return (
            data[self.feature]
            .pct_change()
            .astype(np.float32)
            .rolling(window=self.window)
            .std()
            .fillna(self.fill_value)
            .astype(np.float32)
        )


class RelativeSpread:
    def __init__(
        self,
        denominator: str = "midpoint",
        eps: float = 1e-8,
    ):
        self.denominator = denominator
        self.eps = eps

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.denominator == "midpoint":
            denominator = (
                data["ask_price"] + data["bid_price"]
            ) / 2
        else:
            denominator = data[self.denominator]
        return (data["ask_price"] - data["bid_price"]) / (
            denominator + self.eps
        )
