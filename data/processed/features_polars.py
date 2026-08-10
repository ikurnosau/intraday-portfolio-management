"""Serializable Polars feature implementations used by experiment configs."""

import math

import polars as pl

from data.processed.indicators_polars import EMA, RollingVWAP, SMA, VWAP


class LogReturn:
    def __init__(self, feature: str = "close", eps: float = 1e-8):
        self.feature = feature
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        value = pl.col(self.feature) + self.eps
        return (value / value.shift(1)).log().fill_null(0.0)


class HighLowRange:
    def __init__(self, close_feature: str = "close", eps: float = 1e-8):
        self.close_feature = close_feature
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        return (pl.col("high") - pl.col("low")) / (
            pl.col(self.close_feature) + self.eps
        )


class CloseOpenReturn:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        return (pl.col("close") - pl.col("open")) / (
            pl.col("open") + self.eps
        )


class LogVolumeChange:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        volume = pl.col("volume") + self.eps
        return (volume / volume.shift(1)).log().fill_null(0.0)


class RealizedVolatility:
    def __init__(
        self,
        window: int,
        feature: str = "close",
        eps: float = 1e-8,
    ):
        self.window = window
        self.feature = feature
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        return (
            (pl.col(self.feature) + self.eps)
            .pct_change()
            .rolling_std(window_size=self.window)
            .fill_null(0.0)
        )


class RollingVWAPDistance:
    def __init__(self, period: int, eps: float = 1e-8):
        self.indicator = RollingVWAP(period)
        self.eps = eps

    def __call__(self, frame: pl.LazyFrame) -> pl.Expr:
        return (pl.col("close") - self.indicator(frame)) / (
            pl.col("close") + self.eps
        )


class VWAPDistance:
    def __init__(self, eps: float = 1e-8):
        self.indicator = VWAP()
        self.eps = eps

    def __call__(self, frame: pl.LazyFrame) -> pl.Expr:
        return (pl.col("close") - self.indicator(frame)) / (
            pl.col("close") + self.eps
        )


class LocationInRange:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        return (pl.col("close") - pl.col("low")) / (
            pl.col("high") - pl.col("low") + self.eps
        )


class TimeOfDayCyclic:
    def __init__(
        self,
        function: str,
        trading_day_minutes: float = 390.0,
        offset: float = 0.0,
    ):
        if function not in {"sin", "cos"}:
            raise ValueError("function must be 'sin' or 'cos'")
        self.function = function
        self.trading_day_minutes = trading_day_minutes
        self.offset = offset

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        angle = (
            (pl.col("date").dt.hour() * 60 + pl.col("date").dt.minute())
            .cast(pl.Float32)
            * (2 * math.pi)
            / self.trading_day_minutes
        )
        value = angle.sin() if self.function == "sin" else angle.cos()
        return value + self.offset


class MovingAverageSlope:
    def __init__(
        self,
        kind: str,
        fast_period: int,
        slow_period: int,
        base_feature: str = "close",
    ):
        indicators = {"SMA": SMA, "EMA": EMA}
        try:
            indicator = indicators[kind]
        except KeyError as exc:
            raise ValueError("kind must be 'SMA' or 'EMA'") from exc
        self.fast = indicator(fast_period, base_feature)
        self.slow = indicator(slow_period, base_feature)

    def __call__(self, frame: pl.LazyFrame) -> pl.Expr:
        return self.fast(frame) - self.slow(frame)


class VolatilitySlope:
    def __init__(
        self,
        fast_window: int,
        slow_window: int,
        feature: str = "close",
        eps: float = 1e-8,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.feature = feature
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        returns = (pl.col(self.feature) + self.eps).pct_change()
        return (
            returns.rolling_std(window_size=self.fast_window)
            / (
                returns.rolling_std(window_size=self.slow_window)
                + self.eps
            )
        ).fill_null(0.0)


class Column:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        return pl.col(self.name)


class RelativeSpread:
    def __init__(
        self,
        denominator: str = "close",
        eps: float = 1e-8,
    ):
        self.denominator = denominator
        self.eps = eps

    def __call__(self, _: pl.LazyFrame) -> pl.Expr:
        return (pl.col("ask_price") - pl.col("bid_price")) / (
            pl.col(self.denominator) + self.eps
        )
