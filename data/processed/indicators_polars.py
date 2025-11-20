
# ... existing code ...
# Creating Polars-based versions of most frequently used indicators so that feature engineering can be
# performed directly in Polars and — where required — in lazy mode.
# NOTE: Only the indicators currently required by `cur_experiment.py` are implemented. Add more as needed.

import polars as pl
from typing import Union

_PolarsFrame = Union[pl.DataFrame, pl.LazyFrame]


class EMA:
    """Exponential Moving Average implemented with Polars expressions.

    The object is callable so that it can be plugged directly into the
    `features_polars` mapping. When called with a Polars (Lazy)Frame it returns a
    `pl.Series` (eager execution) which is subsequently converted to numpy/pandas
    by the data-preparation utilities.

    For lazy workflows you can instead use the `expr` property which returns the
    underlying `pl.Expr`.
    """

    def __init__(self, period: int, base_feature: str = "close") -> None:
        self.period = period
        self.base_feature = base_feature

    def __call__(self, lf: pl.LazyFrame) -> pl.Expr:  # noqa: D401
        """Return EMA expression to be used within a Polars select statement."""
        _ = lf  # not used but kept for unified signature
        return (
            pl.col(self.base_feature)
            .ewm_mean(span=self.period, adjust=False)
            .fill_null(strategy="forward")
            .fill_null(0.0)
        )


class RSI:
    """Relative Strength Index (RSI) implemented in Polars.

    This implementation mirrors the Pandas version used elsewhere in the code
    base and employs Wilder's smoothing (simple moving average of gains / losses).
    """

    def __init__(self, period: int = 14, base_feature: str = "close") -> None:
        self.period = period
        self.base_feature = base_feature

    def __call__(self, lf: pl.LazyFrame) -> pl.Expr:
        _ = lf
        delta = pl.col(self.base_feature).diff()

        # Replace NaN diff with 0 so first row behaves like pandas `.where` logic
        delta = delta.fill_null(0.0)

        gain = (
            pl.when(delta > 0)
              .then(delta)
              .otherwise(0.0)
              .fill_null(0.0)
        )

        loss = (
            pl.when(delta < 0)
              .then(-delta)
              .otherwise(0.0)
              .fill_null(0.0)
        )

        avg_gain = gain.rolling_mean(window_size=self.period, min_periods=1)
        avg_loss = loss.rolling_mean(window_size=self.period, min_periods=1)

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fill_null(0.0)


class VWAP:
    """Volume Weighted Average Price implemented with Polars expressions."""

    def __init__(
        self,
        high_feature: str = "high",
        low_feature: str = "low",
        close_feature: str = "close",
        volume_feature: str = "volume",
    ) -> None:
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature
        self.volume_feature = volume_feature

    def __call__(self, lf: pl.LazyFrame) -> pl.Expr:
        _ = lf
        tp = (pl.col(self.high_feature) + pl.col(self.low_feature) + pl.col(self.close_feature)) / 3
        cumulative_tpv = (tp * pl.col(self.volume_feature)).cum_sum()
        cumulative_vol = pl.col(self.volume_feature).cum_sum()
        return cumulative_tpv / (cumulative_vol + 1e-8)
