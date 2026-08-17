import math

import polars as pl
import pytest

from data.processed.features_polars import (
    LogReturn,
    MovingAverageSlope,
)


def test_trend_features_can_use_quote_midpoint():
    frame = pl.DataFrame({
        "bid_price": [99.0, 101.0, 103.0],
        "ask_price": [101.0, 103.0, 105.0],
        "close": [100.0, 90.0, 80.0],
    })

    result = frame.select(
        LogReturn(feature="midpoint", eps=0.0)(
            frame.lazy()
        ).alias("log_return"),
        MovingAverageSlope(
            kind="SMA",
            fast_period=1,
            slow_period=2,
            base_feature="midpoint",
        )(frame.lazy()).alias("sma_slope"),
    )

    assert result["log_return"].to_list() == pytest.approx([
        0.0,
        math.log(102.0 / 100.0),
        math.log(104.0 / 102.0),
    ])
    assert result["sma_slope"].to_list() == pytest.approx([0.0, 1.0, 1.0])
