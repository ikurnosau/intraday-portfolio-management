import numpy as np
import pandas as pd
import polars as pl

from data.processed.features_polars import RelativeSpread as RelativeSpreadFeature
from data.processed.statistics import (
    FutureReturn,
    RelativeSpread,
    RollingVolatility,
)
from modeling.allocator_evaluation import calc_realized_returns


def test_relative_spread_can_use_quote_midpoint() -> None:
    data = pd.DataFrame({
        "bid_price": [99.0],
        "ask_price": [101.0],
        "close": [200.0],
    })

    spread = RelativeSpread(denominator="midpoint", eps=0.0)(data)

    np.testing.assert_allclose(spread.to_numpy(), [0.02])


def test_future_return_can_use_quote_midpoint() -> None:
    data = pd.DataFrame({
        "bid_price": [99.0, 101.0],
        "ask_price": [101.0, 103.0],
        "close": [200.0, 100.0],
    })

    returns = FutureReturn(
        horizon=1,
        feature="midpoint",
    )(data)

    np.testing.assert_allclose(returns.to_numpy(), [0.02, 0.0])


def test_rolling_volatility_can_use_quote_midpoint() -> None:
    data = pd.DataFrame({
        "bid_price": [99.0, 101.0, 100.0],
        "ask_price": [101.0, 103.0, 102.0],
        "close": [200.0, 100.0, 200.0],
    })

    volatility = RollingVolatility(
        window=2,
        feature="midpoint",
    )(data)
    midpoint_returns = pd.Series([100.0, 102.0, 101.0]).pct_change()
    expected = midpoint_returns.rolling(window=2).std().fillna(0.0)

    np.testing.assert_allclose(volatility.to_numpy(), expected.to_numpy())


def test_polars_relative_spread_can_use_quote_midpoint() -> None:
    data = pl.DataFrame({
        "bid_price": [99.0],
        "ask_price": [101.0],
        "close": [200.0],
    })

    spread = data.select(
        RelativeSpreadFeature(denominator="midpoint", eps=0.0)(
            data.lazy()
        ).alias("spread")
    )

    np.testing.assert_allclose(spread["spread"].to_numpy(), [0.02])


def test_midpoint_spread_matches_backtest_cost_as_return_fraction() -> None:
    market = pd.DataFrame({
        "bid_price": [99.0],
        "ask_price": [101.0],
    })
    spread = RelativeSpread(denominator="midpoint", eps=0.0)(
        market
    ).to_numpy().reshape(1, 1)

    realized_returns = calc_realized_returns(
        allocations=np.array([[1.0]]),
        next_returns=np.array([[0.0]]),
        spreads=spread,
        fee=0.0,
        spread_multiplier=1.5,
    )

    # The backtest charges 1.5 * (101 - 99) / 2 = $1.50 per share.
    # At the $100 midpoint that is a 1.5% portfolio return cost.
    np.testing.assert_allclose(realized_returns, [-0.015])
