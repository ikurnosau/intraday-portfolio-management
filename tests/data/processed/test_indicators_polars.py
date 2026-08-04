import polars as pl
import pytest

from data.processed.indicators_polars import RollingVWAP, SMA


def _sma_values(values: list[float], period: int) -> list[float]:
    frame = pl.DataFrame({"close": values})
    return frame.select(SMA(period)(frame.lazy()).alias("sma"))["sma"].to_list()


def test_sma_uses_period_window():
    values = _sma_values([1.0, 2.0, 3.0, 4.0], period=3)

    assert values == pytest.approx([1.0, 1.5, 2.0, 3.0])


def test_sma_is_independent_of_values_before_period_window():
    common_window = [10.0, 11.0, 12.0, 13.0]
    first = _sma_values([-1_000.0, *common_window], period=4)
    second = _sma_values([1_000.0, *common_window], period=4)

    assert first[-1] == pytest.approx(second[-1])


def test_sma_rejects_non_positive_period():
    with pytest.raises(ValueError, match="period must be at least 1"):
        SMA(0)


def _rolling_vwap_values(
    prices: list[float],
    volumes: list[float],
    period: int,
) -> list[float]:
    frame = pl.DataFrame(
        {
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volumes,
        }
    )
    return frame.select(
        RollingVWAP(period)(frame.lazy()).alias("rolling_vwap")
    )["rolling_vwap"].to_list()


def test_rolling_vwap_uses_period_window():
    values = _rolling_vwap_values(
        prices=[1.0, 2.0, 3.0, 4.0],
        volumes=[1.0, 1.0, 2.0, 2.0],
        period=3,
    )

    assert values[-1] == pytest.approx((2.0 + 6.0 + 8.0) / 5.0)


def test_rolling_vwap_is_independent_of_values_before_period_window():
    common_prices = [10.0, 11.0, 12.0]
    common_volumes = [1.0, 2.0, 3.0]
    first = _rolling_vwap_values(
        [-1_000.0, *common_prices],
        [100.0, *common_volumes],
        period=3,
    )
    second = _rolling_vwap_values(
        [1_000.0, *common_prices],
        [100.0, *common_volumes],
        period=3,
    )

    assert first[-1] == pytest.approx(second[-1])


def test_rolling_vwap_rejects_non_positive_period():
    with pytest.raises(ValueError, match="period must be at least 1"):
        RollingVWAP(0)
