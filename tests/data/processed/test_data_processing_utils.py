import pandas as pd
import numpy as np

from data.processed.data_processing_utils import filter_by_regular_hours
from config.constants import Constants


def _make_df(dates):
    """Helper that converts a list of date-like strings into a minimal OHLCV DataFrame."""
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'dummy': np.arange(len(dates))
    })


def test_filter_keeps_only_weekday_regular_hours():
    """Rows outside 13:30–20:00 *or* on weekends must be dropped."""
    dates = [
        # Monday 2 Jan 2023
        "2023-01-02 13:29",  # before window
        "2023-01-02 13:30",  # inclusive lower bound
        "2023-01-02 15:00",  # inside window
        "2023-01-02 20:00",  # inclusive upper bound
        "2023-01-02 20:01",  # after window
        # Saturday 7 Jan 2023 ‒ inside time window but weekend
        "2023-01-07 14:00",
    ]

    df = _make_df(dates)
    result = filter_by_regular_hours(df, 'date')

    expected_dates = pd.to_datetime([
        "2023-01-02 13:30",
        "2023-01-02 15:00",
        "2023-01-02 20:00",
    ])

    # Ensure only expected rows are present and in the same order
    assert list(result['date']) == list(expected_dates)

    # Function must reset the index according to implementation
    assert result.index.equals(pd.RangeIndex(start=0, stop=len(result)))


def test_boundaries_use_project_constants():
    """The start and end times defined in Constants.Data must be honoured."""
    trading_day = pd.Timestamp('2023-01-03')  # Tuesday

    start = pd.Timestamp.combine(trading_day.date(), Constants.Data.REGULAR_TRADING_HOURS_START)
    end = pd.Timestamp.combine(trading_day.date(), Constants.Data.REGULAR_TRADING_HOURS_END)

    off_start = start - pd.Timedelta(minutes=1)
    off_end = end + pd.Timedelta(minutes=1)

    df = _make_df([off_start, start, end, off_end])

    result = filter_by_regular_hours(df, 'date')

    # Only start and end (inclusive) should remain
    assert list(result['date']) == [start, end] 