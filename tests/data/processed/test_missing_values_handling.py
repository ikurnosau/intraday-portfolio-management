import pandas as pd
import numpy as np
from datetime import datetime

import pandas.testing as pdt

from data.processed.missing_values_handling import ForwardFillFlatBars


def _make_partial_day_df():
    """Create a dataframe for one trading day with large gaps so we can test filling."""
    times = [
        "2023-01-02 13:30",  # first minute of the session
        "2023-01-02 13:35",  # 5 minutes later – introduces gap (13:31‒13:34)
        "2023-01-02 14:00",  # another gap (13:36‒13:59)
    ]
    return pd.DataFrame({
        "date": pd.to_datetime(times),
        "open": [10.0, 11.0, 12.0],
        "high": [10.5, 11.5, 12.5],
        "low":  [9.5, 10.5, 11.5],
        "close": [10.2, 11.2, 12.2],
        "volume": [1000, 2000, 3000],
    })


def _apply_forward_fill(df):
    return ForwardFillFlatBars()(df.copy())


def test_forward_fill_creates_complete_minute_grid():
    """After filling, we should have exactly 391 minutes for the trading day."""
    df = _make_partial_day_df()
    filled = _apply_forward_fill(df)

    # Expect 391 rows (inclusive of both ends)
    assert len(filled) == 391

    expected_index = pd.date_range(
        start="2023-01-02 13:30", end="2023-01-02 20:00", freq="min"
    )
    assert list(filled["date"]) == list(expected_index)

    # Ensure there are absolutely no NaNs remaining
    assert filled.isna().sum().sum() == 0


def test_original_data_preserved_verbatim():
    """Values in the original dataframe must be unchanged after filling."""
    original = _make_partial_day_df()
    filled = _apply_forward_fill(original)

    joined = filled.set_index("date").loc[original["date"]]
    # Use pandas testing helper to compare frames ignoring dtype changes
    pdt.assert_frame_equal(
        original.set_index("date").sort_index(),
        joined.sort_index(),
        check_dtype=False,
        check_exact=True,
    )


def test_newly_filled_rows_values():
    """Rows that were not present originally should be flat bars with volume 0 and OHLC equal to close."""
    original = _make_partial_day_df()
    filled = _apply_forward_fill(original)

    # Identify rows that were newly created (volume == 0)
    new_rows = filled[filled["volume"] == 0].copy()
    assert len(new_rows) > 0  # sanity

    # OHLC must all equal close for these flat bars
    for col in ["open", "high", "low"]:
        assert (new_rows[col] == new_rows["close"]).all()

    # Close must be forward-filled relative to the previous available close
    assert filled["close"].equals(filled["close"].ffill()) 