from types import SimpleNamespace

import pandas as pd

from core_data_prep.core_data_prep import DataPreparer


def test_get_daily_slices_uses_exchange_sessions() -> None:
    preparer = DataPreparer(
        normalizer=SimpleNamespace(window=1),
        missing_values_handler=lambda data: data,
        in_seq_len=1,
        frequency="1Min",
    )
    dates = pd.to_datetime([
        "2025-01-17 15:59:00",
        "2025-01-17 16:00:00",
        "2025-01-21 15:59:00",
        "2025-01-21 16:00:00",
    ]).tz_localize("America/New_York")
    data = {
        "AAPL": pd.DataFrame({
            "date": dates,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        })
    }

    slices = preparer._get_daily_slices(
        data=data,
        start_date=pd.Timestamp("2025-01-17", tz="America/New_York"),
        end_date=pd.Timestamp("2025-01-22", tz="America/New_York"),
        slice_length=2,
        verbose=False,
    )

    assert [daily_slice["AAPL"]["date"].iloc[-1] for daily_slice in slices] == [
        pd.Timestamp("2025-01-17 16:00:00", tz="America/New_York"),
        pd.Timestamp("2025-01-21 16:00:00", tz="America/New_York"),
    ]
