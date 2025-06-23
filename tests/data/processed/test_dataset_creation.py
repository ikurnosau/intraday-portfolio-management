import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest

from data.processed.dataset_creation import DatasetCreator
from data.processed.data_processing_utils import filter_by_regular_hours

# -----------------------------------------------------------------------------
# Dummy components so that we isolate DatasetCreator logic itself
# -----------------------------------------------------------------------------

class IdentityClose:  # feature = close price itself
    def __call__(self, df):
        return df["close"]

class DummyNormalizer:
    """Return the input unchanged. Ensures we can reason about raw values."""
    def __call__(self, data):
        return data

class DummyMissing:
    def __call__(self, df):
        return df

class BinaryUpTarget:
    """1 if next bar close > current close else 0 (same rule as BinaryClassification)."""
    def __call__(self, df):
        return (df["close"].pct_change().shift(-1) > 0).astype(np.float32)

# -----------------------------------------------------------------------------
# Synthetic OHLCV generator
# -----------------------------------------------------------------------------

START_DATE = datetime(2023, 1, 2, 13, 30)
N_ROWS = 400
N_TRAIN_ROWS = 300

def make_ohlcv(n: int, start_price: float = 1.0, step: float = 1.0):
    """Create strictly increasing price series so that target==1 except last bar."""
    idx = pd.date_range(START_DATE, periods=n, freq="1min")
    prices = start_price + step * np.arange(n, dtype=float)
    df = pd.DataFrame({
        "date": idx,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": 100,
    })
    return df

# -----------------------------------------------------------------------------
# Parametrised integration test
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("multi_asset,seq_len", [(True, 3), (False, 1)])
def test_create_dataset_numpy_end_to_end(multi_asset, seq_len):
    features = {"close_id": IdentityClose()}
    dc = DatasetCreator(
        features=features,
        target=BinaryUpTarget(),
        normalizer=DummyNormalizer(),
        missing_values_handler=DummyMissing(),
        in_seq_len=seq_len,
        train_set_last_date=START_DATE + timedelta(minutes=N_TRAIN_ROWS),
        multi_asset_prediction=multi_asset,
        cutoff_time=None,  # disable intra-day cutoff for the test
    )

    raw = {
        "AAPL": make_ohlcv(N_ROWS),
        "MSFT": make_ohlcv(N_ROWS),
    }

    Xtr, ytr, Xte, yte = dc.create_dataset_numpy(raw)

    # ------------------------------------------------------------------
    # 1. Shape assertions
    # ------------------------------------------------------------------

    # Compute expected counts by reproducing filtering logic once
    per_asset_df = filter_by_regular_hours(make_ohlcv(N_ROWS), 'date')
    total_rows = len(per_asset_df)
    expected_train_rows_per_asset = (
        per_asset_df['date'] <= START_DATE + timedelta(minutes=N_TRAIN_ROWS)
    ).sum()
    expected_test_rows_per_asset = total_rows - expected_train_rows_per_asset

    if multi_asset:
        expected_train_rows = expected_train_rows_per_asset
        expected_test_rows = expected_test_rows_per_asset
        if seq_len > 1:
            expected_train_rows = expected_train_rows - seq_len + 1   
            expected_test_rows = expected_test_rows - seq_len + 1
        # X shape: (assets, batch, window, features) if seq_len>1 else (assets,batch,features)
        features_axis = 3 if seq_len > 1 else 2
        assert Xtr.shape[0] == 2  # assets
        assert Xtr.shape[1] == expected_train_rows
        assert Xte.shape[1] == expected_test_rows
        assert Xtr.shape[features_axis] == 1  # only 1 feature
    else:
        # single stacked batch
        expected_train_rows = expected_train_rows_per_asset * 2  # stacked assets
        expected_test_rows = expected_test_rows_per_asset * 2
        if seq_len > 1:
            expected_train_rows = expected_train_rows - 2 * (seq_len - 1) 
            expected_test_rows = expected_test_rows - 2 * (seq_len - 1)
        assert Xtr.shape[0] == expected_train_rows
        if seq_len > 1:
            assert Xtr.shape[1] == seq_len  # window
            assert Xtr.shape[2] == 1        # features
        else:
            assert Xtr.shape[1] == 1        # feature count equals 1 after vstack
    # ------------------------------------------------------------------
    # 2. Type / NaN checks
    # ------------------------------------------------------------------
    for arr in (Xtr, ytr, Xte, yte):
        assert arr.dtype == np.float32
        assert not np.isnan(arr).any()

    # ------------------------------------------------------------------
    # 3. Target alignment verification for first training sample
    # ------------------------------------------------------------------
    if seq_len > 1:
        if multi_asset:
            first_window = Xtr[0, 0, :, 0]
            target_first = ytr[0, 0]
        else:
            first_window = Xtr[0, :, 0]
            target_first = ytr[0]
        # because prices strictly increase, target should be 1
        assert target_first == 1.0
        # last element of window equals close price at position seq_len
        assert first_window[-1] == seq_len
    else:
        if multi_asset:
            price_first = Xtr[0, 0, 0]
            target_first = ytr[0, 0]
        else:
            price_first = Xtr[0, 0]
            target_first = ytr[0]
        assert price_first == 1.0
        assert target_first == 1.0

    # ------------------------------------------------------------------
    # 4. Multi-asset arrays equal for synthetic data
    # ------------------------------------------------------------------
    if multi_asset:
        assert np.array_equal(Xtr[0], Xtr[1])
        assert np.array_equal(ytr[0], ytr[1]) 

# -----------------------------------------------------------------------------
# Additional tests: temporal alignment & determinism
# -----------------------------------------------------------------------------


def test_temporal_alignment_sequential():
    """For sequential mode ensure y[t] corresponds to last bar of X window."""
    seq_len = 5
    dc = DatasetCreator(
        features={"id": IdentityClose()},
        target=BinaryUpTarget(),
        normalizer=DummyNormalizer(),
        missing_values_handler=DummyMissing(),
        in_seq_len=seq_len,
        train_set_last_date=START_DATE + timedelta(minutes=N_TRAIN_ROWS),
        multi_asset_prediction=False,
        cutoff_time=None,
    )
    raw = {"AAPL": make_ohlcv(N_ROWS)}
    X, y, _, _ = dc.create_dataset_numpy(raw)

    # X shape (batch, window, features)
    prices = filter_by_regular_hours(make_ohlcv(N_ROWS), 'date')['close'].values.astype(np.float32)
    # after to_sequence, first sample corresponds to original index seq_len-1
    for i in range(len(y)):
        last_price_in_window = X[i, -1, 0]
        original_idx = seq_len - 1 + i
        assert last_price_in_window == prices[original_idx]
        # target rule: price[next] > price[cur] => 1 until penultimate bar
        expected_target = 1.0 if original_idx + 1 < len(prices) else 0.0
        assert y[i] == expected_target


def test_deterministic_output():
    """Running creator twice on same input yields identical arrays (regression style)."""
    dc = DatasetCreator(
        features={"id": IdentityClose()},
        target=BinaryUpTarget(),
        normalizer=DummyNormalizer(),
        missing_values_handler=DummyMissing(),
        in_seq_len=1,
        train_set_last_date=START_DATE + timedelta(minutes=N_TRAIN_ROWS),
        multi_asset_prediction=True,
        cutoff_time=None,
    )
    raw = {
        "AAPL": make_ohlcv(N_ROWS),
        "MSFT": make_ohlcv(N_ROWS),
    }
    out1 = dc.create_dataset_numpy(raw)
    out2 = dc.create_dataset_numpy(raw)

    for a1, a2 in zip(out1, out2):
        assert np.array_equal(a1, a2) 