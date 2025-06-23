import numpy as np
import pandas as pd

from data.processed.targets import Balanced3ClassClassification, BinaryClassification


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _make_random_close_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = rng.random(n) * 100  # Some arbitrary price range 0-100
    return pd.DataFrame({"close": close})


# -----------------------------------------------------------------------------
# Balanced 3-class classification
# -----------------------------------------------------------------------------

def test_balanced3class_distribution_and_constraints():
    df = _make_random_close_df(120)
    target_series = Balanced3ClassClassification()(df)

    # 1) Length equality
    assert len(target_series) == len(df)

    # 2) No NaNs
    assert not target_series.isna().any()

    # 3) Values are only from the set {0,1,2}
    values = set(target_series.unique())
    assert values.issubset({0.0, 1.0, 2.0}), f"Unexpected labels: {values}"

    # 4) All three classes should appear at least once
    assert values == {0.0, 1.0, 2.0}, "Not all three classes present in data"

    # 5) Class distribution should be roughly balanced (each class within 10% of n/3)
    counts = target_series.value_counts()
    expected = len(df) / 3
    tolerance = len(df) * 0.1  # 10% of total samples
    max_deviation = (counts - expected).abs().max()
    assert max_deviation <= tolerance, (
        f"Class distribution deviates more than 10% from perfect balance: {counts.to_dict()}"
    )


# -----------------------------------------------------------------------------
# Binary classification
# -----------------------------------------------------------------------------

def test_binaryclassification_distribution_and_constraints():
    # Design dataset with both positive and negative returns by building a zig-zag price path
    close_prices = [1, 2, 3, 2, 1.5, 1.8, 2.5, 2.4, 2.6, 2.3, 2.0]
    df = pd.DataFrame({"close": close_prices})

    target_series = BinaryClassification()(df)

    # 1) Length equality
    assert len(target_series) == len(df)

    # 2) No NaNs
    assert not target_series.isna().any()

    # 3) Values in {0,1}
    values = set(target_series.unique())
    assert values.issubset({0.0, 1.0}), f"Unexpected labels: {values}"

    # 4) Both classes appear
    assert values == {0.0, 1.0}, "Both binary classes should appear in data"

    # 5) Verify the target equals expected rule (next_return > 0)
    returns = df["close"].pct_change()
    next_return = returns.shift(-1)
    expected_target = (next_return > 0).astype(np.float32).fillna(0)
    pd.testing.assert_series_equal(
        target_series.reset_index(drop=True),
        expected_target.reset_index(drop=True),
        check_names=False,
    ) 