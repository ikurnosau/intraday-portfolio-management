import inspect
from enum import Enum

import numpy as np
import pandas as pd
import pandas.testing as pdt

import data.processed.indicators as ind

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _make_synthetic_ohlcv(n: int = 50) -> pd.DataFrame:
    """Return a deterministic OHLCV dataframe with monotonic prices and volume."""
    idx = pd.RangeIndex(n)
    high = np.arange(1, n + 1, dtype=float)
    low = high - 1.0
    close = (high + low) / 2.0
    open_ = close.copy()
    volume = np.full(n, 1000.0)

    return pd.DataFrame({
        "high": high,
        "low": low,
        "open": open_,
        "close": close,
        "volume": volume,
    })


def _instantiate_all_indicators():
    """Return a mapping name -> callable instance for every indicator class."""
    instances = {}
    for name, cls in inspect.getmembers(ind, inspect.isclass):
        # Only classes defined in this module
        if cls.__module__ != ind.__name__:
            continue
        # Skip internal Enums and nested classes
        if issubclass(cls, Enum):
            continue
        # Instantiate with reasonable defaults
        if name == "BollingerBand":
            instances[f"{name}_lower"] = cls(cls.BBType.LOWER)
            instances[f"{name}_upper"] = cls(cls.BBType.UPPER)
        elif name == "Oscillator":
            instances[f"{name}_K"] = cls(cls.LineType.K)
            instances[f"{name}_D"] = cls(cls.LineType.D)
        elif name == "FRL":
            # Use one Fibonacci ratio for testing
            instances[name] = cls(cls.FIB_RATIOS[2])  # 0.5
        else:
            # Provide mandatory positional args if any
            sig = inspect.signature(cls)
            params = sig.parameters
            # Count required positional args (no default and not self)
            required = [p for p in params.values()
                        if p.default == inspect._empty and p.name != 'self']
            # Simple heuristic: provide 10 for period if needed
            args = []
            if required:
                # Currently only 'period' is required in several indicators
                args = [10]
            instances[name] = cls(*args)
    return instances


# -----------------------------------------------------------------------------
# Generic tests applicable to every indicator
# -----------------------------------------------------------------------------

def test_indicator_output_basic_properties():
    df = _make_synthetic_ohlcv(n=60)
    instances = _instantiate_all_indicators()

    for name, ind_fn in instances.items():
        result = ind_fn(df)
        # Must return a pandas Series
        assert isinstance(result, pd.Series), f"{name} did not return Series"
        # Length must match input
        assert len(result) == len(df), f"{name} length mismatch"
        # Some values (after initial warm-up) must be finite
        finite_count = np.isfinite(result.values).sum()
        assert finite_count > 0, f"{name} produced no finite values"
        # There should be no +/-inf
        assert not np.isinf(result.values).any(), f"{name} produced inf"


# -----------------------------------------------------------------------------
# Specific correctness checks for simple indicators
# -----------------------------------------------------------------------------


def test_rocr_exact_values():
    """ROCR with period=1 should match manual (close[i]/close[i-1])*100"""
    df = _make_synthetic_ohlcv(6)
    roc = ind.ROCR(period=1)(df)
    expected = pd.Series([np.nan] + [df.loc[i, 'close'] / df.loc[i-1, 'close'] * 100
                                     for i in range(1, len(df))])
    pdt.assert_series_equal(roc, expected, check_names=False)


def test_mom_exact_values():
    """Momentum with period=2 should equal close[i]-close[i-2]."""
    df = _make_synthetic_ohlcv(6)
    mom = ind.MOM(period=2)(df)
    expected = pd.Series([np.nan, np.nan] + [df.loc[i, 'close'] - df.loc[i-2, 'close']
                                              for i in range(2, len(df))])
    pdt.assert_series_equal(mom, expected, check_names=False)


def test_willr_range():
    """Williams %R must always be between -100 and 0 (inclusive)."""
    df = _make_synthetic_ohlcv(30)
    will = ind.WILLR(period=10)(df).dropna()
    assert ((will >= -100) & (will <= 0)).all() 