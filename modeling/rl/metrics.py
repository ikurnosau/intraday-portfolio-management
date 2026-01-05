from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np


def _to_numpy(returns: Sequence[float | int]) -> np.ndarray:
    """Convert a sequence of returns to a 1-D float32 NumPy array."""
    return np.asarray(returns, dtype=np.float32).ravel()


def cumulative_return(returns: Sequence[float | int], n_years: int) -> float:
    """Compound cumulative return over *returns*.

    Args:
        returns: Sequence of per-period returns expressed in **fractional** terms,
            e.g. +1 % → ``0.01``.

    Returns:
        The compounded return over the whole period, i.e. ``Π(1+r)−1``.
    """
    r = _to_numpy(returns)
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)

def mean_return_percentage(returns: Sequence[float | int], n_years: int) -> float:
    r = _to_numpy(returns)
    if r.size == 0:
        return float("nan")
    return float(r.mean() * 1e2)


def APR(returns: Sequence[float | int], n_years: int) -> float:
    """Annualised percentage return (CAGR)."""
    r = _to_numpy(returns)
    return float((np.prod(1.0 + r) ** (1.0 / n_years)) - 1.0)


def AVOL(returns: Sequence[float | int], n_years: int) -> float:
    """Annualised volatility (standard deviation of returns)."""
    n_records_per_year = len(returns) / n_years
    r = _to_numpy(returns)
    if r.size <= 1:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(n_records_per_year))


def ASR(returns: Sequence[float | int], n_years: int) -> float:
    """Annualised Sharpe ratio (APR ÷ AVOL)."""
    vol = AVOL(returns, n_years)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return APR(returns, n_years) / vol


def MDD(returns: Sequence[float | int], n_years: int) -> float:
    """Maximum drawdown over the period (expressed as a *negative* fraction)."""
    r = _to_numpy(returns)
    if r.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth / running_max) - 1.0
    return - float(drawdowns.min()) * 100  # transform to positive percentage


def CR(returns: Sequence[float | int], n_years: int) -> float:
    """Calmar ratio — APR divided by the absolute maximum drawdown."""
    dd = MDD(returns, n_years)
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return APR(returns, n_years) / abs(dd)


def DDR(returns: Sequence[float | int], n_years: int) -> float:
    """Downside deviation ratio — APR divided by downside deviation (annualised)."""
    n_records_per_year = len(returns) / n_years

    r = _to_numpy(returns)
    if r.size == 0:
        return float("nan")
    downside = r[r < 0]
    if downside.size == 0:
        return float("nan")
    downside_dev = downside.std(ddof=1) * np.sqrt(n_records_per_year)
    if downside_dev == 0:
        return float("nan")
    return APR(returns, n_years) / downside_dev


# -----------------------------------------------------------------------------
# Additional risk-adjusted performance metrics
# -----------------------------------------------------------------------------


def SoR(returns: Sequence[float | int], n_years: int) -> float:
    """Sortino ratio."""

    r = _to_numpy(returns)
    if r.size == 0:
        return float("nan")

    downside = r[r < 0]
    if downside.size == 0:
        return float("nan")

    # Population standard deviation of downside returns (ddof=1 for an unbiased estimator)
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return float("nan")

    return float(r.mean() / downside_std)


# Default metric set expected by the user
DEFAULT_METRICS: Dict[str, Callable[[Sequence[float]], float]] = {
    "CumulativeReturn": cumulative_return,
    "MeanReturnPercentage": mean_return_percentage,
    "ARR": APR,
    "AVOL": AVOL,
    "MDD": MDD,
    "ASR": ASR,
    "CR": CR,
    "DDR": DDR,
    "SoR": SoR,
}


class MetricsCalculator:
    def __init__(self, metrics: Dict[str, Callable[[Sequence[float | int]], float]] | None = None, n_years: int = 1):
        self.metrics = metrics if metrics is not None else DEFAULT_METRICS
        self.n_years = n_years

    def __call__(self, returns: Sequence[float | int]) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, fn in self.metrics.items():
            results[name] = float(fn(returns, self.n_years))

        return results 