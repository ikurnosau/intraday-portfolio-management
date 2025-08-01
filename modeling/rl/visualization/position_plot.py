import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Sequence, List, Optional, Union


def plot_position_heatmap(
    positions: Union[np.ndarray, Sequence[Sequence[float]]],
    timestamps: Optional[Sequence[Union[datetime, str, int, float]]] = None,
    asset_names: Optional[Sequence[str]] = None,
    max_timestamps: int = 50,
    title: str = "Portfolio positions over time",
):
    """Visualise the evolution of portfolio allocations as a heat-map.

    Parameters
    ----------
    positions : array-like of shape (T, N)
        Historical positions where ``positions[t, i]`` is the allocation to
        *asset i* at time-step *t*.  Each row must already satisfy
        ``∑|aᵢ| = 1`` with values in the interval [-1, 1].

    timestamps : Sequence[datetime | str | int | float], optional
        Labels for the *T* time-steps.  If *None*, the function uses the row
        indices (0, 1, …, T-1).

    asset_names : Sequence[str], optional
        Names of the *N* assets.  If *None*, assets are labelled "Asset 0",
        "Asset 1", …

    max_timestamps : int, default 50
        When ``T`` is large, the heat-map can become unreadable.  The function
        therefore samples at most *max_timestamps* evenly-spaced rows from the
        input.  Set to ``None`` to plot every time-step.

    title : str, default "Portfolio positions over time"
        Figure title.
    """

    # --- Convert & validate ---------------------------------------------------
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError("'positions' must be a 2-D array of shape (T, N)")

    n_timestamps, n_assets = positions.shape

    # Default labels
    if asset_names is None:
        asset_names = [f"Asset {i}" for i in range(n_assets)]
    if len(asset_names) != n_assets:
        raise ValueError("Length of 'asset_names' does not match number of columns in 'positions'.")

    if timestamps is None:
        timestamps = list(range(n_timestamps))
    if len(timestamps) != n_timestamps:
        raise ValueError("Length of 'timestamps' does not match number of rows in 'positions'.")

    # --- Down-sample if necessary --------------------------------------------
    if max_timestamps is not None and n_timestamps > max_timestamps:
        # Evenly spaced indices over the full range [0, T-1]
        idx = np.linspace(0, n_timestamps - 1, max_timestamps, dtype=int)
        positions = positions[idx]
        timestamps = [timestamps[i] for i in idx]
        n_timestamps = max_timestamps

    # Convert timestamps to strings for nicer y-ticks
    timestamp_labels: List[str] = []
    for t in timestamps:
        if isinstance(t, datetime):
            timestamp_labels.append(t.strftime("%Y-%m-%d %H:%M"))
        else:
            timestamp_labels.append(str(t))

    # --- Plot -----------------------------------------------------------------
    plt.figure(figsize=(12, max(4, n_timestamps * 0.3)))
    im = plt.imshow(positions, aspect="auto", cmap="bwr", vmin=-1, vmax=1)

    plt.colorbar(im, label="Position (-1 = short, +1 = long)")

    plt.xticks(range(n_assets), asset_names, rotation=90)
    plt.yticks(range(n_timestamps), timestamp_labels)

    plt.xlabel("Assets")
    plt.ylabel("Time")
    plt.title(title)
    plt.tight_layout()
    plt.show()
