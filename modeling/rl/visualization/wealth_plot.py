import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def plot_cumulative_wealth(returns_dict: dict[str, list[float]], start_time: datetime, end_time: datetime):
    # Create uniform datetime range
    n_points = len(next(iter(returns_dict.values())))  # get length from any series

    time_points = [
        start_time + i * (end_time - start_time) / (n_points - 1)
        for i in range(n_points)
    ]

    # Initial wealth
    initial_wealth = 1.0

    # Plot
    plt.figure(figsize=(12, 6))
    for name, returns in returns_dict.items():
        wealth = initial_wealth * np.cumprod(1 + np.array(returns))
        plt.plot(time_points, wealth, label=name)

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Accumulated Wealth")
    plt.title("Strategy Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()