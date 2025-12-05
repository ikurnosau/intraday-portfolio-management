import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


def plot_cumulative_wealth(
        returns_dict: dict[str, list[float]], 
        start_time: datetime, 
        end_time: datetime, 
        ours_to_include: list[str] = [], 
        compare_to_baseline: bool = False, 
        root_dir: str = '../modeling/rl/visualization'
    ):
    if len(ours_to_include) > 0:
        for our_result in ours_to_include:
            returns_dict[our_result] = list(pd.read_csv(os.path.join(root_dir, f'results_ours', f'realized_returns_{our_result.lower()}.csv'))['0'])

    # Create uniform datetime range
    n_points = len(next(iter(returns_dict.values()))) + 1  # get length from any series

    time_points = [
        start_time + i * (end_time - start_time) / (n_points - 1)
        for i in range(n_points)
    ]

    # Initial wealth
    initial_wealth = 1.0

    # Plot
    plt.figure(figsize=(12, 6))
    for name, returns in returns_dict.items():
        wealth = initial_wealth * np.cumprod(1 + np.array([0] + returns))
        plt.plot(time_points, wealth, label=name)

    if compare_to_baseline:
        baseline_dir = os.path.join(root_dir, 'baselines')
        for baseline_name in os.listdir(baseline_dir):
            baseline_wealth = pd.read_csv(os.path.join(baseline_dir, baseline_name))['y']
            extra_time = np.linspace(
                start_time.timestamp(),
                end_time.timestamp(),
                num=len(baseline_wealth)
            )
            extra_time = [datetime.fromtimestamp(t) for t in extra_time]
            plt.plot(extra_time, baseline_wealth, linestyle="--", label=baseline_name.split('.')[0])

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Accumulated Wealth")
    plt.title("Strategy Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()