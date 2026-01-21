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
        baseline_to_include: list[str] = [], 
        root_dir: str = '../modeling/rl/visualization',
        year_start: int = -1
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

    for baseline_name in baseline_to_include:
        baseline_dir = os.path.join(root_dir, 'baselines')
        for baseline_filename in os.listdir(baseline_dir):
            if baseline_filename.startswith(baseline_name):
                baseline_wealth = pd.read_csv(os.path.join(baseline_dir, baseline_filename))
                if year_start >= 0:
                    baseline_wealth = baseline_wealth[baseline_wealth['x'] >= year_start]
                baseline_wealth = baseline_wealth['y'] / baseline_wealth['y'].iloc[0]
                extra_time = np.linspace(
                    start_time.timestamp(),
                    end_time.timestamp(),
                    num=len(baseline_wealth)
                )
                extra_time = [datetime.fromtimestamp(t) for t in extra_time]
                plt.plot(extra_time, baseline_wealth, linestyle="--", label=baseline_name)

                returns_dict[baseline_name] = pd.Series(baseline_wealth).pct_change().dropna().to_list()

    # Formatting
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Accumulated Wealth", fontsize=16)
    plt.title("Strategy Performance Over Time", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return returns_dict