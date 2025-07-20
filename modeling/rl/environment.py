from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List

import pandas as pd
import torch
import logging

from data.processed.dataset_creation import DatasetCreator
from .state import State


# def _default_reward(current_state: State, next_state: State, fee: float) -> torch.Tensor:
#     """Return incremental portfolio *log*-return given a transaction fee.

#     r_t = Δw · rₜ₊₁ − |Δw| · fee
#     where Δw = wₜ - wₜ₋₁.

#     We output *r_t* (not its log) so the algorithm can combine it flexibly
#     (e.g. into ∑log(1+r_t) or the final cumulative product).
#     """
#     delta_w = next_state.position - current_state.position
#     pnl_component = delta_w * current_state.next_step_return  # (assets,)
#     cost_component = torch.abs(delta_w) * (fee + current_state.spread)  # (assets,)
#     return (pnl_component - cost_component).sum()  # scalar

def _default_reward(current_state: State, next_state: State, fee: float, spread_discount: float = 1) -> torch.Tensor:
    """Return incremental portfolio *log*-return given a transaction fee.

    r_t = Δw · rₜ₊₁ − |Δw| · fee
    where Δw = wₜ - wₜ₋₁.

    We output *r_t* (not its log) so the algorithm can combine it flexibly
    (e.g. into ∑log(1+r_t) or the final cumulative product).
    """
    return_component = current_state.position * current_state.next_step_return  # (assets,)
    cost_component = torch.abs(next_state.position - current_state.position) * (fee + current_state.spread / 2 * spread_discount)  # (assets,)
    return (return_component - cost_component).sum()  # scalar


class PortfolioEnvironment:
    """Minimal episodic environment iterating through a single trading day."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        next_return_train: np.ndarray,
        spread_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        next_return_test: np.ndarray,
        spread_test: np.ndarray,
        trading_days: list[dt.date],
        transaction_fee: float = 0.0005,
        reward_function: Callable[[State, State, float], torch.Tensor] | None = None,
        device: torch.device | str = "cuda",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.next_return_train = next_return_train
        self.spread_train = spread_train
        self.X_test = X_test
        self.y_test = y_test
        self.next_return_test = next_return_test
        self.spread_test = spread_test
        
        self.trading_days = trading_days
        self.minutes_per_day = len(self.X_train) // len(self.trading_days)
        
        self.device = torch.device(device)
        self.transaction_fee = transaction_fee
        self.reward_function = reward_function or _default_reward

        self.day_slices = self._prepare_day_slices()

        self.trading_day: dt.date | None = None
        self.state_index: int = 0
        self.current_state: State | None = None

    def _prepare_day_slices(self) -> None:
        """Pre-compute immutable *template* states for every trading day."""

        day_slices = {}
        for i, day in enumerate(self.trading_days):
            day_slices[day] = slice(i * self.minutes_per_day, (i + 1) * self.minutes_per_day)

        return day_slices

    def _prepare_state_template(self, day: dt.date, i: int) -> List[State]:
        X = self.X_train[self.day_slices[day]][i]
        next_ret = self.next_return_train[self.day_slices[day]][i]
        spread = self.spread_train[self.day_slices[day]][i]
        
        return State(
            signal_features=torch.tensor(X, dtype=torch.float32, device=self.device),
            next_step_return=torch.tensor(next_ret, dtype=torch.float32, device=self.device),
            spread=torch.tensor(spread, dtype=torch.float32, device=self.device),
            position=torch.tensor(0.0, dtype=torch.float32, device=self.device),
        )
        
    def get_trading_days(self) -> list[dt.date]:
        return self.trading_days

    def set_trading_day(self, day: dt.date) -> State:
        self.trading_day = day
        self.state_index = 0

        self.current_state = self._prepare_state_template(day, 0)
        self.current_state.position = torch.zeros(self.X_train.shape[1], device=self.device)

        return self.current_state

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, State | None]:
        if self.trading_day is None:
            raise RuntimeError("Trading day not initialised; call set_trading_day first.")

        next_index = self.state_index + 1
        if next_index >= self.minutes_per_day:
            # End of episode
            return torch.tensor(0.0, device=self.device), None

        next_state = self._prepare_state_template(self.trading_day, next_index)
        next_state.position = action

        reward = self.reward_function(self.current_state, next_state, self.transaction_fee)

        self.state_index = next_index
        self.current_state = next_state

        return reward, next_state
