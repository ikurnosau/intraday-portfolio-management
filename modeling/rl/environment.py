from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List

import pandas as pd
import torch

from data.processed.dataset_creation import DatasetCreator
from .state import State


def _default_reward(prev_state: State, next_state: State, fee: float) -> torch.Tensor:
    """Return incremental portfolio *log*-return given a transaction fee.

    r_t = Δw · rₜ₊₁ − |Δw| · fee
    where Δw = wₜ - wₜ₋₁.

    We output *r_t* (not its log) so the algorithm can combine it flexibly
    (e.g. into ∑log(1+r_t) or the final cumulative product).
    """
    delta_w = next_state.position - prev_state.position
    pnl_component = delta_w * next_state.next_step_return  # (assets,)
    cost_component = torch.abs(delta_w) * (fee + next_state.spread)  # (assets,)
    return (pnl_component - cost_component).sum()  # scalar


class PortfolioEnvironment:
    """Minimal episodic environment iterating through a single trading day."""

    def __init__(
        self,
        retrieval_result: Dict[str, pd.DataFrame],
        dataset_creator: DatasetCreator,
        transaction_fee: float = 0.0005,
        reward_function: Callable[[State, State, float], torch.Tensor] | None = None,
        device: torch.device | str = "cuda",
    ):
        self.retrieval_result = retrieval_result
        self.dataset_creator = dataset_creator
        self.device = torch.device(device)
        self.transaction_fee = transaction_fee
        self.reward_function = reward_function or _default_reward

        self.state_templates: Dict[dt.date, List[State]] = {}
        self._prepare_state_templates()

        self.trading_day: dt.date | None = None
        self.state_index: int = 0
        self.current_state: State | None = None

    def _prepare_state_templates(self) -> None:
        """Pre-compute immutable *template* states for every trading day."""
        for day in self.get_trading_days():
            day_data = self._filter_retrieval_result(day)
            # DatasetCreator returns eight arrays; we only need the first, third and fourth
            X, _, next_ret, spread, *_ = self.dataset_creator.create_dataset_numpy(day_data)

            day_states: List[State] = []
            for feat, nr, sp in zip(X, next_ret, spread):
                day_states.append(
                    State(
                        signal_features=torch.tensor(feat, dtype=torch.float32, device=self.device),
                        next_step_return=torch.tensor(nr, dtype=torch.float32, device=self.device),
                        spread=torch.tensor(sp, dtype=torch.float32, device=self.device),
                        position=torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    )
                )
            self.state_templates[day] = day_states

    def _filter_retrieval_result(self, day: dt.date) -> Dict[str, pd.DataFrame]:
        """Return only those rows that belong to *day* (across all assets)."""
        filtered: Dict[str, pd.DataFrame] = {}
        for symbol, df in self.retrieval_result.items():
            mask = pd.to_datetime(df["date"]).dt.date == day
            cur = df.loc[mask].reset_index(drop=True)
            if not cur.empty:
                filtered[symbol] = cur
        return filtered

    def get_trading_days(self) -> List[dt.date]:
        """Enumerate all distinct trading days in *retrieval_result*."""
        days = set()
        for df in self.retrieval_result.values():
            days.update(pd.to_datetime(df["date"]).dt.date.unique())
        return sorted(days)

    def set_trading_day(self, day: dt.date) -> State:
        if day not in self.state_templates:
            raise ValueError(f"No data available for trading day {day}")
        self.trading_day = day
        self.state_index = 0
        self.current_state = self.state_templates[day][0].copy()

        return self.current_state

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, State | None]:
        if self.trading_day is None:
            raise RuntimeError("Trading day not initialised; call set_trading_day first.")

        prev_state = self.current_state

        next_index = self.state_index + 1
        if next_index >= len(self.state_templates[self.trading_day]):
            # End of episode
            return torch.tensor(0.0, device=self.device), None

        next_state = self.state_templates[self.trading_day][next_index].copy()
        next_state.position = action

        reward = self.reward_function(prev_state, next_state, self.transaction_fee)

        self.state_index = next_index
        self.current_state = next_state

        return reward, next_state
