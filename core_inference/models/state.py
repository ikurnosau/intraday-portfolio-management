from dataclasses import dataclass
import torch
import pandas as pd


@dataclass
class State:
    desired_position: dict[str: float]
    position: dict[str: float]
    available_cash: float
    shares_hold: dict[str: float]

    _position_difference: dict[str: float]
    _buy_positions: dict[str: float]
    _buy_cash_per_asset: dict[str: float]
    _sell_positions: dict[str: float]
    _sell_percentage_per_share: dict[str: float]
    _sell_shares_per_asset: dict[str: float]