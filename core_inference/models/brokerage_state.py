from dataclasses import dataclass
import torch
import pandas as pd


@dataclass
class BrokerageState:
    equity: float
    cash_balance: float
    shares_hold: dict[str: int]