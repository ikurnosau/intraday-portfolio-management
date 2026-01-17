from dataclasses import dataclass
import torch
import pandas as pd


@dataclass
class State:
    allocation: dict[str: float]
    shares_hold: dict[str: float]
    equity: float