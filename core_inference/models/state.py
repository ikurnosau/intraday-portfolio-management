from dataclasses import dataclass
import torch
import pandas as pd


@dataclass
class State:
    assets_df: dict[str: pd.DataFrame]
    allocation: dict[str: float]
    wealth: float