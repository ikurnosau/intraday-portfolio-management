from queue import Queue
import torch
import torch.nn as nn
import pandas as pd

from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.repository import Repository
from core_inference.models.state import State


class Trader:
    def __init__(self, brokerage_proxy: BaseBrokerageProxy, repository: Repository, portfolio_allocator: nn.Module):
        self.brokerage_proxy = brokerage_proxy
        self.repository = repository

        self.portfolio_allocator = portfolio_allocator
        self.portfolio_allocator.eval()

        self.states_queue: Queue[State] = Queue()

    def perform_trading_cycle(self):

        

    def _predict_allocation(self, asset_dfs: dict[str: pd.DataFrame]) -> dict[str: float]:
        with torch.no_grad():
            output = self.portfolio_allocator()
