import pandas as pd
import threading
from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.repository import Repository
from core_inference.models.brokerage_state import BrokerageState


class BacktestBrokerageProxy(BaseBrokerageProxy):
    def __init__(self, repository: Repository, spread_multiplier: float, cash_balance: float = 100000): 
        self.repository = repository
        self.spread_multiplier = spread_multiplier
        self.cash_balance = cash_balance
        
        self.shares_hold = {symbol: 0.0 for symbol in self.repository.symbols}

        self._lock = threading.Lock()

    def get_equity(self) -> float:
        with self._lock:
            cash = self.cash_balance
            shares_hold = dict(self.shares_hold)
        return cash + sum(
            shares * self.repository.get_latest_asset_data(symbol)["close"]
            for symbol, shares in shares_hold.items()
        )

    def market_shares_order(self, symbol: str, shares: float) -> None:
        asset_data = self.repository.get_latest_asset_data(symbol)
        cost = self._transaction_cost(shares, asset_data)

        with self._lock:
            self.shares_hold[symbol] += shares
            self.cash_balance -= shares * asset_data['close'] + cost

    def close_all_positions(self) -> None:
        for symbol, shares in self.shares_hold.items():
            if abs(shares) > 0:
                self.market_shares_order(symbol, -shares)

    def get_all_positions(self) -> dict[str: int]:
        return {symbol: shares for symbol, shares in self.shares_hold.items() if abs(shares) > 0}

    def _transaction_cost(self, shares: int, asset_data: pd.Series) -> float:
        return self.spread_multiplier * (asset_data['ask_price'] - asset_data['bid_price']) * abs(shares) / 2

    def get_named_brokerage_state(self) -> dict[str: BrokerageState]:
        with self._lock:
            cash = self.cash_balance
            shares_hold = dict(self.shares_hold)
        return {
            "backtest": BrokerageState(
                equity=self.get_equity(),
                cash_balance=cash,
                shares_hold=shares_hold,
            )
        }