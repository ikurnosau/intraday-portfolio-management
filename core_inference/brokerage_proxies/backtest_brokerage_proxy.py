import pandas as pd

from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.repository import Repository
from core_inference.models.position import Position


class BacktestBrokerageProxy(BaseBrokerageProxy):
    def __init__(self, repository: Repository, spread_multiplier: float = 1.0, cash_balance: float = 1.0): 
        self.repository = repository
        self.spread_multiplier = spread_multiplier
        self.cash_balance = cash_balance
        
        self.shares_hold = {symbol: 0.0 for symbol in self.repository.symbols}

    def get_cash_balance(self) -> float:
        return self.cash_balance

    def market_buy_notional(self, symbol: str, cash: float) -> None:
        asset_price = self.repository.get_latest_asset_data(symbol)['close']
        self.shares_hold[symbol] += cash / asset_price
        self.cash_balance -= cash

    def market_sell_shares(self, symbol: str, shares: float) -> None:
        asset_data = self.repository.get_latest_asset_data(symbol)
        self.shares_hold[symbol] -= shares
        self.cash_balance += shares * asset_data['close'] * (1 - self._transaction_cost(asset_data))

    def close_all_positions(self) -> None:
        for symbol, shares in self.shares_hold.items():
            if shares > 0:
                self.market_sell_shares(symbol, shares)

    def get_all_positions(self) -> dict[str: Position]:
        return {symbol: Position(quantity=shares, current_price=self.repository.get_latest_asset_data(symbol)['close']) \
            for symbol, shares in self.shares_hold.items() if shares > 0}

    def _transaction_cost(self, asset_data: pd.Series) -> float:
        return self.spread_multiplier * (asset_data['ask_price'] - asset_data['bid_price']) / asset_data['bid_price']