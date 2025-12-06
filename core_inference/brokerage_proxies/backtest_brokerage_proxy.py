from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.repository import Repository


class BacktestBrokerageProxy(BaseBrokerageProxy):
    def __init__(self, repository: Repository): 
        self.repository = repository
        self.cash_balance = 1.
        self.positions = {symbol: 0.0 for symbol in self.repository.symbols}

    def get_cash_balance(self) -> float:
        return self.cash_balance

    def market_buy_notional(self, symbol: str, cash: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def market_sell_shares(self, symbol: str, shares: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def close_all_positions(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def get_shares_hold(self) -> dict[str: float]:
        raise NotImplementedError("Subclasses must implement this method")
