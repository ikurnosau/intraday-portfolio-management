from abc import ABC, abstractmethod


class BaseBrokerageProxy(ABC):
    @abstractmethod
    def get_cash_balance(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def market_buy_notional(self, symbol: str, cash: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def market_sell_shares(self, symbol: str, shares: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def close_all_positions(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_shares_hold(self) -> dict[str: float]:
        raise NotImplementedError("Subclasses must implement this method")