from abc import ABC, abstractmethod

from core_inference.models.brokerage_state import BrokerageState


class BaseBrokerageProxy(ABC):
    @abstractmethod
    def get_equity(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")
        
    @abstractmethod
    def market_shares_order(self, symbol: str, shares: float) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def close_all_positions(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_all_positions(self) -> dict[str: int]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_named_brokerage_state(self) -> dict[str: BrokerageState]:
        raise NotImplementedError("Subclasses must implement this method")