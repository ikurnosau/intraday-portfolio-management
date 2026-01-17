from abc import ABC, abstractmethod

from core_inference.models.position import Position


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