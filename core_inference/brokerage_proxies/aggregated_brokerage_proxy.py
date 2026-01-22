from __future__ import annotations

from typing import Iterable, List

from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.models.brokerage_state import BrokerageState


class AggregatedBrokerageProxy(BaseBrokerageProxy):
    def __init__(self, brokerage_proxies: Iterable[BaseBrokerageProxy]):
        self.brokerage_proxies: List[BaseBrokerageProxy] = list(brokerage_proxies)
        if not self.brokerage_proxies:
            raise ValueError("AggregatedBrokerageProxy requires at least one brokerage proxy.")

    def get_equity(self) -> float:
        return self.brokerage_proxies[0].get_equity()

    def market_shares_order(self, symbol: str, shares: float) -> None:
        for proxy in self.brokerage_proxies:
            proxy.market_shares_order(symbol, shares)

    def close_all_positions(self) -> None:
        for proxy in self.brokerage_proxies:
            proxy.close_all_positions()

    def get_all_positions(self) -> dict[str, int]:
        return self.brokerage_proxies[0].get_all_positions()

    def get_named_brokerage_state(self) -> dict[str, BrokerageState]:
        named_states: dict[str, BrokerageState] = {}
        for proxy in self.brokerage_proxies:
            named_states.update(proxy.get_named_brokerage_state())
        return named_states

