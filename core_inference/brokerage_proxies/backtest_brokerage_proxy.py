import json
import logging
import threading
from typing import Any
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

    def get_cash_balance(self) -> float:
        with self._lock:
            return self.cash_balance

    def market_shares_order(
        self,
        symbol: str,
        shares: float,
        order_context: dict | None = None,
    ) -> None:
        asset_data = self.repository.get_latest_asset_data(symbol)
        cost = self._transaction_cost(shares, asset_data)
        reference_price = float(asset_data["close"])
        effective_fill_price = reference_price + cost / shares

        with self._lock:
            cash_before = self.cash_balance
            self.shares_hold[symbol] += shares
            self.cash_balance -= shares * reference_price + cost
            cash_after = self.cash_balance

        logging.info(
            "execution_metric=%s",
            json.dumps(
                {
                    "event": "shadow_fill",
                    "cycle_id": (order_context or {}).get("cycle_id"),
                    "broker": "backtest",
                    "symbol": symbol,
                    "side": "buy" if shares > 0 else "sell",
                    "quantity": abs(float(shares)),
                    "reference_close": reference_price,
                    "bar_timestamp": asset_data.get("date"),
                    "modeled_bid": float(asset_data["bid_price"]),
                    "modeled_ask": float(asset_data["ask_price"]),
                    "modeled_midpoint": (
                        float(asset_data["bid_price"])
                        + float(asset_data["ask_price"])
                    ) / 2,
                    "execution_market": asset_data,
                    "spread_multiplier": self.spread_multiplier,
                    "modeled_transaction_cost": cost,
                    "effective_fill_price": effective_fill_price,
                    "cash_before": cash_before,
                    "cash_after": cash_after,
                    "cash_delta": cash_after - cash_before,
                },
                default=str,
                sort_keys=True,
            ),
        )

    def close_all_positions(self) -> None:
        for symbol, shares in self.shares_hold.items():
            if abs(shares) > 0:
                self.market_shares_order(symbol, -shares)

    def get_all_positions(self) -> dict[str: int]:
        return {symbol: shares for symbol, shares in self.shares_hold.items() if abs(shares) > 0}

    def _transaction_cost(
        self,
        shares: float,
        asset_data: dict[str, Any],
    ) -> float:
        return self.spread_multiplier * (asset_data['ask_price'] - asset_data['bid_price']) * abs(shares) / 2

    def get_named_brokerage_state(self) -> dict[str: BrokerageState]:
        with self._lock:
            cash = self.cash_balance
            shares_hold = self.get_all_positions()
        return {
            "backtest": BrokerageState(
                equity=self.get_equity(),
                cash_balance=cash,
                shares_hold=shares_hold,
            )
        }