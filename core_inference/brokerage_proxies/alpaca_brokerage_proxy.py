import os
import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.models.brokerage_state import BrokerageState


class AlpacaBrokerageProxy(BaseBrokerageProxy):
    def __init__(self, paper: bool=True): 
        self.paper = paper
        self.trading_client = TradingClient(os.getenv('API_KEY'), os.getenv('API_SECRET'), paper=self.paper)

        open_positions = self.trading_client.get_all_positions()
        logging.info(
            "AlpacaBrokerageProxy instantiated. Available equity: %s. Open positions:",
            self.get_equity(),
        )
        for position in open_positions:
            logging.info("%s", position)

    def get_equity(self) -> float:
        return float(self.trading_client.get_account().equity)

    def market_shares_order(self, symbol: str, shares: float) -> None:
        if shares == 0:
            return
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=abs(shares),
            side=OrderSide.BUY if shares > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        order = self.trading_client.submit_order(order_data=market_order_data)
        logging.info(
            "Market %s shares order submitted for %s with shares %s",
            OrderSide.BUY if shares > 0 else OrderSide.SELL,
            symbol,
            abs(shares),
        )

        self._wait_for_fill(order.id)
        logging.info(
            "Market %s shares order filled for %s",
            OrderSide.BUY if shares > 0 else OrderSide.SELL,
            symbol,
        )

    def close_all_positions(self) -> None:
        self.trading_client.close_all_positions()
        logging.info("All positions closed")

    def get_all_positions(self) -> dict[str: float]:
        open_positions = self.trading_client.get_all_positions()
        positions = {}
        for position in open_positions:
            positions[position.symbol] = float(position.qty)
        return positions

    def get_named_brokerage_state(self) -> dict[str: BrokerageState]:
        account = self.trading_client.get_account()
        cash_value = getattr(account, "cash", 0)
        name = f"alpaca_{'paper' if self.paper else 'live'}"
        return {
            name: BrokerageState(
                equity=float(account.equity),
                cash_balance=float(cash_value),
                shares_hold=self.get_all_positions(),
            )
        }

    def _wait_for_fill(self, order_id: str):
        """
        Could be done better by subscribing to TradingStream
        """

        while True:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                break
            time.sleep(0.2)
