import os
import logging
import asyncio
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.stream import TradingStream

from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy


class AlpacaBrokerageProxy(BaseBrokerageProxy):
    def __init__(self, paper: bool=True): 
        self.paper = paper
        self.trading_client = TradingClient(os.getenv('API_KEY'), os.getenv('API_SECRET'), paper=self.paper)

        open_positions = self.trading_client.get_all_positions()
        logging.info(f"AlpacaBrokerageProxy instantiated. Available cash: {self.get_cash_balance()}. Open positions:")
        for position in open_positions:
            logging.info(f"{position}")

    def get_cash_balance(self) -> float:
        return float(self.trading_client.get_account().cash)

    def market_buy_notional(self, symbol: str, cash: float) -> None:
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            notional=cash,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        order = self.trading_client.submit_order(order_data=market_order_data)
        logging.info(f"Market buy notional order submitted for {symbol} with cash {cash}")

        self._wait_for_fill(order.id)
        logging.info(f"Market buy notional order filled for {symbol}")

    def market_sell_shares(self, symbol: str, shares: float) -> None:
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        order = self.trading_client.submit_order(order_data=market_order_data)
        logging.info(f"Market sell shares order submitted for {symbol} with shares {shares}")

        self._wait_for_fill(order.id)
        logging.info(f"Market sell shares order filled for {symbol}")

    def close_all_positions(self) -> None:
        self.trading_client.close_all_positions()
        logging.info("All positions closed")

    def get_shares_hold(self) -> dict[str: float]:
        open_positions = self.trading_client.get_all_positions()
        return {position.symbol: float(position.qty) for position in open_positions}

    def _wait_for_fill(self, order_id: str):
        """
        Could be done better by subscribing to TradingStream
        """

        while True:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                break
            time.sleep(0.2)
