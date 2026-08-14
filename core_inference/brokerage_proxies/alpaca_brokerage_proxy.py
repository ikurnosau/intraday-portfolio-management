import json
import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

from config.settings import Settings, get_settings
from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.models.brokerage_state import BrokerageState
from core_inference.repository import Repository


class AlpacaBrokerageProxy(BaseBrokerageProxy):
    def __init__(
        self,
        repository: Repository,
        paper: bool = True,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.paper = paper
        self.repository = repository
        self.trading_client = TradingClient(
            settings.alpaca.paper_api_key,
            settings.alpaca.paper_api_secret,
            paper=self.paper,
        )

        open_positions = self.trading_client.get_all_positions()
        if open_positions:
            logging.info(
                "Closing %s Alpaca position(s) during initialization",
                len(open_positions),
            )
            for position in open_positions:
                logging.info("%s", position)
            self.close_all_positions()

        logging.info(
            "AlpacaBrokerageProxy instantiated. Available equity: %s. Cash balance: %s",
            self.get_equity(),
            self.get_cash_balance(),
        )

    def get_equity(self) -> float:
        return float(self.trading_client.get_account().equity)

    def get_cash_balance(self) -> float:
        return float(self.trading_client.get_account().cash)

    def market_shares_order(
        self,
        symbol: str,
        shares: float,
        order_context: dict | None = None,
    ) -> None:
        if shares == 0:
            return
        side = OrderSide.BUY if shares > 0 else OrderSide.SELL
        submit_market = self._latest_asset_data(symbol)
        submit_started_at = time.time()
        submit_started_monotonic = time.perf_counter()
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=abs(shares),
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = self.trading_client.submit_order(order_data=market_order_data)
        submitted_at = time.time()
        submitted_monotonic = time.perf_counter()
        logging.info(
            "Market %s shares order submitted for %s with shares %s",
            side,
            symbol,
            abs(shares),
        )

        filled_order = self._wait_for_fill(order.id)
        fill_observed_at = time.time()
        fill_observed_monotonic = time.perf_counter()
        fill_market = self._latest_asset_data(symbol)
        fill_price = float(filled_order.filled_avg_price)
        decision_market = (order_context or {}).get("decision_market")
        midpoint_slippage = (
            fill_price - submit_market["midpoint"]
            if shares > 0
            else submit_market["midpoint"] - fill_price
        ) if submit_market else None
        touch_slippage = (
            fill_price - submit_market["ask_price"]
            if shares > 0
            else submit_market["bid_price"] - fill_price
        ) if submit_market else None
        quote_move = (
            fill_market["midpoint"] - submit_market["midpoint"]
            if shares > 0
            else submit_market["midpoint"] - fill_market["midpoint"]
        ) if submit_market and fill_market else None
        decision_midpoint_move = (
            submit_market["midpoint"] - decision_market["midpoint"]
            if shares > 0
            else decision_market["midpoint"] - submit_market["midpoint"]
        ) if decision_market and submit_market else None

        logging.info(
            "execution_metric=%s",
            json.dumps(
                {
                    "event": "live_fill",
                    "cycle_id": (order_context or {}).get("cycle_id"),
                    "broker": f"alpaca_{'paper' if self.paper else 'live'}",
                    "order_id": order.id,
                    "client_order_id": getattr(order, "client_order_id", None),
                    "symbol": symbol,
                    "side": "buy" if shares > 0 else "sell",
                    "requested_quantity": abs(float(shares)),
                    "filled_quantity": float(filled_order.filled_qty),
                    "filled_avg_price": fill_price,
                    "order_created_at": getattr(filled_order, "created_at", None),
                    "order_submitted_at": getattr(filled_order, "submitted_at", None),
                    "order_filled_at": getattr(filled_order, "filled_at", None),
                    "submit_started_at": submit_started_at,
                    "submit_completed_at": submitted_at,
                    "fill_observed_at": fill_observed_at,
                    "filled_notional": fill_price * float(filled_order.filled_qty),
                    "commission": getattr(filled_order, "commission", None),
                    "submit_api_latency_ms": (
                        submitted_monotonic - submit_started_monotonic
                    ) * 1000,
                    "submit_to_fill_observed_ms": (
                        fill_observed_monotonic - submitted_monotonic
                    ) * 1000,
                    "submit_market": submit_market,
                    "fill_market": fill_market,
                    "decision_market": decision_market,
                    "decision_to_submit_started_ms": (
                        (
                            submit_started_monotonic
                            - order_context["decision_monotonic"]
                        ) * 1000
                        if order_context and order_context.get("decision_monotonic")
                        else None
                    ),
                    "adverse_midpoint_move_decision_to_submit_per_share": decision_midpoint_move,
                    "slippage_vs_submit_midpoint_per_share": midpoint_slippage,
                    "slippage_vs_submit_touch_per_share": touch_slippage,
                    "adverse_midpoint_move_to_fill_per_share": quote_move,
                    "total_midpoint_slippage": (
                        midpoint_slippage * float(filled_order.filled_qty)
                        if midpoint_slippage is not None
                        else None
                    ),
                },
                default=str,
                sort_keys=True,
            ),
        )
        logging.info(
            "Market %s shares order filled for %s",
            side,
            symbol,
        )

    def close_all_positions(self) -> None:
        close_responses = self.trading_client.close_all_positions()
        for response in close_responses:
            order = getattr(response, "body", response)
            order_id = getattr(order, "id", None)
            if order_id is None:
                logging.warning(
                    "Unable to wait for Alpaca close-position response: %s",
                    response,
                )
                continue
            self._wait_for_fill(order_id)
        logging.info("All Alpaca positions closed")

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

    def _latest_asset_data(self, symbol: str) -> dict:
        return self.repository.get_latest_asset_data(symbol)

    def _wait_for_fill(self, order_id: str):
        """
        Could be done better by subscribing to TradingStream
        """

        previous_status = None
        while True:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status != previous_status:
                logging.info(
                    "execution_metric=%s",
                    json.dumps(
                        {
                            "event": "live_order_status",
                            "order_id": order_id,
                            "status": order.status,
                            "filled_quantity": getattr(order, "filled_qty", None),
                            "filled_avg_price": getattr(order, "filled_avg_price", None),
                            "observed_at": time.time(),
                        },
                        default=str,
                        sort_keys=True,
                    ),
                )
                previous_status = order.status
            if order.status == OrderStatus.FILLED:
                return order
            time.sleep(0.2)
