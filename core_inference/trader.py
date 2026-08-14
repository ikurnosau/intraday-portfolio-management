import json
import torch
import torch.nn as nn
import pandas as pd
from typing import Callable
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import Settings, get_settings
from core_data_prep.core_data_prep import DataPreparer
from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy
from core_inference.repository import Repository
from core_inference.models.trader_state import TraderState


class Trader:
    def __init__(self, 
                 order_size_notional: float,
                 data_preparer: DataPreparer,
                 features: dict[str, Callable],
                 statistics: dict[str, Callable],
                 brokerage_proxy: BaseBrokerageProxy,
                 repository: Repository,
                 portfolio_allocator: nn.Module,
                 settings: Settings | None = None):
        settings = settings or get_settings()
        self.order_size_notional = order_size_notional
        self.data_preparer = data_preparer
        self.features = features
        self.statistics = statistics
        self.repository = repository

        self.brokerage_proxy = brokerage_proxy

        self.portfolio_allocator = portfolio_allocator
        # torch.compile is optional; disable by default to avoid Triton dependency in inference
        if (
            torch.cuda.is_available()
            and settings.runtime.enable_torch_compile
        ):
            try:
                self.portfolio_allocator = torch.compile(self.portfolio_allocator, mode="reduce-overhead")
            except Exception as exc:  # pragma: no cover - defensive fallback
                logging.warning("torch.compile unavailable, using eager mode: %s", exc)
        self.portfolio_allocator.eval()

        initial_brokerage_states = self.brokerage_proxy.get_named_brokerage_state()
        self.session_start_states = initial_brokerage_states
        self.states_history: list[TraderState] = [
            TraderState(
                allocation={symbol: 0.0 for symbol in self.repository.symbols},
                shares_hold={symbol: 0 for symbol in self.repository.symbols},
                brokerage_states=initial_brokerage_states,
            )
        ]
        self._log_reconciliation("session_start", initial_brokerage_states)

    def perform_trading_cycle(self):
        cycle_id = uuid.uuid4().hex
        cycle_started_at = time.time()
        cycle_timer = time.perf_counter()
        logging.info("Starting trading cycle cycle_id=%s...", cycle_id)
        asset_dfs = self.repository.get_asset_dfs()

        logging.info("Transforming data for inference...")
        transform_timer = time.perf_counter()
        x_numpy, statistics = self.data_preparer.transform_data_for_inference(
            data=asset_dfs,
            n_timestamps=1,
            features=self.features,
            include_target=False,
            include_statistics=True,
            statistics=self.statistics,
        )
        transform_ms = (time.perf_counter() - transform_timer) * 1000
        x = torch.from_numpy(x_numpy).float()# .unsqueeze(0)
        x = x.to(next(self.portfolio_allocator.parameters()).device)

        spread = torch.from_numpy(statistics['spread']).float()
        volatility = torch.from_numpy(statistics['volatility']).float()
        spread = spread.to(next(self.portfolio_allocator.parameters()).device)
        volatility = volatility.to(next(self.portfolio_allocator.parameters()).device)

        logging.info("Running portfolio allocator...")
        allocator_timer = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            prediction = self.portfolio_allocator(x, spread, volatility)[0].cpu().numpy()
        allocator_ms = (time.perf_counter() - allocator_timer) * 1000
        new_allocation = {symbol: prediction[0, i] for i, symbol in enumerate(asset_dfs)}

        new_allocation_log = {symbol: new_allocation[symbol] for symbol in new_allocation if new_allocation[symbol] != 0}
        logging.info(f"New allocation predicted: {new_allocation_log}")

        current_positions = self.brokerage_proxy.get_all_positions()
        orders = {}
        order_contexts = {}
        decision_observed_at = time.time()
        decision_monotonic = time.perf_counter()
        for symbol in self.repository.get_symbols():
            latest_data = self.repository.get_latest_asset_data(symbol)
            latest_close = latest_data["close"]
            target_notional = new_allocation[symbol] * self.order_size_notional
            target_shares = target_notional // latest_close
            current_shares = current_positions.get(symbol, 0)
            shares_delta = target_shares - current_shares
            if shares_delta != 0:
                orders[symbol] = shares_delta
                decision_market = latest_data
                order_contexts[symbol] = {
                    "cycle_id": cycle_id,
                    "decision_observed_at": decision_observed_at,
                    "decision_monotonic": decision_monotonic,
                    "decision_market": decision_market,
                }
                logging.info(
                    "execution_metric=%s",
                    json.dumps(
                        {
                            "event": "order_decision",
                            "cycle_id": cycle_id,
                            "symbol": symbol,
                            "side": "buy" if shares_delta > 0 else "sell",
                            "quantity": abs(float(shares_delta)),
                            "allocation": float(new_allocation[symbol]),
                            "current_shares": float(current_shares),
                            "target_shares": float(target_shares),
                            "target_notional": float(target_notional),
                            "bar_close": float(latest_close),
                            "bar_timestamp": latest_data.get("date"),
                            "bar_received_at": latest_data["bar_received_at"],
                            "bar_age_at_decision_ms": latest_data["bar_age_ms"],
                            "decision_observed_at": decision_observed_at,
                            "decision_market": decision_market,
                        },
                        default=str,
                        sort_keys=True,
                    ),
                )

        logging.info("Orders: %s", orders)

        number_of_tasks = len(orders)
        execution_timer = time.perf_counter()
        if number_of_tasks > 0:
            logging.info("Starting order execution...")
            with ThreadPoolExecutor(max_workers=number_of_tasks) as executor:
                futures = {
                    executor.submit(
                        self.brokerage_proxy.market_shares_order,
                        symbol,
                        shares,
                        order_contexts[symbol],
                    ): (symbol, shares)
                    for symbol, shares in orders.items()
                }
                for future in as_completed(futures):
                    symbol, shares = futures[future]
                    try:
                        future.result()
                    except Exception:
                        logging.exception(
                            "Order failed for %s with shares %s",
                            symbol,
                            shares,
                        )
                        raise
        else:
            logging.info("No orders to execute")

        logging.info("Order execution completed!")
        execution_ms = (time.perf_counter() - execution_timer) * 1000

        brokerage_states = self.brokerage_proxy.get_named_brokerage_state()
        logging.info(f"Brokerage states: {brokerage_states}")
        self._log_reconciliation(cycle_id, brokerage_states)
        logging.info(
            "execution_metric=%s",
            json.dumps(
                {
                    "event": "cycle_timing",
                    "cycle_id": cycle_id,
                    "cycle_started_at": cycle_started_at,
                    "transform_ms": transform_ms,
                    "allocator_ms": allocator_ms,
                    "execution_ms": execution_ms,
                    "total_cycle_ms": (time.perf_counter() - cycle_timer) * 1000,
                    "order_count": number_of_tasks,
                },
                sort_keys=True,
            ),
        )

        self.states_history.append(TraderState(
            allocation=new_allocation,
            shares_hold=self.brokerage_proxy.get_all_positions(),
            brokerage_states=brokerage_states,
        ))

    def _log_reconciliation(self, cycle_id: str, brokerage_states: dict) -> None:
        normalized_states = {}
        for name, state in brokerage_states.items():
            baseline = self.session_start_states[name]
            normalized_states[name] = {
                "equity": state.equity,
                "cash": state.cash_balance,
                "positions": state.shares_hold,
                "session_pnl": state.equity - baseline.equity,
                "session_cash_delta": state.cash_balance - baseline.cash_balance,
            }

        comparisons = []
        names = list(brokerage_states)
        if len(names) > 1:
            primary_name = names[0]
            primary = normalized_states[primary_name]
            for comparison_name in names[1:]:
                comparison = normalized_states[comparison_name]
                symbols = set(primary["positions"]) | set(comparison["positions"])
                comparisons.append(
                    {
                        "primary": primary_name,
                        "comparison": comparison_name,
                        "raw_equity_gap_comparison_minus_primary": (
                            comparison["equity"] - primary["equity"]
                        ),
                        "session_pnl_gap_comparison_minus_primary": (
                            comparison["session_pnl"] - primary["session_pnl"]
                        ),
                        "cash_gap_comparison_minus_primary": (
                            comparison["cash"] - primary["cash"]
                        ),
                        "position_differences_comparison_minus_primary": {
                            symbol: (
                                comparison["positions"].get(symbol, 0)
                                - primary["positions"].get(symbol, 0)
                            )
                            for symbol in sorted(symbols)
                            if comparison["positions"].get(symbol, 0)
                            != primary["positions"].get(symbol, 0)
                        },
                    }
                )

        logging.info(
            "execution_metric=%s",
            json.dumps(
                {
                    "event": "brokerage_reconciliation",
                    "cycle_id": cycle_id,
                    "states": normalized_states,
                    "comparisons": comparisons,
                },
                default=str,
                sort_keys=True,
            ),
        )