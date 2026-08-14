import torch
import torch.nn as nn
import pandas as pd
from typing import Callable
import logging
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
        self.brokerage_proxy.close_all_positions()

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

        self.states_history: list[TraderState] = [
            TraderState(
                allocation={symbol: 0.0 for symbol in self.repository.symbols},
                shares_hold={symbol: 0 for symbol in self.repository.symbols},
                brokerage_states=self.brokerage_proxy.get_named_brokerage_state(),
            )
        ]

    def perform_trading_cycle(self):
        logging.info("Starting trading cycle...")
        asset_dfs = self.repository.get_asset_dfs()

        logging.info("Transforming data for inference...")
        x_numpy, statistics = self.data_preparer.transform_data_for_inference(
            data=asset_dfs,
            n_timestamps=1,
            features=self.features,
            include_target=False,
            include_statistics=True,
            statistics=self.statistics,
        )
        x = torch.from_numpy(x_numpy).float()# .unsqueeze(0)
        x = x.to(next(self.portfolio_allocator.parameters()).device)

        spread = torch.from_numpy(statistics['spread']).float()
        volatility = torch.from_numpy(statistics['volatility']).float()
        spread = spread.to(next(self.portfolio_allocator.parameters()).device)
        volatility = volatility.to(next(self.portfolio_allocator.parameters()).device)

        logging.info("Running portfolio allocator...")
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            prediction = self.portfolio_allocator(x, spread, volatility)[0].cpu().numpy()
        new_allocation = {symbol: prediction[0, i] for i, symbol in enumerate(asset_dfs)}

        new_allocation_log = {symbol: new_allocation[symbol] for symbol in new_allocation if new_allocation[symbol] != 0}
        logging.info(f"New allocation predicted: {new_allocation_log}")

        current_positions = self.brokerage_proxy.get_all_positions()
        orders = {}
        for symbol in self.repository.get_symbols():
            latest_close = self.repository.get_latest_asset_data(symbol)["close"]
            target_notional = new_allocation[symbol] * self.order_size_notional
            target_shares = target_notional // latest_close
            current_shares = current_positions.get(symbol, 0)
            shares_delta = target_shares - current_shares
            if shares_delta != 0:
                orders[symbol] = shares_delta

        logging.info("Orders: %s", orders)

        number_of_tasks = len(orders)
        if number_of_tasks > 0:
            logging.info("Starting order execution...")
            with ThreadPoolExecutor(max_workers=number_of_tasks) as executor:
                futures = {
                    executor.submit(
                        self.brokerage_proxy.market_shares_order,
                        symbol,
                        shares,
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

        brokerage_states = self.brokerage_proxy.get_named_brokerage_state()
        logging.info(f"Brokerage states: {brokerage_states}")

        self.states_history.append(TraderState(
            allocation=new_allocation,
            shares_hold=self.brokerage_proxy.get_all_positions(),
            brokerage_states=brokerage_states,
        ))