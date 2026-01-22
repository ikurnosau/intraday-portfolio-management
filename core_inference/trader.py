import os
import torch
import torch.nn as nn
import pandas as pd
from typing import Callable
import math
import logging
from concurrent.futures import ThreadPoolExecutor

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
                 portfolio_allocator: nn.Module):
        self.order_size_notional = order_size_notional
        self.data_preparer = data_preparer
        self.features = features
        self.statistics = statistics
        self.repository = repository

        self.brokerage_proxy = brokerage_proxy
        self.brokerage_proxy.close_all_positions()

        self.portfolio_allocator = portfolio_allocator
        # torch.compile is optional; disable by default to avoid Triton dependency in inference
        if torch.cuda.is_available() and bool(int(os.getenv("ENABLE_TORCH_COMPILE", "0"))):
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

        cur_state = self.states_history[-1]
        cur_allocation = cur_state.allocation

        enter_orders = {}
        exit_orders = {}
        for symbol in self.repository.get_symbols():
            if cur_allocation[symbol] * new_allocation[symbol] > 0: 
                abs_difference = abs(new_allocation[symbol]) - abs(cur_allocation[symbol])
                difference = new_allocation[symbol] - cur_allocation[symbol]
                if abs_difference > 0:
                    cash_to_allocate = difference * self.order_size_notional
                    enter_orders[symbol] =  cash_to_allocate // self.repository.get_latest_asset_data(symbol)['close']
                elif abs_difference < 0:
                    proportion_to_liquidate = difference / abs(cur_allocation[symbol])
                    exit_orders[symbol] = round(cur_state.shares_hold[symbol] * proportion_to_liquidate)
            else: 
                if new_allocation[symbol] != 0:
                    cash_to_allocate = new_allocation[symbol] * self.order_size_notional
                    enter_orders[symbol] = cash_to_allocate // self.repository.get_latest_asset_data(symbol)['close']
                if cur_allocation[symbol] != 0:
                    exit_orders[symbol] = - cur_state.shares_hold[symbol]

        logging.info(f"Enter orders: {enter_orders}")
        logging.info(f"Exit orders: {exit_orders}")

        number_of_tasks = len(enter_orders) + len(exit_orders)
        if number_of_tasks > 0:
            logging.info("Starting order execution...")
            with ThreadPoolExecutor(max_workers=number_of_tasks) as executor:
                exit_orders = [executor.submit(self.brokerage_proxy.market_shares_order, symbol, shares) for symbol, shares in exit_orders.items()]
                enter_orders = [executor.submit(self.brokerage_proxy.market_shares_order, symbol, shares) for symbol, shares in enter_orders.items()]
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