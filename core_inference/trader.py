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
from core_inference.models.state import State
from core_inference.models.position import Position


class Trader:
    def __init__(self, 
                 data_preparer: DataPreparer,
                 features: dict[str, Callable],
                 brokerage_proxy: BaseBrokerageProxy,
                 repository: Repository,
                 portfolio_allocator: nn.Module):
        self.data_preparer = data_preparer
        self.features = features
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

        self.states_history: list[State] = [
            State(
                desired_position={symbol: 0.0 for symbol in self.repository.symbols},
                position={symbol: 0.0 for symbol in self.repository.symbols},
                available_cash=self.brokerage_proxy.get_cash_balance() / 2,
                shares_hold={symbol: 0.0 for symbol in self.repository.symbols},

                _position_difference={symbol: 0.0 for symbol in self.repository.symbols},
                _buy_positions={symbol: 0.0 for symbol in self.repository.symbols},
                _buy_cash_per_asset={symbol: 0.0 for symbol in self.repository.symbols},
                _sell_positions={symbol: 0.0 for symbol in self.repository.symbols},
                _sell_percentage_per_share={symbol: 0.0 for symbol in self.repository.symbols},
                _sell_shares_per_asset={symbol: 0.0 for symbol in self.repository.symbols},
            )
        ]

    def perform_trading_cycle(self):
        logging.info("Starting trading cycle...")
        asset_dfs = self.repository.get_asset_dfs()

        logging.info("Transforming data for inference...")
        x_numpy = self.data_preparer.transform_data_for_inference(
            data=asset_dfs,
            n_timestamps=1,
            features=self.features,
            include_target_and_statistics=False,
        )
        x = torch.from_numpy(x_numpy).float()# .unsqueeze(0)
        x = x.to(next(self.portfolio_allocator.parameters()).device)

        logging.info("Running portfolio allocator...")
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            prediction = self.portfolio_allocator(x).cpu().numpy()
        new_position = {symbol: prediction[0, i] for i, symbol in enumerate(asset_dfs)}

        logging.info("Calculating position difference...")
        cur_state = self.states_history[-1]
        position_difference = {symbol: new_position[symbol] - cur_state.position[symbol] for symbol in new_position}

        buy_positions = {symbol: position_difference[symbol] for symbol in position_difference if position_difference[symbol] > 0}
        buy_cash_per_asset = {symbol: buy_positions[symbol] * cur_state.available_cash for symbol in buy_positions}

        sell_positions = {symbol: - position_difference[symbol] for symbol in position_difference if position_difference[symbol] < 0}
        sell_percentage_per_share = {symbol: math.ceil((sell_positions[symbol] / cur_state.position[symbol]) * 1e5) / 1e5 for symbol in sell_positions}
        sell_shares_per_asset = {symbol: cur_state.shares_hold[symbol] * sell_percentage_per_share[symbol] for symbol in sell_positions }

        number_of_tasks = len(buy_positions) + len(sell_positions)
        if number_of_tasks > 0:
            logging.info("Starting order execution...")
            with ThreadPoolExecutor(max_workers=number_of_tasks) as executor:
                buy_futures = [executor.submit(self.brokerage_proxy.market_buy_notional, symbol, cash) for symbol, cash in buy_cash_per_asset.items()]
                sell_futures = [executor.submit(self.brokerage_proxy.market_sell_shares, symbol, shares) for symbol, shares in sell_shares_per_asset.items()]
        else:
            logging.info("No orders to execute")

        logging.info("Order execution completed!")

        positions = self.brokerage_proxy.get_all_positions()
        positions = {symbol: positions[symbol] if symbol in positions else Position(quantity=0.0, current_price=0.0) for symbol in new_position}
        total_value = sum(position.quantity * position.current_price for position in positions.values())
        executed_position = {symbol: position.quantity * position.current_price / total_value for symbol, position in positions.items()}
        
        self.states_history.append(State(
            desired_position=new_position,
            position=executed_position,
            available_cash=self.brokerage_proxy.get_cash_balance(),
            shares_hold={symbol: position.quantity for symbol, position in positions.items()},
            
            _position_difference=position_difference,
            _buy_positions=buy_positions,
            _buy_cash_per_asset=buy_cash_per_asset,
            _sell_positions=sell_positions,
            _sell_percentage_per_share=sell_percentage_per_share,
            _sell_shares_per_asset=sell_shares_per_asset,
        ))