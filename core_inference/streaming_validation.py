import logging

import numpy as np
import pandas as pd
import torch

from config.constants import Constants
from config.experiment_config import ExperimentConfig
from core_data_prep.core_data_prep import DataPreparer
from core_data_prep.validations import (
    validate_streaming_vs_offline_returns,
    validate_streaming_vs_offline_x,
)
from core_inference.brokerage_proxies.backtest_brokerage_proxy import (
    BacktestBrokerageProxy,
)
from core_inference.repository import Repository
from core_inference.trader import Trader


def run_streaming_validation(
    *,
    config: ExperimentConfig,
    data_preparer: DataPreparer,
    allocator: torch.nn.Module,
    daily_slices: list[dict[str, pd.DataFrame]],
    x_test: np.ndarray,
    allocations: np.ndarray,
    realized_returns: np.ndarray,
    next_returns: np.ndarray,
    validate_inputs: bool = True,
    validate_returns: bool = True,
    initial_cash: float = 1e10,
    order_size_notional: float = 1e9,
    reset_allocator: bool = False,
    n_days: int | None = None,
) -> float:
    if n_days is not None and n_days < 0:
        raise ValueError("n_days must be non-negative or None")

    evaluation_slices = (
        daily_slices if n_days is None else daily_slices[:n_days]
    )
    symbols = sorted(config.data_config.symbol_or_symbols)
    n_day_steps = Constants.Data.TRADING_DAY_LENGTH_MINUTES
    required_steps = len(evaluation_slices) * n_day_steps

    if reset_allocator:
        allocator.reset_counter()

    if validate_inputs:
        assert len(x_test) >= required_steps
        print("Validating streaming inputs against offline X_test")
    if validate_returns:
        assert len(allocations) >= required_steps
        assert len(realized_returns) >= required_steps
        assert len(next_returns) >= required_steps
        print(
            "Validating streaming allocations/returns against offline "
            "test_cum_wealth outputs"
        )

    cur_cash = initial_cash
    for day_i, daily_slice in enumerate(evaluation_slices):
        replay_day = max(
            asset_df["date"].max() for asset_df in daily_slice.values()
        ).normalize()
        replay_start = replay_day + pd.Timedelta(
            hours=Constants.Data.REGULAR_TRADING_HOURS_START.hour,
            minutes=Constants.Data.REGULAR_TRADING_HOURS_START.minute,
        )
        replay_end = replay_day + pd.Timedelta(
            hours=Constants.Data.REGULAR_TRADING_HOURS_END.hour,
            minutes=Constants.Data.REGULAR_TRADING_HOURS_END.minute,
        )
        replay_timestamps = pd.date_range(
            replay_start,
            replay_end,
            freq="1min",
        )

        cur_day_initialization = {
            asset_name: (
                asset_df.loc[asset_df["date"] < replay_start]
                .copy()
                .reset_index(drop=True)
            )
            for asset_name, asset_df in daily_slice.items()
        }
        assert all(
            not asset_df.empty
            for asset_df in cur_day_initialization.values()
        ), "Every asset must have history before the first replay timestamp"

        updates_by_timestamp = {
            timestamp: {} for timestamp in replay_timestamps
        }
        for asset_name, asset_df in daily_slice.items():
            replay_rows = asset_df.loc[
                asset_df["date"].between(
                    replay_start,
                    replay_end,
                    inclusive="both",
                )
            ]
            assert not replay_rows["date"].duplicated().any(), (
                f"Duplicate bar timestamps for {asset_name} on "
                f"{replay_day.date()}"
            )
            for _, row in replay_rows.iterrows():
                updates_by_timestamp[pd.Timestamp(row["date"])][
                    asset_name
                ] = row

        repository = Repository(
            trading_symbols=config.data_config.symbol_or_symbols,
            required_history_depth=(
                config.data_config.in_seq_len
                + config.data_config.normalizer.get_window()
                + 30
            ),
            bars_and_quotes=cur_day_initialization,
        )
        backtest_proxy = BacktestBrokerageProxy(
            repository=repository,
            spread_multiplier=config.rl_config.spread_multiplier,
            cash_balance=cur_cash,
            market_snapshot_key="pre_submit_market",
        )
        trader = Trader(
            order_size_notional=order_size_notional,
            data_preparer=data_preparer,
            features=config.data_config.features_polars,
            statistics={
                "spread": config.data_config.statistics["spread"],
                "volatility": config.data_config.statistics["volatility"],
            },
            brokerage_proxy=backtest_proxy,
            repository=repository,
            portfolio_allocator=allocator,
        )

        trading_cycle_i = 0
        prev_equity_after_trade = None
        prev_offline_idx = None
        prev_trade_cost_return = None
        for timestamp_i, timestamp in enumerate(replay_timestamps):
            timestamp_updates = updates_by_timestamp[timestamp]
            if not timestamp_updates:
                continue

            for stock_name, stock_data_series in timestamp_updates.items():
                stock_data = stock_data_series.to_dict()
                stock_data["symbol"] = stock_name
                repository.add_bar(stock_data)

            offline_idx = day_i * n_day_steps + timestamp_i

            streaming_step_return = None
            offline_realized_return = None
            if (
                validate_returns
                and prev_equity_after_trade is not None
                and prev_offline_idx is not None
                and prev_trade_cost_return is not None
            ):
                equity_before_trade = backtest_proxy.get_equity()
                mtm_return = (
                    equity_before_trade - prev_equity_after_trade
                ) / order_size_notional
                streaming_step_return = mtm_return - prev_trade_cost_return
                offline_realized_return = float(
                    realized_returns[prev_offline_idx]
                )

            if validate_inputs:
                x_streaming, _ = (
                    trader.data_preparer.transform_data_for_inference(
                        data=trader.repository.get_asset_dfs(),
                        n_timestamps=1,
                        features=trader.features,
                        include_target=False,
                        include_statistics=True,
                        statistics=trader.statistics,
                    )
                )
                validate_streaming_vs_offline_x(
                    x_test[offline_idx],
                    x_streaming[0],
                    feature_names=list(
                        config.data_config.features_polars
                    ),
                    symbols=symbols,
                    day_i=day_i,
                    timestamp=timestamp,
                    timestamp_i=timestamp_i,
                )

            equity_before_trade = backtest_proxy.get_equity()
            trader.perform_trading_cycle()
            equity_after_trade = backtest_proxy.get_equity()
            trade_cost_return = (
                equity_before_trade - equity_after_trade
            ) / order_size_notional

            if validate_returns:
                streaming_allocation = np.array(
                    [
                        trader.states_history[-1].allocation[symbol]
                        for symbol in symbols
                    ],
                    dtype=np.float64,
                )
                validate_streaming_vs_offline_returns(
                    streaming_allocation=streaming_allocation,
                    offline_allocation=allocations[offline_idx],
                    streaming_step_return=streaming_step_return,
                    offline_realized_return=offline_realized_return,
                    day_i=day_i,
                    timestamp=timestamp,
                    timestamp_i=timestamp_i,
                    symbols=symbols,
                )

            prev_equity_after_trade = equity_after_trade
            prev_offline_idx = offline_idx
            prev_trade_cost_return = trade_cost_return

            logging.info(
                "Day %s update %s at %s ended with equity %s",
                day_i,
                trading_cycle_i,
                timestamp,
                equity_after_trade,
            )
            trading_cycle_i += 1

        backtest_proxy.close_all_positions()
        cur_cash = backtest_proxy.get_equity()
        logging.info(
            "Day %s ended with equity %s after %s trading cycles",
            day_i,
            cur_cash,
            trading_cycle_i,
        )

    return cur_cash
