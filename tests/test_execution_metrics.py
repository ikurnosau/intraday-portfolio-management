import asyncio
from datetime import datetime, timezone
import json
import logging
import time
from types import SimpleNamespace

import pandas as pd
import pytest
from alpaca.trading.enums import OrderStatus

from core_inference.brokerage_proxies.alpaca_brokerage_proxy import (
    AlpacaBrokerageProxy,
)
from core_inference.brokerage_proxies.backtest_brokerage_proxy import (
    BacktestBrokerageProxy,
)
from core_inference.bars_response_handler import BarsResponseHandler
from core_inference.models.brokerage_state import BrokerageState
from core_inference.quotes_response_handler import QuotesResponseHandler
from core_inference.repository import Repository
from core_inference.streaming_validation import (
    _apply_replay_bar,
    _initial_replay_history,
)
from core_inference.trader import Trader


def _repository() -> Repository:
    frame = pd.DataFrame(
        [
            {
                "open": 99.5,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
                "date": pd.Timestamp("2026-08-14 15:16:00", tz="UTC"),
                "bid_price": 99.9,
                "ask_price": 100.1,
                "bid_size": 10,
                "ask_size": 12,
            }
        ]
    )
    return Repository(
        trading_symbols=["TEST"],
        required_history_depth=1,
        bars_and_quotes={"TEST": frame},
    )


def _metric_records(caplog, event: str) -> list[dict]:
    prefix = "execution_metric="
    return [
        json.loads(record.message[len(prefix):])
        for record in caplog.records
        if record.message.startswith(prefix)
        and json.loads(record.message[len(prefix):])["event"] == event
    ]


def test_repository_snapshot_tracks_quote_timestamp_and_age():
    repository = _repository()
    timestamp = pd.Timestamp("2026-08-14 15:16:01", tz="UTC")

    repository.update_quote(
        SimpleNamespace(
            symbol="TEST",
            bid_price=100.0,
            ask_price=100.2,
            bid_size=15,
            ask_size=20,
            timestamp=timestamp,
        )
    )

    snapshot = repository.get_latest_asset_data("TEST")

    assert snapshot["midpoint"] == pytest.approx(100.1)
    assert snapshot["bid_price"] == pytest.approx(100.0)
    assert snapshot["ask_price"] == pytest.approx(100.2)
    assert snapshot["quote_timestamp"] == timestamp
    assert snapshot["quote_age_ms"] >= 0
    assert repository.get_latest_assets_data()["TEST"]["midpoint"] == pytest.approx(
        100.1
    )


def test_streaming_validation_updates_quote_before_replaying_bar():
    repository = _repository()
    quote_timestamp = pd.Timestamp("2026-08-14 15:17:00", tz="UTC")

    _apply_replay_bar(
        repository,
        "TEST",
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1500,
            "date": quote_timestamp,
            "bid_price": 100.4,
            "ask_price": 100.6,
            "bid_size": 25,
            "ask_size": 30,
            "quote_timestamp": quote_timestamp,
        }
    )

    snapshot = repository.get_latest_asset_data("TEST")

    assert snapshot["bid_price"] == pytest.approx(100.4)
    assert snapshot["ask_price"] == pytest.approx(100.6)
    assert snapshot["midpoint"] == pytest.approx(100.5)
    assert snapshot["quote_timestamp"] == quote_timestamp
    assert snapshot["quote_age_ms"] >= 0


def test_streaming_validation_excludes_quote_timestamp_from_feature_history():
    replay_start = pd.Timestamp("2026-08-14 09:30:00", tz="America/New_York")
    history = pd.DataFrame({
        "date": [replay_start - pd.Timedelta(minutes=1), replay_start],
        "close": [100.0, 101.0],
        "quote_timestamp": [
            replay_start - pd.Timedelta(seconds=1),
            replay_start,
        ],
    })

    replay_history = _initial_replay_history(history, replay_start)

    assert replay_history["date"].tolist() == [
        replay_start - pd.Timedelta(minutes=1)
    ]
    assert "quote_timestamp" not in replay_history.columns


def test_backtest_fill_logs_cost_and_effective_price(caplog):
    repository = _repository()
    pre_submit_market = repository.get_latest_asset_data("TEST")
    pre_submit_market.update(
        {
            "bid_price": 100.0,
            "ask_price": 100.2,
            "midpoint": 100.1,
        }
    )
    proxy = BacktestBrokerageProxy(
        repository=repository,
        spread_multiplier=1.5,
    )

    with caplog.at_level(logging.INFO):
        proxy.market_shares_order(
            "TEST",
            10,
            {
                "cycle_id": "cycle-1",
                "pre_submit_market": pre_submit_market,
            },
        )

    metric = _metric_records(caplog, "shadow_fill")[0]
    assert metric["cycle_id"] == "cycle-1"
    assert metric["modeled_transaction_cost"] == pytest.approx(1.5)
    assert metric["reference_midpoint"] == pytest.approx(100.1)
    assert metric["effective_fill_price"] == pytest.approx(100.25)
    assert metric["cash_delta"] == pytest.approx(-1002.5)


def test_shadow_brokers_measure_cycle_to_submit_market_move():
    repository = _repository()
    cycle_start_market = repository.get_latest_asset_data("TEST")
    pre_submit_market = dict(cycle_start_market)
    pre_submit_market.update(
        {
            "bid_price": 100.1,
            "ask_price": 100.3,
            "midpoint": 100.2,
        }
    )
    context = {
        "cycle_id": "cycle-1",
        "cycle_start_market": cycle_start_market,
        "pre_submit_market": pre_submit_market,
    }
    cycle_start_shadow = BacktestBrokerageProxy(
        repository=repository,
        spread_multiplier=1.5,
        name="cycle_start_shadow",
        market_snapshot_key="cycle_start_market",
    )
    pre_submit_shadow = BacktestBrokerageProxy(
        repository=repository,
        spread_multiplier=1.5,
        name="pre_submit_shadow",
        market_snapshot_key="pre_submit_market",
    )

    cycle_start_shadow.market_shares_order("TEST", 10, context)
    pre_submit_shadow.market_shares_order("TEST", 10, context)

    cycle_start_state = cycle_start_shadow.get_named_brokerage_state()[
        "cycle_start_shadow"
    ]
    pre_submit_state = pre_submit_shadow.get_named_brokerage_state()[
        "pre_submit_shadow"
    ]
    assert pre_submit_state.equity - cycle_start_state.equity == pytest.approx(
        -2.0
    )


def test_trader_captures_market_immediately_before_broker_call():
    recorded = {}
    trader = Trader.__new__(Trader)
    trader.repository = SimpleNamespace(
        get_latest_asset_data=lambda symbol: {
            "symbol": symbol,
            "midpoint": 101.0,
        }
    )
    trader.brokerage_proxy = SimpleNamespace(
        market_shares_order=lambda symbol, shares, context: recorded.update(
            {
                "symbol": symbol,
                "shares": shares,
                "context": context,
            }
        )
    )
    context = {"cycle_id": "cycle-1"}

    trader._execute_order("TEST", 10, context)

    assert recorded["context"]["pre_submit_market"]["midpoint"] == 101.0
    assert recorded["context"]["pre_submit_observed_at"] > 0
    assert recorded["context"]["pre_submit_monotonic"] > 0


def test_alpaca_fill_logs_prices_latency_and_slippage(caplog):
    repository = _repository()
    submitted_order = SimpleNamespace(id="order-1", client_order_id="client-1")
    filled_order = SimpleNamespace(
        id="order-1",
        status=OrderStatus.FILLED,
        filled_qty="10",
        filled_avg_price="100.12",
        created_at=None,
        submitted_at=None,
        filled_at=None,
    )
    proxy = AlpacaBrokerageProxy.__new__(AlpacaBrokerageProxy)
    proxy.paper = True
    proxy.repository = repository
    proxy.trading_client = SimpleNamespace(
        submit_order=lambda order_data: submitted_order,
        get_order_by_id=lambda order_id: filled_order,
    )

    with caplog.at_level(logging.INFO):
        proxy.market_shares_order(
            "TEST",
            10,
            {
                "cycle_id": "cycle-1",
                "decision_observed_at": 1.0,
                "decision_market": repository.get_latest_asset_data("TEST"),
            },
        )

    metric = _metric_records(caplog, "live_fill")[0]
    assert metric["cycle_id"] == "cycle-1"
    assert metric["filled_avg_price"] == pytest.approx(100.12)
    assert metric["slippage_vs_submit_midpoint_per_share"] == pytest.approx(0.12)
    assert metric["slippage_vs_submit_touch_per_share"] == pytest.approx(0.02)
    assert metric["total_midpoint_slippage"] == pytest.approx(1.2)
    assert metric["submit_api_latency_ms"] >= 0
    assert metric["submit_to_fill_observed_ms"] >= 0


def test_alpaca_close_all_positions_waits_for_each_fill():
    proxy = AlpacaBrokerageProxy.__new__(AlpacaBrokerageProxy)
    proxy.trading_client = SimpleNamespace(
        close_all_positions=lambda: [
            SimpleNamespace(body=SimpleNamespace(id="order-1")),
            SimpleNamespace(body=SimpleNamespace(id="order-2")),
        ]
    )
    waited_for = []
    proxy._wait_for_fill = waited_for.append

    proxy.close_all_positions()

    assert waited_for == ["order-1", "order-2"]


def test_trading_cycle_does_not_block_quote_callbacks():
    class SlowTrader:
        def perform_trading_cycle(self):
            time.sleep(0.1)

    class QuoteRepository:
        def __init__(self):
            self.quote_updates = 0

        def update_quote(self, data):
            self.quote_updates += 1

    async def run_scenario():
        repository = QuoteRepository()
        bars_handler = BarsResponseHandler(SlowTrader(), repository)
        quotes_handler = QuotesResponseHandler(repository)

        cycle = asyncio.create_task(bars_handler._trigger_trading_cycle())
        await asyncio.sleep(0.01)
        await quotes_handler.handle(SimpleNamespace())

        assert repository.quote_updates == 1
        assert not cycle.done()
        await cycle

    asyncio.run(run_scenario())


def test_immediate_bar_trigger_returns_before_trading_cycle_finishes():
    class SlowTrader:
        def perform_trading_cycle(self):
            time.sleep(0.1)

    class StreamRepository:
        def __init__(self):
            self.quote_updates = 0

        def add_bar(self, data):
            pass

        def get_symbols(self):
            return ["TEST"]

        def update_quote(self, data):
            self.quote_updates += 1

    async def run_scenario():
        repository = StreamRepository()
        bars_handler = BarsResponseHandler(SlowTrader(), repository)
        quotes_handler = QuotesResponseHandler(repository)
        bar = SimpleNamespace(
            symbol="TEST",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
            timestamp=datetime.now(timezone.utc),
        )

        await bars_handler.handle(bar)
        assert bars_handler._cycle_task is not None
        assert not bars_handler._cycle_task.done()

        await quotes_handler.handle(SimpleNamespace())
        assert repository.quote_updates == 1

        await asyncio.gather(*list(bars_handler._background_tasks))

    asyncio.run(run_scenario())


def test_overlapping_trigger_is_coalesced_into_one_follow_up_cycle():
    class CountingTrader:
        def __init__(self):
            self.calls = 0

        def perform_trading_cycle(self):
            self.calls += 1
            time.sleep(0.05)

    async def run_scenario():
        trader = CountingTrader()
        handler = BarsResponseHandler(trader, SimpleNamespace())

        first_cycle = asyncio.create_task(handler._trigger_trading_cycle())
        await asyncio.sleep(0.01)
        await handler._trigger_trading_cycle()
        await first_cycle

        assert trader.calls == 2

    asyncio.run(run_scenario())


def test_reconciliation_separates_level_gap_from_session_pnl(caplog):
    trader = Trader.__new__(Trader)
    trader.session_start_states = {
        "alpaca_paper": BrokerageState(99_500.0, 99_500.0, {}),
        "cycle_start_shadow": BrokerageState(100_000.0, 100_000.0, {}),
        "pre_submit_shadow": BrokerageState(100_000.0, 100_000.0, {}),
    }
    states = {
        "alpaca_paper": BrokerageState(99_490.0, 109_490.0, {"TEST": -100}),
        "cycle_start_shadow": BrokerageState(
            99_995.0,
            109_995.0,
            {"TEST": -100},
        ),
        "pre_submit_shadow": BrokerageState(
            99_993.0,
            109_993.0,
            {"TEST": -100},
        ),
    }

    with caplog.at_level(logging.INFO):
        trader._log_reconciliation("cycle-1", states)

    metric = _metric_records(caplog, "brokerage_reconciliation")[0]
    comparison = metric["comparisons"][0]
    assert comparison["raw_equity_gap_comparison_minus_primary"] == 505.0
    assert comparison["session_pnl_gap_comparison_minus_primary"] == 5.0
    assert comparison["position_differences_comparison_minus_primary"] == {}
    latency_comparison = metric["comparisons"][2]
    assert latency_comparison["primary"] == "cycle_start_shadow"
    assert latency_comparison["comparison"] == "pre_submit_shadow"
    assert (
        latency_comparison["session_pnl_gap_comparison_minus_primary"]
        == -2.0
    )
