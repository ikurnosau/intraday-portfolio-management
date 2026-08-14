import asyncio
import logging

from core_inference.trader import Trader
from core_inference.repository import Repository
from data.processed.data_processing_utils import convert_time_to_eastern 


class BarsResponseHandler:
    DEBOUNCE_DELAY = 1.
    MIN_DELAY_BETWEEN_TRADES = 30

    def __init__(self, trader: Trader, repository: Repository):
        self.trader = trader
        self.repository = repository

        self.debounce_timer = None
        self._updated_symbols: set[str] = set()
        self._cycle_lock = asyncio.Lock()
        self._cycle_pending = False
        self._cycle_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()

    async def handle(self, data):
        if self.debounce_timer:
            self.debounce_timer.cancel()  # cancel previously scheduled check for last event

            try:
                await self.debounce_timer
            except asyncio.CancelledError:
                pass

        self.process_data(data)

        # If all symbols updated since last cycle, trigger immediately.
        if self._updated_symbols.issuperset(self.repository.get_symbols()):
            logging.info("All symbols updated since last cycle, triggering trading cycle immediately.")
            self._schedule_trading_cycle()
        else:
            # Schedule a fresh debounce callback (new "last event" timer)
            self.debounce_timer = asyncio.create_task(self.identify_last_event())

    def process_data(self, data):
        self.repository.add_bar({
            "symbol": data.symbol,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume,
            "date": convert_time_to_eastern(data.timestamp)
        })
        self._updated_symbols.add(data.symbol)

    async def identify_last_event(self):
        try:
            await asyncio.sleep(self.DEBOUNCE_DELAY)
            self._schedule_trading_cycle()
        except asyncio.CancelledError:
            pass

    def _schedule_trading_cycle(self) -> None:
        self.debounce_timer = None
        self._updated_symbols.clear()
        if self._cycle_task is not None and not self._cycle_task.done():
            self._cycle_pending = True
            logging.info(
                "Trading cycle already running; queued one follow-up cycle."
            )
            return

        task = asyncio.create_task(self._trigger_trading_cycle())
        self._cycle_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._on_trading_cycle_done)

    def _on_trading_cycle_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if self._cycle_task is task:
            self._cycle_task = None
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logging.exception("Background trading cycle failed")

    async def _trigger_trading_cycle(self):
        self.debounce_timer = None
        if self._cycle_lock.locked():
            self._cycle_pending = True
            logging.info(
                "Trading cycle already running; queued one follow-up cycle."
            )
            return

        async with self._cycle_lock:
            while True:
                self._cycle_pending = False
                await asyncio.to_thread(self.trader.perform_trading_cycle)
                if not self._cycle_pending:
                    break
