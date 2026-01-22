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
            await self._trigger_trading_cycle()
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
            await self._trigger_trading_cycle()
        except asyncio.CancelledError:
            pass

    async def _trigger_trading_cycle(self):
        self.debounce_timer = None
        self._updated_symbols.clear()
        self.trader.perform_trading_cycle()
