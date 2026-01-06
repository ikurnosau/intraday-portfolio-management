import os
import logging
from dotenv import load_dotenv

from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from config.experiments.cur_experiment import config
from core_data_prep.core_data_prep import DataPreparer
from core_inference.bars_response_handler import BarsResponseHandler
from core_inference.quotes_response_handler import QuotesResponseHandler
from core_inference.trader import Trader
from core_inference.brokerage_proxies.alpaca_brokerage_proxy import AlpacaBrokerageProxy
from core_inference.brokerage_proxies.backtest_brokerage_proxy import BacktestBrokerageProxy
from core_inference.repository import Repository


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for the log messages
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)


data_preparer = DataPreparer(
    normalizer=config.data_config.normalizer,
    missing_values_handler=config.data_config.missing_values_handler_polars,
    in_seq_len=config.data_config.in_seq_len,
    frequency=str(config.data_config.frequency),
    validator=config.data_config.validator
)
repository = Repository(
    trading_symbols=config.data_config.symbol_or_symbols,
    required_history_depth=config.data_config.in_seq_len + config.data_config.normalizer.get_window() + 30,
)
brokerage_proxy = AlpacaBrokerageProxy()
backtest_proxy = BacktestBrokerageProxy(repository, config.rl_config.spread_multiplier)
trader = Trader(brokerage_proxy)
quotes_response_handler = QuotesResponseHandler(repository)
bars_response_handler = BarsResponseHandler(trader, repository)

async def bars_handler(data):
    await bars_response_handler.handle(data)

async def quotes_handler(data):
    await quotes_response_handler.handle(data)

wss_client = StockDataStream(
    os.getenv('API_KEY'),
    os.getenv('API_SECRET'),
    feed=DataFeed.SIP
)

wss_client.subscribe_bars(bars_handler, *config.data_config.symbol_or_symbols)
wss_client.subscribe_quotes(quotes_handler, *config.data_config.symbol_or_symbols)
wss_client.run()