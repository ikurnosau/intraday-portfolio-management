import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import logging
import torch

from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from config.settings import get_settings
from core_data_prep.core_data_prep import DataPreparer
from core_inference.bars_response_handler import BarsResponseHandler
from core_inference.quotes_response_handler import QuotesResponseHandler
from core_inference.trader import Trader
from core_inference.brokerage_proxies.alpaca_brokerage_proxy import AlpacaBrokerageProxy
from core_inference.brokerage_proxies.backtest_brokerage_proxy import BacktestBrokerageProxy
from core_inference.brokerage_proxies.aggregated_brokerage_proxy import AggregatedBrokerageProxy
from core_inference.repository import Repository
from observability.wandb_integration import load_production_model

settings = get_settings()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
production_model = load_production_model(settings=settings, device=device)
config = production_model.package.config
allocator = production_model.package.allocator

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for the log messages
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)
logging.info(
    "Loaded production model artifact: %s",
    production_model.resolved_artifact_name,
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
    retriever=config.data_config.retriever,
)

alpaca_proxy = AlpacaBrokerageProxy(paper=True, settings=settings)
backtest_proxy = BacktestBrokerageProxy(repository, config.rl_config.spread_multiplier)
aggregated_proxy = AggregatedBrokerageProxy([alpaca_proxy, backtest_proxy])

trader = Trader(
    order_size_notional=10000.,
    data_preparer=data_preparer,
    features=config.data_config.features_polars,
    statistics={
        'spread': config.data_config.statistics['spread'],
        'volatility': config.data_config.statistics['volatility'],
    },
    brokerage_proxy=aggregated_proxy,
    repository=repository,
    portfolio_allocator=allocator,
    settings=settings,
)

quotes_response_handler = QuotesResponseHandler(repository)
bars_response_handler = BarsResponseHandler(trader, repository)

async def bars_handler(data):
    await bars_response_handler.handle(data)

async def quotes_handler(data):
    await quotes_response_handler.handle(data)

wss_client = StockDataStream(
    settings.alpaca.paper_api_key,
    settings.alpaca.paper_api_secret,
    feed=DataFeed.SIP
)

wss_client.subscribe_bars(bars_handler, *config.data_config.symbol_or_symbols)
wss_client.subscribe_quotes(quotes_handler, *config.data_config.symbol_or_symbols)
wss_client.run()