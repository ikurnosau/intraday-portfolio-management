from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os


class AlpacaMarketsRetriever:
    FEED = 'sip'

    def __init__(self, timeframe=TimeFrame.Minute):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')

        self.timeframe = timeframe
        self.client = StockHistoricalDataClient(self.api_key, self.api_secret)

    def get_all_symbols(self):
        trading_client = TradingClient(self.api_key, self.api_key)
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)

        assets = trading_client.get_all_assets(search_params)
        assets = [asset for asset in assets if \
                  asset.status == AssetStatus.ACTIVE and
                  asset.easy_to_borrow and
                  asset.fractionable and
                  not asset.min_order_size and
                  asset.shortable and
                  asset.tradable]
        return [asset.symbol for asset in assets]

    def bars(self,
             symbol_or_symbols,
             start=datetime(2025, 5, 1),
             end=datetime(2025, 5, 2), ):
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=self.timeframe,
            start=start,
            end=end,
            feed=self.FEED
        )
        bars = self.client.get_stock_bars(request_params).data
        return {symbol:
                    pd.DataFrame([data_item.__dict__
                                  for data_item in stock_data]) \
                        .drop(columns=['symbol', 'trade_count']) \
                        .rename(columns={'timestamp': 'date'})
                for symbol, stock_data in bars.items()}

    def todays_bars(self, symbol_or_symbols, limit=100):
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=self.timeframe,
            feed=self.FEED
        )
        bars = self.client.get_stock_bars(request_params).data
        return {symbol:
                    pd.DataFrame([data_item.__dict__
                                  for data_item in stock_data]).head(limit)
                    .drop(columns=['symbol', 'trade_count']) \
                        .rename(columns={'timestamp': 'date'})
                for symbol, stock_data in bars.items()}

    def quotes(self,
               symbol_or_symbols,
               start=datetime(2025, 5, 1),
               end=datetime(2025, 5, 2),
               limit=None
               ):
        request_params = StockQuotesRequest(
            symbol_or_symbols=symbol_or_symbols,
            start=start,
            end=end,
            feed=self.FEED,
            limit=limit
        )
        quotes = self.client.get_stock_quotes(request_params).data
        return quotes

    def latest_quote(self, symbol_or_symbols):
        request_params = StockLatestQuoteRequest(
            symbol_or_symbols=symbol_or_symbols,
            feed=self.FEED
        )
        quotes = self.client.get_stock_latest_quote(request_params)

        return quotes

    def latest_spread(self, symbol_or_symbols):
        quotes = self.latest_quote(symbol_or_symbols)
        return {symbol: (quote.ask_price - quote.bid_price) / quote.bid_price
                for symbol, quote in quotes.items()}