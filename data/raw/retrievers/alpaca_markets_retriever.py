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
import pickle 

from config.constants import Constants


class AlpacaMarketsRetriever:
    FEED = 'sip'

    def __init__(self, timeframe: TimeFrame=TimeFrame.Minute):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')

        self.timeframe = timeframe
        self.client = StockHistoricalDataClient(self.api_key, self.api_secret)

    def build_file_name(self,
                        symbol_or_symbols: str | list[str],
                        start: datetime,
                        end: datetime): 
        return f'{self.timeframe}_{start.date()}-{end.date()}_' \
                + f'{'+'.join(symbol_or_symbols if not isinstance(symbol_or_symbols, str) else [symbol_or_symbols])[:100]}.pkl'
    
    @staticmethod
    def save_data(payload: object, save_dir: str, file_name: str): 
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
        
            with open(os.path.join(save_dir, file_name), 'wb') as output_file: 
                pickle.dump(payload, output_file)

    @staticmethod
    def load_data(save_dir: str, file_name: str) -> object: 
        with open(os.path.join(save_dir, file_name), 'rb') as input_file: 
            return pickle.load(input_file)

    def get_all_symbols(self) -> list[str]:
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

    def _bars(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_SAVE_DIR) -> dict[str: pd.DataFrame]:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=self.timeframe,
            start=start,
            end=end,
            feed=self.FEED
        )
        bars = self.client.get_stock_bars(request_params).data
        response = {symbol:
                    pd.DataFrame([data_item.__dict__
                                  for data_item in stock_data]) \
                        .drop(columns=['symbol', 'trade_count']) \
                        .rename(columns={'timestamp': 'date'})
                for symbol, stock_data in bars.items()}
        
        if save_dir:
            file_name = self.build_file_name(symbol_or_symbols, start, end)
            self.save_data(response, save_dir, file_name)

        return response
    
    def bars(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_SAVE_DIR) -> dict[str: pd.DataFrame]:
        
        if save_dir:
            file_name = self.build_file_name(symbol_or_symbols, start, end)
            potential_load_path = os.path.join(save_dir, file_name)

            if os.path.exists(potential_load_path): 
                return self.load_data(save_dir, file_name)
            
        return self._bars(symbol_or_symbols, start, end, save_dir)
        
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