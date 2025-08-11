from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from datetime import datetime, timedelta, timezone
import pandas as pd
from dotenv import load_dotenv
import os
import pickle 
import gdown
import logging
import numpy as np

from config.constants import Constants

class _NumpyCoreRedirectingUnpickler(pickle.Unpickler):
    """Unpickler that maps obsolete ``numpy._core`` → ``numpy.core``."""

    def find_class(self, module, name):
        # Redirect *any* submodule that starts with the obsolete prefix
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _download_file_from_gdrive(file_id: str, output_dir: str, file_name: str):
    url = f'https://drive.google.com/uc?id={file_id}'
    output_path = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    gdown.download(url, output_path, quiet=False)


def _download_from_gdrive():
    _download_file_from_gdrive(
        file_id= "1oE69lvgomUzIqJHCOJ4N1g6opWuv1coW",
        output_dir='../data/raw/alpaca/bars_with_quotes',
        file_name="1Min_2024-06-01-2025-06-01_AAPL+AMD+BABA+BITU+C+CSCO+DAL+DIA+GLD+GOOG+IJR+MARA+MRVL+MU+NEE+NKE+NVDA+ON+PLTR+PYPL+QLD+QQQ+QQQM+R.pkl"
    )
    # _download_file_from_gdrive(
    #     file_id= "1Vk2sviBbB0srSiq_OsHsnYGVlG-BryBN",
    #     output_dir='../data/raw/alpaca/bars_with_quotes',
    #     file_name="1Min_2016-01-01-2018-01-01_MMM+AXP+AMGN+AMZN+AAPL+BA+CAT+CVX+CSCO+KO+DIS+GS+HD+HON+IBM+JNJ+JPM+MCD+MRK+MSFT+NKE+NVDA+PG+CRM+SHW.pkl"
    # )
    _download_file_from_gdrive(
        file_id= "1fBIwQMGOf-cV5IN-psvWqSV2I3MvPum_",
        output_dir='../modeling/checkpoints',
        file_name="best_model.pth"
    )
    

class AlpacaMarketsRetriever:
    FEED = 'sip'

    def __init__(self, timeframe: TimeFrame=TimeFrame.Minute, download_from_gdrive: bool=False):
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')

        self.timeframe = timeframe

        if not download_from_gdrive:
            self.client = StockHistoricalDataClient(self.api_key, self.api_secret)
        else:
            _download_from_gdrive()

    def build_file_name(self,
                        symbol_or_symbols: str | list[str],
                        start: datetime,
                        end: datetime): 
        return f'{self.timeframe}_{start.date()}-{end.date()}_' \
                + f"{'+'.join(symbol_or_symbols if not isinstance(symbol_or_symbols, str) else [symbol_or_symbols])[:100]}.pkl"
    
    @staticmethod
    def save_data(payload: object, save_dir: str, file_name: str): 
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
        
            with open(os.path.join(save_dir, file_name), 'wb') as output_file: 
                pickle.dump(payload, output_file)

    @staticmethod
    def load_data(save_dir: str, file_name: str) -> object: 
        with open(os.path.join(save_dir, file_name), 'rb') as input_file:
            return _NumpyCoreRedirectingUnpickler(input_file).load()

    def get_all_symbols(self) -> list[str]:
        trading_client = TradingClient(self.api_key, self.api_secret)
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
                        .drop(columns=['symbol', 'trade_count', 'vwap']) \
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

    def _quote_estimation(self, symbol: str, start: datetime, end: datetime) -> dict[str: float]:
        start = pd.to_datetime(start)
        start = datetime.combine(start.date(), Constants.Data.REGULAR_TRADING_HOURS_START) + timedelta(hours=2)
        start = start.replace(tzinfo=timezone.utc)        # re-attach UTC

        rng = pd.date_range(start=start,
                            end=end,
                            freq="30d",
                            tz="UTC",          # optional – adds timezone info
                            inclusive="left")

        quotes = []
        for date in rng:
            start_date = pd.to_datetime(date)
            end_date = start_date + timedelta(hours=1)
            retrieval_result = self.quotes(symbol, start_date, end_date, limit=1)
            if symbol in retrieval_result:
                quotes.append(retrieval_result[symbol][0])

        avg_spread = np.mean([(quote.ask_price - quote.bid_price) for quote in quotes])
        ask_price = np.mean([quote.ask_price for quote in quotes])

        return {
            'ask_price': ask_price,
            'ask_size': int(np.mean([quote.ask_size for quote in quotes])),
            'bid_price': ask_price - avg_spread,
            'bid_size': int(np.mean([quote.bid_size for quote in quotes])),
        }

    def _bars_with_quotes(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_WITH_QUOTES_SAVE_DIR) -> dict[str: pd.DataFrame]:
        bars = self.bars(symbol_or_symbols, start, end, save_dir=Constants.Data.Retrieving.Alpaca.BARS_SAVE_DIR)
        quotes = {symbol: self._quote_estimation(symbol, start, end) for symbol in symbol_or_symbols}

        for symbol, bar_df in bars.items():
            for column_name, value in quotes[symbol].items():
                bar_df[column_name] = value
        
        if save_dir:
            file_name = self.build_file_name(symbol_or_symbols, start, end)
            self.save_data(bars, save_dir, file_name)

        return bars


    def bars_with_quotes(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(2025, 5, 1),
             end: datetime=datetime(2025, 5, 2), 
             save_dir: str=Constants.Data.Retrieving.Alpaca.BARS_WITH_QUOTES_SAVE_DIR) -> dict[str: pd.DataFrame]:
        
        if save_dir:
            file_name = self.build_file_name(symbol_or_symbols, start, end)
            potential_load_path = os.path.join(save_dir, file_name)

            if os.path.exists(potential_load_path): 
                return self.load_data(save_dir, file_name)
            
        return self._bars_with_quotes(symbol_or_symbols, start, end, save_dir)
        
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