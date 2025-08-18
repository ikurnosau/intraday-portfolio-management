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


class StooqRetriever:
    ROOT_DIR = '../data/raw/stooq/bars/djia'
    
    def bars(self,
             start: datetime=datetime(1999, 6, 1, tzinfo=timezone.utc),
             end: datetime=datetime(2019, 1, 1, tzinfo=timezone.utc)) -> dict[str: pd.DataFrame]:
        asset_dfs = {}
        for name in os.listdir(self.ROOT_DIR):
            df = pd.read_csv(os.path.join(self.ROOT_DIR, name))
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df['date'] = df['date'].apply(lambda x: x.replace(hour=17, minute=0, second=0, microsecond=0))
            if df['date'].min() <= start:
                asset_dfs[name.split('_')[0].upper()] = df[df['date'] >= start][df['date'] <= end]
            else:
                logging.info(f'{name} has no data prior to 1999-06-01')
        return asset_dfs

    def bars_with_quotes(self) -> dict[str: pd.DataFrame]:
        return self.bars()