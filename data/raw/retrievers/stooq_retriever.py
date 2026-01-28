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


def _download_folder_from_gdrive(folder_id: str, output_dir: str, folder_name: str):
    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    logging.info(f"Downloading folder {folder_name} from {url} to {output_dir}")
    output_path = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(url=url, output=output_path, quiet=False)


def _download_from_gdrive(folder_name: str):
    if folder_name == 'djia_all':
        _download_folder_from_gdrive(
            folder_id= "1wUeVf7rCWwqjSrv6mC03XreupUnvv3sC",
            output_dir='../data/raw/stooq/bars',
            folder_name=folder_name
        )
    elif folder_name == 'nasdaq_100':
        _download_folder_from_gdrive(
            folder_id= "1kSq0x5WhkLm3T5h6uAKYB0bAizA1Mdyp",
            output_dir='../data/raw/stooq/bars',
            folder_name=folder_name
        )
    else:
        raise ValueError(f"Invalid folder name: {folder_name}")

    
class StooqRetriever:
    ROOT_DIR = '../data/raw/stooq/bars/'

    def __init__(self, download_from_gdrive: bool=False, folder_name: str='djia_all'):
        self.folder_name = folder_name
        
        if download_from_gdrive:
            _download_from_gdrive(folder_name)
    
    def bars(self,
             symbol_or_symbols: str | list[str],
             start: datetime=datetime(1999, 6, 1, tzinfo=Constants.Data.EASTERN_TZ),
             end: datetime=datetime(2019, 1, 1, tzinfo=Constants.Data.EASTERN_TZ)) -> dict[str: pd.DataFrame]:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        asset_dfs = {}
        bars_dir = os.path.join(self.ROOT_DIR, self.folder_name)
        for stock_name in symbol_or_symbols:
            name = f'{stock_name.lower()}_us_d.csv'
            if name in os.listdir(bars_dir):
                df = pd.read_csv(os.path.join(bars_dir, name))
                df.columns = df.columns.str.lower()

                df['date'] = pd.to_datetime(df['date'])
                df['date'] = df['date'].apply(lambda x: x.replace(hour=13, minute=0, second=0, microsecond=0))
                df['date'] = df['date'].dt.tz_localize(Constants.Data.EASTERN_TZ)

                df['ask_price'] = 0
                df['bid_price'] = 0

                asset_df = df.loc[(df['date'] >= start) & (df['date'] <= end)]
                if len(asset_df) > 0:
                    asset_dfs[stock_name] = asset_df
                else: 
                    logging.warning(f"No data found for {stock_name} between {start} and {end}")
            else: 
                logging.warning(f"No data found for {name} in root folder")

        logging.info(f"retrieved {len(asset_dfs)} assets")

        return asset_dfs

    def bars_with_quotes(self, *args, **kwargs) -> dict[str: pd.DataFrame]:
        return self.bars(*args, **kwargs)