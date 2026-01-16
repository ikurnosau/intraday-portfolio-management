import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import os
import gdown
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging
import math
import asyncio

from data.raw.retrievers.alpaca_markets_retriever import AlpacaMarketsRetriever
from config.constants import *



def _download_file_from_gdrive(file_id: str, output_dir: str, file_name: str):
    url = f'https://drive.google.com/uc?id={file_id}'
    output_path = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    gdown.download(url, output_path, quiet=False)


def _download_from_gdrive():
    # _download_file_from_gdrive(
    #     file_id= "1I-3Jy4nFioOUEbILtLbtkY4fMP5PcOC3",
    #     output_dir='../data/raw/alpaca/bars_with_quotes',
    #     file_name="1Hour_2016-01-01-2025-01-01_MMM+AXP+AMGN+AMZN+AAPL+BA+CAT+CVX+CSCO+KO+DIS+GS+HD+HON+IBM+JNJ+JPM+MCD+MRK+MSFT+NKE+NVDA+PG+CRM+SHW.pkl"
    # )
    # _download_file_from_gdrive(
    #     file_id= "1oE69lvgomUzIqJHCOJ4N1g6opWuv1coW",
    #     output_dir='../data/raw/alpaca/bars_with_quotes',
    #     file_name="1Min_2024-06-01-2025-06-01_AAPL+AMD+BABA+BITU+C+CSCO+DAL+DIA+GLD+GOOG+IJR+MARA+MRVL+MU+NEE+NKE+NVDA+ON+PLTR+PYPL+QLD+QQQ+QQQM+R.pkl"
    # )
    # _download_file_from_gdrive(
    #     file_id= "1fBIwQMGOf-cV5IN-psvWqSV2I3MvPum_",
    #     output_dir='../modeling/checkpoints',
    #     file_name="best_model.pth"
    # )
    _download_file_from_gdrive(
        file_id= "1uPa6Szzs9sN3e3PTvPaOeiSCbtfbQ36q",
        output_dir='../data/raw/alpaca/bars_with_quotes',
        file_name="1Min_2024-09-01-2025-10-01_AAPL+AMD+BABA+BITU+C+CSCO+DAL+DIA+GLD+GOOG+IJR+MARA+MRVL+MU+NEE+NKE+NVDA+ON+PLTR+PYPL+QLD+QQQ+QQQM+R.pkl"
    )


def get_quotes_snapshot(retriever: AlpacaMarketsRetriever, symbols: list[str], date: datetime, verbose: bool=False): 
    left_symbols = set(symbols)
    seconds_lookback = 2
    quotes = {}
    while len(left_symbols) > 0:
        if seconds_lookback > 1000: 
            quotes |= {symbol: None for symbol in left_symbols}
            break

        if verbose:
            print(f"Getting quotes for {len(left_symbols)} symbols")

        start_date = date - timedelta(seconds=seconds_lookback)
        result = retriever.quotes(
            left_symbols, 
            start = start_date, 
            end = date, 
            limit=10000
        )

        cur_quotes = {symbol: result[symbol][-1] for symbol in result}
        quotes |= cur_quotes
        left_symbols = left_symbols - set(cur_quotes.keys())

        total_records_in_result = sum(len(value) for value in result.values())
        if total_records_in_result < 10000:
            seconds_lookback *= 4

    return quotes


def _daily_MoAD(asset_df: pd.DataFrame, periods: int):
    if len(asset_df) <= periods:
        return np.nan
    return asset_df['close'].diff(periods=periods).abs().dropna().median()

def _daily_volume(asset_df: pd.DataFrame):
    return asset_df['volume'].sum() / len(asset_df) * 390

def _slippage(asset_df, usd_order_size=30000, Y=1.0):
    current_price = asset_df['close'].iloc[-1]
    Q = usd_order_size / current_price

    daily_volume = _daily_volume(asset_df)
    
    log_returns = np.log(asset_df['close'] / asset_df['close'].shift(1))
    sigma_1m = log_returns.std()
    sigma_daily = sigma_1m * np.sqrt(390)

    impact_pct = Y * sigma_daily * np.sqrt(Q / daily_volume)
    slippage_usd = impact_pct * current_price
    
    return slippage_usd

def _daily_stats_call(symbols, date: datetime, quotes_timestamps: list[datetime.time]):
    retriever = AlpacaMarketsRetriever()

    quotes_snapshots = [
        get_quotes_snapshot(retriever, symbols, pd.Timestamp.combine(date, time_part).tz_localize(Constants.Data.EASTERN_TZ).to_pydatetime())
            for time_part in quotes_timestamps
    ]
    spreads  = {
        symbol: [quotes_snapshot[symbol].ask_price - quotes_snapshot[symbol].bid_price 
            for quotes_snapshot in quotes_snapshots if quotes_snapshot[symbol] is not None]
                for symbol in symbols
    }
    median_spreads = {
        symbol: np.median(symbol_spreads)
            for symbol, symbol_spreads in spreads.items()
                if len(symbol_spreads) > 0
    }

    if len(median_spreads.keys()) == 0:
        1 / 0

    bars = retriever.bars(
        median_spreads.keys(), 
        start = pd.Timestamp.combine(date, pd.to_datetime("9:30:00").time()).tz_localize(Constants.Data.EASTERN_TZ).to_pydatetime(), 
        end = pd.Timestamp.combine(date, pd.to_datetime("16:00:00").time()).tz_localize(Constants.Data.EASTERN_TZ).to_pydatetime(), 
        save_dir=None
    )

    symbol_stats = {}
    skipped_symbols = []
    for symbol in bars: 
        cur_symbol_stats = {
            'spread': median_spreads[symbol],
            'slippage': _slippage(bars[symbol], 30000, 1.0),
            'moad_1m': _daily_MoAD(bars[symbol], 1),
            'moad_5m': _daily_MoAD(bars[symbol], 5),
            'moad_15m': _daily_MoAD(bars[symbol], 15),
            'moad_1h': _daily_MoAD(bars[symbol], 60),
            'daily_volume': _daily_volume(bars[symbol]),
            'price': bars[symbol]['close'].iloc[-1]
        }    

        if all(math.isfinite(value) for value in cur_symbol_stats.values()):
            cur_symbol_stats['E_1m'] = cur_symbol_stats['moad_1m'] / (cur_symbol_stats['spread'] + cur_symbol_stats['slippage'] + 1e-6)
            cur_symbol_stats['E_5m'] = cur_symbol_stats['moad_5m'] / (cur_symbol_stats['spread'] + cur_symbol_stats['slippage'] + 1e-6)
            cur_symbol_stats['E_15m'] = cur_symbol_stats['moad_15m'] / (cur_symbol_stats['spread'] + cur_symbol_stats['slippage'] + 1e-6)
            cur_symbol_stats['E_1h'] = cur_symbol_stats['moad_1h'] / (cur_symbol_stats['spread'] + cur_symbol_stats['slippage'] + 1e-6)

            symbol_stats[symbol] = cur_symbol_stats
        else: 
            skipped_symbols.append(symbol)

    return symbol_stats, skipped_symbols

async def get_daily_stats(symbols: list[str], 
                          date: datetime, 
                          symbols_per_call: int=100, 
                          semaphore: asyncio.Semaphore=asyncio.Semaphore(20),
                          quotes_timestamps: list[datetime.time] = [
                              pd.to_datetime("10:00:00").time(),
                              pd.to_datetime("11:30:00").time(),
                              pd.to_datetime("13:30:00").time(),
                              pd.to_datetime("15:00:00").time(),
                              pd.to_datetime("15:45:00").time()
                          ]
    ):
    async def run(symbols):
        async with semaphore:
            return await asyncio.to_thread(_daily_stats_call, symbols, date, quotes_timestamps)

    tasks = [run(symbols[i:min(i + symbols_per_call, len(symbols))]) for i in range(0, len(symbols), symbols_per_call)]
    results = await asyncio.gather(*tasks)

    daily_stats = {}
    skipped_symbols = []
    for cur_daily_stats, cur_skipped_symbols in results:
        daily_stats |= cur_daily_stats
        skipped_symbols += cur_skipped_symbols

    return daily_stats, skipped_symbols


async def select_portfolio(
    symbols: list[str], 
    start_date: datetime, 
    end_date: datetime, 
    portfolio_size: int = 100,
    criteria: str = 'E_1m',
    max_retries: int = 3
):
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=start_date, end=end_date)
    all_b_days = pd.bdate_range(start=start_date, end=end_date)
    business_days = all_b_days[~all_b_days.isin(holidays)]

    performance_matrix = {symbol: [] for symbol in symbols}
    
    logging.info(f"Starting performance sweep for {len(business_days)} days...")

    for day in business_days:
        logging.info(f"Processing day {day}")
        daily_data = {}
        
        for attempt in range(max_retries):
            try:
                daily_data, skipped_symbols = await get_daily_stats(symbols, day.to_pydatetime())
                break 
                
            except (Exception) as e:
                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                logging.warning(f"Error on {day.date()} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"Failed to retrieve data for {day.date()} after {max_retries} attempts. Skipping.")

        for symbol in symbols:
            val = daily_data.get(symbol, {}).get(criteria, 0)
            performance_matrix[symbol].append(val if np.isfinite(val) else 0)

    final_scores = []
    for symbol, scores in performance_matrix.items():
        q25_value = np.percentile(scores, 25)
        final_scores.append((symbol, q25_value))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    return final_scores[:portfolio_size]