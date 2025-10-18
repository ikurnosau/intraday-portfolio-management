import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import time
import os
import gdown


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


def get_stock_stats(retriever, symbols, stats_start_date, stats_end_date):
    rng = pd.date_range(start=stats_start_date,
                        end=stats_end_date,
                        freq="5h",
                        tz="UTC",          # optional – adds timezone info
                        inclusive="left")  # include start, exclude end

    df = pd.DataFrame({"date": rng})

    from data.processed.data_processing_utils import filter_by_regular_hours
    df = filter_by_regular_hours(df, 'date')
    df['date'] = pd.to_datetime(df['date'])

    symbol_stats = {}
    for i, symbol in enumerate(symbols): 
        if i % 10 == 0:
            print(f"Processing ({i}/{len(symbols)})")

        quotes = []
        for start_date in df['date']:
            end_date = start_date + timedelta(hours=1)
            attempts = 0
            retrieval_result = {}
            while attempts < 2:
                try:
                    retrieval_result = retriever.quotes(symbol, start_date, end_date, limit=1)
                    break
                except Exception as e:
                    attempts += 1
                    if attempts >= 2:
                        print(f"Failed to retrieve quotes for {symbol} between {start_date} and {end_date} after {attempts} attempts: {e}")
                    else:
                        time.sleep(5)
            if symbol in retrieval_result:
                quotes.append(retrieval_result[symbol][0])

        avg_spread = np.mean([(quote.ask_price - quote.bid_price) / quote.ask_price for quote in quotes])

        bars_retrieval_result = retriever.bars(symbol, pd.to_datetime(stats_start_date), pd.to_datetime(stats_end_date))
        if symbol in bars_retrieval_result:
            bars_df = filter_by_regular_hours(bars_retrieval_result[symbol], 'date')
            bars_df['date'] = pd.to_datetime(bars_df['date'])

            # Helper to compute volatility for a single day after skipping first 10 minutes
            def _daily_vol(group: pd.DataFrame, periods: int):
                group_sorted = group.sort_values('date')
                # Skip the first 10 minutes (assumes 1-minute bars)
                group_trimmed = group_sorted.iloc[10:]
                # If there is not enough data after trimming, return NaN
                if len(group_trimmed) <= periods:
                    return np.nan
                return group_trimmed['close'].pct_change(periods=periods).std()

            vol_1m_list, vol_5m_list, vol_15m_list, vol_1h_list = [], [], [], []

            for _, day_group in bars_df.groupby(bars_df['date'].dt.date):
                vol_1m_list.append(_daily_vol(day_group, 1))
                vol_5m_list.append(_daily_vol(day_group, 5))
                vol_15m_list.append(_daily_vol(day_group, 15))
                vol_1h_list.append(_daily_vol(day_group, 60))

            # Average the daily volatilities, ignoring NaNs
            volatility_1m = np.nanmean(vol_1m_list)
            volatility_5m = np.nanmean(vol_5m_list)
            volatility_15m = np.nanmean(vol_15m_list)
            volatility_1h = np.nanmean(vol_1h_list)

            symbol_stats[symbol] = {
                'avg_spread': avg_spread,
                'volatility_1m': volatility_1m,
                'volatility_5m': volatility_5m,
                'volatility_15m': volatility_15m,
                'volatility_1h': volatility_1h,
                'quotes': quotes
            }    

    return symbol_stats
