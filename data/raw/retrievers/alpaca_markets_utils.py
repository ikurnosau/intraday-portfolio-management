import pandas as pd
from datetime import timedelta, datetime
import numpy as np
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
    # _download_file_from_gdrive(
    #     file_id= "1uPa6Szzs9sN3e3PTvPaOeiSCbtfbQ36q",
    #     output_dir='../data/raw/alpaca/bars_with_quotes',
    #     file_name="1Min_2024-09-01-2025-10-01_AAPL+AMD+BABA+BITU+C+CSCO+DAL+DIA+GLD+GOOG+IJR+MARA+MRVL+MU+NEE+NKE+NVDA+ON+PLTR+PYPL+QLD+QQQ+QQQM+R.pkl"
    # )
    _download_file_from_gdrive(
        file_id= "1-lBUC933Kotv-_wUWlAH9EXILxL3AjT1",
        output_dir='../data/raw/alpaca/bars_with_quotes',
        file_name="1Min_2024-11-01-2026-01-01_AAPL+ACWI+AMD+AMZN+APLD+AVGO+BAC+BITB+BITU+BMY+BOIL+C+CIFR+CLSK+CSCO+DIA+DKNG+ETHA+EWY+FBTC+GBTC+GDX.pkl"
    )