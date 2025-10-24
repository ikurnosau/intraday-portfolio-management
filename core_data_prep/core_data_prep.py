import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
from typing import Callable, Tuple
from datetime import datetime, timedelta
import datetime as datetime_lib
from collections import defaultdict
from joblib import Parallel, delayed

import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from data.processed.data_processing_utils import filter_by_regular_hours
from config.constants import Constants


class DataPreparer:
    def __init__(self,
                 normalizer: Callable,
                 missing_values_handler: Callable,
                 in_seq_len: int,
                 date_column: str = 'date'):
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.date_column = date_column

    def _generate_raw_features(self, 
                            data: dict[str, pd.DataFrame], 
                            features: dict[str, Callable]
                            ) -> dict[str, pd.DataFrame]:
        return dict(Parallel(n_jobs=os.cpu_count() // 2, backend="loky")(
            delayed(self._raw_features_for_df)(asset_name, asset_df, features) \
                for asset_name, asset_df in data.items())
        )

    def _raw_features_for_df(self, 
                            asset_name: str, 
                            asset_df: pd.DataFrame, 
                            features: dict[str, Callable]
                            ) -> pd.DataFrame:
        feat_df = pd.DataFrame()
        for name, transform in features.items():
            feat_df[name] = transform(asset_df).astype(np.float32)

        return asset_name, feat_df

    def _features_to_model_input(self, per_asset_df: dict[str, pd.DataFrame | pd.Series]) -> np.ndarray:
        min_asset_length = min(len(df) for df in per_asset_df.values())
        per_asset_dfs_aligned = {asset: df.tail(min_asset_length).reset_index(drop=True) for asset, df in per_asset_df.items()}

        per_asset_array = { a: df.to_numpy(dtype=np.float32) for a, df in per_asset_dfs_aligned.items()}
        
        multi_asset_array = np.stack(
            [array for array in per_asset_array.values()], 
            axis=0
        ).astype(np.float32)

        if len(multi_asset_array.shape) == 3:
            X = multi_asset_array
            array_sequential = (
                sliding_window_view(X, window_shape=self.in_seq_len, axis=len(X.shape) - 2)
                .swapaxes(-2, -1)
            )
        else: 
            y_or_statistics = multi_asset_array
            array_sequential = y_or_statistics[..., self.in_seq_len - 1:]

        return np.swapaxes(array_sequential, 0, 1)

    def _fill_missing_values(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        last_timestamp = max(df[self.date_column].max() for df in data.values())

        return {asset_name: self.missing_values_handler(asset_df, last_timestamp, date_column=self.date_column) \
            for asset_name, asset_df in data.items()}

    def _normalize_features(self, 
                            features: dict[str, pd.DataFrame], 
                            features_to_normalize: list[str]
                            ) -> dict[str, pd.DataFrame]:
        features_normalized = {}
        for asset_name, asset_features in features.items():
            asset_feautures_normalized = asset_features.copy()
            asset_feautures_normalized[features_to_normalize] = self.normalizer(
                asset_features[features_to_normalize]
            ).astype(np.float32)
            features_normalized[asset_name] = asset_feautures_normalized

        return features_normalized

    def transform_data_for_inference(self, 
                                      data: dict[str, pd.DataFrame],
                                      n_timestamps: int,
                                      features: dict[str, Callable],
                                      include_target_and_statistics: bool=False,
                                      statistics: dict[str, Callable] | None = None,
                                      target: Callable | None = None
                                      ) -> np.ndarray:
        data_filled = self._fill_missing_values(data)

        raw_features = self._generate_raw_features(data_filled, features=features)

        features_normalized = self._normalize_features(
            raw_features,
            features_to_normalize=[col for col in features.keys() if col not in {'is_missing', 'tod_sin', 'tod_cos'}]
        )

        X = self._features_to_model_input(features_normalized)[-n_timestamps:]

        if not include_target_and_statistics:
            return X
        else:
            target_and_statistics: dict[str, np.ndarray] = {}
            for statistics_name, statistics_func in (statistics | {"target": target}).items():
                statistics_features = self._generate_raw_features(data_filled, features={statistics_name: statistics_func})
                target_and_statistics[statistics_name] = self._features_to_model_input(statistics_features)[-n_timestamps:]

            return X, target_and_statistics['target'], {k: v for k, v in target_and_statistics.items() if k != 'target'} 

    def get_experiment_data(self, 
                            data: dict[str, pd.DataFrame],
                            start_date: datetime,
                            end_date: datetime,
                            features: dict[str, Callable],
                            statistics: dict[str, Callable],
                            ) -> pd.DataFrame:
        daily_slices = self._get_daily_slices(
            data=data, 
            start_date=start_date, 
            end_date=end_date,
            slice_length=Constants.Data.TRADING_DAY_LENGTH_MINUTES + self.in_seq_len + self.normalizer.window + 30
        )

        raw_features = [self._generate_raw_features(slice, features) for slice in daily_slices]

        return raw_features  

    def _get_daily_slices(self, 
                        data: dict[str, pd.DataFrame],
                        start_date: datetime, 
                        end_date: datetime,
                        slice_length: int=(
                            Constants.Data.TRADING_DAY_LENGTH_MINUTES
                            + 60  # config.data_config.in_seq_len
                            + 60  # config.data_config.normalizer.window
                            + 30  # lookback in features 
                        ),
                        end_hour: int=Constants.Data.REGULAR_TRADING_HOURS_END.hour) -> list[dict[str, pd.DataFrame]]:
        current_day = pd.to_datetime(start_date).normalize()
        end_day = pd.to_datetime(end_date).normalize()

        cur_row_i = defaultdict(int)
        slices: list[dict[str, pd.DataFrame]] = []
        while current_day <= end_day:
            slice_end_target = pd.Timestamp(current_day) + pd.Timedelta(hours=end_hour)
            current_day = current_day + pd.DateOffset(days=1)
            
            if slice_end_target.dayofweek < 5:
                cur_day_slices: dict[str, pd.DataFrame] = {}
                for symbol, df in data.items():
                    while cur_row_i[symbol] < len(df) and df[self.date_column][cur_row_i[symbol]] <= slice_end_target:
                        cur_row_i[symbol] += 1

                    slice_end_i = cur_row_i[symbol] # last index before cutoff
                    slice_start_i = max(0, slice_end_i - slice_length)
                    if slice_end_i - slice_start_i == slice_length:
                        cur_day_slices[symbol] = df.iloc[slice_start_i:slice_end_i].reset_index(drop=True)

                if len(cur_day_slices) == len(data):
                    self._validate_slice_consistency(cur_day_slices, slice_length, slice_end_target)
                    slices.append(cur_day_slices)

        self._log_last_timestamp_distribution(slices, self.date_column)
        return slices

    @staticmethod
    def _validate_slice_consistency(cur_day_slices: dict[str, pd.DataFrame], 
                                     slice_length: int, 
                                     slice_end_target: pd.Timestamp) -> None:
        """
        Validate that all dataframes in a slice have consistent lengths.
        
        """
        slice_lengths = {symbol: len(df) for symbol, df in cur_day_slices.items()}
        unique_lengths = set(slice_lengths.values())
        
        # Check that all dataframes have the same length
        assert len(unique_lengths) == 1, \
            f"Slice at {slice_end_target.date()}: Dataframes have different lengths: {slice_lengths}"
        
        # Verify all dataframes have the expected slice_length
        for symbol, length in slice_lengths.items():
            assert length == slice_length, \
                f"Slice at {slice_end_target.date()}, {symbol}: Expected length {slice_length}, got {length}"

    @staticmethod
    def _log_last_timestamp_distribution(slices: list[dict[str, pd.DataFrame]], 
                                         date_column: str = 'date') -> None:
        """
        Log the distribution of last timestamps across all slices and stocks.
        """
        if not slices:
            return
            
        last_timestamp_counts = defaultdict(int)
        for daily_slice in slices:
            for symbol, df in daily_slice.items():
                last_timestamp = df[date_column].iloc[-1]
                last_timestamp_counts[last_timestamp] += 1
        
        logging.info("Last timestamp counts across all slices and stocks:")
        for timestamp in sorted(last_timestamp_counts.keys()):
            count = last_timestamp_counts[timestamp]
            logging.info("  %s: %d occurrences", timestamp, count)
            

class ContinuousForwardFill: 
    def __init__(self, frequency: str):
        if frequency == '1Min':
            self.frequency = 'min'
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    def __call__(self, data: pd.DataFrame, last_timestamp: datetime, date_column: str = 'date') -> pd.DataFrame:
        first_timestamp = data[date_column].min()

        data[date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(date_column)

        full_idx = pd.date_range(start=first_timestamp, end=last_timestamp, freq=self.frequency)
        data_continuous = data.reindex(full_idx)

        data_continuous['close'] = data_continuous['close'].ffill()
        if 'ask_price' in data_continuous.columns:
            data_continuous['ask_price'] = data_continuous['ask_price'].ffill()
        if 'bid_price' in data_continuous.columns:
            data_continuous['bid_price'] = data_continuous['bid_price'].ffill()

        missing = data_continuous['volume'].isna()
        data_continuous.loc[missing, 'open'] = data_continuous.loc[missing, 'close']
        data_continuous.loc[missing, 'high'] = data_continuous.loc[missing, 'close']
        data_continuous.loc[missing, 'low'] = data_continuous.loc[missing, 'close']
        data_continuous.loc[missing, 'volume'] = 0
        data_continuous['is_missing'] = missing.astype(np.float32)

        return data_continuous.reset_index().rename(columns={'index':date_column})
