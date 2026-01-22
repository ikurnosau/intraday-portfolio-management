import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import threadpoolctl
os.environ["OMP_NUM_THREADS"]  = "1"
os.environ["MKL_NUM_THREADS"]  = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1)

import pandas as pd
import copy
from typing import Callable, Tuple
from datetime import datetime, timedelta
import datetime as datetime_lib
from collections import defaultdict
from joblib import Parallel, delayed

import logging
import numpy as np
import polars as pl
import time
from numpy.lib.stride_tricks import sliding_window_view
from data.processed.data_processing_utils import filter_by_regular_hours
from core_data_prep.validations import Validator
from config.constants import Constants


class DataPreparer:
    def __init__(self,
                 normalizer: Callable,
                 missing_values_handler: Callable,
                 in_seq_len: int,
                 frequency: str,
                 validator: Validator|None=None,
                 date_column: str = 'date'):
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.date_column = date_column
        self.frequency = frequency
        self.validator = validator

    def _generate_raw_features_polars(
        self,
        data: dict[str, pd.DataFrame],
        features: dict[str, Callable],
    ) -> dict[str, pd.DataFrame]:
        lazy_frames: list[pl.LazyFrame] = [
            pl.from_pandas(asset_df).with_columns(pl.lit(asset_name).alias("asset_id")).lazy()
            for asset_name, asset_df in data.items()
        ]
        lf_all = pl.concat(lazy_frames)

        names, exprs = zip(*[
            (name, transform(lf_all).alias(name)) for name, transform in features.items()
        ])
        lf_features = (
            lf_all.group_by("asset_id", maintain_order=True)
            .agg(exprs)
            .explode(names)
        )

        df_all: pd.DataFrame = lf_features.collect().to_pandas()
        return {
            asset_id: grp.drop(columns=["asset_id"]).astype(np.float32).reset_index(drop=True)
            for asset_id, grp in df_all.groupby("asset_id", sort=False)
        }

    def _generate_raw_features(self,
                            data: dict[str, pd.DataFrame],
                            features: dict[str, Callable],
                            ) -> dict[str, pd.DataFrame]:
        return dict(sorted(Parallel(n_jobs=12, backend='threading')(
            delayed(self._raw_features_for_df)(asset_name, asset_df, features) \
                for asset_name, asset_df in data.items()), key=lambda x: x[0])
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

        per_asset_array = { a: df.to_numpy(dtype=np.float32).squeeze() for a, df in sorted(per_asset_dfs_aligned.items())}
        
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

        array_sequential = np.swapaxes(array_sequential, 0, 1)

        if self.validator is not None:
            self.validator.validate_sequential_array(array_sequential)

        return array_sequential

    def _fill_missing_values_polars(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        filled_data = self.missing_values_handler(data)

        if self.validator is not None:
            self.validator.validate_filled_data(filled_data)

        return filled_data

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


        if self.validator is not None:
            self.validator.validate_normalized_features(features_normalized, features_to_normalize)

        return features_normalized

    def transform_data_for_inference(self, 
                                      data: dict[str, pd.DataFrame],
                                      n_timestamps: int,
                                      features: dict[str, Callable],
                                      include_target: bool=False,
                                      include_statistics: bool=False,
                                      statistics: dict[str, Callable] | None = None,
                                      per_asset_target: dict[str, Callable] | None = None,
                                      ) -> np.ndarray | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        data = {asset_name: data[asset_name] for asset_name in sorted(data.keys())}
        if self.validator is not None:
            self.validator.validate_input_data(data)

        data_filled = self._fill_missing_values_polars(data)

        raw_features = self._generate_raw_features_polars(data_filled, features=features)

        if self.validator is not None:
            self.validator.validate_raw_features(raw_features)

        features_normalized = self._normalize_features(
            raw_features,
            features_to_normalize=[col for col in features.keys() if col not in {'is_missing', 'tod_sin', 'tod_cos', 'spread'}]
        )

        X = self._features_to_model_input(features_normalized)[-n_timestamps:]
        if self.validator is not None:
            self.validator.validate_x(X, n_assets=len(data), seq_len=self.in_seq_len)

        output = [X]

        if include_target:
            target_feature = {asset_name: per_asset_target[asset_name](asset_df_filled) \
                for asset_name, asset_df_filled in data_filled.items()}
            y = self._features_to_model_input(target_feature)[-n_timestamps:]

            if self.validator is not None:
                self.validator.validate_target(y)

            output.append(y)

        if include_statistics:
            calculated_statistics: dict[str, np.ndarray] = {}
            for statistics_name, statistics_func in statistics.items():
                statistics_features = self._generate_raw_features(data_filled, features={statistics_name: statistics_func})
                calculated_statistics[statistics_name] = self._features_to_model_input(statistics_features)[-n_timestamps:]
                if self.validator is not None:
                    self.validator.validate_statistics(statistics_name, calculated_statistics[statistics_name])

            if self.validator is not None:
                self.validator.validate_x_target_statistics(X, y, calculated_statistics)
            
            output.append(calculated_statistics)

        return tuple(output) if len(output) > 1 else output[0]

    def get_experiment_data(self, 
                            data: dict[str, pd.DataFrame],
                            start_date: datetime,
                            end_date: datetime,
                            features: dict[str, Callable],
                            statistics: dict[str, Callable],
                            target: Callable,
                            train_set_last_date: datetime | None=None,
                            val_set_last_date: datetime | None=None,
                            n_jobs: int=12,
                            backend: str='loky'
                            ) -> list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
        if self.frequency == '1Min':
            n_timestamps = Constants.Data.TRADING_DAY_LENGTH_MINUTES

            all_slices = self._get_daily_slices(
                data=data, 
                start_date=start_date, 
                end_date=end_date,
                slice_length=n_timestamps + self.in_seq_len + self.normalizer.window + 30,
                verbose=False
            )
            logging.info(f"Found {len(all_slices)} daily slices")

            train_slices, val_slices, test_slices = self._train_val_test_split(all_slices, train_set_last_date, val_set_last_date)
        elif self.frequency == '1Day':
            n_timestamps = - self.normalizer.window
            logging.info(f"Using monolithic slices with {n_timestamps} timestamps")
            train_slices, val_slices, test_slices = self._get_monolith_slices(data, train_set_last_date, val_set_last_date)
            logging.info(f"Found {len(next(iter(train_slices[0].values())))} train slices, {len(next(iter(val_slices[0].values())))} val slices, {len(next(iter(test_slices[0].values())))} test slices")
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")

        per_asset_target = self._train_target_per_asset(
            target,
            train_slices,
            n_timestamps_per_slice=n_timestamps
        )
        logging.info("Trained per-asset targets")

        train_val_test_data = []
        for slices in (train_slices, val_slices, test_slices):
            list_of_x_y_statistics: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = Parallel(
                n_jobs=n_jobs,
                backend=backend
            )(
                delayed(self.transform_data_for_inference)(
                    cur_slice,
                    n_timestamps=n_timestamps,
                    features=features,
                    include_target=True,
                    include_statistics=True,
                    per_asset_target=per_asset_target,
                    statistics=statistics,
                )
                for cur_slice in slices
            )
            x = np.vstack([cur_x for cur_x, _, _ in list_of_x_y_statistics])
            y = np.vstack([cur_y for _, cur_y, _ in list_of_x_y_statistics])

            list_of_statistics = [cur_statistics for _, _, cur_statistics in list_of_x_y_statistics]
            statistics_values = {statistic_name: np.vstack([cur_statistics[statistic_name] for cur_statistics in list_of_statistics]) \
                for statistic_name in list_of_statistics[0].keys()}
            train_val_test_data.append((x, y, statistics_values))

        return train_val_test_data

    def _train_val_test_split(self, 
                              daily_slices: list[dict[str, pd.DataFrame]],
                              train_set_last_date: datetime | None=None,
                              val_set_last_date: datetime | None=None,
                              ) -> tuple[list[dict[str, pd.DataFrame]], list[dict[str, pd.DataFrame]], list[dict[str, pd.DataFrame]]]:
        test_slices, remaining_slices = (
            [slice for slice in daily_slices if next(iter(slice.values()))['date'].max() > val_set_last_date],
            [slice for slice in daily_slices if next(iter(slice.values()))['date'].max() <= val_set_last_date]
        ) if val_set_last_date else ([], daily_slices)

        val_slices, train_slices = (
            [slice for slice in remaining_slices if next(iter(slice.values()))['date'].max() > train_set_last_date],
            [slice for slice in remaining_slices if next(iter(slice.values()))['date'].max() <= train_set_last_date]
        ) if train_set_last_date else ([], remaining_slices)

        return train_slices, val_slices, test_slices

    def _train_target_per_asset(self,
                                target: Callable, 
                                train_slices: list[dict[str, pd.DataFrame]],
                                n_timestamps_per_slice: int=-1,
                                ) -> dict[str, pd.DataFrame]:
        if hasattr(target, "fit"):
            asset_names = set(train_slices[0].keys())
            per_asset_df = {
                asset_name: pd.concat([slice[asset_name].tail(n_timestamps_per_slice) for slice in train_slices], ignore_index=True) \
                    for asset_name in asset_names
            }
            per_asset_target = {
                asset_name: copy.deepcopy(target).fit(asset_df) \
                    for asset_name, asset_df in per_asset_df.items()
            }
        else: 
            per_asset_target = {
                asset_name: copy.deepcopy(target) \
                    for asset_name in train_slices[0].keys()
            }

        return per_asset_target

    def _get_daily_slices(self, 
                        data: dict[str, pd.DataFrame],
                        start_date: datetime, 
                        end_date: datetime,
                        slice_length: int,
                        end_hour: int=Constants.Data.REGULAR_TRADING_HOURS_END.hour,
                        verbose: bool=True
                        ) -> list[dict[str, pd.DataFrame]]:
        current_day = pd.to_datetime(start_date).normalize()
        end_day = pd.to_datetime(end_date).normalize()

        cur_row_i = defaultdict(int)
        slices: list[dict[str, pd.DataFrame]] = []
        while current_day < end_day:
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
                    if self.validator is not None:
                        self.validator.validate_slice_consistency(cur_day_slices, slice_length, slice_end_target)
                    slices.append(cur_day_slices)
                else: 
                    logging.info(f"Skipping day {current_day} because it has less than {len(data)} assets")

        if verbose:
            self._log_last_timestamp_distribution(slices, self.date_column)

        return slices

    def _get_monolith_slices(self,
                             data: dict[str, pd.DataFrame],
                             train_set_last_date: datetime | None=None,
                             val_set_last_date: datetime | None=None,
                             ) -> tuple[list[dict[str, pd.DataFrame]], list[dict[str, pd.DataFrame]], list[dict[str, pd.DataFrame]]]:
        cutoff_compensation = self.in_seq_len + self.normalizer.window
        first_train_timestamp = min(df[self.date_column].min() for df in data.values())
        first_val_timestamp = min(df[(df[self.date_column] > (train_set_last_date - timedelta(days=cutoff_compensation)))][self.date_column].min() for df in data.values())
        first_test_timestamp = min(df[(df[self.date_column] > (val_set_last_date - timedelta(days=cutoff_compensation)))][self.date_column].min() for df in data.values())
        for asset_name, df in list(data.items()):
            for first_timestamp in [first_test_timestamp, first_val_timestamp, first_train_timestamp]:
                if min(df[self.date_column]) > first_timestamp:
                    new_first_row = df.iloc[0].copy()
                    new_first_row[self.date_column] = first_timestamp
                    df = pd.concat([pd.DataFrame([new_first_row]), df], ignore_index=True)
            data[asset_name] = df

        train_data = { asset_name: df[df[self.date_column] <= train_set_last_date] for asset_name, df in data.items()}
        val_data = { asset_name: df[(df[self.date_column] > (train_set_last_date - timedelta(days=cutoff_compensation))) & (df[self.date_column] <= val_set_last_date)] for asset_name, df in data.items()}
        test_data = { asset_name: df[df[self.date_column] > (val_set_last_date - timedelta(days=cutoff_compensation))] for asset_name, df in data.items()}

        return [train_data], [val_data], [test_data]


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
