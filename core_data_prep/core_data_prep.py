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
                 missing_values_handler_polars: Callable,
                 in_seq_len: int,
                 frequency: str,
                 validator: Validator|None=None,
                 date_column: str = 'date',
                 raw_features_backend: str = 'threading',
                 transform_data_for_inference_backend: str = 'loky'):
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.missing_values_handler_polars = missing_values_handler_polars
        self.in_seq_len = in_seq_len
        self.date_column = date_column
        self.frequency = frequency
        self.validator = validator

        self.raw_features_backend = raw_features_backend
        self.transform_data_for_inference_backend = transform_data_for_inference_backend

    def _generate_raw_features_polars(
        self,
        data: dict[str, pd.DataFrame],
        features: dict[str, Callable],
    ) -> dict[str, pd.DataFrame]:
        start_time = time.time()
        lazy_frames: list[pl.LazyFrame] = [
            pl.from_pandas(asset_df).with_columns(pl.lit(asset_name).alias("asset_id")).lazy()
            for asset_name, asset_df in data.items()
        ]
        lf_all = pl.concat(lazy_frames)
        end_time = time.time()
        logging.info(f"Time taken to concatenate lazy frames: {end_time - start_time} seconds")

        start_time = time.time()
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
                            n_jobs: int=os.cpu_count() // 2,
                            batch_size: int=4
                            ) -> dict[str, pd.DataFrame]:
        return dict(sorted(Parallel(n_jobs=n_jobs, backend=self.raw_features_backend, batch_size=batch_size)(
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

    def _fill_missing_values(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        last_timestamp = max(df[self.date_column].max() for df in data.values())
        filled_data = {asset_name: self.missing_values_handler(asset_df, last_timestamp, date_column=self.date_column) \
            for asset_name, asset_df in data.items()}

        if self.validator is not None:
            self.validator.validate_filled_data(filled_data)
        
        return filled_data

    def fill_missing_values_polars(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        filled_data = self.missing_values_handler_polars(data)
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
                                      features_polars: dict[str, Callable],
                                      include_target_and_statistics: bool=False,
                                      statistics: dict[str, Callable] | None = None,
                                      per_asset_target: dict[str, Callable] | None = None,
                                      n_jobs: int=os.cpu_count() // 2
                                      ) -> np.ndarray | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        data = {asset_name: data[asset_name] for asset_name in sorted(data.keys())}
        if self.validator is not None:
            self.validator.validate_input_data(data)

        # start_time = time.time()
        # data_filled = self._fill_missing_values(data)
        # end_time = time.time()
        # logging.info(f"Time taken to fill missing values: {end_time - start_time} seconds")

        start_time = time.time()
        polars_filled = self.fill_missing_values_polars(data)       # new routine
        end_time = time.time()
        logging.info(f"Time taken to fill missing values with Polars: {end_time - start_time} seconds")
        data_filled = polars_filled

        # for asset, df_p in data_filled.items():
        #     df_pl = polars_filled[asset]
        #     pd.DataFrame(df_pl).to_csv(f'df_pl.csv', index=False)
        #     pd.DataFrame(df_p).to_csv(f'df_p.csv', index=False)
        #     num_p  = df_p.select_dtypes(include=[np.number]).to_numpy()
        #     num_pl = df_pl.select_dtypes(include=[np.number]).to_numpy()
        #     assert np.allclose(num_p, num_pl, atol=1e-5, equal_nan=True), f"mismatch for {asset}"

        # raw_features = self._generate_raw_features(data_filled, features=features, n_jobs=n_jobs)
        start_time = time.time()
        raw_features_polars = self._generate_raw_features_polars(data_filled, features=features_polars)
        end_time = time.time()
        logging.info(f"Time taken to generate raw features with Polars: {end_time - start_time} seconds")
        raw_features = raw_features_polars

        # --- Sanity-check parity between Pandas-based and Polars-based pipelines (debug aid) -------

        # for asset_name in raw_features.keys():
        #     pd_arr  = raw_features[asset_name].to_numpy()
        #     pl_arr  = raw_features_polars[asset_name].to_numpy()

        #     assert pd_arr.shape == pl_arr.shape, f"Shape mismatch for {asset_name}: {pd_arr.shape} != {pl_arr.shape}"
        #     assert np.allclose(pd_arr, pl_arr, atol=1e-1, equal_nan=True), f"Feature mismatch for {asset_name}"


        if self.validator is not None:
            self.validator.validate_raw_features(raw_features)

        start_time = time.time()
        features_normalized = self._normalize_features(
            raw_features,
            features_to_normalize=[col for col in features.keys() if col not in {'is_missing', 'tod_sin', 'tod_cos'}]
        )
        end_time = time.time()
        logging.info(f"Time taken to normalize features: {end_time - start_time} seconds")

        X = self._features_to_model_input(features_normalized)[-n_timestamps:]
        if self.validator is not None:
            self.validator.validate_x(X, n_assets=len(data), seq_len=self.in_seq_len)

        if not include_target_and_statistics:
            return X
        else:
            start_time = time.time()
            target_feature = {asset_name: per_asset_target[asset_name](asset_df_filled) \
                for asset_name, asset_df_filled in data_filled.items()}
            end_time = time.time()
            logging.info(f"Time taken to generate target features: {end_time - start_time} seconds")
            y = self._features_to_model_input(target_feature)[-n_timestamps:]
            if self.validator is not None:
             self.validator.validate_target(y)

            calculated_statistics: dict[str, np.ndarray] = {}
            for statistics_name, statistics_func in statistics.items():
                start_time = time.time()
                statistics_features = self._generate_raw_features(data_filled, features={statistics_name: statistics_func}, n_jobs=n_jobs)
                end_time = time.time()
                logging.info(f"Time taken to generate statistics features: {end_time - start_time} seconds")
                calculated_statistics[statistics_name] = self._features_to_model_input(statistics_features)[-n_timestamps:]
                if self.validator is not None:
                    self.validator.validate_statistics(statistics_name, calculated_statistics[statistics_name])
            
            if self.validator is not None:
                self.validator.validate_x_target_statistics(X, y, calculated_statistics)

            return X, y, calculated_statistics

    def get_experiment_data(self, 
                            data: dict[str, pd.DataFrame],
                            start_date: datetime,
                            end_date: datetime,
                            features: dict[str, Callable],
                            features_polars: dict[str, Callable],
                            statistics: dict[str, Callable],
                            target: Callable,
                            train_set_last_date: datetime | None=None,
                            val_set_last_date: datetime | None=None,
                            n_jobs: int=os.cpu_count()
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
            list_of_x_y_statistics = [
                self.transform_data_for_inference(
                    cur_slice,
                    n_timestamps=n_timestamps,
                    features=features,
                    features_polars=features_polars,
                    include_target_and_statistics=True,
                    per_asset_target=per_asset_target,
                    statistics=statistics,
                    n_jobs=1,
                )
                for cur_slice in slices
            ]
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
        first_timestamp = min(df[self.date_column].min() for df in data.values())
        for asset_name, df in list(data.items()):
            if min(df[self.date_column]) > first_timestamp:
                new_first_row = df.iloc[0].copy()
                new_first_row[self.date_column] = first_timestamp
                data[asset_name] = pd.concat([pd.DataFrame([new_first_row]), df], ignore_index=True)

        train_data = { asset_name: df[df[self.date_column] <= train_set_last_date] for asset_name, df in data.items()}
        val_data = { asset_name: df[(df[self.date_column] > train_set_last_date) & (df[self.date_column] <= val_set_last_date)] for asset_name, df in data.items()}
        test_data = { asset_name: df[df[self.date_column] > val_set_last_date] for asset_name, df in data.items()}

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
            

class ContinuousForwardFill: 
    def __init__(self, frequency: str):
        if frequency == '1Min':
            self.frequency = 'min'
        elif frequency == '1Day':
            self.frequency = 'd'
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    def __call__(self, data: pd.DataFrame, last_timestamp: datetime, date_column: str = 'date') -> pd.DataFrame:
        first_timestamp = data[date_column].min()

        data.loc[:, date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(date_column)

        full_idx = pd.date_range(start=first_timestamp, end=last_timestamp, freq=self.frequency)
        data_continuous = data.reindex(full_idx)

        data_continuous['close'] = data_continuous['close'].ffill()
        if 'ask_price' in data_continuous.columns:
            data_continuous['ask_price'] = data_continuous['ask_price'].ffill()
        if 'ask_size' in data_continuous.columns: 
            data_continuous['ask_size'] = data_continuous['ask_size'].ffill()
        if 'bid_price' in data_continuous.columns:
            data_continuous['bid_price'] = data_continuous['bid_price'].ffill()
        if 'bid_size' in data_continuous.columns:
            data_continuous['bid_size'] = data_continuous['bid_size'].ffill()

        missing = data_continuous['volume'].isna()
        data_continuous.loc[missing, 'open'] = data_continuous.loc[missing, 'close']
        data_continuous.loc[missing, 'high'] = data_continuous.loc[missing, 'close']
        data_continuous.loc[missing, 'low'] = data_continuous.loc[missing, 'close']
        data_continuous.loc[missing, 'volume'] = 0
        data_continuous['is_missing'] = missing.astype(np.float32)

        assert not data_continuous.isna().values.any(), "Filling missing values resulted in NaNs!"

        return data_continuous.reset_index().rename(columns={'index':date_column})


class ContinuousForwardFillPolars:
    """
    Lazily forward-fills every asset to a continuous time-grid and produces the
    same columns (including `is_missing`) that ContinuousForwardFill does,
    but in a single Polars query.

    Result for each asset – after `.collect()` – is **bit-for-bit identical**
    to the Pandas implementation.
    """

    FILL_COLS_FFILL = (
        "close", "ask_price", "ask_size", "bid_price", "bid_size"
    )

    def __init__(self, frequency: str):
        if frequency == "1Min":
            self.freq = "1m"
        elif frequency == "1Day":
            self.freq = "1d"
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

    # ------------------------------------------------------------------
    def __call__(
        self, data: dict[str, pd.DataFrame], date_column: str = "date"
    ) -> dict[str, pd.DataFrame]:
        # --- gather column names ahead of lazy context -----------------
        available_cols: set[str] = set()
        for df in data.values():
            available_cols.update(df.columns)

        # --- 1. concat original frames with an asset_id ----------------
        # Ensure consistent dtypes (e.g., `volume` must be Float64 across all frames)
        processed_items = []
        for asset, df in data.items():
            if 'volume' in df.columns and df['volume'].dtype != np.float64:
                df = df.copy()
                df['volume'] = df['volume'].astype(np.float64)
            processed_items.append((asset, df))

        lazy_frames = [
            pl.from_pandas(df)
              .with_columns(pl.lit(asset).alias("asset_id"))
              .lazy()
            for asset, df in processed_items
        ]
        lf_all = pl.concat(lazy_frames)

        # --- 2. find global last_timestamp (same as Pandas path) -------
        last_ts_expr = max(df[date_column].max() for df in data.values())

        # --- 3. build a “calendar” per asset_id ------------------------
        calendars = []
        for asset in data:
            tz_str = (
                Constants.Data.EASTERN_TZ
                if isinstance(Constants.Data.EASTERN_TZ, str)
                else str(Constants.Data.EASTERN_TZ)
            )
            dates = pl.datetime_range(
                start=data[asset][date_column].min(),
                end=last_ts_expr,
                interval=self.freq,
                eager=True,
                time_unit="ns",
                time_zone=tz_str,
            )
            cal_df = (
                pl.DataFrame({
                    date_column: dates,
                    "asset_id": [asset] * len(dates),
                }).lazy()
            )
            calendars.append(cal_df)
        lf_calendar = pl.concat(calendars)

        # --- 4. left-join calendar with original -----------------------
        lf_joined = lf_calendar.join(
            lf_all,
            on=[date_column, "asset_id"],
            how="left",
        ).sort(["asset_id", date_column])

        # --- 5. forward-fill and other null-handling -------------------
        # Perform forward-fill within a window over each asset without aggregating,
        # so that the original `date` column is retained (avoids ColumnNotFoundError).
        forward_fill_cols = [
            pl.col(col).forward_fill().over("asset_id").alias(col)
            for col in self.FILL_COLS_FFILL if col in available_cols
        ]

        lf_filled = (
            lf_joined
            .with_columns(forward_fill_cols)  # ensure `close` is forward-filled first
            .with_columns([
                (pl.col("volume").is_null()).cast(pl.Float32).alias("is_missing"),
                pl.col("open").fill_null(pl.col("close")),
                pl.col("high").fill_null(pl.col("close")),
                pl.col("low").fill_null(pl.col("close")),
                pl.col("volume").fill_null(0),
            ])
        )

        # --- 6. collect & partition back to dict -----------------------
        df_all = lf_filled.collect().to_pandas()
        return {
            asset: grp.drop(columns=["asset_id"]).reset_index(drop=True)
            for asset, grp in df_all.groupby("asset_id", sort=False)
        }