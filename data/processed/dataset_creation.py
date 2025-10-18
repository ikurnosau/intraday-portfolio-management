import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
from typing import Callable, Tuple
from datetime import datetime, timedelta
import datetime as datetime_lib

import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from data.processed.data_processing_utils import filter_by_regular_hours
from config.constants import Constants


class DatasetCreator: 
    def __init__(self, 
                 features: dict[str, Callable],
                 target: Callable,
                 normalizer: Callable,
                 missing_values_handler: Callable,
                 in_seq_len: int,
                 train_set_last_date: datetime | None,
                 val_set_last_date: datetime,
                 multi_asset_prediction: bool,
                 cutoff_time: datetime_lib.time):
        self.features = features
        self.target = target
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.train_last_date = train_set_last_date
        self.val_last_date = val_set_last_date
        self.multi_asset_prediction = multi_asset_prediction
        self.cutoff_time = cutoff_time

    def create_dataset_numpy(self, 
                             data: dict[str, pd.DataFrame],
                             date_column: str = 'date') -> Tuple[
                                 np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                  np.ndarray,
                                  np.ndarray,
                             ]:
        """
        High-level orchestrator that transforms raw per-asset OHLCV data into
        numpy arrays suitable for model consumption.
        """

        processed_assets: dict[str, pd.DataFrame] = {}
        for asset_name, asset_df in data.items():
            logging.info("Processing %s …", asset_name)
            features_df = self._process_single_asset(asset_df, date_column=date_column)
            processed_assets[asset_name] = features_df

        processed_assets = self._filter_and_align_assets(processed_assets, date_column=date_column)

        # Perform the 3-way temporal split
        train_dict, val_dict, test_dict = self._train_val_test_split(processed_assets)

        # Convert each split into the final numpy representation
        X_train, y_train, next_return_train, spread_train, volatility_train = self._to_numpy_and_stack(train_dict)
        X_val, y_val, next_return_val, spread_val, volatility_val = self._to_numpy_and_stack(val_dict)
        X_test, y_test, next_return_test, spread_test, volatility_test = self._to_numpy_and_stack(test_dict)

        return (
            X_train,
            y_train,
            next_return_train,
            spread_train,
            volatility_train,
            X_val,
            y_val,
            next_return_val,
            spread_val,
            volatility_val,
            X_test,
            y_test,
            next_return_test,
            spread_test,
            volatility_test,
        )

    def _process_single_asset(self, asset_df: pd.DataFrame, *, date_column: str = 'date') -> pd.DataFrame:
        asset_df = filter_by_regular_hours(asset_df, datetime_column=date_column)

        asset_df = self.missing_values_handler(asset_df)

        feat_df = pd.DataFrame()
        feat_df[date_column] = pd.to_datetime(asset_df[date_column])

        for name, transform in self.features.items():
            feat_df[name] = transform(asset_df).astype(np.float32)

        feature_cols = list(self.features.keys())

        if hasattr(self.normalizer, "fit") and callable(getattr(self.normalizer, "fit")):
            training_mask = feat_df[date_column] <= self.train_last_date
            self.normalizer.fit(feat_df.loc[training_mask, feature_cols])
            feat_df.loc[:, feature_cols] = self.normalizer.transform(
                feat_df[feature_cols]
            ).astype(np.float32)
        else:
            feat_df.loc[:, feature_cols] = self.normalizer(feat_df[feature_cols]).astype(
                np.float32
            )

        if hasattr(self.target, "fit") and callable(getattr(self.target, "fit")):
            training_slice = asset_df[pd.to_datetime(asset_df[date_column]) <= self.train_last_date]
            self.target.fit(training_slice)

        feat_df['target'] = self.target(asset_df)

        base_feature = getattr(self.target, 'base_feature', 'close')
        feat_df['next_return'] = asset_df[base_feature].pct_change().shift(-1).astype(np.float32)
        # Volatility based on returns over the previous 10 records
        rolling_returns = asset_df[base_feature].pct_change().astype(np.float32)
        feat_df['volatility'] = rolling_returns.rolling(window=10).std().fillna(0.0).astype(np.float32)
        feat_df = feat_df.dropna(subset=["next_return"]).reset_index(drop=True)

        if {'ask_price', 'bid_price'}.issubset(asset_df.columns):
            feat_df['spread'] = (
                (asset_df['ask_price'] - asset_df['bid_price']) / asset_df['ask_price']
            )
        else:
            logging.warning("'ask_price' or 'bid_price' column missing; filling spread with 0.")
            feat_df['spread'] = 0.0
        logging.info("Spread has %d NaNs", int(feat_df['spread'].isna().sum()))
        feat_df['spread'] = feat_df['spread'].fillna(0.0).astype(np.float32)

        if self.cutoff_time is not None:
            feat_df = feat_df[feat_df['date'].dt.time >= self.cutoff_time]

        if hasattr(self.target, "horizon"):
            new_end_time = (datetime.combine(datetime.today(), Constants.Data.REGULAR_TRADING_HOURS_END) - timedelta(minutes=self.target.horizon)).time()
            feat_df = feat_df[feat_df['date'].dt.time <= new_end_time]

        num_null_rows = int(feat_df.isnull().any(axis=1).sum())
        if num_null_rows:
            logging.info("Imputing %d NaN rows with 0.5 sentinel value", num_null_rows)
        feat_df[feature_cols] = feat_df[feature_cols].fillna(0.5)
        assert feat_df.isna().sum().sum() == 0, "There are still NaNs in the dataset"

        return feat_df.reset_index(drop=True)

    def _filter_and_align_assets(self, processed_assets: dict[str, pd.DataFrame], date_column: str = 'date') -> dict[str, pd.DataFrame]:
        # 1) Drop assets that are shorter than 99% of the longest
        lengths = {a: len(df) for a, df in processed_assets.items()}
        max_len = max(lengths.values()) if lengths else 0
        threshold = int(0.99 * max_len)

        filtered_assets: dict[str, pd.DataFrame] = {
            a: df for a, df in processed_assets.items() if len(df) >= threshold
        }
        dropped_by_length = [a for a, df in processed_assets.items() if len(df) < threshold]

        # 2) Intersect dates across remaining assets and trim each to the intersection
        common_dates: set[pd.Timestamp] | None = None
        for df in filtered_assets.values():
            dates_set = set(pd.to_datetime(df[date_column]))
            common_dates = dates_set if common_dates is None else (common_dates & dates_set)

        aligned_assets: dict[str, pd.DataFrame] = {}
        for a, df in filtered_assets.items():
            aligned_df = df[df[date_column].isin(common_dates)].sort_values(by=date_column).reset_index(drop=True)
            aligned_assets[a] = aligned_df

        # Sanity: ensure equal lengths and alignment
        final_lengths = {a: len(df) for a, df in aligned_assets.items()}
        assert len(set(final_lengths.values())) == 1, "Aligned assets must have identical lengths"

        aligned_rows = next(iter(final_lengths.values())) if final_lengths else 0
        logging.info(
            "Finished feature generation. Dropped %d assets by length threshold. Kept %d assets with %d aligned rows each. Max features len prior to alignment: %d",
            len(dropped_by_length),
            len(aligned_assets),
            aligned_rows,
            max_len,
        )

        return aligned_assets

    def _train_val_test_split(self, full_dict: dict[str, pd.DataFrame]):
        train_dict: dict[str, pd.DataFrame] = {}
        val_dict: dict[str, pd.DataFrame] = {}
        test_dict: dict[str, pd.DataFrame] = {}

        extra_rows = max(self.in_seq_len - 1, 0)

        for asset, df in full_dict.items():
            # ---------------------------
            # Train slice (≤ train_last_date)
            # ---------------------------
            train_mask = df['date'] <= self.train_last_date
            train_dict[asset] = df[train_mask]

            # ---------------------------
            # Validation slice (train_last_date < d ≤ val_last_date)
            # + context rows
            # ---------------------------
            val_mask = (df['date'] > self.train_last_date) & (df['date'] <= self.val_last_date)


            first_val_idx = val_mask.idxmax()  # first True index
            context_start_idx = max(first_val_idx - extra_rows, 0)

            last_val_idx = val_mask[::-1].idxmax()  # last True index
            # ``iloc`` is [start:stop) so add 1 to include last_val_idx
            val_dict[asset] = df.iloc[context_start_idx : last_val_idx + 1]


            # ---------------------------
            # Test slice (d > val_last_date) + context rows
            # ---------------------------
            test_mask = df['date'] > self.val_last_date
            first_test_idx = test_mask.idxmax()
            context_start_idx_test = max(first_test_idx - extra_rows, 0)
            test_dict[asset] = df.iloc[context_start_idx_test:]

        return train_dict, val_dict, test_dict

    def _to_numpy_and_stack(
        self,
        per_asset_df: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        per_asset_X = {
            a: df.drop(["date", "target", "next_return", "spread", "volatility"], axis=1).to_numpy(dtype=np.float32)
            for a, df in per_asset_df.items()
        }
        per_asset_y = {a: df["target"].to_numpy(dtype=np.float32) for a, df in per_asset_df.items()}
        per_asset_next_return = {
            a: df["next_return"].to_numpy(dtype=np.float32) for a, df in per_asset_df.items()
        }
        per_asset_spread = {
            a: df["spread"].to_numpy(dtype=np.float32) for a, df in per_asset_df.items()
        }
        per_asset_volatility = {
            a: df["volatility"].to_numpy(dtype=np.float32) for a, df in per_asset_df.items()
        }

        X, y, next_ret, spread, volatility = self._stack(
            per_asset_X,
            per_asset_y,
            per_asset_next_return,
            per_asset_spread,
            per_asset_volatility,
        )

        # Convert to sequential format if required
        if self.in_seq_len > 1:
            X, y, next_ret, spread, volatility = self.transform_data_to_sequential(
                X, y, next_ret, spread, volatility
            )

        if self.multi_asset_prediction:
            X = np.swapaxes(X, 0, 1)
            y = np.swapaxes(y, 0, 1)
            next_ret = np.swapaxes(next_ret, 0, 1)
            spread = np.swapaxes(spread, 0, 1)
            volatility = np.swapaxes(volatility, 0, 1)

        return X, y, next_ret, spread, volatility

    def _stack(
        self,
        per_asset_X: dict[str, np.ndarray],
        per_asset_y: dict[str, np.ndarray],
        per_asset_next_ret: dict[str, np.ndarray],
        per_asset_spread: dict[str, np.ndarray],
        per_asset_volatility: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Combine per-asset arrays into final tensors.

        X  → (asset, batch, features) or (batch, features)
        y  → (asset, batch)          or (batch)
        nr → (asset, batch)          or (batch)
        """

        if self.multi_asset_prediction:
            X = np.stack(list(per_asset_X.values()), axis=0)
            y = np.stack(list(per_asset_y.values()), axis=0)
            next_ret = np.stack(list(per_asset_next_ret.values()), axis=0)
            spread = np.stack(list(per_asset_spread.values()), axis=0)
            volatility = np.stack(list(per_asset_volatility.values()), axis=0)
        else:
            X = np.vstack(list(per_asset_X.values()))
            y = np.concatenate(list(per_asset_y.values()))
            next_ret = np.concatenate(list(per_asset_next_ret.values()))
            spread = np.concatenate(list(per_asset_spread.values()))
            volatility = np.concatenate(list(per_asset_volatility.values()))

        return (
            X.astype(np.float32),
            y.astype(np.float32),
            next_ret.astype(np.float32),
            spread.astype(np.float32),
            volatility.astype(np.float32),
        )

    def transform_data_to_sequential(self, X: np.ndarray, y: np.ndarray, next_ret: np.ndarray, spread: np.ndarray, volatility: np.ndarray):
        """Convert flat [batch, feat] data into sliding-window sequences along the *batch* axis.

        The sliding window is applied one axis before the last (which is the feature dimension). For
        multi-asset mode the layout is (asset, batch, feat) so `axis=-2` still refers to batch.
        """

        X_seq = (
            sliding_window_view(X, window_shape=self.in_seq_len, axis=len(X.shape) - 2)
            .swapaxes(-2, -1)
        )
        y_seq = y[..., self.in_seq_len - 1:]
        next_ret_seq = next_ret[..., self.in_seq_len - 1:]
        spread_seq = spread[..., self.in_seq_len - 1:]
        volatility_seq = volatility[..., self.in_seq_len - 1:]

        return X_seq, y_seq, next_ret_seq, spread_seq, volatility_seq