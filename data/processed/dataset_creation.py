import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
from typing import Callable, Tuple
from datetime import datetime
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
                 multi_asset_prediction: bool,
                 cutoff_time: datetime_lib.time | None = datetime_lib.time(hour=14, minute=10), 
                 include_test_data: bool = True):
        self.features = features
        self.target = target
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.train_last_date = train_set_last_date
        self.multi_asset_prediction = multi_asset_prediction
        self.cutoff_time = cutoff_time
        self.include_test_data = include_test_data

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
        """High-level orchestrator that transforms raw per-asset OHLCV data into
        numpy arrays suitable for model consumption.

        The method now delegates to a set of small private helpers so that each
        step can be unit-tested independently.
        """

        processed_assets: dict[str, pd.DataFrame] = {}
        skipped_assets: list[str] = []

        for asset_name, asset_df in data.items():
            logging.info(f"Processing {asset_name} …")
            features_df = self._process_single_asset(asset_df, date_column=date_column)

            if not processed_assets:
                required_rows = len(features_df)
            else:
                required_rows = len(next(iter(processed_assets.values())))

            if len(features_df) == required_rows:
                processed_assets[asset_name] = features_df
            else:
                logging.info(
                    f"{asset_name} has {len(features_df)} rows, but {required_rows} are "
                    "expected. Skipping …"
                )
                skipped_assets.append(asset_name)

        logging.info(
            f"Finished feature generation. {len(skipped_assets)} assets skipped due to insufficient rows."
        )

        train_dict, test_dict = self._train_test_split(processed_assets)

        X_train, y_train, next_return_train, spread_train, volatility_train = self._to_numpy_and_stack(train_dict)
        X_test, y_test, next_return_test, spread_test, volatility_test = self._to_numpy_and_stack(test_dict) \
            if self.include_test_data else (None, None, None, None, None)

        return (
            X_train,
            y_train,
            next_return_train,
            spread_train,
            volatility_train,
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
        logging.info(f"Spread has {feat_df['spread'].isna().sum()} NaNs")
        feat_df['spread'] = feat_df['spread'].fillna(0.0).astype(np.float32)

        if self.cutoff_time is not None:
            feat_df = feat_df[feat_df['date'].dt.time >= self.cutoff_time]

        if hasattr(self.target, "horizon"):
            feat_df = feat_df[feat_df['date'].dt.time <= Constants.Data.REGULAR_TRADING_HOURS_END - self.target.horizon]

        num_null_rows = feat_df.isnull().any(axis=1).sum()
        if num_null_rows:
            logging.info(f"Imputing {num_null_rows} NaN rows with 0.5 sentinel value")
        feat_df[feature_cols] = feat_df[feature_cols].fillna(0.5)
        assert feat_df.isna().sum().sum() == 0, "There are still NaNs in the dataset"

        return feat_df.reset_index(drop=True)

    def _train_test_split(self, full_dict: dict[str, pd.DataFrame]):
        """Split each asset DataFrame into train and test parts."""
        train_dict = {
            asset: df[df['date'] <= self.train_last_date]
            for asset, df in full_dict.items()
        }
        test_dict = {
            asset: df[df['date'] > self.train_last_date]
            for asset, df in full_dict.items()
        }
        return train_dict, test_dict

    def _to_numpy_and_stack(
        self,
        per_asset_df: dict[str, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert a per-asset DataFrame dictionary into stacked (X, y, next_ret, spread, volatility) numpy arrays.

        This consolidates all logic that was previously duplicated for the train and test
        splits inside ``create_dataset_numpy``:

        1. Extract feature matrix, target, next-period return and spreads as *per-asset* numpy arrays.
        2. Call :py:meth:`_stack` to combine them into either a multi-asset tensor or a flat batch,
           depending on ``self.multi_asset_prediction``.
        3. Optionally convert the flat representation into sliding windows when
           ``self.in_seq_len`` > 1.
        """

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