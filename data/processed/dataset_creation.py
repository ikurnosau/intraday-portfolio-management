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


class DatasetCreator: 
    def __init__(self, 
                 features: dict[str, Callable],
                 target: Callable,
                 normalizer: Callable,
                 missing_values_handler: Callable,
                 in_seq_len: int,
                 train_set_last_date: datetime,
                 multi_asset_prediction: bool,
                 cutoff_time: datetime_lib.time | None = datetime_lib.time(hour=14, minute=10)):
        self.features = features
        self.target = target
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.train_last_date = train_set_last_date
        self.multi_asset_prediction = multi_asset_prediction
        self.cutoff_time = cutoff_time

    def create_dataset_numpy(self, 
                             data: dict[str, pd.DataFrame],
                             date_column: str = 'date') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """High-level orchestrator that transforms raw per-asset OHLCV data into
        numpy arrays suitable for model consumption.

        The method now delegates to a set of small private helpers so that each
        step can be unit-tested independently.
        """

        # ------------------------------------------------------------------
        # 1. Pre-process every asset independently
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2. Train / test split per asset
        # ------------------------------------------------------------------
        train_dict, test_dict = self._train_test_split(processed_assets)

        # ------------------------------------------------------------------
        # 3. Convert per-asset DataFrames → numpy dictionaries
        # ------------------------------------------------------------------
        per_asset_X_train = {a: df.drop(['date', 'target'], axis=1).to_numpy(dtype=np.float32)
                              for a, df in train_dict.items()}
        per_asset_y_train = {a: df['target'].to_numpy(dtype=np.float32)
                              for a, df in train_dict.items()}

        per_asset_X_test = {a: df.drop(['date', 'target'], axis=1).to_numpy(dtype=np.float32)
                             for a, df in test_dict.items()}
        per_asset_y_test = {a: df['target'].to_numpy(dtype=np.float32)
                             for a, df in test_dict.items()}

        # ------------------------------------------------------------------
        # 4. Stack into final tensors
        # ------------------------------------------------------------------
        X_train, y_train = self._stack(per_asset_X_train, per_asset_y_train)
        X_test, y_test = self._stack(per_asset_X_test, per_asset_y_test)

        # ------------------------------------------------------------------
        # 5. Convert to sequential format if requested
        # ------------------------------------------------------------------
        if self.in_seq_len > 1:
            X_train, y_train = self.transform_data_to_sequential(X_train, y_train)
            X_test, y_test = self.transform_data_to_sequential(X_test, y_test)

        return X_train, y_train, X_test, y_test
    
    def transform_data_to_sequential(self, X, y): 
        X = sliding_window_view(X, window_shape=self.in_seq_len, axis=len(X.shape) - 2).swapaxes(-2, -1)
        y = y[..., self.in_seq_len - 1:]

        return X, y

    def _process_single_asset(self, asset_df: pd.DataFrame, *, date_column: str = 'date') -> pd.DataFrame:
        """Run full feature/target engineering pipeline for a single asset."""

        # Filter to regular trading hours
        asset_df = filter_by_regular_hours(asset_df, datetime_column=date_column)

        # Handle missing values (can expand later)
        asset_df = self.missing_values_handler(asset_df)

        # --------------------------------------------------------------
        # Build features DataFrame (date + engineered features)
        # --------------------------------------------------------------
        feat_df = pd.DataFrame()
        feat_df[date_column] = pd.to_datetime(asset_df[date_column])

        for name, transform in self.features.items():
            feat_df[name] = transform(asset_df).astype(np.float32)

        # Normalise only the feature columns
        feature_cols = list(self.features.keys())
        feat_df.loc[:, feature_cols] = self.normalizer(feat_df[feature_cols]).astype(np.float32)

        # Add target
        feat_df['target'] = self.target(asset_df)

        # Optional intra-day cutoff
        if self.cutoff_time is not None:
            feat_df = feat_df[feat_df['date'].dt.time >= self.cutoff_time]

        # Fill remaining NaNs with 0.5 sentinel (kept for backward compat)
        num_null_rows = feat_df.isnull().any(axis=1).sum()
        if num_null_rows:
            logging.info(f"Imputing {num_null_rows} NaN rows with 0.5 sentinel value")
        feat_df = feat_df.fillna(0.5)

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

    def _stack(self, per_asset_X: dict[str, np.ndarray], per_asset_y: dict[str, np.ndarray]):
        """Combine per-asset arrays into the final model-ready structure."""
        if self.multi_asset_prediction:
            # shape → (asset, batch, features)
            X = np.stack(list(per_asset_X.values()), axis=0)
            y = np.stack(list(per_asset_y.values()), axis=0)
        else:
            # shape → (batch, features)
            X = np.vstack(list(per_asset_X.values()))
            y = np.vstack(list(per_asset_y.values())).flatten()
        return X.astype(np.float32), y.astype(np.float32)