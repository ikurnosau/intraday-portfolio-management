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
                 multi_asset_prediction: bool):
        self.features = features
        self.target = target
        self.normalizer = normalizer
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.train_last_date = train_set_last_date
        self.multi_asset_prediction = multi_asset_prediction

    def create_dataset_numpy(self, 
                             data: dict[str, pd.DataFrame],
                             date_column='date') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_features = {}
        for asset_name, asset_data in data.items(): 
            logging.info(f'Processing {asset_name}...')

            asset_data = filter_by_regular_hours(asset_data, datetime_column='date')
            logging.info('Filtered by regular hours!')

            asset_data = self.missing_values_handler(asset_data)
            logging.info(f'Missing values are handled!')

            asset_features = pd.DataFrame()
            asset_features[date_column] = pd.to_datetime(asset_data[date_column])

            for indicator_name, indicator_transformation in self.features.items():
                asset_features[indicator_name] = indicator_transformation(asset_data).astype(np.float32)

            logging.info(f'Features calculated!')

            feature_names = list(self.features.keys())
            asset_features.loc[:, feature_names] = self.normalizer(asset_features[feature_names]).astype(np.float32)

            logging.info(f'Features normalized!')

            asset_features['target'] = self.target(asset_data)
            logging.info(f'Target calculated!')

            asset_features = asset_features[asset_features['date'].dt.time >= datetime_lib.time(hour=14, minute=10)]

            num_null_rows = asset_features.isnull().any(axis=1).sum()
            asset_features = asset_features.dropna()
            logging.info(f'Dropped {num_null_rows} rows with NaN values')

            all_features[asset_name] = asset_features

        all_features_train = {
            asset: asset_features[asset_features['date'].apply(lambda date: date <= self.train_last_date)] 
                for asset, asset_features in all_features.items()
        }
        all_features_test = {
            asset: asset_features[asset_features['date'].apply(lambda date: date > self.train_last_date)] 
                for asset, asset_features in all_features.items()
        }

        per_asset_X_train = {asset: asset_features.drop(['date', 'target'], axis=1).to_numpy() for asset, asset_features in all_features_train.items()}
        per_asset_y_train = {asset: asset_features['target'].to_numpy() for asset, asset_features in all_features_train.items()}

        per_asset_X_test = {asset: asset_features.drop(['date', 'target'], axis=1).to_numpy() for asset, asset_features in all_features_test.items()}
        per_asset_y_test = {asset: asset_features['target'].to_numpy() for asset, asset_features in all_features_test.items()}

        if self.multi_asset_prediction: 
            # creating (asset, batch, features) shape
            X_train = np.stack(list(per_asset_X_train.values()), axis=0)
            y_train = np.stack(list(per_asset_y_train.values()), axis=0)

            X_test = np.stack(list(per_asset_X_test.values()), axis=0)
            y_test = np.stack(list(per_asset_y_test.values()), axis=0)
        else: 
            # creating (batch, features) shape
            X_train = np.vstack(list(per_asset_X_train.values()))
            y_train = np.vstack(list(per_asset_y_train.values())).flatten()

            X_test = np.vstack(list(per_asset_X_test.values()))
            y_test = np.vstack(list(per_asset_y_test.values())).flatten()

        if self.in_seq_len > 1:
            # creating (asset, batch, window, features) or (batch, window, features) shape depending on multi_asset_prediction value
            X_train, y_train = self.transform_data_to_sequential(X_train, y_train)
            X_test, y_test = self.transform_data_to_sequential(X_test, y_test)   
             

        # all_features_df = pd.concat(all_features, ignore_index=True)

        # train_df = all_features_df[all_features_df['date'].apply(lambda date: date <= self.train_last_date)]
        # test_df = all_features_df[all_features_df['date'].apply(lambda date: date > self.train_last_date)]

        # X_train, y_train = train_df.drop(['date', 'target'], axis=1).to_numpy(), train_df['target'].to_numpy()
        # X_test, y_test = test_df.drop(['date', 'target'], axis=1).to_numpy(), test_df['target'].to_numpy()

        # if self.in_seq_len > 1: 
        #     X_train, y_train = self.transform_data_to_sequential(X_train, y_train)
        #     X_test, y_test = self.transform_data_to_sequential(X_test, y_test)        

        #     if self.flatten_sequence:
        #         X_train = X_train.reshape((X_train.shape[0], -1))
        #         X_test = X_test.reshape((X_test.shape[0], -1))

        return X_train, y_train, X_test, y_test
    
    def transform_data_to_sequential(self, X, y): 
        X = sliding_window_view(X, window_shape=self.in_seq_len, axis=0).transpose(-3, -1, -2)
        y = y[..., self.in_seq_len - 1:]
        return X, y



