import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
from typing import Callable, Tuple
from datetime import datetime
import logging
import numpy as np


def get_x_y(df): 
    return df.drop(['date', 'should_buy'], axis=1).to_numpy(), df['should_buy'].to_numpy()


class DatasetCreator: 
    def __init__(self, 
                 features: dict[str, Callable],
                 target: Callable,
                 normalizer: Callable,
                 missing_values_handler: Callable,
                 in_seq_len: int,
                 train_set_last_date: datetime):
        self.features = features
        self.target = target
        self.normalizer = normalizer,
        self.missing_values_handler = missing_values_handler
        self.in_seq_len = in_seq_len
        self.train_last_date = train_set_last_date

    def create_dataset_numpy(self, 
                             data: dict[str, pd.DataFrame],
                             date_column='date') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_features = []
        for asset_name, asset_data in data.items(): 
            logging.info(f'Processing {asset_name}...')

            data = self.missing_values_handler(data)
            logging.info(f'Missing values are handled!')

            asset_features = pd.DataFrame()
            asset_features[date_column] = pd.to_datetime(asset_data[date_column])

            for indicator_name, indicator_transformation in self.features.items():
                asset_features[indicator_name] = indicator_transformation(asset_data).astype(np.float32)

            asset_data['volume'] += 1e-12
            asset_data['volume'] -= 1e-12

            logging.info(f'Features calculated!')


            feature_names = list(self.features.keys())
            asset_features.loc[:, feature_names] = self.normalizer(asset_features[feature_names]).astype(np.float32)

            logging.info(f'Features normalized!')

            asset_features['target'] = self.target(asset_data)
            logging.info(f'Target calculated!')

            num_null_rows = asset_features.isnull().any(axis=1).sum()
            asset_features = asset_features.dropna()
            logging.info(f'Dropped {num_null_rows} rows with NaN values')

            all_features.append(asset_features)

        all_features_df = pd.concat(all_features, ignore_index=True)

        """
        Add logic for creating sequential datastes
        """
        
        train_df = all_features_df[all_features_df['date'].apply(lambda date: date <= self.train_last_date)]
        test_df = all_features_df[all_features_df['date'].apply(lambda date: date > self.train_last_date)]

        X_train, y_train = train_df.drop(['date', 'target'], axis=1).to_numpy(), train_df['target'].to_numpy()
        X_test, y_test = test_df.drop(['date', 'target'], axis=1).to_numpy(), test_df['target'].to_numpy()

        return X_train, y_train, X_test, y_test



