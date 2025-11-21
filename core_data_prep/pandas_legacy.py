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


def _fill_missing_values(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    last_timestamp = max(df[self.date_column].max() for df in data.values())
    filled_data = {asset_name: self.missing_values_handler(asset_df, last_timestamp, date_column=self.date_column) \
        for asset_name, asset_df in data.items()}

    if self.validator is not None:
        self.validator.validate_filled_data(filled_data)
    
    return filled_data

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