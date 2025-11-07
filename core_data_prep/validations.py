import pandas as pd
import numpy as np
import logging


class Validator: 
    def __init__(self, visualization_depth: int = 50):
        self.validation_index = 0
        self.snapshots = dict()
        self.visualization_depth = visualization_depth

    def validate_input_data(self, data: dict[str, pd.DataFrame]) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_input_data"] = {asset_name: self._head_tail(df) for asset_name, df in data.items()}

        REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

        for asset_name, df in data.items():
            # Basic structural checks ------------------------------------------------
            missing_cols = REQUIRED_COLUMNS.difference(df.columns)
            assert not missing_cols, f"{asset_name}: missing required columns {missing_cols}"

            self._ensure_no_nan_inf(df, f"input data for {asset_name}")
            self._ensure_monotonic_increasing(df["date"], f"{asset_name}: 'date' column")
            assert all(df[["open", "high", "low", "close"]] > 0), f"{asset_name}: prices are not strictly positive"
            assert all(df["volume"] >= 0), f"{asset_name}: volume is not non-negative"

        logging.info(f"Input data validated!")

    def validate_raw_features(self, raw_features: dict[str, pd.DataFrame]) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_raw_features"] = {asset_name: self._head_tail(df) for asset_name, df in raw_features.items()}

        for asset_name, df in raw_features.items():
            self._ensure_no_nan_inf(df, f"features for {asset_name}")

    def validate_filled_data(self, filled_data: dict[str, pd.DataFrame]) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_filled_data"] = {asset_name: self._head_tail(df) for asset_name, df in filled_data.items()}

        for asset_name, asset_df in filled_data.items():
            self._ensure_no_nan_inf(asset_df, f"filled data for {asset_name}")
            if 'is_missing' in asset_df.columns:
                assert asset_df[asset_df['is_missing'] == 1]['volume'].sum() == 0, "Volume is not 0 for missing rows"

        # assert len(set([len(df) for df in filled_data.values()])) == 1, "Filled data has different lengths"

        logging.info(f"Filled data validated!")

    def validate_normalized_features(self,
        normalized_features: dict[str, pd.DataFrame], features_to_normalize: list[str]
    ) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_normalized_features"] = {asset_name: self._head_tail(df) for asset_name, df in normalized_features.items()}
    
        for asset_name, df in normalized_features.items():
            missing_cols = set(features_to_normalize).difference(df.columns)
            assert not missing_cols, f"{asset_name}: missing normalised columns {missing_cols}"

            self._ensure_no_nan_inf(df[features_to_normalize], f"normalised features for {asset_name}")

            # Light-weight statistical sanity checks
            means = df[features_to_normalize].mean()
            stds = df[features_to_normalize].std()

            if not (abs(means) < 1).all():
                raise AssertionError(
                    f"{asset_name}: large mean detected in normalised features – stats: {means.to_dict()}"
                )

            if not ((stds > 0).all() and (stds < 2).all()):
                raise AssertionError(
                    f"{asset_name}: abnormal std detected in normalised features – stats: {stds.to_dict()}"
                )
        
        logging.info(f"Normalised features validated!")

    def validate_sequential_array(self, array_sequential: np.ndarray) -> None:
        assert array_sequential.ndim >= 2, "Sequential array should have at least 2 dimensions"
        assert not np.isnan(array_sequential).any(), "NaNs in sequential array"
        assert not np.isinf(array_sequential).any(), "Infs in sequential array"

    def validate_x(self, x: np.ndarray, n_assets: int, seq_len: int) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_x"] = pd.DataFrame(self._array_head_tail(self._x_per_asset_timestamp(x, asset_i=0, timestamp_i=-1)))

        assert x.ndim == 4, f"Expected X to be 4-D (samples × assets × seq_len × features), got {x.ndim}-D"
        assert x.shape[1] == n_assets, f"Expected X to have {n_assets} assets, got {x.shape[1]}"
        assert x.shape[2] == seq_len, f"Expected X to have {seq_len} sequence length, got {x.shape[2]}"

        logging.info(f"X validated!")

    def validate_target(self, target: np.ndarray) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_target"] = pd.DataFrame(self._array_head_tail(target))

        assert target.ndim == 2, f"Expected target to be 2-D (samples × assets), got {target.ndim}-D"
        assert not np.isnan(target).any(), "NaNs in target array"
        assert not np.isinf(target).any(), "Infs in target array"

        logging.info(f"Target mean: {target.mean()}")

        assert abs(target.mean() - 0.5) < 0.2, "Target mean should be close to 0.5"

        logging.info(f"Target validated!")

    def validate_statistics(self, statistics_name: str, statistics: np.ndarray) -> None:
        if f"{self.validation_index}_statistics" not in self.snapshots:
            self.validation_index += 1
            self.snapshots[f"{self.validation_index}_statistics"] = {}
        self.snapshots[f"{self.validation_index}_statistics"][statistics_name] = pd.DataFrame(self._array_head_tail(statistics))

        assert statistics.ndim == 2, (
            f"{statistics_name}: expected statistics array to be 2-D (samples × assets), got {statistics.ndim}-D"
        )
        assert not np.isnan(statistics).any(), f"NaNs in statistics '{statistics_name}'"
        assert not np.isinf(statistics).any(), f"Infs in statistics '{statistics_name}'"

        logging.info(f"Statistics '{statistics_name}' validated!")

    def validate_x_target_statistics(self, x: np.ndarray, target: np.ndarray, statistics: dict[str, np.ndarray]) -> None:
        lengths = [len(x), len(target)] + [len(statistics[statistic_name]) for statistic_name in statistics.keys()]
        assert len(set(lengths)) == 1, "X, target and statistics have different number of samples"

    def validate_slice_consistency(self,
                                    cur_day_slices: dict[str, pd.DataFrame], 
                                    slice_length: int, 
                                    slice_end_target: pd.Timestamp) -> None:
        self.validation_index += 1
        self.snapshots[f"{self.validation_index}_slices"] = {symbol: self._head_tail(df) for symbol, df in cur_day_slices.items()}

        slice_lengths = {symbol: len(df) for symbol, df in cur_day_slices.items()}
        unique_lengths = set(slice_lengths.values())
        
        # Check that all dataframes have the same length
        assert len(unique_lengths) == 1, \
            f"Slice at {slice_end_target.date()}: Dataframes have different lengths: {slice_lengths}"
        
        # Verify all dataframes have the expected slice_length
        for symbol, length in slice_lengths.items():
            assert length == slice_length, \
                f"Slice at {slice_end_target.date()}, {symbol}: Expected length {slice_length}, got {length}"

    def _head_tail(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= 2 * self.visualization_depth:
            return df.copy()
        gap = pd.DataFrame(
            {col: ["..."] for col in df.columns},
            index=[f"... ({len(df) - 2*self.visualization_depth} rows omitted) ..."]
        )
        return pd.concat([df.head(self.visualization_depth), gap, df.tail(self.visualization_depth)])
        
    def _array_head_tail(self, array: np.ndarray) -> np.ndarray:
        if array.shape[0] <= 2 * self.visualization_depth:
            return array.copy()
        gap = np.array([["..."] * array.shape[1]])

        return np.concatenate([array[:self.visualization_depth], gap, array[-self.visualization_depth:]])

    @staticmethod
    def _ensure_no_nan_inf(df: pd.DataFrame, context: str = "") -> None:
        assert not df.isna().any().any(), f"NaNs detected in {context}"
        assert not df.isin([np.inf, -np.inf]).any().any(), f"Infs detected in {context}"

    @staticmethod
    def _ensure_monotonic_increasing(series: pd.Series, context: str = "") -> None:
        assert series.is_monotonic_increasing, f"{context}: values are not monotonically increasing"

    @staticmethod
    def _x_per_asset_timestamp(x: np.ndarray, asset_i: int, timestamp_i: int) -> np.ndarray:
        return x[:, asset_i, timestamp_i, :]