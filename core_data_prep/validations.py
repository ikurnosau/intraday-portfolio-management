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

            if not ((stds >= 0).all() and (stds < 2).all()):
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

        # assert abs(target.mean() - 0.5) < 0.2, "Target mean should be close to 0.5"

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
        for col in df.columns:
            assert not df[col].isna().any(), f"NaNs detected in {context} for column {col}"
            if df[col].isin([np.inf, -np.inf]).any():
                logging.warning(f"Infs detected in {context} for column {col}")
        assert not df.isna().any().any(), f"NaNs detected in {context}"
        assert not df.isin([np.inf, -np.inf]).any().any(), f"Infs detected in {context}"

    @staticmethod
    def _ensure_monotonic_increasing(series: pd.Series, context: str = "") -> None:
        assert series.is_monotonic_increasing, f"{context}: values are not monotonically increasing"

    @staticmethod
    def _x_per_asset_timestamp(x: np.ndarray, asset_i: int, timestamp_i: int) -> np.ndarray:
        return x[:, asset_i, timestamp_i, :]


def validate_streaming_vs_offline_x(
    x_offline: np.ndarray,
    x_streaming: np.ndarray,
    *,
    feature_names: list[str],
    symbols: list[str],
    day_i: int,
    timestamp,
    timestamp_i: int,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    raise_on_mismatch: bool = False,
) -> dict:
    """Compare one offline tensor with its streaming counterpart.

    Expected shape for each input: (assets, seq_len, features).
    On mismatch, logs per-sequence and per-feature breakdowns plus the
    worst cell, then optionally raises.
    """
    assert x_offline.shape == x_streaming.shape, (
        f"Shape mismatch: offline {x_offline.shape} vs streaming {x_streaming.shape}"
    )
    assert x_offline.ndim == 3, (
        f"Expected (assets, seq_len, features), got ndim={x_offline.ndim}"
    )
    n_assets, seq_len, n_features = x_offline.shape
    assert n_assets == len(symbols), (
        f"Expected {len(symbols)} symbols, got {n_assets} assets in X"
    )
    assert n_features == len(feature_names), (
        f"Expected {len(feature_names)} features, got {n_features} in X"
    )

    matches = np.allclose(
        x_offline,
        x_streaming,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    abs_difference = np.abs(x_offline - x_streaming)
    summary = {
        "matches": matches,
        "day_i": day_i,
        "timestamp": timestamp,
        "timestamp_i": timestamp_i,
        "max_abs_diff": float(np.nanmax(abs_difference)),
        "median_abs_diff": float(np.nanmedian(abs_difference)),
        "mean_abs_diff": float(np.nanmean(abs_difference)),
    }
    if matches:
        return summary

    logging.warning(
        "Input mismatch on day %s at %s: max_abs_diff=%s, "
        "median_abs_diff=%s, mean_abs_diff=%s",
        day_i,
        timestamp,
        summary["max_abs_diff"],
        summary["median_abs_diff"],
        summary["mean_abs_diff"],
    )

    per_seq = []
    for seq_i in range(seq_len):
        seq_diff = abs_difference[:, seq_i, :]
        per_seq.append({
            "seq_i": seq_i,
            "allclose": bool(np.allclose(
                x_offline[:, seq_i, :],
                x_streaming[:, seq_i, :],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )),
            "max_abs_diff": float(np.nanmax(seq_diff)),
            "median_abs_diff": float(np.nanmedian(seq_diff)),
            "mean_abs_diff": float(np.nanmean(seq_diff)),
        })
    mismatch_seq = [row for row in per_seq if not row["allclose"]]
    summary["mismatch_seq"] = mismatch_seq
    logging.warning(
        "seq positions mismatched: %s/%s\n%s",
        len(mismatch_seq),
        seq_len,
        pd.DataFrame(mismatch_seq if mismatch_seq else per_seq),
    )

    per_feature = []
    for feat_i, feat_name in enumerate(feature_names):
        feat_diff = abs_difference[:, :, feat_i]
        per_feature.append({
            "feature": feat_name,
            "allclose": bool(np.allclose(
                x_offline[:, :, feat_i],
                x_streaming[:, :, feat_i],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )),
            "max_abs_diff": float(np.nanmax(feat_diff)),
            "median_abs_diff": float(np.nanmedian(feat_diff)),
            "mean_abs_diff": float(np.nanmean(feat_diff)),
            "n_mismatch_cells": int(np.sum(
                ~np.isclose(
                    x_offline[:, :, feat_i],
                    x_streaming[:, :, feat_i],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            )),
        })
    mismatch_features = [row for row in per_feature if not row["allclose"]]
    summary["mismatch_features"] = mismatch_features
    logging.warning(
        "features mismatched: %s/%s\n%s",
        len(mismatch_features),
        n_features,
        pd.DataFrame(mismatch_features if mismatch_features else per_feature),
    )

    flat_idx = int(np.nanargmax(abs_difference))
    asset_i, seq_i, feat_i = np.unravel_index(flat_idx, abs_difference.shape)
    worst = {
        "symbol": symbols[asset_i],
        "seq_i": int(seq_i),
        "feature": feature_names[feat_i],
        "diff": float(abs_difference[asset_i, seq_i, feat_i]),
        "offline": float(x_offline[asset_i, seq_i, feat_i]),
        "streaming": float(x_streaming[asset_i, seq_i, feat_i]),
    }
    summary["worst_cell"] = worst
    logging.warning(
        "worst cell: symbol=%s, seq_i=%s, feature=%s, diff=%s, "
        "offline=%s, streaming=%s",
        worst["symbol"],
        worst["seq_i"],
        worst["feature"],
        worst["diff"],
        worst["offline"],
        worst["streaming"],
    )

    if raise_on_mismatch:
        raise AssertionError(
            f"Streaming/offline input mismatch on day {day_i} at {timestamp}: "
            f"max_abs_diff={summary['max_abs_diff']}, "
            f"worst={worst}"
        )
    return summary


def validate_streaming_vs_offline_returns(
    *,
    streaming_allocation: np.ndarray,
    offline_allocation: np.ndarray,
    day_i: int,
    timestamp,
    timestamp_i: int,
    symbols: list[str],
    streaming_step_return: float | None = None,
    offline_realized_return: float | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-4,
    return_rtol: float = 0.2,
    return_atol: float = 1e-4,
    raise_on_mismatch: bool = True,
) -> dict:
    """Compare streaming trader allocation / step return to offline backtest.

    Offline ``realized_returns[t]`` is
    ``sum(allocation[t] * next_return[t]) - cost(allocation[t-1] -> allocation[t])``.

    Streaming should pass the same quantity as ``streaming_step_return`` once the
    previous bar's mark-to-market and the trade cost at ``t`` are available
    (typically checked on the following timestamp, before the next trade).

    Allocations use tight ``rtol``/``atol``. Step returns use looser
    ``return_rtol``/``return_atol`` because streaming trades whole shares, so
    effective notional (and thus step return) can differ slightly from the
    continuous offline portfolio return. ``np.isclose`` is preferred over a raw
    ratio check: near-zero bars would otherwise divide by ~0.
    """
    streaming_allocation = np.asarray(streaming_allocation, dtype=np.float64)
    offline_allocation = np.asarray(offline_allocation, dtype=np.float64)
    assert streaming_allocation.shape == offline_allocation.shape, (
        f"Allocation shape mismatch: streaming {streaming_allocation.shape} "
        f"vs offline {offline_allocation.shape}"
    )
    assert streaming_allocation.shape == (len(symbols),), (
        f"Expected allocation length {len(symbols)}, got {streaming_allocation.shape}"
    )

    alloc_matches = np.allclose(
        streaming_allocation,
        offline_allocation,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    alloc_abs_diff = np.abs(streaming_allocation - offline_allocation)
    summary = {
        "allocation_matches": alloc_matches,
        "day_i": day_i,
        "timestamp": timestamp,
        "timestamp_i": timestamp_i,
        "allocation_max_abs_diff": float(np.nanmax(alloc_abs_diff)),
        "allocation_mean_abs_diff": float(np.nanmean(alloc_abs_diff)),
    }
    if not alloc_matches:
        worst_i = int(np.nanargmax(alloc_abs_diff))
        summary["allocation_worst"] = {
            "symbol": symbols[worst_i],
            "streaming": float(streaming_allocation[worst_i]),
            "offline": float(offline_allocation[worst_i]),
            "diff": float(alloc_abs_diff[worst_i]),
        }
        logging.warning(
            "Allocation mismatch on day %s at %s (idx=%s): max_abs_diff=%s, worst=%s",
            day_i,
            timestamp,
            timestamp_i,
            summary["allocation_max_abs_diff"],
            summary["allocation_worst"],
        )

    return_matches = True
    if streaming_step_return is not None and offline_realized_return is not None:
        return_matches = bool(np.isclose(
            streaming_step_return,
            offline_realized_return,
            rtol=return_rtol,
            atol=return_atol,
            equal_nan=True,
        ))
        summary["return_matches"] = return_matches
        summary["streaming_step_return"] = float(streaming_step_return)
        summary["offline_realized_return"] = float(offline_realized_return)
        summary["return_abs_diff"] = float(
            abs(streaming_step_return - offline_realized_return)
        )
        if abs(offline_realized_return) > 0:
            summary["return_ratio"] = float(
                streaming_step_return / offline_realized_return
            )
        else:
            summary["return_ratio"] = None
        if not return_matches:
            logging.warning(
                "Return mismatch on day %s at %s (idx=%s): "
                "streaming=%s, offline=%s, abs_diff=%s, ratio=%s "
                "(return_rtol=%s, return_atol=%s)",
                day_i,
                timestamp,
                timestamp_i,
                summary["streaming_step_return"],
                summary["offline_realized_return"],
                summary["return_abs_diff"],
                summary["return_ratio"],
                return_rtol,
                return_atol,
            )

    summary["matches"] = alloc_matches and return_matches
    if raise_on_mismatch and not summary["matches"]:
        raise AssertionError(
            f"Streaming/offline returns mismatch on day {day_i} at {timestamp}: "
            f"{summary}"
        )
    return summary