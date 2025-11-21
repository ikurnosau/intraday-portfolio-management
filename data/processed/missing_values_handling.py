import pandas as pd
import polars as pl
import numpy as np
import logging

from config.constants import Constants


class DummyMissingValuesHandler:
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna()


class ForwardFillFlatBars:
    def __init__(self, frequency: str):
        self.frequency = frequency

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_orig = data.copy()

        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

        data_filled = (data
                       .groupby(data.index.date)
                       .apply(self.fill_day)
                       .droplevel(0)
                       .reset_index()
                       .rename(columns={'index':'date'}))
        
        data_filled['close'] = data_filled['close'].ffill()
        if 'ask_price' in data_filled.columns:
            data_filled['ask_price'] = data_filled['ask_price'].ffill()
        if 'bid_price' in data_filled.columns:
            data_filled['bid_price'] = data_filled['bid_price'].ffill()

        missing = data_filled['volume'].isna()

        logging.info("Imputing %d NaN rows out of %d with forward fill..", missing.sum(), len(data_filled))

        data_filled.loc[missing, 'open'] = data_filled.loc[missing, 'close']
        data_filled.loc[missing, 'high'] = data_filled.loc[missing, 'close']
        data_filled.loc[missing, 'low'] = data_filled.loc[missing, 'close']
        data_filled.loc[missing, 'volume'] = 0        

        if self.frequency == 'min':
            self.test_original_data_preserved(data_orig, data_filled)

        return data_filled.reset_index(drop=True)
    
    @staticmethod
    def test_original_data_preserved(original_df, filled_df):
        assert len(filled_df) % 391 == 0

        original_df = original_df.copy().set_index('date')
        filled_df = filled_df.copy().set_index('date')

        filled_subset = filled_df.loc[original_df.index]

        pd.testing.assert_frame_equal(
            original_df.sort_index(), 
            filled_subset.sort_index(),
            check_dtype=False,  # In case NaNs promoted types in filled_df
            check_exact=True
        )

    def fill_day(self, day_slice):        
        day = day_slice.index[0].normalize()  # midnight of that day

        # Use constants for regular trading hours
        trading_start = Constants.Data.REGULAR_TRADING_HOURS_START
        trading_end = Constants.Data.REGULAR_TRADING_HOURS_END
        
        start = day + pd.Timedelta(hours=trading_start.hour, minutes=trading_start.minute)
        end = day + pd.Timedelta(hours=trading_end.hour, minutes=trading_end.minute)
        
        if self.frequency == '1Min':
            freq = 'min'
        elif self.frequency in ('5Min', '15Min'):
            freq = self.frequency.lower()
        elif self.frequency == '1Hour':
            # For hourly frequency, start one hour after market open
            start = day + pd.Timedelta(hours=trading_start.hour + 1)
            freq = 'h'
        elif self.frequency == '1Day':
            # For daily frequency, use one specific time during the day
            start = day + pd.Timedelta(hours=trading_start.hour + 1)
            freq = 'd'
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")

        full_idx = pd.date_range(start=start, end=end, freq=freq)
        return day_slice.reindex(full_idx)

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