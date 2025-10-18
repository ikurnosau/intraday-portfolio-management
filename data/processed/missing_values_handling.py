import pandas as pd
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
