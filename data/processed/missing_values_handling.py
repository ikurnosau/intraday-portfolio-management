import pandas as pd
import logging
from datetime import timedelta


class DummyMissingValuesHandler:
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna()


class ForwardFillFlatBars:
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data_orig = data.copy()

        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

        data_filled = (data
                       .groupby(data.index.date)
                       .apply(self.fill_day_minutes)
                       .droplevel(0)
                       .reset_index()
                       .rename(columns={'index':'date'}))
        
        data_filled['close'] = data_filled['close'].ffill()
        if 'ask_price' in data_filled.columns:
            data_filled['ask_price'] = data_filled['ask_price'].ffill()
        if 'bid_price' in data_filled.columns:
            data_filled['bid_price'] = data_filled['bid_price'].ffill()

        missing = data_filled['volume'].isna()

        logging.info(f"Imputing {missing.sum()} NaN rows out of {len(data_filled)} with forward fill..")

        data_filled.loc[missing, 'open'] = data_filled.loc[missing, 'close']
        data_filled.loc[missing, 'high'] = data_filled.loc[missing, 'close']
        data_filled.loc[missing, 'low'] = data_filled.loc[missing, 'close']
        data_filled.loc[missing, 'volume'] = 0        

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

    @staticmethod
    def fill_day_minutes(day_slice):
        day = day_slice.index[0].normalize()  # midnight of that day
        full_idx = pd.date_range(
            start=day + pd.Timedelta(hours=13, minutes=30),
            end=  day + pd.Timedelta(hours=20,     minutes=0),
            freq='min'
        )
        return day_slice.reindex(full_idx)
