import sys
import os
import pandas as pd
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.constants import Constants



def filter_by_regular_hours(data, datetime_column):
    return data[(data[datetime_column].dt.time >= Constants.Data.REGULAR_TRADING_HOURS_START) & \
                (data[datetime_column].dt.time <= Constants.Data.REGULAR_TRADING_HOURS_END) & \
                (data[datetime_column].dt.dayofweek < 5)].reset_index(drop=True)


def convert_to_eastern(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Convert datetime column to US Eastern Time (America/New_York).
        
        Args:
            df: DataFrame with datetime column
            date_column: Name of the datetime column to convert
            
        Returns:
            DataFrame with datetime column converted to Eastern Time
        """

        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # If timezone-naive, assume UTC
        if df[date_column].dt.tz is None:
            df[date_column] = df[date_column].dt.tz_localize('UTC')
        
        # Convert to Eastern Time
        df[date_column] = df[date_column].dt.tz_convert(Constants.Data.EASTERN_TZ)
        
        return df