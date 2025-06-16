import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.constants import Constants


def filter_by_regular_hours(data, datetime_column):
    return data[(data[datetime_column].dt.time >= Constants.Data.REGULAR_TRADING_HOURS_START) & \
                (data[datetime_column].dt.time <= Constants.Data.REGULAR_TRADING_HOURS_END) & \
                (data[datetime_column].dt.dayofweek < 5)].reset_index(drop=True)