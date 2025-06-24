import pandas as pd
import numpy as np
from enum import Enum


class WILLR:
    def __init__(self, period=10):
        self.period = period

    def __call__(self, df):
        # Calculate the highest high and lowest low over the period
        highest_high = df['high'].rolling(window=self.period).max()
        lowest_low = df['low'].rolling(window=self.period).min()

        # Calculate Williams %R
        williams_r = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
        return williams_r


class ROCR:
    def __init__(self, period=10, base_feature='close'):
        self.period = period
        self.base_feature = base_feature

    def __call__(self, df):
        # Calculate the Rate of Change
        previous_values = df[self.base_feature].shift(self.period)
        roc = (df[self.base_feature] / previous_values) * 100
        return roc


class MOM:
    def __init__(self, period, base_feature='close'):
        self.period = period
        self.base_feature = base_feature

    def __call__(self, df):
        # Calculate the Momentum
        momentum = df[self.base_feature].diff(periods=self.period)
        return momentum


class RSI:
    def __init__(self, period=14, base_feature='close'):
        self.period = period
        self.base_feature = base_feature

    def __call__(self, df):
        delta = df[self.base_feature].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate the average gains and losses
        avg_gain = gain.rolling(window=self.period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=1).mean()

        # Use Wilder's method for smoothing
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class CCI:
    def __init__(self,
                 period=14,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close'):
        self.period = period
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature

    def __call__(self, df):
        # Calculate the Typical Price
        tp = (df[self.high_feature] + df[self.low_feature] + df[self.close_feature]) / 3

        # Calculate the moving average of the Typical Price
        tp_avg = tp.rolling(window=self.period).mean()

        # Calculate the Mean Deviation
        tp_md = (abs(tp - tp_avg)).rolling(window=self.period).mean()

        # Calculate the Commodity Channel Index
        cci = (tp - tp_avg) / (0.15 * tp_md)

        return cci


class ADX:
    def __init__(self,
                 period=14,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close'):
        self.period = period
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature

    def __call__(self, df):
        high = df[self.high_feature]
        low = df[self.low_feature]
        close = df[self.close_feature]

        # Calculate True Range
        prev_close = close.shift(1)
        tr = (high - low).combine(abs(high - prev_close), max).combine(abs(low - prev_close), max)

        # Calculate ATR (Average True Range)
        atr = tr.rolling(
            window=self.period).mean()  # Alternatively, you can use an exponential moving average for smoothing

        # Calculate +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smooth +DM and -DM
        plus_dm_smooth = plus_dm.rolling(window=self.period).mean()
        minus_dm_smooth = minus_dm.rolling(window=self.period).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # Calculate DX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

        # Calculate ADX
        adx = dx.rolling(window=self.period).mean()  # Alternatively, use an exponential moving average for smoothing

        return adx


class TRIX:
    def __init__(self, period=20, base_feature='close'):
        self.period = period
        self.base_feature = base_feature

    def __call__(self, df):
        # First EMA
        ema1 = df[self.base_feature].ewm(span=self.period, adjust=False).mean()
        # Second EMA
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        # Third EMA
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()

        # Triple Exponential Moving Average
        tema = (3 * ema1) - (3 * ema2) + ema3

        return tema


class MACD:
    def __init__(self, base_feature='close'):
        self.base_feature = base_feature

    def __call__(self, df):
        short_ema = df[self.base_feature].ewm(span=12, adjust=False).mean()
        long_ema = df[self.base_feature].ewm(span=26, adjust=False).mean()

        # Calculate the MACD Line
        macd = short_ema - long_ema

        # Calculate the Signal Line
        signal_line = macd.ewm(span=9, adjust=False).mean()

        # Calculate the MACD Histogram
        return macd - signal_line


class OBV:
    def __init__(self, base_feature='close', volume_feature='volume'):
        self.base_feature = base_feature
        self.volume_feature = volume_feature

    def __call__(self, df):
        # Calculate daily price change
        change = df[self.base_feature].diff()

        # Apply the OBV formula
        obv = (df[self.volume_feature] * change.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()

        return obv


class ATR:
    def __init__(self,
                 period,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close'):
        self.period = period
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature

    def __call__(self, df):
        # Calculate the True Range (TR)
        prev_close = df[self.close_feature].shift(1)

        temp_df = pd.DataFrame()
        temp_df['high-low'] = df[self.high_feature] - df[self.low_feature]
        temp_df['high-Previous close'] = np.abs(df[self.high_feature] - prev_close)
        temp_df['low-Previous close'] = np.abs(df[self.low_feature] - prev_close)

        tr = temp_df[['high-low', 'high-Previous close', 'low-Previous close']].max(axis=1)

        # Calculate the ATR using an Exponential Moving Average (EMA)
        return tr.ewm(span=self.period, adjust=False).mean()


class MFI:
    def __init__(self,
                 period,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close',
                 volume_feature='volume'):
        self.period = period
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature
        self.volume_feature = volume_feature

    def __call__(self, df):
        # Calculate Typical Price
        tp = (df[self.high_feature] + df[self.low_feature] + df[self.close_feature]) / 3

        # Calculate Raw Money Flow
        raw_money_flow = tp * df[self.volume_feature]

        # Determine Positive and Negative Money Flow
        change = tp.diff()
        positive_flow = raw_money_flow.where(change > 0, 0)
        negative_flow = raw_money_flow.where(change < 0, 0)

        # Calculate the Money Flow Ratio
        pos_flow_sum = positive_flow.rolling(window=self.period).sum()
        neg_flow_sum = negative_flow.rolling(window=self.period).sum()

        money_flow_ratio = pos_flow_sum / neg_flow_sum

        # Calculate the Money Flow Index
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return mfi


class EMA:
    def __init__(self, period, base_feature='close'):
        self.period = period
        self.base_feature = base_feature

    def __call__(self, df):
        return df[self.base_feature].ewm(span=self.period, adjust=False).mean()


class BollingerBand:
    class BBType(Enum):
        LOWER = 1
        UPPER = 2

    def __init__(self, bb_type: BBType, period=20, base_feature='close'):
        self.period = period
        self.bb_type = bb_type
        self.base_feature = base_feature

    def __call__(self, df):
        sma = df[self.base_feature].rolling(window=self.period).mean()

        # Calculate the rolling standard deviation
        std = df[self.base_feature].rolling(window=self.period).std()

        # Calculate the Bollinger Bands
        if self.bb_type == self.BBType.LOWER:
            return sma - (std * 2)
        else:
            return sma + (std * 2)


class VWAP:
    def __init__(self,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close',
                 volume_feature='volume'):
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature
        self.volume_feature = volume_feature

    def __call__(self, df):
        typical_price = (df[self.high_feature] + df[self.low_feature] + df[self.close_feature]) / 3

        # Calculate Cumulative TPV and Cumulative Volume
        cumulative_tvp = (typical_price * df[self.volume_feature]).cumsum()
        cumulative_volume = df[self.volume_feature].cumsum()

        # Calculate the VWAP
        return cumulative_tvp / cumulative_volume


class Oscillator:
    class LineType(Enum):
        K = 1
        D = 2

    def __init__(self,
                 line_type: LineType,
                 period=14,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close'):
        self.period = period
        self.line_type = line_type
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature

    def __call__(self, df):
        lowest_low = df[self.low_feature].rolling(window=self.period).min()
        highest_high = df[self.high_feature].rolling(window=self.period).max()

        k_line = ((df[self.close_feature] - lowest_low) / (highest_high - lowest_low)) * 100
        if self.line_type == self.LineType.K:
            # Calculate the %K Line
            return k_line
        else:
            # Calculate the %D Line
            return k_line.rolling(window=3).mean()


class FRL:
    FIB_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.764)

    def __init__(self,
                 fib_ratio,
                 period=10,
                 high_feature='high',
                 low_feature='low',
                 close_feature='close'):
        self.fib_ratio = fib_ratio
        self.period = period
        self.high_feature = high_feature
        self.low_feature = low_feature
        self.close_feature = close_feature

    def __call__(self, df):
        swing_high = df[self.high_feature].rolling(window=self.period).max()
        swing_low = df[self.low_feature].rolling(window=self.period).min()

        # Calculate Fibonacci levels
        return swing_low + (swing_high - swing_low) * self.fib_ratio


class Vol:
    def __init__(self,
                 period,
                 close_feature='close'):
        self.period = period
        self.close_feature = close_feature

    def __call__(self, df):
        prev_close = df[self.close_feature].shift(1)
        vol = abs(df[self.close_feature] - prev_close) / df[self.close_feature]
        return vol.ewm(span=self.period, adjust=False).mean().fillna(0)


class LogVolumeReturn:
    """Computes log(volume_t / volume_{t-1} + epsilon) to get scale-invariant volume changes."""
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # Add epsilon to both numerator and denominator to handle zeros
        volume_t = data['volume'] + self.epsilon
        volume_tm1 = data['volume'].shift(1) + self.epsilon
        
        # Compute log ratio and clip extreme values
        log_ratio = np.log(volume_t / volume_tm1)
        # Clip to reasonable range (-10, 10) to prevent extreme values
        return np.clip(log_ratio, -10, 10).astype(np.float32)


class IntradayTime:
    """Converts timestamp to normalized minutes from market open (0.0 to 1.0)."""
    def __init__(self, 
                market_open: pd.Timestamp = pd.Timestamp('13:30').time(),
                market_close: pd.Timestamp = pd.Timestamp('20:00').time()):
        self.market_open = market_open
        self.market_close = market_close
        # Pre-compute total trading minutes for normalization
        open_minutes = market_open.hour * 60 + market_open.minute
        close_minutes = market_close.hour * 60 + market_close.minute
        self.total_trading_minutes = close_minutes - open_minutes

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # Convert timestamp to minutes since market open
        minutes_from_open = (data['date'].dt.hour * 60 + data['date'].dt.minute) - \
                          (self.market_open.hour * 60 + self.market_open.minute)
        # Normalize to 0-1 range and ensure bounds
        normalized = minutes_from_open / self.total_trading_minutes
        return np.clip(normalized, 0, 1).astype(np.float32)