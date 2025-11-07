import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ZScoreOverWindowNormalizer: 
    def __init__(self, window: int=60, fit_feature=None):
        self.window=window
        self.fit_feature = fit_feature

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled_data = pd.DataFrame()
        if self.fit_feature:
             roll = data[self.fit_feature].rolling(self.window)

        for column in data.columns: 
            if not self.fit_feature:
                roll = data[column].rolling(self.window)
                
            scaled_data[column] = (data[column] - roll.mean()) / (roll.std(ddof=0) + 1e-12)
        
        return scaled_data


class ZScoreNormalizer: 
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)
    
    
class MinMaxNormalizer:
    def __init__(self, fit_feature=None):
        # Keep a dedicated scaler instance; initially unfitted
        self._scaler = MinMaxScaler()
        self.fit_feature = fit_feature

    # ------------------------------------------------------------------
    # scikit-learn-style API (fit / transform)
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame):
        if self.fit_feature:  
            self._scaler.fit(data[[self.fit_feature]])
        else:
            self._scaler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.transform(data),
            columns=data.columns,
            index=data.index,
        )

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.fit_transform(data),
            columns=data.columns,
            index=data.index,
        )

    # Back-compat: calling the instance directly behaves like old code
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(data)


class SmartMinMaxNormalizer:
    def __init__(self, fit_feature=None):
        # Keep a dedicated scaler instance; initially unfitted
        self._scaler = MinMaxScaler()
        self.fit_feature = fit_feature

    # ------------------------------------------------------------------
    # scikit-learn-style API (fit / transform)
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame):
        if self.fit_feature:  
            self._scaler.fit(data[[self.fit_feature]])
        else:
            self._scaler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.transform(data),
            columns=data.columns,
            index=data.index,
        )

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.fit_transform(data),
            columns=data.columns,
            index=data.index,
        )

    # Back-compat: calling the instance directly behaves like old code
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(data)


class MinMaxNormalizerOverWindow:
    """Rolling-window Min-Max scaler.

    For each timestamp *t* the value ``x_t`` is rescaled using the *minimum* and
    *maximum* of the preceding ``window`` samples (inclusive)::

        \hat{x}_t = (x_t - min_{t-w+1:t} x) / (max_{t-w+1:t} x - min_{t-w+1:t} x + eps)

    The result lies in the range \[0, 1\] (unless the window has zero
    variance, in which case the denominator defaults to ``eps``).  The first
    ``window-1`` rows will contain NaNs because the window is not yet filled —
    this is consistent with the behaviour of :class:`ZScoreOverWindowNormalizer`.

    Parameters
    ----------
    window : int, default 60
        Size of the trailing window (in *rows*) used to compute the running
        min/max statistics.
    fit_feature : str | None, default None
        If provided, the rolling min/max are **computed only on this feature**
        (column).  The resulting statistics are then *broadcast* to all other
        columns when scaling.  This mirrors the API of :class:`MinMaxNormalizer`.
    eps : float, default 1e-12
        Numerical stability constant added to the denominator.
    """

    def __init__(self, window: int = 60, fit_feature: str | None = None, eps: float = 1e-12):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.fit_feature = fit_feature
        self.eps = eps

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if self.fit_feature:
            if self.fit_feature not in data.columns:
                raise KeyError(f"fit_feature '{self.fit_feature}' not found in data columns")
            roll_min = data[self.fit_feature].rolling(self.window, min_periods=1).min()
            roll_max = data[self.fit_feature].rolling(self.window, min_periods=1).max()
            denom = (roll_max - roll_min).abs() + self.eps
            # Align index for broadcasting
            scaled = (data.subtract(roll_min, axis=0)).divide(denom, axis=0)
        else:
            roll_min = data.rolling(self.window, min_periods=1).min()
            roll_max = data.rolling(self.window, min_periods=1).max()
            denom = (roll_max - roll_min).abs() + self.eps
            scaled = (data - roll_min) / denom

        return scaled.astype(float)