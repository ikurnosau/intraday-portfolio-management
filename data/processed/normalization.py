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