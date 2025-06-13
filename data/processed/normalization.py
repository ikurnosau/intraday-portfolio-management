import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ZScoreOverWindowNormalizer: 
    def __init__(self, window: int=60):
        self.window=window

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled_data = pd.DataFrame()
        for column in data.columns: 
            roll = data[column].rolling(self.window)
            scaled_data[column] = (data[column] - roll.mean()) / (roll.std(ddof=0) + 1e-12)
        
        return scaled_data


class ZScoreNormalizer: 
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)
    
    
class MinMaxNormalizer: 
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns, index=data.index)