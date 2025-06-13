import pandas as pd


class DummyMissingValuesHandler: 
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return data