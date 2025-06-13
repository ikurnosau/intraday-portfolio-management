import pandas as pd
import numpy as np


class Balanced3ClassClassification:
    def __init__(self, base_feature: str='close'):
        self.base_feature = base_feature

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return_feature = data[self.base_feature].pct_change()
        next_return = return_feature.shift(-1)

        lower_tertile = next_return.quantile(1/3)
        upper_tertile = next_return.quantile(2/3)

        target = next_return.apply(lambda r: 0 if r < lower_tertile else (2 if r > upper_tertile else 1)).astype(np.float32)

        return target