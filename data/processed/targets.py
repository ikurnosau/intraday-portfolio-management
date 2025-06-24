import pandas as pd
import numpy as np


class Balanced3ClassClassification:
    def __init__(self, base_feature: str='close'):
        self.base_feature = base_feature
        self.lower_tertile: float | None = None
        self.upper_tertile: float | None = None

    def fit(self, train_df: pd.DataFrame):
        """Compute class boundaries **only** on the training subset.

        This must be called exactly once *before* the first __call__ so that
        validation/test data are encoded with the *same* thresholds with
        which the model was trained.
        """
        returns = train_df[self.base_feature].pct_change().shift(-1)
        # drop the NaN introduced by pct_change / shift so that quantiles are stable
        returns = returns.dropna()

        self.lower_tertile = returns.quantile(1 / 3)
        self.upper_tertile = returns.quantile(2 / 3)

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # ENSURE boundaries are available --------------------------------------------------------
        if self.lower_tertile is None or self.upper_tertile is None:
            raise RuntimeError("Balanced3ClassClassification.fit() must be called before generating targets.")

        return_feature = data[self.base_feature].pct_change()
        next_return = return_feature.shift(-1)

        target = next_return.apply(
            lambda r: 0 if r < self.lower_tertile else (2 if r > self.upper_tertile else 1)
        ).astype(np.float32)

        return target
    

class BinaryClassification:
    def __init__(self, base_feature: str='close'):
        self.base_feature = base_feature

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return_feature = data[self.base_feature].pct_change()
        next_return = return_feature.shift(-1)

        target = next_return.apply(lambda r: r > 0).astype(np.float32)

        return target