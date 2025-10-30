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

        return self

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


class MeanReturnSignClassification:
    """Binary target based on the *sign* of the **mean** return over the next *horizon* periods.

    For each timestamp *t* the class label is defined as::

        label_t = 1  if  mean(R[t+1], ..., R[t+horizon]) > 0
                 0  otherwise

    where ``R`` denotes the simple return series of the selected ``base_feature``.

    The last ``horizon`` samples of the series will have no sufficient look-ahead
    information available and therefore yield ``NaN`` mean returns; these are
    encoded as the negative class (0) to stay consistent with the behaviour of
    :class:`BinaryClassification` which maps ``NaN`` to ``0`` as well.
    """

    def __init__(self, horizon: int = 5, base_feature: str = "close"):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon
        self.base_feature = base_feature

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # Compute simple returns of the chosen price/feature
        returns = data[self.base_feature].pct_change()

        # Mean of returns for the *next* `horizon` periods.
        # 1. shift(-1) aligns the *next* return with the current timestamp
        # 2. rolling(window=horizon) takes a trailing window on the shifted
        #    series (i.e. a forward-looking window on the original series)
        # 3. shift(-(horizon-1)) realigns the window's last element back to the
        #    current timestamp so that the label at *t* depends only on data
        #    strictly after *t*.
        future_mean_return = (
            returns.shift(-1)
            .rolling(window=self.horizon, min_periods=self.horizon)
            .mean()
            .shift(-(self.horizon - 1))
        )

        # Binary label: 1 if positive mean, else 0 (includes NaN -> False)
        target = future_mean_return.apply(lambda r: r > 0).astype(np.float32)

        return target


class Balanced5ClassClassification:
    """Balanced 5-class classification target based on next-period return.

    The continuous distribution of *next* returns is partitioned into five
    equally sized buckets (quintiles). Each bucket is mapped to a numerical
    encoding according to the scheme

        < 20% quantile              → 0.00
        20%–40% quantile interval   → 0.25
        40%–60% quantile interval   → 0.50
        60%–80% quantile interval   → 0.75
        > 80% quantile              → 1.00

    The quantile thresholds are *fitted only on the training subset* to avoid
    look-ahead bias. Once fitted, the same thresholds are used for validation
    and test data.
    """

    def __init__(self, horizon=5, class_values: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0], base_feature: str = "close"):
        self.horizon = horizon
        self.class_values = class_values
        self.base_feature = base_feature
        # Quantile thresholds initialised to None until `fit` is called
        self.q20: float | None = None
        self.q40: float | None = None
        self.q60: float | None = None
        self.q80: float | None = None

    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # Fitting (training-set only)
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        """Compute quintile boundaries from *train_df* only."""
        returns = self._calculate_returns(train_df).dropna()
        
        self.q20 = returns.quantile(0.20)
        self.q40 = returns.quantile(0.40)
        self.q60 = returns.quantile(0.60)
        self.q80 = returns.quantile(0.80)

        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # Ensure `fit` has been executed
        if None in {self.q20, self.q40, self.q60, self.q80}:  # type: ignore[comparison-overlap]
            raise RuntimeError("Balanced5ClassClassification.fit() must be called before generating targets.")

        returns = self._calculate_returns(data)

        def encode(r: float | np.float64):
            if r < self.q20:
                return self.class_values[0]
            elif r < self.q40:
                return self.class_values[1]
            elif r < self.q60:
                return self.class_values[2]
            elif r < self.q80:
                return self.class_values[3]
            else:
                return self.class_values[4]

        target = returns.apply(encode).astype(np.float32)
        return target


class FutureMeanReturnClassification(Balanced5ClassClassification):
    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        returns = data[self.base_feature].pct_change()
        future_mean_return = (
            returns.shift(-self.horizon)
            .rolling(window=self.horizon, min_periods=self.horizon)
            .mean()
        )
        return future_mean_return


class FutureHorizonReturnClassification(Balanced5ClassClassification):
    """Balanced 5-class target based on the total return over the next horizon.

    For each timestamp t, computes the cumulative simple return across the next
    `horizon` periods as (P[t+horizon] / P[t]) - 1 and then encodes it into
    quintile-based classes using the training-set thresholds.
    """

    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        prices = data[self.base_feature]
        future_total_return = prices.shift(-self.horizon) / prices - 1.0
        return future_total_return