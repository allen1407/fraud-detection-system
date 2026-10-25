"""Feature engineering + preprocessing (fit on the training split only)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

ENGINEERED_NUMERIC = ["hour", "is_night", "log_amount"]
NUMERIC_FEATURES = C.PCA_INPUTS + ["Amount"] + ENGINEERED_NUMERIC


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Derives time-of-day and log-amount features; raw Time (seconds since the window start) is not used."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["hour"] = (X[C.TIME_COL].astype(float) // 3600) % 24
        X["is_night"] = (X["hour"] < 6).astype(int)
        X["log_amount"] = np.log1p(X["Amount"].astype(float))
        return X[NUMERIC_FEATURES]


def build_preprocessor() -> Pipeline:
    ct = ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES)])
    return Pipeline([("fe", FeatureBuilder()), ("ct", ct)])


def feature_names(pre: Pipeline):
    return list(pre.named_steps["ct"].get_feature_names_out())
