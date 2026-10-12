"""Feature engineering + preprocessing (fit on the training split only)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

ENGINEERED_NUMERIC = [
    "hour", "day_of_week", "month", "is_weekend",
    "Amount_to_Income", "Amount_to_Balance", "occupation_freq", "merchant_freq",
]
NUMERIC_FEATURES = C.NUMERIC_INPUTS + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = C.CATEGORICAL_INPUTS


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Derives time, ratio and frequency-encoded features from raw input columns."""

    def fit(self, X, y=None):
        self.occ_freq_ = X["Customer_Occupation"].value_counts(normalize=True).to_dict()
        self.mer_freq_ = X["Merchant_Name"].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        d = pd.to_datetime(X[C.DATE_COL])
        X["hour"] = d.dt.hour
        X["day_of_week"] = d.dt.dayofweek
        X["month"] = d.dt.month
        X["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
        X["Amount_to_Income"] = X["Transaction_Amount"] / (X["Customer_Income"] + 1.0)
        X["Amount_to_Balance"] = X["Transaction_Amount"] / (X["Account_Balance"] + 1.0)
        X["occupation_freq"] = X["Customer_Occupation"].map(self.occ_freq_).fillna(0.0)
        X["merchant_freq"] = X["Merchant_Name"].map(self.mer_freq_).fillna(0.0)
        return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> Pipeline:
    ct = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("fe", FeatureBuilder()), ("ct", ct)])


def feature_names(pre: Pipeline):
    return list(pre.named_steps["ct"].get_feature_names_out())
