import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, col_indexes):
        self.col_indexes = col_indexes

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        cols_to_drop = [X.columns[i] for i in range(len(self.col_indexes)) if self.col_indexes[i]]
        df = X.drop(cols_to_drop, axis=1)
        return df
