from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
import unittest
from sklearn.pipeline import FeatureUnion


class DummyTransformer(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.loc[:, ~X.columns.duplicated()]


# feature for variance in the row
class RmsTransformer(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        if "era" in X.columns:
            df = X.drop(["era"], axis=1)
        else:
            df = pd.DataFrame(X)
        feature = pd.DataFrame(df).std(axis=1)
        df["rms"] = feature
        return pd.DataFrame(df["rms"], columns=["rms"])


class TrendTransformer(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        if "era" in X.columns:
            df = X.drop(["era"], axis=1)
        else:
            df = pd.DataFrame(X)
        num_cols = len(df.columns)
        comparison = pd.Series([i for i in range(num_cols)], index=df.columns)
        df["trend"] = df.apply(lambda row: row.corr(comparison, method="spearman"), axis=1)
        return pd.DataFrame(df["trend"], columns=["trend"])


class AutoCorrTransformer(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        if "era" in X.columns:
            df = X.drop(["era"], axis=1)
        else:
            df = pd.DataFrame(X)
        df["auto"] = df.apply(lambda row: row.shift(1).corr(row), axis=1)
        return pd.DataFrame(df["auto"], columns=["auto"])


class TestTransformers(unittest.TestCase):
    def test_rms(self):
        a1 = np.random.normal(0, 1, [100, 10])
        a2 = np.random.normal(0, 2, [100, 10])
        a = np.vstack((a1, a2))
        df = pd.DataFrame(a)
        transformer = RmsTransformer()
        df["rms"] = transformer.transform(df)

    def test_trend(self):
        a1 = np.random.normal(0, 1, [10, 10])
        a2 = np.random.normal(0, 1, [10, 10])
        trend = [2**i for i in range(10)]  # to differentiate from correlation coef, make sure we use spearman
        a2 = a2 + trend
        a = np.vstack((a1, a2))
        df = pd.DataFrame(a)
        transformer = TrendTransformer()
        result = transformer.transform(df)

    def test_auto(self):
        a1 = np.random.normal(0, 1, [10, 10])
        a2 = np.random.normal(0, 1, [10, 10])
        trend = [1,2,3,4,5,6,5,4,3,2]  # to differentiate from correlation coef, make sure we use spearman
        a2 = a2 + trend
        a = np.vstack((a1, a2))
        df = pd.DataFrame(a)
        transformer = AutoCorrTransformer()
        df["trend_score"] = transformer.transform(df)

    def test_feature_union(self):
        a1 = np.random.normal(0, 1, [10, 10])
        a2 = np.random.normal(0, 1, [10, 10])
        trend = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2]  # to differentiate from correlation coef, make sure we use spearman
        a2 = a2 + trend
        a = np.vstack((a1, a2))
        df = pd.DataFrame(a)
        features = [('dummy', DummyTransformer())]
        # features.append(('rms', RmsTransformer()))
        # features.append(('autocorr', AutoCorrTransformer()))
        # features.append(('trend_transform', TrendTransformer()))
        fu = FeatureUnion(features)
        result = fu.transform(df)
