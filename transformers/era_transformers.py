from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from sklearn.preprocessing import StandardScaler

def era_to_number(s):
    result = ''.join(i for i in s if i.isdigit())
    if len(result):
        return int(result)
    else:
        return 199

class EraToNumber(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if "era" in X.columns:
            X.loc[:, "era"] = X["era"].apply(era_to_number)
        return X

class EraDropper(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        return X.drop(["era"], axis=1, errors="ignore")

# scale by era instead of overall
class EraScaler(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if "era" in X.columns:
            eras = set(X["era"])
            scaled_eras = []
            for era in eras:
                era_scaler = StandardScaler()
                era_df = X[X["era"] == era]
                era_df_scaled = pd.DataFrame(era_scaler.fit_transform(era_df), index=era_df.index, columns=era_df.columns)
                scaled_eras.append(era_df_scaled)
            return pd.concat(scaled_eras)
        else:
            return X


# number which row of this era we're in
class EraNumberer(BaseEstimator, TransformerMixin):
    def init(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if "era" in X.columns:
            eras = set(X["era"])
            era_dfs = []
            for era in eras:
                era_df = X[X["era"] == era].copy()
                era_df["num_col"] = [i for i in range(len(era_df))]
                era_dfs.append(era_df)
            return pd.concat(era_dfs)
        else:
            return X
