"""
for each era number we want to make a single row with a shit ton of features that describe that era.
then we can cluster on these eras blindly (without looking at targets)
then for each cluster, we have a separate model that only trains on those eras.
then in live, we cluster first, and then pass the results to it's respective model
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers.pca import VarianceSumPCA
import unittest
import timeit


class EraClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, max_length=10000):
        self.n_clusters = n_clusters
        self.one_hot_encoder = None
        self.max_length = max_length
        self.scaler = StandardScaler()
        self.pca = VarianceSumPCA(0.95)

    def generate_features(self, X):
        assert isinstance(X, pd.DataFrame)
        assert "era" in X.columns

        era_feature_rows = []
        eras = []
        for era in set(X["era"]):
            era_df = X[X["era"] == era].drop(columns=["era"])

            means = pd.Series(era_df.mean())
            vars = pd.Series(era_df.var())

            # downsample because corr() takes forevs
            if len(era_df) > self.max_length:
                era_df = era_df.iloc[::(len(era_df)//self.max_length)]

            corrs = era_df.corr()
            upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(np.bool)).values.flatten()
            corrs = pd.Series([u for u in upper if not np.isnan(u)])
            # generate neat features
            neat_features = pd.concat([means, vars, corrs])

            # add the features to a row in a dataframe with the era number
            era_feature_rows.append(neat_features.values.tolist())
            eras.append(era)
        result_df = pd.DataFrame(era_feature_rows)
        return result_df, eras

    def fit(self, X, y=None):
        """
        Make a bunch of features for each era and turn those into per-era rows.
        now we can just cluster with self.num_clusters and then yeet alpha into existence
        """

        result = X.copy()
        self.clusterer = MiniBatchKMeans(self.n_clusters)

        # TODO: for each era make a bunch of features
        start_generate_features = timeit.default_timer()
        feature_df, eras = self.generate_features(X)
        print(f"generate features took: {timeit.default_timer() - start_generate_features }")

        # TODO: fit with MiniBatchKMeans
        scaled = self.scaler.fit_transform(feature_df)
        pca = self.pca.fit_transform(scaled)
        start_cluster_fit = timeit.default_timer()
        cluster_col = self.clusterer.fit(pca)
        print(f"cluster fit took: {timeit.default_timer() - start_cluster_fit}")
        cluster_dict = {}
        for era in set(X["era"]):
            eras_index = eras.index(era)
            if eras_index >= len(feature_df):
                feature_df_copy = feature_df.iloc[eras_index:]
            else:
                feature_df_copy = feature_df.iloc[eras_index:eras_index+1]
            scaled = self.scaler.transform(feature_df_copy)
            pca = self.pca.transform(scaled)
            cluster = self.clusterer.predict(pca)[0]
            cluster_dict[era] = cluster

        # TODO: stick the predicts as a feature onto X
        for era in cluster_dict:
            cluster = cluster_dict[era]
            result.loc[result["era"] == era, "cluster"] = cluster

        self.one_hot_encoder = OneHotEncoder(n_values=self.n_clusters, categorical_features=[len(result.columns)-1])
        self.one_hot_encoder.fit(result)

        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        assert "era" in X.columns

        result = X.copy()

        feature_df, eras = self.generate_features(X)

        cluster_dict = {}
        # TODO: for each era generate features
        for era in set(X["era"]):
            eras_index = eras.index(era)
            if eras_index >= len(feature_df):
                feature_df_copy = feature_df.iloc[eras_index:]
            else:
                feature_df_copy = feature_df.iloc[eras_index:eras_index + 1]
            scaled = self.scaler.transform(feature_df)
            pca = self.pca.transform(scaled)
            cluster = self.clusterer.predict(pca)[0]
            cluster_dict[era] = cluster

        # TODO: stick the predicts as a feature onto X
        for era in cluster_dict:
            cluster = cluster_dict[era]
            result.loc[result["era"] == era, "cluster"] = cluster

        # TODO: now one-hot encode the predicts feature, and return X yeet
        result = pd.DataFrame(self.one_hot_encoder.transform(result).toarray())

        return result

    def get_neigboring_corrs(self, df):
        corrs = []
        for i in range(len(df.columns) - 1):
            c = np.corrcoef(df.iloc[:, i], df.iloc[:, i + 1])
            corrs.append(c[0, 1])
        return corrs


class TestClusterer(unittest.TestCase):
    def test_sanity(self):
        era_1 = np.random.normal(0, 1, [40, 2])
        era_2 = np.random.normal(10, 1, [40, 2])
        era_3 = np.random.normal(-10, 1, [40, 2])

        all_data = np.vstack([era_1, era_2, era_3])
        df = pd.DataFrame(all_data)
        df["era"] = [1]*40 + [2]*40 +[3]*40

        clusterer = EraClusterer(n_clusters=3)
        result = clusterer.fit_transform(df)


if __name__ == "__main__":
    unittest.main()
