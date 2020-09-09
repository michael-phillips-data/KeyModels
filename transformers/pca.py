from sklearn.decomposition import IncrementalPCA, PCA
import numpy as np


class VarianceSumPCA(IncrementalPCA):
    def __init__(self, var_limit, max_components=50):
        super(VarianceSumPCA, self).__init__()
        self.var_limit = var_limit
        self.n_components = None
        self.max_components = max_components

    def fit(self, X, y=None):
        # find number of features
        self.max_components = min(self.max_components, len(X.T), len(X))
        temp_PCA = IncrementalPCA(n_components=self.max_components)
        temp_PCA.fit(X)
        variance_list = temp_PCA.explained_variance_ratio_
        var_sum = 0
        self.n_components = 0
        for v in variance_list:
            var_sum += v
            self.n_components += 1
            if var_sum > self.var_limit and self.n_components >= 2:
                break
        return super().fit(X)

    def transform(self, X):
        return super().transform(X=X)


class MinorPCA(PCA):
    def __init__(self, minor_components, n_components=None, svd_solver="randomized"):
        super(MinorPCA, self).__init__(svd_solver=svd_solver)
        self.minor_components = minor_components

    def transform(self, X):
        pred = super().transform(X)
        return pred[:, -self.minor_components:]

    def fit_transform(self, X, y=None):
         self.fit(X)
         return self.transform(X)

class GenesisPCA(VarianceSumPCA):
    def __init__(self, var_limit, minor_components=None):
        super(GenesisPCA, self).__init__(var_limit=var_limit)
        self.minor_components = minor_components
        self.minor_pca = MinorPCA(minor_components=minor_components)

    def fit(self, X, y=None):
        if self.minor_components:
            self.minor_pca.fit(X)
        return super().fit(X)

    def transform(self, X, y=None):
        if self.minor_components:
            minor_preds = self.minor_pca.transform(X)
            var_sum_preds = super().transform(X)
            return np.hstack((var_sum_preds, minor_preds))
        else:
            return super().transform(X)