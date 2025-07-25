import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ["FAME3RScoreEstimator"]


class FAME3RScoreEstimator(BaseEstimator, _SetOutputMixin):
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors: int = n_neighbors

    def fit(self, X, y=None):
        X = check_array(
            X,
            dtype="numeric",
            ensure_2d=True,
            ensure_min_samples=self.n_neighbors,
            estimator=FAME3RScoreEstimator,
        )

        # We use ball tree here because it supports the Jaccard
        # metric, which is equivalent to the Tanimoto distance.
        self.nearest_neighbors_ = BallTree(X, metric="jaccard")

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(
            X,
            dtype="numeric",
            ensure_2d=True,
            ensure_min_samples=0,
            estimator=FAME3RScoreEstimator,
        )

        distances, _ = self.nearest_neighbors_.query(
            X, k=self.n_neighbors, return_distance=True
        )
        similarities = 1 - distances

        return np.mean(similarities, axis=1)

    def get_feature_names_out(self, input_features=None):
        return ["FAME3RScore"]
