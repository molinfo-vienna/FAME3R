import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["FAME3RScoreEstimator"]


class FAME3RScoreEstimator(BaseEstimator, _SetOutputMixin):
    """Computes the FAME score for a set of features.

    The FAME score is defined as the mean Tanimoto similarity of the
    feature vector to the ``n`` closest vectors in the training set.

    It is originally only defined for binary feature vectors, but this
    implementation can also be applied to any numeric features. This
    is because we use an extended definition of Tanimoto similarity as
    described in :func:`~sklearn.metrics.jaccard_score`.

    Parameters
    ----------
    n_neighbors : int, default=3

        Number of nearest neigbors to consider during FAME score
        calculation. Defaults to 3, as defined in the original paper.


    Examples
    --------
    >>> from fame3r import FAME3RVectorizer, FAME3RScoreEstimator
    >>> from sklearn.pipeline import make_pipeline
    >>> pipeline = make_pipeline(
    >>>    FAME3RVectorizer(output=["fingerprint"]),
    >>>    FAME3RScoreEstimator()
    >>> ).fit([["CC[C:1]"], ["CC[N:1]"], ["CC[O:1]"]])
    >>> pipeline.predict([["[C:1]CC"]])
    array([0.66666667])

    """

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors: int = n_neighbors

    def fit(self, X, y=None):
        """Fit the estimator to the training set of known samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : (ignored)
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            FAME3RVectorizer class instance.
        """

        X = validate_data(
            self,
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
        """Compute the FAME score of the given samples.

        Parameters
        ----------

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Query data.

        Returns
        -------

        y : ndarray of shape (n_samples,)
            The predicted FAME scores.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            reset=False,
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
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        return ["FAME3RScore"]
