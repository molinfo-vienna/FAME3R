from sklearn.utils.estimator_checks import check_estimator

from fame3r import FAME3RScoreEstimator


def test_check_estimator():
    check_estimator(FAME3RScoreEstimator())
