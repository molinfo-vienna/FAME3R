import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from syrupy.assertion import SnapshotAssertion

from fame3r.descriptors import FAME3RVectorizer
from fame3r.score import FAME3RScoreEstimator

som_marked_smiles_train = [
    "O=Cc1ccc([O:1])c(OC)c1",
    "CC(=O)NCCC1=CNc2c1cc(O[C:1])cc2",
    "CN1CCC[C@H]1c2ccc[n:1]c2",
]

som_marked_smiles_test = [
    "O=Cc1ccc([O:1])c(OC)c1",
    "CC(=O)NCCC1=CNc2c1cc(O[C:1])cc2",
    "CN1CCC[C@H]1c2ccc[n:1]c2",
]


@pytest.fixture(autouse=True)
def full_numpy_output():
    # This is required so that the full arrays are compared and saved
    # as snapshots, not just summaries.
    with np.printoptions(threshold=np.inf):  # pyright:ignore
        yield


def test_descriptors(snapshot: SnapshotAssertion):
    inputs = [[smiles] for smiles in som_marked_smiles_test]

    vectorizer = FAME3RVectorizer().fit()

    assert snapshot == vectorizer.transform(inputs)


def test_descriptor_names(snapshot: SnapshotAssertion):
    vectorizer = FAME3RVectorizer().fit()

    assert snapshot == vectorizer.get_feature_names_out()


def test_score(snapshot: SnapshotAssertion):
    train_inputs = [[smiles] for smiles in som_marked_smiles_train]
    test_inputs = [[smiles] for smiles in som_marked_smiles_test]

    pipeline = make_pipeline(
        FAME3RVectorizer(),
        FAME3RScoreEstimator(),
    ).fit(train_inputs)

    assert snapshot == pipeline.predict(test_inputs)
