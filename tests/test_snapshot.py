import sys

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from syrupy.assertion import SnapshotAssertion

from fame3r import FAME3RScoreEstimator, FAME3RVectorizer

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
    with np.printoptions(threshold=sys.maxsize):
        yield


DESCRIPTOR_COMBINATIONS = [
    ["fingerprint"],
    ["counts"],
    ["physicochemical"],
    ["topological"],
    ["fingerprint", "physicochemical", "topological"],
    ["counts", "physicochemical", "topological"],
]


@pytest.mark.parametrize("radius", [0, 3, 5])
@pytest.mark.parametrize("descriptors", DESCRIPTOR_COMBINATIONS)
def test_descriptors(snapshot: SnapshotAssertion, radius, descriptors):
    inputs = [[smiles] for smiles in som_marked_smiles_test]

    vectorizer = FAME3RVectorizer(radius=radius, output=descriptors).fit()

    assert snapshot == vectorizer.transform(inputs)


@pytest.mark.parametrize("radius", [0, 3, 5])
@pytest.mark.parametrize("descriptors", DESCRIPTOR_COMBINATIONS)
def test_descriptor_names(snapshot: SnapshotAssertion, radius, descriptors):
    vectorizer = FAME3RVectorizer(radius=radius, output=descriptors).fit()

    assert snapshot == vectorizer.get_feature_names_out()


def test_score(snapshot: SnapshotAssertion):
    train_inputs = [[smiles] for smiles in som_marked_smiles_train]
    test_inputs = [[smiles] for smiles in som_marked_smiles_test]

    pipeline = make_pipeline(
        FAME3RVectorizer(output=["fingerprint"]),
        FAME3RScoreEstimator(),
    ).fit(train_inputs)

    assert snapshot == pipeline.predict(test_inputs)
