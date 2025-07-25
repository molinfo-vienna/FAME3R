from typing import Any

import numpy as np
import numpy.typing as npt
from CDPL.Chem import AtomProperty, parseSMILES  # type:ignore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.validation import check_array, check_is_fitted

from fame3r._internal import DescriptorGenerator, MoleculeProcessor

__all__ = ["FAME3RVectorizer"]


class FAME3RVectorizer(BaseEstimator, TransformerMixin, _SetOutputMixin):
    def __init__(self, radius: int = 5) -> None:
        self.radius: int = radius

    def fit(self, X: Any = None, y: None = None):
        example_mol = parseSMILES("C")
        MoleculeProcessor.perceive_mol(example_mol)

        self.inner_ = DescriptorGenerator(radius=self.radius)
        self.feature_names_ = self.inner_.generate_descriptors(
            example_mol.getAtom(0), example_mol
        )[0]

        return self

    def transform(self, X: Any):
        check_is_fitted(self)
        X = check_array(
            X,
            dtype="str",
            ensure_2d=True,
            ensure_min_samples=0,
            estimator=FAME3RVectorizer,
        )

        return np.apply_along_axis(lambda row: self.transform_one(row[0]), 1, X)

    def transform_one(self, X: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self)

        if not isinstance(X, str):
            return self._empty_value()

        cdpkit_marked_educt = parseSMILES(X)
        MoleculeProcessor.perceive_mol(cdpkit_marked_educt)

        som_atoms_unordered = {
            atom.getProperty(AtomProperty.ATOM_MAPPING_ID): atom
            for atom in cdpkit_marked_educt.getAtoms()
            if atom.getProperty(AtomProperty.ATOM_MAPPING_ID)
        }

        # TODO: Add warning or error
        if len(som_atoms_unordered) != 1:
            return self._empty_value()

        descriptors = [
            self.inner_.generate_descriptors(
                som_atoms_unordered[i], cdpkit_marked_educt
            )[1]
            for i in sorted(som_atoms_unordered.keys())
        ]

        return np.asarray(descriptors, dtype=np.float64)[0]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)

        return self.feature_names_

    def _empty_value(self):
        check_is_fitted(self)

        return np.full(len(self.feature_names_), np.nan)


# check_estimator(Fame3RVectorizer())
