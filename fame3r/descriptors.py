from typing import Literal

import numpy as np
import numpy.typing as npt
from CDPL.Chem import Atom, AtomProperty, parseSMILES  # pyright:ignore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.validation import check_array, check_is_fitted

from fame3r._internal import (
    PHYSICOCHEMICAL_DESCRIPTOR_NAMES,
    TOPOLOGICAL_DESCRIPTOR_NAMES,
    generate_fingerprint_names,
    generate_fingerprints,
    generate_physicochemical_descriptors,
    generate_topological_descriptors,
    prepare_mol,
)

__all__ = ["FAME3RVectorizer"]


class FAME3RVectorizer(BaseEstimator, TransformerMixin, _SetOutputMixin):
    def __init__(
        self,
        *,
        radius: int = 5,
        input: Literal["smiles", "cdpkit"] = "smiles",
        output: list[Literal["fingerprint", "physicochemical", "topological"]] = [
            "fingerprint",
            "physicochemical",
            "topological",
        ],
    ) -> None:
        self.radius = radius
        self.input = input
        self.output = output

    def fit(self, X=None, y=None):
        self.n_features_in_ = 1
        self.feature_names_ = []

        for subset in self.output:
            if subset == "fingerprint":
                self.feature_names_.extend(generate_fingerprint_names(self.radius))
            elif subset == "physicochemical":
                self.feature_names_.extend(PHYSICOCHEMICAL_DESCRIPTOR_NAMES)
            elif subset == "topological":
                self.feature_names_.extend(TOPOLOGICAL_DESCRIPTOR_NAMES)

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(
            X,
            dtype="object",
            ensure_2d=True,
            ensure_min_samples=0,
            estimator=FAME3RVectorizer,
        )

        return np.apply_along_axis(lambda row: self.transform_one(row), 1, X)

    def transform_one(self, X) -> npt.NDArray[np.float64]:
        check_is_fitted(self)

        if len(X) != 1:
            ValueError(
                f"Found array with {len(X)} feature(s) while 1 feature is required."
            )

        if self.input == "smiles":
            if not isinstance(X[0], str):
                raise ValueError("must pass SOM encoded as a SMILES string")
            som_atoms = _extract_marked_atoms(X[0])
        elif self.input == "cdpkit":
            if not isinstance(X[0], Atom):
                raise ValueError("must pass SOM encoded as a CDPKit atom")
            som_atoms = [X[0]]
        else:
            raise ValueError(f"unsupported input type: {self.input}")

        if len(som_atoms) != 1:
            raise ValueError(f"only one SOM atom per sample is supported: {X}")

        descriptors = []

        prepare_mol(som_atoms[0].molecule)

        for subset in self.output:
            if subset == "fingerprint":
                descriptors.append(
                    generate_fingerprints(
                        som_atoms[0], som_atoms[0].molecule, radius=self.radius
                    )
                )
            elif subset == "physicochemical":
                descriptors.append(
                    generate_physicochemical_descriptors(
                        som_atoms[0], som_atoms[0].molecule
                    ).round(4)
                )
            elif subset == "topological":
                descriptors.append(
                    generate_topological_descriptors(
                        som_atoms[0], som_atoms[0].molecule
                    ).round(4)
                )

        return np.concat(descriptors, dtype=float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)

        return self.feature_names_


def _extract_marked_atoms(smiles: str) -> list[Atom]:
    marked_mol = parseSMILES(smiles)

    som_atoms_unordered: dict[int, Atom] = {
        atom.getProperty(AtomProperty.ATOM_MAPPING_ID): atom
        for atom in marked_mol.getAtoms()
        if atom.getProperty(AtomProperty.ATOM_MAPPING_ID)
    }

    return [som_atoms_unordered[i] for i in sorted(som_atoms_unordered.keys())]
