from typing import Literal

import numpy as np
import numpy.typing as npt
from CDPL.Chem import Atom, AtomProperty, parseSMILES  # pyright:ignore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.validation import check_is_fitted, validate_data

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
    """Transforms atom environments into FAME3-like features.

    Parameters
    ----------
    radius : int, default=5
        Radius used for circular fingerprint generation.

    input : {"smiles", "cdpkit"}, default="smiles"

        Format of the provided atoms and their environment. Can be either
        "smiles" to accept SMILES codes with a single atom mapping number each
        specifying the atom used to generate descriptors, or "cdpkit" to accept
        CDPKit Atom objects.

    output : list of {"fingerprint", "physicochemical", "topological"}

        Which molecular descriptors to generate, and in which order. Defaults to
        generating all supported descriptors.

    Examples
    --------
    >>> from fame3r import FAME3RVectorizer
    >>> FAME3RVectorizer().fit_transform([["CC[C:1]"]])
    array([[1., 0., 0., ..., 2., 0., 1.]], shape=(1, 5006))
    """

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
        """Set up the vectorizer for usage.

        Parameters
        ----------
        X : (ignored)
            Ignored parameter.

        y : (ignored)
            Ignored parameter.

        Returns
        -------
        self : object
            FAME3RVectorizer class instance.
        """
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
        """Transform atom environments to feature matrix.

        Samples are provided as a 2d array with shape (n_samples, 1).

        The single provided feature should either be a SMILES string or a CPDKit
        Atom, depending on the value of ``input`` given to the constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input samples.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """

        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            reset=False,
            dtype="object",
            ensure_2d=True,
            ensure_min_samples=0,
            estimator=FAME3RVectorizer,
        )

        return np.apply_along_axis(lambda row: self.transform_one(row), 1, X)

    def transform_one(self, X) -> npt.NDArray[np.floating]:
        """Transform a single atom environment to feature vector.

        The sample is provided as an 1d array with shape (1,).

        The single provided feature should either be a SMILES string or a CPDKit
        Atom, depending on the value of ``input`` given to the constructor.

        Parameters
        ----------
        X : array-like of shape (1,)
            Input sample.

        Returns
        -------
        X_new : ndarray array of shape (n_features_new,)
            Feature vector.
        """

        check_is_fitted(self)

        if len(X) != 1:
            ValueError(
                f"Found array with {len(X)} feature(s) while 1 feature is required."
            )

        if self.input == "smiles":
            if not isinstance(X[0], str):
                raise ValueError(
                    "must pass atom encoded as a SMILES string + mapping number"
                )
            som_atoms = _extract_marked_atoms(X[0])
        elif self.input == "cdpkit":
            if not isinstance(X[0], Atom):
                raise ValueError("must pass atom encoded as a CDPKit Atom")
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

        return np.concatenate(descriptors, dtype=float)

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
