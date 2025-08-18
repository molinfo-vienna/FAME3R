import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from CDPL.Chem import parseSMILES  # pyright:ignore[reportAttributeAccessIssue]
from nerdd_module import Model
from nerdd_module.preprocessing import Sanitize
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles
from sklearn.pipeline import make_pipeline

from fame3r import FAME3RVectorizer

MODEL_DIRECTORY = Path(os.environ["FAME3R_MODEL_DIRECTORY"])
THRESHOLD = 0.3


class FAME3RModel(Model):
    def __init__(self, preprocessing_steps=[Sanitize()]):
        super().__init__(preprocessing_steps)

        self._model = make_pipeline(
            FAME3RVectorizer(radius=5, input="cdpkit").fit(),
            joblib.load(MODEL_DIRECTORY / "random_forest_classifier.joblib"),
        )

    def _predict_mols(self, mols: list[Mol]) -> Iterable[dict]:
        cdpkit_mols = [parseSMILES(MolToSmiles(mol)) for mol in mols]
        atoms = [
            (atom, mol_id)
            for mol_id, mol in enumerate(cdpkit_mols)
            for atom in mol.atoms
        ]

        # This is required to get CDPKit atoms, which define a __getitem__
        # method, into NumPy arrays. Assigning to an existing array prevents
        # NumPy from trying to access the "items" of an Atom.
        atom_array = np.empty((len(atoms), 1), dtype=object)
        atom_array[:, 0] = [atom for atom, _ in atoms]

        predictions = self._model.predict_proba(atom_array)[:, 1]

        for (atom, mol_id), probability in zip(atoms, predictions):
            yield {
                "mol_id": mol_id,
                "atom_id": atom.index,
                "prediction": probability,
                "prediction_binary": probability > THRESHOLD,
            }
