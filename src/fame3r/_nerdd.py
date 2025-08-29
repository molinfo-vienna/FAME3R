import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, get_args

import joblib
import numpy as np
from CDPL.Chem import parseSMILES  # pyright:ignore[reportAttributeAccessIssue]
from nerdd_module import Model
from nerdd_module.preprocessing import Sanitize
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles
from scipy.stats import entropy
from sklearn.pipeline import Pipeline, make_pipeline

from fame3r import FAME3RVectorizer

MODEL_DIRECTORY = Path(os.environ["FAME3R_MODEL_DIRECTORY"])
THRESHOLD = 0.3


MetabolismSubset = Literal["all", "phase1", "phase2", "cyp"]


@dataclass
class Models:
    classifier: Pipeline
    fame_scorer: Pipeline


class FAME3RModel(Model):
    def __init__(self, preprocessing_steps=[Sanitize()]):
        super().__init__(preprocessing_steps)

        self._vectorizer = FAME3RVectorizer(radius=5, input="cdpkit").fit()
        self._models: dict[MetabolismSubset, Models] = {}

        for phase in get_args(MetabolismSubset):
            self._models[phase] = Models(
                classifier=make_pipeline(
                    FAME3RVectorizer(input="cdpkit").fit(),
                    joblib.load(
                        MODEL_DIRECTORY / phase / "random_forest_classifier.joblib",
                    )
                ),
                fame_scorer=make_pipeline(
                    FAME3RVectorizer(input="cdpkit", output=["fingerprint"]).fit(),
                    joblib.load(
                        MODEL_DIRECTORY / phase / "fame3r_score_estimator.joblib",
                    ),
                ),
            )

    def _predict_mols(
        self,
        mols: list[Mol],
        metabolism_subset: MetabolismSubset = "all",
        fame_score: bool = False,
        shannon_entropy: bool = False,
    ) -> Iterable[dict]:
        models = self._models[metabolism_subset]

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

        predictions = models.classifier.predict_proba(atom_array)[:, 1]

        if fame_score:
            fame_scores = models.fame_scorer.predict(atom_array)
        else:
            fame_scores = np.full_like(predictions, np.nan)

        if shannon_entropy:
            shannon_entropies = entropy([predictions, 1-predictions], base=2)
        else:
            shannon_entropies = np.full_like(predictions, np.nan)

        for (atom, mol_id), probability, fame_score, shannon_entropy in zip(
            atoms, predictions, fame_scores, shannon_entropies, strict=True
        ):
            yield {
                "mol_id": mol_id,
                "atom_id": atom.index,
                "prediction": probability,
                "prediction_binary": probability > THRESHOLD,
                "fame_score": fame_score,
                "shannon_entropy": shannon_entropy,
            }
