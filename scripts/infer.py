# pyright: reportAttributeAccessIssue=false

"""Applies a trained re-implementation of the FAME3 model to unlabeled data.

This script saves the per-atom predictions to a CSV file.
The radius of the atom environment is not part of the hyperparameter search, \
    but can be set by changing the radius argument. Default is 5.
The decision threshold can be changed by changing the threshold argument. Default is 0.3.
The script also computes FAME scores if the -fs flag is set.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import joblib
from CDPL.Chem import (
    Atom,
    BasicMolecule,
    FileSDFMoleculeReader,
    MolecularGraph,
    generateSMILES,
    getStructureData,
)

from fame3r import FAME3RVectorizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Applies a trained re-implementation of the FAME3 model to unlabeled data"
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<Input data file>",
        help="Input data file",
        type=str,
    )
    parser.add_argument(
        "-m",
        dest="model_folder",
        required=True,
        metavar="<Model folder>",
        help="Model folder",
        type=str,
    )
    parser.add_argument(
        "-o",
        dest="out_file",
        required=True,
        metavar="<Output file>",
        help="Output file",
        type=str,
    )
    parser.add_argument(
        "-r",
        dest="radius",
        required=False,
        metavar="<radius>",
        default=5,
        help="Max. atom environment radius in number of bonds",
        type=int,
    )
    parser.add_argument(
        "-t",
        dest="threshold",
        required=False,
        metavar="<binary decision threshold>",
        default=0.3,
        help="Binary decision threshold",
        type=float,
    )
    parser.add_argument(
        "-fs",
        dest="compute_fame_scores",
        action="store_true",
        help="Compute FAME scores (optional)",
    )

    return parser.parse_args()


def extract_som_labels(mol: MolecularGraph) -> list[tuple[Atom, bool]]:
    structure_data = {
        entry.header[2:].split(">")[0]: entry.data for entry in getStructureData(mol)
    }
    som_indices = eval(structure_data["soms"])

    return [(atom, int(atom.index in som_indices)) for atom in mol.getAtoms()]  # pyright:ignore


def main():
    """Application entry point."""
    start_time = datetime.now()

    args = parse_arguments()
    print(f"Radius: {args.radius}")

    som_atoms_labeled: list[tuple[Atom, bool]] = []

    reader = FileSDFMoleculeReader(args.input_file)
    mol = BasicMolecule()
    while reader.read(mol):  # pyright:ignore
        som_atoms_labeled.extend(extract_som_labels(mol))
        mol = BasicMolecule()

    print(f"Data: {reader.getNumRecords()} molecules, {len(som_atoms_labeled)} atoms")

    print("Loading models...")
    classifier = joblib.load(
        Path(args.model_folder) / "random_forest_classifier.joblib"
    )
    if args.compute_fame_scores:
        score_estimator = joblib.load(Path(args.model_folder) / "fame_scorer.joblib")

    print("Extracting features...")

    # Unfortunately, this monkey-patching is required to get CDPKit
    # objects like atoms and molecules into NumPy arrays...
    del Atom.__getitem__

    features = FAME3RVectorizer(radius=args.radius, input="cdpkit").fit_transform(
        [[som_atom] for som_atom, _ in som_atoms_labeled]
    )

    print("Predicting SOMs...")

    predictions = classifier.predict_proba(features)[:, 1]
    predictions_binary = (predictions > args.threshold).astype(int)

    if args.compute_fame_scores:
        print("Computing FAME scores...")
        # Compute the FAME scores for the test set, excluding the
        # physicochemical and topological descriptors
        fame_scores = score_estimator.predict(features[:, :-14])
        print(fame_scores.shape)

    with Path(args.out_file).open("w", encoding="UTF-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["smiles", "atom_id", "y_pred", "y_true", "y_prob"]
            + (["fame_score"] if args.compute_fame_scores else []),
        )
        writer.writeheader()

        for i in range(len(som_atoms_labeled)):
            writer.writerow(
                {
                    "smiles": generateSMILES(som_atoms_labeled[i][0].molecule),
                    "atom_id": som_atoms_labeled[i][0].index,
                    "y_pred": predictions_binary[i],
                    "y_true": som_atoms_labeled[i][1],
                    "y_prob": predictions[i],
                }
                | ({"fame_score": fame_scores[i]} if args.compute_fame_scores else {})
            )

    print(f"Predictions saved to {args.out_file}")
    print("Finished in:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
