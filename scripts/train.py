# pyright: reportAttributeAccessIssue=false


"""Trains a re-implementation of the FAME3 model.

This script trains a random forest classifier using the FAME descriptors.
The model is saved in the as a joblib file.

The hyperparameters of the model and the radius of the atom environment are not optimized in this script (see cv_hp_search.py).
The hyperparameters can be set in the RandomForestClassifier constructor.
The radius can be set by changing the radius command line argument (default is 5).
"""

import argparse
from datetime import datetime
from pathlib import Path

import joblib
from CDPL.Chem import (
    Atom,
    BasicMolecule,
    FileSDFMoleculeReader,
    MolecularGraph,
    getStructureData,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from fame3r import FAME3RScoreEstimator, FAME3RVectorizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trains a re-implementation of the FAME3 model."
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<training data file>",
        help="Training data file",
        type=str,
    )
    parser.add_argument(
        "-o",
        dest="out_folder",
        required=True,
        metavar="<Output folder>",
        help="Output location",
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
        "-n",
        dest="n_estimators",
        required=False,
        metavar="<n_estimators>",
        default=100,
        help="Number of trees in the forest",
        type=int,
    )
    parser.add_argument(
        "-md",
        dest="max_depth",
        required=False,
        metavar="<max_depth>",
        default=None,
        help="Maximum depth of the tree (int, None)",
        type=lambda x: None if x == "None" else int(x),
    )
    parser.add_argument(
        "-mss",
        dest="min_samples_split",
        required=False,
        metavar="<min_samples_split>",
        default=2,
        help="Minimum number of samples required to split an internal node",
        type=int,
    )
    parser.add_argument(
        "-msl",
        dest="min_samples_leaf",
        required=False,
        metavar="<min_samples_leaf>",
        default=1,
        help="Minimum number of samples required to be at a leaf node",
        type=int,
    )
    parser.add_argument(
        "-mf",
        dest="max_features",
        required=False,
        metavar="<max_features>",
        default="sqrt",
        help="Number of features to consider when looking for the best split (sqrt, log2, None)",
        type=lambda x: None if x == "None" else str(x),
    )
    parser.add_argument(
        "-c",
        dest="class_weight",
        required=False,
        metavar="<class_weight>",
        default="balanced",
        help="Class weight (balanced, balanced_subsample, None)",
        type=lambda x: None if x == "None" else str(x),
    )

    return parser.parse_args()


def extract_som_labels(mol: MolecularGraph) -> list[tuple[Atom, bool]]:
    structure_data = {
        entry.header[2:].split(">")[0]: entry.data for entry in getStructureData(mol)
    }
    som_indices = eval(structure_data["soms"])

    return [(atom, int(atom.index in som_indices)) for atom in mol.getAtoms()]  # pyright:ignore


def main():
    start_time = datetime.now()

    args = parse_arguments()
    print(f"Radius: {args.radius}")

    print("Loading training data...")

    som_atoms_labeled: list[tuple[Atom, bool]] = []

    reader = FileSDFMoleculeReader(args.input_file)
    mol = BasicMolecule()
    while reader.read(mol):  # pyright:ignore
        som_atoms_labeled.extend(extract_som_labels(mol))
        mol = BasicMolecule()

    print(f"Training data: {len(som_atoms_labeled)} data points")

    print("Training random forest model...")

    # Unfortunately, this monkey-patching is required to get CDPKit
    # objects like atoms and molecules into NumPy arrays...
    del Atom.__getitem__

    rf_pipeline = make_pipeline(
        FAME3RVectorizer(radius=args.radius, input="cdpkit"),
        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight=args.class_weight,
            random_state=42,
        ),
    ).fit(
        [[som_atom] for som_atom, _ in som_atoms_labeled],
        [label for _, label in som_atoms_labeled],
    )

    print("Training FAME score model...")

    # TODO: fingerprint only vectorizer
    positive_labeled_fingerprints = FAME3RVectorizer(
        radius=args.radius, input="cdpkit"
    ).fit_transform([[som_atom] for som_atom, label in som_atoms_labeled if label])[
        :, :-14
    ]
    scorer = FAME3RScoreEstimator().fit(positive_labeled_fingerprints)

    print("Saving models...")

    Path(args.out_folder).mkdir(exist_ok=True)
    joblib.dump(
        rf_pipeline.named_steps["randomforestclassifier"],
        Path(args.out_folder) / "random_forest_classifier.joblib",
    )
    joblib.dump(scorer, Path(args.out_folder) / "fame_scorer.joblib")

    print("Finished in:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
