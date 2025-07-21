"""Trains a re-implementation of the FAME3 model.

This script trains a random forest classifier using the FAME descriptors.
The model is saved in the output folder as a joblib file.
The hyperparameters of the model and the radius of the atom environment \
    are not optimized in this script (see cv_hp_search.py). \
        The hyperparameters can be set in the RandomForestClassifier constructor. \
            The radius can be set by changing the radius command line argument (default is 5).

"""

import argparse
import os
import sys
from datetime import datetime

from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from fame3r import FAMEDescriptors


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
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


def main():
    """Application entry point."""
    start_time = datetime.now()

    args = parse_arguments()
    print(f"Radius: {args.radius}")

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print("The new output folder is created.")

    print("Computing descriptors...")

    descriptors_generator = FAMEDescriptors(args.radius)
    (
        mol_num_ids,
        _mol_ids,
        _atom_ids,
        som_labels,
        descriptors,
    ) = descriptors_generator.compute_fame_descriptors(
        args.input_file, args.out_folder, has_soms=True
    )

    print(f"Training data: {len(set(mol_num_ids))} molecules")

    print("Training model...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=args.class_weight,
        random_state=42,
    )
    clf.fit(descriptors, som_labels)

    print("Saving model...")
    dump(clf, os.path.join(args.out_folder, "model.joblib"))

    print("Finished in:", datetime.now() - start_time)

    sys.exit(0)


if __name__ == "__main__":
    main()
