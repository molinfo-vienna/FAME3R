import argparse
import csv
import os
import sys
from datetime import datetime

from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from fame3r import FAMEDescriptors

THRESHOLD = 0.3


def main() -> None:
    args = parseArgs()
    print(f"Radius: {args.radius}")

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print("The new output folder is created.")

    print(f"Computing descriptors...")

    descriptors_generator = FAMEDescriptors(args.radius)
    (
        mol_ids,
        atom_ids,
        som_labels,
        descriptors,
    ) = descriptors_generator.compute_FAME_descriptors(
        args.input_file, args.out_folder, has_soms=True
    )

    print(f"Training data: {len(set(mol_ids))} molecules")

    print(f"Training model...")
    clf = RandomForestClassifier(
        n_estimators=250, class_weight="balanced_subsample", random_state=42
    )
    clf.fit(descriptors, som_labels)

    print(f"Saving model...")
    dump(clf, os.path.join(args.out_folder, "model.joblib"))


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trains a re-implementation of the FAME.AL model."
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<training data file>",
        help="Trainig data file",
    )
    parser.add_argument(
        "-o",
        dest="out_folder",
        required=True,
        metavar="<output folder>",
        help="Model output location",
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

    parse_args = parser.parse_args()

    return parse_args


if __name__ == "__main__":
    start_time = datetime.now()

    main()

    print("Finished in:")
    print(datetime.now() - start_time)

    sys.exit(0)
