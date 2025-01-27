# pylint: disable=R0801

"""Tests a trained re-implementation of the FAME.AL model to unlabeled data.

The script saves the predictions to a CSV file.
The radius of the atom environment is not part of the hyperparameter search, \
    but can be set by changing the radius argument. Default is 5.
The decision threshold can be changed by modifying the THRESHOLD variable. Default is 0.3.
"""


import argparse
import csv
import os
import sys
from datetime import datetime

from joblib import load

from fame3r import FAMEDescriptors

THRESHOLD = 0.3


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Applies a trained re-implementation of the FAME.AL model to unlabeled data"
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<Input data file>",
        help="Input data file",
    )
    parser.add_argument(
        "-m",
        dest="model_file",
        required=True,
        metavar="<Model file>",
        help="Model file",
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

    args = parse_arguments()
    print(f"Radius: {args.radius}")

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print("The new output folder is created.")

    print("Computing descriptors...")

    descriptors_generator = FAMEDescriptors(args.radius)
    (
        mol_ids_test,
        atom_ids_test,
        _,
        descriptors_test,
    ) = descriptors_generator.compute_fame_descriptors(
        args.input_file, args.out_folder, has_soms=False
    )

    print(f"Data: {len(set(mol_ids_test))} molecules")

    print("Loading model...")
    clf = load(args.model_file)

    print("Testing model...")
    predictions = clf.predict_proba(descriptors_test)[:, 1]
    predictions_binary = (predictions > THRESHOLD).astype(int)

    predictions_file = os.path.join(args.out_folder, "predictions.csv")
    with open(predictions_file, "w", encoding="UTF-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mol_id", "atom_id", "predictions", "predictions_binary"])
        for mol_id, atom_id, prediction, prediction_binary in zip(
            mol_ids_test, atom_ids_test, predictions, predictions_binary
        ):
            writer.writerow([mol_id, atom_id, prediction, prediction_binary])
    print(f"Predictions saved to {predictions_file}")

    print("Finished in:", datetime.now() - start_time)

    sys.exit(0)
