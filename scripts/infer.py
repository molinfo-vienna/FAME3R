"""Applies a trained re-implementation of the FAME3 model to unlabeled data.

This script saves the per-atom predictions to a CSV file.
The radius of the atom environment is not part of the hyperparameter search, \
    but can be set by changing the radius argument. Default is 5.
The decision threshold can be changed by changing the threshold argument. Default is 0.2.
"""

import argparse
import csv
import glob
import os
import sys
from datetime import datetime

import pandas as pd
from joblib import load

from fame3r import FAMEDescriptors, FAMEScores


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Applies a trained re-implementation of the FAME3 model to unlabeled data"
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
        dest="model_folder",
        required=True,
        metavar="<Model folder>",
        help="Model folder",
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
    parser.add_argument(
        "-t",
        dest="threshold",
        required=False,
        metavar="<binary decision threshold>",
        default=0.2,
        help="Binary decision threshold",
        type=float,
    )

    parse_args = parser.parse_args()

    return parse_args


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
        mol_ids,
        atom_ids,
        _,
        descriptors,
    ) = descriptors_generator.compute_fame_descriptors(
        args.input_file, args.out_folder, has_soms=False
    )

    print(f"Data: {len(set(mol_num_ids))} molecules")

    print("Loading model...")
    clf = load(os.path.join(args.model_folder, "model.joblib"))

    print("Predicting SOMs...")
    predictions = clf.predict_proba(descriptors)[:, 1]
    predictions_binary = (predictions > args.threshold).astype(int)

    print("Computing FAME scores...")
    # Get the path of the CSV file that ends with "descriptors.csv"
    csv_files = glob.glob(os.path.join(args.model_folder, "*descriptors.csv"))

    if len(csv_files) == 1:
        train_descriptors_df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(
            f"Expected one CSV file ending with 'descriptors.csv', but found {len(csv_files)}."
        )
    train_descriptors = train_descriptors_df.drop(
        columns=["mol_num_id", "mol_id", "atom_id", "som_label"]
    ).values
    fame_scores_generator = FAMEScores(train_descriptors)
    fame_scores = fame_scores_generator.compute_fame_scores(descriptors)

    predictions_file = os.path.join(args.out_folder, "predictions.csv")
    with open(predictions_file, "w", encoding="UTF-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["mol_id", "atom_id", "predictions", "predictions_binary", "fame_scores"]
        )
        for mol_id, atom_id, prediction, prediction_binary, fame_score in zip(
            mol_ids, atom_ids, predictions, predictions_binary, fame_scores
        ):
            writer.writerow(
                [mol_id, atom_id, prediction, prediction_binary, fame_score]
            )
    print(f"Predictions saved to {predictions_file}")

    print("Finished in:", datetime.now() - start_time)

    sys.exit(0)


if __name__ == "__main__":
    main()
