import argparse
import csv
import os
import sys

from datetime import datetime
from joblib import load

from fameal import FAMEDescriptors


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
        mol_ids_test,
        atom_ids_test,
        _,
        descriptors_test,
    ) = descriptors_generator.compute_FAME_descriptors(
        args.input_file, args.out_folder, has_soms=False
    )

    print(f"Data: {len(set(mol_ids_test))} molecules")

    print(f"Loading model...")
    clf = load(args.model_file)

    print(f"Testing model...")
    predictions = clf.predict_proba(descriptors_test)[:, 1]
    predictions_binary = (predictions > THRESHOLD).astype(int)

    predictions_file = os.path.join(args.out_folder, "predictions.csv")
    with open(predictions_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mol_id", "atom_id", "predictions", "predictions_binary"])
        for mol_id, atom_id, prediction, prediction_binary in zip(
            mol_ids_test, atom_ids_test, predictions, predictions_binary
        ):
            writer.writerow([mol_id, atom_id, prediction, prediction_binary])
    print(f"Predictions saved to {predictions_file}")


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Applied a pretrained a re-implementation of the FAME.AL model to unlabeled data"
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

    main()

    print("Finished in:")
    print(datetime.now() - start_time)

    sys.exit(0)
