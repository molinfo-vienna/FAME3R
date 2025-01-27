# pylint: disable=C0114,R0801

import argparse
import csv
import os
import sys
from datetime import datetime
from statistics import mean, stdev

import numpy as np
from joblib import load

from fame3r import FAMEDescriptors, compute_metrics

NUM_BOOTSTRAPS = 1000
THRESHOLD = 0.3


# pylint: disable=C0116
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tests a trained re-implementation of the FAME.AL model."
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
        mol_ids,
        atom_ids,
        som_labels,
        descriptors,
    ) = descriptors_generator.compute_fame_descriptors(
        args.input_file, args.out_folder, has_soms=True
    )

    print(f"Testing data: {len(set(mol_ids))} molecules")

    print("Loading model...")
    clf = load(args.model_file)

    print("Testing model...")
    y_prob = clf.predict_proba(descriptors)[:, 1]
    y_pred = (y_prob > THRESHOLD).astype(int)

    metrics = {
        "AUROC": [],
        "Average precision": [],
        "F1": [],
        "MCC": [],
        "Precision": [],
        "Recall": [],
        "Top-2 correctness rate": [],
    }

    for i in range(NUM_BOOTSTRAPS):
        print(f"Bootstrap {i + 1}...")
        sampled_mol_ids = np.random.choice(
            list(set(mol_ids)), len(set(mol_ids)), replace=True
        )
        mask = np.zeros(len(mol_ids), dtype=bool)
        for mol_id in sampled_mol_ids:
            mask = mask | (mol_ids == mol_id)
        print("Computing metrics...")
        (
            auroc,
            average_precision,
            f1,
            mcc,
            precision,
            recall,
            top2,
        ) = compute_metrics(som_labels[mask], y_prob[mask], y_pred[mask], mol_ids[mask])

        metrics["AUROC"].append(auroc)
        metrics["Average precision"].append(average_precision)
        metrics["F1"].append(f1)
        metrics["MCC"].append(mcc)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["Top-2 correctness rate"].append(top2)

    metrics_file = os.path.join(args.out_folder, "test_metrics.txt")
    with open(metrics_file, "w", encoding="UTF-8") as file:
        for metric, scores in metrics.items():
            file.write(
                f"{metric}: {round(mean(scores), 4)} +/- {round(stdev(scores), 4)}\n"
            )
    print(f"Metrics saved to {metrics_file}")

    predictions_file = os.path.join(args.out_folder, "predictions.csv")
    with open(predictions_file, "w", encoding="UTF-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mol_id", "atom_id", "y_prob", "y_pred", "y_true"])
        for mol_id, atom_id, prediction, prediction_binary, true_label in zip(
            mol_ids,
            atom_ids,
            y_prob,
            y_pred,
            som_labels,
        ):
            writer.writerow(
                [mol_id, atom_id, prediction, prediction_binary, true_label]
            )
    print(f"Predictions saved to {predictions_file}")

    print("Finished in:", datetime.now() - start_time)

    sys.exit(0)
