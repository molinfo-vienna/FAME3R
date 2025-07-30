"""Tests a trained re-implementation of the FAME3 model.

This script saves the test metrics to a text file.
This script saves the per-atom predictions to a CSV file.
This script performs bootstrapping to estimate the uncertainty in the metrics.\
    The number of bootstraps can be set by changing the NUM_BOOTSTRAPS variable. Default is 1000.
The radius of the atom environment is not part of the hyperparameter search, \
    but can be set by changing the radius argument. Default is 5.
The decision threshold can be changed by changing the threshold argument. Default is 0.3.
The script also computes FAME scores if the -fs flag is set.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

NUM_BOOTSTRAPS = 1000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tests a trained re-implementation of the FAME3 model."
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<Predictions data file>",
        help="Predictions data file",
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
        "-fs",
        dest="use_fame_score",
        action="store_true",
        help="Use FAME score for ranking",
    )

    parse_args = parser.parse_args()

    return parse_args


def top2_rate_score(y_true, y_prob, groups):
    unique_groups, _ = np.unique(groups, return_index=True)
    top2_sucesses = 0

    for current_group in unique_groups:
        mask = groups == current_group

        # Sort by predicted probability (descending) and take the top 2
        top_2_indices = np.argsort(y_prob[mask])[-2:]
        if y_true[mask][top_2_indices].sum() > 0:
            top2_sucesses += 1

    return top2_sucesses / len(unique_groups)


def compute_metrics(y_true, y_prob, y_pred, groups):
    auroc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    top2_rate = top2_rate_score(y_true, y_prob, groups)

    return float(auroc), float(ap), f1, mcc, precision, recall, top2_rate


def main():
    """Application entry point."""
    start_time = datetime.now()

    args = parse_arguments()

    rows = [row for row in csv.DictReader(Path(args.input_file).open())]
    smiles, y_true, y_pred, y_prob, fame_score = (
        np.array([row[key] for row in rows], dtype=object if key == "smiles" else float)
        for key in ["smiles", "y_true", "y_pred", "y_prob", "fame_score"]
    )

    print("Computing metrics...")

    metrics: dict[str, list[float]] = {
        "AUROC": [],
        "Average precision": [],
        "F1": [],
        "MCC": [],
        "Precision": [],
        "Recall": [],
        "Top-2 correctness rate": [],
    }

    rng = np.random.default_rng(0)

    for i in range(NUM_BOOTSTRAPS):
        print(f"Bootstrap run: {i}")

        mask = np.zeros(len(smiles), dtype=bool)
        for random_smiles in rng.choice(np.unique(smiles), len(np.unique(smiles))):
            mask = mask | (smiles == random_smiles)

        (
            auroc,
            average_precision,
            f1,
            mcc,
            precision,
            recall,
            top2,
        ) = compute_metrics(
            y_true[mask],
            fame_score[mask] if args.use_fame_score else y_prob[mask],
            y_pred[mask],
            smiles[mask],
        )

        metrics["AUROC"].append(auroc)
        metrics["Average precision"].append(average_precision)
        metrics["F1"].append(f1)
        metrics["MCC"].append(mcc)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["Top-2 correctness rate"].append(top2)

    print(f"Saving metrics...")

    with Path(args.out_file).open("w", encoding="UTF-8") as f:
        for metric, scores in metrics.items():
            f.write(
                f"{metric}: {np.mean(scores).round(4)} +/- {np.std(scores).round(4)}\n"
            )

    print("Finished in:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
