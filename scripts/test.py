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

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.discovery import Path

NUM_BOOTSTRAPS = 1000


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
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
        "-fs",
        dest="use_fame_score",
        action="store_true",
        help="Use FAME score for ranking",
    )

    parse_args = parser.parse_args()

    return parse_args


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    mol_num_id: np.ndarray,
):
    """
    Compute various performance metrics for binary classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        y_pred (np.ndarray): Predicted binary labels.
        mol_num_id (np.ndarray): Array of numerical molecule IDs corresponding to each data point.

    Returns:
        tuple[float, float, float, float, float, float, float]:
            A tuple containing AUROC, AP, F1, MCC, precision, recall, and top-2 success rate.
    """
    # Basic metrics
    auroc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Calculate Top-2 success rate
    unique_mol_num_ids, _ = np.unique(mol_num_id, return_index=True)
    top2_sucesses = 0

    for i in unique_mol_num_ids:
        mask = mol_num_id == i
        masked_y_true = y_true[mask]
        masked_y_prob = y_prob[mask]

        # Sort by predicted probability (descending) and take the top 2
        top_2_indices = np.argsort(masked_y_prob)[-2:]
        if masked_y_true[top_2_indices].sum() > 0:
            top2_sucesses += 1

    top2_rate = top2_sucesses / len(unique_mol_num_ids)

    return float(auroc), float(ap), f1, mcc, precision, recall, top2_rate


def main():
    """Application entry point."""
    start_time = datetime.now()

    args = parse_arguments()

    data = [row for row in csv.DictReader(Path(args.input_file).open())]
    data = np.array(
        [tuple(row.values()) for row in data],
        dtype=[(key, "object" if key == "smiles" else "<f8") for key in data[0].keys()],
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

        mask = np.zeros(len(data), dtype=bool)
        for random_smiles in rng.choice(
            list(set(data["smiles"])), len(set(data["smiles"]))
        ):
            mask = mask | (data["smiles"] == random_smiles)
        (
            auroc,
            average_precision,
            f1,
            mcc,
            precision,
            recall,
            top2,
        ) = compute_metrics(
            data["y_true"][mask],
            data["fame_score"][mask] if args.use_fame_score else data["y_prob"],
            data["y_pred"][mask],
            data["smiles"][mask],
        )

        print(data["smiles"])

        metrics["AUROC"].append(auroc)
        metrics["Average precision"].append(average_precision)
        metrics["F1"].append(f1)
        metrics["MCC"].append(mcc)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["Top-2 correctness rate"].append(top2)

    metrics_file = "metrics.txt"

    with Path(metrics_file).open("w", encoding="UTF-8") as file:
        for metric, scores in metrics.items():
            file.write(
                f"{metric}: {np.mean(scores).round(4)} +/- {np.std(scores).round(4)}\n"
            )
    print(f"Metrics saved to {metrics_file}")

    print("Finished in:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
