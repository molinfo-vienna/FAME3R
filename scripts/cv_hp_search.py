# pylint: disable=R0801

"""Performs K-fold grid-search cross-validation to find the best model hyperparameters.

The searching space can be set in the param_grid dictionary. \
    The script saves the best hyperparameters to a text file.
The script also saves the optimal decision threshold to a text file. \
    The optimal threshold is determined by the majority vote of \
        the best thresholds found in each fold.
The script saves the optimal k-fold CV metrics (mean and standard deviation) \
    of the model to a text file.
The number of folds can be set by changing the NUM_FOLDS variable. Default is 10.
The radius of the atom environment is not part of the hyperparameter search, \
    but can be set by changing the radius argument. Default is 5.
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from statistics import mean, stdev

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, make_scorer,
                             matthews_corrcoef)
from sklearn.model_selection import GridSearchCV, GroupKFold

from fame3r import FAMEDescriptors, compute_metrics

NUM_FOLDS = 10  # Number of folds for cross-validation


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Performs K-fold cross-validation to find the \
            best hyperparameters of a re-implementation of the FAME.AL model."
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<Input data file>",
        help="Input data file",
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
    print(f"Descriptor radius: {args.radius}")

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print("New output folder is created.")

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

    # Define parameter grid for RandomForest
    param_grid = {
        "n_estimators": [100, 250, 500, 750],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    scorer = make_scorer(average_precision_score, greater_is_better=True)

    kf = GroupKFold(n_splits=NUM_FOLDS, random_state=42, shuffle=True)

    print("Performing hyperparameter optimization using GridSearchCV...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=kf.split(descriptors, som_labels, groups=mol_ids),
        scoring=scorer,
        n_jobs=-1,
        verbose=3,
    )

    grid_search.fit(descriptors, som_labels)
    best_model = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")

    # Save best hyperparameters to a file
    best_params_file = os.path.join(args.out_folder, "best_hyperparameters.txt")
    with open(best_params_file, "w", encoding="UTF-8") as file:
        for param, value in grid_search.best_params_.items():
            file.write(f"{param}: {value}\n")
    print(f"Best hyperparameters saved to {best_params_file}")

    metrics = {
        "AUROC": [],
        "Average precision": [],
        "F1": [],
        "MCC": [],
        "Precision": [],
        "Recall": [],
        "Top-2 correctness rate": [],
    }

    best_thresholds = []
    for i, (train_index, val_index) in enumerate(
        kf.split(descriptors, som_labels, groups=mol_ids)
    ):
        print(f"Fold {i+1}")

        descriptors_train = descriptors[train_index, :]
        y_true_train = som_labels[train_index]

        decriptors_val = descriptors[val_index, :]
        y_true_val = som_labels[val_index]
        mol_id_val = mol_ids[val_index]

        print("Training model with best parameters...")
        best_model.fit(descriptors_train, y_true_train)

        print("Predicting on validation set...")
        y_prob = best_model.predict_proba(decriptors_val)[:, 1]

        print("Searching for best decision threshold...")
        thresholds = np.linspace(0.1, 0.9, 9)
        BEST_MCC = -1
        BEST_THRESHOLD = 0.5

        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(int)
            mcc = matthews_corrcoef(y_true_val, y_pred)

            if mcc > BEST_MCC:
                BEST_MCC = mcc
                BEST_THRESHOLD = threshold

        print(f"Best threshold for fold {i+1}: {BEST_THRESHOLD}")
        best_thresholds.append(BEST_THRESHOLD)

        y_pred = (y_prob > BEST_THRESHOLD).astype(int)

        print("Computing metrics...")
        (
            auroc,
            average_precision,
            f1,
            mcc,
            precision,
            recall,
            top2,
        ) = compute_metrics(y_true_val, y_prob, y_pred, mol_id_val)

        metrics["AUROC"].append(auroc)
        metrics["Average precision"].append(average_precision)
        metrics["F1"].append(f1)
        metrics["MCC"].append(mcc)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["Top-2 correctness rate"].append(top2)

    # Determine the majority vote for the best threshold
    threshold_counts = Counter(best_thresholds)
    majority_threshold = threshold_counts.most_common(1)[0][0]
    print(f"Majority vote threshold: {majority_threshold}")

    # Save optimal thresholds
    with open(best_params_file, "a", encoding="UTF-8") as file:
        file.write(f"decision_threshold: {majority_threshold}\n")
    print(f"Optimal threshold saved to {best_params_file}")

    # Save metrics
    metrics_file = os.path.join(args.out_folder, f"{NUM_FOLDS}_fold_cv_metrics.txt")
    with open(metrics_file, "w", encoding="UTF-8") as file:
        for metric, scores in metrics.items():
            file.write(
                f"{metric}: {round(mean(scores), 4)} +/- {round(stdev(scores), 4)}\n"
            )
    print(f"Metrics saved to {metrics_file}")

    print("Finished in:", datetime.now() - start_time)
    sys.exit(0)
