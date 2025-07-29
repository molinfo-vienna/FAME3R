# pyright: reportAttributeAccessIssue=false

"""Performs K-fold grid-search cross-validation to find the best model hyperparameters.

The searching space can be set in the param_grid dictionary. \
    The script saves the best hyperparameters to a text file.
The script also saves the optimal binary decision threshold to a text file. \
    The optimal threshold is determined by the majority vote of \
        the best thresholds found in each fold.
This script saves the optimal k-fold CV metrics (mean and standard deviation) \
    of the model to a text file.
The radius of the atom environment is not part of the hyperparameter search, \
    but can be set by changing the radius argument. Default is 5.
The number of folds can be set by changing the num_folds argument. Default is 10.
"""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from CDPL.Chem import (
    Atom,
    BasicMolecule,
    FileSDFMoleculeReader,
    MolecularGraph,
    getStructureData,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

from fame3r import FAME3RVectorizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Performs K-fold cross-validation to find the \
            best hyperparameters of a re-implementation of the FAME3 model."
    )

    parser.add_argument(
        "-i",
        dest="input_file",
        required=True,
        metavar="<input data file>",
        help="Input data file",
        type=str,
    )
    parser.add_argument(
        "-o",
        dest="out_folder",
        required=True,
        metavar="<output folder>",
        help="Model output location",
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
        dest="num_folds",
        required=False,
        metavar="<number of cross-validation folds>",
        default=10,
        help="Number of cross-validation folds",
        type=int,
    )

    parse_args = parser.parse_args()

    return parse_args


def compute_metrics(y_true, y_prob, y_pred, mol_num_id):
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


def extract_som_labels(mol: MolecularGraph) -> list[tuple[Atom, bool]]:
    structure_data = {
        entry.header[2:].split(">")[0]: entry.data for entry in getStructureData(mol)
    }
    som_indices = eval(structure_data["soms"])

    return [(atom, int(atom.index in som_indices)) for atom in mol.getAtoms()]  # pyright:ignore


def main():
    """Application entry point."""
    start_time = datetime.now()

    args = parse_arguments()
    print(f"Descriptor radius: {args.radius}")

    print("Loading training data...")

    som_atoms_labeled: list[tuple[Atom, bool]] = []

    reader = FileSDFMoleculeReader(args.input_file)
    mol = BasicMolecule()
    while reader.read(mol):  # pyright:ignore
        som_atoms_labeled.extend(extract_som_labels(mol))
        mol = BasicMolecule()

    print(f"Training data: {len(som_atoms_labeled)} data points")

    print("Extracting features...")

    # Unfortunately, this monkey-patching is required to get CDPKit
    # objects like atoms and molecules into NumPy arrays...
    del Atom.__getitem__

    som_atoms = [[som_atom] for som_atom, _ in som_atoms_labeled]
    labels = [label for _, label in som_atoms_labeled]
    containing_mols = [atom.molecule.getObjectID() for atom, _ in som_atoms_labeled]

    descriptors = FAME3RVectorizer(radius=args.radius, input="cdpkit").fit_transform(
        som_atoms
    )

    print("Performing hyperparameter optimization using GridSearchCV...")

    param_grid = {
        "n_estimators": [100, 250, 500, 750],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    scorer = make_scorer(average_precision_score, greater_is_better=True)

    kf = GroupKFold(n_splits=args.num_folds, random_state=42, shuffle=True)

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=kf.split(descriptors, labels, groups=containing_mols),
        scoring=scorer,
        n_jobs=1,
        verbose=3,
    )

    grid_search.fit(descriptors, labels)
    best_model = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")

    # Save best hyperparameters to a file
    best_params_file = Path(args.out_folder) / "best_hyperparameters.txt"
    with best_params_file.open("w", encoding="UTF-8") as f:
        for param, value in grid_search.best_params_.items():
            f.write(f"{param}: {value}\n")

    print(f"Best hyperparameters saved to {best_params_file}")

    metrics: dict[str, list[float]] = {
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
        kf.split(descriptors, labels, groups=containing_mols)
    ):
        print(f"Fold {i + 1}")

        y_true_train = np.array(labels)[train_index]
        descriptors_train = np.array(descriptors)[train_index, :]

        decriptors_val = np.array(descriptors)[val_index, :]
        y_true_val = np.array(labels)[val_index]
        mol_num_id_val = np.array(containing_mols)[val_index]

        print("Training model with best parameters...")
        best_model.fit(descriptors_train, y_true_train)

        print("Predicting on validation set...")
        y_prob = best_model.predict_proba(decriptors_val)[:, 1]

        print("Searching for best decision threshold...")
        thresholds = np.linspace(0.1, 0.9, 9)
        best_mcc = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(int)
            mcc = matthews_corrcoef(y_true_val, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

        print(f"Best threshold for fold {i + 1}: {best_threshold}")
        best_thresholds.append(best_threshold)

        y_pred = (y_prob > best_threshold).astype(int)

        print("Computing metrics...")
        (
            auroc,
            average_precision,
            f1,
            mcc,
            precision,
            recall,
            top2,
        ) = compute_metrics(y_true_val, y_prob, y_pred, mol_num_id_val)

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
    with open(best_params_file, "a", encoding="UTF-8") as f:
        f.write(f"decision_threshold: {round(majority_threshold, 1)}\n")
    print(f"Optimal threshold saved to {best_params_file}")

    # Save metrics
    metrics_file = Path(args.out_folder) / f"{args.num_folds}_fold_cv_metrics.txt"
    with open(metrics_file, "w", encoding="UTF-8") as f:
        for metric, scores in metrics.items():
            f.write(
                f"{metric}: {np.mean(scores).round(4)} +/- {np.std(scores).round(4)}\n"
            )
    print(f"Metrics saved to {metrics_file}")

    print("Finished in:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
