import argparse
import os
import sys
from datetime import datetime
from statistics import mean, stdev

import numpy as np
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from fame3r import FAMEDescriptors, PerformanceMetrics

THRESHOLD = 0.3
K = 10


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

    mcc_scores = []
    precision_scores = []
    recall_scores = []
    auroc_scores = []
    top2_success_rate_scores = []

    kf = KFold(n_splits=K, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(kf.split(np.unique(mol_ids))):
        print(f"Fold {i+1}")
        training_mol_ids = np.unique(mol_ids)[train_index]
        training_indexes = np.where(np.isin(mol_ids, training_mol_ids))[0]
        val_mol_ids = np.unique(mol_ids)[val_index]
        val_indexes = np.where(np.isin(mol_ids, val_mol_ids))[0]

        descriptors_train = descriptors[training_indexes, :]
        y_true_train = som_labels[training_indexes]

        decriptors_val = descriptors[val_indexes, :]
        y_true_val = som_labels[val_indexes]
        mol_id_val = mol_ids[val_indexes]

        print("Training model...")
        clf = RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample", random_state=42
        )
        clf.fit(descriptors_train, y_true_train)

        print("Predicting on validation set...")
        y_prob = clf.predict_proba(decriptors_val)[:, 1]
        y_pred = (y_prob > THRESHOLD).astype(int)

        print("Computing metrics...")
        (
            mcc,
            precision,
            recall,
            auroc,
            top2,
        ) = PerformanceMetrics.compute_metrics(y_true_val, y_prob, y_pred, mol_id_val)

        mcc_scores.append(mcc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        auroc_scores.append(auroc)
        top2_success_rate_scores.append(top2)

    metrics_file = os.path.join(args.out_folder, "metrics.txt")
    with open(metrics_file, "w") as file:
        file.write(
            f"MCC: {round(mean(mcc_scores), 4)} +/- {round(stdev(mcc_scores), 4)}\n"
        )
        file.write(
            f"Precision: {round(mean(precision_scores), 4)} +/- {round(stdev(precision_scores), 4)}\n"
        )
        file.write(
            f"Recall: {round(mean(recall_scores), 4)} +/- {round(stdev(recall_scores), 4)}\n"
        )
        file.write(
            f"AUROC: {round(mean(auroc_scores), 4)} +/- {round(stdev(auroc_scores), 4)}\n"
        )
        file.write(
            f"Top-2 correctness rate: {round(mean(top2_success_rate_scores), 4)} +/- {round(stdev(top2_success_rate_scores), 4)}\n"
        )
    print(f"Metrics saved to {metrics_file}")


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulates k-fold cross-validation results for a re-implementation of the FAME.AL model."
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

    main()

    print("Finished in:")
    print(datetime.now() - start_time)

    sys.exit(0)
