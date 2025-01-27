# pylint: disable=R0914

"""Module for computing performance metrics for binary classification task."""

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    mol_id: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Compute various performance metrics for binary classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        y_pred (np.ndarray): Predicted binary labels.
        mol_id (np.ndarray): Array of molecule IDs corresponding to each data point.

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
    unique_mol_ids, _ = np.unique(mol_id, return_index=True)
    top2_sucesses = 0

    for i in unique_mol_ids:
        mask = mol_id == i
        masked_y_true = y_true[mask]
        masked_y_prob = y_prob[mask]

        # Sort by predicted probability (descending) and take the top 2
        top_2_indices = np.argsort(masked_y_prob)[-2:]
        if masked_y_true[top_2_indices].sum() > 0:
            top2_sucesses += 1

    top2_rate = top2_sucesses / len(unique_mol_ids)

    return auroc, ap, f1, mcc, precision, recall, top2_rate
