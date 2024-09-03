import numpy as np

from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class PerformanceMetrics:
    @classmethod
    def compute_metrics(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        mol_id: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        mcc = matthews_corrcoef(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_prob)

        top2 = 0
        for id in list(
            dict.fromkeys(mol_id.tolist())
        ):  # This is a somewhat complicated way to get an ordered set, but it works
            mask = np.where(mol_id == id)[0]
            masked_y_true = y_true[mask]
            masked_y_pred = y_prob[mask]
            masked_sorted_y_true = np.take(
                masked_y_true,
                indices=np.argsort(masked_y_pred)[::-1],
                axis=0,
            )
            if np.sum(masked_sorted_y_true[:2]).item() > 0:
                top2 += 1
        top2 /= len(set(mol_id.tolist()))

        return mcc, precision, recall, auroc, top2
