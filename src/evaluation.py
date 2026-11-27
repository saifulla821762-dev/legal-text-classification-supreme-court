"""Evaluation metrics and model comparison helpers."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)


def evaluate_model(name: str, y_true, y_pred, labels=None):
    """Return a metric dictionary and confusion matrix."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (weighted)": precision,
        "Recall (weighted)": recall,
        "F1 (weighted)": f1,
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, matrix


def compare_models(metrics_rows: list[dict]) -> pd.DataFrame:
    """Create a sorted comparison table from metric dictionaries."""
    return pd.DataFrame(metrics_rows).sort_values(
        "F1 (weighted)", ascending=False, ignore_index=True
    )
