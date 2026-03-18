"""Reusable evaluation metrics for fair comparison across protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics computed on one evaluation fold."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FoldMetrics:
    """Compute standardized classification metrics."""

    accuracy = float(np.mean(y_true == y_pred))
    return FoldMetrics(
        accuracy=accuracy,
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )


def aggregate_fold_metrics(metrics: list[FoldMetrics]) -> dict[str, float | np.ndarray]:
    """Aggregate fold-level metrics into report-friendly summaries."""

    if not metrics:
        raise ValueError("At least one fold metric is required for aggregation.")

    accuracy = np.asarray([m.accuracy for m in metrics], dtype=float)
    balanced = np.asarray([m.balanced_accuracy for m in metrics], dtype=float)
    macro_f1 = np.asarray([m.macro_f1 for m in metrics], dtype=float)
    confusion = np.sum([m.confusion_matrix for m in metrics], axis=0)

    n = len(metrics)
    accuracy_sem = float(accuracy.std(ddof=0) / np.sqrt(n))
    balanced_sem = float(balanced.std(ddof=0) / np.sqrt(n))
    macro_f1_sem = float(macro_f1.std(ddof=0) / np.sqrt(n))

    return {
        "n": n,
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=0)),
        "accuracy_sem": accuracy_sem,
        "accuracy_ci95_low": float(accuracy.mean() - 1.96 * accuracy_sem),
        "accuracy_ci95_high": float(accuracy.mean() + 1.96 * accuracy_sem),
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_std": float(balanced.std(ddof=0)),
        "balanced_accuracy_sem": balanced_sem,
        "balanced_accuracy_ci95_low": float(balanced.mean() - 1.96 * balanced_sem),
        "balanced_accuracy_ci95_high": float(balanced.mean() + 1.96 * balanced_sem),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=0)),
        "macro_f1_sem": macro_f1_sem,
        "macro_f1_ci95_low": float(macro_f1.mean() - 1.96 * macro_f1_sem),
        "macro_f1_ci95_high": float(macro_f1.mean() + 1.96 * macro_f1_sem),
        "confusion_matrix_sum": confusion,
    }
