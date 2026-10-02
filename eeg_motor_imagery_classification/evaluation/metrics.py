"""Reusable evaluation metrics for fair comparison across protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
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


def _mean_std_ci(values: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Mean, sample std (ddof=1), SEM, and a t-distribution confidence interval."""

    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "sem": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(values.std(ddof=1))
    sem = std / np.sqrt(n)
    half_width = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * sem)
    return {"mean": mean, "std": std, "sem": float(sem), "ci_low": mean - half_width, "ci_high": mean + half_width}


def aggregate_fold_metrics(metrics: list[FoldMetrics]) -> dict[str, float | np.ndarray]:
    """Aggregate fold- or subject-level metrics into report-friendly summaries.

    ``*_std`` is the sample standard deviation (ddof=1) across the aggregated units and
    ``*_ci95_*`` is a t-distribution interval with ``n - 1`` degrees of freedom.
    """

    if not metrics:
        raise ValueError("At least one fold metric is required for aggregation.")

    summary: dict[str, float | np.ndarray] = {"n": len(metrics), "std_ddof": 1}
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = np.asarray([getattr(m, name) for m in metrics], dtype=float)
        described = _mean_std_ci(values)
        summary[f"{name}_mean"] = described["mean"]
        summary[f"{name}_std"] = described["std"]
        summary[f"{name}_sem"] = described["sem"]
        summary[f"{name}_ci95_low"] = described["ci_low"]
        summary[f"{name}_ci95_high"] = described["ci_high"]
    summary["confusion_matrix_sum"] = np.sum([m.confusion_matrix for m in metrics], axis=0)
    return summary
