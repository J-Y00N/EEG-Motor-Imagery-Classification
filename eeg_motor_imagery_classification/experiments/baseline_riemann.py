"""Within-subject baseline for Riemannian tangent-space models."""

from __future__ import annotations

import time
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from eeg_motor_imagery_classification.evaluation import (
    FoldMetrics,
    SubjectResult,
    aggregate_fold_metrics,
    compute_classification_metrics,
    summarize_subject_results,
)
from eeg_motor_imagery_classification.models import build_riemann_tangent_pipeline


def run_riemann_within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate the tangent-space baseline under the shared CV protocol."""

    pipeline = build_riemann_tangent_pipeline()
    runtime_seconds = 0.0
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_metrics = []

        for train_idx, test_idx in cv.split(X, y):
            start = time.perf_counter()
            model = clone(pipeline)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            runtime_seconds += time.perf_counter() - start
            fold_metrics.append(compute_classification_metrics(y[test_idx], y_pred))

        return {"summary": aggregate_fold_metrics(fold_metrics), "rows": [], "runtime_seconds": float(runtime_seconds)}

    subject_results: list[SubjectResult] = []
    for subject_id in np.unique(groups):
        subject_idx = np.flatnonzero(groups == subject_id)
        X_subject, y_subject = X[subject_idx], y[subject_idx]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_metrics = []

        for train_idx, test_idx in cv.split(X_subject, y_subject):
            start = time.perf_counter()
            model = clone(pipeline)
            model.fit(X_subject[train_idx], y_subject[train_idx])
            y_pred = model.predict(X_subject[test_idx])
            runtime_seconds += time.perf_counter() - start
            fold_metrics.append(compute_classification_metrics(y_subject[test_idx], y_pred))

        subject_summary = aggregate_fold_metrics(fold_metrics)
        subject_results.append(
            SubjectResult(
                label=f"S{int(subject_id)}",
                metrics=FoldMetrics(
                    accuracy=float(subject_summary["accuracy_mean"]),
                    balanced_accuracy=float(subject_summary["balanced_accuracy_mean"]),
                    macro_f1=float(subject_summary["macro_f1_mean"]),
                    confusion_matrix=np.asarray(subject_summary["confusion_matrix_sum"]),
                ),
            )
        )

    result = summarize_subject_results(subject_results)
    result["runtime_seconds"] = float(runtime_seconds)
    return result
