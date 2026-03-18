"""Classical baseline experiment built on the canonical data pipeline."""

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
from eeg_motor_imagery_classification.models import (
    build_csp_pipeline,
    build_fbcsp_pipeline,
    build_raw_power_pipeline,
)


def run_classical_within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sfreq: float,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, object]]:
    """Evaluate raw power, CSP, and FBCSP under one consistent CV protocol."""

    pipelines = {
        "raw_power": build_raw_power_pipeline(),
        "csp": build_csp_pipeline(),
        "fbcsp": build_fbcsp_pipeline(sfreq=sfreq),
    }
    runtimes = {name: 0.0 for name in pipelines}

    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = {name: [] for name in pipelines}

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for name, pipeline in pipelines.items():
                start = time.perf_counter()
                model = clone(pipeline)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                runtimes[name] += time.perf_counter() - start
                fold_results[name].append(compute_classification_metrics(y_test, y_pred))

        return {
            name: {
                "summary": aggregate_fold_metrics(metrics),
                "rows": [],
                "runtime_seconds": float(runtimes[name]),
            }
            for name, metrics in fold_results.items()
        }

    subject_results = {name: [] for name in pipelines}
    for subject_id in np.unique(groups):
        subject_idx = np.flatnonzero(groups == subject_id)
        X_subject, y_subject = X[subject_idx], y[subject_idx]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = {name: [] for name in pipelines}

        for train_idx, test_idx in cv.split(X_subject, y_subject):
            X_train, X_test = X_subject[train_idx], X_subject[test_idx]
            y_train, y_test = y_subject[train_idx], y_subject[test_idx]

            for name, pipeline in pipelines.items():
                start = time.perf_counter()
                model = clone(pipeline)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                runtimes[name] += time.perf_counter() - start
                fold_results[name].append(compute_classification_metrics(y_test, y_pred))

        for name, metrics in fold_results.items():
            subject_summary = aggregate_fold_metrics(metrics)
            subject_results[name].append(
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

    outputs = {name: summarize_subject_results(results) for name, results in subject_results.items()}
    for name in outputs:
        outputs[name]["runtime_seconds"] = float(runtimes[name])
    return outputs
