"""EEGNet baseline experiment under the canonical CV protocol."""

from __future__ import annotations

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.evaluation import (
    FoldMetrics,
    SubjectResult,
    aggregate_fold_metrics,
    compute_classification_metrics,
    summarize_subject_results,
)
from eeg_motor_imagery_classification.models.eegnet import EEGNet
from eeg_motor_imagery_classification.train import (
    TrainingConfig,
    build_train_validation_datasets,
    fit_model,
    predict_model,
)


def run_eegnet_within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    training_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Evaluate EEGNet with the same fold definition used by classical baselines."""

    cfg = training_config or TrainingConfig()
    runtime_seconds = 0.0
    training_histories: list[dict[str, object]] = []
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_metrics = []

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_dataset, val_dataset = build_train_validation_datasets(X_train, y_train, config=cfg)
            test_dataset = EpochDataset(X_test, y_test, scaler=train_dataset.scaler)

            start = time.perf_counter()
            model = EEGNet(
                n_channels=X.shape[1],
                n_times=X.shape[2],
                n_classes=len(np.unique(y)),
            )
            training = fit_model(model, train_dataset, validation_dataset=val_dataset, config=cfg)
            y_pred = predict_model(training.model, test_dataset, config=cfg)
            runtime_seconds += time.perf_counter() - start
            fold_metrics.append(compute_classification_metrics(y_test, y_pred))
            training_histories.append({"label": f"fold_{fold_id}", **training.history})

        return {
            "summary": aggregate_fold_metrics(fold_metrics),
            "rows": [],
            "runtime_seconds": float(runtime_seconds),
            "training_histories": training_histories,
        }

    subject_results: list[SubjectResult] = []
    for subject_id in np.unique(groups):
        subject_idx = np.flatnonzero(groups == subject_id)
        X_subject, y_subject = X[subject_idx], y[subject_idx]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_metrics = []

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_subject, y_subject), start=1):
            X_train, X_test = X_subject[train_idx], X_subject[test_idx]
            y_train, y_test = y_subject[train_idx], y_subject[test_idx]

            train_dataset, val_dataset = build_train_validation_datasets(X_train, y_train, config=cfg)
            test_dataset = EpochDataset(X_test, y_test, scaler=train_dataset.scaler)

            start = time.perf_counter()
            model = EEGNet(
                n_channels=X.shape[1],
                n_times=X.shape[2],
                n_classes=len(np.unique(y)),
            )
            training = fit_model(model, train_dataset, validation_dataset=val_dataset, config=cfg)
            y_pred = predict_model(training.model, test_dataset, config=cfg)
            runtime_seconds += time.perf_counter() - start
            fold_metrics.append(compute_classification_metrics(y_test, y_pred))
            training_histories.append({"label": f"S{int(subject_id)}_fold_{fold_id}", **training.history})

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
    result["training_histories"] = training_histories
    return result
