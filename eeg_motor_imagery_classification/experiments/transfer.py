"""Cross-subject transfer protocols with explicit calibration handling."""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from sklearn.base import clone

from eeg_motor_imagery_classification.data.datasets import EpochDataset
from eeg_motor_imagery_classification.data.splits import transfer_split
from eeg_motor_imagery_classification.evaluation.metrics import FoldMetrics, compute_classification_metrics
from eeg_motor_imagery_classification.evaluation.protocols import (
    SubjectResult,
    aggregate_transfer_seed_runs,
    metric_from_row,
    metric_from_summary,
    summarize_subject_results,
)
from eeg_motor_imagery_classification.models import EEGNet, build_fbcsp_pipeline
from eeg_motor_imagery_classification.train import (
    TrainingConfig,
    build_train_validation_datasets,
    fit_model,
    predict_model,
)


def _sample_k_shot_indices(y: np.ndarray, shots_per_class: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    indices = []
    for class_id in np.unique(y):
        class_idx = np.flatnonzero(y == class_id)
        if len(class_idx) < shots_per_class:
            selected = class_idx
        else:
            selected = rng.choice(class_idx, size=shots_per_class, replace=False)
        indices.extend(selected.tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def run_classical_transfer_fbcsp(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    sfreq: float,
    target_subject: int,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate FBCSP transfer from source subjects to one target subject."""

    split = transfer_split(
        groups,
        y,
        target_subject=target_subject,
        calibration_size=calibration_size,
        random_state=random_state,
    )
    X_source, y_source = X[split.train_idx], y[split.train_idx]
    X_cal, y_cal = X[split.calibration_idx], y[split.calibration_idx]
    X_eval, y_eval = X[split.test_idx], y[split.test_idx]

    results = []
    runtime_seconds = 0.0

    zero_shot = clone(build_fbcsp_pipeline(sfreq=sfreq))
    start = time.perf_counter()
    zero_shot.fit(X_source, y_source)
    zero_pred = zero_shot.predict(X_eval)
    runtime_seconds += time.perf_counter() - start
    results.append(SubjectResult(label="zero_shot", metrics=compute_classification_metrics(y_eval, zero_pred)))

    for shot in calibration_shots:
        shot_idx = _sample_k_shot_indices(y_cal, shot, random_state=random_state + shot)
        X_shot = X_cal[shot_idx]
        y_shot = y_cal[shot_idx]
        adapted = clone(build_fbcsp_pipeline(sfreq=sfreq))
        start = time.perf_counter()
        adapted.fit(np.concatenate([X_source, X_shot], axis=0), np.concatenate([y_source, y_shot], axis=0))
        y_pred = adapted.predict(X_eval)
        runtime_seconds += time.perf_counter() - start
        results.append(
            SubjectResult(
                label=f"{shot}_shot",
                metrics=compute_classification_metrics(y_eval, y_pred),
            )
        )

    result = summarize_subject_results(results)
    result["runtime_seconds"] = float(runtime_seconds)
    return result


def run_classical_transfer_fbcsp_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    sfreq: float,
    target_subjects: tuple[int, ...] | None = None,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate FBCSP transfer across multiple target subjects."""

    subjects = target_subjects or tuple(int(subject_id) for subject_id in np.unique(groups))
    target_results: dict[str, dict[str, object]] = {}
    grouped_results: dict[str, list[SubjectResult]] = {}

    for target_subject in subjects:
        result = run_classical_transfer_fbcsp(
            X,
            y,
            groups,
            sfreq=sfreq,
            target_subject=int(target_subject),
            calibration_size=calibration_size,
            calibration_shots=calibration_shots,
            random_state=random_state,
        )
        target_label = f"S{int(target_subject)}"
        target_results[target_label] = result

        rows = result["rows"]
        if not isinstance(rows, list):
            raise ValueError("Transfer result rows must be a list.")
        for row in rows:
            label = str(row["label"])
            grouped_results.setdefault(label, []).append(
                SubjectResult(
                    label=target_label,
                    metrics=metric_from_row(row),
                )
            )

    aggregate_rows = []
    for label, results in grouped_results.items():
        summary = summarize_subject_results(results)
        aggregate_rows.append(
            SubjectResult(
                label=label,
                metrics=metric_from_summary(summary["summary"]),
            )
        )

    result = {
        "aggregate_by_setting": summarize_subject_results(aggregate_rows),
        "targets": target_results,
    }
    result["runtime_seconds"] = float(sum(float(target_results[target]["runtime_seconds"]) for target in target_results))
    return result


def run_classical_transfer_fbcsp_repeated_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    sfreq: float,
    seeds: tuple[int, ...],
    target_subjects: tuple[int, ...] | None = None,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
) -> dict[str, object]:
    """Evaluate FBCSP transfer across repeated calibration splits."""

    seed_runs = {}
    for seed in seeds:
        seed_runs[f"seed_{seed}"] = run_classical_transfer_fbcsp_sweep(
            X,
            y,
            groups,
            sfreq=sfreq,
            target_subjects=target_subjects,
            calibration_size=calibration_size,
            calibration_shots=calibration_shots,
            random_state=seed,
        )
    result = aggregate_transfer_seed_runs(seed_runs)
    result["runtime_seconds"] = float(
        sum(float(seed_run.get("runtime_seconds", 0.0)) for seed_run in seed_runs.values() if isinstance(seed_run, dict))
    )
    return result


def run_eegnet_transfer(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    target_subject: int,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    random_state: int = 42,
    training_config: TrainingConfig | None = None,
    finetune_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Evaluate EEGNet zero-shot and few-shot transfer on one target subject."""

    cfg = training_config or TrainingConfig()
    ft_cfg = finetune_config or TrainingConfig(
        batch_size=16,
        epochs=20,
        learning_rate=5e-4,
        device=cfg.device,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
        validation_split=0.0,
        early_stopping=False,
        min_epochs=20,
        patience=20,
    )
    split = transfer_split(
        groups,
        y,
        target_subject=target_subject,
        calibration_size=calibration_size,
        random_state=random_state,
    )

    X_source, y_source = X[split.train_idx], y[split.train_idx]
    X_cal, y_cal = X[split.calibration_idx], y[split.calibration_idx]
    X_eval, y_eval = X[split.test_idx], y[split.test_idx]

    pretrain_dataset, pretrain_val_dataset = build_train_validation_datasets(X_source, y_source, config=cfg)
    eval_dataset = EpochDataset(X_eval, y_eval, scaler=pretrain_dataset.scaler)

    base_model = EEGNet(
        n_channels=X.shape[1],
        n_times=X.shape[2],
        n_classes=len(np.unique(y)),
    )
    runtime_seconds = 0.0
    start = time.perf_counter()
    pretraining = fit_model(base_model, pretrain_dataset, validation_dataset=pretrain_val_dataset, config=cfg)
    base_model = pretraining.model
    runtime_seconds += time.perf_counter() - start

    results = []
    zero_pred = predict_model(base_model, eval_dataset, config=cfg)
    results.append(SubjectResult(label="zero_shot", metrics=compute_classification_metrics(y_eval, zero_pred)))
    training_histories: list[dict[str, object]] = [{"label": "source_pretraining", **pretraining.history}]

    for shot in calibration_shots:
        shot_idx = _sample_k_shot_indices(y_cal, shot, random_state=random_state + shot)
        shot_dataset = EpochDataset(X_cal[shot_idx], y_cal[shot_idx], scaler=pretrain_dataset.scaler)
        adapted_model = deepcopy(base_model)
        start = time.perf_counter()
        finetuning = fit_model(adapted_model, shot_dataset, validation_dataset=None, config=ft_cfg)
        y_pred = predict_model(finetuning.model, eval_dataset, config=cfg)
        runtime_seconds += time.perf_counter() - start
        results.append(
            SubjectResult(
                label=f"{shot}_shot",
                metrics=compute_classification_metrics(y_eval, y_pred),
            )
        )
        training_histories.append(
            {
                "label": f"{shot}_shot_finetune",
                **finetuning.history,
            }
        )

    result = summarize_subject_results(results)
    result["runtime_seconds"] = float(runtime_seconds)
    result["training_histories"] = training_histories
    return result


def run_eegnet_transfer_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    target_subjects: tuple[int, ...] | None = None,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    random_state: int = 42,
    training_config: TrainingConfig | None = None,
    finetune_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Evaluate EEGNet transfer across multiple target subjects."""

    subjects = target_subjects or tuple(int(subject_id) for subject_id in np.unique(groups))
    target_results: dict[str, dict[str, object]] = {}
    grouped_results: dict[str, list[SubjectResult]] = {}

    for target_subject in subjects:
        result = run_eegnet_transfer(
            X,
            y,
            groups,
            target_subject=int(target_subject),
            calibration_size=calibration_size,
            calibration_shots=calibration_shots,
            random_state=random_state,
            training_config=training_config,
            finetune_config=finetune_config,
        )
        target_label = f"S{int(target_subject)}"
        target_results[target_label] = result

        rows = result["rows"]
        if not isinstance(rows, list):
            raise ValueError("Transfer result rows must be a list.")
        for row in rows:
            label = str(row["label"])
            grouped_results.setdefault(label, []).append(
                SubjectResult(
                    label=target_label,
                    metrics=metric_from_row(row),
                ),
            )

    aggregate_rows = []
    for label, results in grouped_results.items():
        summary = summarize_subject_results(results)
        aggregate_rows.append(
            SubjectResult(
                label=label,
                metrics=metric_from_summary(summary["summary"]),
            )
        )

    result = {
        "aggregate_by_setting": summarize_subject_results(aggregate_rows),
        "targets": target_results,
    }
    result["runtime_seconds"] = float(sum(float(target_results[target]["runtime_seconds"]) for target in target_results))
    return result


def run_eegnet_transfer_repeated_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seeds: tuple[int, ...],
    target_subjects: tuple[int, ...] | None = None,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    training_config: TrainingConfig | None = None,
    finetune_config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Evaluate EEGNet transfer across repeated calibration splits."""

    seed_runs = {}
    for seed in seeds:
        seed_runs[f"seed_{seed}"] = run_eegnet_transfer_sweep(
            X,
            y,
            groups,
            target_subjects=target_subjects,
            calibration_size=calibration_size,
            calibration_shots=calibration_shots,
            random_state=seed,
            training_config=training_config,
            finetune_config=finetune_config,
        )
    result = aggregate_transfer_seed_runs(seed_runs)
    result["runtime_seconds"] = float(
        sum(float(seed_run.get("runtime_seconds", 0.0)) for seed_run in seed_runs.values() if isinstance(seed_run, dict))
    )
    return result
