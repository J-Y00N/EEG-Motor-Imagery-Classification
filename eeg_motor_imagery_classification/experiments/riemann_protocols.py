"""LOSO and transfer protocols for the Riemannian tangent-space baseline."""

from __future__ import annotations

import time
import numpy as np
from sklearn.base import clone

from eeg_motor_imagery_classification.data.splits import leave_one_subject_out, transfer_split
from eeg_motor_imagery_classification.evaluation.metrics import compute_classification_metrics
from eeg_motor_imagery_classification.evaluation.protocols import (
    SubjectResult,
    aggregate_transfer_seed_runs,
    metric_from_row,
    metric_from_summary,
    summarize_subject_results,
)
from eeg_motor_imagery_classification.models import build_riemann_tangent_pipeline


def _sample_k_shot_indices(y: np.ndarray, shots_per_class: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    indices = []
    for class_id in np.unique(y):
        class_idx = np.flatnonzero(y == class_id)
        selected = class_idx if len(class_idx) < shots_per_class else rng.choice(class_idx, size=shots_per_class, replace=False)
        indices.extend(selected.tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def run_riemann_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    """Evaluate the tangent-space baseline under LOSO splits."""

    subject_results: list[SubjectResult] = []
    pipeline = build_riemann_tangent_pipeline()
    runtime_seconds = 0.0

    for split in leave_one_subject_out(groups):
        subject_id = int(groups[split.test_idx][0])
        start = time.perf_counter()
        model = clone(pipeline)
        model.fit(X[split.train_idx], y[split.train_idx])
        y_pred = model.predict(X[split.test_idx])
        runtime_seconds += time.perf_counter() - start
        subject_results.append(
            SubjectResult(label=f"S{subject_id}", metrics=compute_classification_metrics(y[split.test_idx], y_pred))
        )

    result = summarize_subject_results(subject_results)
    result["runtime_seconds"] = float(runtime_seconds)
    return result


def run_riemann_transfer(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    target_subject: int,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate zero-shot and few-shot tangent-space transfer on one target subject."""

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
    pipeline = build_riemann_tangent_pipeline()
    runtime_seconds = 0.0

    zero_shot = clone(pipeline)
    start = time.perf_counter()
    zero_shot.fit(X_source, y_source)
    zero_pred = zero_shot.predict(X_eval)
    runtime_seconds += time.perf_counter() - start
    results.append(SubjectResult(label="zero_shot", metrics=compute_classification_metrics(y_eval, zero_pred)))

    for shot in calibration_shots:
        shot_idx = _sample_k_shot_indices(y_cal, shot, random_state + shot)
        model = clone(pipeline)
        start = time.perf_counter()
        model.fit(
            np.concatenate([X_source, X_cal[shot_idx]], axis=0),
            np.concatenate([y_source, y_cal[shot_idx]], axis=0),
        )
        y_pred = model.predict(X_eval)
        runtime_seconds += time.perf_counter() - start
        results.append(SubjectResult(label=f"{shot}_shot", metrics=compute_classification_metrics(y_eval, y_pred)))

    result = summarize_subject_results(results)
    result["runtime_seconds"] = float(runtime_seconds)
    return result


def run_riemann_transfer_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    target_subjects: tuple[int, ...] | None = None,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate tangent-space transfer across multiple target subjects."""

    subjects = target_subjects or tuple(int(subject_id) for subject_id in np.unique(groups))
    target_results: dict[str, dict[str, object]] = {}
    grouped_results: dict[str, list[SubjectResult]] = {}

    for target_subject in subjects:
        result = run_riemann_transfer(
            X,
            y,
            groups,
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


def run_riemann_transfer_repeated_sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seeds: tuple[int, ...],
    target_subjects: tuple[int, ...] | None = None,
    calibration_size: float = 0.5,
    calibration_shots: tuple[int, ...] = (5, 10, 20, 30),
) -> dict[str, object]:
    """Evaluate tangent-space transfer across repeated calibration splits."""

    seed_runs = {}
    for seed in seeds:
        seed_runs[f"seed_{seed}"] = run_riemann_transfer_sweep(
            X,
            y,
            groups,
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
