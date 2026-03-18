"""Protocol-level result helpers for cross-subject experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_motor_imagery_classification.evaluation.metrics import FoldMetrics, aggregate_fold_metrics


@dataclass(frozen=True)
class SubjectResult:
    """Metrics associated with one target subject or one transfer setting."""

    label: str
    metrics: FoldMetrics


def metric_from_row(row: dict[str, object]) -> FoldMetrics:
    """Convert one serialized result row back into metric form."""

    confusion = row.get("confusion_matrix")
    if confusion is None:
        confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    else:
        confusion_matrix = np.asarray(confusion, dtype=np.int64)
    return FoldMetrics(
        accuracy=float(row["accuracy"]),
        balanced_accuracy=float(row["balanced_accuracy"]),
        macro_f1=float(row["macro_f1"]),
        confusion_matrix=confusion_matrix,
    )


def metric_from_summary(summary: dict[str, object]) -> FoldMetrics:
    """Convert an aggregated summary dictionary back into metric form."""

    return FoldMetrics(
        accuracy=float(summary["accuracy_mean"]),
        balanced_accuracy=float(summary["balanced_accuracy_mean"]),
        macro_f1=float(summary["macro_f1_mean"]),
        confusion_matrix=np.asarray(summary["confusion_matrix_sum"]),
    )


def summarize_subject_results(results: list[SubjectResult]) -> dict[str, object]:
    """Aggregate per-subject results while preserving the detailed rows."""

    if not results:
        raise ValueError("At least one subject result is required.")

    aggregate = aggregate_fold_metrics([item.metrics for item in results])
    rows = [
        {
            "label": item.label,
            "accuracy": item.metrics.accuracy,
            "balanced_accuracy": item.metrics.balanced_accuracy,
            "macro_f1": item.metrics.macro_f1,
            "confusion_matrix": np.asarray(item.metrics.confusion_matrix).tolist(),
        }
        for item in results
    ]
    return {"summary": aggregate, "rows": rows}


def aggregate_transfer_seed_runs(seed_runs: dict[str, dict[str, object]]) -> dict[str, object]:
    """Aggregate repeated-seed transfer sweeps into one summary object."""

    if not seed_runs:
        raise ValueError("At least one seed run is required for repeated transfer aggregation.")

    grouped_by_setting: dict[str, list[SubjectResult]] = {}
    grouped_by_target: dict[str, dict[str, list[SubjectResult]]] = {}
    seed_summaries: dict[str, dict[str, object]] = {}

    for seed_label, seed_run in seed_runs.items():
        aggregate = seed_run.get("aggregate_by_setting")
        targets = seed_run.get("targets")
        if not isinstance(aggregate, dict) or not isinstance(targets, dict):
            raise ValueError("Each seed run must contain 'aggregate_by_setting' and 'targets'.")

        seed_summaries[seed_label] = aggregate

        for target_label, target_result in targets.items():
            if not isinstance(target_result, dict):
                raise ValueError("Each target result must be a dictionary.")
            rows = target_result.get("rows")
            if not isinstance(rows, list):
                raise ValueError("Each target result must contain a 'rows' list.")
            for row in rows:
                setting_label = str(row["label"])
                metrics = metric_from_row(row)
                grouped_by_setting.setdefault(setting_label, []).append(
                    SubjectResult(label=f"{seed_label}:{target_label}", metrics=metrics)
                )
                grouped_by_target.setdefault(target_label, {}).setdefault(setting_label, []).append(
                    SubjectResult(label=seed_label, metrics=metrics)
                )

    aggregate_rows = [
        SubjectResult(label=setting_label, metrics=metric_from_summary(summarize_subject_results(results)["summary"]))
        for setting_label, results in grouped_by_setting.items()
    ]

    target_rows: dict[str, dict[str, object]] = {}
    for target_label, grouped_settings in grouped_by_target.items():
        rows = [
            SubjectResult(label=setting_label, metrics=metric_from_summary(summarize_subject_results(results)["summary"]))
            for setting_label, results in grouped_settings.items()
        ]
        target_rows[target_label] = summarize_subject_results(rows)

    observations_by_setting = {
        setting_label: summarize_subject_results(results) for setting_label, results in grouped_by_setting.items()
    }

    return {
        "aggregate_by_setting": summarize_subject_results(aggregate_rows),
        "targets": target_rows,
        "seed_summaries": seed_summaries,
        "observations_by_setting": observations_by_setting,
        "n_seeds": len(seed_runs),
    }


def format_metric_table(result: dict[str, object]) -> str:
    """Create a compact plain-text table for README or report drafting."""

    rows = result["rows"]
    if not isinstance(rows, list):
        raise ValueError("Result rows must be a list.")

    header = "label\taccuracy\tbalanced_accuracy\tmacro_f1"
    body = [
        f"{row['label']}\t{row['accuracy']:.4f}\t{row['balanced_accuracy']:.4f}\t{row['macro_f1']:.4f}"
        for row in rows
    ]
    summary = result["summary"]
    if not isinstance(summary, dict):
        raise ValueError("Result summary must be a dictionary.")
    footer = (
        f"mean\t{summary['accuracy_mean']:.4f}\t"
        f"{summary['balanced_accuracy_mean']:.4f}\t{summary['macro_f1_mean']:.4f}"
    )
    return "\n".join([header, *body, footer])


def format_metric_markdown_table(result: dict[str, object]) -> str:
    """Create a Markdown table from subject- or shot-level rows."""

    rows = result["rows"]
    if not isinstance(rows, list):
        raise ValueError("Result rows must be a list.")

    summary = result["summary"]
    if not isinstance(summary, dict):
        raise ValueError("Result summary must be a dictionary.")

    lines = [
        "| Label | Accuracy | Balanced Accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.append(
        f"| Mean | {summary['accuracy_mean']:.4f} | "
        f"{summary['balanced_accuracy_mean']:.4f} | {summary['macro_f1_mean']:.4f} |"
    )
    return "\n".join(lines)


def format_transfer_sweep_markdown(result: dict[str, object]) -> str:
    """Create a Markdown summary for multi-target transfer sweeps."""

    aggregate = result.get("aggregate_by_setting")
    targets = result.get("targets")
    if not isinstance(aggregate, dict) or not isinstance(targets, dict):
        raise ValueError("Transfer sweep result must contain 'aggregate_by_setting' and 'targets' dictionaries.")

    lines = [
        "## Aggregate By Setting",
        "",
        format_metric_markdown_table(aggregate),
        "",
    ]

    seed_summaries = result.get("seed_summaries")
    if isinstance(seed_summaries, dict) and seed_summaries:
        lines.extend(["## Seed Aggregates", ""])
        for seed_label, seed_result in seed_summaries.items():
            if not isinstance(seed_result, dict):
                raise ValueError("Each seed summary must be a dictionary.")
            lines.extend([f"### {seed_label}", "", format_metric_markdown_table(seed_result), ""])

    for target_label, target_result in targets.items():
        if not isinstance(target_result, dict):
            raise ValueError("Each transfer target result must be a dictionary.")
        lines.extend(
            [
                f"## {target_label}",
                "",
                format_metric_markdown_table(target_result),
                "",
            ]
        )

    return "\n".join(lines).strip()
