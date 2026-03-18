"""Evaluation helpers for EEG experiments."""

from eeg_motor_imagery_classification.evaluation.metrics import (
    FoldMetrics,
    aggregate_fold_metrics,
    compute_classification_metrics,
)
from eeg_motor_imagery_classification.evaluation.protocols import (
    SubjectResult,
    aggregate_transfer_seed_runs,
    format_metric_markdown_table,
    format_metric_table,
    metric_from_row,
    metric_from_summary,
    summarize_subject_results,
)
from eeg_motor_imagery_classification.evaluation.statistics import compare_paired_result_rows, paired_permutation_test

__all__ = [
    "FoldMetrics",
    "SubjectResult",
    "aggregate_transfer_seed_runs",
    "aggregate_fold_metrics",
    "compute_classification_metrics",
    "compare_paired_result_rows",
    "format_metric_markdown_table",
    "format_metric_table",
    "metric_from_row",
    "metric_from_summary",
    "paired_permutation_test",
    "summarize_subject_results",
]
