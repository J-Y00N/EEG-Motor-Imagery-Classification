from __future__ import annotations

import json
from pathlib import Path

from eeg_motor_imagery_classification.figures import export_report_assets


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_assets_handles_missing_outputs_and_generates_confusion_figures(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    outputs = project_root / "outputs"
    assets = project_root / "docs" / "assets" / "generated"

    _write_json(
        outputs / "within_subject_classical" / "result.json",
        {
            "raw_power": {"summary": {"accuracy_mean": 0.5, "accuracy_std": 0.1}},
            "csp": {"summary": {"accuracy_mean": 0.7, "accuracy_std": 0.1}},
            "fbcsp": {
                "summary": {
                    "accuracy_mean": 0.8,
                    "accuracy_std": 0.1,
                    "confusion_matrix_sum": [[8, 2], [1, 9]],
                }
            },
        },
    )
    _write_json(
        outputs / "within_subject_eegnet_e30" / "result.json",
        {
            "summary": {
                "accuracy_mean": 0.7,
                "accuracy_std": 0.12,
                "confusion_matrix_sum": [[16, 4], [5, 15]],
            },
            "rows": [
                {
                    "label": "S1",
                    "accuracy": 0.7,
                    "balanced_accuracy": 0.7,
                    "macro_f1": 0.69,
                    "confusion_matrix": [[16, 4], [5, 15]],
                }
            ],
            "training_histories": [
                {
                    "label": "S1_fold_1",
                    "train_loss": [0.68, 0.55, 0.44],
                    "val_loss": [0.7, 0.6, 0.5],
                    "epochs_ran": 3,
                    "best_epoch": 3,
                    "early_stopped": False,
                    "used_validation": True,
                }
            ],
            "runtime_seconds": 150.0,
        },
    )
    _write_json(
        outputs / "loso_eegnet_e30" / "result.json",
        {
            "summary": {
                "accuracy_mean": 0.68,
                "accuracy_std": 0.15,
                "confusion_matrix_sum": [[30, 12], [10, 32]],
            },
            "rows": [
                {
                    "label": "S1",
                    "accuracy": 0.68,
                    "balanced_accuracy": 0.68,
                    "macro_f1": 0.67,
                    "confusion_matrix": [[30, 12], [10, 32]],
                }
            ],
            "training_histories": [
                {
                    "label": "S1",
                    "train_loss": [0.73, 0.61, 0.5],
                    "val_loss": [0.75, 0.66, 0.58],
                    "epochs_ran": 3,
                    "best_epoch": 3,
                    "early_stopped": False,
                    "used_validation": True,
                }
            ],
            "runtime_seconds": 100.0,
        },
    )
    _write_json(
        outputs / "transfer_riemann_all_targets_seed42_43_v2" / "result.json",
        {
            "aggregate_by_setting": {
                "summary": {
                    "accuracy_mean": 0.67,
                    "accuracy_ci95_low": 0.65,
                    "accuracy_ci95_high": 0.69,
                    "confusion_matrix_sum": [[120, 40], [35, 125]],
                },
                "rows": [
                    {"label": "zero_shot", "accuracy": 0.63},
                    {"label": "5_shot", "accuracy": 0.66},
                    {"label": "10_shot", "accuracy": 0.67},
                    {"label": "20_shot", "accuracy": 0.69},
                    {"label": "30_shot", "accuracy": 0.70},
                ],
            },
            "observations_by_setting": {
                "zero_shot": {
                    "rows": [
                        {
                            "label": "S1",
                            "accuracy": 0.63,
                            "balanced_accuracy": 0.63,
                            "macro_f1": 0.62,
                            "confusion_matrix": [[12, 8], [7, 13]],
                        }
                    ]
                }
            },
            "runtime_seconds": 120.0,
        },
    )
    _write_json(
        outputs / "transfer_eegnet_all_targets_seed42_43_e30" / "result.json",
        {
            "aggregate_by_setting": {
                "summary": {
                    "accuracy_mean": 0.74,
                    "accuracy_ci95_low": 0.71,
                    "accuracy_ci95_high": 0.76,
                    "confusion_matrix_sum": [[135, 25], [22, 138]],
                },
                "rows": [
                    {"label": "zero_shot", "accuracy": 0.69},
                    {"label": "5_shot", "accuracy": 0.72},
                    {"label": "10_shot", "accuracy": 0.75},
                    {"label": "20_shot", "accuracy": 0.76},
                    {"label": "30_shot", "accuracy": 0.77},
                ],
            },
            "observations_by_setting": {
                "zero_shot": {
                    "rows": [
                        {
                            "label": "S1",
                            "accuracy": 0.69,
                            "balanced_accuracy": 0.69,
                            "macro_f1": 0.68,
                            "confusion_matrix": [[14, 6], [5, 15]],
                        }
                    ]
                }
            },
            "runtime_seconds": 180.0,
        },
    )

    generated = export_report_assets(project_root=project_root, assets_dir=assets)

    assert Path(generated["pipeline_figure"]).exists()
    assert Path(generated["within_subject_table"]).exists()
    assert Path(generated["within_subject_confusion"]).exists()
    assert Path(generated["within_subject_eegnet_learning_curve"]).exists()
    assert Path(generated["loso_confusion"]).exists()
    assert Path(generated["loso_eegnet_learning_curve"]).exists()
    assert Path(generated["transfer_riemann_confusion"]).exists()
    assert Path(generated["transfer_confusion"]).exists()
    assert Path(generated["transfer_figure"]).exists()
    assert "within_subject_figure" in generated
    assert "loso_figure" in generated
