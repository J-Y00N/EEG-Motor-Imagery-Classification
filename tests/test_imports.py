"""Minimal smoke tests for package-level reproducibility checks."""

from __future__ import annotations


def test_package_imports() -> None:
    import eeg_motor_imagery_classification.cli as cli
    import eeg_motor_imagery_classification.eda as eda
    import eeg_motor_imagery_classification.figures as figures
    import eeg_motor_imagery_classification.models.riemann as riemann
    import eeg_motor_imagery_classification.evaluation.protocols as protocols
    import eeg_motor_imagery_classification.evaluation.statistics as statistics
    import eeg_motor_imagery_classification.train as train

    assert cli.build_parser() is not None
    assert callable(eda.export_subject_eda_assets)
    assert callable(figures.export_report_assets)
    assert callable(riemann.build_riemann_tangent_pipeline)
    assert callable(protocols.aggregate_transfer_seed_runs)
    assert callable(statistics.compare_paired_result_rows)
    assert callable(train.seed_everything)


def test_cli_accepts_seed_list() -> None:
    import eeg_motor_imagery_classification.cli as cli

    parser = cli.build_parser()
    args = parser.parse_args(["--experiment", "riemann_transfer", "--all-target-subjects", "--seed-list", "42,43"])
    assert args.seed_list == "42,43"


def test_cli_accepts_export_assets() -> None:
    import eeg_motor_imagery_classification.cli as cli

    parser = cli.build_parser()
    args = parser.parse_args(["--experiment", "export_assets"])
    assert args.experiment == "export_assets"


def test_cli_accepts_export_eda() -> None:
    import eeg_motor_imagery_classification.cli as cli

    parser = cli.build_parser()
    args = parser.parse_args(["--experiment", "export_eda", "--eda-subject", "2"])
    assert args.experiment == "export_eda"
    assert args.eda_subject == 2


def test_cli_accepts_early_stopping_options() -> None:
    import eeg_motor_imagery_classification.cli as cli

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--experiment",
            "eegnet_baseline",
            "--epochs",
            "50",
            "--validation-split",
            "0.2",
            "--patience",
            "8",
            "--min-epochs",
            "12",
        ]
    )
    assert args.validation_split == 0.2
    assert args.patience == 8
    assert args.min_epochs == 12
