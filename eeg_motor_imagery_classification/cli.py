"""Command-line entry point for canonical EEG experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_motor_imagery_classification.config import DatasetConfig, PreprocessingConfig
from eeg_motor_imagery_classification.constants import DEFAULT_SUBJECTS
from eeg_motor_imagery_classification.data import load_all_subject_epochs
from eeg_motor_imagery_classification.eda import export_grand_average_sensorimotor_erds, export_subject_eda_assets
from eeg_motor_imagery_classification.evaluation import format_metric_markdown_table, format_metric_table
from eeg_motor_imagery_classification.evaluation.protocols import format_transfer_sweep_markdown
from eeg_motor_imagery_classification.experiments import (
    run_classical_loso,
    run_classical_transfer_fbcsp,
    run_classical_transfer_fbcsp_repeated_sweep,
    run_classical_transfer_fbcsp_sweep,
    run_classical_within_subject_cv,
    run_eegnet_loso,
    run_eegnet_transfer,
    run_eegnet_transfer_repeated_sweep,
    run_eegnet_transfer_sweep,
    run_eegnet_within_subject_cv,
    run_riemann_loso,
    run_riemann_transfer,
    run_riemann_transfer_repeated_sweep,
    run_riemann_transfer_sweep,
    run_riemann_within_subject_cv,
)
from eeg_motor_imagery_classification.figures import export_report_assets
from eeg_motor_imagery_classification.train import TrainingConfig
from eeg_motor_imagery_classification.utils import ensure_directory, to_jsonable, write_json, write_text


def _parse_subjects(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_SUBJECTS
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_seeds(raw: str | None, default_seed: int) -> tuple[int, ...]:
    if not raw:
        return (default_seed,)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular EEG motor imagery experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=(
            "classical_baseline",
            "eegnet_baseline",
            "classical_loso",
            "eegnet_loso",
            "classical_transfer",
            "eegnet_transfer",
            "riemann_baseline",
            "riemann_loso",
            "riemann_transfer",
            "export_assets",
            "export_eda",
            "export_group_eda",
        ),
    )
    parser.add_argument("--subjects", default=None, help="Comma-separated subject IDs. Default: 1-9")
    parser.add_argument("--target-subject", type=int, default=2, help="Target subject for transfer experiments.")
    parser.add_argument("--eda-subject", type=int, default=1, help="Subject ID for subject-level EDA export.")
    parser.add_argument(
        "--all-target-subjects",
        action="store_true",
        help="Run transfer experiments for every subject as a target and aggregate by shot setting.",
    )
    parser.add_argument("--calibration-size", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for EEGNet experiments.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split used for EEGNet early stopping.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience for EEGNet validation loss.")
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stopping can trigger.")
    parser.add_argument("--disable-early-stopping", action="store_true", help="Disable EEGNet early stopping and use all epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CV splits and deep learning runs.")
    parser.add_argument(
        "--seed-list",
        default=None,
        help="Comma-separated seeds for repeated transfer evaluation. Default: use --seed only.",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Allow non-deterministic torch kernels for potentially faster deep learning runs.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for saving result artifacts.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a plain-text summary.")
    return parser


def _save_outputs(output_dir: str, result) -> None:
    directory = ensure_directory(output_dir)
    jsonable = to_jsonable(result)
    write_json(directory / "result.json", jsonable)

    if isinstance(result, dict) and "rows" in result:
        write_text(directory / "result.tsv", format_metric_table(result) + "\n")
        write_text(directory / "result.md", format_metric_markdown_table(result) + "\n")
    elif isinstance(result, dict) and "aggregate_by_setting" in result and "targets" in result:
        aggregate = result["aggregate_by_setting"]
        if not isinstance(aggregate, dict):
            raise ValueError("Transfer sweep aggregate result must be a dictionary.")
        write_text(directory / "aggregate.tsv", format_metric_table(aggregate) + "\n")
        write_text(directory / "result.md", format_transfer_sweep_markdown(result) + "\n")
    else:
        write_text(directory / "result.md", "```json\n" + json.dumps(jsonable, indent=2) + "\n```\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    subjects = _parse_subjects(args.subjects)
    seeds = _parse_seeds(args.seed_list, args.seed)
    if args.experiment == "export_assets":
        output_dir = args.output_dir or "docs/assets"
        result = export_report_assets(
            project_root=Path(__file__).resolve().parents[1],
            assets_dir=Path(__file__).resolve().parents[1] / output_dir,
        )
        if args.output_dir:
            _save_outputs(args.output_dir, result)
        print(json.dumps(to_jsonable(result), indent=2))
        return
    if args.experiment == "export_eda":
        output_dir = args.output_dir or f"docs/assets/eda_subject_{args.eda_subject}"
        result = export_subject_eda_assets(
            subject_id=args.eda_subject,
            output_dir=Path(__file__).resolve().parents[1] / output_dir,
            config=PreprocessingConfig(),
        )
        if args.output_dir:
            _save_outputs(args.output_dir, result)
        print(json.dumps(to_jsonable(result), indent=2))
        return
    if args.experiment == "export_group_eda":
        output_dir = args.output_dir or "docs/assets/group_eda"
        result = export_grand_average_sensorimotor_erds(
            subject_ids=subjects,
            output_dir=Path(__file__).resolve().parents[1] / output_dir,
            config=PreprocessingConfig(),
        )
        if args.output_dir:
            _save_outputs(args.output_dir, result)
        print(json.dumps(to_jsonable(result), indent=2))
        return

    preprocessing = PreprocessingConfig()
    _dataset_config = DatasetConfig(subjects=subjects, preprocessing=preprocessing)
    data_bundle = load_all_subject_epochs(subjects=subjects, config=preprocessing)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        early_stopping=not args.disable_early_stopping,
        patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed,
        deterministic=not args.non_deterministic,
    )

    if args.experiment == "classical_baseline":
        result = run_classical_within_subject_cv(
            data_bundle.X,
            data_bundle.y,
            sfreq=next(iter(data_bundle.metadata.values())).sfreq,
            groups=data_bundle.groups,
            random_state=args.seed,
        )
    elif args.experiment == "eegnet_baseline":
        result = run_eegnet_within_subject_cv(
            data_bundle.X,
            data_bundle.y,
            groups=data_bundle.groups,
            random_state=args.seed,
            training_config=training_config,
        )
    elif args.experiment == "riemann_baseline":
        result = run_riemann_within_subject_cv(
            data_bundle.X,
            data_bundle.y,
            groups=data_bundle.groups,
            random_state=args.seed,
        )
    elif args.experiment == "classical_loso":
        result = run_classical_loso(
            data_bundle.X,
            data_bundle.y,
            data_bundle.groups,
            sfreq=next(iter(data_bundle.metadata.values())).sfreq,
        )
    elif args.experiment == "eegnet_loso":
        result = run_eegnet_loso(
            data_bundle.X,
            data_bundle.y,
            data_bundle.groups,
            training_config=training_config,
        )
    elif args.experiment == "riemann_loso":
        result = run_riemann_loso(
            data_bundle.X,
            data_bundle.y,
            data_bundle.groups,
        )
    elif args.experiment == "classical_transfer":
        if args.all_target_subjects and len(seeds) > 1:
            result = run_classical_transfer_fbcsp_repeated_sweep(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                sfreq=next(iter(data_bundle.metadata.values())).sfreq,
                seeds=seeds,
                target_subjects=subjects,
                calibration_size=args.calibration_size,
            )
        elif args.all_target_subjects:
            result = run_classical_transfer_fbcsp_sweep(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                sfreq=next(iter(data_bundle.metadata.values())).sfreq,
                target_subjects=subjects,
                calibration_size=args.calibration_size,
                random_state=args.seed,
            )
        else:
            result = run_classical_transfer_fbcsp(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                sfreq=next(iter(data_bundle.metadata.values())).sfreq,
                target_subject=args.target_subject,
                calibration_size=args.calibration_size,
                random_state=args.seed,
            )
    elif args.experiment == "eegnet_transfer":
        if args.all_target_subjects and len(seeds) > 1:
            result = run_eegnet_transfer_repeated_sweep(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                seeds=seeds,
                target_subjects=subjects,
                calibration_size=args.calibration_size,
                training_config=training_config,
            )
        elif args.all_target_subjects:
            result = run_eegnet_transfer_sweep(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                target_subjects=subjects,
                calibration_size=args.calibration_size,
                random_state=args.seed,
                training_config=training_config,
            )
        else:
            result = run_eegnet_transfer(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                target_subject=args.target_subject,
                calibration_size=args.calibration_size,
                random_state=args.seed,
                training_config=training_config,
            )
    else:
        if args.all_target_subjects and len(seeds) > 1:
            result = run_riemann_transfer_repeated_sweep(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                seeds=seeds,
                target_subjects=subjects,
                calibration_size=args.calibration_size,
            )
        elif args.all_target_subjects:
            result = run_riemann_transfer_sweep(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                target_subjects=subjects,
                calibration_size=args.calibration_size,
                random_state=args.seed,
            )
        else:
            result = run_riemann_transfer(
                data_bundle.X,
                data_bundle.y,
                data_bundle.groups,
                target_subject=args.target_subject,
                calibration_size=args.calibration_size,
                random_state=args.seed,
            )

    if args.output_dir:
        _save_outputs(args.output_dir, result)

    if args.json:
        print(json.dumps(to_jsonable(result), indent=2))
        return

    if isinstance(result, dict) and "rows" in result:
        print(format_metric_table(result))
        return

    if isinstance(result, dict) and "aggregate_by_setting" in result:
        aggregate = result["aggregate_by_setting"]
        if not isinstance(aggregate, dict):
            raise ValueError("Transfer sweep aggregate result must be a dictionary.")
        print(format_metric_table(aggregate))
        return

    if isinstance(result, dict):
        print(json.dumps(to_jsonable(result), indent=2))
        return

    print(result)


if __name__ == "__main__":
    main()
