"""Cross-platform runner for the full reproduction protocol (macOS, Linux, Windows).

Usage examples::

    python -m eeg_motor_imagery_classification.reproduce --list
    python -m eeg_motor_imagery_classification.reproduce --device auto --stages primary
    python -m eeg_motor_imagery_classification.reproduce --device cuda --stages all --dry-run

Each stage calls the CLI with the current Python interpreter, writes a log to
``outputs/logs/<stage>.log``, and is skipped when ``outputs/<name>/result.json`` already exists
with the same experiment settings. A result produced with different settings is moved to
``outputs/_archive/`` before the stage runs again, so nothing is overwritten silently.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from eeg_motor_imagery_classification.train import DEVICE_CHOICES
from eeg_motor_imagery_classification.utils import ensure_directory, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pre-specified EEGNet recipe (docs/analysis_plan.md): long budget with early stopping,
# repeated over five training seeds. Single-subject training sets use batch 32, pooled ones 64.
EEGNET_SEEDS = "42,43,44,45,46"
EEGNET_COMMON = ["--epochs", "300", "--patience", "30", "--min-epochs", "30", "--validation-split", "0.2", "--seed", "42"]
EEGNET_SUBJECT = [*EEGNET_COMMON, "--batch-size", "32", "--seed-list", EEGNET_SEEDS]
EEGNET_POOLED = [*EEGNET_COMMON, "--batch-size", "64", "--seed-list", EEGNET_SEEDS]
TRANSFER_SEEDS = ["--all-target-subjects", "--seed-list", "42,43"]

# Settings that define a result; the device and output paths are deliberately excluded.
COMPARED_SETTINGS = ("experiment", "epochs", "patience", "min_epochs", "batch_size", "validation_split",
                     "learning_rate", "seed", "seed_list", "all_target_subjects", "calibration_size", "subjects",
                     "disable_early_stopping")


@dataclass(frozen=True)
class Stage:
    name: str
    group: str
    args: list[str]
    output: str | None = None  # outputs/<output>/result.json
    marker: str | None = None  # file whose existence marks a finished non-result stage
    deterministic: bool = True  # classical/Riemann results do not depend on device or torch seeds


STAGES: list[Stage] = [
    # Primary (pre-specified confirmatory comparisons)
    Stage("cross_session_classical", "primary", ["--experiment", "classical_cross_session"], output="cross_session_classical"),
    Stage("cross_session_riemann", "primary", ["--experiment", "riemann_cross_session"], output="cross_session_riemann"),
    Stage("cross_session_eegnet", "primary", ["--experiment", "eegnet_cross_session", *EEGNET_SUBJECT], output="cross_session_eegnet", deterministic=False),
    Stage("loso_riemann", "primary", ["--experiment", "riemann_loso"], output="loso_riemann"),
    Stage("loso_eegnet", "primary", ["--experiment", "eegnet_loso", *EEGNET_POOLED], output="loso_eegnet", deterministic=False),
    # Secondary (exploratory)
    Stage("within_classical", "secondary", ["--experiment", "classical_baseline", "--seed", "42"], output="within_subject_classical"),
    Stage("within_riemann", "secondary", ["--experiment", "riemann_baseline", "--seed", "42"], output="within_subject_riemann"),
    Stage("within_eegnet", "secondary", ["--experiment", "eegnet_baseline", *EEGNET_SUBJECT], output="within_subject_eegnet", deterministic=False),
    Stage("loso_classical", "secondary", ["--experiment", "classical_loso"], output="loso_classical"),
    Stage("transfer_fbcsp", "secondary", ["--experiment", "classical_transfer", *TRANSFER_SEEDS], output="transfer_classical_all_targets_seed42_43"),
    Stage("transfer_riemann", "secondary", ["--experiment", "riemann_transfer", *TRANSFER_SEEDS], output="transfer_riemann_all_targets_seed42_43"),
    Stage("transfer_eegnet", "secondary",
          ["--experiment", "eegnet_transfer", *TRANSFER_SEEDS, *EEGNET_COMMON, "--batch-size", "64"],
          output="transfer_eegnet_all_targets_seed42_43", deterministic=False),
    # Exploratory figures and exact ERD/ERS summaries
    *[
        Stage(f"eda_subject_{subject}", "eda", ["--experiment", "export_eda", "--eda-subject", str(subject)],
              marker=f"docs/assets/eda_subject_{subject}/subject_{subject}_erds_summary.csv")
        for subject in (1, 2, 8)
    ],
    Stage("eda_group", "eda", ["--experiment", "export_group_eda"], marker="docs/assets/group_eda/grand_average_erds_summary.md"),
    # Aggregation (always re-run)
    Stage("export_assets", "export", ["--experiment", "export_assets", "--output-dir", "docs/assets/generated"]),
    Stage("export_stats", "export", ["--experiment", "export_stats"]),
]
GROUPS = ("primary", "secondary", "eda", "export")


def _parse_cli_settings(args: list[str]) -> dict[str, object]:
    from eeg_motor_imagery_classification.cli import build_parser

    return vars(build_parser().parse_args(args))


def _result_status(stage: Stage) -> str:
    """Return "missing", "current", or "stale" for a stage with a result.json output."""

    path = PROJECT_ROOT / "outputs" / stage.output / "result.json"
    if not path.exists():
        return "missing"
    saved = read_json(path).get("config")
    if saved is None:
        # Results written before settings were recorded: deterministic pipelines are still valid.
        return "current" if stage.deterministic else "stale"
    expected = _parse_cli_settings(stage.args)
    same = all(str(saved.get(key)) == str(expected.get(key)) for key in COMPARED_SETTINGS)
    return "current" if same else "stale"


def _archive(stage: Stage) -> Path:
    source = PROJECT_ROOT / "outputs" / stage.output
    target = ensure_directory(PROJECT_ROOT / "outputs" / "_archive") / f"{stage.output}-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(source), str(target))
    return target


def _run(stage: Stage, device: str, dry_run: bool) -> int:
    command = [sys.executable, "-m", "eeg_motor_imagery_classification.cli", *stage.args, "--device", device]
    if stage.output:
        command += ["--output-dir", f"outputs/{stage.output}"]
    print(f"[{stage.name}] {' '.join(command)}", flush=True)
    if dry_run:
        return 0
    log_path = ensure_directory(PROJECT_ROOT / "outputs" / "logs") / f"{stage.name}.log"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
        command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    print(f"[{stage.name}] exit {code} after {time.perf_counter() - start:.0f} s (log: {log_path})", flush=True)
    return code


def select_stages(selection: str) -> list[Stage]:
    names = [part.strip() for part in selection.split(",") if part.strip()]
    if "all" in names:
        return list(STAGES)
    unknown = [n for n in names if n not in GROUPS and n not in {s.name for s in STAGES}]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}. Use --list to see the available stages.")
    return [s for s in STAGES if s.group in names or s.name in names]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the reproduction protocol stage by stage.")
    parser.add_argument("--stages", default="all", help=f"Comma-separated groups {GROUPS} and/or stage names, or 'all'.")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES, help="Torch device for EEGNet stages.")
    parser.add_argument("--force", action="store_true", help="Re-run stages even when a current result exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    parser.add_argument("--list", action="store_true", help="List stages with their status and exit.")
    args = parser.parse_args(argv)

    stages = select_stages(args.stages)
    if args.list:
        for stage in stages:
            status = _result_status(stage) if stage.output else ("done" if stage.marker and (PROJECT_ROOT / stage.marker).exists() else "-")
            print(f"{stage.group:9s} {stage.name:26s} {status}")
        return

    for stage in stages:
        if not args.force and stage.output and _result_status(stage) == "current":
            print(f"[{stage.name}] skipped: current result exists")
            continue
        if not args.force and stage.marker and (PROJECT_ROOT / stage.marker).exists():
            print(f"[{stage.name}] skipped: {stage.marker} exists")
            continue
        if stage.output and (PROJECT_ROOT / "outputs" / stage.output).exists() and not args.dry_run:
            print(f"[{stage.name}] archived previous result to {_archive(stage)}")
        if _run(stage, args.device, args.dry_run) != 0:
            raise SystemExit(f"Stage {stage.name} failed; see outputs/logs/{stage.name}.log")


if __name__ == "__main__":
    main()
