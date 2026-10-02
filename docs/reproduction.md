# Reproduction Protocol

This document describes how to reproduce every result and figure on macOS, Linux, or Windows, with Apple MPS, NVIDIA CUDA, or the CPU. All steps after the environment setup are the same on every platform, because the runner is a Python module rather than a shell script.

## 1. Environment

Python 3.10 or newer is required.

### macOS (Apple Silicon, MPS) and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

In `cmd.exe`, activate with `.venv\Scripts\activate.bat` instead.

### NVIDIA CUDA (Linux or Windows)

The default `torch` wheel from PyPI may not match your CUDA driver.

1. Install the CUDA build of PyTorch first, using the command that https://pytorch.org/get-started/locally/ gives for your operating system and CUDA version. It has the form `python -m pip install torch --index-url https://download.pytorch.org/whl/<cuda tag>`.
2. Then run `python -m pip install -e ".[dev]"`. pip keeps the installed CUDA build because it already satisfies the requirement.

### Check the environment

```bash
python -m pytest -q
python -c "from eeg_motor_imagery_classification.utils import environment_info; import json; print(json.dumps(environment_info(), indent=2))"
python -m pip freeze > requirements-lock.txt
```

The second command prints the Python, package, CUDA/cuDNN, and MPS information that every result also stores under `environment`.

## 2. Data

The BNCI 2014-001 files are downloaded by MOABB on first use, by default to `~/mne_data` (macOS/Linux) or `%USERPROFILE%\mne_data` (Windows). To use another location, set the `MNE_DATA` environment variable before the first run.

## 3. Device selection

Every command accepts `--device {auto,cpu,cuda,mps}`:

- `auto` (default): CUDA if available, then Apple MPS, then CPU
- `cuda` / `mps`: stop with an error if the device is not available, instead of silently falling back
- `cpu`: the most reproducible option, and the slowest for EEGNet

Classical and Riemannian models always run on the CPU and are deterministic. For EEGNet:

- Deterministic torch algorithms are requested, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set for CUDA.
- Bit-identical results are expected only for repeated runs on the same device, software versions, and seed.
- Results from different devices (for example MPS and CUDA) can differ, so a comparison should use results from one device. The device is recorded in each `result.json` and listed in `statistics.md`.

## 4. Running the protocol

```bash
python -m eeg_motor_imagery_classification.reproduce --list                      # stages and their status
python -m eeg_motor_imagery_classification.reproduce --stages primary --dry-run  # show the commands only
python -m eeg_motor_imagery_classification.reproduce --stages primary --device auto
python -m eeg_motor_imagery_classification.reproduce --stages secondary,eda,export --device auto
```

Stage groups:

| Group | Stages | Purpose |
|---|---|---|
| `primary` | cross-session classical / Riemann / EEGNet, LOSO Riemann / EEGNet | pre-specified comparisons (`docs/analysis_plan.md`) |
| `secondary` | within-subject CV, LOSO classical, transfer | exploratory results |
| `eda` | subjects 1, 2, 8 and the grand average | figures and exact ERD/ERS summaries |
| `export` | `export_assets`, `export_stats` | report tables, figures, statistics |

Runner behaviour:

- Each stage writes `outputs/<name>/result.json` and a log at `outputs/logs/<stage>.log`.
- A stage is skipped when its result already exists with the same experiment settings.
  - A result produced with different settings is moved to `outputs/_archive/<name>-<timestamp>/` and the stage runs again.
  - Classical and Riemannian results from before settings were recorded are kept, because those pipelines are deterministic.
- `--force` re-runs selected stages anyway.
- Single stages can be selected by name, for example `--stages loso_eegnet,export_stats`.

The EEGNet stages train five seeds with up to 300 epochs and dominate the run time. Expected durations depend on the hardware and are unknown in advance. As a reference, `statistics.md` lists the measured runtime of every result.

## 5. Outputs used by the report

| File | Content |
|---|---|
| `outputs/statistics/statistics.md` | primary comparisons, per-protocol descriptives and paired tests, per-subject accuracy, EEGNet between-seed SD, runtime, environment |
| `docs/assets/generated/*.png` | report figures |
| `docs/assets/eda_subject_<s>/subject_<s>_erds_summary.md` / `.csv` | exact ERD/ERS window means, peak values, and latencies of one subject |
| `docs/assets/eda_subject_<s>/subject_<s>_erds_curves.csv` | the ERD/ERS curves behind the figure |
| `docs/assets/eda_subject_<s>/subject_<s>_topomap_values.csv` | per-channel log-variance values behind the topomap |
| `docs/assets/group_eda/grand_average_erds_summary.md` | grand-average values, per-subject mean ± SEM, contralateral vs ipsilateral tests |
| `docs/assets/group_eda/grand_average_erds_curves.csv` / `grand_average_erds_per_subject.csv` | the grand-average curves (mean, SEM) and per-subject summaries |

## 6. Platform notes

- All text and JSON outputs are written as UTF-8, so they read the same on Windows.
- Paths are handled with `pathlib`, and `/` in the commands above also works in PowerShell.
- Data loading does not use worker processes, so no `if __name__ == "__main__"` guard is needed on Windows.
