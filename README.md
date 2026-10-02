# EEG Motor Imagery Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE-EEG%20Analysis-0A7E8C)
![Dataset](https://img.shields.io/badge/Dataset-BNCI2014__001-1F6FEB)
![Protocol](https://img.shields.io/badge/Protocols-Within%20%7C%20LOSO%20%7C%20Transfer-444444)

This repository presents an EEG motor imagery classification project with unified preprocessing, explicit evaluation protocols, reproducible experiment entry points, and report-ready artifacts.
It compares classical, geometric, and deep baselines for left-versus-right motor imagery under clearly separated within-subject, subject-independent, and transfer settings.

Summary (9 subjects; mean accuracy, Holm-corrected exact paired tests):

- within-subject: `FBCSP` has the highest mean (`0.818`), but it does not differ significantly from `Riemann` (`0.798`) or `CSP` (`0.781`)
- LOSO: `EEGNet` has the highest mean (`0.707`) and is the best model for 8 of 9 held-out subjects, but this is not significant after correction
- transfer: `EEGNet > Riemann > FBCSP` at every calibration budget, again without significance after correction
- EEGNet results depend on the training budget and vary between runs; see the report for details

## Report

- full paper-style report: [docs/report.md](docs/report.md)
- pre-specified analysis plan for the confirmatory comparisons: [docs/analysis_plan.md](docs/analysis_plan.md)
- reproduction protocol for macOS / Linux / Windows and MPS / CUDA / CPU: [docs/reproduction.md](docs/reproduction.md)
- generated figures: [docs/assets/generated](docs/assets/generated)

## Scope

- dataset: `BNCI2014_001`
- task: binary motor imagery classification, `left_hand` vs `right_hand`
- classical baselines: raw power + LDA, CSP + LDA, FBCSP + LDA
- riemannian baseline: covariance + tangent space + LDA
- deep baseline: EEGNet
- protocol families: cross-session (session 1 -> 2), within-subject CV, LOSO, cross-subject transfer

## Status

Included:

- installable modules, canonical preprocessing, and split utilities
- classical, Riemannian, and EEGNet baselines
- within-subject, LOSO, and cross-subject transfer protocols
- all-target transfer runs, repeated-seed transfer checks, and artifact export
- subject-level statistics: sample SD, t-based 95% CIs, exact paired sign-flip tests with Holm correction (`export_stats`)
- cross-session evaluation, EEGNet repeated over training seeds, device selection (`--device`), and a cross-platform runner
- exact ERD/ERS summaries (CSV/Markdown) next to every EDA figure
- report figures, EDA figures, and a paper-style report in `docs/report.md`

Possible extensions:

- more transfer seeds and additional datasets to increase statistical power
- subject-alignment methods (for example Riemannian re-centering) for transfer

## Repository Structure

```text
EEG-Motor-Imagery-Classification/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── assets/
│   └── report.md
├── eeg_motor_imagery_classification/
├── notebooks/
├── outputs/
└── tests/
```

Notebook entry points:

- `notebooks/01_quickstart.ipynb`: thin execution notebook for launching packaged experiments
- `notebooks/02_results_overview.ipynb`: lightweight viewer for exported figures used in the report

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

On Windows (PowerShell) create and activate the environment with `py -3 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. For an NVIDIA GPU, install the CUDA build of PyTorch first; see [docs/reproduction.md](docs/reproduction.md).

Install developer checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Experiment Protocols

- `classical_baseline`: within-subject stratified CV for raw power, CSP, and FBCSP
- `eegnet_baseline`: within-subject stratified CV for EEGNet
- `classical_loso`: leave-one-subject-out evaluation for classical baselines
- `eegnet_loso`: leave-one-subject-out evaluation for EEGNet
- `classical_transfer`: zero-shot and few-shot transfer for FBCSP
- `eegnet_transfer`: zero-shot and few-shot transfer for EEGNet
- `riemann_baseline`: within-subject stratified CV for tangent-space Riemannian baseline
- `riemann_loso`: leave-one-subject-out evaluation for tangent-space baseline
- `riemann_transfer`: zero-shot and few-shot transfer for tangent-space baseline
- `classical_cross_session`, `riemann_cross_session`, `eegnet_cross_session`: train on session 1 and test on session 2 of each subject

These protocols are intentionally separated so within-subject, subject-independent, and adaptation claims are not mixed together.

## Usage

Run the whole protocol, or a group of stages, on any platform:

```bash
python -m eeg_motor_imagery_classification.reproduce --list
python -m eeg_motor_imagery_classification.reproduce --stages primary --device auto
```

`--device` accepts `auto`, `cpu`, `cuda`, or `mps` for every command. `--seed-list` repeats `eegnet_baseline`, `eegnet_loso`, and `eegnet_cross_session` over training seeds and averages per subject.

Individual experiments:

Run a classical within-subject baseline:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment classical_baseline
```

Run EEGNet LOSO:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment eegnet_loso \
  --epochs 50 \
  --validation-split 0.2
```

Run the Riemannian tangent-space baseline:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment riemann_baseline
```

Run transfer evaluation against subject 2:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment eegnet_transfer \
  --target-subject 2 \
  --calibration-size 0.5 \
  --epochs 50 \
  --validation-split 0.2
```

Print JSON output:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment classical_loso \
  --json
```

Save report-friendly artifacts:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment classical_transfer \
  --target-subject 2 \
  --output-dir outputs/classical_transfer_s2
```

Run a multi-target transfer sweep:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment riemann_transfer \
  --all-target-subjects \
  --output-dir outputs/transfer_riemann_all_targets
```

Run a repeated-seed transfer sweep for stability checks:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment riemann_transfer \
  --all-target-subjects \
  --seed-list 42,43,44 \
  --output-dir outputs/transfer_riemann_all_targets_seed42_43_44
```

Export report-ready tables and figures from saved results:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment export_assets \
  --output-dir docs/assets/generated
```

When saved experiment outputs are available locally, this export step also generates confusion-matrix figures for within-subject, LOSO, and per-setting transfer results, and prints the result files it used (`sources`) and those it could not find (`missing_sources`).

Compute subject-level statistics and paired model comparisons from the saved outputs:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment export_stats
```

This writes `outputs/statistics/statistics.md` and `statistics.json`.

## Reproducibility

- CV splits, transfer calibration splits, and EEGNet training default to `seed=42`; `--seed-list` repeats transfer sweeps, and each seed also sets the EEGNet training seed
- EEGNet requests deterministic torch kernels (`--non-deterministic` opts out), but runs on GPU or Apple MPS are not guaranteed to be bit-for-bit reproducible
- EEGNet uses validation-based early stopping (CLI default: `50` max epochs) and restores the best validation epoch; the analysis plan fixes `300` max epochs with patience `30` and five training seeds
- every `result.json` records the experiment settings (`config`) and the software/hardware environment (`environment`)
- reported results were produced with Python `3.14.7`, MOABB `1.7.2`, MNE `1.13.2`, PyTorch `2.14.0` (Apple MPS), scikit-learn `1.9.1`, pyRiemann `0.12`; the full environment is listed in `requirements-lock.txt`

The results currently in the report (EEGNet: one training seed, 50 max epochs) were produced with the commands below; output names are the ones the export steps look for. The next round follows the pre-specified recipe in [docs/analysis_plan.md](docs/analysis_plan.md) and is run with `python -m eeg_motor_imagery_classification.reproduce` ([docs/reproduction.md](docs/reproduction.md)).

```bash
python -m eeg_motor_imagery_classification.cli --experiment classical_baseline --seed 42 --output-dir outputs/within_subject_classical
python -m eeg_motor_imagery_classification.cli --experiment riemann_baseline --seed 42 --output-dir outputs/within_subject_riemann
python -m eeg_motor_imagery_classification.cli --experiment eegnet_baseline --epochs 50 --validation-split 0.2 --seed 42 --output-dir outputs/within_subject_eegnet

python -m eeg_motor_imagery_classification.cli --experiment classical_loso --output-dir outputs/loso_classical
python -m eeg_motor_imagery_classification.cli --experiment riemann_loso --output-dir outputs/loso_riemann
python -m eeg_motor_imagery_classification.cli --experiment eegnet_loso --epochs 50 --validation-split 0.2 --seed 42 --output-dir outputs/loso_eegnet

python -m eeg_motor_imagery_classification.cli --experiment classical_transfer --all-target-subjects --seed-list 42,43 --output-dir outputs/transfer_classical_all_targets_seed42_43
python -m eeg_motor_imagery_classification.cli --experiment riemann_transfer --all-target-subjects --seed-list 42,43 --output-dir outputs/transfer_riemann_all_targets_seed42_43
python -m eeg_motor_imagery_classification.cli --experiment eegnet_transfer --all-target-subjects --seed-list 42,43 --epochs 50 --validation-split 0.2 --output-dir outputs/transfer_eegnet_all_targets_seed42_43

python -m eeg_motor_imagery_classification.cli --experiment export_assets --output-dir docs/assets/generated
python -m eeg_motor_imagery_classification.cli --experiment export_stats
```

## Results

All values are accuracy; mean ± sample SD across the 9 subjects (transfer: target subjects, seeds averaged within target). Full tables, per-subject values, and paired tests are in [docs/report.md](docs/report.md).

| Model | Within-subject CV | LOSO |
|---|---:|---:|
| Raw Power + LDA | `0.7099 ± 0.1331` | `0.6134 ± 0.0938` |
| CSP + LDA | `0.7810 ± 0.1368` | `0.5907 ± 0.1148` |
| FBCSP + LDA | `0.8183 ± 0.1383` | `0.5648 ± 0.0678` |
| Riemann + Tangent Space + LDA | `0.7983 ± 0.1245` | `0.6285 ± 0.1039` |
| EEGNet (`50` max epochs + early stopping) | `0.6933 ± 0.1862` | `0.7068 ± 0.1481` |

| Transfer setting | FBCSP | Riemann | EEGNet |
|---|---:|---:|---:|
| `zero_shot` | `0.5664 ± 0.0650` | `0.6331 ± 0.1056` | `0.7079 ± 0.1455` |
| `5_shot` | `0.5687 ± 0.0698` | `0.6671 ± 0.1208` | `0.7195 ± 0.1567` |
| `10_shot` | `0.5745 ± 0.0710` | `0.6740 ± 0.1232` | `0.7469 ± 0.1611` |
| `20_shot` | `0.5826 ± 0.0700` | `0.6968 ± 0.1218` | `0.7569 ± 0.1560` |
| `30_shot` | `0.5887 ± 0.0830` | `0.7106 ± 0.1327` | `0.7612 ± 0.1570` |

`k_shot` means `k` calibration trials per class. FBCSP and Riemann adapt by retraining on source plus calibration trials; EEGNet is fine-tuned on the calibration trials.

Statistical notes:

- Within-subject: only Raw Power vs CSP and Raw Power vs Riemann are significant after Holm correction (`p_Holm = 0.039`); CSP, FBCSP, Riemann, and EEGNet do not differ significantly.
- LOSO: EEGNet is higher than every other model in 8 of 9 subjects (uncorrected `p = 0.012-0.020`, `p_Holm = 0.12-0.18`).
- Transfer: every ordering is consistent across settings, but with 9 targets and 15 tests the smallest attainable Holm-adjusted p-value is `0.0586`.
- With a `300`-epoch budget, EEGNet reaches `0.7423` within-subject and `0.7218` in LOSO.

## Selected References

- BNCI Horizon 2020. *001-2014: Left and right hand motor imagery*. https://bnci-horizon-2020.eu/database/data-sets
- Tangermann M, Muller KR, Aertsen A, et al. *Review of the BCI Competition IV*. Front Neurosci. 2012. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full
- Ramoser H, Muller-Gerking J, Pfurtscheller G. *Optimal spatial filtering of single trial EEG during imagined hand movement*. IEEE Trans Rehabil Eng. 2000. https://pubmed.ncbi.nlm.nih.gov/11204034/
- Ang KK, Chin ZY, Wang C, Guan C, Zhang H. *Filter Bank Common Spatial Pattern (FBCSP) in brain-computer interface*. Proc IJCNN. 2008. https://pubmed.ncbi.nlm.nih.gov/19963675/
- Barachant A, Bonnet S, Congedo M, Jutten C. *Multiclass brain-computer interface classification by Riemannian geometry*. IEEE Trans Biomed Eng. 2012. https://pubmed.ncbi.nlm.nih.gov/22010143/
- Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*. J Neural Eng. 2018. https://pubmed.ncbi.nlm.nih.gov/29932424/
- Jayaram V, Barachant A. *MOABB: trustworthy algorithm benchmarking for BCIs*. J Neural Eng. 2018. https://pubmed.ncbi.nlm.nih.gov/30177583/
- Gramfort A, Luessi M, Larson E, et al. *MNE software for processing MEG and EEG data*. Neuroimage. 2013. https://pubmed.ncbi.nlm.nih.gov/24161808/

## Notes

- Environment files such as `.venv/` and `.vscode/` are intentionally not included.
