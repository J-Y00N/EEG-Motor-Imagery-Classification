# EEG Motor Imagery Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE-EEG%20Analysis-0A7E8C)
![Dataset](https://img.shields.io/badge/Dataset-BNCI2014__001-1F6FEB)
![Protocol](https://img.shields.io/badge/Protocols-Within%20%7C%20LOSO%20%7C%20Transfer-444444)

This repository presents an EEG motor imagery classification project with unified preprocessing, explicit evaluation protocols, reproducible experiment entry points, and report-ready artifacts.
It compares classical, geometric, and deep baselines for left-versus-right motor imagery under clearly separated within-subject, subject-independent, and transfer settings.

Pre-specified results (9 subjects; exact paired sign-flip tests, Holm-adjusted across the three primary comparisons; see [docs/analysis_plan.md](docs/analysis_plan.md)):

- cross-session (train on session 1, test on session 2): `EEGNet` vs `FBCSP` `-0.055` (95% CI `-0.130` to `+0.021`, `p_Holm = 0.20`); `Riemann` vs `FBCSP` `-0.002` (`-0.072` to `+0.069`, `p_Holm = 0.98`); no significant difference
- LOSO: `EEGNet` vs `Riemann` `+0.095` (`+0.055` to `+0.136`), higher in all 9 subjects, `p_Holm = 0.012` (a replication of an earlier observation)
- exploratory: EEGNet's advantage is larger in LOSO than cross-session, but the two protocols also differ in training-set size (2304 vs 144 trials)

## Report

- full paper-style report: [docs/report.md](docs/report.md)
- pre-specified analysis plan for the confirmatory comparisons: [docs/analysis_plan.md](docs/analysis_plan.md), and a follow-up fixed before it was run: [docs/analysis_plan_addendum.md](docs/analysis_plan_addendum.md)
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
- `classical_loso_matched`, `riemann_loso_matched`, `eegnet_loso_matched`: LOSO with 144 training trials from the other subjects, tested on session 2 (addendum)

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
- EEGNet uses validation-based early stopping and restores the best validation epoch; the CLI default is `50` max epochs, and the reported results use the analysis-plan recipe (`300` max epochs, patience `30`, five training seeds)
- every `result.json` records the experiment settings (`config`) and the software/hardware environment (`environment`)
- reported EEGNet results were produced with Python `3.14.7`, MOABB `1.7.2`, MNE `1.13.2`, PyTorch `2.14.0` (Apple MPS), scikit-learn `1.9.1`, pyRiemann `0.12`; the full environment is listed in `requirements-lock.txt`

The reported results were produced with the cross-platform runner, which runs every experiment with the settings fixed in [docs/analysis_plan.md](docs/analysis_plan.md) and then regenerates the figures and statistics ([docs/reproduction.md](docs/reproduction.md)):

```bash
python -m eeg_motor_imagery_classification.reproduce --stages all --device auto
python -m eeg_motor_imagery_classification.reproduce --stages all --dry-run   # print every underlying CLI command
```

## Results

All values are accuracy; mean ± sample SD across the 9 subjects (transfer: target subjects, seeds averaged within target). EEGNet uses the pre-specified recipe (up to 300 epochs with early stopping, mean of 5 training seeds). Full tables, per-subject values, confidence intervals, and paired tests are in [docs/report.md](docs/report.md).

| Model | Cross-session | Within-subject CV (sessions pooled) | LOSO |
|---|---:|---:|---:|
| Raw Power + LDA | `0.6898 ± 0.1381` | `0.7099 ± 0.1331` | `0.6134 ± 0.0938` |
| CSP + LDA | `0.7230 ± 0.1641` | `0.7810 ± 0.1368` | `0.5907 ± 0.1148` |
| FBCSP + LDA | `0.7600 ± 0.1600` | `0.8183 ± 0.1383` | `0.5648 ± 0.0678` |
| Riemann + Tangent Space + LDA | `0.7585 ± 0.1505` | `0.7983 ± 0.1245` | `0.6285 ± 0.1039` |
| EEGNet | `0.7054 ± 0.1810` | `0.7459 ± 0.1885` | `0.7239 ± 0.1234` |

| Transfer setting | FBCSP | Riemann | EEGNet |
|---|---:|---:|---:|
| `zero_shot` | `0.5664 ± 0.0650` | `0.6331 ± 0.1056` | `0.7276 ± 0.1205` |
| `5_shot` | `0.5687 ± 0.0698` | `0.6671 ± 0.1208` | `0.7515 ± 0.1261` |
| `10_shot` | `0.5745 ± 0.0710` | `0.6740 ± 0.1232` | `0.7650 ± 0.1363` |
| `20_shot` | `0.5826 ± 0.0700` | `0.6968 ± 0.1218` | `0.7936 ± 0.1191` |
| `30_shot` | `0.5887 ± 0.0830` | `0.7106 ± 0.1327` | `0.7940 ± 0.1268` |

`k_shot` means `k` calibration trials per class. FBCSP and Riemann adapt by retraining on source plus calibration trials; EEGNet is fine-tuned on the calibration trials.

Notes:

- Cross-session is the standard protocol for this dataset; pooled within-subject CV mixes both sessions and is reported as a reference (every model is 2-6 points lower cross-session).
- Secondary comparisons are exploratory. In LOSO, EEGNet is higher than CSP and Riemann in all 9 subjects (`p_Holm = 0.039`). In transfer, EEGNet is higher than Riemann in all 9 targets at every setting, but with 15 tests the smallest attainable Holm-adjusted p-value is `0.0586`.
- EEGNet trained on a single subject's session (144 trials) can be unstable across seeds (between-seed SD up to `0.199`).

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
