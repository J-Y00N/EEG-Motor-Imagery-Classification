# EEG Motor Imagery Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![MNE](https://img.shields.io/badge/MNE-EEG%20Analysis-0A7E8C)
![Dataset](https://img.shields.io/badge/Dataset-BNCI2014__001-1F6FEB)
![Protocol](https://img.shields.io/badge/Protocols-Within%20%7C%20LOSO%20%7C%20Transfer-444444)

This repository presents an EEG motor imagery classification project with unified preprocessing, explicit evaluation protocols, reproducible experiment entry points, and report-ready artifacts.
It compares classical, geometric, and deep baselines for left-versus-right motor imagery under clearly separated within-subject, subject-independent, and transfer settings.

Interpretation:

- within-subject: `FBCSP` is the strongest verified baseline
- LOSO: `EEGNet` is the strongest verified subject-independent baseline
- cross-subject transfer: `EEGNet` is the strongest verified transfer model, with `Riemann` as the strongest non-deep alternative

## Scope

- dataset: `BNCI2014_001`
- task: binary motor imagery classification, `left_hand` vs `right_hand`
- classical baselines: raw power + LDA, CSP + LDA, FBCSP + LDA
- riemannian baseline: covariance + tangent space + LDA
- deep baseline: EEGNet
- protocol families: within-subject CV, LOSO, cross-subject transfer

## Status

Included:

- installable modules, canonical preprocessing, and split utilities
- classical, Riemannian, and EEGNet baselines
- within-subject, LOSO, and cross-subject transfer protocols
- all-target transfer runs, repeated-seed transfer checks, and artifact export
- report figures, EDA figures, and a paper-style report in `docs/report.md`

Possible extensions:

- broader repeated-seed transfer sweeps beyond the current verified seeds
- deeper statistical testing and sensitivity analysis
- optional expansion of interpretability figures and harder transfer variants such as domain-adaptation extensions

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

Install developer checks:

```bash
python -m pip install -e ".[dev]"
pytest tests/test_imports.py
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

These protocols are intentionally separated so within-subject, subject-independent, and adaptation claims are not mixed together.

## Usage

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

When saved experiment outputs are available locally, this export step also generates representative confusion-matrix figures for within-subject, LOSO, and repeated-seed transfer summaries.

## Reproducibility

- classical CV splits and transfer calibration splits default to `seed=42`
- repeated transfer evaluation can be enabled with `--seed-list`
- EEGNet runs seed Python, NumPy, and torch via [`train.py`](eeg_motor_imagery_classification/train.py)
- deterministic torch algorithms are enabled by default for repeatable deep-learning runs
- EEGNet now uses validation-based early stopping by default, with `50` max epochs and the best validation epoch restored after training
- you can override the seed with `--seed` and opt out of deterministic kernels with `--non-deterministic`

Recommended reproduction flow:

```bash
python -m eeg_motor_imagery_classification.cli \
  --experiment riemann_baseline \
  --seed 42 \
  --output-dir outputs/repro_riemann_within

python -m eeg_motor_imagery_classification.cli \
  --experiment eegnet_loso \
  --epochs 50 \
  --batch-size 64 \
  --validation-split 0.2 \
  --seed 42 \
  --output-dir outputs/repro_eegnet_loso

python -m eeg_motor_imagery_classification.cli \
  --experiment riemann_transfer \
  --all-target-subjects \
  --seed-list 42,43 \
  --output-dir outputs/repro_transfer_riemann_seed42_43
```

## Sanity Check

The following numbers are a verified subject-level sanity check on `subject 1` using the project pipeline:

| Model | Protocol | Setting | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| Raw Power + LDA | within-subject CV | default | `0.5000` | `0.5000` | `0.3333` |
| CSP + LDA | within-subject CV | default | `0.8368` | `0.8371` | `0.8353` |
| FBCSP + LDA | within-subject CV | default | `0.9132` | `0.9137` | `0.9129` |
| Riemann + Tangent Space + LDA | within-subject CV | default | `0.8368` | `0.8374` | `0.8360` |
| EEGNet | within-subject CV | `5` epochs verification run | `0.5381` | `0.5365` | `0.5039` |

These numbers are sanity-check references, not the main all-subject result tables.

## Main Results

The `within-subject CV` section below summarizes the classical baselines, the Riemannian baseline, and the all-subject `EEGNet` baseline under one shared protocol.

Summary:

| Model | Protocol | Subjects | Accuracy Mean ± Std | Balanced Accuracy Mean ± Std | Macro F1 Mean ± Std |
|---|---|---:|---:|---:|---:|
| Raw Power + LDA | within-subject CV | `9` | `0.5355 ± 0.0625` | `0.5364 ± 0.0620` | `0.4210 ± 0.1162` |
| CSP + LDA | within-subject CV | `9` | `0.7810 ± 0.1290` | `0.7810 ± 0.1287` | `0.7793 ± 0.1303` |
| FBCSP + LDA | within-subject CV | `9` | `0.8183 ± 0.1304` | `0.8184 ± 0.1304` | `0.8175 ± 0.1310` |
| Riemann + Tangent Space + LDA | within-subject CV | `9` | `0.7956 ± 0.1171` | `0.7956 ± 0.1171` | `0.7946 ± 0.1176` |
| EEGNet | within-subject CV, `50` max epochs + early stopping | `9` | `0.6857 ± 0.1835` | `0.6859 ± 0.1832` | `0.6566 ± 0.2124` |

The completed full runs show the same overall pattern as the verification snapshot: `raw power` stays near chance, `CSP` and `Riemann` are strong, `FBCSP` is currently the strongest verified classical baseline, and the current `EEGNet` recipe with validation-based early stopping still does not surpass the best classical methods under the same within-subject protocol.

Subject-level spread is substantial. In practice, subjects such as `S2` remain difficult across model families, whereas subjects such as `S8` are much easier, which is why this project treats per-subject reporting and protocol separation as first-class requirements rather than optional extras.

## Completed LOSO Run

The classical, `Riemann + Tangent Space + LDA`, and `EEGNet` LOSO results give a subject-independent comparison table under one shared protocol.

| Model | Protocol | Subjects | Accuracy Mean ± Std | Balanced Accuracy Mean ± Std | Macro F1 Mean ± Std |
|---|---|---:|---:|---:|---:|
| Raw Power + LDA | LOSO | `9` | `0.5270 ± 0.0517` | `0.5270 ± 0.0517` | `0.4045 ± 0.1043` |
| CSP + LDA | LOSO | `9` | `0.5907 ± 0.1082` | `0.5907 ± 0.1082` | `0.5258 ± 0.1516` |
| FBCSP + LDA | LOSO | `9` | `0.5648 ± 0.0639` | `0.5648 ± 0.0639` | `0.5104 ± 0.1143` |
| Riemann + Tangent Space + LDA | LOSO | `9` | `0.6258 ± 0.0989` | `0.6258 ± 0.0989` | `0.5895 ± 0.1307` |
| EEGNet | LOSO, `50` max epochs + early stopping | `9` | `0.6971 ± 0.1358` | `0.6971 ± 0.1358` | `0.6829 ± 0.1481` |

Under LOSO, `EEGNet` is currently the strongest verified subject-independent baseline, followed by `Riemann`, while `CSP` slightly outperforms `FBCSP`. That ordering differs from the within-subject setting, which is exactly why protocol separation matters.

Taken together with the EDA exports, this also makes the qualitative interpretation clearer: some subjects express cleaner sensorimotor lateralization than others, and that variability propagates into the downstream decoding tables. The key protocol message is simple: within-subject `FBCSP` is strongest in the subject-specific regime, but under strict subject-independent evaluation `EEGNet` rises to the top among the verified baselines.

## Completed Transfer Runs

Transfer evaluation is now verified across all available target subjects, using `zero_shot`, `5_shot`, `10_shot`, `20_shot`, and `30_shot` adaptation settings.

All-target transfer summary:

| Setting | FBCSP | Riemann | EEGNet |
|---|---:|---:|---:|
| `zero_shot` | `0.5640` | `0.6366` | `0.6960` |
| `5_shot` | `0.5687` | `0.6744` | `0.7145` |
| `10_shot` | `0.5772` | `0.6829` | `0.7492` |
| `20_shot` | `0.5818` | `0.7006` | `0.7701` |
| `30_shot` | `0.5903` | `0.7191` | `0.7670` |

Aggregate summaries:

| Model | Targets | Accuracy Mean ± Std | Balanced Accuracy Mean ± Std | Macro F1 Mean ± Std |
|---|---|---:|---:|---:|
| FBCSP Transfer | `all` | `0.5764 ± 0.0099` | `0.5764 ± 0.0099` | `0.5211 ± 0.0112` |
| Riemann Transfer | `all` | `0.6827 ± 0.0280` | `0.6827 ± 0.0280` | `0.6675 ± 0.0399` |
| EEGNet Transfer, `seed 42`, `50` max epochs + early stopping | `all` | `0.7394 ± 0.0293` | `0.7394 ± 0.0293` | `0.7269 ± 0.0371` |

Across all verified targets, `EEGNet` is the strongest transfer baseline, while `Riemann` is clearly stronger than `FBCSP` and improves steadily as calibration shots increase. The all-target view is more reliable than any single-target case and is therefore the main basis for interpretation.

Single-target `S2` remains available as a reference case, but it should now be treated as an illustrative example rather than the main transfer conclusion.

## Repeated-Seed Transfer Check

To test whether the transfer ranking depends too heavily on one calibration split, repeated-seed all-target transfer sweeps were run for `seed=42,43`.

| Model | Seeds | Accuracy Mean | 95% CI |
|---|---:|---:|---:|
| FBCSP Transfer | `2` | `0.5762` | `[0.5688, 0.5835]` |
| Riemann Transfer | `2` | `0.6756` | `[0.6511, 0.7001]` |
| EEGNet Transfer, `50` max epochs + early stopping | `2` | `0.7411` | `[0.7150, 0.7672]` |

Repeated-seed paired comparisons also favored the stronger models consistently:

- `Riemann` beat `FBCSP` at every shot setting, with permutation-test `p` values from about `0.0438` down to `0.0006`.
- `EEGNet` beat `Riemann` at every shot setting, with updated permutation-test `p` values from about `0.0234` down to `0.0005`.
- `EEGNet` stayed above `FBCSP` at every shot setting in the repeated-seed sweep.

These repeated-seed results make the transfer interpretation more robust: `EEGNet` does not lead on only one favorable split, and `Riemann` remains a strong non-deep transfer baseline.

## Representative Runtime Reference

The regenerated representative runs also provide a simple wall-clock reference for model cost on the same local environment. These values are useful for comparing practical cost within this repository, but they should not be treated as hardware-independent benchmark claims.

| Protocol | Representative Model | Runtime (s) | Runtime (min) |
|---|---|---:|---:|
| within-subject | `FBCSP` | `405.20` | `6.75` |
| LOSO | `EEGNet` | `685.16` | `11.42` |
| repeated-seed transfer | `Riemann` | `171.50` | `2.86` |
| repeated-seed transfer | `EEGNet` | `1255.64` | `20.93` |

In short, `FBCSP` remains the strongest verified within-subject model, `EEGNet` is the strongest verified LOSO and transfer model, and `Riemann` offers the lightest representative transfer run among the stronger transfer baselines.

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
