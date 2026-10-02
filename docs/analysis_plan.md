# Analysis Plan

This plan fixes the confirmatory analysis before the cross-session results and the multi-seed EEGNet results exist. Everything not listed under "Primary comparisons" is exploratory.

- Date fixed: 2026-10-02
- Code: the commit that adds this file
- Status of the data when the plan was written:
  - Within-subject CV (sessions pooled), LOSO, and transfer results had already been computed. In those runs EEGNet was trained once with a 50-epoch budget (`docs/report.md`).
  - No cross-session result and no result with the EEGNet recipe below had been computed.

## Research question

Do the relative accuracies of a filter-bank spatial-filtering pipeline (FBCSP), a Riemannian tangent-space pipeline, and EEGNet depend on whether the model is evaluated on a new session of the same subject or on a new subject?

## Data

- BNCI 2014-001: all 9 subjects, both sessions, left- versus right-hand imagery (288 trials per subject, 144 per session)
- no subject or trial exclusions
- preprocessing as in `docs/report.md` Section 2.2

## Protocols

- **Cross-session**: train on the first session of a subject and test on the second session (recorded on a different day). This is the standard protocol for this dataset.
- **LOSO**: train on eight subjects and test on the held-out subject (both sessions).

## Models

- FBCSP + LDA, Riemann + tangent space + LDA, and the other classical baselines exactly as implemented in the codebase. No hyperparameter tuning.
- **EEGNet, recipe R**, fixed before the new runs:
  - Training: Adam, learning rate `1e-3`, at most `300` epochs, early stopping on validation loss with patience `30` and a minimum of `30` epochs, best validation epoch restored.
  - Validation split: stratified `20%` taken from the training data only.
  - Batch size: `32` when training on one subject (cross-session, within-subject CV) and `64` when training on pooled subjects (LOSO, transfer).
  - Rationale: in a sensitivity run, every within-subject fit stopped before 300 epochs, while the earlier 50-epoch budget stopped training before convergence.
  - Training seeds: `42, 43, 44, 45, 46`. The per-subject accuracy of EEGNet is the mean over these five seeds. Evaluation splits do not change across seeds.

## Outcome

Test-set accuracy per subject. The classes are balanced, so accuracy equals balanced accuracy.

## Primary comparisons (confirmatory)

All tests are two-sided, and the three tests form one family.

| ID | Protocol | Comparison |
|---|---|---|
| P1 | Cross-session | EEGNet (recipe R) vs FBCSP + LDA |
| P2 | Cross-session | Riemann + tangent space + LDA vs FBCSP + LDA |
| P3 | LOSO | EEGNet (recipe R) vs Riemann + tangent space + LDA |

P3 is a replication. Its direction (EEGNet higher) was observed in the earlier single-seed, 50-epoch LOSO run, so it is not a fully independent test.

Statistical procedure:

- Exact paired sign-flip permutation test on the nine per-subject accuracy differences (all `2^9` sign patterns). The smallest attainable p-value is `0.0039`.
- Holm adjustment across P1-P3. A comparison is called significant if its Holm-adjusted p-value is below `0.05`; the smallest attainable adjusted value is `0.0117`.
- Report for each comparison: the mean paired difference, the number of subjects favouring each model, the exact p-value, and the Holm-adjusted p-value.

These comparisons are implemented in `eeg_motor_imagery_classification/stats_report.py` (`PRIMARY_COMPARISONS`) and appear first in `outputs/statistics/statistics.md`.

## Secondary and exploratory analyses

These are reported with Holm adjustment within each family and are not used for confirmatory claims:

- all model pairs within cross-session, LOSO, and pooled-session within-subject CV
- transfer comparisons per calibration setting
- contralateral versus ipsilateral ERD/ERS at C3/C4 (four tests)
- between-seed variability of EEGNet

## Execution rules

- The primary stages are run once with the settings above: `python -m eeg_motor_imagery_classification.reproduce --stages primary`.
- A run that fails for technical reasons (crash, out of memory) is repeated with identical settings, and the repeat is documented.
- No setting is changed after seeing primary results. Any deviation from this plan is reported in `docs/report.md` together with its reason.
- The compute device and software versions are recorded in every `result.json` and listed in `statistics.md`. Results from different devices are not mixed within one comparison.
