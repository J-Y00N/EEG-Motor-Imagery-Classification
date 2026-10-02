# Experiment Report

## Abstract

This report compares five decoders of left- versus right-hand motor imagery on BNCI 2014-001 (nine subjects): raw log-variance power, CSP, FBCSP, a Riemannian tangent-space model, and EEGNet. Three comparisons were fixed in an analysis plan before the corresponding results were computed. In cross-session evaluation (train on one recording day, test on the other), EEGNet did not differ significantly from FBCSP: the mean difference was −5.5 percentage points (95% CI −13.0 to +2.1; Holm-adjusted p = 0.20). The Riemannian model and FBCSP were practically identical (−0.2 points, 95% CI −7.2 to +6.9; p = 0.98). In leave-one-subject-out (LOSO) evaluation, EEGNet was more accurate than the Riemannian model in all nine subjects (+9.5 points, 95% CI +5.5 to +13.6; Holm-adjusted p = 0.012); this comparison replicates a direction seen in earlier runs. A follow-up fixed before it was run trained the cross-subject models on only 144 trials, as many as one session, and tested them on the same trials as the cross-session models. With this training set, EEGNet was close to chance (0.534) and below the Riemannian model in all nine subjects. Its standing relative to Riemann and FBCSP no longer differed between the two protocols (Holm-adjusted p = 0.94 for both). EEGNet's advantage in LOSO therefore depends on the large pooled training set rather than on cross-subject evaluation as such.

Keywords: EEG motor imagery classification, BNCI2014_001, CSP, FBCSP, Riemannian geometry, EEGNet, cross-session, LOSO, transfer learning

## 1. Introduction

The project asks whether the relative accuracy of a filter-bank spatial-filtering pipeline (FBCSP), a Riemannian tangent-space pipeline, and EEGNet depends on what a decoder has to generalize to:

- a new recording session of the same subject (cross-session)
- a new subject (LOSO)
- a new subject with a few calibration trials (transfer)

This revision corrects design errors of an earlier version of the project (Appendix B) and restricts additional analyses to those needed to answer this question. Each claim is tied to the protocol in which it was measured. Confirmatory claims are restricted to the comparisons fixed in [`docs/analysis_plan.md`](analysis_plan.md) before the corresponding results existed. All other results are exploratory.

## 2. Data and Method

### 2.1 Dataset

- source: `BNCI2014_001` (BCI Competition IV 2a), loaded with MOABB 1.7.2
- subjects `1-9`, two sessions per subject recorded on different days, six runs per session
- classes: `left_hand` and `right_hand` only, giving `288` trials per subject (`144` per session, balanced classes)
- `22` EEG channels at `250 Hz`; MOABB converts the signals from microvolts to volts
- MOABB places each trial annotation at cue onset (its dataset interval is `[2, 6] s` from trial start), so `0 s` in this report is the cue
- no subject or trial was excluded

### 2.2 Preprocessing

One preprocessing path is shared by all models:

- `50 Hz` notch and `8-32 Hz` FIR band-pass filtering on the continuous runs
- epochs from `0.0` to `4.0 s` after the cue, without baseline correction

Average referencing and ICA are available in the configuration but disabled.

### 2.3 Models

- **Raw Power + LDA**: per-channel log-variance, standardization, shrinkage LDA
- **CSP + LDA**: four CSP components with log-power features, shrinkage LDA
- **FBCSP + LDA**: six `4 Hz` bands between `8` and `32 Hz`, four CSP components per band, the eight best features by ANOVA F-score selected inside the training pipeline, shrinkage LDA. This is a simplified variant of Ang et al. (2008), which used a wider band range and mutual-information-based selection.
- **Riemann + Tangent Space + LDA**: OAS covariance estimation, tangent-space projection at the Riemannian mean of the training covariances, standardization, shrinkage LDA
- **EEGNet**: `F1 = 8`, `D = 2`, `F2 = 16`, temporal kernel of `64` samples, dropout `0.5`, applied to `250 Hz` input without resampling and without the max-norm constraints of the original architecture. Inputs are standardized per channel with training-data statistics. All reported EEGNet results use **recipe R** from the analysis plan:
  - Adam with learning rate `1e-3`, at most `300` epochs
  - early stopping on a stratified `20%` validation split of the training data (patience `30`, minimum `30` epochs), with the best validation epoch restored
  - batch size `32` when training on one subject and `64` when training on pooled subjects
  - five training seeds (`42-46`); per-subject accuracy is the mean over seeds, and evaluation splits are identical across seeds

The classical and Riemannian models were not tuned. EEGNet runs used Apple MPS; the other models run on the CPU and are deterministic.

## 3. Evaluation

### 3.1 Protocols

- **Cross-session**: train on the first session of a subject (`144` trials) and test on the second session (`144` trials). This is the standard protocol for this dataset.
- **Within-subject CV (sessions pooled)**: stratified, shuffled 5-fold cross-validation over all `288` trials of a subject. Trials from both sessions appear in training and test folds, so the result is a session-pooled reference, not a test of transfer to a new session.
- **LOSO**: train on eight subjects (`2304` trials) and test on all `288` trials of the held-out subject.
- **Matched LOSO** (addendum): train on `144` trials from the other eight subjects (9 per class per subject, drawn with seeds `42-46`) and test on session 2 of the held-out subject. The training-set size and the test trials are the same as in cross-session evaluation; only the source of the training data differs. EEGNet uses recipe R with batch size `32`.
- **Cross-subject transfer**:
  - For each target subject, the target trials are split 50/50 into a calibration pool and an evaluation set of `144` trials.
  - A `k_shot` setting draws `k` calibration trials per class (`k = 5, 10, 20, 30`); `zero_shot` uses source subjects only.
  - FBCSP and Riemann adapt by retraining on the source trials plus the `2k` target trials. EEGNet (recipe R, batch size `64`) is pretrained on the source subjects and then fine-tuned on the `2k` target trials only (all layers, `20` epochs, learning rate `5e-4`, batch size `16`).
  - Seeds `42` and `43` set the calibration split, the shot sampling, and the EEGNet training seed. Seeds are averaged within each target.

Because the adaptation mechanism differs between EEGNet and the other two models, transfer comparisons measure model and adaptation strategy together.

![Evaluation pipeline overview](assets/generated/evaluation_pipeline.png)

*Figure 1. Evaluation pipeline. One preprocessing path feeds all models; within-subject, cross-session, and LOSO evaluation use all five models, and transfer uses FBCSP, Riemann, and EEGNet.*

### 3.2 Analysis plan and statistics

- The unit of analysis is the subject (`n = 9`); in transfer it is the target subject after averaging seeds.
- Summaries report the mean, the sample standard deviation (`ddof = 1`), and a t-distribution 95% confidence interval.
- Model differences are tested with exact two-sided paired sign-flip permutation tests on per-subject accuracy. They are reported with the mean paired difference and its t-based 95% confidence interval.

Primary comparisons, fixed in the analysis plan and Holm-adjusted as one family of three tests:

| ID | Protocol | Comparison |
|---|---|---|
| P1 | Cross-session | EEGNet vs FBCSP + LDA |
| P2 | Cross-session | Riemann + Tangent Space + LDA vs FBCSP + LDA |
| P3 | LOSO | EEGNet vs Riemann + Tangent Space + LDA |

- P3 had been observed in the same direction in earlier single-seed runs, so it is a replication rather than an independent test.
- With `n = 9`, the smallest attainable exact p-value is `2 / 2^9 = 0.0039`. For the three primary tests, the smallest attainable Holm-adjusted value is `0.0117`.

Secondary analyses are Holm-adjusted within their own families and are exploratory:

- all `10` model pairs per protocol
- `15` transfer tests
- `4` lateralization tests
- `2` protocol-by-model tests
- the `2` addendum tests A1-A2 ([`docs/analysis_plan_addendum.md`](analysis_plan_addendum.md)), fixed before the matched runs, with their own Holm family and a pre-stated interpretation rule

## 4. Results

### 4.1 Primary comparisons

| ID | Protocol | Comparison | Mean difference | 95% CI | Subjects A / B higher | p (exact) | p (Holm) |
|---|---|---|---:|---:|---:|---:|---:|
| P1 | Cross-session | EEGNet − FBCSP | `−0.0546` | `[−0.1304, +0.0212]` | `3 / 6` | `0.1016` | `0.2031` |
| P2 | Cross-session | Riemann − FBCSP | `−0.0015` | `[−0.0716, +0.0686]` | `4 / 4` (one tie) | `0.9844` | `0.9844` |
| P3 | LOSO | EEGNet − Riemann | `+0.0954` | `[+0.0552, +0.1357]` | `9 / 0` | `0.0039` | `0.0117` |

- **P1**: In cross-session evaluation, EEGNet is not significantly different from FBCSP. The point estimate favors FBCSP, and the confidence interval ranges from a 13-point disadvantage to a 2-point advantage for EEGNet, so the data do not establish equivalence either.
- **P2**: The Riemannian model and FBCSP reach practically the same mean cross-session accuracy, with an interval of about ±7 points.
- **P3**: In LOSO, EEGNet is more accurate than the Riemannian model for every subject, and the difference remains significant after Holm adjustment.

### 4.2 Cross-session

| Model | Accuracy Mean ± SD | 95% CI |
|---|---:|---:|
| Raw Power + LDA | `0.6898 ± 0.1381` | `[0.5837, 0.7959]` |
| CSP + LDA | `0.7230 ± 0.1641` | `[0.5969, 0.8491]` |
| FBCSP + LDA | `0.7600 ± 0.1600` | `[0.6370, 0.8830]` |
| Riemann + Tangent Space + LDA | `0.7585 ± 0.1505` | `[0.6428, 0.8742]` |
| EEGNet (recipe R) | `0.7054 ± 0.1810` | `[0.5662, 0.8446]` |

| Subject | Raw Power | CSP | FBCSP | Riemann | EEGNet | EEGNet seed SD |
|---|---:|---:|---:|---:|---:|---:|
| S1 | `0.6944` | `0.7917` | `0.9028` | `0.8611` | `0.8389` | `0.0205` |
| S2 | `0.5486` | `0.6042` | `0.5694` | `0.5694` | `0.5014` | `0.0227` |
| S3 | `0.8819` | `0.8958` | `0.9444` | `0.9653` | `0.8556` | `0.1989` |
| S4 | `0.6042` | `0.5972` | `0.6111` | `0.7431` | `0.6528` | `0.0461` |
| S5 | `0.5417` | `0.5833` | `0.7847` | `0.6875` | `0.5000` | `0.0208` |
| S6 | `0.6319` | `0.6667` | `0.5764` | `0.7153` | `0.5694` | `0.0517` |
| S7 | `0.5972` | `0.5139` | `0.6389` | `0.5347` | `0.5750` | `0.0284` |
| S8 | `0.8681` | `0.9514` | `0.9167` | `0.9375` | `0.9486` | `0.0188` |
| S9 | `0.8403` | `0.9028` | `0.8958` | `0.8125` | `0.9069` | `0.0160` |

- None of the ten secondary pairwise comparisons is significant after Holm adjustment. The smallest uncorrected p-values are Raw Power vs FBCSP (`0.039`) and vs Riemann (`0.043`), both with `p_Holm = 0.39`.
- EEGNet is at chance level for S2 and S5.
- For S3, the between-seed standard deviation of `0.199` comes from a single failed fit.
  - Seed 46 stopped at epoch `31` with its best validation loss at epoch `1`, so early stopping restored weights close to the initialization and the fit reached an accuracy of `0.500`.
  - The other four seeds reached `0.938` to `0.951`.
- **Collapsed fits.** The same failure (best epoch `1`) occurred in `4` of the `45` cross-session EEGNet fits: S2 seeds 44 and 45, S5 seed 45, and S3 seed 46. It did not occur in any of the `45` LOSO fits.
  - The failure is a property of recipe R when only `144` trials (about `115` after the validation split) are available: if the validation loss never improves after the first epoch, the recipe restores the first-epoch weights.
  - As specified in the analysis plan, these fits are included in the primary analysis.
- **Post-hoc sensitivity check (not part of the plan).** Excluding the four collapsed fits raises cross-session EEGNet to `0.7165`. P1 then becomes `−0.0435` (95% CI `−0.1185` to `+0.0314`, exact `p = 0.21`), so the conclusion of P1 is unchanged.

![Cross-session accuracy](assets/generated/cross_session_accuracy.png)

*Figure 2. Cross-session mean accuracy (train on session 1, test on session 2). Error bars show the standard deviation across subjects.*

### 4.3 LOSO

| Model | Accuracy Mean ± SD | 95% CI |
|---|---:|---:|
| Raw Power + LDA | `0.6134 ± 0.0938` | `[0.5413, 0.6855]` |
| CSP + LDA | `0.5907 ± 0.1148` | `[0.5025, 0.6789]` |
| FBCSP + LDA | `0.5648 ± 0.0678` | `[0.5127, 0.6169]` |
| Riemann + Tangent Space + LDA | `0.6285 ± 0.1039` | `[0.5486, 0.7083]` |
| EEGNet (recipe R) | `0.7239 ± 0.1234` | `[0.6291, 0.8188]` |

| Subject | Raw Power | CSP | FBCSP | Riemann | EEGNet | EEGNet seed SD |
|---|---:|---:|---:|---:|---:|---:|
| S1 | `0.6285` | `0.5139` | `0.6701` | `0.7153` | `0.8090` | `0.0185` |
| S2 | `0.4792` | `0.5069` | `0.5243` | `0.4965` | `0.6139` | `0.0186` |
| S3 | `0.7674` | `0.5590` | `0.5486` | `0.8090` | `0.8632` | `0.0364` |
| S4 | `0.6285` | `0.6146` | `0.6458` | `0.5486` | `0.6222` | `0.0300` |
| S5 | `0.5035` | `0.5174` | `0.5000` | `0.5139` | `0.5736` | `0.0247` |
| S6 | `0.6042` | `0.5035` | `0.5000` | `0.6111` | `0.7333` | `0.0328` |
| S7 | `0.5625` | `0.5729` | `0.5556` | `0.5972` | `0.6174` | `0.0464` |
| S8 | `0.7292` | `0.8576` | `0.5035` | `0.7153` | `0.9160` | `0.0135` |
| S9 | `0.6181` | `0.6701` | `0.6354` | `0.6493` | `0.7667` | `0.0280` |

- In the secondary LOSO family, EEGNet is higher than CSP (`9/9` subjects, `p_Holm = 0.039`) and Riemann (`9/9`, `p_Holm = 0.039`).
- EEGNet is also higher than FBCSP and Raw Power in `8/9` subjects (`p_Holm = 0.063` each).
- No pair of non-deep models differs significantly.
- Several classical subject-independent models stay at chance for individual held-out subjects, for example FBCSP for S5, S6, and S8.

![LOSO accuracy comparison](assets/generated/loso_accuracy.png)

*Figure 3. LOSO mean accuracy. Error bars show the standard deviation across held-out subjects.*

![LOSO EEGNet confusion matrix](assets/generated/loso_eegnet_confusion.png)

*Figure 4. Row-normalized confusion matrix for LOSO EEGNet, summed over held-out subjects and training seeds.*

### 4.4 Within-subject CV with pooled sessions

| Model | Within-subject CV (pooled) | Cross-session | Difference |
|---|---:|---:|---:|
| Raw Power + LDA | `0.7099` | `0.6898` | `−0.0201` |
| CSP + LDA | `0.7810` | `0.7230` | `−0.0580` |
| FBCSP + LDA | `0.8183` | `0.7600` | `−0.0583` |
| Riemann + Tangent Space + LDA | `0.7983` | `0.7585` | `−0.0398` |
| EEGNet (recipe R) | `0.7459` | `0.7054` | `−0.0405` |

- Every model is less accurate cross-session than in session-pooled CV.
- The difference mixes two factors: transfer to a new recording day, and less training data per fit (`144` versus about `230` trials).
- Within the pooled CV family, only Raw Power vs CSP and Raw Power vs Riemann are significant after Holm adjustment (`p_Holm = 0.039`). FBCSP is higher than EEGNet in `5` of `9` subjects (`p = 0.20`).

![Within-subject accuracy comparison](assets/generated/within_subject_accuracy.png)

*Figure 5. Within-subject CV mean accuracy with both sessions pooled. Error bars show the standard deviation across subjects.*

![Within-subject FBCSP confusion matrix](assets/generated/within_subject_fbcsp_confusion.png)

*Figure 6. Row-normalized confusion matrix for within-subject FBCSP, summed over subjects and folds.*

### 4.5 Cross-subject transfer

Mean accuracy across the nine target subjects (seeds averaged within target), with the standard deviation across targets:

| Setting | FBCSP | Riemann | EEGNet (recipe R) |
|---|---:|---:|---:|
| `zero_shot` | `0.5664 ± 0.0650` | `0.6331 ± 0.1056` | `0.7276 ± 0.1205` |
| `5_shot` | `0.5687 ± 0.0698` | `0.6671 ± 0.1208` | `0.7515 ± 0.1261` |
| `10_shot` | `0.5745 ± 0.0710` | `0.6740 ± 0.1232` | `0.7650 ± 0.1363` |
| `20_shot` | `0.5826 ± 0.0700` | `0.6968 ± 0.1218` | `0.7936 ± 0.1191` |
| `30_shot` | `0.5887 ± 0.0830` | `0.7106 ± 0.1327` | `0.7940 ± 0.1268` |

- **EEGNet vs Riemann**: EEGNet is higher in all nine targets at every setting (mean difference `+0.083` to `+0.097`, exact `p = 0.0039`).
- **EEGNet vs FBCSP**: EEGNet is higher in `8` to `9` targets (`+0.161` to `+0.211`).
- **Riemann vs FBCSP**: Riemann is higher in `7` to `8` targets (`+0.067` to `+0.122`, `p = 0.012` to `0.148`).
- **Multiple-comparison limit**: with `15` tests and `n = 9`, the smallest attainable Holm-adjusted p-value is `0.0586`, which is the value reached by most comparisons. The design cannot reach `0.05` here, so these results remain exploratory.
- **Gains from calibration**: between `zero_shot` and `30_shot`, accuracy rises by `0.022` (FBCSP), `0.078` (Riemann), and `0.066` (EEGNet).
- **Overlap with LOSO**: `zero_shot` uses source-only models evaluated on half of each target's trials, so it overlaps with LOSO.

![Transfer accuracy by calibration budget](assets/generated/transfer_repeated_accuracy.png)

*Figure 7. Transfer accuracy by calibration budget. Lines show the mean across target subjects and shaded bands the t-based 95% confidence interval across targets.*

![Riemann transfer confusion matrices](assets/generated/transfer_repeated_riemann_confusion.png)

*Figure 8. Row-normalized confusion matrices for Riemann transfer, one panel per calibration setting, summed over targets and seeds.*

![EEGNet transfer confusion matrices](assets/generated/transfer_repeated_eegnet_confusion.png)

*Figure 9. Row-normalized confusion matrices for EEGNet transfer, one panel per calibration setting, summed over targets and seeds.*

### 4.6 Exploratory: does EEGNet's advantage depend on the protocol?

That P3 is significant while P1 is not does not by itself show that the protocols differ. The following test therefore compares, per subject, a model difference under LOSO with the same difference under cross-session. This test was not pre-specified.

| Model difference | LOSO − cross-session | 95% CI | Subjects larger in LOSO | p (exact) | p (Holm, 2 tests) |
|---|---:|---:|---:|---:|---:|
| EEGNet − Riemann | `+0.1485` | `[+0.0752, +0.2219]` | `8 / 9` | `0.0078` | `0.0156` |
| EEGNet − FBCSP | `+0.2137` | `[+0.0975, +0.3300]` | `8 / 9` | `0.0078` | `0.0156` |

EEGNet's advantage over both non-deep models is larger in LOSO than in cross-session. The two protocols differ in more than the generalization target, however:

- EEGNet trains on `2304` pooled trials in LOSO and on `144` single-subject trials cross-session.
- The test sets contain both sessions in LOSO and only the second session cross-session.

The result is therefore consistent with EEGNet benefiting from larger training sets as much as with a specific cross-subject advantage, and this comparison alone cannot separate the two. A follow-up that matches both the training-set size and the test trials of the two protocols was specified in [`docs/analysis_plan_addendum.md`](analysis_plan_addendum.md) before it was run (Section 4.7).

### 4.7 Addendum: training-size-matched LOSO

Matched LOSO uses the same `144` test trials per subject as cross-session evaluation and a training set of the same size, so the two protocols differ only in whether the training data come from the test subject's first session or from eight other subjects.

| Model | Matched LOSO, mean ± SD | 95% CI | Cross-session (same test trials) | Change |
|---|---:|---:|---:|---:|
| Raw Power + LDA | `0.5804 ± 0.0675` | `[0.5285, 0.6323]` | `0.6898` | `−0.1094` |
| CSP + LDA | `0.5806 ± 0.0625` | `[0.5325, 0.6286]` | `0.7230` | `−0.1424` |
| FBCSP + LDA | `0.5602 ± 0.0594` | `[0.5145, 0.6059]` | `0.7600` | `−0.1998` |
| Riemann + Tangent Space + LDA | `0.6017 ± 0.0697` | `[0.5481, 0.6553]` | `0.7585` | `−0.1568` |
| EEGNet (recipe R) | `0.5340 ± 0.0318` | `[0.5095, 0.5584]` | `0.7054` | `−0.1714` |

Pre-specified addendum tests (per-subject model difference under matched LOSO minus the same difference under cross-session; Holm across two tests):

| ID | Model difference | Matched LOSO − cross-session | 95% CI | Subjects larger in matched LOSO | p (exact) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| A1 | EEGNet − Riemann | `−0.0147` | `[−0.1053, +0.0760]` | `4 / 9` | `0.7109` | `0.9375` |
| A2 | EEGNet − FBCSP | `+0.0284` | `[−0.0537, +0.1105]` | `6 / 9` | `0.4688` | `0.9375` |

- **Pre-specified rule.** Neither test is significant. Under the rule fixed in the addendum, EEGNet's larger advantage in full LOSO is therefore not shown to be independent of the training-set size.
  - The confidence intervals are wide (about ±9 points), so moderate effects of the data source cannot be excluded.
- **Descriptive pattern.** With `144` cross-subject training trials, EEGNet is close to chance for every subject (`0.488` to `0.596`) and is the least accurate model. In the secondary family, Riemann is higher than EEGNet in all nine subjects (`+0.068`, `p_Holm = 0.039`).
- **Full versus matched LOSO.** The test sets differ (both sessions versus session 2), so the following is descriptive only. Going from `144` to `2304` cross-subject training trials changes the mean accuracy by:

  | Model | Matched LOSO (144 trials) | Full LOSO (2304 trials) | Change |
  |---|---:|---:|---:|
  | EEGNet | `0.534` | `0.724` | `+0.190` |
  | Riemann | `0.602` | `0.629` | `+0.027` |
  | FBCSP | `0.560` | `0.565` | `+0.005` |

  EEGNet's LOSO advantage appears only with the large pooled training set.
- **Value of own-subject data.** Every model is `11` to `20` points more accurate when its `144` training trials come from the test subject's own first session rather than from other subjects (descriptive, same test trials).
- **Seed variability.** The between-seed standard deviation of matched-LOSO EEGNet is `0.023` to `0.083` per subject. This is larger than in full LOSO, as expected when each seed also draws a different training subsample.

![Matched LOSO accuracy](assets/generated/loso_matched_accuracy.png)

*Figure 9a. LOSO with `144` training trials from the other subjects, tested on session 2 of each held-out subject. Error bars show the standard deviation across subjects.*

### 4.8 EEGNet training budget and variability

| Protocol | Earlier run: 1 seed, 50 max epochs | Recipe R: 5 seeds, 300 max epochs |
|---|---:|---:|
| Within-subject CV (pooled) | `0.6933` | `0.7459` |
| LOSO | `0.7068` | `0.7239` |
| Transfer `zero_shot` / `30_shot` | `0.7079` / `0.7612` | `0.7276` / `0.7940` |

- The 50-epoch budget had stopped within-subject training before convergence. In a single-seed sensitivity run with a 300-epoch cap, all 45 within-subject fits stopped early, at a mean best epoch of `77`.
- **Recipe R, cross-session**: `44` of `45` fits stopped early (mean best epoch `82.1`, mean executed epochs `112.0`).
- **Recipe R, LOSO**: `35` of `45` fits stopped early (mean best epoch `213.1`). The other `10` reached the `300`-epoch cap with best epochs between `278` and `298`, so the cap still limited some LOSO fits. LOSO EEGNet accuracy may therefore be slightly underestimated, which is conservative for P3.
- The between-seed standard deviation per subject ranges from `0.004` to `0.073` (pooled CV), `0.014` to `0.046` (LOSO), and `0.016` to `0.199` (cross-session).
- Single-seed EEGNet results for individual subjects should not be interpreted.

### 4.9 Exploratory EEG analysis

The exploratory figures use epochs from `-1.5` to `4.5 s` around the cue, so that the reference interval and the analysed task interval stay away from wavelet edge effects. Time-frequency power uses Morlet wavelets (`8-30 Hz`, `n_cycles = f / 2`). ERD/ERS follows the classical definition (Pfurtscheller and Lopes da Silva, 1999): power is averaged over trials first and then expressed relative to the pre-cue reference interval `[-1.0, -0.2] s`:

$$
\mathrm{ERD/ERS}(t, f) = 100 \times \frac{\bar{P}(t, f) - \bar{R}(f)}{\bar{R}(f)}
$$

Here $\bar{P}(t, f)$ is the trial-averaged power and $\bar{R}(f)$ its mean over the reference interval; negative values (ERD) mean less power than before the cue. The numerical summaries below report two values per curve:

- the mean over the pre-defined window `0.5-2.5 s` after the cue
- the most negative value (peak ERD) in `0-4 s`, with its latency

All values come from the exported files (`*_erds_summary.md`, `*_erds_curves.csv`).

Grand average over the nine subjects:

| Class | Channel | Side | Band | Window mean (%) | Peak ERD (%) | Peak latency (s) |
|---|---|---|---|---:|---:|---:|
| Left hand | C4 | contralateral | mu | `−17.2` | `−27.3` | `0.61` |
| Left hand | C3 | ipsilateral | mu | `−13.3` | `−25.1` | `1.57` |
| Left hand | C4 | contralateral | beta | `−10.5` | `−16.2` | `1.51` |
| Left hand | C3 | ipsilateral | beta | `−9.9` | `−16.2` | `1.52` |
| Right hand | C3 | contralateral | mu | `−17.9` | `−33.6` | `0.68` |
| Right hand | C4 | ipsilateral | mu | `−11.1` | `−21.5` | `0.69` |
| Right hand | C3 | contralateral | beta | `−13.5` | `−20.5` | `0.52` |
| Right hand | C4 | ipsilateral | beta | `−5.6` | `−11.4` | `0.50` |

Contralateral versus ipsilateral window mean, per subject (exact sign-flip test, Holm across four tests):

| Class | Band | Contra − ipsi (%) | Subjects with stronger contralateral ERD | p (exact) | p (Holm) |
|---|---|---:|---:|---:|---:|
| Left hand | mu | `−3.9` | `6 / 9` | `0.1523` | `0.4570` |
| Left hand | beta | `−0.6` | `5 / 9` | `0.8164` | `0.8164` |
| Right hand | mu | `−6.8` | `6 / 9` | `0.2773` | `0.5547` |
| Right hand | beta | `−7.9` | `9 / 9` | `0.0039` | `0.0156` |

- Motor imagery produces mu and beta ERD over both hemispheres. The per-subject window means range from `−5.6%` to `−17.9%`, with standard errors of `2.9` to `7.8` points.
- The ERD tends to be stronger over the hemisphere contralateral to the imagined hand. This lateralization is significant only for right-hand imagery in the beta band; the mu-band differences vary between subjects.
- Mu ERD peaks around `0.6-0.7 s` after the cue, except for left-hand imagery at C3, which peaks at `1.57 s`.

![Subject 1 PSD by class](assets/eda_subject_1/subject_1_psd.png)

*Figure 10. Subject 1 class-wise power spectral density (Welch), averaged over channels and trials, in dB relative to 1 µV²/Hz.*

![Subject 1 PCA](assets/eda_subject_1/subject_1_pca.png)

*Figure 11. Subject 1 trials projected onto the first two principal components of standardized channel log-variance features.*

![Subject 1 t-SNE](assets/eda_subject_1/subject_1_tsne.png)

*Figure 12. Subject 1 t-SNE embedding of the same channel log-variance features, for visual inspection only.*

![Subject 1 channel topography](assets/eda_subject_1/subject_1_topomap.png)

*Figure 13. Subject 1 mean log-variance per channel for left- and right-hand imagery and their difference (natural-log ratio, left minus right). The difference is largest at CP3 (`+0.109`, about 11% more variance during left- than right-hand imagery), P1 (`+0.082`), and FC3 (`+0.074`), and most negative at CP4 (`−0.063`, about 6% less) and P2 (`−0.047`). This matches lower power over the hemisphere contralateral to the imagined hand, with the largest effect at centro-parietal rather than central electrodes. Values are in `subject_1_topomap_values.csv`.*

![Subject 1 ERD/ERS maps](assets/eda_subject_1/subject_1_erds.png)

*Figure 14. Subject 1 ERD/ERS time-frequency maps at C3 and C4 for each class, relative to the pre-cue reference interval.*

![Subject 1 sensorimotor ERD/ERS summary](assets/eda_subject_1/subject_1_sensorimotor_erds.png)

*Figure 15. Subject 1 mu (8-12 Hz) and beta (13-30 Hz) ERD/ERS at C3 and C4; the grey area marks the reference interval and the dotted line the cue.*

- Right-hand imagery: mu ERD at contralateral C3 has a window mean of `−24.7%` and a peak of `−38.9%` at `0.74 s`; ipsilateral C4 shows `−16.8%` and `−31.5%`.
- Left-hand imagery: contralateral C4 shows `−19.2%` and `−29.9%` at `0.71 s`; ipsilateral C3 shows `−13.0%` and `−24.9%`.
- Beta window means lie between `−6.2%` and `+4.1%`.

![Subject 2 sensorimotor ERD/ERS summary](assets/eda_subject_2/subject_2_sensorimotor_erds.png)

*Figure 16. Subject 2 mu and beta ERD/ERS at C3 and C4. Subject 2 is decoded close to chance by most models in most protocols.*

![Subject 8 sensorimotor ERD/ERS summary](assets/eda_subject_8/subject_8_sensorimotor_erds.png)

*Figure 17. Subject 8 mu and beta ERD/ERS at C3 and C4. Subject 8 is one of the best-decoded subjects for most models and protocols.*

![Grand-average sensorimotor ERD/ERS summary](assets/group_eda/grand_average_sensorimotor_erds.png)

*Figure 18. Grand-average mu and beta ERD/ERS across the nine subjects; shaded bands show the standard error across subjects. Values are in the tables above.*

![Subject 1 classical spatial patterns](assets/eda_subject_1/subject_1_classical_patterns.png)

*Figure 19. Subject 1 CSP patterns (components 1 and 4) and the first FBCSP pattern of the lowest and highest frequency bands, fitted on all trials of the subject. The FBCSP panels show fixed bands, not necessarily the selected features.*

![Subject 1 Riemann tangent-space view](assets/eda_subject_1/subject_1_riemann_3d.png)

*Figure 20. Subject 1 tangent-space features projected onto three principal components, fitted on all trials (qualitative view).*

![Subject 1 Riemann + LDA out-of-fold scores](assets/eda_subject_1/subject_1_riemann_lda_distribution.png)

*Figure 21. Subject 1 Riemann + LDA decision scores, each obtained from a model trained without that trial (5-fold cross-validation).*

![Subject 1 CSP + LDA out-of-fold scores](assets/eda_subject_1/subject_1_csp_lda_distribution.png)

*Figure 22. Subject 1 CSP + LDA decision scores, obtained out-of-fold in the same way as Figure 21.*

![Subject 1 CSP feature projection](assets/eda_subject_1/subject_1_csp_projection.png)

*Figure 23. Subject 1 trials in the space of the first two CSP components, fitted and plotted on the same trials (in-sample, so separability is optimistic).*

![Subject 1 EEGNet saliency topomaps](assets/eda_subject_1/subject_1_eegnet_saliency_topomap.png)

*Figure 24. Subject 1 input-gradient saliency of an EEGNet trained on all trials of the subject for 15 epochs. This is a first-order attribution summary of one model, not a mechanistic explanation.*

## 5. Discussion

**Confirmatory results.** Of the three pre-specified comparisons, only the LOSO comparison is significant: EEGNet decodes held-out subjects more accurately than the Riemannian model, by about ten percentage points and in every subject. When the model has to transfer to a new session of the same subject, there is no evidence that EEGNet outperforms FBCSP, and the Riemannian model and FBCSP perform alike. The cross-session confidence intervals are wide (about ±7 points for Riemann vs FBCSP), so these non-significant results do not show equivalence.

**Protocol dependence.** EEGNet's relative standing differs between full LOSO and cross-session evaluation, in eight of nine subjects (exploratory). The addendum shows that this difference is tied to the amount of training data:

- When cross-subject models receive only as many trials as one session, EEGNet falls to near chance.
- Its standing relative to Riemann and FBCSP then no longer differs from cross-session evaluation (A1, A2).

The data therefore do not support a specific cross-subject advantage of EEGNet. Two things do hold:

- EEGNet profits much more than the non-deep models from a large pooled training set (descriptively `+0.19` versus at most `+0.03` from `144` to `2304` trials).
- For every model, a session of the subject's own data is worth more than the same number of trials from other subjects.

The confidence intervals of A1 and A2 leave room for moderate data-source effects, and nine subjects limit what can be ruled out.

**Session effects.** Every model loses accuracy from session-pooled CV to cross-session evaluation (`2` to `6` points). Session-pooled CV therefore overstates how well a decoder trained on one day works on another day, although part of the loss reflects smaller training sets.

**EEGNet training.** EEGNet results depend on the training budget and the seed:

- Recipe R raised accuracy relative to the earlier 50-epoch runs.
- With only `144` training trials, single training runs can fail: in `4` of `45` cross-session fits, the validation loss never improved after the first epoch and early stopping restored near-initial weights.
- Averaging over five seeds reduces the influence of such failures but does not remove the underlying instability. Excluding the failed fits does not change the conclusion of P1.

**ERD/ERS.** The time-frequency analysis confirms task-related mu and beta desynchronization after the cue in all conditions. Its lateralization is clear only for right-hand imagery in the beta band, which is in line with the moderate decoding accuracy of several subjects.

**Limitations.**

- Nine subjects from one dataset limit statistical power.
- P3 replicates a direction that had already been observed.
- The addendum was specified after the primary results were known, although before the matched runs. Its training subsamples (five seeds) add sampling variability, and EEGNet's recipe was not re-tuned for small cross-subject training sets.
- FBCSP and EEGNet are simplified relative to their original publications, and no hyperparameters were tuned.
- EEGNet ran on Apple MPS, which is not guaranteed to be bit-for-bit reproducible, and results from other devices may differ.

## 6. Conclusion

- In cross-session evaluation, EEGNet (`0.705`) was not significantly different from FBCSP (`0.760`), and the Riemannian model (`0.759`) matched FBCSP. The confidence intervals do not rule out differences of several percentage points in either direction.
- In LOSO evaluation, EEGNet (`0.724`) was more accurate than the Riemannian model (`0.629`) for all nine subjects. This was the only significant pre-specified comparison, and it replicates an earlier observation.
- With the training-set size and the test trials matched to cross-session evaluation, EEGNet's standing relative to the Riemannian model and FBCSP was the same as in cross-session evaluation (addendum, not significant). With only `144` cross-subject trials, EEGNet was near chance. EEGNet's LOSO advantage therefore reflects its use of a large pooled training set, not cross-subject evaluation as such.
- Exploratory analyses further show that all models lose accuracy across recording days, and that the transfer ordering (EEGNet > Riemann > FBCSP) is consistent but not significant after correction.

## Appendix A. EEGNet optimization curves

![Cross-session EEGNet learning curve](assets/generated/cross_session_eegnet_learning_curve.png)

*Figure A1. Mean cross-session EEGNet training and validation loss over subjects and training seeds. Each epoch averages only the fits still training, so later epochs are computed from fewer fits.*

![Within-subject EEGNet learning curve](assets/generated/within_subject_eegnet_learning_curve.png)

*Figure A2. Mean within-subject (pooled CV) EEGNet training and validation loss over folds and training seeds, averaged in the same way.*

![LOSO EEGNet learning curve](assets/generated/loso_eegnet_learning_curve.png)

*Figure A3. Mean LOSO EEGNet training and validation loss over held-out subjects and training seeds, averaged in the same way.*

## Appendix B. Corrections to earlier versions of this report

1. **Raw Power + LDA.** A fixed floor of `1e-10` on volt-scale variances (about `2.5e-11 V²`) made every feature constant. The earlier values `0.5355` (within-subject) and `0.5270` (LOSO) were artifacts.
2. **ERD/ERS figures.** The earliest version normalized each trial by its own `0.5 s` reference interval at the epoch edge. This produced an apparent sustained power increase of `+50%` to `+100%` even for data without a task effect.
3. **LOSO EEGNet spread, transfer intervals, and p-values.**
   - The reported LOSO standard deviation did not match the per-subject values of the same run.
   - The transfer "± Std" and "95% CI" were computed across calibration settings rather than subjects.
   - The transfer p-values counted each target twice and were not corrected for multiple comparisons.
4. **Repeated-seed EEGNet transfer.** Seeds now also set the network training seed.
5. **Approximate ERD/ERS values.** The previous revision quoted ERD/ERS values read from figures. They are replaced by the exported values. The largest change: left-hand imagery at C3 peaks at `−25.1%` (`1.57 s`), not about `−20%` as previously read. The previous revision also described contralateral dominance more strongly than the per-subject tests support.
6. **EEGNet configuration.** The previous revision reported single-seed, 50-epoch EEGNet results. All EEGNet results now use recipe R; the earlier runs are listed in Section 4.8 for comparison.
7. **Unchanged results.** Classical and Riemannian results are unchanged since the previous revision.

## Appendix C. Reproducibility

Software and hardware:

- EEGNet results: Python `3.14.7`, PyTorch `2.14.0` on Apple MPS (macOS arm64), MOABB `1.7.2`, MNE `1.13.2`, scikit-learn `1.9.1`, pyRiemann `0.12`, NumPy `2.5.3`, SciPy `1.18.1`
- Classical and Riemannian cross-session and matched-LOSO results record the same environment (they run on the CPU regardless of the torch device setting). The classical and Riemannian within-subject, LOSO, and transfer results were produced earlier with the same code for these pipelines, and their result files do not record the environment.

Wall-clock fit-and-predict time on that machine (EEGNet totals include all training seeds):

| Result | Runtime (s) |
|---|---:|
| Cross-session: Raw Power / CSP / FBCSP / Riemann | `0.1` / `1.5` / `40.0` / `1.2` |
| Cross-session: EEGNet (5 seeds) | `394.0` |
| Within-subject CV: Raw Power / CSP / FBCSP / Riemann | `0.5` / `11.4` / `251.0` / `7.1` |
| Within-subject CV: EEGNet (5 seeds) | `2776.0` |
| LOSO: Raw Power / CSP / FBCSP / Riemann | `0.9` / `23.6` / `492.6` / `13.3` |
| LOSO: EEGNet (5 seeds) | `9787.0` |
| Matched LOSO, 5 seeds: Raw Power / CSP / FBCSP / Riemann | `0.6` / `7.4` / `203.0` / `6.1` |
| Matched LOSO: EEGNet (5 seeds) | `235.6` |
| Transfer, 9 targets × 2 seeds: FBCSP / Riemann / EEGNet | `5118.2` / `190.2` / `3626.2` |

The full protocol is described in [`docs/reproduction.md`](reproduction.md) and is run with `python -m eeg_motor_imagery_classification.reproduce`. `python -m eeg_motor_imagery_classification.cli --experiment export_stats` regenerates every statistic in this report from the saved outputs.

## Appendix D. Adherence to the analysis plan

- **Plan commit.** The analysis plan was committed in `1699c44` at `2026-10-02 14:25:32 +0900`, together with the code that implements the cross-session protocol and recipe R. The primary stages were run after this commit, following the documented workflow.
  - The result files of that round do not store creation times, so this order cannot be checked from the result files alone.
  - From now on, every `result.json` records `created_at`.
- **Settings.** The settings recorded in the primary result files match the plan: `300` max epochs, patience `30`, minimum `30` epochs, batch size `32` for cross-session and `64` for LOSO, seeds `42-46`, Apple MPS.
- **Collapsed fits.** The four collapsed cross-session EEGNet fits (Section 4.2) were kept, as the plan requires; a post-hoc sensitivity check is reported separately.
- **Additions after the primary results.** The paired-difference confidence intervals and the protocol-by-model tests (Section 4.6) were added after the primary results were known and are reported as supplementary or exploratory. A training-size-matched follow-up was specified in [`docs/analysis_plan_addendum.md`](analysis_plan_addendum.md) before it was run and was executed as specified. The addendum was committed in `5d19d90` at `2026-10-02 20:29:39 +0900` (`11:29:39 UTC`); the matched-LOSO result files record `created_at` times of `11:33:36`, `11:33:57`, and `11:38:08 UTC` on the same day, after the commit.
- **Other deviations.** None was recorded.

## References

1. BNCI Horizon 2020. *001-2014: Left and right hand motor imagery*. https://bnci-horizon-2020.eu/database/data-sets
2. Tangermann M, Muller KR, Aertsen A, et al. *Review of the BCI Competition IV*. Front Neurosci. 2012. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full
3. Ramoser H, Muller-Gerking J, Pfurtscheller G. *Optimal spatial filtering of single trial EEG during imagined hand movement*. IEEE Trans Rehabil Eng. 2000. https://pubmed.ncbi.nlm.nih.gov/11204034/
4. Ang KK, Chin ZY, Wang C, Guan C, Zhang H. *Filter Bank Common Spatial Pattern (FBCSP) in brain-computer interface*. Proc IJCNN. 2008. https://pubmed.ncbi.nlm.nih.gov/19963675/
5. Barachant A, Bonnet S, Congedo M, Jutten C. *Multiclass brain-computer interface classification by Riemannian geometry*. IEEE Trans Biomed Eng. 2012. https://pubmed.ncbi.nlm.nih.gov/22010143/
6. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*. J Neural Eng. 2018. https://pubmed.ncbi.nlm.nih.gov/29932424/
7. Jayaram V, Barachant A. *MOABB: trustworthy algorithm benchmarking for BCIs*. J Neural Eng. 2018. https://pubmed.ncbi.nlm.nih.gov/30177583/
8. Gramfort A, Luessi M, Larson E, et al. *MNE software for processing MEG and EEG data*. Neuroimage. 2013. https://pubmed.ncbi.nlm.nih.gov/24161808/
9. Pfurtscheller G, Lopes da Silva FH. *Event-related EEG/MEG synchronization and desynchronization: basic principles*. Clin Neurophysiol. 1999.
10. Holm S. *A simple sequentially rejective multiple test procedure*. Scand J Stat. 1979.
