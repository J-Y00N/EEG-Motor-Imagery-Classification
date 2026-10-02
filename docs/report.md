# Experiment Report

## Abstract

This report evaluates left-versus-right hand motor imagery decoding on BNCI 2014-001 with five model families (raw log-variance power, CSP, FBCSP, a Riemannian tangent-space model, and EEGNet) under three separated protocols: within-subject cross-validation, leave-one-subject-out (LOSO), and cross-subject transfer with zero- and few-shot calibration. All summaries use the subject as the unit of analysis, report sample standard deviations and t-based 95% confidence intervals, and compare models with exact paired sign-flip tests corrected with the Holm procedure. Mean accuracy ranks the models differently by protocol: FBCSP has the highest within-subject mean (0.818), EEGNet has the highest LOSO mean (0.707), and in transfer the ordering EEGNet > Riemann > FBCSP holds at every calibration budget. With nine subjects, however, none of the comparisons between the competitive models remains significant after multiple-comparison correction, so these rankings are descriptive rather than statistically established. EEGNet results were also sensitive to the training budget and varied between runs.

Keywords: EEG motor imagery classification, BNCI2014_001, CSP, FBCSP, Riemannian geometry, EEGNet, LOSO, transfer learning

## 1. Introduction

The project compares classical, geometric, and deep approaches to left-versus-right motor imagery classification under a single experimental design. Within-subject, subject-independent (LOSO), and calibration-aware transfer results answer different questions, so each claim in this report is tied to the protocol in which it was measured.

## 2. Data and Method

### 2.1 Dataset

- source: `BNCI2014_001` (BCI Competition IV 2a), loaded with MOABB 1.7.2
- subjects `1-9`, two sessions per subject recorded on different days, six runs per session
- classes: `left_hand` and `right_hand` only, giving `288` trials per subject (`144` per class)
- `22` EEG channels at `250 Hz`; MOABB converts the signals from microvolts to volts
- MOABB places each trial annotation at cue onset (its dataset interval is `[2, 6] s` from trial start), so `0 s` in this report is the cue

### 2.2 Preprocessing

One preprocessing path is shared by all models:

- `50 Hz` notch and `8-32 Hz` FIR band-pass filtering on the continuous runs
- epochs from `0.0` to `4.0 s` after the cue, without baseline correction
- both sessions of each subject are pooled

Average referencing and ICA are available in the configuration but disabled for all reported results.

### 2.3 Models

- **Raw Power + LDA**: per-channel log-variance, standardization, shrinkage LDA
- **CSP + LDA**: four CSP components with log-power features, shrinkage LDA
- **FBCSP + LDA**: six `4 Hz` bands between `8` and `32 Hz`, four CSP components per band, the eight best features by ANOVA F-score selected inside the training pipeline, shrinkage LDA. This is a simplified variant of Ang et al. (2008), which used a wider band range and mutual-information-based selection.
- **Riemann + Tangent Space + LDA**: OAS covariance estimation, tangent-space projection at the Riemannian mean of the training covariances, standardization, shrinkage LDA
- **EEGNet**: `F1 = 8`, `D = 2`, `F2 = 16`, temporal kernel of `64` samples, dropout `0.5`, applied to `250 Hz` input without resampling and without the max-norm constraints of the original architecture. Inputs are standardized per channel with statistics from the training data only. Training uses Adam (learning rate `1e-3`, batch size `64`) for at most `50` epochs, with early stopping on a stratified `20%` validation split taken from the training data (patience `10`, minimum `10` epochs) and restoration of the best validation epoch.

## 3. Evaluation

### 3.1 Protocols

- **Within-subject CV**: stratified, shuffled 5-fold cross-validation inside each subject, with both sessions pooled. Per-subject accuracy is the mean over folds.
- **LOSO**: each subject is held out once; the model is trained on the other eight subjects.
- **Cross-subject transfer**: for each target subject, the target trials are split 50/50 (stratified) into a calibration pool and an evaluation set of `144` trials. A `k_shot` setting draws `k` trials per class from the calibration pool (`k = 5, 10, 20, 30`). The `zero_shot` model uses source subjects only.
  - FBCSP and Riemann adapt by retraining on the source trials plus the `2k` target trials.
  - EEGNet is pretrained on the source subjects and then fine-tuned on the `2k` target trials only (all layers, `20` epochs, learning rate `5e-4`, batch size `16`).
  - The sweep is repeated with seeds `42` and `43`. A seed sets the calibration split, the shot sampling, and the EEGNet training seed. Seeds are averaged within each target before aggregation.

Because the adaptation mechanism differs between EEGNet and the other two models, transfer comparisons measure model and adaptation strategy together.

![Evaluation pipeline overview](assets/generated/evaluation_pipeline.png)

*Figure 1. Evaluation pipeline. One preprocessing path feeds all models; within-subject CV and LOSO use all five models, and transfer uses FBCSP, Riemann, and EEGNet.*

### 3.2 Statistics

- The unit of analysis is the subject (`n = 9`); in transfer it is the target subject after averaging seeds.
- Summaries report the mean, the sample standard deviation (`ddof = 1`), and a t-distribution 95% confidence interval.
- Models are compared with exact two-sided paired sign-flip permutation tests on per-subject accuracy.
- p-values are Holm-adjusted within each family: all `10` model pairs for within-subject CV, all `10` pairs for LOSO, and `3` pairs at each of `5` settings (`15` tests) for transfer.
- With `n = 9`, the smallest attainable exact p-value is `2 / 2^9 = 0.0039`. After Holm adjustment the smallest attainable value is therefore `0.039` for a family of `10` tests and `0.0586` for a family of `15` tests. No transfer comparison can reach `0.05` under this family definition, regardless of effect size.

## 4. Results

### 4.1 Within-subject CV

| Model | Subjects | Accuracy Mean ± SD | 95% CI |
|---|---:|---:|---:|
| Raw Power + LDA | `9` | `0.7099 ± 0.1331` | `[0.6075, 0.8122]` |
| CSP + LDA | `9` | `0.7810 ± 0.1368` | `[0.6758, 0.8861]` |
| FBCSP + LDA | `9` | `0.8183 ± 0.1383` | `[0.7120, 0.9246]` |
| Riemann + Tangent Space + LDA | `9` | `0.7983 ± 0.1245` | `[0.7025, 0.8940]` |
| EEGNet (`50` max epochs) | `9` | `0.6933 ± 0.1862` | `[0.5502, 0.8364]` |

Paired comparisons:

- Raw Power is lower than CSP (`0/9` subjects higher, `p_Holm = 0.039`) and Riemann (`0/9`, `p_Holm = 0.039`). Its deficit to FBCSP (`-0.108`, `1/8`) gives `p = 0.0078`, `p_Holm = 0.063`.
- CSP, FBCSP, and Riemann do not differ significantly (all `p_Holm >= 0.70`).
- EEGNet has the lowest mean, but its differences to FBCSP (`-0.125`, `p_Holm = 0.35`) and Riemann (`-0.105`, `p_Holm = 0.22`) are not significant.

EEGNet per-subject accuracy (`50` max epochs):

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | `0.5552` | `0.4757` | `0.9236` | `0.7535` | `0.5420` | `0.5454` | `0.6111` | `0.9548` | `0.8785` |
| Macro F1 | `0.4616` | `0.4745` | `0.9235` | `0.7524` | `0.4732` | `0.4883` | `0.5794` | `0.9548` | `0.8779` |

Four subjects (S1, S2, S5, S6) are at or near chance, and for S1 and S5 macro F1 is clearly below accuracy, which indicates predictions biased toward one class. Section 4.4 shows that this is partly a training-budget effect.

![Within-subject accuracy comparison](assets/generated/within_subject_accuracy.png)

*Figure 2. Within-subject mean accuracy with error bars showing the standard deviation across subjects.*

![Within-subject FBCSP confusion matrix](assets/generated/within_subject_fbcsp_confusion.png)

*Figure 3. Row-normalized confusion matrix for within-subject FBCSP, summed over subjects and folds.*

### 4.2 LOSO

| Model | Subjects | Accuracy Mean ± SD | 95% CI |
|---|---:|---:|---:|
| Raw Power + LDA | `9` | `0.6134 ± 0.0938` | `[0.5413, 0.6855]` |
| CSP + LDA | `9` | `0.5907 ± 0.1148` | `[0.5025, 0.6789]` |
| FBCSP + LDA | `9` | `0.5648 ± 0.0678` | `[0.5127, 0.6169]` |
| Riemann + Tangent Space + LDA | `9` | `0.6285 ± 0.1039` | `[0.5486, 0.7083]` |
| EEGNet (`50` max epochs) | `9` | `0.7068 ± 0.1481` | `[0.5929, 0.8207]` |

Paired comparisons:

- EEGNet is higher than each of the other four models in `8` of `9` subjects. Uncorrected p-values are `0.012` (Riemann) and `0.020` (Raw Power, CSP, FBCSP), and Holm-adjusted values are `0.12` to `0.18`.
- No other pair differs significantly.

Among the non-deep models, Riemann has the highest mean and the subject-specific spatial filters of CSP and FBCSP transfer worst across subjects. These differences are not significant.

EEGNet per-subject accuracy (`50` max epochs):

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | `0.7917` | `0.6042` | `0.9236` | `0.5903` | `0.5208` | `0.6840` | `0.5764` | `0.9132` | `0.7569` |
| Macro F1 | `0.7887` | `0.6041` | `0.9232` | `0.5203` | `0.4160` | `0.6711` | `0.5717` | `0.9127` | `0.7552` |

![LOSO accuracy comparison](assets/generated/loso_accuracy.png)

*Figure 4. LOSO mean accuracy with error bars showing the standard deviation across held-out subjects.*

![LOSO EEGNet confusion matrix](assets/generated/loso_eegnet_confusion.png)

*Figure 5. Row-normalized confusion matrix for LOSO EEGNet, summed over held-out subjects.*

### 4.3 Cross-subject transfer

Mean accuracy across the nine target subjects (seeds averaged within target), with the standard deviation across targets:

| Setting | FBCSP | Riemann | EEGNet |
|---|---:|---:|---:|
| `zero_shot` | `0.5664 ± 0.0650` | `0.6331 ± 0.1056` | `0.7079 ± 0.1455` |
| `5_shot` | `0.5687 ± 0.0698` | `0.6671 ± 0.1208` | `0.7195 ± 0.1567` |
| `10_shot` | `0.5745 ± 0.0710` | `0.6740 ± 0.1232` | `0.7469 ± 0.1611` |
| `20_shot` | `0.5826 ± 0.0700` | `0.6968 ± 0.1218` | `0.7569 ± 0.1560` |
| `30_shot` | `0.5887 ± 0.0830` | `0.7106 ± 0.1327` | `0.7612 ± 0.1570` |

Paired comparisons per setting (`15` tests, Holm-adjusted):

- EEGNet vs Riemann: EEGNet is higher in `7` to `8` of `9` targets at every setting (mean difference `+0.050` to `+0.075`, uncorrected `p = 0.016` to `0.039`, `p_Holm = 0.14` to `0.19`).
- Riemann vs FBCSP: Riemann is higher in `7` to `8` targets (mean difference `+0.067` to `+0.122`, uncorrected `p = 0.008` to `0.148`, `p_Holm >= 0.10`).
- EEGNet vs FBCSP: EEGNet is higher in `8` to `9` targets (mean difference `+0.142` to `+0.174`, uncorrected `p = 0.004` to `0.031`, `p_Holm = 0.059` to `0.19`).

The direction EEGNet > Riemann > FBCSP is the same at every setting, but no single comparison is significant after correction (see Section 3.2 for the attainable minimum). Between `zero_shot` and `30_shot`, mean accuracy rises by `0.022` for FBCSP, `0.078` for Riemann, and `0.053` for EEGNet. For FBCSP and Riemann the calibration trials make up only about `0.4%` to `2.5%` of the pooled training set. The `zero_shot` setting uses the same source models as LOSO evaluated on half of each target's trials, so it is not independent evidence.

![Transfer accuracy by calibration budget](assets/generated/transfer_repeated_accuracy.png)

*Figure 6. Transfer accuracy by calibration budget. Lines show the mean across target subjects and shaded bands the t-based 95% confidence interval across targets.*

![Riemann transfer confusion matrices](assets/generated/transfer_repeated_riemann_confusion.png)

*Figure 7. Row-normalized confusion matrices for Riemann transfer, one panel per calibration setting, summed over targets and seeds.*

![EEGNet transfer confusion matrices](assets/generated/transfer_repeated_eegnet_confusion.png)

*Figure 8. Row-normalized confusion matrices for EEGNet transfer, one panel per calibration setting, summed over targets and seeds.*

### 4.4 EEGNet training-budget sensitivity

To test whether the `50`-epoch cap limits EEGNet, the within-subject and LOSO runs were repeated with at most `300` epochs, patience `30`, and a minimum of `30` epochs (within-subject batch size `32`, LOSO batch size `64`).

| Protocol | `50` max epochs | `300` max epochs | Early stopping at `300` | Mean best / executed epoch at `300` |
|---|---:|---:|---:|---:|
| Within-subject | `0.6933` | `0.7423` | `45 / 45` fits | `77.2 / 107.2` |
| LOSO | `0.7068` | `0.7218` | `7 / 9` fits | `214.9 / 239.3` |

Per-subject accuracy with `300` max epochs:

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Within-subject | `0.8924` | `0.4930` | `0.9583` | `0.7849` | `0.5351` | `0.5036` | `0.6490` | `0.9582` | `0.9062` |
| LOSO | `0.8021` | `0.6146` | `0.8750` | `0.5972` | `0.5868` | `0.6840` | `0.6424` | `0.9340` | `0.7604` |

These runs support four observations.

- Every within-subject fit stopped early before `300` epochs, at a mean best epoch of `77`, so the `50`-epoch cap stopped training before convergence.
- The longer budget raises within-subject EEGNet from `0.693` to `0.742`, largely through S1 (`0.555` to `0.892`), with smaller gains in most other subjects. S2, S5, and S6 stay near chance, and S6 has macro F1 `0.418`.
- LOSO fits continue for about `240` epochs on average and gain `0.015` in mean accuracy.
- Even with the longer budget, within-subject EEGNet stays below the FBCSP mean (`0.818`). This comparison was not tested formally.

EEGNet also varies between runs. Relative to an earlier run of the same `50`-epoch configuration on a different environment, within-subject S1 changed from `0.706` to `0.555` and LOSO S9 from `0.594` to `0.757`, while the means moved by about `0.01`. It is unknown how much of this comes from library versions, the compute device (Apple MPS here), or nondeterministic kernels. Single-run EEGNet results for individual subjects should therefore be read with caution.

### 4.5 Exploratory analysis

The exploratory figures use epochs from `-1.5` to `4.5 s` around the cue, so that the reference interval and the task interval stay away from the wavelet edge effects at both ends of each epoch. Time-frequency power uses Morlet wavelets (`8-30 Hz`, `n_cycles = f / 2`). Event-related desynchronization and synchronization (ERD/ERS) follow the classical definition (Pfurtscheller and Lopes da Silva, 1999): power is first averaged over trials and then expressed relative to the mean of the pre-cue reference interval `[-1.0, -0.2] s`:

$$
\mathrm{ERD/ERS}(t, f) = 100 \times \frac{\bar{P}(t, f) - \bar{R}(f)}{\bar{R}(f)}
$$

where $\bar{P}(t, f)$ is the trial-averaged power and $\bar{R}(f)$ its mean over the reference interval. Negative values (ERD) mean less power than during the reference interval. Normalizing each trial by its own short reference interval before averaging is avoided because it biases the result upward.

![Subject 1 PSD by class](assets/eda_subject_1/subject_1_psd.png)

*Figure 9. Subject 1 class-wise power spectral density (Welch), averaged over channels and trials, in dB relative to 1 µV²/Hz.*

![Subject 1 PCA](assets/eda_subject_1/subject_1_pca.png)

*Figure 10. Subject 1 trials projected onto the first two principal components of standardized channel log-variance features.*

![Subject 1 t-SNE](assets/eda_subject_1/subject_1_tsne.png)

*Figure 11. Subject 1 t-SNE embedding of the same channel log-variance features. Unsupervised embeddings are shown for visual inspection only.*

![Subject 1 channel topography](assets/eda_subject_1/subject_1_topomap.png)

*Figure 12. Subject 1 mean log-variance per channel for left- and right-hand imagery, and their difference (log ratio). The left-minus-right difference is positive over the left hemisphere and negative over the right, with its largest magnitude (about ±0.1) over centro-parietal and parietal sites rather than directly at C3 and C4.*

![Subject 1 ERD/ERS maps](assets/eda_subject_1/subject_1_erds.png)

*Figure 13. Subject 1 ERD/ERS time-frequency maps at C3 and C4 for each class, relative to the pre-cue reference interval.*

![Subject 1 sensorimotor ERD/ERS summary](assets/eda_subject_1/subject_1_sensorimotor_erds.png)

*Figure 14. Subject 1 mu (8-12 Hz) and beta (13-30 Hz) ERD/ERS at C3 and C4. The grey area marks the reference interval and the dotted line the cue. After the cue, mu power falls by roughly 20-30% at C4 during left-hand imagery and by up to about 39% at C3 during right-hand imagery; beta changes are smaller.*

![Subject 2 sensorimotor ERD/ERS summary](assets/eda_subject_2/subject_2_sensorimotor_erds.png)

*Figure 15. Subject 2 mu and beta ERD/ERS at C3 and C4. Subject 2 is one of the subjects that most models decode close to chance.*

![Subject 8 sensorimotor ERD/ERS summary](assets/eda_subject_8/subject_8_sensorimotor_erds.png)

*Figure 16. Subject 8 mu and beta ERD/ERS at C3 and C4. Subject 8 is one of the best-decoded subjects across protocols.*

![Grand-average sensorimotor ERD/ERS summary](assets/group_eda/grand_average_sensorimotor_erds.png)

*Figure 17. Grand-average mu and beta ERD/ERS across the nine subjects (shaded: ± standard error across subjects).*

Approximate values read from Figure 17:

- After the cue, mu and beta power decrease at both electrodes, and the decrease is stronger over the hemisphere contralateral to the imagined hand.
- For right-hand imagery, mu ERD reaches about -33% at C3 versus about -21% at C4.
- For left-hand imagery, mu ERD reaches about -27% at C4 versus about -20% at C3.
- The mu ERD weakens after about 2 s, and a short positive deflection right after the cue may reflect the visual cue response.

![Subject 1 classical spatial patterns](assets/eda_subject_1/subject_1_classical_patterns.png)

*Figure 18. Subject 1 CSP patterns (components 1 and 4) and the first FBCSP pattern of the lowest and highest frequency bands, fitted on all trials of the subject. The FBCSP panels show fixed bands, not necessarily the selected features.*

![Subject 1 Riemann tangent-space view](assets/eda_subject_1/subject_1_riemann_3d.png)

*Figure 19. Subject 1 tangent-space features projected onto three principal components, fitted on all trials (qualitative view).*

![Subject 1 Riemann + LDA out-of-fold scores](assets/eda_subject_1/subject_1_riemann_lda_distribution.png)

*Figure 20. Subject 1 Riemann + LDA decision scores, where each trial is scored by a model trained without it (5-fold cross-validation).*

![Subject 1 CSP + LDA out-of-fold scores](assets/eda_subject_1/subject_1_csp_lda_distribution.png)

*Figure 21. Subject 1 CSP + LDA decision scores obtained out-of-fold in the same way as Figure 20.*

![Subject 1 CSP feature projection](assets/eda_subject_1/subject_1_csp_projection.png)

*Figure 22. Subject 1 trials in the space of the first two CSP components, fitted and plotted on the same trials (in-sample, so separability is optimistic).*

![Subject 1 EEGNet saliency topomaps](assets/eda_subject_1/subject_1_eegnet_saliency_topomap.png)

*Figure 23. Subject 1 input-gradient saliency of an EEGNet trained on all trials of the subject for 15 epochs. This is a first-order attribution summary of a single model, not a mechanistic explanation.*

## 5. Discussion

Mean accuracy suggests that the ranking depends on the protocol:

- FBCSP, Riemann, and CSP lead within subjects.
- EEGNet leads in LOSO and transfer.
- Riemann is the strongest non-deep model across subjects.

The evidence for these rankings is limited. With nine subjects and the multiple-comparison families defined above, no comparison among CSP, FBCSP, Riemann, and EEGNet is significant in any protocol. The most consistent signals are directional: EEGNet is higher than every other model in eight of nine LOSO subjects, and the transfer ordering EEGNet > Riemann > FBCSP holds at all five calibration budgets. A study designed to test these claims would need more subjects or datasets, or a smaller set of pre-specified primary comparisons.

The EEGNet results depend on how the network is trained. The `50`-epoch budget stopped within-subject training before convergence, a longer budget raised the within-subject mean by about five points, and single-subject results changed noticeably between runs. The within-subject gap between EEGNet and the best classical pipelines is therefore partly a property of the training recipe. A larger training set in LOSO also gives EEGNet more optimization steps per epoch than within-subject training (about `29` versus `3` mini-batches per epoch at batch size `64`), which confounds the comparison of deep and classical models across protocols.

In transfer, EEGNet fine-tunes all layers, including batch-normalization statistics, on the target trials, while FBCSP and Riemann only add the target trials to a much larger source training set. The transfer comparison therefore reflects both the model family and the adaptation strategy.

Limitations:

- Within-subject CV pools the two recording sessions. Cross-session evaluation (train on session 1, test on session 2) is the standard for this dataset and is likely to give lower accuracy.
- Nine subjects, two transfer seeds, and a single training seed for within-subject and LOSO EEGNet limit precision.
- FBCSP and EEGNet are simplified relative to their original publications, and no hyperparameters were tuned.
- Runs on Apple MPS are not guaranteed to be bit-for-bit reproducible.

Further work:

- add a cross-session protocol
- repeat EEGNet within-subject and LOSO runs over several training seeds
- pre-specify primary comparisons
- evaluate subject-alignment methods (for example Riemannian re-centering) that adapt the non-deep models more directly to a new subject

## 6. Conclusion

Under one preprocessing path and three explicitly separated protocols:

- **Within-subject**: FBCSP has the highest mean accuracy (`0.818`), but it does not differ significantly from Riemann (`0.798`) or CSP (`0.781`).
- **LOSO**: EEGNet has the highest mean accuracy (`0.707`) and is the best model for eight of nine held-out subjects, without significance after correction.
- **Transfer**: the ordering EEGNet > Riemann > FBCSP is consistent across calibration budgets, again without significance after correction.
- **Raw log-variance power**: it is a meaningful baseline (`0.710` within-subject, `0.613` LOSO) once the features are computed at the correct signal scale.

These results are a consistent descriptive picture rather than statistically established rankings, and the EEGNet results depend on the training budget.

## Appendix A. EEGNet optimization curves

![Within-subject EEGNet learning curve](assets/generated/within_subject_eegnet_learning_curve.png)

*Figure A1. Mean within-subject EEGNet training and validation loss (`50` max epochs). Each epoch averages only the folds still training, so the tail of the curve is computed from fewer folds.*

![LOSO EEGNet learning curve](assets/generated/loso_eegnet_learning_curve.png)

*Figure A2. Mean LOSO EEGNet training and validation loss (`50` max epochs), averaged in the same way.*

## Appendix B. Corrections to the previous version of this report

1. **Raw Power + LDA.** The log-variance features used a fixed floor of `1e-10`. Because the signals are in volts, typical channel variances (about `2.5e-11 V²`) fell below the floor and every feature became the same constant, so the classifier predicted a single class. The reported `0.5355` (within-subject) and `0.5270` (LOSO) were artifacts; the corrected values are `0.7099` and `0.6134`.
2. **ERD/ERS figures.** The previous figures normalized each trial by its own `0.5 s` reference interval at the edge of the epoch before averaging. This produced a sustained apparent power increase of `+50%` to `+100%` even for data without any task effect. The figures now use the trial-averaged definition and a reference interval away from the epoch edges.
3. **LOSO EEGNet spread.** The previously reported standard deviation (`0.1358`) did not match the per-subject values of the same run (`0.0927`, population SD).
4. **Transfer "± Std" and "95% CI".** These were computed across the five calibration settings rather than across subjects or seeds. All intervals are now computed across target subjects.
5. **Transfer p-values.** The previous tests treated seed-target pairs as independent, counting each target twice, and were not corrected for multiple comparisons. Tests are now exact, use target subjects as units, and are Holm-adjusted.
6. **Repeated-seed EEGNet transfer.** The previous seeds changed only the calibration split; the network training seed was fixed. Each seed now also sets the EEGNet training seed.
7. **Figures.** The transfer figure now includes FBCSP, and transfer confusion matrices are shown per calibration setting instead of pooled over settings.
8. **Classical, Riemann, and EEGNet reruns.** CSP and FBCSP reproduced their previous values exactly. Riemann changed slightly (within-subject `0.7956` to `0.7983`, LOSO `0.6258` to `0.6285`). EEGNet changed by about `0.01` in mean accuracy and more for individual subjects. The software and hardware of the previous runs are unknown.
9. **Removed content.** The earlier subject-1 sanity-check table and runtime table were removed: the first contained the invalid Raw Power value, and the second came from an undocumented environment.

## Appendix C. Reproducibility

- environment for the reported results: Python `3.14.7`, MOABB `1.7.2`, MNE `1.13.2`, PyTorch `2.14.0` (Apple MPS), scikit-learn `1.9.1`, pyRiemann `0.12`, NumPy `2.5.3`, SciPy `1.18.1`, macOS arm64
- seeds: `42` for CV splits and EEGNet; transfer seeds `42` and `43`
- tables and statistics are regenerated from the saved outputs with `--experiment export_assets` and `--experiment export_stats`, which also list the result files they used

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
