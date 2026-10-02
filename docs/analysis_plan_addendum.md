# Analysis Plan Addendum: Training-Size-Matched LOSO

This addendum was written after the primary results of [`analysis_plan.md`](analysis_plan.md) were known. The original plan is unchanged. The addendum specifies one follow-up analysis before it is run, and that analysis serves the original research question. It is not part of the confirmatory family P1-P3.

- Date fixed: 2026-10-02
- Code: the commit that adds this file
- Status of the data when the addendum was written:
  - P1-P3 and the exploratory protocol-by-model tests had been computed.
  - No result of the matched protocol below existed.

## Motivation

In the primary results, EEGNet was more accurate than the Riemannian model in LOSO (P3) but not more accurate than FBCSP in cross-session evaluation (P1). The exploratory protocol-by-model test found EEGNet's advantage to be larger in LOSO.

The two protocols differ in more than the source of the training data:

| | Cross-session | LOSO |
|---|---|---|
| Training trials per fit | 144 | 2304 |
| Test trials | session 2 only | both sessions |

The protocol difference can therefore not be attributed to cross-subject generalization. This addendum removes both confounds.

## Protocol: matched LOSO

- **Training set**: for each held-out subject, 144 trials from the other eight subjects, 9 trials per class per subject, drawn from both sessions without replacement.
- **Test set**: session 2 of the held-out subject (144 trials). These are the same test trials as in cross-session evaluation.
- **Sampling**: the training subsample is drawn with a seed. Seeds `42, 43, 44, 45, 46` are used for every model, so each model sees five different subsamples, and per-subject accuracy is the mean over seeds. For EEGNet the same seed also sets the network training seed.
- **Models and recipe**: all models as in the original plan. EEGNet uses recipe R with batch size `32`, the size rule for a training set of one session.
- **Result**: cross-session and matched LOSO now differ only in where the 144 training trials come from (the test subject's first session versus eight other subjects).

## Comparisons

The two tests below form one family, Holm-adjusted. Each is an exact paired sign-flip test over the nine subjects, applied to the per-subject difference of a model pair between the two protocols.

| ID | Model pair | Compared quantity |
|---|---|---|
| A1 | EEGNet − Riemann | (difference under matched LOSO) − (difference under cross-session) |
| A2 | EEGNet − FBCSP | (difference under matched LOSO) − (difference under cross-session) |

Interpretation rule, fixed in advance:

- If A1/A2 remain positive and significant, EEGNet's relative advantage depends on the source of the training data (other subjects versus the same subject), not only on its amount.
- If they are not significant, the larger advantage observed in full LOSO is not shown to be independent of training-set size.

Descriptive results of matched LOSO (all models, per subject) are reported alongside without further tests beyond the standard all-pairs family.

## Execution

```bash
python -m eeg_motor_imagery_classification.reproduce --stages addendum,export
```

The same execution rules as in the original plan apply.
