# Statistical Comparisons

Units are subjects (transfer: target subjects, seeds averaged within target). SD uses ddof=1, CIs use the t distribution, paired sign-flip tests are exact, and p-values are Holm-adjusted within each family.

## Primary (pre-specified) comparisons

Family of 3 tests defined in docs/analysis_plan.md; Holm adjustment across this family only. 3 of 3 comparisons have results available.

| Setting | A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Cross-Session (session 1 -> 2) | EEGNet | FBCSP + LDA | 9 | -0.0546 | [-0.1304, +0.0212] | 3/6 | 0.1016 | 0.2031 |
| Cross-Session (session 1 -> 2) | Riemann + Tangent Space + LDA | FBCSP + LDA | 9 | -0.0015 | [-0.0716, +0.0686] | 4/4 | 0.9844 | 0.9844 |
| LOSO | EEGNet | Riemann + Tangent Space + LDA | 9 | +0.0954 | [+0.0552, +0.1357] | 9/0 | 0.0039 | 0.0117 |

## Within-Subject CV (sessions pooled)

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| Raw Power + LDA | 9 | 0.7099 | 0.1331 | [0.6075, 0.8122] |
| CSP + LDA | 9 | 0.7810 | 0.1368 | [0.6758, 0.8861] |
| FBCSP + LDA | 9 | 0.8183 | 0.1383 | [0.7120, 0.9246] |
| Riemann + Tangent Space + LDA | 9 | 0.7983 | 0.1245 | [0.7025, 0.8940] |
| EEGNet | 9 | 0.7459 | 0.1885 | [0.6010, 0.8909] |

| A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|
| Raw Power + LDA | CSP + LDA | 9 | -0.0711 | [-0.0971, -0.0450] | 0/9 | 0.0039 | 0.0391 |
| Raw Power + LDA | FBCSP + LDA | 9 | -0.1084 | [-0.1851, -0.0317] | 1/8 | 0.0078 | 0.0625 |
| Raw Power + LDA | Riemann + Tangent Space + LDA | 9 | -0.0884 | [-0.1235, -0.0533] | 0/9 | 0.0039 | 0.0391 |
| Raw Power + LDA | EEGNet | 9 | -0.0361 | [-0.1023, +0.0302] | 4/5 | 0.2344 | 1.0000 |
| CSP + LDA | FBCSP + LDA | 9 | -0.0373 | [-0.1127, +0.0380] | 3/6 | 0.3281 | 1.0000 |
| CSP + LDA | Riemann + Tangent Space + LDA | 9 | -0.0173 | [-0.0439, +0.0092] | 3/6 | 0.1758 | 1.0000 |
| CSP + LDA | EEGNet | 9 | +0.0350 | [-0.0415, +0.1116] | 4/5 | 0.3281 | 1.0000 |
| FBCSP + LDA | Riemann + Tangent Space + LDA | 9 | +0.0200 | [-0.0631, +0.1032] | 6/3 | 0.6523 | 1.0000 |
| FBCSP + LDA | EEGNet | 9 | +0.0724 | [-0.0374, +0.1822] | 5/4 | 0.1953 | 1.0000 |
| Riemann + Tangent Space + LDA | EEGNet | 9 | +0.0523 | [-0.0205, +0.1251] | 6/3 | 0.1445 | 1.0000 |

Per-subject accuracy:

| Subject | Raw Power + LDA | CSP + LDA | FBCSP + LDA | Riemann + Tangent Space + LDA | EEGNet |
|---|---:|---:|---:|---:|---:|
| S1 | 0.7429 | 0.8368 | 0.9132 | 0.8472 | 0.8744 |
| S2 | 0.5348 | 0.5491 | 0.6007 | 0.5938 | 0.4999 |
| S3 | 0.8819 | 0.9584 | 0.9583 | 0.9792 | 0.9645 |
| S4 | 0.6147 | 0.6913 | 0.7121 | 0.7849 | 0.7480 |
| S5 | 0.5767 | 0.6524 | 0.9134 | 0.6528 | 0.5447 |
| S6 | 0.6248 | 0.7325 | 0.6246 | 0.7535 | 0.5901 |
| S7 | 0.7082 | 0.8232 | 0.8473 | 0.8055 | 0.6121 |
| S8 | 0.9063 | 0.9549 | 0.9515 | 0.9514 | 0.9680 |
| S9 | 0.7987 | 0.8300 | 0.8437 | 0.8162 | 0.9118 |

EEGNet accuracy is averaged over training seeds 42, 43, 44, 45, 46; between-seed SD per subject:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0071 | 0.0245 | 0.0038 | 0.0261 | 0.0302 | 0.0729 | 0.0382 | 0.0080 | 0.0162 |

## LOSO

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| Raw Power + LDA | 9 | 0.6134 | 0.0938 | [0.5413, 0.6855] |
| CSP + LDA | 9 | 0.5907 | 0.1148 | [0.5025, 0.6789] |
| FBCSP + LDA | 9 | 0.5648 | 0.0678 | [0.5127, 0.6169] |
| Riemann + Tangent Space + LDA | 9 | 0.6285 | 0.1039 | [0.5486, 0.7083] |
| EEGNet | 9 | 0.7239 | 0.1234 | [0.6291, 0.8188] |

| A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|
| Raw Power + LDA | CSP + LDA | 9 | +0.0228 | [-0.0553, +0.1008] | 4/5 | 0.5352 | 1.0000 |
| Raw Power + LDA | FBCSP + LDA | 9 | +0.0486 | [-0.0342, +0.1314] | 5/4 | 0.2617 | 1.0000 |
| Raw Power + LDA | Riemann + Tangent Space + LDA | 9 | -0.0150 | [-0.0499, +0.0198] | 2/7 | 0.3633 | 1.0000 |
| Raw Power + LDA | EEGNet | 9 | -0.1105 | [-0.1589, -0.0621] | 1/8 | 0.0078 | 0.0625 |
| CSP + LDA | FBCSP + LDA | 9 | +0.0258 | [-0.0784, +0.1301] | 6/3 | 0.6523 | 1.0000 |
| CSP + LDA | Riemann + Tangent Space + LDA | 9 | -0.0378 | [-0.1349, +0.0592] | 5/4 | 0.4023 | 1.0000 |
| CSP + LDA | EEGNet | 9 | -0.1333 | [-0.2201, -0.0465] | 0/9 | 0.0039 | 0.0391 |
| FBCSP + LDA | Riemann + Tangent Space + LDA | 9 | -0.0637 | [-0.1508, +0.0235] | 2/7 | 0.1445 | 0.8672 |
| FBCSP + LDA | EEGNet | 9 | -0.1591 | [-0.2644, -0.0538] | 1/8 | 0.0078 | 0.0625 |
| Riemann + Tangent Space + LDA | EEGNet | 9 | -0.0954 | [-0.1357, -0.0552] | 0/9 | 0.0039 | 0.0391 |

Per-subject accuracy:

| Subject | Raw Power + LDA | CSP + LDA | FBCSP + LDA | Riemann + Tangent Space + LDA | EEGNet |
|---|---:|---:|---:|---:|---:|
| S1 | 0.6285 | 0.5139 | 0.6701 | 0.7153 | 0.8090 |
| S2 | 0.4792 | 0.5069 | 0.5243 | 0.4965 | 0.6139 |
| S3 | 0.7674 | 0.5590 | 0.5486 | 0.8090 | 0.8632 |
| S4 | 0.6285 | 0.6146 | 0.6458 | 0.5486 | 0.6222 |
| S5 | 0.5035 | 0.5174 | 0.5000 | 0.5139 | 0.5736 |
| S6 | 0.6042 | 0.5035 | 0.5000 | 0.6111 | 0.7333 |
| S7 | 0.5625 | 0.5729 | 0.5556 | 0.5972 | 0.6174 |
| S8 | 0.7292 | 0.8576 | 0.5035 | 0.7153 | 0.9160 |
| S9 | 0.6181 | 0.6701 | 0.6354 | 0.6493 | 0.7667 |

EEGNet accuracy is averaged over training seeds 42, 43, 44, 45, 46; between-seed SD per subject:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0185 | 0.0186 | 0.0364 | 0.0300 | 0.0247 | 0.0328 | 0.0464 | 0.0135 | 0.0280 |

## Cross-Session (session 1 -> 2)

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| Raw Power + LDA | 9 | 0.6898 | 0.1381 | [0.5837, 0.7959] |
| CSP + LDA | 9 | 0.7230 | 0.1641 | [0.5969, 0.8491] |
| FBCSP + LDA | 9 | 0.7600 | 0.1600 | [0.6370, 0.8830] |
| Riemann + Tangent Space + LDA | 9 | 0.7585 | 0.1505 | [0.6428, 0.8742] |
| EEGNet | 9 | 0.7054 | 0.1810 | [0.5662, 0.8446] |

| A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|
| Raw Power + LDA | CSP + LDA | 9 | -0.0332 | [-0.0749, +0.0086] | 2/7 | 0.1094 | 0.8125 |
| Raw Power + LDA | FBCSP + LDA | 9 | -0.0702 | [-0.1435, +0.0031] | 1/8 | 0.0391 | 0.3906 |
| Raw Power + LDA | Riemann + Tangent Space + LDA | 9 | -0.0687 | [-0.1293, -0.0081] | 2/7 | 0.0430 | 0.3906 |
| Raw Power + LDA | EEGNet | 9 | -0.0156 | [-0.0706, +0.0395] | 5/4 | 0.5469 | 1.0000 |
| CSP + LDA | FBCSP + LDA | 9 | -0.0370 | [-0.1086, +0.0345] | 4/5 | 0.2812 | 1.0000 |
| CSP + LDA | Riemann + Tangent Space + LDA | 9 | -0.0355 | [-0.0917, +0.0207] | 3/6 | 0.1992 | 0.9961 |
| CSP + LDA | EEGNet | 9 | +0.0176 | [-0.0332, +0.0683] | 5/4 | 0.4609 | 1.0000 |
| FBCSP + LDA | Riemann + Tangent Space + LDA | 9 | +0.0015 | [-0.0686, +0.0716] | 4/4 | 0.9844 | 1.0000 |
| FBCSP + LDA | EEGNet | 9 | +0.0546 | [-0.0212, +0.1304] | 6/3 | 0.1016 | 0.8125 |
| Riemann + Tangent Space + LDA | EEGNet | 9 | +0.0531 | [-0.0173, +0.1235] | 6/3 | 0.1289 | 0.8125 |

Per-subject accuracy:

| Subject | Raw Power + LDA | CSP + LDA | FBCSP + LDA | Riemann + Tangent Space + LDA | EEGNet |
|---|---:|---:|---:|---:|---:|
| S1 | 0.6944 | 0.7917 | 0.9028 | 0.8611 | 0.8389 |
| S2 | 0.5486 | 0.6042 | 0.5694 | 0.5694 | 0.5014 |
| S3 | 0.8819 | 0.8958 | 0.9444 | 0.9653 | 0.8556 |
| S4 | 0.6042 | 0.5972 | 0.6111 | 0.7431 | 0.6528 |
| S5 | 0.5417 | 0.5833 | 0.7847 | 0.6875 | 0.5000 |
| S6 | 0.6319 | 0.6667 | 0.5764 | 0.7153 | 0.5694 |
| S7 | 0.5972 | 0.5139 | 0.6389 | 0.5347 | 0.5750 |
| S8 | 0.8681 | 0.9514 | 0.9167 | 0.9375 | 0.9486 |
| S9 | 0.8403 | 0.9028 | 0.8958 | 0.8125 | 0.9069 |

EEGNet accuracy is averaged over training seeds 42, 43, 44, 45, 46; between-seed SD per subject:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0205 | 0.0227 | 0.1989 | 0.0461 | 0.0208 | 0.0517 | 0.0284 | 0.0188 | 0.0160 |

## LOSO, 144 training trials (tested on session 2)

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| Raw Power + LDA | 9 | 0.5804 | 0.0675 | [0.5285, 0.6323] |
| CSP + LDA | 9 | 0.5806 | 0.0625 | [0.5325, 0.6286] |
| FBCSP + LDA | 9 | 0.5602 | 0.0594 | [0.5145, 0.6059] |
| Riemann + Tangent Space + LDA | 9 | 0.6017 | 0.0697 | [0.5481, 0.6553] |
| EEGNet | 9 | 0.5340 | 0.0318 | [0.5095, 0.5584] |

| A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|
| Raw Power + LDA | CSP + LDA | 9 | -0.0002 | [-0.0560, +0.0557] | 4/5 | 1.0000 | 1.0000 |
| Raw Power + LDA | FBCSP + LDA | 9 | +0.0202 | [-0.0398, +0.0802] | 6/3 | 0.4883 | 1.0000 |
| Raw Power + LDA | Riemann + Tangent Space + LDA | 9 | -0.0213 | [-0.0702, +0.0276] | 3/6 | 0.3594 | 1.0000 |
| Raw Power + LDA | EEGNet | 9 | +0.0465 | [-0.0082, +0.1011] | 7/2 | 0.0781 | 0.4688 |
| CSP + LDA | FBCSP + LDA | 9 | +0.0204 | [-0.0108, +0.0515] | 6/3 | 0.1914 | 0.9570 |
| CSP + LDA | Riemann + Tangent Space + LDA | 9 | -0.0211 | [-0.0612, +0.0189] | 3/6 | 0.3047 | 1.0000 |
| CSP + LDA | EEGNet | 9 | +0.0466 | [+0.0178, +0.0754] | 8/1 | 0.0078 | 0.0703 |
| FBCSP + LDA | Riemann + Tangent Space + LDA | 9 | -0.0415 | [-0.0854, +0.0024] | 2/7 | 0.0312 | 0.2500 |
| FBCSP + LDA | EEGNet | 9 | +0.0262 | [-0.0014, +0.0539] | 8/1 | 0.0352 | 0.2500 |
| Riemann + Tangent Space + LDA | EEGNet | 9 | +0.0677 | [+0.0235, +0.1120] | 9/0 | 0.0039 | 0.0391 |

Per-subject accuracy:

| Subject | Raw Power + LDA | CSP + LDA | FBCSP + LDA | Riemann + Tangent Space + LDA | EEGNet |
|---|---:|---:|---:|---:|---:|
| S1 | 0.6125 | 0.5653 | 0.5625 | 0.7083 | 0.5264 |
| S2 | 0.4958 | 0.5083 | 0.4944 | 0.5333 | 0.5153 |
| S3 | 0.7139 | 0.6347 | 0.5403 | 0.6667 | 0.5306 |
| S4 | 0.5875 | 0.5903 | 0.5736 | 0.5931 | 0.5597 |
| S5 | 0.5875 | 0.4972 | 0.5278 | 0.5236 | 0.4875 |
| S6 | 0.5750 | 0.5514 | 0.5167 | 0.5250 | 0.5153 |
| S7 | 0.5361 | 0.5542 | 0.5556 | 0.5833 | 0.5194 |
| S8 | 0.6181 | 0.6847 | 0.7028 | 0.6806 | 0.5958 |
| S9 | 0.4972 | 0.6389 | 0.5681 | 0.6014 | 0.5556 |

EEGNet accuracy is averaged over training seeds 42, 43, 44, 45, 46; between-seed SD per subject:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0750 | 0.0528 | 0.0828 | 0.0656 | 0.0292 | 0.0227 | 0.0288 | 0.0600 | 0.0772 |

## Transfer (per shot setting, across target subjects)

### FBCSP

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| zero_shot | 9 | 0.5664 | 0.0650 | [0.5164, 0.6163] |
| 5_shot | 9 | 0.5687 | 0.0698 | [0.5150, 0.6223] |
| 10_shot | 9 | 0.5745 | 0.0710 | [0.5199, 0.6290] |
| 20_shot | 9 | 0.5826 | 0.0700 | [0.5288, 0.6364] |
| 30_shot | 9 | 0.5887 | 0.0830 | [0.5249, 0.6525] |

### Riemann

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| zero_shot | 9 | 0.6331 | 0.1056 | [0.5519, 0.7143] |
| 5_shot | 9 | 0.6671 | 0.1208 | [0.5742, 0.7599] |
| 10_shot | 9 | 0.6740 | 0.1232 | [0.5793, 0.7687] |
| 20_shot | 9 | 0.6968 | 0.1218 | [0.6031, 0.7904] |
| 30_shot | 9 | 0.7106 | 0.1327 | [0.6087, 0.8126] |

### EEGNet

| Label | n | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| zero_shot | 9 | 0.7276 | 0.1205 | [0.6350, 0.8202] |
| 5_shot | 9 | 0.7515 | 0.1261 | [0.6546, 0.8485] |
| 10_shot | 9 | 0.7650 | 0.1363 | [0.6602, 0.8699] |
| 20_shot | 9 | 0.7936 | 0.1191 | [0.7021, 0.8851] |
| 30_shot | 9 | 0.7940 | 0.1268 | [0.6965, 0.8915] |

### Paired comparisons

| Setting | A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| zero_shot | EEGNet | Riemann | 9 | +0.0945 | [+0.0527, +0.1363] | 9/0 | 0.0039 | 0.0586 |
| zero_shot | Riemann | FBCSP | 9 | +0.0667 | [-0.0272, +0.1607] | 7/2 | 0.1484 | 0.1484 |
| zero_shot | EEGNet | FBCSP | 9 | +0.1613 | [+0.0524, +0.2701] | 8/1 | 0.0078 | 0.0586 |
| 5_shot | EEGNet | Riemann | 9 | +0.0845 | [+0.0550, +0.1140] | 9/0 | 0.0039 | 0.0586 |
| 5_shot | Riemann | FBCSP | 9 | +0.0984 | [-0.0060, +0.2027] | 7/2 | 0.0469 | 0.0938 |
| 5_shot | EEGNet | FBCSP | 9 | +0.1829 | [+0.0801, +0.2856] | 9/0 | 0.0039 | 0.0586 |
| 10_shot | EEGNet | Riemann | 9 | +0.0910 | [+0.0411, +0.1410] | 9/0 | 0.0039 | 0.0586 |
| 10_shot | Riemann | FBCSP | 9 | +0.0995 | [-0.0003, +0.1994] | 8/1 | 0.0195 | 0.0586 |
| 10_shot | EEGNet | FBCSP | 9 | +0.1906 | [+0.0838, +0.2973] | 9/0 | 0.0039 | 0.0586 |
| 20_shot | EEGNet | Riemann | 9 | +0.0968 | [+0.0547, +0.1390] | 9/0 | 0.0039 | 0.0586 |
| 20_shot | Riemann | FBCSP | 9 | +0.1142 | [+0.0267, +0.2017] | 8/0 | 0.0078 | 0.0586 |
| 20_shot | EEGNet | FBCSP | 9 | +0.2110 | [+0.1256, +0.2964] | 9/0 | 0.0039 | 0.0586 |
| 30_shot | EEGNet | Riemann | 9 | +0.0833 | [+0.0386, +0.1281] | 9/0 | 0.0039 | 0.0586 |
| 30_shot | Riemann | FBCSP | 9 | +0.1219 | [+0.0222, +0.2216] | 8/1 | 0.0117 | 0.0586 |
| 30_shot | EEGNet | FBCSP | 9 | +0.2052 | [+0.1076, +0.3029] | 9/0 | 0.0039 | 0.0586 |

## Exploratory: model difference between protocols

Per subject, the accuracy difference of the listed model pair under protocol A is compared with the same difference under protocol B (exact paired sign-flip test, Holm across these tests). Not pre-specified.

| Setting | A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| EEGNet - Riemann + Tangent Space + LDA | LOSO | Cross-Session (session 1 -> 2) | 9 | +0.1485 | [+0.0752, +0.2219] | 8/1 | 0.0078 | 0.0156 |
| EEGNet - FBCSP + LDA | LOSO | Cross-Session (session 1 -> 2) | 9 | +0.2137 | [+0.0975, +0.3300] | 8/1 | 0.0078 | 0.0156 |

## Addendum: training-size-matched LOSO vs cross-session

Fixed in docs/analysis_plan_addendum.md before the matched runs. Both protocols use 144 training trials and the same test trials (session 2 of each subject); the training data come from the subject itself (cross-session) or from the other eight subjects (matched LOSO). Per subject, the model difference under matched LOSO is compared with the same difference under cross-session (exact paired sign-flip test, Holm across these two tests).

| Setting | A | B | n | Mean diff (A-B) | 95% CI of diff | Wins A/B | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| EEGNet - Riemann + Tangent Space + LDA | LOSO, 144 training trials (tested on session 2) | Cross-Session (session 1 -> 2) | 9 | -0.0147 | [-0.1053, +0.0760] | 4/5 | 0.7109 | 0.9375 |
| EEGNet - FBCSP + LDA | LOSO, 144 training trials (tested on session 2) | Cross-Session (session 1 -> 2) | 9 | +0.0284 | [-0.0537, +0.1105] | 6/3 | 0.4688 | 0.9375 |

## Runtime (wall-clock fit + predict, machine-specific)

| Result | Model | Runtime (s) |
|---|---|---:|
| within_classical | raw_power | 0.5 |
| within_classical | csp | 11.4 |
| within_classical | fbcsp | 251.0 |
| within_riemann | - | 7.1 |
| within_eegnet | - | 2776.0 |
| loso_classical | raw_power | 0.9 |
| loso_classical | csp | 23.6 |
| loso_classical | fbcsp | 492.6 |
| loso_riemann | - | 13.3 |
| loso_eegnet | - | 9787.0 |
| cross_session_classical | raw_power | 0.1 |
| cross_session_classical | csp | 1.5 |
| cross_session_classical | fbcsp | 40.0 |
| cross_session_riemann | - | 1.2 |
| cross_session_eegnet | - | 394.0 |
| loso_matched_classical | raw_power | 0.6 |
| loso_matched_classical | csp | 7.4 |
| loso_matched_classical | fbcsp | 203.0 |
| loso_matched_riemann | - | 6.1 |
| loso_matched_eegnet | - | 235.6 |
| transfer_fbcsp | - | 5118.2 |
| transfer_riemann | - | 190.2 |
| transfer_eegnet | - | 3626.2 |

## Environment

| Result | Python | torch | Torch device (EEGNet only) | CUDA device | Platform |
|---|---|---|---|---|---|
| within_classical | not recorded | | | | |
| within_riemann | not recorded | | | | |
| within_eegnet | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| loso_classical | not recorded | | | | |
| loso_riemann | not recorded | | | | |
| loso_eegnet | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| cross_session_classical | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| cross_session_riemann | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| cross_session_eegnet | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| loso_matched_classical | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| loso_matched_riemann | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| loso_matched_eegnet | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |
| transfer_fbcsp | not recorded | | | | |
| transfer_riemann | not recorded | | | | |
| transfer_eegnet | 3.14.7 | 2.14.0 | mps | - | macOS-27.0-arm64-arm-64bit-Mach-O |

## Sources

- `within_classical`: `outputs/within_subject_classical/result.json`
- `within_riemann`: `outputs/within_subject_riemann/result.json`
- `within_eegnet`: `outputs/within_subject_eegnet/result.json`
- `loso_classical`: `outputs/loso_classical/result.json`
- `loso_riemann`: `outputs/loso_riemann/result.json`
- `loso_eegnet`: `outputs/loso_eegnet/result.json`
- `cross_session_classical`: `outputs/cross_session_classical/result.json`
- `cross_session_riemann`: `outputs/cross_session_riemann/result.json`
- `cross_session_eegnet`: `outputs/cross_session_eegnet/result.json`
- `loso_matched_classical`: `outputs/loso_matched_classical/result.json`
- `loso_matched_riemann`: `outputs/loso_matched_riemann/result.json`
- `loso_matched_eegnet`: `outputs/loso_matched_eegnet/result.json`
- `transfer_fbcsp`: `outputs/transfer_classical_all_targets_seed42_43/result.json`
- `transfer_riemann`: `outputs/transfer_riemann_all_targets_seed42_43/result.json`
- `transfer_eegnet`: `outputs/transfer_eegnet_all_targets_seed42_43/result.json`
