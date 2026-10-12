# Final ECG classification results

## Experiment configuration

- Heartbeats: 107,049
- Samples per heartbeat: 252
- Wavelet: bior6.8, level 8, symmetric extension
- Retained wavelet coefficients: cD1-cD8 (365 values)
- PCA components: 12, fitted on each training fold only
- SVM: RBF, C=10, gamma=0.1, one-versus-one
- RR source: `beat_only`
- Class weighting: `balanced`, calculated from each training fold only
- Beat evaluation: shuffled stratified 10-fold split
- Record evaluation: ten predefined overlapping paper holdouts

## Fold-mean results (%)

| Scheme | Model | Overall accuracy | Balanced accuracy | Macro-F1 | Paper SEN | Paper SPE | Paper ACC |
|---|---|---:|---:|---:|---:|---:|---:|
| beat | Baseline | 99.13 | 96.33 | 97.40 | 99.13 | 99.83 | 99.71 |
| beat | Weighted SVM | 98.46 | 97.80 | 95.75 | 98.46 | 99.69 | 99.49 |
| beat | PCA + RR | 99.51 | 98.35 | 98.76 | 99.51 | 99.90 | 99.84 |
| beat | Weighted PCA + RR | 99.38 | 98.63 | 98.38 | 99.38 | 99.88 | 99.79 |
| record | Baseline | 43.89 | 29.64 | 19.28 | 43.89 | 88.78 | 81.30 |
| record | Weighted SVM | 38.20 | 29.46 | 19.39 | 38.20 | 87.64 | 79.40 |
| record | PCA + RR | 50.00 | 38.27 | 28.81 | 50.00 | 90.00 | 83.33 |
| record | Weighted PCA + RR | 50.14 | 39.63 | 29.57 | 50.14 | 90.03 | 83.38 |

## Main extension result

The best RR-based record model by fold-mean macro-F1 was **Weighted PCA + RR**.
Its record-based macro-F1 was **29.57%**, compared with
**19.28%** for the morphology-only baseline, a
change of **+10.29 percentage points**. The paired fold comparison
improved in **10 of 10** holdouts and worsened in
**0 of 10**.

The primary scientific comparison should emphasize record-based balanced
accuracy and macro-F1 rather than accuracy alone because class N dominates the
dataset.

## Figures

- [Beat-based model comparison](figures/beat_model_comparison.png)
- [Record-based model comparison](figures/record_model_comparison.png)
- [Record-based per-class sensitivity](figures/record_per_class_sensitivity.png)
- [Record-based confusion matrices](figures/record_confusion_comparison.png)

## Interpretation cautions

1. The beat-based scheme shares records between training and testing and is
   therefore optimistic for unseen-patient performance.
2. The paper record experiments overlap across holdouts. Only
   32 of 48
   records appear in a test holdout, so pooled results are repeat-weighted.
   Fold means are used for the main comparisons.
3. The paper reports 373 wavelet candidates, while the reproducible stated
   252-sample bior6.8 decomposition produces 365. No speculative padding was
   introduced.
4. Beat-only RR intervals exclude non-beat annotation markers. This differs
   from the partner-compatible reference RR matrix and is the primary extension.
5. `post_RR` requires observing the following beat. These results represent
   offline classification, not zero-latency real-time prediction.
6. These experiments use one database and do not constitute clinical
   validation.
