# ECG Heartbeat Classification Using Wavelet Features, PCA and SVM

This repository contains the research paper selected for the **BM4152 Biosignal Processing** project and the presentation prepared for the mid-review. The project studies automated classification of ECG heartbeats using wavelet multi-resolution analysis, principal component analysis (PCA), and a radial basis function support vector machine (RBF-SVM).

## Repository contents

- [Research paper](./Combining%20Low-dimensional.pdf)
- [Mid-review presentation](./BM4152PAPER%20IMPLEMENTATION.pdf)
- [Implementation 1](./Implementation_1)
- [Implementation 2: audited reproduction and completed extension](./Implementation_2)

## Research objective

The selected study classifies six heartbeat types from the MIT-BIH Arrhythmia Database:

- **N** — Normal beat
- **A** — Atrial premature beat
- **V** — Premature ventricular contraction
- **L** — Left bundle branch block beat
- **R** — Right bundle branch block beat
- **P** — Paced beat

Its baseline processing pipeline is:

```text
ECG record
   -> heartbeat segmentation around the R peak (252 samples)
   -> eight-level bior6.8 wavelet decomposition
   -> concatenation of detail coefficients cD1-cD8 (373 coefficients)
   -> PCA dimensionality reduction (12 principal components)
   -> one-versus-one RBF-SVM classification
```

## Novelty of the selected literature

The main contribution is not simply the use of wavelets or SVM, since both had already been widely used for ECG classification. The novelty is the way the wavelet information is represented and reduced:

1. **All detail levels are retained.** Instead of choosing only selected wavelet sub-bands, the method concatenates the detail coefficients from levels 1 through 8 into a single 373-element vector.
2. **PCA is applied directly to the complete wavelet vector.** The method transforms the redundant coefficient vector into 12 compact, mutually orthogonal principal-component scores instead of manually selecting a subset of individual coefficients.
3. **A low-dimensional representation is obtained.** Only 12 features—approximately 3.2% of the original 373-dimensional vector—are passed to the classifier without a significant reduction in the reported beat-based performance.
4. **The complete effective beat set is used.** The study evaluates 107,049 beats from the MIT-BIH Arrhythmia Database across six classes, rather than demonstrating the method on only a small subset.
5. **Two validation perspectives are investigated.** Beat-based validation produces very high performance, while record-based validation tests generalization to unseen ECG records and exposes a substantial performance reduction.

The reported overall accuracy is approximately **99.70% under beat-based validation** and **81.47% under record-based validation**. This difference is important: randomly separating individual beats can place beats from the same record in both the training and test sets, producing an optimistic estimate of generalization.

## Identified limitations

- The dataset is strongly imbalanced, with normal beats forming the majority class.
- The SVM can therefore become biased toward predicting the normal class.
- Wavelet and PCA features describe beat morphology but do not explicitly describe the timing of a beat relative to neighbouring beats.
- Atrial premature beats can have morphology similar to normal beats, making morphology-only discrimination difficult.
- Strong beat-based results do not transfer directly to unseen records, as shown by the weaker record-based results.

## Proposed extension

We propose combining **morphological information, temporal information, and imbalance-aware classification**.

### 1. Add RR-interval features

The 12 PCA scores will be combined with three timing features:

- Previous RR interval
- Next RR interval
- Locally normalized RR interval

This produces a 15-dimensional feature vector:

```text
[PC1, PC2, ..., PC12, previous_RR, next_RR, normalized_RR]
```

The PCA scores describe heartbeat morphology, while the RR features describe whether a beat occurs earlier or later than expected. These temporal features are expected to be particularly useful for atrial and ventricular premature beats. They are not expected to improve all classes equally; bundle branch block beats, for example, are primarily distinguished by waveform morphology.

### 2. Use a class-weighted RBF-SVM

A class-weighted, one-versus-one RBF-SVM will assign larger misclassification penalties to minority classes. This is intended to reduce the model's tendency to classify abnormal beats as the majority normal class. The weights and SVM hyperparameters must be selected using only the training data in each fold.

### 3. Use leakage-free, record-based evaluation

All preprocessing operations—including feature standardization, PCA fitting, class-weight calculation, and hyperparameter selection—will be performed using training data only. Complete ECG records will be kept separate between training and testing so that the final evaluation measures generalization to unseen records.

### 4. Perform an ablation study

The following systems will be compared under identical record-based folds:

1. PCA features + standard RBF-SVM (baseline)
2. PCA features + RR features + standard RBF-SVM
3. PCA features + class-weighted RBF-SVM
4. PCA features + RR features + class-weighted RBF-SVM

This comparison will show whether any improvement comes from the temporal features, class weighting, or their combination.

## Evaluation metrics

Because ordinary accuracy can hide poor minority-class performance, the implementation will report:

- Per-class sensitivity/recall
- Precision
- Specificity
- F1-score
- Macro-averaged F1-score
- Balanced accuracy
- Confusion matrix

The main success criterion is improved minority-class recognition and record-based generalization without an unacceptable reduction in precision—not merely a higher overall accuracy.

## Current project status

- [x] Research paper selected and reviewed
- [x] Mid-review presentation prepared
- [ ] MIT-BIH data acquisition and preprocessing
- [ ] Baseline wavelet-PCA-SVM reproduction
- [ ] RR-feature implementation
- [ ] Class-weighted SVM implementation
- [ ] Record-based evaluation and ablation study
- [ ] Final report and presentation

## Important implementation note

The next RR interval is only known after the following heartbeat occurs. It is suitable for offline ECG analysis, but a real-time implementation would introduce a one-beat delay or would need to use only previous and historical RR features.

## Reference

The full selected article is included in this repository for academic project use. Please consult the paper for its complete methodology, experimental conditions, authorship, and citation details.
