# ECG Heartbeat Classification Using Wavelet, PCA, RR Features, and SVM

This repository reimplements the selected BM4152 research paper on six-class
MIT-BIH heartbeat classification and evaluates an extension that adds
RR-interval features and class-weighted RBF-SVM classification.

## Included material

- [Selected research paper](../Combining%20Low-dimensional.pdf)
- [Mid-review presentation](../BM4152PAPER%20IMPLEMENTATION.pdf)
- Reproducible Python implementation in [`src`](./src)
- Automated tests in [`tests`](./tests)

## Results at a glance

Mean over the ten record-based holdouts (%):

| Model | Paper SEN / ordinary accuracy | Balanced accuracy | Macro-F1 | Paper ACC |
|---|---:|---:|---:|---:|
| Reimplemented baseline | 43.89 | 29.64 | 19.28 | 81.30 |
| PCA + RR | 50.00 | 38.27 | 28.81 | 83.33 |
| Weighted PCA + RR | **50.14** | **39.63** | **29.57** | **83.38** |

For comparison, the paper reports record-based SEN = 44.40%, SPE = 88.88%,
and ACC = 81.47%. The close baseline values validate the reproduction. The
weighted PCA + RR extension improves record-based macro-F1 by 10.29 percentage
points and improves it in all ten paired holdouts. See the
[complete report](./results/final/FINAL_RESULTS.md) and
[model comparison table](./results/final/model_comparison.csv).

## Classes

- **N** - normal beat
- **A** - atrial premature beat
- **V** - premature ventricular contraction
- **L** - left bundle branch block beat
- **R** - right bundle branch block beat
- **P** - paced beat

## Reimplemented pipeline

```text
MIT-BIH ECG and annotations
  -> select one lead per record
  -> extract 252-sample beats around annotated R peaks
  -> 8-level bior6.8 DWT with symmetric boundary handling
  -> concatenate cD1-cD8 (365 reproducible coefficients)
  -> fit PCA on each training fold and retain 12 components
  -> one-versus-one RBF-SVM (C=10, gamma=0.1)
  -> beat-based and record-based evaluation
```

The paper reports 373 wavelet coefficients. With the stated 252-sample beat,
`bior6.8`, eight levels, and symmetric extension, both PyWavelets and the
partner implementation produce 365 retained detail coefficients. This project
records that discrepancy instead of adding artificial coefficients.

The dataset audit found 107,049 usable beats, matching the paper's total after
252-sample boundary filtering.

## Literature contribution being reproduced

The selected work differs from many earlier wavelet ECG classifiers in three
important ways:

1. It concatenates every detail band from cD1 through cD8 rather than manually
   selecting only a few sub-bands.
2. It applies PCA directly to the complete wavelet vector, creating 12 new
   components rather than selecting 12 original coefficients.
3. It reports both beat-based validation and the more difficult record-based
   validation, which better tests performance on unseen patients.

## Implemented extension

The extension combines morphology, rhythm, and imbalance-aware learning.

### Six RR features

The 12 PCA morphology scores are augmented with:

1. `pre_RR`
2. `post_RR`
3. `pre_RR / record_mean`
4. `post_RR / record_mean`
5. `pre_RR / local_mean`
6. `pre_RR / post_RR`

This gives 18 classifier inputs for RR models. Two RR matrices are retained:

- `reference`: partner-compatible intervals based on the full annotation
  sequence, including non-beat markers;
- `beat_only`: corrected intervals calculated from valid selected heartbeats
  only. This is the default scientific extension.

`post_RR` requires the next beat, so this configuration is intended for
offline analysis or classification with one-beat delay.

### Class-weighted RBF-SVM

Balanced class weights are computed from each training fold only:

```text
weight_c = number_of_training_samples / (number_of_classes * samples_in_class_c)
```

Minority-class errors therefore receive a larger penalty. The implementation
also supports the partner convention `max_class_count / class_count` for a
separate compatibility experiment.

### Leakage controls

PCA, RR scaling, and class-weight calculation are fitted only on the training
portion of every fold. Test records never affect those fitted transformations.

## Ablation study

Four systems are evaluated on identical folds:

1. `baseline`: PCA + ordinary RBF-SVM
2. `weighted`: PCA + class-weighted RBF-SVM
3. `rr`: PCA + six RR features + ordinary RBF-SVM
4. `weighted_rr`: PCA + six RR features + class-weighted RBF-SVM

The primary record-based outcomes are macro-F1 and balanced accuracy. Overall
accuracy is reported but is not sufficient for this imbalanced dataset.

## Current status

- [x] Dataset inspection and class audit
- [x] 252-sample heartbeat extraction
- [x] Wavelet feature extraction
- [x] Reference and corrected RR feature extraction
- [x] Beat-based and paper record-based split generation
- [x] Leakage-free PCA/SVM baseline implementation
- [x] Class-weighted and RR ablation implementations
- [x] Automated tests and one-fold smoke validation
- [x] Execute all 80 model/fold fits
- [x] Generate the final tables, plots, and interpretation report

## Reproduce the project

Activate the environment from Git Bash:

```bash
source .venv/Scripts/activate
```

If starting from raw MIT-BIH files, run the preparation stages:

```bash
python src/inspect_dataset.py
python src/extract_beats.py
python src/extract_features.py
python src/create_splits.py
python -m pytest -q
```

Run the complete ablation study:

```bash
python src/run_experiments.py \
  --scheme both \
  --models baseline,weighted,rr,weighted_rr \
  --rr-source beat_only \
  --weighting balanced
```

This performs 4 models x 2 validation schemes x 10 folds = 80 CPU-based SVM
fits. The runner saves after each fold and can be resumed by running the same
command again. A GPU or Google Colab is not required; scikit-learn's RBF-SVM
implementation used here runs on the CPU.

After all fits finish, create the final results package:

```bash
python src/make_results.py
```

The generated `results/final` directory will contain:

- `model_comparison.csv`
- `paired_fold_changes.csv`
- `FINAL_RESULTS.md`
- beat- and record-based comparison figures
- per-class record sensitivity and confusion-matrix figures

## Evaluation cautions

- Beat-based folds mix beats from the same records across training and test
  sets, so they are optimistic for unseen-patient generalization.
- The paper's ten record holdouts overlap: 32 of 48 records are tested, 16 are
  never tested, and some beats are tested more than once. This implementation
  reproduces those splits but emphasizes fold-mean metrics and documents the
  repeat-weighted pooled result.
- The extension is evaluated on one database and is not clinical validation.

## Reference

The complete selected article is included for academic project use. Consult it
for the original methodology, authorship, and citation details.
