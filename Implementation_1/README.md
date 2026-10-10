# BM4152 – Wavelet + PCA + SVM Arrhythmia Beat Classification

Implementation of **Qin et al., "Combining Low-dimensional Wavelet Features and Support Vector
Machine for Arrhythmia Beat Classification", *Scientific Reports* 7, 6067 (2017)**, plus two
extensions: a class-weighted SVM and RR-interval (timing) features.

Boralugoda M.S. (220074B) · Ranaweera R.P.D.D.H. (220511N)
BM4152 Biosignal Processing, University of Moratuwa

The original authors did not release code; everything here was implemented for this project.

## Results at a glance

Mean over 10 folds (%), paper Table 5 vs. ours:

| Scheme | Source | SEN | SPE | ACC |
|---|---|---|---|---|
| Beat-based | Paper | 99.09 | 99.82 | 99.70 |
| | Ours | 99.13 | 99.83 | 99.71 |
| Record-based | Paper | 44.40 | 88.88 | 81.47 |
| | Ours | 43.89 | 88.78 | 81.30 |

Record-based (unseen patients), pooled over folds:

| Model | Accuracy | Balanced acc. | Macro-F1 |
|---|---|---|---|
| Paper (baseline) | 44.12 | 30.38 | 19.55 |
| + scaling | 44.25 | 30.58 | 20.12 |
| + class weights | 38.40 | 28.50 | 18.47 |
| **+ RR features** | **50.55** | 37.35 | 29.42 |
| + RR + weights | 50.31 | 37.46 | 29.74 |

Full numbers are in `results/summary.csv` and `results/*.json`; plots are in `figures/`.

## Files

```
ecg_features.py          segmentation, 8-level bior6.8 wavelet features, RR features
experiments.py           CV folds (beat-based + the paper's record folds), models, metrics
01_download_data.py      Step 1  download MIT-BIH (or download the zip manually, see below)
02_extract_features.py   Step 2  build features/beats.npz (107,049 beats)
03_pca_sweep.py          Step 3  2..50 PCs under both CV schemes (paper Fig. 5)
04_run_experiments.py    Step 4  run models x schemes
05_make_figures.py       Step 5  all figures
results/  figures/  logs/   outputs from our runs
```

Models in `04_run_experiments.py`:

| Name | Features | SVM |
|---|---|---|
| `paper` | 12 PCA scores | unweighted (paper settings: RBF, C = 10, γ = 0.1, one-vs-one) |
| `scaled` | 12 PCA scores, standardised | unweighted |
| `weighted` | 12 PCA scores | class-weighted (inverse frequency) |
| `rr` | 12 PCA scores + 6 RR features, standardised | unweighted |
| `weighted_rr` | 12 PCA scores + 6 RR features, standardised | class-weighted |

PCA and the scaler are always fitted on the training fold only.

## How to run

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

**Data.** Download the ZIP from https://physionet.org/content/mitdb/1.0.0/ and extract it into
`data/` (or run `python 01_download_data.py`). Then:

```bash
python 02_extract_features.py --data-dir data/mit-bih-arrhythmia-database-1.0.0
```

This should print 107,049 beats with exactly the paper's per-class counts.

**Experiments** (Windows PowerShell; limit maths threads per worker first):

```powershell
$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
python 04_run_experiments.py --models paper --n-jobs 6
python 04_run_experiments.py --models weighted,weighted_rr --n-jobs 6
python 04_run_experiments.py --models scaled,rr --n-jobs 6
python 03_pca_sweep.py --n-jobs 6 --subsample 0.3
python 05_make_figures.py
```

One fold takes about 90 s on a Ryzen 7 5700U. Each 10-fold run of one model under both schemes
takes about 10–20 minutes with 6 workers. Use fewer workers (`--n-jobs 4`) if you run out of RAM.

## Implementation notes

- **365 vs. 373 coefficients.** An 8-level bior6.8 DWT of a 252-sample window gives 365 detail
  coefficients; the paper's 373 corresponds to a 259–260-sample window. We kept 252 samples.
- **Lead II.** Records 102 and 104 have no MLII channel; V5 is used for them.
- **PCA memory.** PCA uses `svd_solver="covariance_eigh"` (eigendecomposition of the 365×365
  covariance). A full SVD per fold ran out of memory with parallel folds.
- **Paper metrics.** The paper's "Total" SEN equals plain 6-class accuracy, and its total SPE and
  ACC are fixed functions of it (SPE = 1 − e/5, ACC = 1 − e/3 for error rate e). We therefore also
  report balanced accuracy and macro-F1.

## Reference

Q. Qin, J. Li, L. Zhang, Y. Yue and C. Liu, *Scientific Reports* 7, 6067 (2017).
doi:10.1038/s41598-017-06596-z. Data: MIT-BIH Arrhythmia Database, PhysioNet.
