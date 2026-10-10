"""Step 3 - choose the number of principal components (replicates paper Fig. 5).

    python 03_pca_sweep.py --n-jobs 4                       # full (slow)
    python 03_pca_sweep.py --n-jobs 4 --subsample 0.3       # faster, ~same curve

Sweeps 2, 4, ..., 50 PCs with the paper model under both CV schemes and saves
results/pca_sweep.json. Also prints the cumulative explained variance of PCA.
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA

from experiments import run_cv

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="features/beats.npz")
ap.add_argument("--pcs", default="2:50:2", help="start:stop:step (inclusive stop)")
ap.add_argument("--schemes", default="beat,record")
ap.add_argument("--subsample", type=float, default=1.0,
                help="fraction of beats to use (stratified by class+record) for speed")
ap.add_argument("--n-jobs", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/pca_sweep.json")
args = ap.parse_args()

d = dict(np.load(args.features))
if args.subsample < 1.0:
    rng = np.random.default_rng(args.seed)
    keep = rng.random(len(d["y"])) < args.subsample
    d = {k: (v[keep] if len(v) == len(keep) else v) for k, v in d.items()}
    print(f"Subsampled to {len(d['y'])} beats")

ev = PCA(svd_solver="covariance_eigh").fit(d["X_wavelet"]).explained_variance_ratio_.cumsum()
for k in (5, 10, 12, 20, 50):
    print(f"cumulative explained variance with {k:2d} PCs: {100 * ev[k - 1]:.2f}%")

a, b, s = map(int, args.pcs.split(":"))
grid = list(range(a, b + 1, s))
out = {"explained_variance_cumsum": ev[:100].tolist(), "grid": grid, "results": {}}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
for scheme in args.schemes.split(","):
    out["results"][scheme] = []
    for k in grid:
        r = run_cv(d, scheme, "paper", n_pcs=k, n_jobs=args.n_jobs, verbose=False)
        t = r["pooled"]["per_class"]["Total(paper Eq.3-5)"]
        row = dict(n_pcs=k, **t, balanced_accuracy=r["pooled"]["balanced_accuracy"],
                   macro_F1=r["pooled"]["macro_F1"])
        out["results"][scheme].append(row)
        print(f"{scheme:6s} PCs={k:2d}  SEN={t['SEN']:.2f}  SPE={t['SPE']:.2f}  "
              f"ACC={t['ACC']:.2f}  balAcc={row['balanced_accuracy']:.2f}")
        with open(args.out, "w") as f:      # save progress as we go
            json.dump(out, f, indent=1)
print(f"Saved -> {args.out}")
