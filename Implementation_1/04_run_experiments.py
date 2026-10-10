"""Step 4 - run the main experiments (replication + extensions).

    python 04_run_experiments.py --n-jobs 4                       # everything
    python 04_run_experiments.py --models paper --schemes beat    # one run
    python 04_run_experiments.py --models weighted,weighted_rr --tune --n-jobs 4

Models: paper | weighted | weighted_rr | scaled | rr      Schemes: beat | record
Each run is saved to results/<model>_<scheme>.json and a summary row is
appended to results/summary.csv.
"""
import argparse
import csv
import json
import os

import numpy as np

from ecg_features import CLASSES
from experiments import run_cv

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="features/beats.npz")
ap.add_argument("--models", default="paper,weighted,weighted_rr")
ap.add_argument("--schemes", default="beat,record")
ap.add_argument("--n-pcs", type=int, default=12)
ap.add_argument("--C", type=float, default=10.0)
ap.add_argument("--gamma", type=float, default=0.1)
ap.add_argument("--alpha", type=float, default=1.0,
                help="class-weight exponent for weighted models (1 = 'balanced')")
ap.add_argument("--tune", action="store_true",
                help="tune alpha per fold on training records only (slower)")
ap.add_argument("--n-jobs", type=int, default=1)
ap.add_argument("--max-folds", type=int, default=None, help="for quick tests")
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

d = dict(np.load(args.features))
os.makedirs(args.outdir, exist_ok=True)
summary_path = os.path.join(args.outdir, "summary.csv")
new_file = not os.path.exists(summary_path)

for model in args.models.split(","):
    for scheme in args.schemes.split(","):
        r = run_cv(d, scheme, model, n_pcs=args.n_pcs, C=args.C, gamma=args.gamma,
                   alpha=args.alpha, tune=args.tune, n_jobs=args.n_jobs,
                   max_folds=args.max_folds)
        tag = f"{model}{'_tuned' if args.tune and model != 'paper' else ''}_{scheme}"
        with open(os.path.join(args.outdir, tag + ".json"), "w") as f:
            json.dump(r, f, indent=1)

        P = r["pooled"]["per_class"]
        T = P["Total(paper Eq.3-5)"]
        print(f"\n=== {tag} (pooled over folds) ===")
        print(f"{'class':>6} {'SEN':>7} {'SPE':>7} {'ACC':>7} {'PPV':>7} {'F1':>7} {'n':>6}")
        for c in CLASSES:
            m = P[c]
            print(f"{c:>6} {m['SEN']:7.2f} {m['SPE']:7.2f} {m['ACC']:7.2f} "
                  f"{m['PPV']:7.2f} {m['F1']:7.2f} {m['support']:6d}")
        print(f"{'Total':>6} {T['SEN']:7.2f} {T['SPE']:7.2f} {T['ACC']:7.2f}   (paper Eqs. 3-5)")
        print(f"overall acc {r['pooled']['overall_accuracy']:.2f} | balanced acc "
              f"{r['pooled']['balanced_accuracy']:.2f} | macro F1 {r['pooled']['macro_F1']:.2f}")
        print("confusion (rows=true, cols=pred, order A L N P R V):")
        print(np.array(r["pooled"]["confusion"]))

        with open(summary_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["run", "model", "scheme", "n_pcs", "SEN_total", "SPE_total",
                            "ACC_total", "overall_acc", "balanced_acc", "macro_F1"]
                           + [f"SEN_{c}" for c in CLASSES] + [f"PPV_{c}" for c in CLASSES])
                new_file = False
            w.writerow([tag, model, scheme, args.n_pcs,
                        *(round(T[k], 2) for k in ("SEN", "SPE", "ACC")),
                        round(r["pooled"]["overall_accuracy"], 2),
                        round(r["pooled"]["balanced_accuracy"], 2),
                        round(r["pooled"]["macro_F1"], 2)]
                       + [round(P[c]["SEN"], 2) for c in CLASSES]
                       + [round(P[c]["PPV"], 2) for c in CLASSES])
print(f"\nSummary -> {summary_path}")
