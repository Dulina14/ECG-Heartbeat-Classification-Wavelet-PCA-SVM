"""Step 5 - make all report/presentation figures from the saved results.

    python 05_make_figures.py

Reads features/beats.npz and results/*.json, writes PNGs to figures/.
Missing results are skipped, so you can run this at any stage.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt

from ecg_features import CLASSES, FS, LEVEL, PRE, WAVELET

FEAT, RES, FIG = "features/beats.npz", "results", "figures"
os.makedirs(FIG, exist_ok=True)
COL = {"A": "#E69F00", "L": "#56B4E9", "N": "#444444", "P": "#009E73",
       "R": "#CC79A7", "V": "#D55E00"}
MODEL_COL = {"paper": "#888888", "weighted": "#0072B2", "scaled": "#56B4E9",
             "rr": "#E69F00", "weighted_rr": "#D55E00"}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def load(tag):
    p = os.path.join(RES, tag + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


# ---- 1. Mean beat per class (dataset slide) --------------------------------
if os.path.exists(FEAT):
    d = np.load(FEAT)
    W, y = d["windows"], d["y"]
    t = (np.arange(W.shape[1]) - PRE) / FS
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), sharex=True)
    for ax, c in zip(axes.ravel(), CLASSES):
        sel = W[y == c]
        if len(sel) == 0:
            ax.set_title(f"{c} (none)"); continue
        mu, sd = sel.mean(0), sel.std(0)
        ax.fill_between(t, mu - sd, mu + sd, color=COL[c], alpha=0.2, lw=0)
        ax.plot(t, mu, color=COL[c], lw=1.5)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{c}  (n = {len(sel):,})")
    for ax in axes[1]: ax.set_xlabel("time from R-peak (s)")
    for ax in axes[:, 0]: ax.set_ylabel("mV")
    fig.suptitle("Mean ± 1 SD beat per class (0.7 s windows, MLII)")
    save(fig, "fig1_mean_beats.png")

    # ---- 2. Wavelet decomposition of one beat ------------------------------
    i = int(np.where(y == "N")[0][0])
    coeffs = pywt.wavedec(W[i], WAVELET, level=LEVEL)
    fig, axes = plt.subplots(LEVEL + 2, 1, figsize=(6, 9))
    axes[0].plot(t, W[i], "k", lw=1); axes[0].set_ylabel("beat")
    names = ["cA8"] + [f"cD{j}" for j in range(LEVEL, 0, -1)]
    for ax, cf, nm in zip(axes[1:], coeffs, names):
        ax.stem(cf, markerfmt=" ", basefmt=" ", linefmt="C0-")
        ax.set_ylabel(nm, rotation=0, ha="right", va="center")
        ax.set_xticks([])
    axes[1].set_ylabel("cA8\n(discarded)", rotation=0, ha="right", va="center")
    fig.suptitle(f"{LEVEL}-level {WAVELET} decomposition of one N beat")
    save(fig, "fig2_wavelet_decomposition.png")

# ---- 3. PCA sweep (paper Fig. 5) -------------------------------------------
sw = os.path.join(RES, "pca_sweep.json")
if os.path.exists(sw):
    S = json.load(open(sw))
    schemes = list(S["results"])
    fig, axes = plt.subplots(1, len(schemes) + 1, figsize=(4 * (len(schemes) + 1), 3.2))
    for ax, sc in zip(axes, schemes):
        rows = S["results"][sc]
        k = [r["n_pcs"] for r in rows]
        for m, mk in (("SEN", "<"), ("SPE", "*"), ("ACC", "D"), ("balanced_accuracy", "o")):
            ax.plot(k, [r[m] for r in rows], marker=mk, ms=3, label=m.replace("_", " "))
        ax.axvline(12, color="r", ls="--", lw=0.8)
        ax.set_title(f"{sc}-based"); ax.set_xlabel("number of PCs"); ax.set_ylabel("%")
        ax.legend(fontsize=7)
    ev = 100 * np.array(S["explained_variance_cumsum"][:50])
    axes[-1].plot(np.arange(1, len(ev) + 1), ev, "k.-", ms=3)
    axes[-1].axvline(12, color="r", ls="--", lw=0.8)
    axes[-1].set_title("cumulative explained variance"); axes[-1].set_xlabel("PCs")
    save(fig, "fig3_pca_sweep.png")

# ---- 4. Per-fold performance (paper Fig. 6) --------------------------------
runs = {s: load(f"paper_{s}") for s in ("beat", "record")}
if any(runs.values()):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (s, r) in zip(axes, runs.items()):
        if not r: continue
        f = [x["fold"] for x in r["folds"]]
        for m, mk in (("SEN", "o"), ("SPE", "<"), ("ACC", ">")):
            ax.plot(f, [x["metrics"]["per_class"]["Total(paper Eq.3-5)"][m] for x in r["folds"]],
                    marker=mk, label=m)
        ax.set_title(f"({'a' if s == 'beat' else 'b'}) {s}-based"); ax.set_xlabel("fold")
        ax.set_ylabel("%"); ax.legend(fontsize=7)
    save(fig, "fig4_per_fold.png")

# ---- 5. Confusion matrices --------------------------------------------------
for p in sorted(glob.glob(os.path.join(RES, "*_*.json"))):
    tag = os.path.basename(p)[:-5]
    if tag == "pca_sweep": continue
    r = json.load(open(p))
    cm = np.array(r["pooled"]["confusion"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(6), CLASSES); ax.set_yticks(range(6), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{tag}\n(colour = row-normalised)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, f"cm_{tag}.png")

# ---- 6. Model comparison per class -----------------------------------------
for scheme in ("record", "beat"):
    avail = [(m, load(f"{m}_{scheme}") or load(f"{m}_tuned_{scheme}"))
             for m in ("paper", "weighted", "scaled", "rr", "weighted_rr")]
    avail = [(m, r) for m, r in avail if r]
    if len(avail) < 2: continue
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    w = 0.8 / len(avail)
    for metric, ax in zip(("SEN", "PPV"), axes):
        for i, (m, r) in enumerate(avail):
            v = [r["pooled"]["per_class"][c][metric] for c in CLASSES]
            ax.bar(np.arange(6) + (i - (len(avail) - 1) / 2) * w, v, w,
                   label=m, color=MODEL_COL[m])
        ax.set_xticks(range(6), CLASSES); ax.set_ylim(0, 105)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"{scheme}-based CV: per-class {'sensitivity' if metric == 'SEN' else 'precision'}")
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.95)
    save(fig, f"fig5_model_comparison_{scheme}.png")

print("done")
