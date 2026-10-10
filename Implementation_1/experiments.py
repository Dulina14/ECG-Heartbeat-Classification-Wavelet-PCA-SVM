"""
experiments.py
--------------
Cross-validation schemes, models and metrics (paper Steps 3-4 + our extensions).

Models
    paper        PCA(12) on raw wavelet coefficients -> one-vs-one RBF SVM,
                 C = 10, gamma = 0.1 (exactly the paper's settings).
    weighted     Same features, but the SVM penalty C is multiplied by a class
                 weight w_c = (n_max / n_c) ** alpha, so mistakes on minority
                 classes (A, R, ...) cost more. alpha = 1 equals sklearn's
                 'balanced'. alpha can be tuned on the training records only.
    weighted_rr  12 PCA scores + 6 RR-interval features, standardised, then the
                 class-weighted SVM. (Morphology + timing.)

  Ablations (to separate the three changes inside weighted_rr):
    scaled       12 PCA scores, standardised, unweighted SVM  (effect of scaling)
    rr           12 PCA scores + RR, standardised, unweighted  (effect of RR)

Cross-validation
    beat    10-fold stratified split of individual beats (paper Table 1).
            Beats of one patient land in both train and test -> optimistic.
    record  10 folds of whole records exactly as listed in paper Table 3.
            Train = every other record, so the test patients are unseen.

PCA (and the scaler) are always fitted on the training fold only.
"""

from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ecg_features import CLASSES

# ---------------------------------------------------------------- folds ----
# Paper Table 3: record-based 10-fold test sets (train = all other records)
PAPER_RECORD_FOLDS = [
    ["100", "101", "109", "118", "217"],
    ["105", "106", "107", "111", "124"],
    ["104", "112", "113", "114", "207"],
    ["102", "116", "117", "212", "214"],
    ["107", "109", "122", "123", "231"],
    ["111", "202", "203", "217", "232"],
    ["102", "118", "207", "209", "210"],
    ["104", "124", "214", "215", "219"],
    ["109", "207", "217", "222"],
    ["102", "111", "212", "233"],
]


def beat_folds(y, n_splits=10, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


def record_folds(records, fold_table=PAPER_RECORD_FOLDS):
    folds = []
    for test_recs in fold_table:
        te = np.where(np.isin(records, test_recs))[0]
        tr = np.where(~np.isin(records, test_recs))[0]
        if len(te) and len(tr):
            folds.append((tr, te))
    return folds


def get_folds(scheme, y, records, n_splits=10, seed=0):
    if scheme == "beat":
        return beat_folds(y, n_splits, seed)
    if scheme == "record":
        return record_folds(records)
    raise ValueError(scheme)


# --------------------------------------------------------------- models ----
def class_weights(y, alpha):
    labels, counts = np.unique(y, return_counts=True)
    return {c: (counts.max() / n) ** alpha for c, n in zip(labels, counts)}


def make_model(name, n_pcs=12, C=10.0, gamma=0.1, alpha=1.0, y_train=None):
    """Returns (svm, class_weight_dict_or_None)."""
    if name in ("paper", "rr", "scaled"):
        cw = None
    elif name in ("weighted", "weighted_rr"):
        cw = class_weights(y_train, alpha)
    else:
        raise ValueError(name)
    svm = SVC(kernel="rbf", C=C, gamma=gamma, decision_function_shape="ovo",
              class_weight=cw, cache_size=500)
    return svm, cw


class FoldModel:
    """PCA on wavelet coefficients (+ optional RR features) -> SVM."""

    def __init__(self, name, n_pcs=12, C=10.0, gamma=0.1, alpha=1.0):
        self.name, self.n_pcs, self.C, self.gamma, self.alpha = name, n_pcs, C, gamma, alpha

    def _features(self, Xw, Xrr, fit):
        if fit:
            # "covariance_eigh" diagonalises the 365x365 covariance matrix instead of
            # running an SVD on the full 96k x 365 data matrix: same principal
            # components, a fraction of the memory (the full SVD ran out of RAM
            # with 8 parallel folds).
            self.pca = PCA(n_components=self.n_pcs, svd_solver="covariance_eigh").fit(Xw)
        Z = self.pca.transform(Xw)
        if self.name in ("weighted_rr", "rr"):
            Z = np.hstack([Z, Xrr])
        if self.name in ("weighted_rr", "rr", "scaled"):
            if fit:
                self.scaler = StandardScaler().fit(Z)
            Z = self.scaler.transform(Z)
        return Z

    def fit(self, Xw, Xrr, y):
        Z = self._features(Xw, Xrr, fit=True)
        self.svm, self.cw = make_model(self.name, self.n_pcs, self.C, self.gamma,
                                       self.alpha, y_train=y)
        self.svm.fit(Z, y)
        return self

    def predict(self, Xw, Xrr):
        return self.svm.predict(self._features(Xw, Xrr, fit=False))

    @property
    def explained_variance(self):
        return float(self.pca.explained_variance_ratio_.sum())


# -------------------------------------------------------------- metrics ----
def per_class_counts(y_true, y_pred, classes=CLASSES):
    """One-vs-rest TP, TN, FP, FN for every class (paper Table 4)."""
    out = {}
    for c in classes:
        t, p = (y_true == c), (y_pred == c)
        out[c] = dict(TP=int((t & p).sum()), TN=int((~t & ~p).sum()),
                      FP=int((~t & p).sum()), FN=int((t & ~p).sum()))
    return out


def _safe(a, b):
    return 100.0 * a / b if b else float("nan")


def metrics_from_counts(counts):
    """Per-class SEN/SPE/ACC/PPV/F1 + the paper's 'Total' (Eqs. 3-5: sums of
    TP, TN, FP, FN over all classes = micro average) + macro averages."""
    res = {}
    for c, k in counts.items():
        sen = _safe(k["TP"], k["TP"] + k["FN"])
        spe = _safe(k["TN"], k["TN"] + k["FP"])
        acc = _safe(k["TP"] + k["TN"], sum(k.values()))
        ppv = _safe(k["TP"], k["TP"] + k["FP"])
        f1 = (2 * sen * ppv / (sen + ppv)) if (sen + ppv) > 0 else 0.0
        res[c] = dict(SEN=sen, SPE=spe, ACC=acc, PPV=ppv, F1=f1,
                      support=k["TP"] + k["FN"])
    T = {m: sum(k[m] for k in counts.values()) for m in ("TP", "TN", "FP", "FN")}
    res["Total(paper Eq.3-5)"] = dict(
        SEN=_safe(T["TP"], T["TP"] + T["FN"]),
        SPE=_safe(T["TN"], T["TN"] + T["FP"]),
        ACC=_safe(T["TP"] + T["TN"], sum(T.values())))
    present = [c for c in counts if res[c]["support"] > 0]
    res["Macro"] = {m: float(np.nanmean([res[c][m] for c in present]))
                    for m in ("SEN", "SPE", "ACC", "PPV", "F1")}
    return res


def summary_metrics(y_true, y_pred):
    counts = per_class_counts(y_true, y_pred)
    m = metrics_from_counts(counts)
    overall_acc = 100.0 * float((y_true == y_pred).mean())
    present = [c for c in CLASSES if (y_true == c).any()]
    bal_acc = float(np.mean([m[c]["SEN"] for c in present]))
    return dict(per_class=m, overall_accuracy=overall_acc,
                balanced_accuracy=bal_acc,
                macro_F1=m["Macro"]["F1"],
                confusion=confusion_matrix(y_true, y_pred, labels=CLASSES).tolist())


# ---------------------------------------------------- inner weight tuning ----
def tune_alpha(Xw, Xrr, y, records, name, n_pcs, C, gamma,
               alphas=(0.25, 0.5, 0.75, 1.0), inner_splits=3, max_train=30000, seed=0):
    """Choose alpha for class weights using ONLY the training data of a fold.

    Inner split is grouped by record (patient-independent) so the choice
    rewards generalisation to new patients. Criterion: balanced accuracy.
    Training sets are subsampled to `max_train` beats to keep this affordable.
    """
    rng = np.random.default_rng(seed)
    n_groups = len(np.unique(records))
    splits = (GroupKFold(n_splits=min(inner_splits, n_groups)).split(Xw, y, records)
              if n_groups >= 2 else beat_folds(y, inner_splits, seed))
    splits = list(splits)
    scores = {}
    for a in alphas:
        s = []
        for tr, va in splits:
            if len(tr) > max_train:
                tr = rng.choice(tr, max_train, replace=False)
            m = FoldModel(name, n_pcs, C, gamma, a).fit(Xw[tr], Xrr[tr], y[tr])
            yp = m.predict(Xw[va], Xrr[va])
            present = np.unique(y[va])
            s.append(np.mean([(yp[y[va] == c] == c).mean() for c in present]))
        scores[a] = float(np.mean(s))
    best = max(sorted(scores, reverse=True), key=scores.get)  # ties -> larger alpha
    return best, scores


# ------------------------------------------------------------ main loop ----
def _run_fold(k, tr, te, Xw, Xrr, y, records, name, n_pcs, C, gamma, alpha, tune):
    t0 = time.time()
    tune_info = None
    if tune and name != "paper":
        alpha, tune_info = tune_alpha(Xw[tr], Xrr[tr], y[tr], records[tr],
                                      name, n_pcs, C, gamma)
    m = FoldModel(name, n_pcs, C, gamma, alpha).fit(Xw[tr], Xrr[tr], y[tr])
    yp = m.predict(Xw[te], Xrr[te])
    met = summary_metrics(y[te], yp)
    return dict(fold=k + 1, n_train=int(len(tr)), n_test=int(len(te)),
                test_records=sorted(set(records[te].tolist())),
                alpha=alpha, tune_scores=tune_info,
                explained_variance=m.explained_variance,
                n_support_vectors=int(m.svm.n_support_.sum()),
                seconds=round(time.time() - t0, 1),
                y_true=y[te].tolist(), y_pred=yp.tolist(), metrics=met)


def run_cv(data, scheme, name, n_pcs=12, C=10.0, gamma=0.1, alpha=1.0,
           tune=False, n_jobs=1, max_folds=None, seed=0, verbose=True):
    Xw, Xrr, y, rec = data["X_wavelet"], data["X_rr"], data["y"], data["record"]
    folds = get_folds(scheme, y, rec, seed=seed)
    if max_folds:
        folds = folds[:max_folds]
    if verbose:
        print(f"[{name} | {scheme}-based | {n_pcs} PCs] {len(folds)} folds, n_jobs={n_jobs}")
    fold_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_run_fold)(k, tr, te, Xw, Xrr, y, rec, name, n_pcs, C, gamma, alpha, tune)
        for k, (tr, te) in enumerate(folds))

    # Pooled metrics over all test predictions + mean/std over folds
    yt = np.concatenate([np.array(f["y_true"]) for f in fold_results])
    yp = np.concatenate([np.array(f["y_pred"]) for f in fold_results])
    pooled = summary_metrics(yt, yp)
    fold_table = {}
    for key in list(CLASSES) + ["Total(paper Eq.3-5)", "Macro"]:
        for met in ("SEN", "SPE", "ACC"):
            vals = [f["metrics"]["per_class"][key][met] for f in fold_results]
            fold_table[f"{key}_{met}"] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    for f in fold_results:            # keep json small
        f.pop("y_true"), f.pop("y_pred")
    return dict(model=name, scheme=scheme, n_pcs=n_pcs, C=C, gamma=gamma,
                folds=fold_results, pooled=pooled, fold_mean_std=fold_table)
