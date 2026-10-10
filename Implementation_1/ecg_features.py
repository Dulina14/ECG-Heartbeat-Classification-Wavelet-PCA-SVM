"""
ecg_features.py
---------------
Data loading, beat segmentation and feature extraction for the replication of

    Qin Q. et al., "Combining Low-dimensional Wavelet Features and Support Vector
    Machine for Arrhythmia Beat Classification", Sci. Rep. 7, 6067 (2017).

Pipeline (paper Steps 1-2):
    1. Read the MLII (Lead II) channel of each MIT-BIH record.
    2. Keep beats annotated A, L, N, P, R, V (paper uses symbol '/' for paced = P).
    3. Cut a 0.7 s window around every R-peak: 0.25 s before (90 samples) and
       0.45 s after (162 samples) -> 252 samples at 360 Hz.
    4. 8-level wavelet multi-resolution analysis with bior6.8; keep cD1..cD8,
       discard cA8 (0-0.70 Hz, baseline drift). Concatenate cD1 -> cD8.

Extension features (our work):
    RR-interval features: pre-RR, post-RR, and both normalised by the record's
    mean RR and by a local (running) mean RR. These carry the *timing*
    information that a single 0.7 s morphology window cannot see (e.g. the
    prematurity of an A beat, whose shape is almost identical to N).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pywt
import wfdb

FS = 360                      # MIT-BIH sampling rate (Hz)
PRE = int(round(0.25 * FS))   # 90 samples before R-peak
POST = int(round(0.45 * FS))  # 162 samples after R-peak  -> 252 total
WAVELET = "bior6.8"
LEVEL = 8

# MIT-BIH annotation symbol -> paper class label
SYMBOL_TO_CLASS = {"N": "N", "L": "L", "R": "R", "A": "A", "V": "V", "/": "P"}
CLASSES = ["A", "L", "N", "P", "R", "V"]     # order used in the paper's tables

# The 48 MIT-BIH arrhythmia records
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def download_mitdb(data_dir: str = "data/mitdb") -> None:
    """Download the full MIT-BIH Arrhythmia Database (~100 MB) from PhysioNet."""
    os.makedirs(data_dir, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=data_dir)


def available_records(data_dir: str) -> list[str]:
    return [r for r in MITDB_RECORDS
            if os.path.exists(os.path.join(data_dir, r + ".dat"))
            and os.path.exists(os.path.join(data_dir, r + ".atr"))]


def load_record(record: str, data_dir: str):
    """Return (signal_mV, r_peak_samples, symbols, lead_name) for one record.

    The paper says 'Lead II'. MIT-BIH stores modified lead II as 'MLII' in 46 of
    the 48 records; records 102 and 104 have no MLII (the patients wear
    pacemaker dressings), so we fall back to channel 0 (V5) for those and
    record that fact so it can be reported.
    """
    rec = wfdb.rdrecord(os.path.join(data_dir, record))
    ann = wfdb.rdann(os.path.join(data_dir, record), "atr")
    names = [n.upper() for n in rec.sig_name]
    ch = names.index("MLII") if "MLII" in names else 0
    sig = rec.p_signal[:, ch].astype(np.float64)
    return sig, np.asarray(ann.sample), list(ann.symbol), rec.sig_name[ch]


# --------------------------------------------------------------------------
# Step 1 - beat segmentation
# --------------------------------------------------------------------------
def segment_beats(sig, r_peaks, symbols):
    """Cut 252-sample windows around every annotated A/L/N/P/R/V R-peak.

    Returns windows (n,252), labels (n,), index of each kept beat in the full
    annotation list (needed for RR features), and the R-peak samples.
    Beats whose window would run past the start or end of the record are
    dropped (this is why the count is slightly below the paper's 107,049).
    """
    windows, labels, ann_idx = [], [], []
    n = len(sig)
    for i, (r, s) in enumerate(zip(r_peaks, symbols)):
        if s not in SYMBOL_TO_CLASS:
            continue
        if r - PRE < 0 or r + POST > n:
            continue
        windows.append(sig[r - PRE:r + POST])
        labels.append(SYMBOL_TO_CLASS[s])
        ann_idx.append(i)
    return (np.asarray(windows), np.asarray(labels),
            np.asarray(ann_idx, dtype=int))


# --------------------------------------------------------------------------
# Step 2 - wavelet multi-resolution analysis
# --------------------------------------------------------------------------
def wavelet_features(windows: np.ndarray, wavelet: str = WAVELET,
                     level: int = LEVEL, mode: str = "symmetric") -> np.ndarray:
    """8-level DWT of each beat; returns concatenated [cD1, cD2, ..., cD8].

    pywt.wavedec returns [cA8, cD8, cD7, ..., cD1]; we drop cA8 and reverse
    the detail list so the vector starts at cD1 (highest band, 90-180 Hz) and
    ends at cD8 (0.70-1.41 Hz), exactly as in the paper's Fig. 4.

    With a 252-sample window and bior6.8 (filter length 18) this gives 365
    coefficients. The paper reports 373, which corresponds to a 259/260-sample
    window - see README 'Known discrepancies'.
    """
    with warnings.catch_warnings():
        # pywt warns that level 8 exceeds the 'useful' level for 252 samples
        # (boundary effects dominate the deepest levels). The paper does it
        # anyway, so we follow the paper.
        warnings.simplefilter("ignore", UserWarning)
        feats = []
        for w in windows:
            coeffs = pywt.wavedec(w, wavelet, level=level, mode=mode)
            details = coeffs[1:][::-1]           # cD1 ... cD8
            feats.append(np.concatenate(details))
    return np.asarray(feats)


def band_edges(fs: float = FS, level: int = LEVEL):
    """Nominal frequency band of each detail level (for the report)."""
    return {f"cD{j}": (fs / 2 ** (j + 1), fs / 2 ** j) for j in range(1, level + 1)}


# --------------------------------------------------------------------------
# Extension - RR-interval (temporal) features
# --------------------------------------------------------------------------
def rr_features(r_peaks: np.ndarray, kept_idx: np.ndarray,
                fs: float = FS, local_window: int = 10) -> np.ndarray:
    """Per kept beat: [pre_RR, post_RR, pre/record_mean, post/record_mean,
    pre/local_mean, pre/post].

    RR intervals are computed over *all* annotated beats (the R-peak
    sequence), so a kept beat's neighbours may be other beat types.
    Local mean = mean of the previous `local_window` RR intervals (causal).
    """
    r = r_peaks.astype(float) / fs
    rr = np.diff(r)                                   # rr[k] = r[k+1]-r[k]
    n = len(r)
    pre = np.empty(n)
    post = np.empty(n)
    pre[1:] = rr
    pre[0] = rr[0]
    post[:-1] = rr
    post[-1] = rr[-1]

    rec_mean = np.mean(rr)
    local = np.empty(n)
    csum = np.concatenate([[0.0], np.cumsum(pre)])
    for k in range(n):
        a = max(0, k - local_window + 1)
        local[k] = (csum[k + 1] - csum[a]) / (k + 1 - a)

    pre, post, local = pre[kept_idx], post[kept_idx], local[kept_idx]
    return np.column_stack([
        pre, post,
        pre / rec_mean, post / rec_mean,
        pre / local,
        pre / post,
    ])


RR_FEATURE_NAMES = ["pre_RR", "post_RR", "pre_RR/rec_mean", "post_RR/rec_mean",
                    "pre_RR/local_mean", "pre_RR/post_RR"]


# --------------------------------------------------------------------------
# Whole-database feature table
# --------------------------------------------------------------------------
def build_dataset(data_dir: str, records=None, verbose: bool = True):
    """Extract everything for all records. Returns a dict of numpy arrays."""
    records = records or available_records(data_dir)
    if not records:
        raise FileNotFoundError(
            f"No MIT-BIH records found in '{data_dir}'. Run 01_download_data.py first.")

    Xw, Xrr, Y, REC, WIN, leads = [], [], [], [], [], {}
    for rec in records:
        sig, rp, sym, lead = load_record(rec, data_dir)
        win, lab, idx = segment_beats(sig, rp, sym)
        leads[rec] = lead
        if len(lab) == 0:
            continue
        Xw.append(wavelet_features(win))
        Xrr.append(rr_features(rp, idx))
        Y.append(lab)
        REC.append(np.full(len(lab), rec))
        WIN.append(win.astype(np.float32))
        if verbose:
            counts = {c: int((lab == c).sum()) for c in CLASSES if (lab == c).any()}
            print(f"  record {rec} [{lead:>4}]: {len(lab):5d} beats  {counts}")

    return {
        "X_wavelet": np.vstack(Xw),
        "X_rr": np.vstack(Xrr),
        "y": np.concatenate(Y),
        "record": np.concatenate(REC),
        "windows": np.vstack(WIN),
        "leads": np.array([f"{k}:{v}" for k, v in leads.items()]),
    }
