"""Step 2 - segment beats and extract wavelet + RR features for every record.

    python 02_extract_features.py

Writes features/beats.npz and prints a class-count table you can compare with
the paper (Table 1: A 2546, L 8072, N 75022, P 7025, R 7255, V 7129 = 107,049).
"""
import argparse
import os

import numpy as np

from ecg_features import CLASSES, WAVELET, LEVEL, band_edges, build_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/mitdb")
ap.add_argument("--out", default="features/beats.npz")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
print(f"Extracting: 0.7 s windows, {LEVEL}-level {WAVELET} WMRA (cD1..cD8)")
d = build_dataset(args.data_dir)
np.savez_compressed(args.out, **d)

y = d["y"]
paper = {"A": 2546, "L": 8072, "N": 75022, "P": 7025, "R": 7255, "V": 7129}
print("\nClass   ours    paper")
for c in CLASSES:
    print(f"  {c}   {int((y == c).sum()):6d}  {paper[c]:6d}")
print(f"  total {len(y):6d}  {sum(paper.values()):6d}")
print(f"\nWavelet feature vector length: {d['X_wavelet'].shape[1]} (paper: 373)")
print(f"RR feature matrix: {d['X_rr'].shape}")
print("Detail-band edges (Hz):", {k: (round(a, 2), round(b, 2)) for k, (a, b) in band_edges().items()})
print(f"Saved -> {args.out}")
