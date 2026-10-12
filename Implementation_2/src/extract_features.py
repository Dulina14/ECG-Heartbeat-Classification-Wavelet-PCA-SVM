"""Extract wavelet and RR features from the segmented heartbeat dataset.

This stage follows the audited Implementation_1 convention for the paper
baseline: an eight-level bior6.8 DWT with symmetric extension, retaining
cD1..cD8 and discarding cA8. For a 252-sample beat this produces 365
coefficients.

Two RR matrices are saved:

* reference: reproduces Implementation_1 by treating every MIT-BIH annotation
  as an RR-sequence entry;
* beat_only: computes intervals only between true beat/QRS annotations.

Keeping both definitions lets later experiments reproduce the partner result
and separately test the scientifically cleaner beat-to-beat extension.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb

try:
    from src.inspect_record import BEAT_LENGTH
except ModuleNotFoundError:  # Support: python src/extract_features.py
    from inspect_record import BEAT_LENGTH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

BEATS_PATH = INTERIM_DIR / "beats_252.npy"
METADATA_PATH = INTERIM_DIR / "beat_metadata.csv"
WAVELET_OUTPUT_PATH = (
    INTERIM_DIR / "wavelet_features_bior68_l8_symmetric.npy"
)
REFERENCE_RR_OUTPUT_PATH = INTERIM_DIR / "rr_features_reference.npy"
BEAT_ONLY_RR_OUTPUT_PATH = INTERIM_DIR / "rr_features_beat_only.npy"
SUMMARY_PATH = MANIFEST_DIR / "feature_extraction_summary.json"

SAMPLING_FREQUENCY_HZ = 360.0
WAVELET = "bior6.8"
DECOMPOSITION_LEVEL = 8
BOUNDARY_MODE = "symmetric"
EXPECTED_WAVELET_FEATURES = 365
PAPER_REPORTED_WAVELET_FEATURES = 373
LOCAL_RR_WINDOW = 10

# WFDB symbols that represent actual beat/QRS annotations. Markers such as
# '+', '~', '|', 'x', '[' and ']' are intentionally excluded.
BEAT_ANNOTATION_SYMBOLS = frozenset(
    {
        "N",
        "L",
        "R",
        "a",
        "V",
        "F",
        "J",
        "A",
        "S",
        "E",
        "j",
        "/",
        "Q",
        "e",
        "n",
        "f",
        "r",
    }
)

RR_FEATURE_NAMES = [
    "pre_RR",
    "post_RR",
    "pre_RR/record_mean",
    "post_RR/record_mean",
    "pre_RR/local_mean",
    "pre_RR/post_RR",
]


def wavelet_detail_vector(
    beat: np.ndarray,
    wavelet: str = WAVELET,
    level: int = DECOMPOSITION_LEVEL,
    mode: str = BOUNDARY_MODE,
) -> tuple[np.ndarray, list[int]]:
    """Return concatenated cD1..cD8 and PyWavelets coefficient lengths."""

    beat = np.asarray(beat, dtype=np.float64)

    if beat.shape != (BEAT_LENGTH,):
        raise ValueError(
            f"Expected one ({BEAT_LENGTH},) heartbeat, got {beat.shape}"
        )

    with warnings.catch_warnings():
        # The paper deliberately uses level 8. With a 252-sample input and
        # bior6.8, PyWavelets warns that all levels experience boundary effects.
        warnings.filterwarnings(
            "ignore",
            message="Level value of .* is too high.*",
            category=UserWarning,
        )
        coefficients = pywt.wavedec(
            beat,
            wavelet=wavelet,
            level=level,
            mode=mode,
        )

    lengths = [len(values) for values in coefficients]

    # wavedec returns [cA8, cD8, ..., cD1]. Reverse only the details.
    detail_vector = np.concatenate(coefficients[:0:-1])

    return detail_vector, lengths


def rr_features_from_sequence(
    r_peak_samples: np.ndarray,
    selected_positions: np.ndarray,
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    local_window: int = LOCAL_RR_WINDOW,
) -> np.ndarray:
    """Calculate six RR features for selected positions in an R-peak sequence."""

    samples = np.asarray(r_peak_samples, dtype=np.float64)
    positions = np.asarray(selected_positions, dtype=np.int64)

    if samples.ndim != 1 or len(samples) < 2:
        raise ValueError("RR calculation requires at least two R peaks")

    if np.any(np.diff(samples) <= 0):
        raise ValueError("R-peak samples must be strictly increasing")

    if np.any(positions < 0) or np.any(positions >= len(samples)):
        raise IndexError("Selected RR position is outside the sequence")

    times = samples / sampling_frequency_hz
    intervals = np.diff(times)

    pre = np.empty(len(times), dtype=np.float64)
    post = np.empty(len(times), dtype=np.float64)

    pre[1:] = intervals
    pre[0] = intervals[0]
    post[:-1] = intervals
    post[-1] = intervals[-1]

    record_mean = float(np.mean(intervals))

    # Causal running mean of the current and previous pre-RR intervals.
    local_mean = np.empty(len(times), dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(pre)))

    for index in range(len(times)):
        start = max(0, index - local_window + 1)
        local_mean[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)

    selected_pre = pre[positions]
    selected_post = post[positions]
    selected_local_mean = local_mean[positions]

    features = np.column_stack(
        [
            selected_pre,
            selected_post,
            selected_pre / record_mean,
            selected_post / record_mean,
            selected_pre / selected_local_mean,
            selected_pre / selected_post,
        ]
    )

    if not np.isfinite(features).all():
        raise ValueError("RR feature matrix contains NaN or infinite values")

    return features.astype(np.float32)


def extract_wavelet_matrix(
    beats: np.ndarray,
    output_path: Path,
    progress_every: int,
) -> tuple[list[int], float]:
    """Extract all wavelet vectors into a disk-backed NumPy array."""

    if beats.ndim != 2 or beats.shape[1] != BEAT_LENGTH:
        raise ValueError(
            f"Expected beat matrix (n, {BEAT_LENGTH}), got {beats.shape}"
        )

    first_vector, coefficient_lengths = wavelet_detail_vector(beats[0])
    feature_count = len(first_vector)

    if feature_count != EXPECTED_WAVELET_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_WAVELET_FEATURES} wavelet features, "
            f"received {feature_count}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(beats), feature_count),
    )

    started = time.perf_counter()

    for index in range(len(beats)):
        if index == 0:
            vector = first_vector
            lengths = coefficient_lengths
        else:
            vector, lengths = wavelet_detail_vector(beats[index])

        if lengths != coefficient_lengths:
            raise RuntimeError(
                f"Coefficient lengths changed at beat row {index}: {lengths}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Wavelet vector contains non-finite values at row {index}"
            )

        matrix[index] = vector.astype(np.float32, copy=False)

        completed = index + 1
        if completed % progress_every == 0 or completed == len(beats):
            elapsed = time.perf_counter() - started
            print(
                f"Wavelets: {completed:,}/{len(beats):,} beats "
                f"({elapsed:.1f} seconds)"
            )

    matrix.flush()
    del matrix

    elapsed = time.perf_counter() - started
    return coefficient_lengths, elapsed


def extract_rr_matrices(
    metadata: pd.DataFrame,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return reference and beat-only RR matrices aligned to beat rows."""

    required_columns = {
        "beat_row_index",
        "record_id",
        "annotation_index",
    }
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    row_indices = metadata["beat_row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_indices, np.arange(len(metadata))):
        raise ValueError("beat_row_index must be consecutive and row-aligned")

    reference = np.empty((len(metadata), len(RR_FEATURE_NAMES)), dtype=np.float32)
    beat_only = np.empty_like(reference)

    nonbeat_before = 0
    nonbeat_after = 0

    for record_id, group in metadata.groupby("record_id", sort=False):
        record_id = str(record_id)
        annotation = wfdb.rdann(str(data_dir / record_id), "atr")

        all_samples = np.asarray(annotation.sample, dtype=np.int64)
        all_symbols = np.asarray(annotation.symbol, dtype="U1")
        annotation_indices = group["annotation_index"].to_numpy(dtype=np.int64)
        output_rows = group["beat_row_index"].to_numpy(dtype=np.int64)

        if np.any(annotation_indices >= len(all_samples)):
            raise IndexError(
                f"Metadata annotation index is invalid for record {record_id}"
            )

        # Exact Implementation_1 behavior: every .atr entry is treated as a
        # member of the interval sequence, including non-beat markers.
        reference[output_rows] = rr_features_from_sequence(
            all_samples,
            annotation_indices,
        )

        beat_mask = np.isin(all_symbols, list(BEAT_ANNOTATION_SYMBOLS))
        beat_annotation_indices = np.flatnonzero(beat_mask)
        beat_samples = all_samples[beat_mask]

        beat_position_by_annotation = np.full(
            len(all_samples),
            -1,
            dtype=np.int64,
        )
        beat_position_by_annotation[beat_annotation_indices] = np.arange(
            len(beat_samples),
            dtype=np.int64,
        )
        selected_beat_positions = beat_position_by_annotation[annotation_indices]

        if np.any(selected_beat_positions < 0):
            raise ValueError(
                f"Selected annotation is not a beat in record {record_id}"
            )

        beat_only[output_rows] = rr_features_from_sequence(
            beat_samples,
            selected_beat_positions,
        )

        has_previous = annotation_indices > 0
        previous_indices = np.maximum(annotation_indices - 1, 0)
        previous_is_beat = np.isin(
            all_symbols[previous_indices],
            list(BEAT_ANNOTATION_SYMBOLS),
        )
        nonbeat_before += int(np.sum(~has_previous | ~previous_is_beat))

        has_next = annotation_indices < len(all_symbols) - 1
        next_indices = np.minimum(annotation_indices + 1, len(all_symbols) - 1)
        next_is_beat = np.isin(
            all_symbols[next_indices],
            list(BEAT_ANNOTATION_SYMBOLS),
        )
        nonbeat_after += int(np.sum(~has_next | ~next_is_beat))

        print(
            f"RR features: record {record_id}, "
            f"selected beats={len(group):,}, "
            f"beat annotations={len(beat_samples):,}"
        )

    audit = {
        "selected_beats_with_nonbeat_previous_annotation": nonbeat_before,
        "selected_beats_with_nonbeat_next_annotation": nonbeat_after,
    }
    return reference, beat_only, audit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats", type=Path, default=BEATS_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--wavelet-output",
        type=Path,
        default=WAVELET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--reference-rr-output",
        type=Path,
        default=REFERENCE_RR_OUTPUT_PATH,
    )
    parser.add_argument(
        "--beat-only-rr-output",
        type=Path,
        default=BEAT_ONLY_RR_OUTPUT_PATH,
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--progress-every", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    beats = np.load(args.beats, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(args.metadata, dtype={"record_id": str})

    if len(beats) != len(metadata):
        raise ValueError(
            f"Beat and metadata counts differ: {len(beats)} vs {len(metadata)}"
        )

    coefficient_lengths, wavelet_seconds = extract_wavelet_matrix(
        beats,
        args.wavelet_output,
        args.progress_every,
    )

    reference_rr, beat_only_rr, rr_audit = extract_rr_matrices(
        metadata,
        args.data_dir,
    )

    args.reference_rr_output.parent.mkdir(parents=True, exist_ok=True)
    args.beat_only_rr_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.reference_rr_output, reference_rr, allow_pickle=False)
    np.save(args.beat_only_rr_output, beat_only_rr, allow_pickle=False)

    different_rr_rows = int(
        np.sum(~np.all(np.isclose(reference_rr, beat_only_rr), axis=1))
    )

    summary = {
        "number_of_beats": len(beats),
        "samples_per_beat": int(beats.shape[1]),
        "wavelet": WAVELET,
        "decomposition_level": DECOMPOSITION_LEVEL,
        "boundary_mode": BOUNDARY_MODE,
        "pywt_coefficient_order": [
            "cA8",
            "cD8",
            "cD7",
            "cD6",
            "cD5",
            "cD4",
            "cD3",
            "cD2",
            "cD1",
        ],
        "coefficient_lengths": coefficient_lengths,
        "retained_order": [f"cD{level}" for level in range(1, 9)],
        "approximation_discarded": "cA8",
        "wavelet_features_per_beat": EXPECTED_WAVELET_FEATURES,
        "paper_reported_wavelet_features": PAPER_REPORTED_WAVELET_FEATURES,
        "wavelet_output_shape": [len(beats), EXPECTED_WAVELET_FEATURES],
        "wavelet_output_dtype": "float32",
        "wavelet_elapsed_seconds": wavelet_seconds,
        "rr_feature_names": RR_FEATURE_NAMES,
        "rr_local_window": LOCAL_RR_WINDOW,
        "rr_reference_definition": (
            "Exact Implementation_1 behavior: intervals between adjacent "
            "entries in the complete MIT-BIH annotation sequence."
        ),
        "rr_beat_only_definition": (
            "Intervals between adjacent true beat/QRS annotations only."
        ),
        "rr_rows_different_between_definitions": different_rr_rows,
        **rr_audit,
        "files": {
            "wavelet": str(args.wavelet_output),
            "rr_reference": str(args.reference_rr_output),
            "rr_beat_only": str(args.beat_only_rr_output),
        },
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nFeature extraction completed.")
    print(f"Wavelet matrix: ({len(beats):,}, {EXPECTED_WAVELET_FEATURES})")
    print(f"Coefficient lengths: {coefficient_lengths}")
    print(f"Reference RR matrix: {reference_rr.shape}")
    print(f"Beat-only RR matrix: {beat_only_rr.shape}")
    print(f"Rows changed by corrected RR definition: {different_rr_rows:,}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
