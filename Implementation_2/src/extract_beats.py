"""Extract all selected MIT-BIH beats into a 252-sample matrix."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from inspect_dataset import read_official_records, verify_record_files
from inspect_record import (
    BEAT_LENGTH,
    LABEL_MAP,
    SAMPLES_AFTER_R,
    SAMPLES_BEFORE_R,
    choose_lead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

EXPECTED_FS = 360
PAPER_FINAL_BEAT_COUNT = 107_049
CLASS_ORDER = ["N", "A", "V", "L", "R", "P"]


def extract_beat_window(
    signal: np.ndarray,
    r_peak_sample: int,
) -> tuple[np.ndarray, int, int] | None:
    """Extract a complete 252-sample beat around one R peak."""

    start = int(r_peak_sample) - SAMPLES_BEFORE_R
    stop = int(r_peak_sample) + SAMPLES_AFTER_R

    if start < 0 or stop > len(signal):
        return None

    beat = np.asarray(signal[start:stop], dtype=np.float32)

    if beat.shape != (BEAT_LENGTH,):
        raise AssertionError(
            f"Expected shape ({BEAT_LENGTH},), received {beat.shape}"
        )

    if not np.isfinite(beat).all():
        raise ValueError("Beat contains NaN or infinite samples")

    return beat, start, stop


def main() -> None:
    records = read_official_records(DATA_DIR)

    beats: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    before_counts: Counter[str] = Counter()
    discarded_counts: Counter[str] = Counter()

    for record_number, record_name in enumerate(records, start=1):
        verify_record_files(DATA_DIR, record_name)

        record_path = DATA_DIR / record_name
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), "atr")

        if int(record.fs) != EXPECTED_FS:
            raise ValueError(
                f"Record {record_name} uses {record.fs} Hz, "
                f"expected {EXPECTED_FS} Hz"
            )

        signal_names = list(record.sig_name)
        lead_index = choose_lead(signal_names)
        lead_name = signal_names[lead_index]

        signal = np.asarray(
            record.p_signal[:, lead_index],
            dtype=np.float32,
        )

        retained_in_record = 0
        discarded_in_record = 0

        annotation_pairs = zip(
            annotation.sample,
            annotation.symbol,
            strict=True,
        )

        for annotation_index, (r_peak, raw_symbol) in enumerate(
            annotation_pairs
        ):
            if raw_symbol not in LABEL_MAP:
                continue

            mapped_label = LABEL_MAP[raw_symbol]
            before_counts[mapped_label] += 1

            extracted = extract_beat_window(signal, int(r_peak))

            if extracted is None:
                discarded_counts[mapped_label] += 1
                discarded_in_record += 1
                continue

            beat, start, stop = extracted
            row_index = len(beats)

            beats.append(beat)

            metadata_rows.append(
                {
                    "beat_id": f"{record_name}_{int(r_peak):07d}",
                    "beat_row_index": row_index,
                    "record_id": record_name,
                    "annotation_index": annotation_index,
                    "r_peak_sample": int(r_peak),
                    "window_start_sample": start,
                    "window_stop_exclusive": stop,
                    "selected_lead": lead_name,
                    "selected_lead_index": lead_index,
                    "raw_symbol": raw_symbol,
                    "label": mapped_label,
                }
            )

            retained_in_record += 1

        print(
            f"[{record_number:02d}/{len(records)}] "
            f"record {record_name}: "
            f"lead={lead_name}, "
            f"retained={retained_in_record}, "
            f"boundary-discarded={discarded_in_record}"
        )

    if not beats:
        raise RuntimeError("No beats were extracted")

    beat_matrix = np.stack(beats).astype(np.float32, copy=False)
    metadata = pd.DataFrame(metadata_rows)

    labels = metadata["label"].to_numpy(dtype="U1")
    record_ids = metadata["record_id"].to_numpy(dtype="U3")

    if beat_matrix.shape[0] != len(metadata):
        raise AssertionError(
            "Beat matrix and metadata row counts do not match"
        )

    if beat_matrix.shape[1] != BEAT_LENGTH:
        raise AssertionError(
            f"Expected {BEAT_LENGTH} samples per beat"
        )

    if metadata["beat_id"].duplicated().any():
        raise AssertionError("Duplicate beat IDs were generated")

    if metadata["beat_row_index"].duplicated().any():
        raise AssertionError("Duplicate row indices were generated")

    retained_counts = Counter(labels.tolist())

    comparison_rows = []

    for label in CLASS_ORDER:
        before = int(before_counts[label])
        discarded = int(discarded_counts[label])
        retained = int(retained_counts[label])

        if before != discarded + retained:
            raise AssertionError(
                f"Count reconciliation failed for class {label}"
            )

        comparison_rows.append(
            {
                "class": label,
                "before_window_filtering": before,
                "boundary_beats_discarded": discarded,
                "after_window_filtering": retained,
                "percentage_after_filtering": (
                    retained / len(metadata) * 100.0
                ),
            }
        )

    class_counts = pd.DataFrame(comparison_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    beats_path = OUTPUT_DIR / "beats_252.npy"
    labels_path = OUTPUT_DIR / "labels.npy"
    record_ids_path = OUTPUT_DIR / "record_ids.npy"
    metadata_path = OUTPUT_DIR / "beat_metadata.csv"

    counts_path = (
        MANIFEST_DIR / "class_counts_after_window_filtering.csv"
    )
    summary_path = (
        MANIFEST_DIR / "beat_extraction_summary.json"
    )

    np.save(beats_path, beat_matrix, allow_pickle=False)
    np.save(labels_path, labels, allow_pickle=False)
    np.save(record_ids_path, record_ids, allow_pickle=False)

    metadata.to_csv(metadata_path, index=False)
    class_counts.to_csv(counts_path, index=False)

    final_count = len(metadata)
    total_before = int(sum(before_counts.values()))
    total_discarded = int(sum(discarded_counts.values()))
    matches_paper = final_count == PAPER_FINAL_BEAT_COUNT

    summary = {
        "records_processed": len(records),
        "sampling_frequency_hz": EXPECTED_FS,
        "samples_before_r_peak": SAMPLES_BEFORE_R,
        "samples_after_r_peak": SAMPLES_AFTER_R,
        "beat_length": BEAT_LENGTH,
        "matrix_shape": list(beat_matrix.shape),
        "matrix_dtype": str(beat_matrix.dtype),
        "selected_annotations_before_filtering": total_before,
        "boundary_beats_discarded": total_discarded,
        "final_beat_count": final_count,
        "paper_reported_final_beat_count": PAPER_FINAL_BEAT_COUNT,
        "matches_paper_reported_total": matches_paper,
        "class_counts_after_filtering": {
            label: int(retained_counts[label])
            for label in CLASS_ORDER
        },
        "discarded_by_class": {
            label: int(discarded_counts[label])
            for label in CLASS_ORDER
        },
        "files": {
            "beats": str(beats_path),
            "labels": str(labels_path),
            "record_ids": str(record_ids_path),
            "metadata": str(metadata_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nClass counts after window filtering:")
    print(class_counts.to_string(index=False))

    print(f"\nBeat matrix shape: {beat_matrix.shape}")
    print(f"Selected annotations before filtering: {total_before:,}")
    print(f"Boundary beats discarded: {total_discarded}")
    print(f"Final heartbeat count: {final_count:,}")
    print(f"Paper-reported count: {PAPER_FINAL_BEAT_COUNT:,}")
    print(f"Exact match: {matches_paper}")

    print(f"\nSaved beat matrix: {beats_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved record IDs: {record_ids_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved class counts: {counts_path}")
    print(f"Saved summary: {summary_path}")

    if not matches_paper:
        raise RuntimeError(
            "The final count does not match the paper. "
            "Do not continue to wavelet extraction until investigated."
        )


if __name__ == "__main__":
    main()