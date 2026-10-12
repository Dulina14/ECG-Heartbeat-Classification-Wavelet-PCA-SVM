"""Inspect one MIT-BIH record before building the full processing pipeline.

This script verifies that WFDB can read the downloaded signal and annotation
files, selects an ECG lead, summarizes annotations, and plots one correctly
segmented 252-sample heartbeat.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

FS_EXPECTED = 360
SAMPLES_BEFORE_R = 90
SAMPLES_AFTER_R = 162
BEAT_LENGTH = SAMPLES_BEFORE_R + SAMPLES_AFTER_R

LABEL_MAP = {
    "N": "N",
    "A": "A",
    "V": "V",
    "L": "L",
    "R": "R",
    "/": "P",
}


def choose_lead(signal_names: list[str]) -> int:
    """Return the preferred MLII/II lead index, or fall back to channel 0."""
    normalized = [name.strip().upper() for name in signal_names]
    for preferred in ("MLII", "II"):
        if preferred in normalized:
            return normalized.index(preferred)
    return 0


def find_example_beat(
    samples: np.ndarray,
    symbols: list[str],
    signal_length: int,
) -> tuple[int, str, str]:
    """Find the first selected-class beat with a complete 252-sample window."""
    for sample, symbol in zip(samples, symbols, strict=True):
        start = int(sample) - SAMPLES_BEFORE_R
        stop = int(sample) + SAMPLES_AFTER_R
        if symbol in LABEL_MAP and start >= 0 and stop <= signal_length:
            return int(sample), symbol, LABEL_MAP[symbol]
    raise RuntimeError("No selected-class beat has a complete extraction window.")


def inspect_record(record_name: str, data_dir: Path, output_dir: Path) -> None:
    record_path = data_dir / record_name
    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(f"Missing header file: {record_path.with_suffix('.hea')}")
    if not record_path.with_suffix(".atr").exists():
        raise FileNotFoundError(f"Missing annotation file: {record_path.with_suffix('.atr')}")

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    if int(record.fs) != FS_EXPECTED:
        raise ValueError(
            f"Expected {FS_EXPECTED} Hz but record {record_name} uses {record.fs} Hz."
        )

    lead_index = choose_lead(list(record.sig_name))
    lead_name = record.sig_name[lead_index]
    signal = record.p_signal[:, lead_index]

    annotation_counts = Counter(annotation.symbol)
    selected_counts = {
        mapped: annotation_counts.get(raw, 0)
        for raw, mapped in LABEL_MAP.items()
    }

    r_sample, raw_symbol, mapped_label = find_example_beat(
        annotation.sample,
        list(annotation.symbol),
        len(signal),
    )
    start = r_sample - SAMPLES_BEFORE_R
    stop = r_sample + SAMPLES_AFTER_R
    beat = signal[start:stop]

    if len(beat) != BEAT_LENGTH:
        raise AssertionError(f"Expected {BEAT_LENGTH} samples, received {len(beat)}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "record": record_name,
        "sampling_frequency_hz": float(record.fs),
        "signal_shape": list(record.p_signal.shape),
        "duration_seconds": float(record.sig_len / record.fs),
        "available_leads": list(record.sig_name),
        "selected_lead": lead_name,
        "annotation_count": len(annotation.sample),
        "selected_class_counts": selected_counts,
        "example": {
            "r_peak_sample": r_sample,
            "raw_symbol": raw_symbol,
            "mapped_label": mapped_label,
            "window_start": start,
            "window_stop_exclusive": stop,
            "beat_length": len(beat),
        },
    }

    summary_path = output_dir / f"record_{record_name}_inspection.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    count_rows = [
        {
            "raw_symbol": raw,
            "mapped_class": LABEL_MAP.get(raw, "not_selected"),
            "count": count,
        }
        for raw, count in sorted(annotation_counts.items())
    ]
    pd.DataFrame(count_rows).to_csv(
        output_dir / f"record_{record_name}_annotation_counts.csv",
        index=False,
    )

    time_ms = (
        np.arange(-SAMPLES_BEFORE_R, SAMPLES_AFTER_R, dtype=float)
        / record.fs
        * 1000.0
    )
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(time_ms, beat, color="#155EEF", linewidth=1.2)
    axis.axvline(0.0, color="#D92D20", linestyle="--", linewidth=1.0, label="R peak")
    axis.set_title(
        f"MIT-BIH record {record_name}: class {mapped_label} ({lead_name})"
    )
    axis.set_xlabel("Time relative to annotated R peak (ms)")
    axis.set_ylabel("Amplitude (mV)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / f"record_{record_name}_example_beat.png",
        dpi=180,
    )
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved inspection outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default="100", help="MIT-BIH record name")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing MIT-BIH .dat/.hea/.atr files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the inspection JSON, CSV, and PNG",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    inspect_record(arguments.record, arguments.data_dir, arguments.output_dir)

