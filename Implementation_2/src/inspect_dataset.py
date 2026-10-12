"""Inspect all official MIT-BIH records and build dataset manifest files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wfdb

try:
    from src.inspect_record import LABEL_MAP, choose_lead
except ModuleNotFoundError:  # Support: python src/inspect_dataset.py
    from inspect_record import LABEL_MAP, choose_lead


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mitdb"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

EXPECTED_FS = 360
EXPECTED_RECORD_COUNT = 48
CLASS_ORDER = ["N", "A", "V", "L", "R", "P"]


def read_official_records(data_dir: Path) -> list[str]:
    """Read record names from the database's official RECORDS file."""
    records_file = data_dir / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"Missing official record list: {records_file}")

    records = [
        line.strip()
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RECORD_COUNT} records, found {len(records)} in {records_file}."
        )
    if len(records) != len(set(records)):
        raise ValueError("The official RECORDS file contains duplicate record names.")
    return records


def verify_record_files(data_dir: Path, record_name: str) -> None:
    """Ensure that a signal, header and reference annotation exist."""
    missing = [
        suffix
        for suffix in (".dat", ".hea", ".atr")
        if not (data_dir / f"{record_name}{suffix}").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Record {record_name} is missing required files: {', '.join(missing)}"
        )


def inspect_dataset(
    data_dir: Path,
    manifest_dir: Path,
    figure_dir: Path,
) -> None:
    records = read_official_records(data_dir)
    manifest_rows: list[dict[str, object]] = []

    for position, record_name in enumerate(records, start=1):
        verify_record_files(data_dir, record_name)
        record_path = data_dir / record_name
        header = wfdb.rdheader(str(record_path))
        annotation = wfdb.rdann(str(record_path), "atr")

        if int(header.fs) != EXPECTED_FS:
            raise ValueError(
                f"Record {record_name} has sampling frequency {header.fs}, "
                f"expected {EXPECTED_FS}."
            )

        signal_names = list(header.sig_name)
        lead_index = choose_lead(signal_names)
        annotation_counts = Counter(annotation.symbol)
        selected_counts = {
            mapped: int(annotation_counts.get(raw, 0))
            for raw, mapped in LABEL_MAP.items()
        }

        row: dict[str, object] = {
            "record_id": record_name,
            "sampling_frequency_hz": float(header.fs),
            "signal_length_samples": int(header.sig_len),
            "duration_seconds": float(header.sig_len / header.fs),
            "number_of_channels": int(header.n_sig),
            "available_leads": "|".join(signal_names),
            "selected_lead": signal_names[lead_index],
            "selected_lead_index": lead_index,
            "annotation_count": len(annotation.sample),
            "selected_beat_count": sum(selected_counts.values()),
        }
        row.update({f"count_{label}": selected_counts[label] for label in CLASS_ORDER})
        manifest_rows.append(row)

        print(
            f"[{position:02d}/{len(records)}] record {record_name}: "
            f"lead={signal_names[lead_index]}, selected beats={row['selected_beat_count']}"
        )

    manifest = pd.DataFrame(manifest_rows)
    class_counts = pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "count_before_window_filtering": [
                int(manifest[f"count_{label}"].sum()) for label in CLASS_ORDER
            ],
        }
    )
    selected_total = int(class_counts["count_before_window_filtering"].sum())
    class_counts["percentage"] = (
        class_counts["count_before_window_filtering"] / selected_total * 100.0
    )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "record_manifest.csv"
    counts_path = manifest_dir / "class_counts_before_window_filtering.csv"
    summary_path = manifest_dir / "dataset_inspection_summary.json"
    figure_path = figure_dir / "class_distribution_before_window_filtering.png"

    manifest.to_csv(manifest_path, index=False)
    class_counts.to_csv(counts_path, index=False)

    lead_counts = manifest["selected_lead"].value_counts().to_dict()
    summary = {
        "record_count": len(records),
        "sampling_frequency_hz": EXPECTED_FS,
        "selected_beat_count_before_window_filtering": selected_total,
        "paper_reported_final_beat_count": 107049,
        "note": (
            "The final count is checked after removing beats whose complete "
            "252-sample window does not fit inside a record."
        ),
        "class_counts_before_window_filtering": {
            row["class"]: int(row["count_before_window_filtering"])
            for _, row in class_counts.iterrows()
        },
        "selected_lead_counts": {str(key): int(value) for key, value in lead_counts.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        class_counts["class"],
        class_counts["count_before_window_filtering"],
        color=["#155EEF", "#7F56D9", "#D92D20", "#039855", "#DC6803", "#475467"],
    )
    axis.set_yscale("log")
    axis.set_title("MIT-BIH selected heartbeat classes before window filtering")
    axis.set_xlabel("Heartbeat class")
    axis.set_ylabel("Number of annotated beats (log scale)")
    axis.grid(axis="y", alpha=0.25, which="both")
    for bar, count in zip(
        bars,
        class_counts["count_before_window_filtering"],
        strict=True,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            float(count) * 1.06,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print("\nClass counts before 252-sample window filtering:")
    print(class_counts.to_string(index=False))
    print(f"\nTotal selected annotations: {selected_total:,}")
    print(f"Saved record manifest: {manifest_path}")
    print(f"Saved class counts: {counts_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved class-distribution figure: {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    inspect_dataset(arguments.data_dir, arguments.manifest_dir, arguments.figure_dir)

