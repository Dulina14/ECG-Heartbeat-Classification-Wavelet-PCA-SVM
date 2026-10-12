"""Create audited beat-based and paper record-based evaluation splits.

The output indices are generated before PCA or SVM fitting. This makes the
training/test boundary explicit and prevents PCA leakage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

LABELS_PATH = INTERIM_DIR / "labels.npy"
RECORD_IDS_PATH = INTERIM_DIR / "record_ids.npy"
SPLIT_DIR = INTERIM_DIR / "splits"
BEAT_SPLITS_PATH = SPLIT_DIR / "beat_based_10_folds.npz"
RECORD_SPLITS_PATH = SPLIT_DIR / "record_based_paper_10_holdouts.npz"
SUMMARY_PATH = MANIFEST_DIR / "cv_split_summary.json"

CLASS_ORDER = ["A", "L", "N", "P", "R", "V"]
N_SPLITS = 10
RANDOM_SEED = 0

# Paper Table 3, also used by Implementation_1. These are ten predefined
# record-based test subsets. They overlap across experiments and therefore do
# not form a conventional disjoint GroupKFold partition.
PAPER_RECORD_TEST_FOLDS = (
    ("100", "101", "109", "118", "217"),
    ("105", "106", "107", "111", "124"),
    ("104", "112", "113", "114", "207"),
    ("102", "116", "117", "212", "214"),
    ("107", "109", "122", "123", "231"),
    ("111", "202", "203", "217", "232"),
    ("102", "118", "207", "209", "210"),
    ("104", "124", "214", "215", "219"),
    ("109", "207", "217", "222"),
    ("102", "111", "212", "233"),
)


def create_beat_folds(
    labels: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return shuffled stratified folds of individual heartbeat rows."""

    labels = np.asarray(labels)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    placeholder = np.zeros(len(labels), dtype=np.uint8)
    return [
        (train.astype(np.int32), test.astype(np.int32))
        for train, test in splitter.split(placeholder, labels)
    ]


def create_paper_record_holdouts(
    record_ids: np.ndarray,
    fold_table: tuple[tuple[str, ...], ...] = PAPER_RECORD_TEST_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the paper's ten predefined record-level experiments."""

    record_ids = np.asarray(record_ids).astype("U3")
    available = set(np.unique(record_ids).tolist())
    requested = {record for fold in fold_table for record in fold}
    missing = requested.difference(available)
    if missing:
        raise ValueError(f"Paper record folds request missing records: {sorted(missing)}")

    folds = []
    for test_records in fold_table:
        test_mask = np.isin(record_ids, test_records)
        test = np.flatnonzero(test_mask).astype(np.int32)
        train = np.flatnonzero(~test_mask).astype(np.int32)
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Empty train/test partition for records {test_records}")
        folds.append((train, test))
    return folds


def validate_fold_indices(
    folds: list[tuple[np.ndarray, np.ndarray]],
    number_of_rows: int,
) -> None:
    """Check bounds, duplicates, and within-fold train/test separation."""

    all_rows = np.arange(number_of_rows)
    for fold_number, (train, test) in enumerate(folds, start=1):
        if len(np.unique(train)) != len(train):
            raise AssertionError(f"Duplicate training index in fold {fold_number}")
        if len(np.unique(test)) != len(test):
            raise AssertionError(f"Duplicate test index in fold {fold_number}")
        if np.intersect1d(train, test).size:
            raise AssertionError(f"Train/test index leakage in fold {fold_number}")
        if not np.array_equal(np.sort(np.concatenate((train, test))), all_rows):
            raise AssertionError(f"Fold {fold_number} does not cover every row")


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(labels).tolist())
    return {label: int(counts[label]) for label in CLASS_ORDER}


def fold_summary(
    folds: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    record_ids: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for fold_number, (train, test) in enumerate(folds, start=1):
        train_records = set(record_ids[train].tolist())
        test_records = set(record_ids[test].tolist())
        rows.append(
            {
                "fold": fold_number,
                "training_beats": len(train),
                "test_beats": len(test),
                "training_class_counts": class_counts(labels[train]),
                "test_class_counts": class_counts(labels[test]),
                "training_record_count": len(train_records),
                "test_record_count": len(test_records),
                "test_records": sorted(test_records),
                "records_shared_between_train_and_test": sorted(
                    train_records.intersection(test_records)
                ),
            }
        )
    return rows


def save_folds(
    output_path: Path,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for fold_number, (train, test) in enumerate(folds, start=1):
        arrays[f"train_{fold_number:02d}"] = train
        arrays[f"test_{fold_number:02d}"] = test
    np.savez_compressed(output_path, **arrays)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--record-ids", type=Path, default=RECORD_IDS_PATH)
    parser.add_argument("--beat-output", type=Path, default=BEAT_SPLITS_PATH)
    parser.add_argument("--record-output", type=Path, default=RECORD_SPLITS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    labels = np.load(args.labels, allow_pickle=False)
    record_ids = np.load(args.record_ids, allow_pickle=False).astype("U3")

    if labels.shape != record_ids.shape:
        raise ValueError(
            f"Label and record arrays differ: {labels.shape} vs {record_ids.shape}"
        )

    unexpected_labels = set(np.unique(labels).tolist()).difference(CLASS_ORDER)
    if unexpected_labels:
        raise ValueError(f"Unexpected class labels: {sorted(unexpected_labels)}")

    beat_folds = create_beat_folds(labels, seed=args.seed)
    record_folds = create_paper_record_holdouts(record_ids)
    validate_fold_indices(beat_folds, len(labels))
    validate_fold_indices(record_folds, len(labels))

    beat_test_frequency = np.zeros(len(labels), dtype=np.uint8)
    for _, test in beat_folds:
        beat_test_frequency[test] += 1
    if not np.all(beat_test_frequency == 1):
        raise AssertionError("Beat-based test folds must partition all beats once")

    record_test_frequency = np.zeros(len(labels), dtype=np.uint8)
    for _, test in record_folds:
        record_test_frequency[test] += 1

    save_folds(args.beat_output, beat_folds)
    save_folds(args.record_output, record_folds)

    record_test_counts = Counter(
        record
        for fold in PAPER_RECORD_TEST_FOLDS
        for record in fold
    )
    never_tested_records = sorted(
        set(np.unique(record_ids).tolist()).difference(record_test_counts)
    )

    summary = {
        "number_of_beats": len(labels),
        "number_of_records": len(np.unique(record_ids)),
        "classes": CLASS_ORDER,
        "random_seed": args.seed,
        "beat_based": {
            "description": (
                "Shuffled 10-fold stratified split of individual beats. "
                "Patients/records may occur in both train and test sets."
            ),
            "test_frequency_min": int(beat_test_frequency.min()),
            "test_frequency_max": int(beat_test_frequency.max()),
            "folds": fold_summary(beat_folds, labels, record_ids),
        },
        "record_based": {
            "description": (
                "Ten predefined paper Table 3 record holdouts. Train/test "
                "records are disjoint within each experiment, but test "
                "records overlap across experiments."
            ),
            "is_disjoint_ten_fold_partition": False,
            "unique_test_records": len(record_test_counts),
            "never_tested_records": never_tested_records,
            "record_test_frequencies": dict(sorted(record_test_counts.items())),
            "beats_never_tested": int(np.sum(record_test_frequency == 0)),
            "beats_tested_more_than_once": int(np.sum(record_test_frequency > 1)),
            "test_frequency_min": int(record_test_frequency.min()),
            "test_frequency_max": int(record_test_frequency.max()),
            "folds": fold_summary(record_folds, labels, record_ids),
        },
        "leakage_rule": (
            "PCA, scaling, class weights, and the SVM must be fitted only on "
            "the train indices stored for each experiment."
        ),
        "files": {
            "beat_based_indices": str(args.beat_output),
            "record_based_indices": str(args.record_output),
        },
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Cross-validation split generation completed.")
    print(f"Beat-based folds: {len(beat_folds)}")
    print("Beat test coverage: every beat appears exactly once")
    print(f"Paper record holdouts: {len(record_folds)}")
    print(f"Unique records tested: {len(record_test_counts)}")
    print(f"Records never tested: {len(never_tested_records)}")
    print(f"Beats tested more than once: {int(np.sum(record_test_frequency > 1)):,}")
    print(f"Saved beat splits: {args.beat_output}")
    print(f"Saved record splits: {args.record_output}")
    print(f"Saved audit summary: {args.summary}")


if __name__ == "__main__":
    main()
