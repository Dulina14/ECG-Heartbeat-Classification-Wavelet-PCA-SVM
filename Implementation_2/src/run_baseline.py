"""Run the paper's fold-local PCA(12) plus RBF-SVM baseline.

PCA is fitted separately on the training rows of every experiment. Test rows
are transformed with that fitted PCA and never participate in fitting,
scaling, weighting, or parameter selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"

WAVELET_PATH = INTERIM_DIR / "wavelet_features_bior68_l8_symmetric.npy"
LABELS_PATH = INTERIM_DIR / "labels.npy"
RECORD_IDS_PATH = INTERIM_DIR / "record_ids.npy"
BEAT_SPLITS_PATH = INTERIM_DIR / "splits" / "beat_based_10_folds.npz"
RECORD_SPLITS_PATH = (
    INTERIM_DIR / "splits" / "record_based_paper_10_holdouts.npz"
)

CLASS_ORDER = ["A", "L", "N", "P", "R", "V"]
DEFAULT_COMPONENTS = 12
DEFAULT_C = 10.0
DEFAULT_GAMMA = 0.1


def safe_percentage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def metrics_from_confusion(
    matrix: np.ndarray,
    classes: list[str] = CLASS_ORDER,
) -> dict[str, Any]:
    """Calculate paper-compatible and modern metrics from a confusion matrix."""

    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)):
        raise ValueError(
            f"Expected {(len(classes), len(classes))} confusion matrix, "
            f"got {matrix.shape}"
        )

    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot calculate metrics from an empty confusion matrix")

    per_class: dict[str, dict[str, int | float | None]] = {}
    count_rows = []

    for index, label in enumerate(classes):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(total - tp - fn - fp)

        sensitivity = safe_percentage(tp, tp + fn)
        specificity = safe_percentage(tn, tn + fp)
        accuracy = safe_percentage(tp + tn, total)
        precision = safe_percentage(tp, tp + fp)
        f1 = safe_percentage(2 * tp, 2 * tp + fp + fn)

        per_class[label] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "support": tp + fn,
            "SEN": sensitivity,
            "SPE": specificity,
            "ACC": accuracy,
            "PPV": precision,
            "F1": f1,
        }
        count_rows.append((tp, tn, fp, fn))

    sum_tp = sum(row[0] for row in count_rows)
    sum_tn = sum(row[1] for row in count_rows)
    sum_fp = sum(row[2] for row in count_rows)
    sum_fn = sum(row[3] for row in count_rows)

    paper_total = {
        "SEN": safe_percentage(sum_tp, sum_tp + sum_fn),
        "SPE": safe_percentage(sum_tn, sum_tn + sum_fp),
        "ACC": safe_percentage(
            sum_tp + sum_tn,
            sum_tp + sum_tn + sum_fp + sum_fn,
        ),
    }

    present = [label for label in classes if per_class[label]["support"] > 0]
    balanced_accuracy = float(
        np.mean([per_class[label]["SEN"] for label in present])
    )
    macro_f1 = float(
        np.mean(
            [
                per_class[label]["F1"]
                if per_class[label]["F1"] is not None
                else 0.0
                for label in present
            ]
        )
    )

    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "paper_total": paper_total,
        "overall_accuracy": 100.0 * float(np.trace(matrix)) / total,
        "balanced_accuracy": balanced_accuracy,
        "macro_F1": macro_f1,
        "number_of_predictions": total,
    }


def load_folds(
    split_path: Path,
    max_folds: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load train/test arrays from the split archive."""

    with np.load(split_path, allow_pickle=False) as archive:
        fold_numbers = sorted(
            int(name.split("_")[1])
            for name in archive.files
            if name.startswith("train_")
        )
        if max_folds is not None:
            fold_numbers = fold_numbers[:max_folds]
        return [
            (
                archive[f"train_{fold:02d}"].astype(np.int64),
                archive[f"test_{fold:02d}"].astype(np.int64),
            )
            for fold in fold_numbers
        ]


def validate_fold(
    train: np.ndarray,
    test: np.ndarray,
    number_of_rows: int,
) -> None:
    if np.intersect1d(train, test).size:
        raise AssertionError("Training and test indices overlap")
    if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
        raise AssertionError("Fold contains duplicate row indices")
    if len(train) + len(test) != number_of_rows:
        raise AssertionError("Fold does not include every dataset row")
    if train.min() < 0 or test.min() < 0:
        raise IndexError("Fold contains a negative row index")
    if train.max() >= number_of_rows or test.max() >= number_of_rows:
        raise IndexError("Fold contains an out-of-range row index")


def fit_predict_baseline(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    test_features: np.ndarray,
    n_components: int = DEFAULT_COMPONENTS,
    c_value: float = DEFAULT_C,
    gamma: float = DEFAULT_GAMMA,
    cache_size_mb: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit training-only PCA and SVM, then predict the test rows."""

    pca_started = time.perf_counter()
    pca = PCA(
        n_components=n_components,
        svd_solver="covariance_eigh",
    )
    transformed_training = pca.fit_transform(training_features)
    transformed_test = pca.transform(test_features)
    pca_seconds = time.perf_counter() - pca_started

    svm_started = time.perf_counter()
    classifier = SVC(
        kernel="rbf",
        C=c_value,
        gamma=gamma,
        decision_function_shape="ovo",
        class_weight=None,
        cache_size=cache_size_mb,
    )
    classifier.fit(transformed_training, training_labels)
    svm_fit_seconds = time.perf_counter() - svm_started

    prediction_started = time.perf_counter()
    predictions = classifier.predict(transformed_test)
    prediction_seconds = time.perf_counter() - prediction_started

    details = {
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_seconds": pca_seconds,
        "svm_fit_seconds": svm_fit_seconds,
        "prediction_seconds": prediction_seconds,
        "support_vectors_total": int(classifier.n_support_.sum()),
        "support_vectors_by_class": {
            str(label): int(count)
            for label, count in zip(
                classifier.classes_,
                classifier.n_support_,
                strict=True,
            )
        },
    }
    return predictions, details


def numeric_stats(values: list[float | None]) -> dict[str, float | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"mean": None, "standard_deviation": None}
    return {
        "mean": float(np.mean(usable)),
        "standard_deviation": float(np.std(usable)),
    }


def aggregate_results(folds: list[dict[str, Any]]) -> dict[str, Any]:
    matrices = [np.asarray(fold["metrics"]["confusion_matrix"]) for fold in folds]
    pooled_matrix = np.sum(matrices, axis=0)
    pooled = metrics_from_confusion(pooled_matrix)

    fold_mean_standard_deviation = {
        "overall_accuracy": numeric_stats(
            [fold["metrics"]["overall_accuracy"] for fold in folds]
        ),
        "balanced_accuracy": numeric_stats(
            [fold["metrics"]["balanced_accuracy"] for fold in folds]
        ),
        "macro_F1": numeric_stats(
            [fold["metrics"]["macro_F1"] for fold in folds]
        ),
        "paper_total_SEN": numeric_stats(
            [fold["metrics"]["paper_total"]["SEN"] for fold in folds]
        ),
        "paper_total_SPE": numeric_stats(
            [fold["metrics"]["paper_total"]["SPE"] for fold in folds]
        ),
        "paper_total_ACC": numeric_stats(
            [fold["metrics"]["paper_total"]["ACC"] for fold in folds]
        ),
    }

    per_class_fold_statistics = {}
    for label in CLASS_ORDER:
        per_class_fold_statistics[label] = {
            metric: numeric_stats(
                [fold["metrics"]["per_class"][label][metric] for fold in folds]
            )
            for metric in ("SEN", "SPE", "ACC", "PPV", "F1")
        }

    return {
        "pooled_or_repeat_weighted": pooled,
        "fold_mean_standard_deviation": fold_mean_standard_deviation,
        "per_class_fold_statistics": per_class_fold_statistics,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def run_scheme(
    scheme: str,
    feature_path: Path,
    labels_path: Path,
    record_ids_path: Path,
    split_path: Path,
    output_path: Path,
    n_components: int,
    c_value: float,
    gamma: float,
    cache_size_mb: int,
    max_folds: int | None,
) -> dict[str, Any]:
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    record_ids = np.load(record_ids_path, allow_pickle=False).astype("U3")

    if len(features) != len(labels) or len(labels) != len(record_ids):
        raise ValueError("Feature, label and record arrays are not row-aligned")
    if features.shape[1] != 365:
        raise ValueError(f"Expected 365 wavelet features, got {features.shape[1]}")

    folds = load_folds(split_path, max_folds=max_folds)
    result: dict[str, Any] = {
        "status": "in_progress",
        "model": "paper_baseline",
        "scheme": scheme,
        "configuration": {
            "input_features": "365 concatenated bior6.8 cD1-cD8 coefficients",
            "pca_components": n_components,
            "pca_solver": "covariance_eigh",
            "pca_fit_scope": "training fold only",
            "scaling": None,
            "svm_kernel": "rbf",
            "svm_C": c_value,
            "svm_gamma": gamma,
            "svm_multiclass": "one-versus-one",
            "svm_class_weight": None,
        },
        "number_of_available_beats": len(labels),
        "number_of_folds_requested": len(folds),
        "folds": [],
    }

    for fold_number, (train, test) in enumerate(folds, start=1):
        validate_fold(train, test, len(labels))
        train_records = set(record_ids[train].tolist())
        test_records = set(record_ids[test].tolist())
        shared_records = sorted(train_records.intersection(test_records))

        if scheme == "record" and shared_records:
            raise AssertionError(
                f"Record leakage in fold {fold_number}: {shared_records}"
            )

        print(
            f"[{scheme} fold {fold_number}/{len(folds)}] "
            f"train={len(train):,}, test={len(test):,}, "
            f"shared records={len(shared_records)}"
        )

        training_features = np.asarray(features[train], dtype=np.float32)
        test_features = np.asarray(features[test], dtype=np.float32)

        predictions, model_details = fit_predict_baseline(
            training_features,
            labels[train],
            test_features,
            n_components=n_components,
            c_value=c_value,
            gamma=gamma,
            cache_size_mb=cache_size_mb,
        )

        matrix = confusion_matrix(
            labels[test],
            predictions,
            labels=CLASS_ORDER,
        )
        metrics = metrics_from_confusion(matrix)

        fold_result = {
            "fold": fold_number,
            "training_beats": len(train),
            "test_beats": len(test),
            "training_record_count": len(train_records),
            "test_record_count": len(test_records),
            "test_records": sorted(test_records),
            "shared_train_test_records": shared_records,
            **model_details,
            "metrics": metrics,
        }
        result["folds"].append(fold_result)
        write_json(output_path, result)

        print(
            f"  overall={metrics['overall_accuracy']:.2f}% | "
            f"balanced={metrics['balanced_accuracy']:.2f}% | "
            f"macro-F1={metrics['macro_F1']:.2f}% | "
            f"PCA variance={100 * model_details['pca_explained_variance']:.2f}%"
        )

    result["status"] = "complete"
    result["aggregation_note"] = (
        "For beat-based folds, pooled predictions form one complete test "
        "partition. For the paper record scheme, pooled metrics are repeat-"
        "weighted because test records overlap; use fold means for direct "
        "comparison with the paper."
    )
    result["aggregate"] = aggregate_results(result["folds"])
    write_json(output_path, result)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme",
        choices=("beat", "record", "both"),
        default="beat",
    )
    parser.add_argument("--features", type=Path, default=WAVELET_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--record-ids", type=Path, default=RECORD_IDS_PATH)
    parser.add_argument("--beat-splits", type=Path, default=BEAT_SPLITS_PATH)
    parser.add_argument("--record-splits", type=Path, default=RECORD_SPLITS_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--cache-size-mb", type=int, default=512)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N folds for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    schemes = ("beat", "record") if args.scheme == "both" else (args.scheme,)

    for scheme in schemes:
        split_path = (
            args.beat_splits if scheme == "beat" else args.record_splits
        )
        suffix = (
            f"_first_{args.max_folds}_folds" if args.max_folds is not None else ""
        )
        output_path = args.output_dir / f"paper_baseline_{scheme}{suffix}.json"

        result = run_scheme(
            scheme=scheme,
            feature_path=args.features,
            labels_path=args.labels,
            record_ids_path=args.record_ids,
            split_path=split_path,
            output_path=output_path,
            n_components=args.components,
            c_value=args.C,
            gamma=args.gamma,
            cache_size_mb=args.cache_size_mb,
            max_folds=args.max_folds,
        )

        aggregate = result["aggregate"]
        means = aggregate["fold_mean_standard_deviation"]
        print(f"\nCompleted {scheme}-based baseline")
        print(
            "Fold-mean paper total: "
            f"SEN={means['paper_total_SEN']['mean']:.2f}%, "
            f"SPE={means['paper_total_SPE']['mean']:.2f}%, "
            f"ACC={means['paper_total_ACC']['mean']:.2f}%"
        )
        print(f"Saved: {output_path}\n")


if __name__ == "__main__":
    main()
