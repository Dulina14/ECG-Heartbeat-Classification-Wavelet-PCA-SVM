"""Run baseline, class-weighted, RR, and class-weighted RR experiments.

One command can evaluate all four controlled models under both the beat-based
and paper record-based schemes. Every fold fits PCA, scaling, and class weights
using training rows only. Runs are resumable at fold boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.run_baseline import (
        BEAT_SPLITS_PATH,
        CLASS_ORDER,
        DEFAULT_C,
        DEFAULT_COMPONENTS,
        DEFAULT_GAMMA,
        LABELS_PATH,
        RECORD_IDS_PATH,
        RECORD_SPLITS_PATH,
        WAVELET_PATH,
        aggregate_results,
        load_folds,
        metrics_from_confusion,
        validate_fold,
        write_json,
    )
except ModuleNotFoundError:  # Support: python src/run_experiments.py
    from run_baseline import (
        BEAT_SPLITS_PATH,
        CLASS_ORDER,
        DEFAULT_C,
        DEFAULT_COMPONENTS,
        DEFAULT_GAMMA,
        LABELS_PATH,
        RECORD_IDS_PATH,
        RECORD_SPLITS_PATH,
        WAVELET_PATH,
        aggregate_results,
        load_folds,
        metrics_from_confusion,
        validate_fold,
        write_json,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"

REFERENCE_RR_PATH = INTERIM_DIR / "rr_features_reference.npy"
BEAT_ONLY_RR_PATH = INTERIM_DIR / "rr_features_beat_only.npy"

MODEL_NAMES = ("baseline", "weighted", "rr", "weighted_rr")
WEIGHTING_METHODS = ("balanced", "partner")


def calculate_class_weights(
    labels: np.ndarray,
    method: str = "balanced",
) -> dict[str, float]:
    """Calculate weights using training labels only."""

    classes, counts = np.unique(labels, return_counts=True)
    if method == "balanced":
        # Standard inverse-frequency weighting used by scikit-learn:
        # n_samples / (n_classes * n_class_samples).
        values = len(labels) / (len(classes) * counts.astype(float))
    elif method == "partner":
        # Exact convention used by Implementation_1. This has the same class
        # ratios as inverse frequency but a larger common scale, changing the
        # effective SVM C value.
        values = counts.max() / counts.astype(float)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return {
        str(label): float(weight)
        for label, weight in zip(classes, values, strict=True)
    }


def model_uses_rr(model: str) -> bool:
    return model in ("rr", "weighted_rr")


def model_uses_weights(model: str) -> bool:
    return model in ("weighted", "weighted_rr")


def experiment_tag(
    model: str,
    scheme: str,
    rr_source: str,
    weighting: str,
) -> str:
    parts = [model]
    if model_uses_rr(model):
        parts.append(rr_source)
    if model_uses_weights(model):
        parts.append(weighting)
    parts.append(scheme)
    return "_".join(parts)


def fit_predict_experiment(
    training_wavelets: np.ndarray,
    training_labels: np.ndarray,
    test_wavelets: np.ndarray,
    training_rr: np.ndarray | None,
    test_rr: np.ndarray | None,
    model: str,
    weighting: str = "balanced",
    n_components: int = DEFAULT_COMPONENTS,
    c_value: float = DEFAULT_C,
    gamma: float = DEFAULT_GAMMA,
    cache_size_mb: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one training-fold model and predict its test fold."""

    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")

    pca_started = time.perf_counter()
    pca = PCA(
        n_components=n_components,
        svd_solver="covariance_eigh",
    )
    transformed_training = pca.fit_transform(training_wavelets)
    transformed_test = pca.transform(test_wavelets)
    pca_seconds = time.perf_counter() - pca_started

    scaling_seconds = 0.0
    scaler_used = model_uses_rr(model)
    if scaler_used:
        if training_rr is None or test_rr is None:
            raise ValueError(f"Model {model} requires RR features")
        if len(training_rr) != len(training_wavelets):
            raise ValueError("Training RR and wavelet rows do not align")
        if len(test_rr) != len(test_wavelets):
            raise ValueError("Test RR and wavelet rows do not align")

        combined_training = np.hstack((transformed_training, training_rr))
        combined_test = np.hstack((transformed_test, test_rr))

        scaling_started = time.perf_counter()
        scaler = StandardScaler()
        transformed_training = scaler.fit_transform(combined_training)
        transformed_test = scaler.transform(combined_test)
        scaling_seconds = time.perf_counter() - scaling_started

    class_weights = (
        calculate_class_weights(training_labels, method=weighting)
        if model_uses_weights(model)
        else None
    )

    svm_started = time.perf_counter()
    classifier = SVC(
        kernel="rbf",
        C=c_value,
        gamma=gamma,
        decision_function_shape="ovo",
        class_weight=class_weights,
        cache_size=cache_size_mb,
    )
    classifier.fit(transformed_training, training_labels)
    svm_fit_seconds = time.perf_counter() - svm_started

    prediction_started = time.perf_counter()
    predictions = classifier.predict(transformed_test)
    prediction_seconds = time.perf_counter() - prediction_started

    details = {
        "model_input_dimensions": int(transformed_training.shape[1]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_seconds": pca_seconds,
        "scaler_used": scaler_used,
        "scaling_seconds": scaling_seconds,
        "class_weights": class_weights,
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


def expected_configuration(
    model: str,
    rr_source: str,
    weighting: str,
    n_components: int,
    c_value: float,
    gamma: float,
) -> dict[str, Any]:
    uses_rr = model_uses_rr(model)
    uses_weights = model_uses_weights(model)
    return {
        "input_features": (
            "12 PCA scores plus 6 RR features"
            if uses_rr
            else "12 PCA scores"
        ),
        "wavelet_features_before_pca": 365,
        "pca_components": n_components,
        "pca_solver": "covariance_eigh",
        "pca_fit_scope": "training fold only",
        "rr_source": rr_source if uses_rr else None,
        "scaling": "training-fold StandardScaler" if uses_rr else None,
        "svm_kernel": "rbf",
        "svm_C": c_value,
        "svm_gamma": gamma,
        "svm_multiclass": "one-versus-one",
        "class_weighting": weighting if uses_weights else None,
        "class_weight_fit_scope": "training fold only" if uses_weights else None,
    }


def load_existing_result(
    output_path: Path,
    configuration: dict[str, Any],
    model: str,
    scheme: str,
    requested_folds: int,
    force: bool,
) -> dict[str, Any] | None:
    if force or not output_path.exists():
        return None

    existing = json.loads(output_path.read_text(encoding="utf-8"))
    if existing.get("configuration") != configuration:
        raise ValueError(
            f"Existing result configuration differs: {output_path}. "
            "Use --force to replace it."
        )
    if existing.get("model") != model or existing.get("scheme") != scheme:
        raise ValueError(f"Existing result identity differs: {output_path}")
    if existing.get("number_of_folds_requested") != requested_folds:
        raise ValueError(
            f"Existing result has a different fold count: {output_path}. "
            "Use --force to replace it."
        )
    return existing


def run_model_scheme(
    model: str,
    scheme: str,
    wavelet_path: Path,
    rr_path: Path,
    labels_path: Path,
    record_ids_path: Path,
    split_path: Path,
    output_path: Path,
    rr_source: str,
    weighting: str,
    n_components: int,
    c_value: float,
    gamma: float,
    cache_size_mb: int,
    max_folds: int | None,
    force: bool,
) -> dict[str, Any]:
    wavelets = np.load(wavelet_path, mmap_mode="r", allow_pickle=False)
    rr_features = (
        np.load(rr_path, mmap_mode="r", allow_pickle=False)
        if model_uses_rr(model)
        else None
    )
    labels = np.load(labels_path, allow_pickle=False)
    record_ids = np.load(record_ids_path, allow_pickle=False).astype("U3")

    if len(wavelets) != len(labels) or len(labels) != len(record_ids):
        raise ValueError("Wavelet, label and record arrays are not aligned")
    if wavelets.shape[1] != 365:
        raise ValueError(f"Expected 365 wavelet columns, got {wavelets.shape[1]}")
    if rr_features is not None:
        if rr_features.shape != (len(labels), 6):
            raise ValueError(f"Expected RR shape {(len(labels), 6)}, got {rr_features.shape}")
        if not np.isfinite(rr_features).all():
            raise ValueError("RR matrix contains non-finite values")

    folds = load_folds(split_path, max_folds=max_folds)
    configuration = expected_configuration(
        model,
        rr_source,
        weighting,
        n_components,
        c_value,
        gamma,
    )
    existing = load_existing_result(
        output_path,
        configuration,
        model,
        scheme,
        len(folds),
        force,
    )

    if existing is not None and existing.get("status") == "complete":
        print(f"Skipping completed experiment: {output_path.name}")
        return existing

    if existing is None:
        result: dict[str, Any] = {
            "status": "in_progress",
            "model": model,
            "scheme": scheme,
            "configuration": configuration,
            "number_of_available_beats": len(labels),
            "number_of_folds_requested": len(folds),
            "folds": [],
        }
    else:
        result = existing
        print(
            f"Resuming {output_path.name} after "
            f"{len(result['folds'])} completed folds"
        )

    completed_folds = len(result["folds"])

    for zero_based_fold, (train, test) in enumerate(folds):
        if zero_based_fold < completed_folds:
            continue

        fold_number = zero_based_fold + 1
        validate_fold(train, test, len(labels))

        train_records = set(record_ids[train].tolist())
        test_records = set(record_ids[test].tolist())
        shared_records = sorted(train_records.intersection(test_records))
        if scheme == "record" and shared_records:
            raise AssertionError(
                f"Record leakage in fold {fold_number}: {shared_records}"
            )

        print(
            f"[{model} | {scheme} fold {fold_number}/{len(folds)}] "
            f"train={len(train):,}, test={len(test):,}, "
            f"shared records={len(shared_records)}"
        )

        training_wavelets = np.asarray(wavelets[train], dtype=np.float32)
        test_wavelets = np.asarray(wavelets[test], dtype=np.float32)
        training_rr = (
            np.asarray(rr_features[train], dtype=np.float32)
            if rr_features is not None
            else None
        )
        test_rr = (
            np.asarray(rr_features[test], dtype=np.float32)
            if rr_features is not None
            else None
        )

        predictions, details = fit_predict_experiment(
            training_wavelets,
            labels[train],
            test_wavelets,
            training_rr,
            test_rr,
            model=model,
            weighting=weighting,
            n_components=n_components,
            c_value=c_value,
            gamma=gamma,
            cache_size_mb=cache_size_mb,
        )

        matrix = confusion_matrix(labels[test], predictions, labels=CLASS_ORDER)
        metrics = metrics_from_confusion(matrix)

        result["folds"].append(
            {
                "fold": fold_number,
                "training_beats": len(train),
                "test_beats": len(test),
                "training_record_count": len(train_records),
                "test_record_count": len(test_records),
                "test_records": sorted(test_records),
                "shared_train_test_records": shared_records,
                **details,
                "metrics": metrics,
            }
        )
        write_json(output_path, result)

        print(
            f"  overall={metrics['overall_accuracy']:.2f}% | "
            f"balanced={metrics['balanced_accuracy']:.2f}% | "
            f"macro-F1={metrics['macro_F1']:.2f}% | "
            f"support vectors={details['support_vectors_total']:,}"
        )

    result["status"] = "complete"
    result["aggregation_note"] = (
        "Beat-based pooled results form one complete partition. Paper record-"
        "based pooled results are repeat-weighted because holdouts overlap; "
        "use fold means for direct paper comparison."
    )
    result["aggregate"] = aggregate_results(result["folds"])
    write_json(output_path, result)
    return result


def summary_row(tag: str, result: dict[str, Any]) -> dict[str, Any]:
    fold_stats = result["aggregate"]["fold_mean_standard_deviation"]
    pooled = result["aggregate"]["pooled_or_repeat_weighted"]
    return {
        "experiment": tag,
        "model": result["model"],
        "scheme": result["scheme"],
        "fold_mean_paper_SEN": fold_stats["paper_total_SEN"]["mean"],
        "fold_mean_paper_SPE": fold_stats["paper_total_SPE"]["mean"],
        "fold_mean_paper_ACC": fold_stats["paper_total_ACC"]["mean"],
        "fold_mean_overall_accuracy": fold_stats["overall_accuracy"]["mean"],
        "fold_mean_balanced_accuracy": fold_stats["balanced_accuracy"]["mean"],
        "fold_mean_macro_F1": fold_stats["macro_F1"]["mean"],
        "pooled_overall_accuracy": pooled["overall_accuracy"],
        "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        "pooled_macro_F1": pooled["macro_F1"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("beat", "record", "both"), default="both")
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help="Comma-separated: baseline,weighted,rr,weighted_rr",
    )
    parser.add_argument(
        "--rr-source",
        choices=("beat_only", "reference"),
        default="beat_only",
    )
    parser.add_argument(
        "--weighting",
        choices=WEIGHTING_METHODS,
        default="balanced",
    )
    parser.add_argument("--wavelets", type=Path, default=WAVELET_PATH)
    parser.add_argument("--rr-reference", type=Path, default=REFERENCE_RR_PATH)
    parser.add_argument("--rr-beat-only", type=Path, default=BEAT_ONLY_RR_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--record-ids", type=Path, default=RECORD_IDS_PATH)
    parser.add_argument("--beat-splits", type=Path, default=BEAT_SPLITS_PATH)
    parser.add_argument("--record-splits", type=Path, default=RECORD_SPLITS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--cache-size-mb", type=int, default=512)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing matching result files instead of resuming/skipping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(models).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    schemes = ("beat", "record") if args.scheme == "both" else (args.scheme,)
    rr_path = (
        args.rr_beat_only if args.rr_source == "beat_only" else args.rr_reference
    )
    summary_rows = []

    for model in models:
        for scheme in schemes:
            tag = experiment_tag(model, scheme, args.rr_source, args.weighting)
            if args.max_folds is not None:
                tag += f"_first_{args.max_folds}_folds"
            output_path = args.output_dir / f"{tag}.json"
            split_path = args.beat_splits if scheme == "beat" else args.record_splits

            result = run_model_scheme(
                model=model,
                scheme=scheme,
                wavelet_path=args.wavelets,
                rr_path=rr_path,
                labels_path=args.labels,
                record_ids_path=args.record_ids,
                split_path=split_path,
                output_path=output_path,
                rr_source=args.rr_source,
                weighting=args.weighting,
                n_components=args.components,
                c_value=args.C,
                gamma=args.gamma,
                cache_size_mb=args.cache_size_mb,
                max_folds=args.max_folds,
                force=args.force,
            )
            summary_rows.append(summary_row(tag, result))

    suffix = f"_first_{args.max_folds}_folds" if args.max_folds is not None else ""
    summary_path = args.output_dir / f"summary{suffix}.csv"
    write_summary_csv(summary_path, summary_rows)

    print("\n=== Experiment summary: fold means (%) ===")
    for row in summary_rows:
        print(
            f"{row['experiment']}: "
            f"overall={row['fold_mean_overall_accuracy']:.2f}, "
            f"balanced={row['fold_mean_balanced_accuracy']:.2f}, "
            f"macro-F1={row['fold_mean_macro_F1']:.2f}"
        )
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
