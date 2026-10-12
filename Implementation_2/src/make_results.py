"""Create final tables, figures, and a Markdown report from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.run_baseline import CLASS_ORDER
    from src.run_experiments import MODEL_NAMES, experiment_tag
except ModuleNotFoundError:  # Support: python src/make_results.py
    from run_baseline import CLASS_ORDER
    from run_experiments import MODEL_NAMES, experiment_tag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "experiments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "final"
FEATURE_SUMMARY_PATH = PROJECT_ROOT / "data" / "manifests" / "feature_extraction_summary.json"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "data" / "manifests" / "cv_split_summary.json"

MODEL_LABELS = {
    "baseline": "Baseline",
    "weighted": "Weighted SVM",
    "rr": "PCA + RR",
    "weighted_rr": "Weighted PCA + RR",
}

MODEL_COLORS = {
    "baseline": "#475467",
    "weighted": "#7F56D9",
    "rr": "#039855",
    "weighted_rr": "#155EEF",
}


def result_filename(
    model: str,
    scheme: str,
    rr_source: str,
    weighting: str,
) -> str:
    return experiment_tag(model, scheme, rr_source, weighting) + ".json"


def load_completed_results(
    input_dir: Path,
    rr_source: str,
    weighting: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the required four models under both evaluation schemes."""

    results = {}
    missing = []
    incomplete = []

    for model in MODEL_NAMES:
        for scheme in ("beat", "record"):
            filename = result_filename(model, scheme, rr_source, weighting)
            path = input_dir / filename
            if not path.exists():
                missing.append(str(path))
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "complete" or len(result.get("folds", [])) != 10:
                incomplete.append(
                    f"{path} (status={result.get('status')}, "
                    f"folds={len(result.get('folds', []))})"
                )
                continue
            results[(model, scheme)] = result

    if missing or incomplete:
        messages = []
        if missing:
            messages.append("Missing result files:\n- " + "\n- ".join(missing))
        if incomplete:
            messages.append("Incomplete result files:\n- " + "\n- ".join(incomplete))
        messages.append(
            "Run: python src/run_experiments.py --scheme both "
            "--models baseline,weighted,rr,weighted_rr "
            f"--rr-source {rr_source} --weighting {weighting}"
        )
        raise RuntimeError("\n\n".join(messages))

    return results


def summary_row(model: str, scheme: str, result: dict[str, Any]) -> dict[str, Any]:
    stats = result["aggregate"]["fold_mean_standard_deviation"]
    pooled = result["aggregate"]["pooled_or_repeat_weighted"]
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "scheme": scheme,
        "fold_mean_overall_accuracy": stats["overall_accuracy"]["mean"],
        "fold_std_overall_accuracy": stats["overall_accuracy"]["standard_deviation"],
        "fold_mean_balanced_accuracy": stats["balanced_accuracy"]["mean"],
        "fold_std_balanced_accuracy": stats["balanced_accuracy"]["standard_deviation"],
        "fold_mean_macro_F1": stats["macro_F1"]["mean"],
        "fold_std_macro_F1": stats["macro_F1"]["standard_deviation"],
        "fold_mean_paper_SEN": stats["paper_total_SEN"]["mean"],
        "fold_mean_paper_SPE": stats["paper_total_SPE"]["mean"],
        "fold_mean_paper_ACC": stats["paper_total_ACC"]["mean"],
        "pooled_overall_accuracy": pooled["overall_accuracy"],
        "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        "pooled_macro_F1": pooled["macro_F1"],
    }


def build_summary_rows(
    results: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        summary_row(model, scheme, results[(model, scheme)])
        for scheme in ("beat", "record")
        for model in MODEL_NAMES
    ]


def paired_delta_rows(
    results: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate fold-paired changes relative to the baseline."""

    rows = []
    for scheme in ("beat", "record"):
        baseline = results[("baseline", scheme)]["folds"]
        for model in ("weighted", "rr", "weighted_rr"):
            comparison = results[(model, scheme)]["folds"]
            for metric in ("overall_accuracy", "balanced_accuracy", "macro_F1"):
                differences = np.array(
                    [
                        new["metrics"][metric] - old["metrics"][metric]
                        for old, new in zip(baseline, comparison, strict=True)
                    ]
                )
                rows.append(
                    {
                        "scheme": scheme,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "metric": metric,
                        "mean_paired_change_percentage_points": float(
                            differences.mean()
                        ),
                        "std_paired_change_percentage_points": float(
                            differences.std()
                        ),
                        "folds_improved": int(np.sum(differences > 0)),
                        "folds_unchanged": int(np.sum(differences == 0)),
                        "folds_worsened": int(np.sum(differences < 0)),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scheme_comparison(
    summary_rows: list[dict[str, Any]],
    scheme: str,
    output_path: Path,
) -> None:
    rows = [row for row in summary_rows if row["scheme"] == scheme]
    metrics = [
        ("fold_mean_overall_accuracy", "Overall accuracy"),
        ("fold_mean_balanced_accuracy", "Balanced accuracy"),
        ("fold_mean_macro_F1", "Macro-F1"),
    ]
    x = np.arange(len(rows))
    width = 0.24

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for metric_index, (key, label) in enumerate(metrics):
        values = [row[key] for row in rows]
        positions = x + (metric_index - 1) * width
        bars = axis.bar(positions, values, width=width, label=label)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.7,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(x, [row["model_label"] for row in rows])
    axis.set_ylabel("Fold-mean performance (%)")
    axis.set_title(f"{scheme.capitalize()}-based model comparison")
    axis.set_ylim(0, 108)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_record_sensitivity(
    results: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
) -> None:
    x = np.arange(len(CLASS_ORDER))
    width = 0.19
    figure, axis = plt.subplots(figsize=(11, 5.5))

    for model_index, model in enumerate(MODEL_NAMES):
        pooled = results[(model, "record")]["aggregate"][
            "pooled_or_repeat_weighted"
        ]
        values = [pooled["per_class"][label]["SEN"] or 0.0 for label in CLASS_ORDER]
        offset = (model_index - 1.5) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )

    axis.set_xticks(x, CLASS_ORDER)
    axis.set_xlabel("Heartbeat class")
    axis.set_ylabel("Repeat-weighted sensitivity (%)")
    axis.set_title("Record-based per-class sensitivity")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def normalized_confusion(result: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(
        result["aggregate"]["pooled_or_repeat_weighted"]["confusion_matrix"],
        dtype=float,
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def plot_record_confusions(
    baseline: dict[str, Any],
    extension: dict[str, Any],
    extension_label: str,
    output_path: Path,
) -> None:
    matrices = [normalized_confusion(baseline), normalized_confusion(extension)]
    titles = ["Baseline", extension_label]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for axis, matrix, title in zip(axes, matrices, titles, strict=True):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(title)
        axis.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        axis.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        for row in range(len(CLASS_ORDER)):
            for column in range(len(CLASS_ORDER)):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.55 else "black",
                )

    figure.colorbar(image, ax=axes, shrink=0.85, label="Row-normalized proportion")
    figure.suptitle("Record-based repeat-weighted confusion matrices (%)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def markdown_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Scheme | Model | Overall accuracy | Balanced accuracy | Macro-F1 | Paper SEN | Paper SPE | Paper ACC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['scheme']} | {row['model_label']} | "
            f"{row['fold_mean_overall_accuracy']:.2f} | "
            f"{row['fold_mean_balanced_accuracy']:.2f} | "
            f"{row['fold_mean_macro_F1']:.2f} | "
            f"{row['fold_mean_paper_SEN']:.2f} | "
            f"{row['fold_mean_paper_SPE']:.2f} | "
            f"{row['fold_mean_paper_ACC']:.2f} |"
        )
    return "\n".join(lines)


def create_report(
    results: dict[tuple[str, str], dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    output_path: Path,
    rr_source: str,
    weighting: str,
) -> None:
    record_extensions = [
        row
        for row in summary_rows
        if row["scheme"] == "record" and row["model"] in ("rr", "weighted_rr")
    ]
    best = max(record_extensions, key=lambda row: row["fold_mean_macro_F1"])
    baseline = next(
        row
        for row in summary_rows
        if row["scheme"] == "record" and row["model"] == "baseline"
    )

    feature_summary = (
        json.loads(FEATURE_SUMMARY_PATH.read_text(encoding="utf-8"))
        if FEATURE_SUMMARY_PATH.exists()
        else {}
    )
    split_summary = (
        json.loads(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
        if SPLIT_SUMMARY_PATH.exists()
        else {}
    )

    best_delta = best["fold_mean_macro_F1"] - baseline["fold_mean_macro_F1"]
    paired_best = [
        row
        for row in delta_rows
        if row["scheme"] == "record"
        and row["model"] == best["model"]
        and row["metric"] == "macro_F1"
    ][0]

    report = f"""# Final ECG classification results

## Experiment configuration

- Heartbeats: {feature_summary.get('number_of_beats', 107049):,}
- Samples per heartbeat: {feature_summary.get('samples_per_beat', 252)}
- Wavelet: bior6.8, level 8, symmetric extension
- Retained wavelet coefficients: cD1-cD8 ({feature_summary.get('wavelet_features_per_beat', 365)} values)
- PCA components: 12, fitted on each training fold only
- SVM: RBF, C=10, gamma=0.1, one-versus-one
- RR source: `{rr_source}`
- Class weighting: `{weighting}`, calculated from each training fold only
- Beat evaluation: shuffled stratified 10-fold split
- Record evaluation: ten predefined overlapping paper holdouts

## Fold-mean results (%)

{markdown_table(summary_rows)}

## Main extension result

The best RR-based record model by fold-mean macro-F1 was **{best['model_label']}**.
Its record-based macro-F1 was **{best['fold_mean_macro_F1']:.2f}%**, compared with
**{baseline['fold_mean_macro_F1']:.2f}%** for the morphology-only baseline, a
change of **{best_delta:+.2f} percentage points**. The paired fold comparison
improved in **{paired_best['folds_improved']} of 10** holdouts and worsened in
**{paired_best['folds_worsened']} of 10**.

The primary scientific comparison should emphasize record-based balanced
accuracy and macro-F1 rather than accuracy alone because class N dominates the
dataset.

## Figures

- [Beat-based model comparison](figures/beat_model_comparison.png)
- [Record-based model comparison](figures/record_model_comparison.png)
- [Record-based per-class sensitivity](figures/record_per_class_sensitivity.png)
- [Record-based confusion matrices](figures/record_confusion_comparison.png)

## Interpretation cautions

1. The beat-based scheme shares records between training and testing and is
   therefore optimistic for unseen-patient performance.
2. The paper record experiments overlap across holdouts. Only
   {split_summary.get('record_based', {}).get('unique_test_records', 32)} of 48
   records appear in a test holdout, so pooled results are repeat-weighted.
   Fold means are used for the main comparisons.
3. The paper reports 373 wavelet candidates, while the reproducible stated
   252-sample bior6.8 decomposition produces 365. No speculative padding was
   introduced.
4. Beat-only RR intervals exclude non-beat annotation markers. This differs
   from the partner-compatible reference RR matrix and is the primary extension.
5. `post_RR` requires observing the following beat. These results represent
   offline classification, not zero-latency real-time prediction.
6. These experiments use one database and do not constitute clinical
   validation.
"""
    output_path.write_text(report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rr-source",
        choices=("beat_only", "reference"),
        default="beat_only",
    )
    parser.add_argument(
        "--weighting",
        choices=("balanced", "partner"),
        default="balanced",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        results = load_completed_results(
            args.input_dir,
            rr_source=args.rr_source,
            weighting=args.weighting,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    summary_rows = build_summary_rows(results)
    delta_rows = paired_delta_rows(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "model_comparison.csv", summary_rows)
    write_csv(args.output_dir / "paired_fold_changes.csv", delta_rows)

    plot_scheme_comparison(
        summary_rows,
        "beat",
        figure_dir / "beat_model_comparison.png",
    )
    plot_scheme_comparison(
        summary_rows,
        "record",
        figure_dir / "record_model_comparison.png",
    )
    plot_record_sensitivity(
        results,
        figure_dir / "record_per_class_sensitivity.png",
    )

    best_model = max(
        ("rr", "weighted_rr"),
        key=lambda model: results[(model, "record")]["aggregate"][
            "fold_mean_standard_deviation"
        ]["macro_F1"]["mean"],
    )
    plot_record_confusions(
        results[("baseline", "record")],
        results[(best_model, "record")],
        MODEL_LABELS[best_model],
        figure_dir / "record_confusion_comparison.png",
    )

    report_path = args.output_dir / "FINAL_RESULTS.md"
    create_report(
        results,
        summary_rows,
        delta_rows,
        report_path,
        rr_source=args.rr_source,
        weighting=args.weighting,
    )

    print("Final results package created.")
    print(f"Comparison table: {args.output_dir / 'model_comparison.csv'}")
    print(f"Paired changes: {args.output_dir / 'paired_fold_changes.csv'}")
    print(f"Figures: {figure_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
