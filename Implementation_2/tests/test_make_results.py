from src.make_results import paired_delta_rows, summary_row


def fake_result(model: str, scheme: str, values: list[float]) -> dict:
    folds = []
    for value in values:
        folds.append(
            {
                "metrics": {
                    "overall_accuracy": value,
                    "balanced_accuracy": value - 5,
                    "macro_F1": value - 10,
                }
            }
        )
    return {
        "model": model,
        "scheme": scheme,
        "folds": folds,
        "aggregate": {
            "fold_mean_standard_deviation": {
                "overall_accuracy": {"mean": 50.0, "standard_deviation": 2.0},
                "balanced_accuracy": {"mean": 40.0, "standard_deviation": 3.0},
                "macro_F1": {"mean": 30.0, "standard_deviation": 4.0},
                "paper_total_SEN": {"mean": 50.0, "standard_deviation": 2.0},
                "paper_total_SPE": {"mean": 90.0, "standard_deviation": 1.0},
                "paper_total_ACC": {"mean": 83.0, "standard_deviation": 1.0},
            },
            "pooled_or_repeat_weighted": {
                "overall_accuracy": 51.0,
                "balanced_accuracy": 41.0,
                "macro_F1": 31.0,
            },
        },
    }


def test_summary_row_uses_fold_means_for_primary_results() -> None:
    row = summary_row("baseline", "record", fake_result("baseline", "record", [1]))

    assert row["fold_mean_overall_accuracy"] == 50.0
    assert row["pooled_overall_accuracy"] == 51.0


def test_paired_delta_rows_compare_matching_folds() -> None:
    results = {}
    for scheme in ("beat", "record"):
        results[("baseline", scheme)] = fake_result("baseline", scheme, [10, 20])
        results[("weighted", scheme)] = fake_result("weighted", scheme, [11, 18])
        results[("rr", scheme)] = fake_result("rr", scheme, [13, 24])
        results[("weighted_rr", scheme)] = fake_result(
            "weighted_rr", scheme, [12, 22]
        )

    rows = paired_delta_rows(results)
    rr_record_overall = next(
        row
        for row in rows
        if row["scheme"] == "record"
        and row["model"] == "rr"
        and row["metric"] == "overall_accuracy"
    )

    assert rr_record_overall["mean_paired_change_percentage_points"] == 3.5
    assert rr_record_overall["folds_improved"] == 2
