import numpy as np

from src.run_baseline import (
    fit_predict_baseline,
    metrics_from_confusion,
    validate_fold,
)


def test_metrics_from_confusion() -> None:
    matrix = np.array([[8, 2], [1, 9]])
    metrics = metrics_from_confusion(matrix, classes=["A", "B"])

    assert metrics["overall_accuracy"] == 85.0
    assert metrics["per_class"]["A"]["SEN"] == 80.0
    assert metrics["per_class"]["A"]["SPE"] == 90.0
    assert metrics["paper_total"]["SEN"] == 85.0


def test_validate_fold_accepts_complete_disjoint_partition() -> None:
    validate_fold(
        train=np.array([0, 1, 2]),
        test=np.array([3, 4]),
        number_of_rows=5,
    )


def test_fit_predict_baseline_returns_one_prediction_per_test_row() -> None:
    rng = np.random.default_rng(4)
    class_centres = np.array(
        [
            [-5, 0, 0, 0, 0],
            [-3, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    labels = np.array(["A", "L", "N", "P", "R", "V"])

    training_features = np.vstack(
        [centre + rng.normal(scale=0.1, size=(10, 5)) for centre in class_centres]
    )
    training_labels = np.repeat(labels, 10)
    test_features = class_centres.copy()

    predictions, details = fit_predict_baseline(
        training_features,
        training_labels,
        test_features,
        n_components=3,
        cache_size_mb=50,
    )

    assert predictions.shape == (6,)
    assert 0.0 < details["pca_explained_variance"] <= 1.0
    assert details["support_vectors_total"] > 0
