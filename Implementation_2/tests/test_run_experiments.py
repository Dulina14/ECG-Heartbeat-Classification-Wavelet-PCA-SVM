import numpy as np

from src.run_experiments import (
    calculate_class_weights,
    experiment_tag,
    fit_predict_experiment,
)


def test_balanced_class_weights_use_training_frequency() -> None:
    labels = np.array(["A", "A", "N", "N", "N", "N"])
    weights = calculate_class_weights(labels, method="balanced")

    assert weights == {"A": 1.5, "N": 0.75}


def test_partner_class_weights_preserve_implementation_one_convention() -> None:
    labels = np.array(["A", "A", "N", "N", "N", "N"])
    weights = calculate_class_weights(labels, method="partner")

    assert weights == {"A": 2.0, "N": 1.0}


def test_experiment_tag_records_rr_and_weighting_choices() -> None:
    assert (
        experiment_tag("weighted_rr", "record", "beat_only", "balanced")
        == "weighted_rr_beat_only_balanced_record"
    )


def test_weighted_rr_model_runs_end_to_end() -> None:
    rng = np.random.default_rng(8)
    labels = np.array(["A", "L", "N", "P", "R", "V"])
    centres = np.arange(6, dtype=float)[:, None] * np.ones((1, 5))
    training_wavelets = np.vstack(
        [centre + rng.normal(scale=0.1, size=(8, 5)) for centre in centres]
    )
    training_labels = np.repeat(labels, 8)
    test_wavelets = centres.copy()
    training_rr = rng.normal(size=(48, 6))
    test_rr = rng.normal(size=(6, 6))

    predictions, details = fit_predict_experiment(
        training_wavelets,
        training_labels,
        test_wavelets,
        training_rr,
        test_rr,
        model="weighted_rr",
        n_components=3,
        cache_size_mb=50,
    )

    assert predictions.shape == (6,)
    assert details["model_input_dimensions"] == 9
    assert details["scaler_used"] is True
    assert details["class_weights"] is not None
