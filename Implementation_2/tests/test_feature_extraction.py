import warnings

import numpy as np
import pywt

from src.extract_features import (
    EXPECTED_WAVELET_FEATURES,
    rr_features_from_sequence,
    wavelet_detail_vector,
)


def test_wavelet_feature_length_and_coefficient_lengths() -> None:
    beat = np.linspace(-1.0, 1.0, 252)

    features, lengths = wavelet_detail_vector(beat)

    assert lengths == [17, 17, 18, 20, 24, 31, 46, 75, 134]
    assert features.shape == (EXPECTED_WAVELET_FEATURES,)
    assert np.isfinite(features).all()


def test_wavelet_features_are_ordered_cd1_to_cd8() -> None:
    beat = np.sin(np.linspace(0.0, 8.0 * np.pi, 252))
    actual, _ = wavelet_detail_vector(beat)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coefficients = pywt.wavedec(
            beat,
            wavelet="bior6.8",
            level=8,
            mode="symmetric",
        )

    expected = np.concatenate(coefficients[:0:-1])
    np.testing.assert_allclose(actual, expected)


def test_rr_features_use_selected_sequence_positions() -> None:
    samples = np.array([0, 360, 900, 1260])
    selected_positions = np.array([1, 2])

    features = rr_features_from_sequence(samples, selected_positions)

    assert features.shape == (2, 6)
    np.testing.assert_allclose(features[:, 0], [1.0, 1.5])
    np.testing.assert_allclose(features[:, 1], [1.5, 1.0])
    assert np.isfinite(features).all()
