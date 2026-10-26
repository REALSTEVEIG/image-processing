"""Tests for grayscale conversion and normalisation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imgfeatures import normalize_minmax, standardize, to_grayscale


class TestToGrayscale:
    def test_pure_channels_give_bt601_weights(self) -> None:
        img = np.zeros((1, 3, 3), dtype=np.float64)
        img[0, 0, 0] = 1.0  # red
        img[0, 1, 1] = 1.0  # green
        img[0, 2, 2] = 1.0  # blue
        np.testing.assert_allclose(to_grayscale(img)[0], [0.299, 0.587, 0.114])

    def test_white_stays_white(self) -> None:
        img = np.full((4, 5, 3), 255, dtype=np.uint8)
        np.testing.assert_allclose(to_grayscale(img), 255.0)

    def test_shape_and_dtype(self) -> None:
        out = to_grayscale(np.zeros((7, 9, 3), dtype=np.uint8))
        assert out.shape == (7, 9)
        assert out.dtype == np.float64

    def test_2d_passthrough_returns_copy(self) -> None:
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        out = to_grayscale(img)
        np.testing.assert_array_equal(out, img)
        out[0, 0] = 99.0
        assert img[0, 0] == 0.0

    def test_matches_opencv(self) -> None:
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
        # OpenCV rounds to uint8 using fixed-point weights, so allow 1 level.
        expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        np.testing.assert_allclose(to_grayscale(rgb), expected, atol=1.0)

    @pytest.mark.parametrize("shape", [(4, 4, 4), (4, 4, 1), (4,), (2, 2, 2, 3)])
    def test_bad_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            to_grayscale(np.zeros(shape))

    def test_non_array_raises(self) -> None:
        with pytest.raises(TypeError):
            to_grayscale([[1, 2], [3, 4]])  # type: ignore[arg-type]


class TestNormalizeMinMax:
    def test_range_is_zero_to_one(self, random_image: np.ndarray) -> None:
        out = normalize_minmax(random_image)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_known_values(self) -> None:
        img = np.array([[10, 20], [30, 50]], dtype=np.int32)
        np.testing.assert_allclose(normalize_minmax(img), [[0, 0.25], [0.5, 1.0]])

    def test_constant_image_gives_zeros(self) -> None:
        out = normalize_minmax(np.full((3, 3), 7.0))
        np.testing.assert_array_equal(out, 0.0)
        assert np.all(np.isfinite(out))

    def test_input_not_modified(self) -> None:
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        before = img.copy()
        normalize_minmax(img)
        np.testing.assert_array_equal(img, before)

    def test_negative_values(self) -> None:
        out = normalize_minmax(np.array([[-5.0, 0.0, 5.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]])


class TestStandardize:
    def test_zero_mean_unit_std(self, random_image: np.ndarray) -> None:
        out = standardize(random_image)
        assert out.mean() == pytest.approx(0.0, abs=1e-12)
        assert out.std() == pytest.approx(1.0)

    def test_constant_image_gives_zeros(self) -> None:
        out = standardize(np.full((4, 4), 3, dtype=np.uint8))
        np.testing.assert_array_equal(out, 0.0)

    def test_dtype_and_shape(self) -> None:
        out = standardize(np.arange(6, dtype=np.uint8).reshape(2, 3))
        assert out.shape == (2, 3)
        assert out.dtype == np.float64


@pytest.mark.parametrize("func", [normalize_minmax, standardize])
class TestNormalisationValidation:
    def test_rejects_color(self, func) -> None:
        with pytest.raises(ValueError):
            func(np.zeros((3, 3, 3)))

    def test_rejects_empty(self, func) -> None:
        with pytest.raises(ValueError):
            func(np.zeros((0, 3)))

    def test_rejects_nan(self, func) -> None:
        img = np.ones((2, 2))
        img[0, 0] = np.nan
        with pytest.raises(ValueError):
            func(img)

    def test_rejects_inf(self, func) -> None:
        img = np.ones((2, 2))
        img[0, 0] = np.inf
        with pytest.raises(ValueError):
            func(img)

    @pytest.mark.parametrize(
        "bad",
        [np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=complex),
         np.array([["a", "b"]]), None, 3.0],
    )
    def test_rejects_bad_types(self, func, bad) -> None:
        with pytest.raises(TypeError):
            func(bad)
