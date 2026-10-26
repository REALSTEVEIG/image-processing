"""Tests for the NumPy 2-D convolution."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imgfeatures import convolve2d


def naive_convolve_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reference implementation: explicit loops, valid region only."""
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    out = np.zeros((image.shape[0] - kh + 1, image.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * flipped)
    return out


def test_identity_kernel(random_image: np.ndarray) -> None:
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0
    np.testing.assert_allclose(convolve2d(random_image, kernel), random_image)


def test_matches_naive_loop_valid_mode() -> None:
    rng = np.random.default_rng(42)
    img = rng.normal(size=(12, 15))
    kernel = rng.normal(size=(3, 5))
    np.testing.assert_allclose(
        convolve2d(img, kernel, padding="valid"),
        naive_convolve_valid(img, kernel),
    )


def test_kernel_is_flipped() -> None:
    """A single impulse convolved with a kernel reproduces the kernel."""
    img = np.zeros((5, 5))
    img[2, 2] = 1.0
    kernel = np.arange(9, dtype=np.float64).reshape(3, 3)
    out = convolve2d(img, kernel, padding="constant")
    np.testing.assert_allclose(out[1:4, 1:4], kernel)


@pytest.mark.parametrize(
    ("mode", "border"),
    [
        ("reflect", cv2.BORDER_REFLECT_101),
        ("edge", cv2.BORDER_REPLICATE),
        ("constant", cv2.BORDER_CONSTANT),
    ],
)
def test_matches_opencv_filter2d(
    random_image: np.ndarray, mode: str, border: int
) -> None:
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(5, 5))
    img = random_image.astype(np.float64)
    # filter2D computes correlation, so give it the flipped kernel.
    expected = cv2.filter2D(img, cv2.CV_64F, kernel[::-1, ::-1].copy(),
                            borderType=border)
    np.testing.assert_allclose(convolve2d(img, kernel, padding=mode), expected,
                               atol=1e-8)


def test_wrap_padding() -> None:
    img = np.zeros((4, 4))
    img[0, 0] = 1.0
    kernel = np.ones((3, 3))
    out = convolve2d(img, kernel, padding="wrap")
    # The impulse spreads to its wrapped neighbours, e.g. the opposite corner.
    assert out[3, 3] == 1.0
    assert out.sum() == pytest.approx(9.0)


@pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap"])
def test_same_shape_for_padded_modes(random_image: np.ndarray, mode: str) -> None:
    out = convolve2d(random_image, np.ones((3, 3)), padding=mode)
    assert out.shape == random_image.shape
    assert out.dtype == np.float64


def test_valid_mode_shape() -> None:
    out = convolve2d(np.ones((10, 8)), np.ones((3, 5)), padding="valid")
    assert out.shape == (8, 4)


def test_box_filter_on_constant_image() -> None:
    out = convolve2d(np.full((6, 6), 2.0), np.full((3, 3), 1 / 9))
    np.testing.assert_allclose(out, 2.0)


def test_1x1_kernel_scales() -> None:
    img = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.testing.assert_allclose(convolve2d(img, np.array([[3.0]])), 3 * img)


class TestValidation:
    def test_even_kernel_rejected_for_same_modes(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            convolve2d(np.ones((5, 5)), np.ones((2, 2)))

    def test_even_kernel_allowed_in_valid_mode(self) -> None:
        out = convolve2d(np.ones((5, 5)), np.ones((2, 2)), padding="valid")
        np.testing.assert_allclose(out, 4.0)

    def test_unknown_padding(self) -> None:
        with pytest.raises(ValueError, match="padding"):
            convolve2d(np.ones((5, 5)), np.ones((3, 3)), padding="mirror")  # type: ignore[arg-type]

    def test_non_string_padding(self) -> None:
        with pytest.raises(TypeError):
            convolve2d(np.ones((5, 5)), np.ones((3, 3)), padding=0)  # type: ignore[arg-type]

    def test_kernel_larger_than_image_valid(self) -> None:
        with pytest.raises(ValueError, match="larger"):
            convolve2d(np.ones((2, 2)), np.ones((3, 3)), padding="valid")

    def test_image_too_small_for_reflect(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            convolve2d(np.ones((1, 5)), np.ones((3, 3)), padding="reflect")

    def test_kernel_must_be_2d(self) -> None:
        with pytest.raises(ValueError):
            convolve2d(np.ones((5, 5)), np.ones(3))

    def test_kernel_must_be_array(self) -> None:
        with pytest.raises(TypeError):
            convolve2d(np.ones((5, 5)), [[1.0]])  # type: ignore[arg-type]

    def test_color_image_rejected(self) -> None:
        with pytest.raises(ValueError):
            convolve2d(np.ones((5, 5, 3)), np.ones((3, 3)))
