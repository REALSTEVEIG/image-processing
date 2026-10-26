"""Tests for Sobel, non-maximum suppression and the Canny wrapper."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imgfeatures import (
    canny_opencv,
    gradient_magnitude_direction,
    non_maximum_suppression,
    sobel_edges,
    sobel_gradients,
)


class TestSobel:
    def test_vertical_step_gives_horizontal_gradient(
        self, vertical_step: np.ndarray
    ) -> None:
        gx, gy = sobel_gradients(vertical_step)
        np.testing.assert_allclose(gy, 0.0, atol=1e-12)
        # Dark-to-bright going right -> positive gx, peak 4 at the edge.
        assert gx[:, 4].min() == pytest.approx(4.0)
        assert gx[:, 5].min() == pytest.approx(4.0)
        np.testing.assert_allclose(gx[:, :3], 0.0)
        np.testing.assert_allclose(gx[:, 7:], 0.0)

    def test_horizontal_step_gives_vertical_gradient(
        self, horizontal_step: np.ndarray
    ) -> None:
        gx, gy = sobel_gradients(horizontal_step)
        np.testing.assert_allclose(gx, 0.0, atol=1e-12)
        assert gy[4, :].min() == pytest.approx(4.0)

    def test_constant_image_gives_zero_gradient(self) -> None:
        gx, gy = sobel_gradients(np.full((8, 8), 123.0))
        np.testing.assert_allclose(gx, 0.0)
        np.testing.assert_allclose(gy, 0.0)

    def test_linear_ramp_has_constant_gradient(self) -> None:
        img = np.tile(np.arange(10, dtype=np.float64), (6, 1))  # +1 per column
        gx, _ = sobel_gradients(img)
        # Sobel x on a unit ramp = 8 (weights sum 4, central difference 2).
        np.testing.assert_allclose(gx[:, 1:-1], 8.0)

    @pytest.mark.parametrize(
        ("padding", "border"),
        [("reflect", cv2.BORDER_REFLECT_101),
         ("edge", cv2.BORDER_REPLICATE),
         ("constant", cv2.BORDER_CONSTANT)],
    )
    def test_matches_cv2_sobel(
        self, random_image: np.ndarray, padding: str, border: int
    ) -> None:
        img = random_image.astype(np.float64)
        gx, gy = sobel_gradients(img, padding=padding)
        ref_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=border)
        ref_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=border)
        np.testing.assert_allclose(gx, ref_x, atol=1e-8)
        np.testing.assert_allclose(gy, ref_y, atol=1e-8)

    def test_shapes_and_dtype(self, random_image: np.ndarray) -> None:
        gx, gy = sobel_gradients(random_image)
        assert gx.shape == gy.shape == random_image.shape
        assert gx.dtype == gy.dtype == np.float64

    def test_valid_padding_rejected(self) -> None:
        with pytest.raises(ValueError):
            sobel_gradients(np.ones((5, 5)), padding="valid")

    def test_color_rejected(self) -> None:
        with pytest.raises(ValueError):
            sobel_gradients(np.ones((5, 5, 3)))


class TestMagnitudeDirection:
    def test_direction_of_vertical_step_is_zero(
        self, vertical_step: np.ndarray
    ) -> None:
        mag, ang = gradient_magnitude_direction(vertical_step)
        edge = mag > 0
        np.testing.assert_allclose(ang[edge], 0.0)

    def test_direction_of_horizontal_step_is_half_pi(
        self, horizontal_step: np.ndarray
    ) -> None:
        mag, ang = gradient_magnitude_direction(horizontal_step)
        np.testing.assert_allclose(ang[mag > 0], np.pi / 2)

    def test_reversed_step_points_opposite(self, vertical_step: np.ndarray) -> None:
        mag, ang = gradient_magnitude_direction(1.0 - vertical_step)
        np.testing.assert_allclose(np.abs(ang[mag > 0]), np.pi)

    def test_magnitude_matches_cv2(self, random_image: np.ndarray) -> None:
        img = random_image.astype(np.float64)
        mag, ang = gradient_magnitude_direction(img)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        np.testing.assert_allclose(mag, cv2.magnitude(gx, gy), atol=1e-8)
        assert np.all((ang >= -np.pi) & (ang <= np.pi))
        assert np.all(mag >= 0)


class TestNonMaximumSuppression:
    def test_thins_step_edge(self) -> None:
        img = np.zeros((9, 9))
        img[:, 4:] = 1.0
        mag, ang = gradient_magnitude_direction(img)
        assert np.count_nonzero(mag[4]) == 2  # two-pixel-wide response
        thin = non_maximum_suppression(mag, ang)
        # Ties are kept (>=), so a symmetric edge may keep both pixels, but
        # nothing away from the edge survives.
        assert np.count_nonzero(thin[4]) <= 2
        np.testing.assert_array_equal(thin[:, :2], 0.0)

    def test_ridge_keeps_single_peak(self) -> None:
        mag = np.tile(np.array([0.0, 1.0, 3.0, 1.0, 0.0]), (5, 1))
        ang = np.zeros_like(mag)  # gradients point along x
        out = non_maximum_suppression(mag, ang)
        np.testing.assert_array_equal(out[:, 2], 3.0)
        np.testing.assert_array_equal(np.delete(out, 2, axis=1), 0.0)

    def test_diagonal_direction(self) -> None:
        # Ridge along the main diagonal; its cross-section runs along the
        # 135-degree axis, i.e. through the k=-2 and k=+2 diagonals.
        mag = np.eye(7) * 2.0 + (np.eye(7, k=2) + np.eye(7, k=-2)) * 1.0
        ang = np.full_like(mag, np.deg2rad(135.0))
        out = non_maximum_suppression(mag, ang)
        np.testing.assert_array_equal(np.diag(out), 2.0)
        np.testing.assert_array_equal(np.diag(out, k=2), 0.0)
        np.testing.assert_array_equal(np.diag(out, k=-2), 0.0)

    def test_negative_angle_equivalent(self) -> None:
        mag = np.tile(np.array([0.0, 2.0, 5.0, 2.0, 0.0]), (3, 1))
        a = non_maximum_suppression(mag, np.zeros_like(mag))
        b = non_maximum_suppression(mag, np.full_like(mag, -np.pi))
        np.testing.assert_array_equal(a, b)

    def test_output_never_exceeds_input(self, random_image: np.ndarray) -> None:
        mag, ang = gradient_magnitude_direction(random_image)
        out = non_maximum_suppression(mag, ang)
        assert np.all((out == 0) | (out == mag))

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shapes"):
            non_maximum_suppression(np.ones((3, 3)), np.ones((3, 4)))


class TestSobelEdges:
    def test_square_outline(self, square_image: np.ndarray) -> None:
        edges = sobel_edges(square_image)
        assert edges.dtype == np.bool_
        assert edges[32, 16] or edges[32, 15]
        assert not edges[32, 32]  # interior is flat
        assert not edges[2, 2]  # background is flat

    def test_constant_image_has_no_edges(self) -> None:
        assert not sobel_edges(np.full((10, 10), 5.0)).any()

    def test_thinning_reduces_edge_count(self, square_image: np.ndarray) -> None:
        thick = sobel_edges(square_image, thin=False).sum()
        thin = sobel_edges(square_image, thin=True).sum()
        assert thin <= thick

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
    def test_bad_threshold_value(self, bad: float) -> None:
        with pytest.raises(ValueError):
            sobel_edges(np.ones((5, 5)), threshold=bad)

    @pytest.mark.parametrize("bad", ["0.5", None, True])
    def test_bad_threshold_type(self, bad: object) -> None:
        with pytest.raises(TypeError):
            sobel_edges(np.ones((5, 5)), threshold=bad)  # type: ignore[arg-type]


class TestCannyOpenCV:
    def test_detects_square_outline(self, square_image: np.ndarray) -> None:
        edges = canny_opencv(square_image)
        assert edges.dtype == np.bool_
        assert edges.shape == square_image.shape
        assert edges.any()
        assert not edges[32, 32]
        assert not edges[2, 2]

    def test_agrees_with_sobel_nms_location(self, square_image: np.ndarray) -> None:
        """Both detectors should fire within 1 px of the true boundary."""
        canny = canny_opencv(square_image)
        ours = sobel_edges(square_image)
        boundary = np.zeros_like(canny)
        boundary[14:50, 14:50] = True
        boundary[18:46, 18:46] = False
        assert canny[~boundary].sum() == 0
        assert ours[~boundary].sum() == 0

    def test_accepts_float_input(self) -> None:
        img = np.zeros((20, 20))
        img[:, 10:] = 0.8
        assert canny_opencv(img).any()

    def test_constant_image(self) -> None:
        assert not canny_opencv(np.full((10, 10), 3.0)).any()

    def test_low_above_high(self) -> None:
        with pytest.raises(ValueError):
            canny_opencv(np.ones((5, 5)), low_threshold=200, high_threshold=100)

    def test_negative_threshold(self) -> None:
        with pytest.raises(ValueError):
            canny_opencv(np.ones((5, 5)), low_threshold=-1)

    def test_threshold_type(self) -> None:
        with pytest.raises(TypeError):
            canny_opencv(np.ones((5, 5)), high_threshold="150")  # type: ignore[arg-type]
