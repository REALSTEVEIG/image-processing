"""Tests for feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from imgfeatures import (
    edge_density,
    feature_vector,
    feature_vector_length,
    gradient_statistics,
    hog,
    intensity_histogram,
)


class TestIntensityHistogram:
    def test_density_sums_to_one(self, random_image: np.ndarray) -> None:
        hist = intensity_histogram(random_image, bins=8)
        assert hist.shape == (8,)
        assert hist.sum() == pytest.approx(1.0)

    def test_counts(self) -> None:
        img = np.array([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
        hist = intensity_histogram(img, bins=2, density=False)
        np.testing.assert_array_equal(hist, [2, 6])

    def test_value_range(self) -> None:
        img = np.array([[0.0, 0.5], [0.5, 1.0]])
        hist = intensity_histogram(img, bins=4, value_range=(0.0, 1.0))
        np.testing.assert_allclose(hist, [0.25, 0.0, 0.5, 0.25])

    def test_constant_image(self) -> None:
        hist = intensity_histogram(np.full((3, 3), 4.0), bins=5)
        assert hist.sum() == pytest.approx(1.0)
        assert np.count_nonzero(hist) == 1

    @pytest.mark.parametrize("bins", [0, -3])
    def test_bad_bins_value(self, bins: int) -> None:
        with pytest.raises(ValueError):
            intensity_histogram(np.ones((3, 3)), bins=bins)

    @pytest.mark.parametrize("bins", [2.5, "8", True])
    def test_bad_bins_type(self, bins: object) -> None:
        with pytest.raises(TypeError):
            intensity_histogram(np.ones((3, 3)), bins=bins)  # type: ignore[arg-type]

    def test_bad_range(self) -> None:
        with pytest.raises(ValueError):
            intensity_histogram(np.ones((3, 3)), value_range=(1.0, 0.0))


class TestGradientStatistics:
    def test_constant_image_is_zero(self) -> None:
        np.testing.assert_allclose(gradient_statistics(np.ones((8, 8))), 0.0)

    def test_values_on_step(self, vertical_step: np.ndarray) -> None:
        mean, std, peak, p90 = gradient_statistics(vertical_step)
        assert peak == pytest.approx(4.0)
        assert mean == pytest.approx(4.0 * 2 / 10)  # 2 of 10 columns are 4
        assert std > 0
        assert 0 <= p90 <= peak

    def test_shape(self, random_image: np.ndarray) -> None:
        stats = gradient_statistics(random_image)
        assert stats.shape == (4,)
        assert stats.dtype == np.float64


class TestEdgeDensity:
    def test_constant_is_zero(self) -> None:
        assert edge_density(np.zeros((10, 10))) == 0.0

    def test_in_unit_interval(self, random_image: np.ndarray) -> None:
        assert 0.0 <= edge_density(random_image) <= 1.0

    def test_more_edges_more_density(self) -> None:
        one = np.zeros((32, 32))
        one[:, 16:] = 1.0
        stripes = np.tile(np.repeat([0.0, 1.0], 4), (32, 4))
        assert edge_density(stripes) > edge_density(one)


class TestHOG:
    def test_shape(self) -> None:
        out = hog(np.zeros((35, 50)), cell_size=8, n_bins=9)
        assert out.shape == (4, 6, 9)  # trailing pixels dropped

    def test_vertical_edge_votes_bin_zero(self) -> None:
        img = np.zeros((16, 16))
        img[:, 8:] = 1.0
        out = hog(img, cell_size=8, n_bins=9)
        # Every cell touching the edge puts all its weight in the 0-degree bin.
        for cell in out.reshape(-1, 9):
            if cell.any():
                assert np.argmax(cell) == 0
                assert cell[0] == pytest.approx(1.0)

    def test_horizontal_edge_votes_90_degrees(self) -> None:
        img = np.zeros((16, 16))
        img[8:, :] = 1.0
        out = hog(img, cell_size=8, n_bins=9)  # bins of 20 deg; 90 -> 4.5
        cell = out[0, 0]
        assert cell[4] == pytest.approx(cell[5])
        assert cell[4] + cell[5] == pytest.approx(cell.sum())

    def test_unsigned_orientation(self, vertical_step: np.ndarray) -> None:
        np.testing.assert_allclose(
            hog(vertical_step, cell_size=5), hog(1.0 - vertical_step, cell_size=5)
        )

    def test_cells_are_l2_normalised(self, random_image: np.ndarray) -> None:
        out = hog(random_image, cell_size=8)
        np.testing.assert_allclose(np.linalg.norm(out, axis=2), 1.0)

    def test_constant_image_is_zero_without_nan(self) -> None:
        out = hog(np.ones((16, 16)))
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out, 0.0)

    def test_vote_wraps_around(self) -> None:
        """An angle near 180 deg splits between the last and the first bin."""
        rows, cols = np.mgrid[0:24, 0:24].astype(np.float64)
        img = cols - 0.18 * rows  # gradient approx -10 deg -> 170 deg unsigned
        cell = hog(img, cell_size=8, n_bins=9)[1, 1]  # interior: no border
        assert cell[8] > 0 and cell[0] > 0
        assert cell[8] + cell[0] == pytest.approx(cell.sum())

    def test_image_smaller_than_cell(self) -> None:
        with pytest.raises(ValueError, match="cell"):
            hog(np.ones((4, 20)), cell_size=8)

    @pytest.mark.parametrize("kwargs", [{"cell_size": 0}, {"n_bins": 0}])
    def test_bad_params(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            hog(np.ones((16, 16)), **kwargs)

    def test_bad_param_type(self) -> None:
        with pytest.raises(TypeError):
            hog(np.ones((16, 16)), cell_size=8.0)  # type: ignore[arg-type]


class TestFeatureVector:
    def test_length_matches_helper(self, square_image: np.ndarray) -> None:
        vec = feature_vector(square_image)
        assert vec.ndim == 1
        assert vec.dtype == np.float64
        assert vec.shape == (feature_vector_length(square_image.shape),)

    def test_accepts_rgb(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, 16:] = (255, 0, 0)
        vec = feature_vector(rgb, cell_size=8)
        assert vec.shape == (feature_vector_length((32, 32), cell_size=8),)
        assert np.all(np.isfinite(vec))

    def test_scale_invariant(self, square_image: np.ndarray) -> None:
        """Min-max normalisation makes the vector independent of scale."""
        a = feature_vector(square_image)
        b = feature_vector(square_image.astype(np.float64) / 255.0)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_layout(self, square_image: np.ndarray) -> None:
        vec = feature_vector(square_image, hist_bins=4)
        assert vec[:4].sum() == pytest.approx(1.0)  # histogram first
        assert 0.0 <= vec[4 + 4] <= 1.0  # edge density after 4 gradient stats

    def test_constant_image_is_finite(self) -> None:
        assert np.all(np.isfinite(feature_vector(np.full((16, 16), 9.0))))

    def test_bad_edge_threshold(self) -> None:
        with pytest.raises(ValueError):
            feature_vector(np.ones((16, 16)), edge_threshold=2.0)

    def test_bad_input_type(self) -> None:
        with pytest.raises(TypeError):
            feature_vector("not an image")  # type: ignore[arg-type]
