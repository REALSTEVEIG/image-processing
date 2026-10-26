"""Hand-crafted image features built on the from-scratch primitives."""

from __future__ import annotations

import numpy as np

from ._validation import as_float_image, check_positive_int
from .edges import _check_fraction, gradient_magnitude_direction, sobel_edges
from .preprocessing import normalize_minmax, to_grayscale


def intensity_histogram(
    image: np.ndarray,
    bins: int = 16,
    value_range: tuple[float, float] | None = None,
    density: bool = True,
) -> np.ndarray:
    """Histogram of pixel intensities.

    Args:
        image: 2-D grayscale image.
        bins: Number of equal-width bins (>= 1).
        value_range: ``(low, high)`` range covered by the bins. Defaults to
            the image's own ``(min, max)``. Values outside are ignored.
        density: If true, return the fraction of pixels per bin (sums to 1
            when every pixel falls inside the range); otherwise raw counts.

    Returns:
        1-D ``float64`` array of length ``bins``.

    Raises:
        TypeError: If ``image`` or ``bins`` has the wrong type.
        ValueError: If ``bins < 1`` or ``value_range`` is not increasing.
    """
    img = as_float_image(image)
    bins = check_positive_int(bins, "bins")
    if value_range is None:
        lo, hi = float(img.min()), float(img.max())
        if lo == hi:  # constant image: give the single value a non-empty bin
            lo, hi = lo - 0.5, hi + 0.5
    else:
        if len(value_range) != 2:
            raise ValueError("value_range must be a (low, high) pair")
        lo, hi = float(value_range[0]), float(value_range[1])
        if not lo < hi:
            raise ValueError(f"value_range must satisfy low < high, got {value_range}")

    counts, _ = np.histogram(img, bins=bins, range=(lo, hi))
    counts = counts.astype(np.float64)
    if density:
        counts /= img.size
    return counts


def gradient_statistics(image: np.ndarray) -> np.ndarray:
    """Summary statistics of the Sobel gradient magnitude.

    Args:
        image: 2-D grayscale image.

    Returns:
        ``float64`` array ``[mean, std, max, 90th percentile]``.
    """
    magnitude, _ = gradient_magnitude_direction(image)
    return np.array(
        [
            magnitude.mean(),
            magnitude.std(),
            magnitude.max(),
            np.percentile(magnitude, 90),
        ],
        dtype=np.float64,
    )


def edge_density(image: np.ndarray, threshold: float = 0.2) -> float:
    """Fraction of pixels classified as edges by thinned Sobel.

    Args:
        image: 2-D grayscale image.
        threshold: Relative magnitude threshold in ``[0, 1]``; see
            :func:`~imgfeatures.edges.sobel_edges`.

    Returns:
        A float in ``[0, 1]``. ``0.0`` for a constant image.
    """
    return float(sobel_edges(image, threshold=threshold, thin=True).mean())


def hog(
    image: np.ndarray,
    cell_size: int = 8,
    n_bins: int = 9,
) -> np.ndarray:
    """Simplified Histogram of Oriented Gradients.

    Steps:

    1. Sobel gradients give a magnitude and an *unsigned* orientation in
       ``[0, 180)`` degrees for every pixel (a dark-to-light edge and a
       light-to-dark edge at the same angle are treated alike).
    2. The image is split into non-overlapping ``cell_size x cell_size``
       cells; trailing rows/columns that do not fill a cell are dropped.
    3. Each pixel votes its magnitude into the two nearest orientation bins,
       split linearly by angular distance (reduces aliasing at bin edges).
    4. Each cell histogram is L2-normalised.

    Compared with Dalal & Triggs (2005) this omits overlapping block
    normalisation and Gaussian spatial weighting, which keeps it short.

    Args:
        image: 2-D grayscale image of at least ``cell_size x cell_size``.
        cell_size: Cell side length in pixels (>= 1).
        n_bins: Number of orientation bins spanning 180 degrees (>= 1).

    Returns:
        ``(H // cell_size, W // cell_size, n_bins)`` ``float64`` array.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If the image is smaller than one cell.
    """
    img = as_float_image(image)
    cell_size = check_positive_int(cell_size, "cell_size")
    n_bins = check_positive_int(n_bins, "n_bins")

    cells_y, cells_x = img.shape[0] // cell_size, img.shape[1] // cell_size
    if cells_y == 0 or cells_x == 0:
        raise ValueError(
            f"image {img.shape} is smaller than one {cell_size}x{cell_size} cell"
        )

    magnitude, direction = gradient_magnitude_direction(img)
    h, w = cells_y * cell_size, cells_x * cell_size
    magnitude, direction = magnitude[:h, :w], direction[:h, :w]

    # Orientation expressed in units of bins, e.g. 2.3 -> 70% bin 2, 30% bin 3.
    bin_width = 180.0 / n_bins
    pos = (np.rad2deg(direction) % 180.0) / bin_width
    lower = np.floor(pos).astype(np.int64) % n_bins
    upper = (lower + 1) % n_bins  # wraps: 175 deg also votes for the 0 bin
    frac = pos - np.floor(pos)

    # Cell index for every pixel, flattened to one id per cell.
    rows = np.arange(h) // cell_size
    cols = np.arange(w) // cell_size
    cell_id = rows[:, None] * cells_x + cols[None, :]

    # Accumulate all votes in one bincount over (cell, bin) pairs.
    n_slots = cells_y * cells_x * n_bins
    hist = np.bincount(
        (cell_id * n_bins + lower).ravel(),
        weights=(magnitude * (1.0 - frac)).ravel(),
        minlength=n_slots,
    )
    hist += np.bincount(
        (cell_id * n_bins + upper).ravel(),
        weights=(magnitude * frac).ravel(),
        minlength=n_slots,
    )
    hist = hist.reshape(cells_y, cells_x, n_bins)

    norms = np.linalg.norm(hist, axis=2, keepdims=True)
    return hist / (norms + 1e-12)


def feature_vector(
    image: np.ndarray,
    hist_bins: int = 16,
    cell_size: int = 8,
    hog_bins: int = 9,
    edge_threshold: float = 0.2,
) -> np.ndarray:
    """Concatenate all features into one 1-D vector.

    The image is converted to grayscale and min-max normalised first, so the
    features do not depend on the input dtype or overall brightness scale.

    Layout: ``[intensity histogram (hist_bins) | gradient stats (4) |
    edge density (1) | flattened HOG (cells_y * cells_x * hog_bins)]``.

    The HOG part depends on image size, so images should share a size (e.g.
    resize beforehand) if their vectors are to be compared or fed to a model.

    Args:
        image: ``(H, W)`` grayscale or ``(H, W, 3)`` RGB image.
        hist_bins: Bins for the intensity histogram.
        cell_size: HOG cell size in pixels.
        hog_bins: HOG orientation bins.
        edge_threshold: Relative threshold for edge density.

    Returns:
        1-D ``float64`` feature vector.
    """
    _check_fraction(edge_threshold, "edge_threshold")
    gray = normalize_minmax(to_grayscale(image))
    parts = [
        intensity_histogram(gray, bins=hist_bins, value_range=(0.0, 1.0)),
        gradient_statistics(gray),
        np.array([edge_density(gray, threshold=edge_threshold)]),
        hog(gray, cell_size=cell_size, n_bins=hog_bins).ravel(),
    ]
    return np.concatenate(parts)


def feature_vector_length(
    shape: tuple[int, int],
    hist_bins: int = 16,
    cell_size: int = 8,
    hog_bins: int = 9,
) -> int:
    """Length of :func:`feature_vector` output for an image of ``shape``."""
    h, w = shape[:2]
    return hist_bins + 4 + 1 + (h // cell_size) * (w // cell_size) * hog_bins
