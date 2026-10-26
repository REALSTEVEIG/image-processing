"""Image preprocessing: grayscale conversion and intensity normalisation."""

from __future__ import annotations

import numpy as np

from ._validation import as_float_image

#: ITU-R BT.601 luma coefficients for (R, G, B).
BT601_WEIGHTS: np.ndarray = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using ITU-R BT.601 luminance.

    Each output pixel is ``0.299 * R + 0.587 * G + 0.114 * B``. The weights
    sum to 1, so the output keeps the input's intensity range (e.g. a
    ``uint8`` image stays within [0, 255]).

    Args:
        image: ``(H, W, 3)`` array in **RGB** channel order, or an
            ``(H, W)`` array, which is returned unchanged (as ``float64``).
            Note that ``cv2.imread`` returns BGR; reverse the channels first.

    Returns:
        ``(H, W)`` ``float64`` grayscale image.

    Raises:
        TypeError: If ``image`` is not a numeric NumPy array.
        ValueError: If ``image`` has an unsupported shape, is empty, or
            contains non-finite values.
    """
    img = as_float_image(image, allow_color=True)
    if img.ndim == 2:
        return img.copy()
    # Matrix-vector product over the channel axis: (H, W, 3) @ (3,) -> (H, W).
    return img @ BT601_WEIGHTS


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Linearly rescale intensities to the range [0, 1].

    Computes ``(x - min) / (max - min)``. A constant image has no range to
    stretch, so it is mapped to all zeros instead of dividing by zero.

    Args:
        image: 2-D grayscale image.

    Returns:
        ``float64`` array of the same shape with values in [0, 1].

    Raises:
        TypeError: If ``image`` is not a numeric NumPy array.
        ValueError: If ``image`` is not 2-D, is empty, or is non-finite.
    """
    img = as_float_image(image)
    lo, hi = img.min(), img.max()
    span = hi - lo
    if span == 0:
        return np.zeros_like(img)
    return (img - lo) / span


def standardize(image: np.ndarray) -> np.ndarray:
    """Standardise intensities to zero mean and unit variance (z-score).

    Computes ``(x - mean) / std``. A constant image has zero standard
    deviation, so it is mapped to all zeros (its deviation from the mean).

    Args:
        image: 2-D grayscale image.

    Returns:
        ``float64`` array of the same shape.

    Raises:
        TypeError: If ``image`` is not a numeric NumPy array.
        ValueError: If ``image`` is not 2-D, is empty, or is non-finite.
    """
    img = as_float_image(image)
    centered = img - img.mean()
    std = img.std()
    if std == 0:
        return np.zeros_like(img)
    return centered / std
