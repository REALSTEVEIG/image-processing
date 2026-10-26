"""Shared input validation helpers.

Every public function in the package funnels its image argument through
:func:`as_float_image` so that error messages and accepted inputs are
consistent across modules.
"""

from __future__ import annotations

import numpy as np


def as_float_image(
    image: np.ndarray,
    *,
    name: str = "image",
    allow_color: bool = False,
) -> np.ndarray:
    """Validate an image array and return it as a ``float64`` copy-or-view.

    Args:
        image: Input array. Must be 2-D (H x W), or 3-D (H x W x 3) when
            ``allow_color`` is true.
        name: Argument name used in error messages.
        allow_color: Whether a 3-channel image is acceptable.

    Returns:
        The image as a ``float64`` array with the same shape.

    Raises:
        TypeError: If ``image`` is not a NumPy array or has a non-real,
            non-numeric dtype (e.g. bool, complex, object, string).
        ValueError: If the shape is unsupported, the image is empty, or it
            contains NaN/inf values.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(image).__name__}"
        )
    if image.dtype == np.bool_ or not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError(
            f"{name} must have an integer or floating dtype, got {image.dtype}"
        )

    if image.ndim == 2:
        pass
    elif allow_color and image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        expected = "(H, W) or (H, W, 3)" if allow_color else "(H, W)"
        raise ValueError(f"{name} must have shape {expected}, got {image.shape}")

    if image.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {image.shape}")

    out = image.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return out


def check_positive_int(value: int, name: str) -> int:
    """Validate that ``value`` is a positive Python/NumPy integer.

    Raises:
        TypeError: If ``value`` is not an integer (bools are rejected).
        ValueError: If ``value`` is less than 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)
