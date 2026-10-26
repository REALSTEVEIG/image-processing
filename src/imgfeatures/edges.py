"""Edge detection: from-scratch Sobel and non-maximum suppression, plus an
OpenCV Canny wrapper used for comparison.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._validation import as_float_image
from .convolution import Padding, convolve2d
from .preprocessing import normalize_minmax

# Sobel kernels written in *correlation* form, i.e. as they are usually
# printed and as OpenCV applies them. SOBEL_X responds positively when
# intensity increases left-to-right; SOBEL_Y when it increases top-to-bottom
# (image rows grow downwards).
SOBEL_X: np.ndarray = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float64,
)
SOBEL_Y: np.ndarray = SOBEL_X.T.copy()


def sobel_gradients(
    image: np.ndarray,
    padding: Padding = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical Sobel derivatives.

    The correlation-form kernels are flipped before being passed to
    :func:`~imgfeatures.convolution.convolve2d` (which flips them again), so
    the result equals ``cv2.Sobel`` with the same border handling. With the
    default ``"reflect"`` padding it matches OpenCV's default border.

    Args:
        image: 2-D grayscale image.
        padding: Border mode forwarded to ``convolve2d``. ``"valid"`` is
            rejected because the gradients must align with the input pixels.

    Returns:
        ``(gx, gy)`` - ``float64`` arrays with the shape of ``image``.

    Raises:
        TypeError: If ``image`` is not a numeric NumPy array.
        ValueError: On invalid image shape or ``padding="valid"``.
    """
    img = as_float_image(image)
    if padding == "valid":
        raise ValueError("padding='valid' is not supported for gradients")
    gx = convolve2d(img, SOBEL_X[::-1, ::-1], padding=padding)
    gy = convolve2d(img, SOBEL_Y[::-1, ::-1], padding=padding)
    return gx, gy


def gradient_magnitude_direction(
    image: np.ndarray,
    padding: Padding = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """Return Sobel gradient magnitude and direction.

    Args:
        image: 2-D grayscale image.
        padding: Border mode, see :func:`sobel_gradients`.

    Returns:
        ``(magnitude, direction)`` where ``magnitude = sqrt(gx**2 + gy**2)``
        and ``direction = arctan2(gy, gx)`` in radians, in ``[-pi, pi]``.
        Angles follow image coordinates (positive y points down).

    Raises:
        TypeError, ValueError: As for :func:`sobel_gradients`.
    """
    gx, gy = sobel_gradients(image, padding=padding)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def non_maximum_suppression(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Thin edges by keeping only local maxima along the gradient direction.

    The direction is quantised to one of four axes (0, 45, 90, 135 degrees).
    A pixel survives if its magnitude is >= both neighbours along that axis;
    otherwise it is set to zero. Pixels outside the image count as zero.
    This is the thinning step of the Canny detector, done without loops.

    Args:
        magnitude: 2-D gradient magnitude (non-negative).
        direction: 2-D gradient direction in radians, same shape.

    Returns:
        ``float64`` array: ``magnitude`` with non-maxima zeroed.

    Raises:
        TypeError: If inputs are not numeric NumPy arrays.
        ValueError: If shapes differ or inputs are not 2-D / finite.
    """
    mag = as_float_image(magnitude, name="magnitude")
    ang = as_float_image(direction, name="direction")
    if mag.shape != ang.shape:
        raise ValueError(
            f"magnitude and direction shapes differ: {mag.shape} vs {ang.shape}"
        )

    # Fold to [0, 180) degrees - opposite directions share an axis - and
    # round to the nearest multiple of 45 degrees -> sector 0..3.
    deg = np.rad2deg(ang) % 180.0
    sector = np.round(deg / 45.0).astype(np.int64) % 4

    padded = np.pad(mag, 1, mode="constant")
    h, w = mag.shape

    def shifted(dr: int, dc: int) -> np.ndarray:
        """Magnitude of the neighbour at offset (dr, dc) for every pixel."""
        return padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]

    # (row, col) offsets for each sector. With y pointing down, a 45-degree
    # gradient points to the lower-right, so its neighbours are (+1, +1)
    # and (-1, -1).
    offsets = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros_like(mag, dtype=bool)
    for s, (dr, dc) in offsets.items():
        is_max = (mag >= shifted(dr, dc)) & (mag >= shifted(-dr, -dc))
        keep |= (sector == s) & is_max
    return np.where(keep, mag, 0.0)


def sobel_edges(
    image: np.ndarray,
    threshold: float = 0.2,
    thin: bool = True,
) -> np.ndarray:
    """Binary edge map from Sobel magnitude (optionally thinned by NMS).

    Args:
        image: 2-D grayscale image.
        threshold: Fraction in ``[0, 1]`` of the maximum magnitude above
            which a pixel is an edge.
        thin: Apply :func:`non_maximum_suppression` before thresholding.

    Returns:
        Boolean ``(H, W)`` mask. All ``False`` for a constant image.

    Raises:
        TypeError: If ``threshold`` is not a real number.
        ValueError: If ``threshold`` is outside ``[0, 1]``.
    """
    _check_fraction(threshold, "threshold")
    mag, ang = gradient_magnitude_direction(image)
    if thin:
        mag = non_maximum_suppression(mag, ang)
    peak = mag.max()
    if peak == 0:
        return np.zeros(mag.shape, dtype=bool)
    return mag > threshold * peak


def canny_opencv(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> np.ndarray:
    """Canny edge detection via OpenCV, for comparison with the NumPy code.

    The image is min-max scaled to ``uint8`` [0, 255] first so thresholds
    have the same meaning regardless of the input dtype or range.

    Args:
        image: 2-D grayscale image.
        low_threshold: Hysteresis lower threshold (>= 0).
        high_threshold: Hysteresis upper threshold (>= ``low_threshold``).

    Returns:
        Boolean ``(H, W)`` edge mask.

    Raises:
        TypeError: If thresholds are not real numbers.
        ValueError: If thresholds are negative or ``low > high``.
    """
    for value, name in ((low_threshold, "low_threshold"),
                        (high_threshold, "high_threshold")):
        _check_real(value, name)
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if low_threshold > high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be <= "
            f"high_threshold ({high_threshold})"
        )
    img_u8 = np.round(normalize_minmax(image) * 255.0).astype(np.uint8)
    return cv2.Canny(img_u8, float(low_threshold), float(high_threshold)) > 0


def _check_real(value: float, name: str) -> None:
    """Raise TypeError unless ``value`` is a real (non-bool) number."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _check_fraction(value: float, name: str) -> None:
    """Raise unless ``value`` is a real number in ``[0, 1]``."""
    _check_real(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
