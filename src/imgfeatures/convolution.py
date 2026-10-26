"""Vectorised 2-D convolution implemented with NumPy."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._validation import as_float_image

Padding = Literal["constant", "reflect", "edge", "wrap", "valid"]

#: Supported padding modes. ``"valid"`` means no padding (smaller output).
PADDING_MODES: tuple[str, ...] = ("constant", "reflect", "edge", "wrap", "valid")


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: Padding = "reflect",
) -> np.ndarray:
    """Convolve a 2-D image with a 2-D kernel.

    This is true convolution: the kernel is flipped on both axes before
    being slid over the image (as opposed to cross-correlation, which is
    what ``cv2.filter2D`` computes).

    Implementation: the image is padded, ``sliding_window_view`` creates a
    zero-copy ``(H, W, kh, kw)`` view of every neighbourhood, and
    ``np.einsum`` takes the weighted sum of each window in one vectorised
    call instead of a Python loop over pixels.

    Args:
        image: 2-D grayscale image.
        kernel: 2-D kernel. For any padding mode other than ``"valid"`` both
            dimensions must be odd so the kernel has a well-defined centre.
        padding: How to handle borders:

            * ``"constant"`` - pad with zeros.
            * ``"reflect"`` - mirror without repeating the edge pixel
              (``d c b | a b c d | c b a``); matches OpenCV's default
              ``BORDER_REFLECT_101``.
            * ``"edge"`` - repeat the edge pixel (``BORDER_REPLICATE``).
            * ``"wrap"`` - periodic wrap-around.
            * ``"valid"`` - no padding; output shrinks to
              ``(H - kh + 1, W - kw + 1)``.

    Returns:
        ``float64`` array. Same shape as ``image`` unless ``padding`` is
        ``"valid"``.

    Raises:
        TypeError: If ``image``/``kernel`` are not numeric arrays or
            ``padding`` is not a string.
        ValueError: On wrong dimensionality, even-sized kernels with a
            "same" padding mode, an unknown padding mode, or a kernel larger
            than the image in ``"valid"`` mode.
    """
    img = as_float_image(image)
    ker = as_float_image(kernel, name="kernel")

    if not isinstance(padding, str):
        raise TypeError(f"padding must be a str, got {type(padding).__name__}")
    if padding not in PADDING_MODES:
        raise ValueError(
            f"padding must be one of {PADDING_MODES}, got {padding!r}"
        )

    kh, kw = ker.shape
    if padding == "valid":
        if kh > img.shape[0] or kw > img.shape[1]:
            raise ValueError(
                f"kernel {ker.shape} is larger than image {img.shape} "
                "in 'valid' mode"
            )
        padded = img
    else:
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError(
                f"kernel dimensions must be odd for padding={padding!r}, "
                f"got {ker.shape}"
            )
        pad = ((kh // 2, kh // 2), (kw // 2, kw // 2))
        if padding == "reflect" and (
            img.shape[0] <= kh // 2 or img.shape[1] <= kw // 2
        ):
            # NumPy's reflect would silently reflect multiple times; be
            # explicit instead of producing surprising results.
            raise ValueError(
                f"image {img.shape} is too small for 'reflect' padding "
                f"with kernel {ker.shape}"
            )
        padded = np.pad(img, pad, mode=padding)

    windows = sliding_window_view(padded, (kh, kw))  # (H', W', kh, kw) view
    flipped = ker[::-1, ::-1]
    return np.einsum("ijkl,kl->ij", windows, flipped)
