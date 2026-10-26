"""Regenerate ``examples/sample.png``: a small synthetic test image.

Simple shapes (filled rectangle, circle, diagonal line, outlined triangle) on
a dark background with a gentle horizontal gradient, so the pipeline has clear
edges in several orientations. Deterministic: no randomness.

Usage::

    python examples/make_sample_image.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 256, 192


def make_sample_image() -> np.ndarray:
    """Return the sample image as an ``(H, W, 3)`` BGR ``uint8`` array."""
    ramp = np.linspace(20, 60, WIDTH, dtype=np.float64)
    img = np.repeat(np.tile(ramp, (HEIGHT, 1))[..., None], 3, axis=2)
    img = img.astype(np.uint8)

    # Colours are BGR (OpenCV convention).
    cv2.rectangle(img, (24, 24), (104, 96), (40, 160, 230), thickness=-1)
    cv2.circle(img, (176, 72), 40, (220, 120, 40), thickness=-1)
    cv2.line(img, (16, 176), (240, 120), (255, 255, 255), thickness=3)
    triangle = np.array([[60, 168], [110, 112], [150, 168]], dtype=np.int32)
    cv2.polylines(img, [triangle], isClosed=True, color=(60, 220, 60), thickness=2)
    return img


def main() -> None:
    default = Path(__file__).resolve().parent / "sample.png"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not cv2.imwrite(str(out), make_sample_image()):
        raise SystemExit(f"failed to write {out}")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
