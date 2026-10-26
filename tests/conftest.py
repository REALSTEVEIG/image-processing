"""Shared synthetic test images."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def vertical_step() -> np.ndarray:
    """10x10 image: left half 0, right half 1 (a vertical edge)."""
    img = np.zeros((10, 10), dtype=np.float64)
    img[:, 5:] = 1.0
    return img


@pytest.fixture
def horizontal_step() -> np.ndarray:
    """10x10 image: top half 0, bottom half 1 (a horizontal edge)."""
    img = np.zeros((10, 10), dtype=np.float64)
    img[5:, :] = 1.0
    return img


@pytest.fixture
def random_image() -> np.ndarray:
    """Reproducible 32x40 random uint8 image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(32, 40), dtype=np.uint8)


@pytest.fixture
def square_image() -> np.ndarray:
    """64x64 image with a bright filled square in the middle."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    return img
