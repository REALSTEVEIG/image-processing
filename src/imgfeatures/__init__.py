"""imgfeatures - from-scratch NumPy image preprocessing, edge detection and
feature extraction, with OpenCV used only for comparison.
"""

from .convolution import convolve2d
from .edges import (
    canny_opencv,
    gradient_magnitude_direction,
    non_maximum_suppression,
    sobel_edges,
    sobel_gradients,
)
from .features import (
    edge_density,
    feature_vector,
    feature_vector_length,
    gradient_statistics,
    hog,
    intensity_histogram,
)
from .preprocessing import normalize_minmax, standardize, to_grayscale

__version__ = "0.1.0"

__all__ = [
    "canny_opencv",
    "convolve2d",
    "edge_density",
    "feature_vector",
    "feature_vector_length",
    "gradient_magnitude_direction",
    "gradient_statistics",
    "hog",
    "intensity_histogram",
    "non_maximum_suppression",
    "normalize_minmax",
    "sobel_edges",
    "sobel_gradients",
    "standardize",
    "to_grayscale",
]
