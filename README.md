# imgfeatures

A small, self-contained Python package that implements classic image
preprocessing, edge detection and hand-crafted feature extraction **from
scratch in NumPy**. OpenCV is used only as a reference to validate the
from-scratch results (and for a Canny comparison).

The goal is to show the algorithms clearly, with vectorised NumPy code, full
type hints, input validation and a test suite that checks both mathematical
properties and agreement with OpenCV.

## Features

| Module | Function | What it does |
| --- | --- | --- |
| `preprocessing` | `to_grayscale` | RGB to grayscale with ITU-R BT.601 weights |
| | `normalize_minmax` | Rescale intensities to [0, 1] |
| | `standardize` | Z-score: zero mean, unit variance |
| `convolution` | `convolve2d` | Vectorised 2-D convolution with padding modes |
| `edges` | `sobel_gradients` | Sobel x/y derivatives |
| | `gradient_magnitude_direction` | Gradient magnitude and angle |
| | `non_maximum_suppression` | Edge thinning (Canny's NMS step) |
| | `sobel_edges` | Binary edge map from (thinned) Sobel |
| | `canny_opencv` | OpenCV Canny, for comparison |
| `features` | `intensity_histogram` | Intensity distribution |
| | `gradient_statistics` | Mean / std / max / 90th pct of gradient magnitude |
| | `edge_density` | Fraction of edge pixels |
| | `hog` | Simplified Histogram of Oriented Gradients |
| | `feature_vector` | All of the above concatenated into one vector |

## Installation

Requires Python 3.10+.

```bash
cd image-processing
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e ".[dev]"
```

## Usage

```python
import cv2
import numpy as np
import imgfeatures as imf

bgr = cv2.imread("photo.jpg")            # OpenCV loads images as BGR
rgb = bgr[..., ::-1]                      # our grayscale expects RGB

gray = imf.normalize_minmax(imf.to_grayscale(rgb))

gx, gy = imf.sobel_gradients(gray)
magnitude, direction = imf.gradient_magnitude_direction(gray)
thin = imf.non_maximum_suppression(magnitude, direction)

ours = imf.sobel_edges(gray, threshold=0.2)
reference = imf.canny_opencv(gray, low_threshold=50, high_threshold=150)

descriptor = imf.hog(gray, cell_size=8, n_bins=9)   # (H//8, W//8, 9)
vector = imf.feature_vector(rgb)                    # 1-D float64
print(vector.shape, imf.feature_vector_length(rgb.shape))
```

A synthetic example that needs no image file:

```python
img = np.zeros((64, 64))
img[16:48, 16:48] = 1.0                     # bright square
print(imf.edge_density(img))                # small fraction: only the outline
```

## Algorithms

Notation: the image is `H x W` pixels (`N = H·W`), the kernel is `k x k`,
`B` is the number of histogram bins.

### Grayscale conversion - O(N) time, O(N) space

Human eyes are most sensitive to green and least to blue, so a plain average
of R, G, B looks wrong. The BT.601 standard weights them
`0.299 R + 0.587 G + 0.114 B`. Implemented as one matrix-vector product over
the channel axis. 2-D input is passed through unchanged.

### Min-max normalisation - O(N) time, O(N) space

Stretches intensities linearly so the darkest pixel becomes 0 and the
brightest 1: `(x − min) / (max − min)`. Makes later thresholds independent of
the input's dtype (uint8 vs. float) and exposure. A constant image has no
range, so it is mapped to zeros rather than dividing by zero.

### Z-score standardisation - O(N) time, O(N) space

`(x − mean) / std`: centres intensities at 0 with unit spread, which is the
usual input scaling for statistical models. A constant image (std = 0) maps
to zeros.

### 2-D convolution - O(N·k²) time, O(N) extra space

Each output pixel is a weighted sum of its `k x k` neighbourhood, with the
kernel flipped (true convolution, as opposed to the correlation computed by
`cv2.filter2D`). Borders are handled by padding: `constant` (zeros),
`reflect` (mirror; OpenCV's default), `edge` (replicate), `wrap` (periodic),
or `valid` (no padding, smaller output).

Implementation: `sliding_window_view` builds a zero-copy `(H, W, k, k)` view
of all neighbourhoods and `np.einsum` reduces it in one vectorised call, so
there is no Python loop over pixels. The view itself costs O(1) memory; the
padded image and output are O(N). (An FFT-based convolution would be
O(N log N) and win for large kernels; for the 3x3 kernels used here, direct
convolution is simpler and faster.)

### Sobel gradients - O(N) time, O(N) space

Two 3x3 kernels estimate the horizontal and vertical derivative: a central
difference in one direction, smoothed with weights `[1, 2, 1]` in the other
to reduce noise. From `gx` and `gy`:

- magnitude `sqrt(gx² + gy²)` - how strong the edge is;
- direction `atan2(gy, gx)` - which way intensity increases.

Built on `convolve2d` (k = 3, so O(9N) = O(N)). Tests confirm the output
equals `cv2.Sobel` for three border modes.

### Non-maximum suppression - O(N) time, O(N) space

A blurred edge gives a gradient response several pixels wide. NMS keeps a
pixel only if it is at least as strong as both neighbours along its gradient
direction (quantised to 0°, 45°, 90°, 135°), leaving one-pixel-wide ridges.
Fully vectorised: one shifted comparison per direction.

### Canny (OpenCV) - O(N) time, O(N) space

Canny = smoothing + Sobel + NMS + hysteresis thresholding (strong edges, plus
weak edges connected to strong ones). It is wrapped from OpenCV to compare
against the from-scratch Sobel + NMS pipeline; tests check both detectors fire
only on the true boundary of a synthetic square.

### Intensity histogram - O(N + B) time, O(B) space

Counts how many pixels fall into each of `B` equal-width intensity bins;
optionally divided by `N` so it sums to 1. Captures overall brightness and
contrast, independent of where things are in the image.

### Gradient statistics - O(N) time, O(N) space

Mean, standard deviation, maximum and 90th percentile of Sobel magnitude:
a compact summary of how "busy" or textured the image is. (The percentile
uses a selection algorithm, average O(N).)

### Edge density - O(N) time, O(N) space

Fraction of pixels marked as edges by thinned Sobel. Low for smooth images,
high for cluttered or textured ones.

### Histogram of Oriented Gradients (simplified) - O(N + C·B) time, O(N + C·B) space

(`C` = number of cells.) Describes local shape by the distribution of edge
directions:

1. Compute gradient magnitude and *unsigned* orientation in [0°, 180°).
2. Split the image into non-overlapping `cell_size x cell_size` cells.
3. Each pixel adds its magnitude to the two nearest orientation bins,
   split linearly by angle (with wrap-around from 180° back to 0°), which
   avoids abrupt changes at bin boundaries.
4. L2-normalise each cell histogram, so the descriptor ignores contrast.

All votes are accumulated with a single `np.bincount` over (cell, bin)
indices instead of looping over cells. Compared with the original Dalal &
Triggs HOG, overlapping block normalisation and Gaussian weighting are
omitted to keep the code short.

### Combined feature vector - O(N) time, O(N) space

Converts to grayscale, min-max normalises, then concatenates
`[histogram | 4 gradient stats | edge density | flattened HOG]`. Because of
the normalisation, rescaling the input (e.g. uint8 vs. `/255` floats) gives
the same vector. HOG length depends on image size, so resize images to a
common size before comparing vectors; `feature_vector_length` computes the
expected length.

## Design decisions

- **From scratch first, OpenCV as an oracle.** Core algorithms are NumPy only.
  OpenCV appears in `canny_opencv` and in the tests, where it is the reference
  that the from-scratch Sobel, convolution and grayscale must match.
- **Vectorised, not looped.** Sliding-window views, `einsum`, shifted array
  comparisons and `bincount` replace per-pixel Python loops. A naive loop
  convolution exists only in the tests as a readable reference.
- **True convolution semantics.** `convolve2d` flips the kernel, as the
  mathematical definition requires. Sobel kernels are written in their
  familiar correlation form and pre-flipped, so results match `cv2.Sobel`.
- **`reflect` padding by default.** It matches OpenCV's default border and
  avoids false edges that zero padding creates at image borders.
- **Everything is `float64`.** Inputs of any integer/float dtype are converted
  once, avoiding uint8 overflow and keeping results comparable.
- **Fail early with clear errors.** A shared validator raises `TypeError` for
  non-arrays and non-numeric dtypes (bool, complex, strings) and `ValueError`
  for wrong shapes, empty images, NaN/inf, and invalid parameters.
- **Defined behaviour on degenerate input.** Constant images produce zeros
  (normalisation, gradients, HOG) rather than NaN from division by zero.
- **Small and flat.** Four modules, plain functions, no classes or config
  objects - the package is meant to be read.

## Project layout

```
image-processing/
├── pyproject.toml
├── README.md
├── src/imgfeatures/
│   ├── __init__.py        # public API
│   ├── _validation.py     # shared input checks
│   ├── preprocessing.py   # grayscale, normalisation
│   ├── convolution.py     # convolve2d
│   ├── edges.py           # Sobel, NMS, Canny wrapper
│   └── features.py        # histogram, gradient stats, edge density, HOG
└── tests/
    ├── conftest.py        # synthetic image fixtures
    ├── test_preprocessing.py
    ├── test_convolution.py
    ├── test_edges.py
    └── test_features.py
```

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests use small synthetic images with known answers (step edges, ramps,
impulses, constant images), compare against OpenCV (`cv2.Sobel`,
`cv2.filter2D`, `cv2.cvtColor`, `cv2.magnitude`) within tolerance, and check
shapes, dtypes, edge cases and that invalid input raises the right error.

Optional linting (configured in `pyproject.toml`):

```bash
pip install ruff
ruff check src tests
```
