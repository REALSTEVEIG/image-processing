"""Command-line interface: run the full pipeline on one image file.

Pipeline: load (``cv2.imread``) -> grayscale -> min-max normalise -> Sobel
edges (+ optional OpenCV Canny for comparison) -> feature extraction ->
write images and features to an output directory.

Usage::

    imgfeatures examples/sample.png --output out/ --canny
    python -m imgfeatures examples/sample.png -o out/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .edges import canny_opencv, gradient_magnitude_direction, sobel_edges
from .features import (
    edge_density,
    feature_vector,
    gradient_statistics,
)
from .preprocessing import normalize_minmax, to_grayscale


class CLIError(Exception):
    """An expected, user-facing failure (reported without a traceback)."""


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="imgfeatures",
        description=(
            "Run grayscale -> normalise -> Sobel edges -> feature extraction "
            "on an image and write the results to a directory."
        ),
    )
    parser.add_argument("image", type=Path, help="input image file (PNG, JPEG, ...)")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output"),
        help="output directory, created if missing (default: ./output)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Sobel edge threshold as a fraction of max magnitude (default: 0.2)",
    )
    parser.add_argument(
        "--canny", action="store_true",
        help="also run OpenCV Canny and save its edge map for comparison",
    )
    parser.add_argument("--canny-low", type=float, default=50.0,
                        help="Canny lower hysteresis threshold (default: 50)")
    parser.add_argument("--canny-high", type=float, default=150.0,
                        help="Canny upper hysteresis threshold (default: 150)")
    parser.add_argument(
        "--format", choices=("json", "npy", "both"), default="both",
        help="feature output format (default: both)",
    )
    parser.add_argument("--hist-bins", type=int, default=16,
                        help="intensity histogram bins (default: 16)")
    parser.add_argument("--cell-size", type=int, default=8,
                        help="HOG cell size in pixels (default: 8)")
    parser.add_argument("--hog-bins", type=int, default=9,
                        help="HOG orientation bins (default: 9)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def load_rgb(path: Path) -> np.ndarray:
    """Read an image file as an ``(H, W, 3)`` RGB ``uint8`` array."""
    if not path.exists():
        raise CLIError(f"input file not found: {path}")
    if not path.is_file():
        raise CLIError(f"input path is not a file: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise CLIError(f"could not read image (unsupported or corrupt): {path}")
    return np.ascontiguousarray(bgr[..., ::-1])  # OpenCV loads BGR


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale an array (or boolean mask) to ``uint8`` [0, 255] for saving."""
    if image.dtype == bool:
        return image.astype(np.uint8) * 255
    return np.round(normalize_minmax(image) * 255.0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), to_uint8(image)):
        raise CLIError(f"failed to write {path}")


def run(args: argparse.Namespace) -> dict:
    """Execute the pipeline and write outputs. Returns the summary dict."""
    rgb = load_rgb(args.image)
    h, w = rgb.shape[:2]
    if h < args.cell_size or w < args.cell_size:
        raise CLIError(
            f"image {w}x{h} is smaller than one {args.cell_size}x{args.cell_size} "
            "HOG cell; use a larger image or a smaller --cell-size"
        )

    gray = normalize_minmax(to_grayscale(rgb))
    magnitude, _ = gradient_magnitude_direction(gray)
    edges = sobel_edges(gray, threshold=args.threshold)
    canny = (
        canny_opencv(gray, args.canny_low, args.canny_high) if args.canny else None
    )
    features = feature_vector(
        rgb,
        hist_bins=args.hist_bins,
        cell_size=args.cell_size,
        hog_bins=args.hog_bins,
        edge_threshold=args.threshold,
    )
    stats = gradient_statistics(gray)

    out: Path = args.output
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLIError(f"cannot create output directory {out}: {exc}") from exc

    written: list[Path] = []
    for name, img in (
        ("grayscale.png", gray),
        ("edge_magnitude.png", magnitude),
        ("edges_sobel.png", edges),
    ):
        _write_png(out / name, img)
        written.append(out / name)
    if canny is not None:
        _write_png(out / "edges_canny.png", canny)
        written.append(out / "edges_canny.png")

    summary = {
        "input": str(args.image),
        "width": w,
        "height": h,
        "parameters": {
            "threshold": args.threshold,
            "hist_bins": args.hist_bins,
            "cell_size": args.cell_size,
            "hog_bins": args.hog_bins,
            "canny": (
                {"low": args.canny_low, "high": args.canny_high}
                if args.canny else None
            ),
        },
        "gradient_stats": dict(
            zip(("mean", "std", "max", "p90"), map(float, stats), strict=True)
        ),
        "sobel_edge_density": edge_density(gray, threshold=args.threshold),
        "canny_edge_density": float(canny.mean()) if canny is not None else None,
        "feature_vector_length": int(features.size),
        "layout": {
            "histogram": args.hist_bins,
            "gradient_stats": 4,
            "edge_density": 1,
            "hog": int(features.size) - args.hist_bins - 5,
        },
    }

    if args.format in ("json", "both"):
        path = out / "features.json"
        payload = {**summary, "feature_vector": features.tolist()}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    if args.format in ("npy", "both"):
        path = out / "features.npy"
        np.save(path, features)
        written.append(path)

    summary["outputs"] = [str(p) for p in written]
    return summary


def print_summary(summary: dict) -> None:
    """Print a short human-readable summary."""
    g = summary["gradient_stats"]
    print(f"Image:               {summary['input']} "
          f"({summary['width']}x{summary['height']})")
    print(f"Gradient (mean/max): {g['mean']:.4f} / {g['max']:.4f}")
    print(f"Sobel edge density:  {summary['sobel_edge_density']:.4f}")
    if summary["canny_edge_density"] is not None:
        print(f"Canny edge density:  {summary['canny_edge_density']:.4f}")
    print(f"Feature vector:      {summary['feature_vector_length']} values")
    print("Wrote:")
    for path in summary["outputs"]:
        print(f"  {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (CLIError, ValueError, TypeError) as exc:
        print(f"imgfeatures: error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0
