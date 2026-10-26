"""End-to-end tests for the command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from imgfeatures import feature_vector_length
from imgfeatures.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "examples" / "sample.png"


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """A 48x64 colour PNG with a bright rectangle and a circle."""
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.rectangle(img, (8, 8), (30, 36), (0, 200, 255), thickness=-1)
    cv2.circle(img, (46, 24), 10, (255, 255, 255), thickness=-1)
    path = tmp_path / "input.png"
    assert cv2.imwrite(str(path), img)
    return path


def _check_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    assert path.is_file(), path
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"unreadable PNG: {path}"
    assert img.shape == shape
    assert img.dtype == np.uint8
    return img


def test_main_writes_all_outputs(image_file: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "out"
    code = main([str(image_file), "--output", str(out), "--canny"])
    assert code == 0

    gray = _check_png(out / "grayscale.png", (48, 64))
    assert gray.min() == 0 and gray.max() == 255  # normalised to full range
    _check_png(out / "edge_magnitude.png", (48, 64))
    for name in ("edges_sobel.png", "edges_canny.png"):
        edges = _check_png(out / name, (48, 64))
        assert set(np.unique(edges)) <= {0, 255}
        assert edges.any()

    expected_len = feature_vector_length((48, 64))
    vec = np.load(out / "features.npy")
    assert vec.shape == (expected_len,)
    assert np.isfinite(vec).all()

    data = json.loads((out / "features.json").read_text())
    assert data["width"] == 64 and data["height"] == 48
    assert data["feature_vector_length"] == expected_len
    np.testing.assert_allclose(data["feature_vector"], vec)
    assert 0 < data["sobel_edge_density"] < 1
    assert 0 < data["canny_edge_density"] < 1
    assert sum(data["layout"].values()) == expected_len

    stdout = capsys.readouterr().out
    assert "Feature vector" in stdout
    assert "features.json" in stdout


@pytest.mark.parametrize(
    ("fmt", "present", "absent"),
    [("json", "features.json", "features.npy"),
     ("npy", "features.npy", "features.json")],
)
def test_format_option(image_file: Path, tmp_path: Path, fmt, present, absent) -> None:
    out = tmp_path / "out"
    assert main([str(image_file), "-o", str(out), "--format", fmt]) == 0
    assert (out / present).is_file()
    assert not (out / absent).exists()
    assert not (out / "edges_canny.png").exists()  # --canny not given


def test_sample_image_runs(tmp_path: Path) -> None:
    assert SAMPLE.is_file(), "examples/sample.png is missing"
    assert SAMPLE.stat().st_size < 50_000
    out = tmp_path / "out"
    assert main([str(SAMPLE), "-o", str(out), "--canny"]) == 0
    assert (out / "features.json").is_file()


def test_missing_file(tmp_path: Path, capsys) -> None:
    code = main([str(tmp_path / "nope.png"), "-o", str(tmp_path / "out")])
    assert code == 1
    err = capsys.readouterr().err
    assert "not found" in err
    assert "Traceback" not in err
    assert not (tmp_path / "out").exists()


def test_unreadable_file(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    assert main([str(bad), "-o", str(tmp_path / "out")]) == 1
    assert "could not read image" in capsys.readouterr().err


def test_directory_as_input(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path), "-o", str(tmp_path / "out")]) == 1
    assert "not a file" in capsys.readouterr().err


def test_invalid_threshold(image_file: Path, tmp_path: Path, capsys) -> None:
    code = main([str(image_file), "-o", str(tmp_path / "o"), "--threshold", "2"])
    assert code == 1
    assert "threshold" in capsys.readouterr().err


def test_image_smaller_than_cell(tmp_path: Path, capsys) -> None:
    tiny = tmp_path / "tiny.png"
    cv2.imwrite(str(tiny), np.zeros((4, 4), dtype=np.uint8))
    assert main([str(tiny), "-o", str(tmp_path / "o")]) == 1
    assert "smaller than one" in capsys.readouterr().err


def test_bad_argument_exits_2(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-m", "imgfeatures", *args],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )


def test_subprocess_success(image_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _run_module(str(image_file), "--output", str(out), "--canny")
    assert result.returncode == 0, result.stderr
    assert "Sobel edge density" in result.stdout
    for name in ("grayscale.png", "edge_magnitude.png", "edges_sobel.png",
                 "edges_canny.png", "features.json", "features.npy"):
        assert (out / name).is_file(), name


def test_subprocess_missing_file(tmp_path: Path) -> None:
    result = _run_module(str(tmp_path / "missing.png"), "-o", str(tmp_path / "o"))
    assert result.returncode == 1
    assert "error: input file not found" in result.stderr
    assert "Traceback" not in result.stderr
