"""Point stencil through the nanobind wrapper."""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

import rgsaddle
from rgsaddle import Stencil


def bowl(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    v = row[0]
    energy = np.array([2.0 * v[0] * v[0] + 5.0 * v[1] * v[1]], dtype=np.float64)
    gradient = np.array([[4.0 * v[0], 10.0 * v[1]]], dtype=np.float64)
    return energy, gradient


def test_dimer_with_positive_curvature_moves_along_the_mode() -> None:
    x = np.array([0.4, 0.3], dtype=np.float64)
    mode = np.array([1.0, 0.0], dtype=np.float64)
    search = Stencil.point(x, preset="dimer", mode=mode, max_rotations=4)
    report = search.step(bowl)
    positions = np.asarray(search.positions(), dtype=np.float64)
    assert positions[1].tobytes() == np.float64(0.3).tobytes()
    assert positions[0].tobytes() != np.float64(0.4).tobytes()
    assert report.iteration >= 1


def test_band_preset_names_the_zoom_seam() -> None:
    images = np.zeros((3, 2), dtype=np.float64)
    try:
        Stencil.band(images, preset="ci-neb")
    except ValueError as err:
        assert "feat/band-methods-zoom" in str(err)
    else:
        raise AssertionError("band build returned")


def test_a_second_step_is_a_runtime_error() -> None:
    x = np.array([0.4, 0.3], dtype=np.float64)
    mode = np.array([1.0, 0.0], dtype=np.float64)
    search = Stencil.point(x, preset="dimer", mode=mode, max_rotations=1)
    started = threading.Event()
    release = threading.Event()
    seen: list[BaseException] = []

    def holding(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        started.set()
        assert release.wait(timeout=5)
        return bowl(row)

    def other() -> None:
        assert started.wait(timeout=5)
        try:
            search.step(bowl)
        except BaseException as err:
            seen.append(err)
        finally:
            release.set()

    worker = threading.Thread(target=other)
    worker.start()
    search.step(holding)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(seen) == 1
    assert isinstance(seen[0], RuntimeError)
    assert str(seen[0]) == "Stencil is in use by another thread"
    search.step(bowl)


def test_unknown_preset_is_a_value_error() -> None:
    x = np.array([0.0, 0.0], dtype=np.float64)
    try:
        Stencil.point(x, preset="zoom")
    except ValueError as err:
        assert "preset" in str(err)
    else:
        raise AssertionError("unknown preset returned")


def shallow_saddle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    v = row[0]
    energy = np.array([-0.0025 * v[0] * v[0] + 5.0 * v[1] * v[1]], dtype=np.float64)
    gradient = np.array([[-0.005 * v[0], 10.0 * v[1]]], dtype=np.float64)
    return energy, gradient


def test_art_defaults_match_rust_stencil_point() -> None:
    x = np.array([0.4, 0.3], dtype=np.float64)
    mode = np.array([1.0, 0.0], dtype=np.float64)
    default_art = Stencil.point(x, preset="art", mode=mode)
    explicit_art = Stencil.point(x, preset="art", mode=mode, art_threshold=-0.01, art_push=3)
    for step_index in range(4):
        rep_def = default_art.step(shallow_saddle)
        rep_exp = explicit_art.step(shallow_saddle)
        if step_index < 3:
            assert np.isnan(rep_def.curvature)
        else:
            assert np.isfinite(rep_def.curvature)
        assert rep_def.evaluations == rep_exp.evaluations
        assert np.float64(rep_def.curvature).tobytes() == np.float64(rep_exp.curvature).tobytes()
        pos_def = np.asarray(default_art.positions(), dtype=np.float64)
        pos_exp = np.asarray(explicit_art.positions(), dtype=np.float64)
        assert pos_def.tobytes() == pos_exp.tobytes()
    assert np.asarray(default_art.positions(), dtype=np.float64)[0] > 1.0

    doc = Stencil.point.__doc__ or ""
    assert "art_threshold: float = -0.01" in doc
    assert "art_push: int = 3" in doc
    stub = (Path(rgsaddle.__file__).resolve().parent / "_core.pyi").read_text(encoding="utf-8")
    assert "art_threshold: float = -0.01, art_push: int = 3" in stub
