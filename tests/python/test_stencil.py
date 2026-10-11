"""Point stencil through the nanobind wrapper."""

from __future__ import annotations

import numpy as np

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


def test_unknown_preset_is_a_value_error() -> None:
    x = np.array([0.0, 0.0], dtype=np.float64)
    try:
        Stencil.point(x, preset="zoom")
    except ValueError as err:
        assert "preset" in str(err)
    else:
        raise AssertionError("unknown preset returned")
