"""Stencil through the nanobind wrapper: a dimer, a band, and the refusals."""

from __future__ import annotations

import numpy as np
import pytest

from rgsaddle import Stencil


def bowl(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    v = row[0]
    energy = np.array([2.0 * v[0] * v[0] + 5.0 * v[1] * v[1]], dtype=np.float64)
    gradient = np.array([[4.0 * v[0], 10.0 * v[1]]], dtype=np.float64)
    return energy, gradient


def well(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    energy = np.empty(row.shape[0], dtype=np.float64)
    gradient = np.zeros_like(row)
    for i in range(row.shape[0]):
        px = float(row[i, 0])
        py = float(row[i, 1])
        energy[i] = (px * px - 1.0) ** 2 + py * py
        gradient[i, 0] = 4.0 * px * (px * px - 1.0)
        gradient[i, 1] = 2.0 * py
    return energy, gradient


def well_ids(ids: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert ids.dtype == np.int64
    assert ids.shape[0] == positions.shape[0]
    assert int(ids.min()) != int(ids.max()) or positions.shape[0] == 1
    return well(positions)


def test_dimer_with_positive_curvature_moves_along_the_mode() -> None:
    x = np.array([0.4, 0.3], dtype=np.float64)
    mode = np.array([1.0, 0.0], dtype=np.float64)
    search = Stencil.point(x, preset="dimer", mode=mode, max_rotations=4)
    report = search.step(bowl)
    positions = np.asarray(search.positions(), dtype=np.float64)
    assert positions[1].tobytes() == np.float64(0.3).tobytes()
    assert positions[0].tobytes() != np.float64(0.4).tobytes()
    assert report.iteration >= 1


def test_ci_neb_steps_and_keeps_endpoint_ids() -> None:
    images = np.array([[-1.0, 0.0], [0.0, 0.2], [1.0, 0.0]], dtype=np.float64)
    search = Stencil.band(images, preset="ci-neb")
    report = search.step(well_ids)
    positions = np.asarray(search.positions(), dtype=np.float64)
    assert positions.shape == (3, 2)
    assert report.iteration >= 1
    ids = list(search.image_ids())
    assert ids[0] == 0
    assert ids[-1] == 2


def test_zoom_pack_reports_a_window() -> None:
    images = np.zeros((7, 2), dtype=np.float64)
    for i in range(7):
        images[i, 0] = -1.0 + 2.0 * i / 6.0
    search = Stencil.band(
        images,
        preset="zoom-neb",
        zoom_stability=1,
        zoom_force=1.0e9,
        zoom_force_kind=1,
        zoom_ends=0,
        ci_trigger_force=1.0e300,
    )
    phase = None
    for _ in range(8):
        report = search.step(well)
        if report.phase == 1:
            phase = report.phase
            assert report.evaluations == 0
            assert 0 <= report.zoom_lo < report.zoom_hi
            break
    assert phase == 1


def test_idpp_names_the_missing_piece() -> None:
    images = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="not implemented") as err:
        Stencil.band(images, preset="idpp")
    assert "feat/" not in str(err.value)


def test_unknown_preset_is_a_value_error() -> None:
    x = np.array([0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="preset"):
        Stencil.point(x, preset="zoom")


def test_threads_above_one_are_refused() -> None:
    images = np.zeros((3, 2), dtype=np.float64)
    search = Stencil.band(images, preset="neb", threads=2)
    with pytest.raises(ValueError, match="threads > 1"):
        search.step(well)
