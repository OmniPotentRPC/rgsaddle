"""NumPy stencil: CI-NEB, zoom, dimer, and the thread refusal."""

from __future__ import annotations

import numpy as np
import pytest

import rgsaddle
from rgsaddle import StencilPhase, StencilPreset, ZoomEnds

MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
MB_AA = np.array([-1.0, -1.0, -6.5, 0.7])
MB_B = np.array([0.0, 0.0, 11.0, 0.6])
MB_C = np.array([-10.0, -10.0, -6.5, 0.7])
MB_X0 = np.array([1.0, 0.0, -0.5, -1.0])
MB_Y0 = np.array([0.0, 0.5, 1.5, 1.0])
TS2 = np.array([0.21248658, 0.29298833, 0.0])
B = np.array([-0.050010822738, 0.466694104872, 0.0])
C = np.array([0.623499404931, 0.028037758529, 0.0])


def muller_brown(positions: np.ndarray):
    """Müller-Brown plus a unit spring in z. ``positions`` is (rows, 3)."""
    rows = positions.shape[0]
    energy = np.empty(rows)
    gradient = np.zeros_like(positions)
    for row in range(rows):
        x = float(positions[row, 0])
        y = float(positions[row, 1])
        z = float(positions[row, 2])
        value = z * z
        gx = 0.0
        gy = 0.0
        for i in range(4):
            dx = x - MB_X0[i]
            dy = y - MB_Y0[i]
            exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy
            weight = MB_A[i] * np.exp(exponent)
            px = 2.0 * MB_AA[i] * dx + MB_B[i] * dy
            py = MB_B[i] * dx + 2.0 * MB_C[i] * dy
            value += weight
            gx += weight * px
            gy += weight * py
        energy[row] = value
        gradient[row, 0] = gx
        gradient[row, 1] = gy
        gradient[row, 2] = 2.0 * z
    return energy, gradient


def muller_brown_ids(ids: np.ndarray, positions: np.ndarray):
    assert ids.dtype == np.int64
    assert ids.shape[0] == positions.shape[0]
    return muller_brown(positions)


def band(images: int = 7) -> np.ndarray:
    out = np.zeros((images, 3))
    for image in range(images):
        t = image / (images - 1)
        out[image] = (1.0 - t) * C + t * B
    return np.ascontiguousarray(out)


def test_ci_neb_reaches_the_muller_brown_saddle():
    stencil = rgsaddle.Stencil.band(band()).ci_tol(1e-3).build()
    report = stencil.run(muller_brown, 250)
    positions = stencil.positions()
    ci = int(report.ci_index)
    assert 0 < ci < positions.shape[0] - 1
    assert np.linalg.norm(positions[ci] - TS2) < 0.05


def test_zoom_reports_the_window():
    stencil = (
        rgsaddle.Stencil.band(band())
        .preset(int(StencilPreset.EON_ZOOM))
        .zoom_stability(1)
        .zoom_force_abs(1.0e9)
        .ends(int(ZoomEnds.FIXED))
        .ci_tol(1e-2)
        .build()
    )
    window = None
    phase = None
    for _ in range(40):
        report = stencil.step(muller_brown_ids)
        if report.phase == int(StencilPhase.ZOOM):
            window = report.zoom_window()
            phase = report.phase
            assert report.surface_rows == 0
            break
    assert phase == int(StencilPhase.ZOOM)
    assert window is not None
    lo, hi = window
    assert 0 <= lo < hi < band().shape[0]


def test_dimer_from_near_the_saddle():
    x0 = np.array([0.25, 0.30, 0.0])
    stencil = rgsaddle.Stencil.point(np.ascontiguousarray(x0)).build()
    report = stencil.run(muller_brown, 80)
    position = stencil.positions()[0]
    assert np.linalg.norm(position - TS2) < 0.15
    assert np.isfinite(report.curvature)


def test_threads_above_one_are_refused():
    stencil = rgsaddle.Stencil.band(band(5)).threads(2).build()
    with pytest.raises(ValueError, match="threads > 1"):
        stencil.step(muller_brown)
