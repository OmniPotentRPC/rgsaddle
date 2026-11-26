"""Spline reparametrization and the curvature-aware rotation stop."""

from __future__ import annotations

import numpy as np
import pytest

import rgsaddle
from rgsaddle import _core


def _cubic_band(n: int, power: float = 1.5) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)
    t = -1.0 + 2.0 * u**power
    return np.column_stack([t, t**3 - t, 0.5 * t**2])


@pytest.mark.parametrize("scheme", ["linear", "not-a-knot", "spline"])
def test_a_straight_band_is_spaced_exactly(scheme: str) -> None:
    band = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    placed = rgsaddle.reparametrize(band, scheme=scheme)
    np.testing.assert_allclose(placed[:, 0], np.linspace(0.0, 2.0, 4), atol=1e-12)
    assert band[1, 0] == 0.1


def test_spline_keeps_endpoints_and_anchor_and_evens_the_chords() -> None:
    band = _cubic_band(17)
    placed = rgsaddle.reparametrize(band, scheme="not-a-knot", anchor=6)
    for row in (0, 6, 16):
        assert np.array_equal(placed[row], band[row])
    chords = np.linalg.norm(np.diff(placed, axis=0), axis=1)
    for side in (chords[:6], chords[6:]):
        assert np.ptp(side) / side.mean() < 0.05


def test_spline_unwraps_a_periodic_band() -> None:
    cell = np.diag([4.0, 4.0, 4.0])
    unwrapped = 0.5 * _cubic_band(9) + np.array([3.6, 1.0, 1.0])
    placed = rgsaddle.reparametrize(np.mod(unwrapped, 4.0), scheme="spline", cell=cell)
    reference = rgsaddle.reparametrize(unwrapped, scheme="spline")
    delta = placed - reference
    delta -= 4.0 * np.round(delta / 4.0)
    assert np.abs(delta).max() < 1e-12


def test_spline_resample_imports_from_the_package() -> None:
    from rgsaddle import spline_resample

    band = np.ascontiguousarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    moved = np.asarray(spline_resample(band))
    assert moved.shape == (3, 3)
    assert np.allclose(moved[0], band[0])
    assert np.allclose(moved[-1], band[-1])
    assert abs(float(moved[1, 0]) - 0.5) < 1e-12
    assert "spline_resample" in rgsaddle.__all__


def test_reparametrize_rejects_bad_input() -> None:
    band = _cubic_band(5)
    with pytest.raises(ValueError):
        rgsaddle.reparametrize(band, scheme="quintic")
    with pytest.raises(ValueError):
        rgsaddle.reparametrize(band, anchor=0)
    with pytest.raises(ValueError):
        rgsaddle.reparametrize(band, cell=np.eye(2))
    with pytest.raises(ValueError):
        rgsaddle.reparametrize(band[:1])


def _quadratic(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curvatures = np.array([-2.0, 1.0, 4.0, 8.0, 2.0, 3.0])
    x = np.asarray(x).reshape(-1)
    energy = 0.5 * np.sum(curvatures * x * x)
    return np.array([energy]), (curvatures * x).reshape(1, -1)


@pytest.mark.parametrize("kind", ["dimer", "lanczos"])
def test_rotation_stop_length_cuts_rotations(kind: str) -> None:
    x = np.array([0.3, 0.2, 0.1, 0.4, 0.1, 0.2])
    mode = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    stock = _core.MinMode(x, mode, kind=kind, rotation_tol=1e-6).estimate(_quadratic)
    off = _core.MinMode(x, mode, kind=kind, rotation_tol=1e-6, rotation_stop_length=0.0)
    assert off.estimate(_quadratic).rotations == stock.rotations
    on = _core.MinMode(x, mode, kind=kind, rotation_tol=1e-6, rotation_stop_length=1e-3)
    report = on.estimate(_quadratic)
    assert report.rotations <= stock.rotations
    if kind == "dimer":
        assert report.rotations < stock.rotations
        assert report.curvature < 0.0


def test_rotation_stop_length_must_be_finite_and_non_negative() -> None:
    x = np.zeros(3)
    mode = np.array([1.0, 0.0, 0.0])
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _core.MinMode(x, mode, rotation_stop_length=bad)
