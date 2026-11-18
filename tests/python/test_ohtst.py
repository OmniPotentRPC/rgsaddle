"""OH-TST through the nanobind surface: sample, climb, MBAR, work, transmission, rate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rgsaddle import (
    ohtst_mbar,
    ohtst_optimize,
    ohtst_rate,
    ohtst_sample,
    ohtst_transmission,
    ohtst_work,
)


def surface(q: np.ndarray) -> tuple[float, np.ndarray]:
    x = float(q[0])
    y = float(q[1])
    well = x * x - 1.0
    energy = 0.25 * well * well + 0.5 * y * y
    gradient = np.array([x * well, y], dtype=np.float64)
    return energy, gradient


def test_sample_optimize_work_transmission_and_rate() -> None:
    point = np.array([0.0, 0.0], dtype=np.float64)
    normal = np.array([1.0, 0.0], dtype=np.float64)
    sample = ohtst_sample(surface, point, normal, 0.25, 0.05, 1.0, 2, 1, 8, 2, 1)
    assert sample.records == 8
    assert sample.evaluations > 0
    assert 0.0 <= sample.acceptance <= 1.0
    assert math.isfinite(sample.normal_force)
    assert np.asarray(sample.centroid).shape == (2,)
    assert np.asarray(sample.grad_normal).shape == (2,)

    optimum = ohtst_optimize(
        surface,
        np.array([0.0, 0.0], dtype=np.float64),
        normal,
        0.25,
        0.05,
        1.0,
        2,
        1,
        8,
        2,
        1,
        4,
        0.2,
        0.1,
        1,
        True,
        False,
    )
    out_normal = np.asarray(optimum.normal, dtype=np.float64)
    assert optimum.evaluations > 0
    assert abs(float(np.dot(out_normal, out_normal)) - 1.0) < 0.000_001
    assert math.isfinite(optimum.normal_force)

    points = np.array([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    work = ohtst_work(
        surface, points, normals, 0.25, 0.05, 1.0, 2, 1, 8, 2, 1, 3, 2, None, True, 2
    )
    assert math.isfinite(work.work)
    assert math.isfinite(work.mbar_work)
    assert work.evaluations > 0

    starts = np.array([[0.0, 0.0], [0.0, 0.2]], dtype=np.float64)
    kappa = ohtst_transmission(surface, point, normal, starts, 0.25, 0.05, 0.0, 16, 4, 4, 2, 2)
    assert math.isfinite(kappa.kappa)
    assert kappa.shots + kappa.unassigned == 4
    assert kappa.evaluations > 0

    rate = ohtst_rate(0.25, 1.0, 0.0, 0.0, 0.0)
    assert rate.rate > 0.0
    assert abs(rate.rate - rate.flux) <= rate.flux / 1_000_000
    assert abs(rate.log_rate - math.log(rate.rate)) <= 0.000_001


def test_mbar_pins_the_first_state_at_zero() -> None:
    reduced = np.array([[0.0, 4.0], [0.1, 3.8], [3.9, 0.0], [4.2, 0.2]], dtype=np.float64)
    counts = np.array([2, 2], dtype=np.int64)
    solved = ohtst_mbar(reduced, counts)
    free = np.asarray(solved.free_energies, dtype=np.float64)
    overlap = np.asarray(solved.overlap, dtype=np.float64)
    assert free.shape == (2,)
    assert abs(float(free[0])) <= 0.000_001
    assert overlap.shape == (2, 2)
    assert abs(float(overlap.sum(axis=1)[0]) - 1.0) < 0.000_001
    assert abs(float(overlap.sum(axis=1)[1]) - 1.0) < 0.000_001


def test_rate_refuses_a_non_positive_density() -> None:
    with pytest.raises(ValueError):
        ohtst_rate(0.25, 0.0, 0.0, 0.0, 0.0)


def test_rate_refuses_a_negative_standard_error() -> None:
    with pytest.raises(ValueError):
        ohtst_rate(0.25, 1.0, -0.1, 0.0, 0.0)
    with pytest.raises(ValueError):
        ohtst_rate(0.25, 1.0, 0.0, 1.0, -0.1)


def test_transmission_stops_entering_python_after_a_callback_error() -> None:
    calls = {"n": 0}

    def failing(q: np.ndarray) -> tuple[float, np.ndarray]:
        calls["n"] += 1
        if calls["n"] > 8:
            raise RuntimeError("stop the other shots")
        return surface(q)

    point = np.array([0.0, 0.0], dtype=np.float64)
    normal = np.array([1.0, 0.0], dtype=np.float64)
    starts = np.zeros((16, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="stop the other shots"):
        ohtst_transmission(failing, point, normal, starts, 0.25, 0.02, 0.0, 40, 10, 16, 2, 1)
    # Sixteen shots of forty steps would be hundreds of callbacks if the
    # others kept entering Python after the first exception.
    assert calls["n"] < 80


def test_mbar_covariance_is_states_by_states() -> None:
    reduced = np.array([[0.0, 4.0, 8.0], [0.2, 3.5, 7.5], [4.0, 0.1, 4.0], [8.0, 4.0, 0.0]])
    counts = np.array([1, 1, 2], dtype=np.int64)
    solved = ohtst_mbar(reduced, counts)
    covariance = np.asarray(solved.covariance)
    overlap = np.asarray(solved.overlap)
    assert covariance.shape == (3, 3)
    assert overlap.shape == (3, 3)
    assert covariance.size == 9
    assert overlap.size == 9


def test_mbar_rejects_a_state_square_that_overflows() -> None:
    states = 1 << 32
    base = np.zeros(1, dtype=np.float64)
    reduced = np.lib.stride_tricks.as_strided(base, shape=(1, states), strides=(8, 0))
    counts = np.lib.stride_tricks.as_strided(
        np.ones(1, dtype=np.int64), shape=(states,), strides=(0,)
    )
    with pytest.raises(ValueError, match="overflows"):
        ohtst_mbar(reduced, counts)
