"""OH-TST through the nanobind surface: sample, climb, MBAR, work, transmission, rate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rgsaddle import (
    NumericalError,
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


def coupled_surface(
    q: np.ndarray, linear: float = 0.1, quadratic: float = 0.0
) -> tuple[float, np.ndarray]:
    x = float(q[0])
    y = float(q[1])
    well = x * x - 1.0
    stiffness = 1.0 + 2.0 * linear * x + 2.0 * quadratic * x * x
    energy = 0.25 * well * well + 0.5 * stiffness * y * y
    gradient = np.array(
        [
            x * well + (linear + 2.0 * quadratic * x) * y * y,
            stiffness * y,
        ],
        dtype=np.float64,
    )
    return energy, gradient


def coupled_optimum(thermal: float, linear: float, quadratic: float) -> float:
    def mean_gradient(x: float) -> float:
        stiffness = 1.0 + 2.0 * linear * x + 2.0 * quadratic * x * x
        return x * (x * x - 1.0) + (linear + 2.0 * quadratic * x) * thermal / stiffness

    x = linear * thermal
    for _ in range(20):
        h = 1.0e-6
        derivative = (mean_gradient(x + h) - mean_gradient(x - h)) / (2.0 * h)
        x -= mean_gradient(x) / derivative
    assert abs(mean_gradient(x)) < 1.0e-12
    return x


def test_sample_optimize_work_transmission_and_rate() -> None:
    point = np.array([0.3, 0.0], dtype=np.float64)
    normal = np.array([1.0, 0.0], dtype=np.float64)
    sample = ohtst_sample(surface, point, normal, 0.25, 0.05, 1.0, 2, 1, 8, 2, 1)
    expected_force = -0.3 * (0.3 * 0.3 - 1.0)
    assert sample.records == 8
    assert sample.evaluations == 11
    assert sample.acceptance == 1.0
    assert abs(sample.normal_force - expected_force) < 1.0e-12
    assert sample.normal_force_error == 0.0
    centroid = np.asarray(sample.centroid, dtype=np.float64)
    grad_normal = np.asarray(sample.grad_normal, dtype=np.float64)
    assert centroid.shape == (2,)
    assert abs(float(centroid[0]) - 0.3) < 1.0e-12
    assert grad_normal.shape == (2,)
    assert abs(float(grad_normal[0])) < 1.0e-12

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
    out_point = np.asarray(optimum.point, dtype=np.float64)
    out_normal = np.asarray(optimum.normal, dtype=np.float64)
    assert optimum.evaluations > 0
    assert abs(float(out_point[0])) < 1.0e-12
    assert np.allclose(out_normal, np.array([1.0, 0.0], dtype=np.float64), atol=1.0e-12)
    assert abs(optimum.normal_force) < 1.0e-12
    assert optimum.normal_force_error == 0.0

    points = np.array([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    work = ohtst_work(
        surface, points, normals, 0.25, 0.05, 1.0, 2, 1, 8, 2, 1, 3, 2, None, True, 2
    )
    expected_work = 0.25 * ((0.3 * 0.3 - 1.0) ** 2 - 1.0)
    assert abs(work.work - expected_work) < 1.0e-12
    assert abs(work.integrated_work - expected_work) < 1.0e-12
    assert abs(work.bar_work - expected_work) < 1.0e-12
    assert abs(work.mbar_work - expected_work) < 1.0e-12
    assert work.integrated_error == 0.0
    assert work.evaluations > 0

    starts = np.array([[0.0, 0.0], [0.0, 0.2]], dtype=np.float64)
    kappa = ohtst_transmission(
        surface,
        np.array([0.0, 0.0], dtype=np.float64),
        normal,
        starts,
        0.25,
        0.05,
        0.0,
        16,
        4,
        4,
        2,
        2,
    )
    assert abs(kappa.kappa - 1.0) < 1.0e-12
    assert kappa.error == 0.0
    assert kappa.shots == 4
    assert kappa.unassigned == 0
    assert kappa.reactive == 4
    assert kappa.recrossed == 0
    assert kappa.evaluations > 0

    rate = ohtst_rate(0.25, 0.8, 0.04, 0.5, 0.02)
    expected_flux = math.sqrt(0.25 / (2.0 * math.pi))
    expected_log_rate = math.log(expected_flux) + math.log(0.8) - 0.5 / 0.25
    expected_log_error = math.hypot(0.04 / 0.8, 0.02 / 0.25)
    expected_rate = math.exp(expected_log_rate)
    assert abs(rate.flux - expected_flux) < 1.0e-12
    assert abs(rate.log_rate - expected_log_rate) < 1.0e-12
    assert abs(rate.log_rate_error - expected_log_error) < 1.0e-12
    assert abs(rate.rate - expected_rate) < 1.0e-12
    assert abs(rate.error - expected_rate * expected_log_error) < 1.0e-12


def test_optimize_refuses_to_report_converged_when_unconverged() -> None:
    with pytest.raises(ValueError, match="did not converge"):
        ohtst_optimize(
            surface,
            np.array([0.4, 0.0], dtype=np.float64),
            np.array([1.0, 0.3], dtype=np.float64),
            0.25,
            0.05,
            1.0,
            2,
            1,
            8,
            2,
            1,
            1,
            0.05,
            0.05,
            1,
            True,
            True,
        )


@pytest.mark.parametrize(("linear", "quadratic"), [(0.1, 0.0), (0.12, 0.08)])
def test_optimize_converges_on_coupled_surface(
    linear: float, quadratic: float
) -> None:
    thermal = 0.02585
    point = np.array(
        [coupled_optimum(thermal, linear, quadratic), 0.0], dtype=np.float64
    )
    optimum = ohtst_optimize(
        lambda q: coupled_surface(q, linear, quadratic),
        point,
        np.array([1.0, 0.0], dtype=np.float64),
        thermal,
        0.05,
        1.0,
        500,
        5,
        2000,
        20,
        1,
        100,
        0.1,
        0.05,
    )
    assert optimum.normal_force_error > 0.0


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


def test_mbar_accepts_positive_infinite_reduced_energy() -> None:
    reduced = np.array(
        [[0.0, math.inf], [0.1, 3.8], [3.9, 0.0], [4.2, 0.2]], dtype=np.float64
    )
    solved = ohtst_mbar(reduced, np.array([2, 2], dtype=np.int64))
    assert np.asarray(solved.free_energies).shape == (2,)


def test_rate_refuses_a_non_positive_density() -> None:
    with pytest.raises(ValueError, match="density"):
        ohtst_rate(0.25, 0.0, 0.0, 0.0, 0.0)


def test_rate_refuses_a_negative_standard_error() -> None:
    with pytest.raises(ValueError, match="standard errors"):
        ohtst_rate(0.25, 1.0, -0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="standard errors"):
        ohtst_rate(0.25, 1.0, 0.0, 1.0, -0.1)


def test_rate_refuses_non_finite_rate_or_error() -> None:
    with pytest.raises(NumericalError, match="rate"):
        ohtst_rate(1.0, 1.0, 0.0, -1000.0, 0.0)
    with pytest.raises(NumericalError, match="rate"):
        ohtst_rate(1.0, 1.0, 0.1, -1000.0, 0.1)
    with pytest.raises(NumericalError, match="rate"):
        ohtst_rate(1.0, 1.0, 0.0, 0.0, 1.0e300)


def test_python_exceptions_name_the_cause() -> None:
    point = np.array([0.0, 0.0], dtype=np.float64)
    normal = np.array([1.0, 0.0], dtype=np.float64)

    with pytest.raises(NumericalError, match=r"non-finite.*point"):
        ohtst_sample(
            surface,
            np.array([math.nan, 0.0], dtype=np.float64),
            normal,
            0.25,
            0.05,
            1.0,
            2,
            1,
            8,
            2,
            1,
        )

    with pytest.raises(NumericalError, match=r"non-finite.*max_translation"):
        ohtst_optimize(
            surface,
            point,
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
            math.nan,
            0.1,
            1,
            True,
            False,
        )

    with pytest.raises(NumericalError, match=r"non-finite.*initial"):
        ohtst_mbar(
            np.array([[0.0, 4.0], [0.1, 3.8], [3.9, 0.0], [4.2, 0.2]], dtype=np.float64),
            np.array([2, 2], dtype=np.int64),
            np.array([0.0, math.inf], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="estimate"):
        ohtst_work(
            surface,
            np.array([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64),
            np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            0.25,
            0.05,
            1.0,
            2,
            1,
            8,
            2,
            1,
            3,
            0,
            None,
            False,
            99,
        )

    with pytest.raises(NumericalError, match=r"non-finite.*starts"):
        ohtst_transmission(
            surface,
            point,
            normal,
            np.array([[0.0, math.nan]], dtype=np.float64),
            0.25,
            0.05,
            0.0,
            8,
            2,
            2,
            2,
            1,
        )

    with pytest.raises(NumericalError, match="thermal"):
        ohtst_rate(math.nan, 1.0, 0.0, 0.5, 0.0)

    with pytest.raises(ValueError, match="gradient length"):
        ohtst_sample(
            lambda _q: (0.0, np.array([1.0], dtype=np.float64)),
            point,
            normal,
            0.25,
            0.05,
            1.0,
            1,
            1,
            4,
            2,
            1,
        )


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
