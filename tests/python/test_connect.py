"""Double-ended connect through the nanobind wrapper."""

from __future__ import annotations

import math

import numpy as np

from rgsaddle import connect


def toy(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.ascontiguousarray(x, dtype=np.float64)
    a, b, c = row[0]
    u = a * a * a - 4.0 * a
    energy = np.array([u * u + 2.0 * b * b + 2.0 * c * c], dtype=np.float64)
    gradient = np.array([[2.0 * u * (3.0 * a * a - 4.0), 4.0 * b, 4.0 * c]], dtype=np.float64)
    return energy, gradient


def test_connect_finds_the_two_barrier_saddle() -> None:
    calls = {"n": 0}

    def surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls["n"] += 1
        return toy(x)

    left = np.array([-2.0, 0.0, 0.0], dtype=np.float64)
    right = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    report = connect(surface, left, right, force_tol=1e-3, max_calls=2000, quench_nodes=True)
    position = np.asarray(report.position, dtype=np.float64)
    assert report.kind == "s"
    assert report.hessian_index == 1
    assert report.force_norm <= 1e-3
    assert report.evaluations == calls["n"]
    assert report.evaluations < 4000
    assert abs(report.energy - 256.0 / 27.0) < 1e-6
    assert abs(abs(position[0]) - math.sqrt(4.0 / 3.0)) < 1e-4
    assert abs(position[1]) < 1e-4 and abs(position[2]) < 1e-4


def test_connect_rejects_a_length_mismatch() -> None:
    left = np.zeros(3, dtype=np.float64)
    right = np.zeros(6, dtype=np.float64)

    def surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError(x.shape)

    try:
        connect(surface, left, right, max_calls=10)
    except ValueError as err:
        assert "multiple of three" in str(err)
    else:
        raise AssertionError("mismatched endpoints were accepted")
