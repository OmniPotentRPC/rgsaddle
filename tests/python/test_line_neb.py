from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rgsaddle import (
    LineNeb,
    NaturalCubic,
    fixed_energy_action,
    fixed_energy_action_force,
    fixed_energy_mep_seed,
)


FloatArray = npt.NDArray[np.float64]


def double_well(positions: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    energies = (x * x - 1.0) ** 2 + y * y + z * z
    gradients = np.column_stack((4.0 * x * (x * x - 1.0), 2.0 * y, 2.0 * z))
    return energies, gradients


def test_action_force_and_cubic_match_analytic_values() -> None:
    positions = np.array(
        [[-1.0, 0.2, 0.0], [-0.35, 0.55, 0.0], [0.25, 0.48, 0.0], [0.9, -0.1, 0.0]],
        dtype=np.float64,
    )
    masses = np.array([2.0, 5.0, 3.0], dtype=np.float64)
    x = positions[:, 0]
    y = positions[:, 1]
    energies = 0.7 * x * x + 0.4 * x * y + 1.1 * y * y + 2.0
    gradients = np.column_stack(
        (1.4 * x + 0.4 * y, 0.4 * x + 2.2 * y, np.zeros(positions.shape[0]))
    )
    action = fixed_energy_action(positions, energies, masses, 0.3, 1.7)
    force = fixed_energy_action_force(positions, energies, gradients, masses, 0.3, 1.7)
    assert abs(action - 4.451_267_924_111_834) < 2.0e-15
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.406_560_449_872_918_34, -2.593_280_341_805_817_5, 0.0],
            [-0.851_336_823_737_935_1, -2.151_288_222_332_447, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(force, expected, rtol=0.0, atol=2.0e-15)

    cubic = NaturalCubic(
        np.array([0.0, 1.0, 2.0]),
        np.array([[1.0], [3.0], [5.0]]),
    )
    value, first, second = cubic.sample(0.5)
    np.testing.assert_array_equal(value, [2.0])
    np.testing.assert_array_equal(first, [2.0])
    np.testing.assert_array_equal(second, [0.0])


def test_seed_and_session_recover_the_analytic_mep_and_saddle() -> None:
    masses = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    seed = fixed_energy_mep_seed(
        double_well,
        np.zeros(3, dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        masses,
        1.0,
        0.25,
        9,
    )
    assert seed.shape == (9, 3)
    np.testing.assert_array_equal(seed[:, 1:], 0.0)
    assert abs(seed[4, 0]) <= np.finfo(np.float64).eps
    assert abs((seed[4, 0] ** 2 - 1.0) ** 2 - 1.0) <= np.finfo(np.float64).eps

    turning = np.sqrt(1.0 - np.sqrt(0.3))
    fraction = np.linspace(0.0, 1.0, 9)
    positions = np.zeros((9, 3), dtype=np.float64)
    positions[:, 0] = -turning + 2.0 * turning * fraction
    positions[:, 1] = 0.18 * np.sin(np.pi * fraction)
    line = LineNeb(
        positions,
        masses,
        0.25,
        spring_k=0.4,
        endpoint_kappa=1.5,
        force_tol=2.0e-6,
        endpoint_force_tol=2.0e-6,
        sum_force_tol=2.0e-6,
        time_step=0.03,
    )
    report = line.run(double_well, 80_000)
    assert report.converged()
    assert report.max_interior_optimized <= 2.0e-6
    converged = line.positions()
    assert np.max(np.abs(converged[:, 1:])) < 2.0e-5
    assert abs(converged[4, 0]) < 2.0e-5
    saddle_energy = (converged[4, 0] ** 2 - 1.0) ** 2 + converged[4, 1] ** 2
    assert abs(saddle_energy - 1.0) < 2.0e-6
    endpoint_energies, _ = double_well(converged[[0, -1]])
    np.testing.assert_allclose(endpoint_energies, 0.25, rtol=0.0, atol=2.0e-6)
