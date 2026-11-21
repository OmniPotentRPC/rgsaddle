"""Fixed-energy LI-NEB through the nanobind wrapper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rgsaddle import LinebResult, lineb_solve


def eckart(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-atom Eckart barrier V = sech^2(x) on the first Cartesian coordinate."""
    row = np.ascontiguousarray(x, dtype=np.float64).reshape(1, -1)
    coord = row[0, 0]
    s = 1.0 / math.cosh(coord)
    energy = np.array([s * s], dtype=np.float64)
    gradient = np.zeros((1, row.shape[1]), dtype=np.float64)
    gradient[0, 0] = -2.0 * s * s * math.tanh(coord)
    return energy, gradient


def eckart_seed(n_images: int = 9) -> tuple[np.ndarray, float, float]:
    energy = 0.25
    turning = math.acosh(math.sqrt(1.0 / energy))
    end = 0.5 * turning
    xs = np.linspace(-end, end, n_images, dtype=np.float64)
    positions = np.zeros((n_images, 3), dtype=np.float64)
    positions[:, 0] = xs
    spacing = turning * 2.0 / (n_images - 1)
    panels = 8.0
    endpoint_scale = 0.2
    tol = endpoint_scale * spacing / (8.0 * panels**4)
    return positions, energy, tol


def test_lineb_solve_path_only_returns_nan_rate() -> None:
    positions, energy, tol = eckart_seed()
    masses = np.array([1.0], dtype=np.float64)
    result = lineb_solve(
        eckart,
        positions,
        masses,
        energy,
        tol,
        tol,
        molecular=False,
        max_steps=500,
        beads=0,
    )
    assert isinstance(result, LinebResult)
    optimized = np.asarray(result.positions, dtype=np.float64)
    assert optimized.shape == positions.shape
    assert math.isfinite(result.action)
    assert math.isfinite(result.half_period)
    assert math.isfinite(result.half_period_dynamics)
    assert math.isfinite(result.temperature)
    assert math.isfinite(result.temperature_dynamics)
    assert math.isnan(result.log_rate_over_reactant)
    assert result.modes is None and result.negative_modes is None
    assert result.iterations >= 1
    assert result.optimizer_calls >= 1
    assert result.measure_calls >= 1
    assert result.dynamics_calls >= 1
    assert result.rate_calls == 0
    phases = result.surface_calls
    assert phases["seed"] + phases["optimization"] == result.optimizer_calls
    assert phases["path_preparation"] + phases["quadrature"] == result.measure_calls
    assert phases["total"] == sum(count for phase, count in phases.items() if phase != "total")
    exact_action = math.pi * (math.sqrt(2.0) - math.sqrt(0.5))
    exact_half = math.pi * math.sqrt(2.0)
    slope = 0.5 * math.sqrt(0.75)
    spacing = float(optimized[-1, 0] - optimized[0, 0]) / (optimized.shape[0] - 1)
    slack = math.sqrt(spacing / (8.0**4) / slope)
    assert abs(result.action - exact_action) <= slack
    assert abs(result.half_period - exact_half) <= slack


def test_lineb_batches_callback_rows_without_changing_the_solution() -> None:
    positions, energy, tolerance = eckart_seed()
    arguments = (positions, np.ones(1), energy, tolerance, tolerance)
    scalar_sizes = []
    batch_sizes = []

    def scalar_surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scalar_sizes.append(len(x))
        assert len(x) == 1
        return eckart(x)

    def batch_surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_sizes.append(len(x))
        energies, gradients = zip(*(eckart(x[i : i + 1]) for i in range(len(x))))
        return np.concatenate(energies), np.concatenate(gradients)

    scalar = lineb_solve(scalar_surface, *arguments, max_steps=500, beads=8)
    batch = lineb_solve(batch_surface, *arguments, max_steps=500, beads=8, batch_surface=True)

    assert len(positions) in batch_sizes
    assert len(batch_sizes) < len(scalar_sizes)
    assert sum(batch_sizes) == sum(scalar_sizes) == batch.surface_calls["total"]
    assert batch.surface_calls == scalar.surface_calls
    assert batch.iterations == scalar.iterations
    np.testing.assert_array_equal(batch.positions, scalar.positions)
    assert batch.action == scalar.action
    assert batch.temperature == scalar.temperature
    assert batch.log_rate_over_reactant == scalar.log_rate_over_reactant


def test_rate_uses_the_same_energy_zero_as_the_reactant_partition() -> None:
    positions, energy, tol = eckart_seed()

    def solve(offset: float) -> LinebResult:
        def confined(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            value, grad = eckart(x)
            transverse = x.reshape(1, 3)[:, 1:]
            grad[:, 1:] = 2 * transverse
            return value + offset + (transverse**2).sum(), grad

        return lineb_solve(
            confined, positions, np.ones(1), energy + offset, tol, tol,
            molecular=False, max_steps=500, beads=8,
        )

    base = solve(0.0)
    shifted = solve(0.125)
    assert math.isfinite(base.log_rate_over_reactant)
    assert base.rate_calls > 0
    assert base.modes == 8 * 3 - 1
    assert base.negative_modes == 1
    assert shifted.temperature == pytest.approx(base.temperature, rel=1e-7)
    # In the default reduced units k_B = 1, and k Z_r carries exp(-βV).
    difference = shifted.log_rate_over_reactant - base.log_rate_over_reactant
    assert difference == pytest.approx(-0.125 / base.temperature, rel=1e-6)


def test_lineb_solve_rejects_shape_mismatches() -> None:
    positions = np.zeros((3, 6), dtype=np.float64)
    masses = np.array([1.0], dtype=np.float64)

    def surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError(x.shape)

    with pytest.raises(ValueError, match="3 \\* n_atoms"):
        lineb_solve(surface, positions, masses, 0.1, 1e-3, 1e-3, max_steps=1)

    short = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="three images"):
        lineb_solve(surface, short, masses, 0.1, 1e-3, 1e-3, max_steps=1)

    one_atom = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="two atoms"):
        lineb_solve(
            surface,
            one_atom,
            masses,
            0.1,
            1e-3,
            1e-3,
            molecular=True,
            max_steps=1,
        )


def test_lineb_fixed_endpoint_coefficient() -> None:
    positions, energy, _ = eckart_seed()
    result = lineb_solve(eckart, positions, np.ones(1), energy, 0.5, 0.5, endpoint_kappa=0.2)
    assert result.temperature == pytest.approx(1 / (2 * result.half_period))


def test_lineb_solve_preserves_callback_exceptions() -> None:
    positions, energy, tol = eckart_seed(n_images=5)
    masses = np.array([1.0], dtype=np.float64)

    class Boom(RuntimeError):
        pass

    def surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise Boom("lineb-callback")

    with pytest.raises(Boom, match="lineb-callback"):
        lineb_solve(surface, positions, masses, energy, tol, tol, max_steps=2, beads=0)


@pytest.mark.parametrize("beads", [7, 8])
def test_rate_orbit_uses_measured_coordinates_without_surface_calls(beads: int) -> None:
    positions, energy, tol = eckart_seed()
    masses = np.array([2.0])
    observed = {}

    def surface(x):
        value, grad = eckart(x)
        transverse = x.reshape(1, 3)[:, 1:]
        grad[:, 1:] = 2 * transverse
        observed[x.tobytes()] = grad.copy().reshape(-1)
        return value + (transverse**2).sum(), grad

    args = (surface, positions, masses, energy, tol, tol)
    plain = lineb_solve(*args, max_steps=500, beads=beads)
    observed.clear()
    result = lineb_solve(*args, max_steps=500, beads=beads, return_orbit=True)
    assert plain.orbit is None
    assert result.surface_calls == plain.surface_calls
    assert result.log_rate_over_reactant == plain.log_rate_over_reactant
    orbit = result.orbit
    assert orbit is not None
    q = orbit.positions
    assert q.shape == (beads, 3)
    for row, gradient in zip(q, orbit.gradients, strict=True):
        np.testing.assert_array_equal(observed[row.tobytes()], gradient)
    np.testing.assert_array_equal(q[1:], q[:0:-1])
    dt = 2 * result.half_period / beads
    residual = orbit.gradients + masses[0] * (
        (q - np.roll(q, 1, axis=0)) + (q - np.roll(q, -1, axis=0))
    ) / dt**2
    assert orbit.max_abs_residual == pytest.approx(np.abs(residual).max(), abs=1e-13)
    assert orbit.rms_residual == pytest.approx(np.sqrt(np.mean(residual**2)), abs=1e-13)
    assert abs(residual[orbit.max_bead, orbit.max_coordinate]) == pytest.approx(orbit.max_abs_residual)


def test_interpolated_force_clock_uses_one_gradient_per_path_image() -> None:
    positions, energy, tol = eckart_seed()
    evaluations = 0

    def surface(x):
        nonlocal evaluations
        evaluations += x.shape[0]
        value, grad = eckart(x)
        transverse = x.reshape(1, 3)[:, 1:]
        grad[:, 1:] = 2 * transverse
        return value + (transverse**2).sum(), grad

    result = lineb_solve(
        surface, positions, np.ones(1), energy, tol, tol,
        max_steps=500, beads=16, return_orbit=True, interpolated_force_clock=True,
    )
    assert result.surface_calls["total"] == evaluations
    assert result.dynamics_calls == len(positions) + 1
    assert result.turning is not None
    assert result.turning["launch_image"] in (0, len(positions)-1)
    assert result.half_period == result.half_period_dynamics
    assert result.temperature == result.temperature_dynamics
    assert result.orbit is not None
    assert result.orbit.positions.shape == (16, 3)
    assert result.orbit.positions[0, 0] < 0 < result.orbit.positions[8, 0]



@pytest.mark.parametrize("beads", [7, 8])
def test_interpolated_hessian_orbit_reuses_each_measured_replica(beads):
    positions, energy, tol = eckart_seed(5)
    observed = []
    def surface(x):
        value, grad = eckart(x)
        transverse = x[:, 1:]
        grad[:, 1:] = 2 * transverse
        observed.append((x.copy(), grad.copy()))
        return value + (transverse**2).sum(), grad

    args = (surface, positions, np.ones(1), energy, tol, tol)
    kwargs = dict(max_steps=500, beads=beads, interpolated_hessian=True, image_hessian_step=0.005)
    plain = lineb_solve(*args, **kwargs)
    assert plain.surface_calls["total"] == len(observed)
    observed.clear()
    result = lineb_solve(*args, **kwargs, return_orbit=True)
    assert result.surface_calls == plain.surface_calls
    assert result.surface_calls["total"] == len(observed)
    assert result.curvature_evaluations == 2*positions.size
    assert result.bead_evaluations == beads
    assert result.image_hessian_step == 0.005
    assert result.rate_calls == 2*positions.size+beads
    assert result.log_rate_over_reactant == plain.log_rate_over_reactant
    assert result.orbit is not None
    np.testing.assert_array_equal(result.orbit.positions,np.concatenate([row for row, _ in observed[-beads:]]))
    np.testing.assert_array_equal(result.orbit.gradients,np.concatenate([g for _,g in observed[-beads:]]))
