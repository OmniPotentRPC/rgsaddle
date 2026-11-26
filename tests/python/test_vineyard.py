import math
import numpy as np
import pytest

import rgsaddle


def _coupled_barrier_surface(x: np.ndarray):
    pos = np.asarray(x, dtype=np.float64).reshape(-1)
    q, y, z = pos[0], pos[1], pos[2]
    k_trans = 1.4
    k_sad = 2.6
    # V(q, y, z) = (q^2 - 1)^2 + 0.5*(k_trans + alpha*q^2)*y^2 + 0.5*k_sad*z^2
    # Minimum at (1, 0, 0): V_qq = 8.0, V_yy = k_trans + 0.4, V_zz = k_sad
    # Saddle at (0, 0, 0): V_qq = -4.0, V_yy = k_trans, V_zz = k_sad
    alpha = 0.4
    e = (q * q - 1.0) ** 2 + 0.5 * (k_trans + alpha * q * q) * y * y + 0.5 * k_sad * z * z
    gq = 4.0 * q * (q * q - 1.0) + alpha * q * y * y
    gy = (k_trans + alpha * q * q) * y
    gz = k_sad * z
    return np.array([e], dtype=np.float64), np.array([[gq, gy, gz]], dtype=np.float64)


def test_atomic_mass_lookup_matches_standard_values():
    cu_sym = rgsaddle.atomic_mass("Cu")
    cu_z = rgsaddle.atomic_mass(29)
    assert math.isclose(cu_sym, 63.546, rel_tol=0.0, abs_tol=1e-9)
    assert cu_sym == cu_z
    assert math.isclose(rgsaddle.atomic_mass("Fe"), 55.845, abs_tol=1e-9)
    with pytest.raises(ValueError):
        rgsaddle.atomic_mass("Xx")
    with pytest.raises(ValueError):
        rgsaddle.atomic_mass(0)
    with pytest.raises(TypeError):
        rgsaddle.atomic_mass(1.5)


def test_vineyard_prefactor_fd_analytic_and_schemes():
    mass_cu = rgsaddle.atomic_mass("Cu")
    masses = np.array([mass_cu], dtype=np.float64)
    x_min = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_sad = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    rep_central = rgsaddle.vineyard_prefactor_fd(
        _coupled_barrier_surface, x_min, x_sad, masses, step=1e-3, difference="central"
    )
    rep_forward = rgsaddle.vineyard_prefactor_fd(
        _coupled_barrier_surface, x_min, x_sad, masses, step=1e-4, difference="forward"
    )

    omega2_si = 1.602176634e-19 / 1e-20 / 1.66053906660e-27
    # Analytic eigenvalues at min: (8.0/m, 1.8/m, 2.6/m); at saddle: (-4.0/m, 1.4/m, 2.6/m)
    expected_hz = (
        math.sqrt(omega2_si) / (2.0 * math.pi)
    ) * math.sqrt((8.0 / mass_cu) * ((1.8 / mass_cu) / (1.4 / mass_cu)))

    assert rep_central.n_min_modes == 3
    assert rep_central.n_saddle_modes == 2
    assert math.isclose(rep_central.unstable_eigenvalue, -4.0 / mass_cu, rel_tol=1e-4)
    assert math.isclose(rep_central.prefactor_hz, expected_hz, rel_tol=1e-5)
    assert math.isclose(rep_central.nu0_thz, rep_central.prefactor_hz / 1e12, rel_tol=1e-12)
    assert math.isclose(rep_forward.prefactor_hz, rep_central.prefactor_hz, rel_tol=1e-3)
    assert rep_central.evaluations > rep_forward.evaluations
