"""Run-to-completion AutoNEB through the nanobind wrapper."""

from __future__ import annotations

import numpy as np

from rgsaddle._core import autoneb_run


def _double_well(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E = (x^2 - 1)^2 + 2 y^2 + 2 z^2, as a gradient."""
    rows = np.ascontiguousarray(x, dtype=np.float64)
    coord = rows.reshape(rows.shape[0], -1, 3)[:, 0, :]
    xx = coord[:, 0]
    yy = coord[:, 1]
    zz = coord[:, 2]
    energy = (xx * xx - 1.0) ** 2 + 2.0 * yy * yy + 2.0 * zz * zz
    gradient = np.zeros_like(rows)
    flat = gradient.reshape(rows.shape[0], -1, 3)
    flat[:, 0, 0] = 4.0 * xx * (xx * xx - 1.0)
    flat[:, 0, 1] = 4.0 * yy
    flat[:, 0, 2] = 4.0 * zz
    return energy, gradient


def test_autoneb_run_reports_status_force_and_iteration() -> None:
    """A short double well reports a converged stage."""
    positions = np.array(
        [[-1.0, 0.0, 0.0], [0.1, 0.4, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    report = autoneb_run(
        _double_well,
        positions,
        k=5.0,
        climb=True,
        n_simul=1,
        n_max=5,
        force_tol=1e-3,
        local_force_tol=5e-2,
        local_max_steps=400,
        climb_max_steps=400,
        projection="neb",
        tangent="improved",
        max_move=0.1,
        space_energy_ratio=0.5,
        idpp=False,
    )
    assert report.status == 1
    assert 0.0 <= report.max_force <= 1e-3
    assert 0 < report.iteration <= 400
    images = getattr(report, "images", None)
    if images is not None:
        assert len(images) == report.n_images
