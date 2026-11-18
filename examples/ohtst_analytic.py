"""OH-TST on a separable double well, through the Python bindings.

``V = (x**2 - 1)**2 / 4 + y**2 / 2``. The coordinates are mass-weighted.
The run samples a plane, climbs it, integrates the reversible work, solves
a two-state MBAR problem, shoots a transmission coefficient and assembles
the rate.
"""

from __future__ import annotations

import numpy as np

from rgsaddle import (
    ohtst_mbar,
    ohtst_optimize,
    ohtst_rate,
    ohtst_sample,
    ohtst_transmission,
    ohtst_work,
)


def surface(q: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy and gradient of the double well."""
    x = float(q[0])
    y = float(q[1])
    well = x * x - 1.0
    energy = 0.25 * well * well + 0.5 * y * y
    gradient = np.array([x * well, y], dtype=np.float64)
    return energy, gradient


def main() -> None:
    point = np.array([0.0, 0.0], dtype=np.float64)
    normal = np.array([1.0, 0.0], dtype=np.float64)
    sample = ohtst_sample(surface, point, normal, 0.25, 0.05, 1.0, 4, 1, 8, 2, 3)
    print(
        f"sample: normal force {sample.normal_force:.4f}, "
        f"acceptance {sample.acceptance:.2f}, records {sample.records}"
    )

    start = np.array([0.0, 0.0], dtype=np.float64)
    optimum = ohtst_optimize(
        surface, start, normal, 0.25, 0.05, 1.0, 4, 1, 8, 2, 3, 4, 0.2, 0.1, 1, True, False
    )
    print(f"optimum: point {np.asarray(optimum.point)}, iterations {optimum.iterations}")

    points = np.array([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    work = ohtst_work(
        surface, points, normals, 0.25, 0.05, 1.0, 4, 1, 8, 2, 3, 3, 0, None, False, 0
    )
    print(f"work: {work.work:.4f} ± {work.error:.4f}")

    reduced = np.array([[0.0, 4.0], [0.2, 3.5], [4.0, 0.0], [3.6, 0.1]], dtype=np.float64)
    counts = np.array([2, 2], dtype=np.int64)
    mbar = ohtst_mbar(reduced, counts)
    print(f"mbar: f = {np.asarray(mbar.free_energies)}")

    starts = np.array([[0.0, 0.0], [0.0, 0.2]], dtype=np.float64)
    kappa = ohtst_transmission(surface, point, normal, starts, 0.25, 0.05, 0.0, 20, 5, 4, 2, 5)
    print(f"kappa: {kappa.kappa:.4f} ± {kappa.error:.4f}")

    rate = ohtst_rate(0.25, 1.0, 0.0, work.work, work.error)
    print(f"rate: {rate.rate:.6f} ± {rate.error:.6f}")


if __name__ == "__main__":
    main()
