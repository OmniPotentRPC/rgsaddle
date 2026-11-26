"""Downward pathway map of the separable well through the Python session.

V = (x^4 - 2 x^2) / 4 + (y^4 - y^2) / 4 + z^2 / 2. The origin is an
index-2 saddle. The binding estimates Hessian actions by finite
differences of the surface callback.
"""

from __future__ import annotations

import numpy as np
import pytest

import rgsaddle


def wells(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    a, b, c = rows[:, 0], rows[:, 1], rows[:, 2]
    energy = (a**4 - 2.0 * a * a) / 4.0 + (b**4 - b * b) / 4.0 + c * c / 2.0
    gradient = np.stack([a**3 - a, b**3 - 0.5 * b, c], axis=1)
    return energy, gradient


def test_the_map_from_the_origin_finds_every_critical_point() -> None:
    session = rgsaddle.Landscape(
        np.zeros(3),
        push=0.5,
        duplicate_distance=1.0e-2,
        max_steps=800,
        force_tol=1.0e-8,
        relax_force_tol=1.0e-8,
        max_move=0.1,
        max_atom_move=0.1,
        initial_step=0.05,
    )
    pathway = session.run(wells)
    assert isinstance(pathway, rgsaddle.PathwayMap)
    assert len(pathway.nodes) == 9
    assert len(pathway.edges) == 12
    assert pathway.nodes[0].index == 2
    assert pathway.evaluations == pathway.nodes[0].evaluations + sum(edge.evaluations for edge in pathway.edges)
    side = float(np.sqrt(0.5))
    targets = {
        (0.0, 0.0, 0.0): 2,
        (1.0, 0.0, 0.0): 1,
        (-1.0, 0.0, 0.0): 1,
        (0.0, side, 0.0): 1,
        (0.0, -side, 0.0): 1,
        (1.0, side, 0.0): 0,
        (1.0, -side, 0.0): 0,
        (-1.0, side, 0.0): 0,
        (-1.0, -side, 0.0): 0,
    }
    for node in pathway.nodes:
        matched = None
        for target, index in targets.items():
            if np.max(np.abs(node.position - np.array(target))) <= 1.0e-3:
                matched = index
                break
        assert matched is not None, node.position
        assert node.index == matched
    print(
        "PY_WELLS",
        pathway.evaluations,
        [(node.index, int(node.evaluations)) for node in pathway.nodes],
    )


def test_hessian_receives_the_geometry() -> None:
    seen: list[np.ndarray] = []

    def hessian(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        point = np.asarray(x, dtype=np.float64).reshape(-1)
        direction = np.asarray(v, dtype=np.float64).reshape(-1)
        seen.append(point.copy())
        a, b, c = point
        return np.array(
            [
                (3.0 * a * a - 1.0) * direction[0],
                (3.0 * b * b - 0.5) * direction[1],
                direction[2],
            ]
        )

    session = rgsaddle.Landscape(
        np.zeros(3),
        max_evaluations=2,
        push=0.5,
        force_tol=1.0e-8,
    )
    with pytest.raises(ValueError, match="LANDSCAPE_BOUND"):
        session.run(wells, hessian)
    assert seen
    assert np.allclose(seen[0], 0.0)


def test_the_evaluation_limit_is_an_error() -> None:
    session = rgsaddle.Landscape(np.zeros(3), max_evaluations=1, push=0.5, force_tol=1.0e-8)
    with pytest.raises(ValueError, match="LANDSCAPE_BOUND"):
        session.run(wells)
