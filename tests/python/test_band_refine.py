"""Per-step refine stepper on the Python band."""

from __future__ import annotations

import numpy as np
import pytest

import rgsaddle


def _bowl(positions: np.ndarray):
    energy = np.sum(0.5 * positions * positions, axis=1)
    return np.ascontiguousarray(energy), np.ascontiguousarray(positions.copy())


def _band() -> rgsaddle.Band:
    images = np.zeros((5, 3), dtype=np.float64)
    images[:, 0] = np.linspace(-1.0, 1.0, 5)
    # Bow the interior off the line so the projected force is not zero.
    # A straight path in this bowl is already at the NEB force of zero.
    images[1:-1, 1] = 0.4
    return rgsaddle.Band(
        np.ascontiguousarray(images),
        climb=False,
        k=0.1,
        force_tol=1e-8,
        max_move=0.2,
        remove_translation=False,
    )


def test_set_refine_steps_with_lbfgs_not_fire() -> None:
    fire = _band()
    refined = _band()
    refined.set_refine(True, threshold=1.0e6, memory=20)
    assert refined.is_refining() is False
    fire.step(_bowl)
    refined.step(_bowl)
    assert fire.is_refining() is False
    assert refined.is_refining() is True
    assert not np.allclose(fire.positions(), refined.positions())


def test_set_refine_rejects_a_non_positive_threshold() -> None:
    band = _band()
    with pytest.raises(ValueError):
        band.set_refine(True, threshold=-1.0)
    band.set_refine(False, threshold=-1.0)
    band.step(_bowl)
    assert band.is_refining() is False
