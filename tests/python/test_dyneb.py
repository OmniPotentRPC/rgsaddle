"""DyNEB reads the band once, then moves every image that is still free."""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator

from rgsaddle.ase import DyNEB


class Bowl(Calculator):
    """Harmonic bowl. ``calls`` counts calculator evaluations."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, images: list[Atoms], seen: list[np.ndarray], calls: list[int]) -> None:
        super().__init__()
        self._images = images
        self._seen = seen
        self._calls = calls

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        self._calls[0] += 1
        self._seen.append(np.concatenate([image.positions.copy() for image in self._images], axis=0))
        r = atoms.positions
        energy = float(2.0 * r[0, 0] ** 2 + 5.0 * r[0, 1] ** 2)
        forces = np.zeros_like(r)
        forces[0, 0] = -4.0 * r[0, 0]
        forces[0, 1] = -10.0 * r[0, 1]
        self.results = {"energy": energy, "forces": forces}


def _chain(xs: list[float]) -> tuple[list[Atoms], list[int], list[np.ndarray]]:
    images = [Atoms("H", positions=[[x, 0.15 * x, 0.0]]) for x in xs]
    calls = [0]
    seen: list[np.ndarray] = []
    for atoms in images:
        atoms.calc = Bowl(images, seen, calls)
    return images, calls, seen


def test_set_positions_evaluates_the_band_once() -> None:
    images, calls, seen = _chain([-0.8, -0.2, 0.15, 0.55, 1.1])
    neb = DyNEB(images, k=0.1, climb=False, fmax=0.05, method="improvedtangent")
    neb.dynamic_relaxation = False
    raw = np.asarray(neb.get_forces(), dtype=np.float64).reshape(3, 1, 3)
    neb.dynamic_relaxation = True
    per_image = [float(np.sqrt((block**2).sum(axis=1)).max()) for block in raw]
    quiet, louder = sorted(per_image)[:2]
    assert louder > quiet
    # Freeze the quietest image and leave an earlier image free, so a
    # force read after the first move would see mixed coordinates.
    neb.fmax = 0.5 * (quiet + louder)
    free = [value >= neb.fmax for value in per_image]
    assert free[0] and not all(free)

    before = [image.positions.copy() for image in images]
    calls[0] = 0
    seen.clear()
    for image in images:
        image.calc.results.clear()
    proposed = np.asarray(neb.get_positions(), dtype=np.float64) + np.array([0.2, -0.05, 0.0])
    neb.set_positions(proposed)

    n_images = len(images)
    # One band evaluation. The old loop evaluated once per interior image.
    assert calls[0] == n_images, calls[0]
    original = np.concatenate(before, axis=0)
    assert seen, "set_positions did not evaluate the band"
    for snapshot in seen:
        assert np.array_equal(snapshot, original)

    proposed_rows = proposed.reshape(-1, 3)
    for index, image in enumerate(images[1:-1]):
        if free[index]:
            assert np.allclose(image.positions, proposed_rows[index])
        else:
            assert np.array_equal(image.positions, before[index + 1])
