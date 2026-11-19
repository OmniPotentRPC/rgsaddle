"""FixAtoms on ASE images is the band's fixed-atom mask."""

from __future__ import annotations

import os

import numpy as np
import pytest

# RGSADDLE_REQUIRE_ASE=1 turns the skip into a collection error.
if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from rgsaddle._core import project_band
from rgsaddle.ase import NEB

TETHER = np.array([[0.3, 0.2, 0.1], [2.6, -0.2, 0.0]])


class Tethered(Calculator):
    """Atoms 0 and 1 pulled off their sites, atom 2 in a double well."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        r = atoms.positions
        d = r[:2] - TETHER
        x, y, z = r[2, 0] - 1.0, r[2, 1] - 1.5, r[2, 2]
        energy = 0.5 * float(np.sum(d * d)) + (x * x - 1.0) ** 2 + 2.0 * y * y + 2.0 * z * z
        forces = np.zeros_like(r)
        forces[:2] = -d
        forces[2] = [-4.0 * x * (x * x - 1.0), -4.0 * y, -4.0 * z]
        self.results = {"energy": energy, "forces": forces}


def _band(n_images: int = 7, fix: list[int] | None = None) -> list[Atoms]:
    images = []
    for i in range(n_images):
        t = i / (n_images - 1)
        atoms = Atoms(
            "H3",
            positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [2.0 * t, 1.5 + 0.3 * np.sin(np.pi * t), 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        if fix is not None:
            atoms.set_constraint(FixAtoms(indices=fix))
        atoms.calc = Tethered()
        images.append(atoms)
    return images


def test_fix_atoms_keeps_every_bit_through_a_climbing_run() -> None:
    images = _band(fix=[0, 1])
    start = [image.get_positions()[:2].copy() for image in images]
    neb = NEB(images, k=1.0, climb=True, remove_rotation_and_translation=True)
    forces = neb.get_forces().reshape(len(images) - 2, 3, 3)
    assert np.all(forces[:, :2] == 0.0)
    assert np.any(forces[:, 2] != 0.0)
    assert FIRE(neb, logfile=None).run(fmax=1e-3, steps=2000)
    for image, before in zip(images, start):
        assert np.array_equal(image.get_positions()[:2], before)
    assert neb.ci is not None
    assert abs(images[neb.ci].positions[2, 0] - 1.0) < 1e-2


def test_interpolate_uses_ase_constraint_modes() -> None:
    images = _band(fix=[0, 1])
    for image in images[1:-1]:
        image.positions[:2] += 0.5
    neb = NEB(images, k=1.0)
    with pytest.raises(RuntimeError, match="Constraints in image 1"):
        neb.interpolate()
    shifted = [image.positions[:2].copy() for image in images]
    neb.interpolate(apply_constraint=True)
    for image, before in zip(images, shifted):
        assert np.array_equal(image.positions[:2], before)
    neb.interpolate(apply_constraint=False)
    for image in images:
        assert np.array_equal(image.positions[:2], images[0].positions[:2])


def test_fix_atoms_must_agree_between_images() -> None:
    images = _band(fix=[0, 1])
    images[3].set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="image 3"):
        NEB(images, k=1.0).get_forces()
    images = _band(fix=[0, 1])
    images[-1].positions[0, 0] += 1e-6
    with pytest.raises(Exception):
        NEB(images, k=1.0).get_forces()


def test_project_band_takes_a_mask_and_refuses_a_bad_one() -> None:
    images = _band()
    positions = np.stack([image.get_positions().reshape(-1) for image in images])
    energies = np.array([image.get_potential_energy() for image in images])
    gradients = -np.stack([image.get_forces().reshape(-1) for image in images])
    plain, _ = project_band(positions, energies, gradients, k=1.0)
    same, _ = project_band(positions, energies, gradients, k=1.0, fixed=None)
    assert np.array_equal(plain, same)
    masked, _ = project_band(
        positions, energies, gradients, k=1.0, fixed=np.array([1, 1, 0], dtype=np.uint8)
    )
    rows = np.asarray(masked).reshape(len(images), 3, 3)
    assert np.all(rows[:, :2] == 0.0)
    with pytest.raises(ValueError):
        project_band(positions, energies, gradients, fixed=np.array([1, 0], dtype=np.uint8))
    with pytest.raises(Exception):
        project_band(positions, energies, gradients, fixed=np.array([1, 2, 0], dtype=np.uint8))
