"""ASE-shaped NEB, minimum-mode, and Sella drivers."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE

from rgsaddle.ase import MinModeAtoms, NEB, Sella


class Bowl(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        r = atoms.positions
        energy = 2.0 * r[0, 0] ** 2 + 5.0 * r[0, 1] ** 2
        forces = np.zeros_like(r)
        forces[0, 0] = -4.0 * r[0, 0]
        forces[0, 1] = -10.0 * r[0, 1]
        self.results = {"energy": float(energy), "forces": forces}


class Slope(Calculator):
    """Energy increases with x. The last image can slope into the band."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, into: bool, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.into = into

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        x = float(atoms.positions[0, 0])
        forces = np.zeros_like(atoms.positions)
        if x > 1.5:
            forces[0, 0] = -1.0 if self.into else 1.0
        elif 0.5 < x < 1.5:
            forces[0, 0] = -1.0
        self.results = {"energy": x, "forces": forces}


def _chain(calc_factory, xs):  # type: ignore[no-untyped-def]
    images = []
    for x in xs:
        atoms = Atoms("H", positions=[[x, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=False)
        atoms.calc = calc_factory()
        images.append(atoms)
    return images


def test_interpolate_places_the_interior_on_the_segment() -> None:
    images = _chain(Bowl, [-1.0, 0.0, 1.0])
    images[1].positions[0, 0] = 4.0
    neb = NEB(images, k=0.1, climb=False)
    neb.interpolate()
    assert images[1].positions[0, 0] == 0.0
    try:
        neb.interpolate(method="idpp")
    except ValueError as err:
        assert "idpp_path" in str(err)
    else:
        raise AssertionError("idpp interpolation was accepted")


def test_a_slope_into_the_band_does_not_climb() -> None:
    into = NEB(_chain(lambda: Slope(into=True), [0.0, 1.0, 2.0]), k=0.1, climb=True)
    away = NEB(_chain(lambda: Slope(into=False), [0.0, 1.0, 2.0]), k=0.1, climb=True)
    into_force = into.get_forces()
    away_force = away.get_forces()
    assert into.ci is None
    assert abs(float(into_force[0, 0])) < 1e-8
    assert away.ci == 1
    assert float(away_force[0, 0]) > 0.5


def test_neb_is_an_ase_optimizable() -> None:
    images = _chain(Bowl, [-0.4, 0.0, 0.4])
    neb = NEB(images, k=0.5, climb=False)
    neb.interpolate()
    opt = FIRE(neb, logfile=None)
    opt.run(fmax=0.2, steps=8)
    assert np.isfinite(neb.get_positions()).all()


def test_min_mode_drops_the_perpendicular_force_above_the_well() -> None:
    atoms = Atoms("H", positions=[[0.4, 0.3, 0.0]])
    atoms.calc = Bowl()
    search = MinModeAtoms(atoms, mode=[1.0, 0.0, 0.0], kind="dimer", max_rotations=6)
    forces = search.get_forces()
    assert abs(float(forces[0, 1])) < 1e-8
    assert abs(float(forces[0, 0])) > 0.5
    assert search.curvature > 0.0


def test_sella_step_stays_finite() -> None:
    atoms = Atoms("H3", positions=[[0.2, 0.0, 0.0], [1.2, 0.1, 0.0], [0.4, 1.1, 0.0]])
    atoms.calc = Bowl()
    opt = Sella(atoms, order=1, delta=0.05)
    opt.run(fmax=1.0, steps=2)
    assert np.isfinite(atoms.positions).all()
