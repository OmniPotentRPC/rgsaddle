"""Compare the ASE-shaped drivers with ASE 3.29 on the same calculators."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.mep import NEB as AseNeb
from ase.mep.dimer import DimerControl, MinModeAtoms as AseMinMode
from rgsaddle.ase import MinModeAtoms, NEB


class Bowl(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        x, y = (float(atoms.positions[0, 0]), float(atoms.positions[0, 1]))
        energy = 0.5 * x * x + 2.0 * y * y
        self.results = {"energy": energy, "forces": np.array([[-x, -4.0 * y, 0.0]])}


def images(points):  # type: ignore[no-untyped-def]
    built = []
    for x, y in points:
        atoms = Atoms("H", positions=[[x, y, 0.0]])
        atoms.calc = Bowl()
        built.append(atoms)
    return built


def test_neb_forces_match_ase_when_nothing_climbs() -> None:
    points = [(0.0, 0.0), (0.4, 0.2), (1.5, 0.0)]
    ase = AseNeb(images(points), k=0.5, climb=False, method="improvedtangent")
    ours = NEB(images(points), k=0.5, climb=False, method="improvedtangent")
    assert np.allclose(ase.get_forces(), ours.get_forces())
    assert ours.get_potential_energy() == ase.get_potential_energy()


def test_a_slope_into_the_band_is_where_ase_climbs_and_we_do_not() -> None:
    points = [(0.0, 0.0), (0.4, 0.2), (1.5, 0.0)]
    ase = AseNeb(images(points), k=0.5, climb=True, method="improvedtangent")
    climbed = NEB(images(points), k=0.5, climb=True, method="improvedtangent")
    plain = NEB(images(points), k=0.5, climb=False, method="improvedtangent")
    ase_force = ase.get_forces()
    our_force = climbed.get_forces()
    # The right endpoint is higher and its force points back into the band,
    # so there is no saddle between it and the interior image. ASE still
    # reflects that image. This driver leaves the spring in place.
    assert not np.allclose(ase_force, our_force)
    assert np.allclose(our_force, plain.get_forces())
    assert climbed.ci is None
    assert ase.imax == 1


def test_dimer_force_matches_ase_on_the_bowl() -> None:
    def bowl_atom():  # type: ignore[no-untyped-def]
        atoms = Atoms("H", positions=[[0.4, 0.3, 0.0]])
        atoms.calc = Bowl()
        return atoms

    control = DimerControl(
        logfile=None,
        eigenmode_method="dimer",
        max_num_rot=4,
        trial_angle=0.01,
        dimer_separation=0.001,
        f_rot_min=1e-4,
        f_rot_max=1e-4,
        initial_eigenmode_method="displacement",
        displacement_method="vector",
    )
    mode = np.array([[1.0, 0.0, 0.0]])
    ase = AseMinMode(bowl_atom(), control=control, eigenmodes=[mode])
    ours = MinModeAtoms(bowl_atom(), mode=[1.0, 0.0, 0.0], kind="dimer", max_rotations=6, dr=0.001)
    assert np.allclose(ase.get_forces(), ours.get_forces())
    assert abs(float(ase.curvatures[0]) - ours.curvature) < 1e-8
