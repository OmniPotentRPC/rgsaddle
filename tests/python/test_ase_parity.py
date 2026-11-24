"""Same script, ASE's class and rgsaddle's, on a cheap calculator.

Force-call counts are printed. They are not a speed claim.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.mep.autoneb import AutoNEB as AseAutoNEB
from ase.optimize import FIRE

from rgsaddle.ase import AutoNEB, _periodic_cell

# Müller-Brown saddle between minima B and C, from the AutoNEB example.
_SADDLE_XY = np.array([0.212, 0.293])


class Flat(Calculator):
    """Zero energy and force, for tests that only place atoms."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def _muller(x, y):  # type: ignore[no-untyped-def]
    a = np.array([-200.0, -100.0, -170.0, 15.0])
    aa = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])
    energy = 0.0
    gx = 0.0
    gy = 0.0
    for i in range(4):
        dx = x - x0[i]
        dy = y - y0[i]
        weight = a[i] * np.exp(aa[i] * dx * dx + b[i] * dx * dy + c[i] * dy * dy)
        energy += weight
        gx += weight * (2.0 * aa[i] * dx + b[i] * dy)
        gy += weight * (b[i] * dx + 2.0 * c[i] * dy)
    return float(energy), float(gx), float(gy)


class MullerBrown(Calculator):
    implemented_properties = ["energy", "forces"]
    calls = 0

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        MullerBrown.calls += 1
        x, y = (float(atoms.positions[0, 0]), float(atoms.positions[0, 1]))
        energy, gx, gy = _muller(x, y)
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = -gx
        forces[0, 1] = -gy
        self.results = {"energy": energy, "forces": forces}


class TightFIRE(FIRE):
    def __init__(self, atoms, maxstep=0.05, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(atoms, maxstep=maxstep, **kwargs)


def test_autoneb_saddle_matches() -> None:
    start = np.array([-0.05001082, 0.46669410, 0.0])
    end = np.array([0.62349940, 0.02803776, 0.0])

    def run(cls, root):  # type: ignore[no-untyped-def]
        MullerBrown.calls = 0
        prefix = os.path.join(root, "neb")
        os.makedirs(root, exist_ok=True)
        from ase.io import write

        for index, xyz in enumerate((start, end)):
            atoms = Atoms("H", positions=[xyz])
            atoms.calc = MullerBrown()
            atoms.get_potential_energy()
            write(f"{prefix}{index:03d}.traj", atoms)

        def attach(images):  # type: ignore[no-untyped-def]
            for atoms in images:
                atoms.calc = MullerBrown()

        band = cls(
            attach,
            prefix,
            n_simul=3,
            n_max=7,
            fmax=[0.5, 0.05],
            maxsteps=[40, 80],
            k=5.0,
            climb=True,
            method="improvedtangent",
            optimizer=TightFIRE,
            remove_rotation_and_translation=False,
            space_energy_ratio=0.5,
            parallel=False,
            smooth_curve=False,
            interpolate_method="linear",
        )
        band.run()
        energies = []
        positions = []
        for atoms in band.all_images:
            atoms.calc = MullerBrown()
            energies.append(float(atoms.get_potential_energy()))
            positions.append(atoms.positions[0, :2].copy())
        interior = list(range(1, len(energies) - 1))
        peak = max(interior, key=lambda i: energies[i])
        print(
            f"AutoNEB {cls.__name__} calls={MullerBrown.calls} "
            f"peak={positions[peak]} energy={energies[peak]:.6f}"
        )
        return np.asarray(positions[peak]), energies[peak], MullerBrown.calls

    import tempfile

    ase_saddle, ase_energy, ase_calls = run(AseAutoNEB, tempfile.mkdtemp(prefix="ase-autoneb-"))
    our_saddle, our_energy, our_calls = run(AutoNEB, tempfile.mkdtemp(prefix="rgsaddle-autoneb-"))
    assert np.linalg.norm(ase_saddle - _SADDLE_XY) < 0.05
    assert np.linalg.norm(our_saddle - _SADDLE_XY) < 0.05
    assert our_energy == pytest.approx(ase_energy, abs=1.0)
    assert ase_calls > 0 and our_calls > 0
    with pytest.raises(NotImplementedError):
        AutoNEB(
            lambda images: None,
            "unused",
            n_simul=3,
            n_max=7,
            smooth_curve=True,
        )


_PERIODIC_CASES = {
    "slab": ([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]], [True, True, False]),
    "wire": ([[4.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], [True, False, False]),
    "skewed_slab": ([[4.0, 0.0, 0.0], [2.0, 3.5, 0.0], [1.0, 1.0, 10.0]], [True, True, False]),
}


def _hopping(cell, pbc, ends):  # type: ignore[no-untyped-def]
    """Two anchor atoms and an adatom that moves between ``ends``."""
    images = []
    for x, z in ends:
        atoms = Atoms(
            "H3",
            positions=[[0.5, 1.0, 0.0], [2.0, 1.0, 0.0], [x, 1.0, z]],
            cell=cell,
            pbc=pbc,
        )
        atoms.calc = Flat()
        images.append(atoms)
    return images


def test_periodic_cell_completes_the_open_axes() -> None:
    cell, pbc = _PERIODIC_CASES["skewed_slab"]
    images = _hopping(cell, pbc, [(0.6, 1.0), (2.0, 1.0), (3.6, 1.0)])
    built = _periodic_cell(images)
    assert built is not None
    assert abs(np.linalg.det(built)) > 1.0
    assert np.allclose(built[:2], np.asarray(cell)[:2])
    # The open row is normal to both periodic rows and longer than the images span.
    assert np.allclose(built[2] @ built[0], 0.0, atol=1e-9)
    assert np.allclose(built[2] @ built[1], 0.0, atol=1e-9)
    assert np.linalg.norm(built[2]) > 10.0

    wire = _hopping(*_PERIODIC_CASES["wire"], [(0.6, 1.0), (2.0, 1.0), (3.6, 1.0)])
    built = _periodic_cell(wire)
    assert built is not None
    assert abs(np.linalg.det(built)) > 1.0
    assert np.allclose(built[0], [4.0, 0.0, 0.0])
    assert abs(built[1] @ built[0]) < 1e-9 and abs(built[2] @ built[0]) < 1e-9
    assert abs(built[1] @ built[2]) < 1e-9

    free = _hopping(cell, [False, False, False], [(0.6, 1.0), (2.0, 1.0), (3.6, 1.0)])
    assert _periodic_cell(free) is None
    full = _hopping(cell, [True, True, True], [(0.6, 1.0), (2.0, 1.0), (3.6, 1.0)])
    assert np.allclose(_periodic_cell(full), np.asarray(cell))


def test_periodic_cell_refuses_dependent_periodic_vectors() -> None:
    flat = [[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
    images = _hopping(flat, [True, True, False], [(0.6, 1.0), (2.0, 1.0), (3.6, 1.0)])
    with pytest.raises(ValueError, match="linearly independent"):
        _periodic_cell(images)
