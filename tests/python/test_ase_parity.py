"""Same script, ASE's class and rgsaddle's, on a cheap calculator.

Force-call counts are printed. They are not a speed claim.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.mep.autoneb import AutoNEB as AseAutoNEB
from ase.optimize import FIRE

from rgsaddle._core import autoneb_run
from rgsaddle.ase import AutoNEB, _periodic_cell

# Müller-Brown saddle between minima B and C, from the AutoNEB example.
_SADDLE_XY = np.array([0.212, 0.293])


class ThreadCounter:
    """Force-call count for the calling thread.

    One search creates and discards several calculators, so the count
    cannot live on a single instance. A class integer is shared by every
    thread running the test, and a parallel run resets it to zero between
    the search and the read.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def reset(self) -> None:
        self._local.value = 0

    def increment(self) -> None:
        self._local.value = getattr(self._local, "value", 0) + 1

    def get(self) -> int:
        return int(getattr(self._local, "value", 0))


class CountedCalculator(Calculator):
    _calls: ThreadCounter

    @classmethod
    def reset_calls(cls) -> None:
        cls._calls.reset()

    @classmethod
    def call_count(cls) -> int:
        return cls._calls.get()

    def _count_call(self) -> None:
        type(self)._calls.increment()


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


class MullerBrown(CountedCalculator):
    implemented_properties = ["energy", "forces"]
    _calls = ThreadCounter()

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        self._count_call()
        x, y = (float(atoms.positions[0, 0]), float(atoms.positions[0, 1]))
        energy, gx, gy = _muller(x, y)
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = -gx
        forces[0, 1] = -gy
        self.results = {"energy": energy, "forces": forces}


def _count_on_this_thread(
    calc_cls: type[CountedCalculator],
    n: int,
    phase: threading.Barrier,
    seen: list[int],
) -> None:
    calc_cls.reset_calls()
    phase.wait()
    for index in range(n):
        atoms = Atoms("H", positions=[[0.1 + 0.01 * index, 0.0, 0.0]])
        atoms.calc = calc_cls()
        atoms.get_potential_energy()
    phase.wait()
    seen.append(calc_cls.call_count())


def test_force_call_counts_stay_on_their_thread() -> None:
    """A class counter reads zero when another thread resets it."""
    for calc_cls, left, right in (
        (MullerBrown, 1, 6),
    ):
        phase = threading.Barrier(2)
        seen_left: list[int] = []
        seen_right: list[int] = []
        first = threading.Thread(
            target=_count_on_this_thread, args=(calc_cls, left, phase, seen_left)
        )
        second = threading.Thread(
            target=_count_on_this_thread, args=(calc_cls, right, phase, seen_right)
        )
        first.start()
        second.start()
        first.join()
        second.join()
        assert seen_left == [left], calc_cls.__name__
        assert seen_right == [right], calc_cls.__name__


class TightFIRE(FIRE):
    def __init__(self, atoms, maxstep=0.05, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(atoms, maxstep=maxstep, **kwargs)


def test_autoneb_saddle_matches() -> None:
    start = np.array([-0.05001082, 0.46669410, 0.0])
    end = np.array([0.62349940, 0.02803776, 0.0])

    def run(cls, root):  # type: ignore[no-untyped-def]
        MullerBrown.reset_calls()
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
            f"AutoNEB {cls.__name__} calls={MullerBrown.call_count()} "
            f"peak={positions[peak]} energy={energies[peak]:.6f}"
        )
        return np.asarray(positions[peak]), energies[peak], MullerBrown.call_count()

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


@pytest.mark.parametrize("n_max", [-1, -(2**40), 1, 2**62])
def test_autoneb_run_validates_n_max_before_allocating(n_max: int) -> None:
    calls: list[int] = []

    def surface(x):  # type: ignore[no-untyped-def]
        calls.append(int(x.shape[0]))
        return np.zeros(x.shape[0]), np.zeros_like(x)

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="n_max"):
        autoneb_run(surface, positions, n_simul=1, n_max=n_max)
    assert calls == []


@pytest.mark.parametrize(("n_simul", "n_max"), [(3, -1), (3, 4), (0, 7), (-1, 7)])
def test_autoneb_constructor_validates_the_image_counts(n_simul: int, n_max: int) -> None:
    with pytest.raises(ValueError, match="n_max|n_simul"):
        AutoNEB(lambda images: None, "unused", n_simul=n_simul, n_max=n_max)

