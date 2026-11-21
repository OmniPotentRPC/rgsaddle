"""Same script, ASE's class and rgsaddle's, on a cheap calculator.

Force-call counts are printed. They are not a speed claim.
"""

from __future__ import annotations

import inspect
import os
import threading

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixConstraint
from ase.mep import NEB as AseNeb
from ase.mep.autoneb import AutoNEB as AseAutoNEB
from ase.mep.dimer import DimerControl as AseDimerControl
from ase.mep.dimer import MinModeAtoms as AseMinMode
from ase.mep.dimer import MinModeTranslate as AseTranslate
from ase.mep.neb import DyNEB as AseDyNEB
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

from rgsaddle._core import MinMode, autoneb_run
from rgsaddle.ase import (
    AutoNEB,
    DimerControl,
    DyNEB,
    MinModeAtoms,
    MinModeTranslate,
    NEB,
    UnsupportedFeature,
    _periodic_cell,
)

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


class Bowl(CountedCalculator):
    implemented_properties = ["energy", "forces"]
    _calls = ThreadCounter()

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        self._count_call()
        x, y = (float(atoms.positions[0, 0]), float(atoms.positions[0, 1]))
        self.results = {
            "energy": 0.5 * x * x + 2.0 * y * y,
            "forces": np.array([[-x, -4.0 * y, 0.0]]),
        }


class Bond(Calculator):
    """Two atoms, a unit bond, and a soft bend toward the product."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        bond = atoms.positions[1] - atoms.positions[0]
        length = float(np.linalg.norm(bond))
        stretch = length - 1.0
        energy = 0.5 * stretch * stretch
        forces = np.zeros((2, 3))
        if length > 0.0:
            direction = bond / length
            forces[0] = direction * stretch
            forces[1] = -direction * stretch
        self.results = {"energy": energy, "forces": forces}


class QuadraticSaddle(CountedCalculator):
    """E = -2 x^2 + 5 y^2. The index-1 saddle is the origin."""

    implemented_properties = ["energy", "forces"]
    _calls = ThreadCounter()

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        self._count_call()
        x, y = (float(atoms.positions[0, 0]), float(atoms.positions[0, 1]))
        self.results = {
            "energy": -2.0 * x * x + 5.0 * y * y,
            "forces": np.array([[4.0 * x, -10.0 * y, 0.0]]),
        }


class Flat(Calculator):
    """Zero energy and force, for tests that only place atoms."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


class Nudge(FixConstraint):
    """Moves the first atom's x when the constraint is applied."""

    def adjust_positions(self, atoms, positions):  # type: ignore[no-untyped-def]
        del atoms
        positions[0, 0] += 0.25

    def adjust_forces(self, atoms, forces):  # type: ignore[no-untyped-def]
        del atoms, forces


def _images(calc_factory, points):  # type: ignore[no-untyped-def]
    built = []
    for point in points:
        atoms = Atoms("H", positions=[[point[0], point[1], 0.0]])
        atoms.calc = calc_factory()
        built.append(atoms)
    return built


def _pair(factory):  # type: ignore[no-untyped-def]
    left = _images(factory, [(0.0, 0.0), (0.4, 0.2), (1.5, 0.0)])
    right = _images(factory, [(0.0, 0.0), (0.4, 0.2), (1.5, 0.0)])
    return left, right


def test_aseneb_is_the_improved_tangent_not_ases_legacy() -> None:
    """ASE's aseneb keeps an unpublished tangent. This name uses the improved one."""
    ase_images, our_images = _pair(Bowl)
    _, improved_images = _pair(Bowl)
    ase = AseNeb(ase_images, k=0.5, climb=False, method="aseneb")
    ours = NEB(our_images, k=0.5, climb=False, method="aseneb")
    improved = NEB(improved_images, k=0.5, climb=False, method="improvedtangent")
    assert np.allclose(ours.get_forces(), improved.get_forces())
    assert not np.allclose(ase.get_forces(), ours.get_forces())


def test_eb_is_the_plain_elastic_band() -> None:
    """ASE's eb is the full spring of Paper III. This name is the plain elastic band."""
    ase_images, our_images = _pair(Bowl)
    ase = AseNeb(ase_images, k=0.5, climb=False, method="eb")
    ours = NEB(our_images, k=0.5, climb=False, method="eb")
    assert np.isfinite(ours.get_forces()).all()
    assert not np.allclose(ase.get_forces(), ours.get_forces())


def test_a_uniform_spring_list_matches_the_scalar() -> None:
    ase_images, our_images = _pair(Bowl)
    ase = AseNeb(ase_images, k=[0.5, 0.5], climb=False, method="improvedtangent")
    ours = NEB(our_images, k=[0.5, 0.5], climb=False, method="improvedtangent")
    assert np.allclose(ase.get_forces(), ours.get_forces())
    with pytest.raises(UnsupportedFeature):
        NEB(our_images, k=[0.5, 0.2], method="eb")


def test_unset_method_is_improvedtangent() -> None:
    ase_images, our_images = _pair(Bowl)
    with pytest.warns(UserWarning):
        ase = AseNeb(ase_images, k=0.5, climb=False)
    with pytest.warns(UserWarning):
        ours = NEB(our_images, k=0.5, climb=False)
    assert np.allclose(ase.get_forces(), ours.get_forces())


def test_spline_string_and_precon_are_refused() -> None:
    images = _images(Bowl, [(0.0, 0.0), (0.4, 0.0), (1.0, 0.0)])
    for method in ("spline", "string"):
        with pytest.raises(UnsupportedFeature):
            NEB(images, method=method)
    with pytest.raises(UnsupportedFeature):
        NEB(images, method="eb", precon=object())
    with pytest.raises(UnsupportedFeature):
        NEB(images, parallel=True, method="eb")


def test_linear_mic_matches() -> None:
    def chain(cls):  # type: ignore[no-untyped-def]
        images = []
        for x in (0.1, 0.0, 1.9):
            atoms = Atoms("H", positions=[[x, 0.0, 0.0]], cell=[2.0, 8.0, 8.0], pbc=[True, False, False])
            atoms.calc = Bowl()
            images.append(atoms)
        band = cls(images, k=0.1, climb=False, method="improvedtangent")
        band.interpolate(method="linear", mic=True)
        return images[1].positions.copy()

    assert np.allclose(chain(AseNeb), chain(NEB))


def test_idpp_keeps_the_bond() -> None:
    def placed(cls):  # type: ignore[no-untyped-def]
        images = []
        ends = (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        for positions in ends:
            atoms = Atoms("H2", positions=positions)
            atoms.calc = Bond()
            images.append(atoms)
        band = cls(images, k=0.1, climb=False, method="improvedtangent")
        band.interpolate(method="idpp", mic=False)
        bond = images[1].positions[1] - images[1].positions[0]
        return images[1].positions.copy(), float(np.linalg.norm(bond))

    ase_positions, ase_length = placed(AseNeb)
    our_positions, our_length = placed(NEB)
    print(f"idpp bond length ase={ase_length:.6f} rgsaddle={our_length:.6f}")
    print(f"idpp interior ase={ase_positions} rgsaddle={our_positions}")
    # ASE stops an MDMin run at fmax 0.1. The Rust path is L-BFGS to the
    # Smidstrup gradient tolerance, so the bond is tighter and the
    # coordinates are not the same point.
    assert ase_length == pytest.approx(1.0, abs=0.1)
    assert our_length == pytest.approx(1.0, abs=0.01)


def test_apply_constraint_none_matches_ase() -> None:
    def prepared(cls, flag):  # type: ignore[no-untyped-def]
        images = []
        for x in (-1.0, 4.0, 1.0):
            atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
            atoms.calc = Bowl()
            atoms.set_constraint(Nudge())
            images.append(atoms)
        band = cls(images, k=0.1, climb=False, method="improvedtangent")
        band.interpolate(method="linear", apply_constraint=flag)
        return images[1].positions.copy()

    with pytest.raises(RuntimeError):
        prepared(AseNeb, None)
    with pytest.raises(RuntimeError):
        prepared(NEB, None)
    assert np.allclose(prepared(AseNeb, False), prepared(NEB, False))


def test_dyneb_forces_and_one_step_match() -> None:
    points = [(-0.8, 0.0), (-0.2, 0.0), (0.6, 0.1), (0.15, 0.0), (0.9, 0.0)]

    def run(cls):  # type: ignore[no-untyped-def]
        Bowl.reset_calls()
        images = _images(Bowl, points)
        band = cls(images, k=0.5, fmax=0.2, climb=False, method="improvedtangent")
        forces = band.get_forces()
        opt = FIRE(band, logfile=None, maxstep=0.05)
        opt.run(fmax=0.2, steps=1)
        calls = Bowl.call_count()
        print(f"DyNEB {cls.__name__} force calls={calls}")
        return forces, images[2].positions.copy(), calls

    ase_forces, ase_middle, ase_calls = run(AseDyNEB)
    our_forces, our_middle, our_calls = run(DyNEB)
    assert np.allclose(ase_forces, our_forces)
    assert np.allclose(ase_middle, our_middle, atol=1e-8)
    assert ase_calls > 0 and our_calls > 0


def test_neb_dynamic_keyword_is_opt_in() -> None:
    images = _images(Bowl, [(0.0, 0.0), (0.4, 0.0), (1.0, 0.0)])
    neb = NEB(images, k=0.1, method="eb")
    assert neb.dynamic_relaxation is False
    with pytest.warns(FutureWarning):
        opted = NEB(images, k=0.1, method="eb", dynamic_relaxation=True)
    assert opted.dynamic_relaxation is True
    with pytest.raises(ValueError):
        NEB(images, k=0.1, method="eb", scale_fmax=0.1)


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
        (Bowl, 3, 5),
        (QuadraticSaddle, 4, 2),
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
    with pytest.raises(UnsupportedFeature):
        AutoNEB(
            lambda images: None,
            "unused",
            n_simul=3,
            n_max=7,
            smooth_curve=True,
        )


def test_dimer_control_forces_match() -> None:
    def atom():  # type: ignore[no-untyped-def]
        atoms = Atoms("H", positions=[[0.4, 0.3, 0.0]])
        atoms.calc = Bowl()
        return atoms

    kwargs = {"logfile": None, "max_num_rot": 4, "dimer_separation": 0.001, "f_rot_min": 1e-4}
    mode = np.array([[1.0, 0.0, 0.0]])
    ase = AseMinMode(atom(), control=AseDimerControl(**kwargs), eigenmodes=[mode])
    ours = MinModeAtoms(atom(), control=DimerControl(**kwargs), eigenmodes=[mode])
    assert np.allclose(ase.get_forces(), ours.get_forces(), atol=1e-6)
    assert ours.get_curvature() == pytest.approx(float(ase.get_curvature()), abs=1e-6)


def test_min_mode_translate_finds_the_saddle() -> None:
    def search(atoms_cls, translate_cls, control_cls):  # type: ignore[no-untyped-def]
        QuadraticSaddle.reset_calls()
        atoms = Atoms("H", positions=[[0.35, 0.15, 0.0]])
        atoms.calc = QuadraticSaddle()
        control = control_cls(
            logfile=None,
            max_num_rot=6,
            dimer_separation=0.001,
            f_rot_min=1e-4,
            maximum_translation=0.1,
        )
        dimer = atoms_cls(atoms, control=control, eigenmodes=[np.array([[1.0, 0.0, 0.0]])])
        opt = translate_cls(dimer, logfile=None)
        opt.run(fmax=0.05, steps=40)
        print(
            f"MinModeTranslate {translate_cls.__name__} calls={QuadraticSaddle.call_count()} "
            f"position={atoms.positions[0]} energy={atoms.get_potential_energy():.6f}"
        )
        return (
            atoms.positions[0].copy(),
            float(atoms.get_potential_energy()),
            QuadraticSaddle.call_count(),
        )

    ase_pos, ase_energy, ase_calls = search(AseMinMode, AseTranslate, AseDimerControl)
    our_pos, our_energy, our_calls = search(MinModeAtoms, MinModeTranslate, DimerControl)
    assert np.linalg.norm(ase_pos[:2]) < 0.05
    assert np.linalg.norm(our_pos[:2]) < 0.05
    assert our_energy == pytest.approx(ase_energy, abs=0.02)
    assert ase_calls > 0 and our_calls > 0


def test_fixatoms_still_follow_the_first_image() -> None:
    def placed(cls):  # type: ignore[no-untyped-def]
        images = []
        for x, pinned in ((0.0, 0.2), (1.0, 0.5), (2.0, 0.8)):
            atoms = Atoms("H2", positions=[[pinned, 0.0, 0.0], [x, 0.0, 0.0]])
            atoms.calc = Bond()
            atoms.set_constraint(FixAtoms(indices=[0]))
            images.append(atoms)
        cls(images, k=0.1, method="eb").interpolate()
        return images[1].positions[0, 0]

    # ASE leaves a fixed atom on that image. This band pins every image
    # to the first image's fixed coordinates.
    assert placed(AseNeb) == pytest.approx(0.5)
    assert placed(NEB) == pytest.approx(0.2)


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


@pytest.mark.parametrize("case", sorted(_PERIODIC_CASES))
def test_idpp_mic_follows_the_minimum_image_on_mixed_periodicity(case: str) -> None:
    cell, pbc = _PERIODIC_CASES[case]
    images = _hopping(cell, pbc, [(0.6, 1.0), (2.0, 1.0), (3.6, 1.0)])
    band = NEB(images, k=0.1, climb=False, method="improvedtangent")
    band.interpolate(method="idpp", mic=True)
    middle = images[1].positions[2]
    print(f"idpp {case} interior adatom={middle}")
    # The minimum image of 0.6 -> 3.6 crosses the cell edge at x = 0. The
    # straight line would pass x = 2.1.
    assert middle[0] < 1.0
    assert np.all(np.isfinite(images[1].positions))
    assert np.allclose(images[0].positions[2], [0.6, 1.0, 1.0])
    assert np.allclose(images[-1].positions[2], [3.6, 1.0, 1.0])


def test_idpp_mic_does_not_fold_a_non_periodic_axis() -> None:
    cell = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
    images = _hopping(cell, [True, True, False], [(2.0, 0.0), (2.0, 2.0), (2.0, 4.0)])
    band = NEB(images, k=0.1, climb=False, method="improvedtangent")
    band.interpolate(method="idpp", mic=True)
    # The adatom climbs 4 A through a cell 5 A tall. Folding z would send
    # it 1 A the other way.
    assert 1.0 < images[1].positions[2, 2] < 3.0


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


def _quadratic_dimer(method: str):  # type: ignore[no-untyped-def]
    atoms = Atoms("H", positions=[[0.35, 0.15, 0.0]])
    atoms.calc = QuadraticSaddle()
    control = DimerControl(
        logfile=None,
        max_num_rot=6,
        dimer_separation=0.001,
        f_rot_min=1e-4,
        maximum_translation=0.1,
    )
    dimer = MinModeAtoms(
        atoms, control=control, eigenmodes=[np.array([[1.0, 0.0, 0.0]])], method=method
    )
    return atoms, dimer


def test_min_mode_translate_is_an_optimizer_class() -> None:
    atoms, dimer = _quadratic_dimer("lbfgs")
    opt = MinModeTranslate(dimer, logfile=None)
    assert isinstance(opt, MinModeTranslate)
    assert isinstance(opt, Optimizer)
    assert issubclass(MinModeTranslate, Optimizer)
    assert opt.run(fmax=0.05, steps=40)
    assert np.linalg.norm(atoms.positions[0, :2]) < 0.05
    assert opt.nsteps > 0


def test_min_mode_translate_first_argument_is_the_dimeratoms() -> None:
    names = list(inspect.signature(MinModeTranslate).parameters)
    assert names == list(inspect.signature(AseTranslate).parameters)
    assert names[0] == "dimeratoms"
    atoms, dimer = _quadratic_dimer("lbfgs")
    opt = MinModeTranslate(dimeratoms=dimer, logfile=None)
    assert opt.dimeratoms is dimer
    assert opt.control is dimer.get_control()
    with pytest.raises(TypeError, match="MinModeAtoms"):
        MinModeTranslate(atoms, logfile=None)


def test_min_mode_translate_writes_the_trajectory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from ase.io import read

    atoms, dimer = _quadratic_dimer("lbfgs")
    path = tmp_path / "translate.traj"
    opt = MinModeTranslate(dimer, logfile=None, trajectory=str(path))
    opt.run(fmax=0.05, steps=40)
    frames = read(str(path), ":")
    assert len(frames) == opt.nsteps + 1
    assert np.allclose(frames[-1].positions, atoms.positions)


def test_min_mode_translate_optimizer_is_lbfgs_or_fire() -> None:
    assert "L-BFGS" in (MinModeTranslate.__doc__ or "")
    assert "FIRE" in (MinModeTranslate.__doc__ or "")
    first = {}
    for method in ("lbfgs", "fire"):
        atoms, dimer = _quadratic_dimer(method)
        assert dimer.method == method
        opt = MinModeTranslate(dimer, logfile=None)
        assert opt.run(fmax=0.05, steps=60)
        assert np.linalg.norm(atoms.positions[0, :2]) < 0.05
        fresh, other = _quadratic_dimer(method)
        other.translate(0.05)
        first[method] = fresh.positions[0].copy()
    # The first translation is the optimizer's own: L-BFGS and FIRE start differently.
    assert not np.allclose(first["lbfgs"], first["fire"], atol=1e-3)
    assert MinModeAtoms(Atoms("H", positions=[[0.0, 0.0, 0.0]], calculator=Flat())).method == "lbfgs"
    with pytest.raises(ValueError, match="lbfgs or fire"):
        MinModeAtoms(Atoms("H", positions=[[0.0, 0.0, 0.0]], calculator=Flat()), method="quickmin")
    with pytest.raises(ValueError, match="fire or lbfgs"):
        MinMode(np.zeros(3), np.array([1.0, 0.0, 0.0]), method="quickmin")
    for method in ("lbfgs", "fire"):
        assert MinMode(np.zeros(3), np.array([1.0, 0.0, 0.0]), method=method) is not None
