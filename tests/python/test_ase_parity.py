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
from ase.mep.autoneb import AutoNEB as AseAutoNEB
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

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
    left, right = 1, 6
    phase = threading.Barrier(2)
    seen_left: list[int] = []
    seen_right: list[int] = []
    first = threading.Thread(
        target=_count_on_this_thread, args=(MullerBrown, left, phase, seen_left)
    )
    second = threading.Thread(
        target=_count_on_this_thread, args=(MullerBrown, right, phase, seen_right)
    )
    first.start()
    second.start()
    first.join()
    second.join()
    assert seen_left == [left]
    assert seen_right == [right]


class TightFIRE(FIRE):
    def __init__(self, atoms, maxstep=0.05, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(atoms, maxstep=maxstep, **kwargs)


class DocumentedFIRE(FIRE):
    """FIRE whose documented cap is not the stock 0.2."""

    defaults = {"maxstep": 0.07}


def test_autoneb_runs_with_fire() -> None:
    """ASE 3.29 leaves ``maxstep`` as None and applies ``defaults['maxstep']``."""

    from rgsaddle.ase import _optimizer_maxstep

    assert inspect.signature(FIRE).parameters["maxstep"].default is None
    documented = float(FIRE.defaults["maxstep"])
    assert documented == pytest.approx(float(Optimizer.defaults["maxstep"]))
    assert _optimizer_maxstep(FIRE) == pytest.approx(documented)
    assert _optimizer_maxstep(None) == pytest.approx(documented)
    assert _optimizer_maxstep(DocumentedFIRE) == pytest.approx(0.07)
    assert _optimizer_maxstep(TightFIRE) == pytest.approx(0.05)

    import tempfile

    from ase.io import write

    root = tempfile.mkdtemp(prefix="rgsaddle-fire-")
    prefix = os.path.join(root, "neb")
    for index, xyz in enumerate((np.array([-0.4, 0.0, 0.0]), np.array([0.4, 0.0, 0.0]))):
        atoms = Atoms("H", positions=[xyz])
        atoms.calc = Bowl()
        atoms.get_potential_energy()
        write(f"{prefix}{index:03d}.traj", atoms)

    def attach(images):  # type: ignore[no-untyped-def]
        for image in images:
            image.calc = Bowl()

    band = AutoNEB(
        attach,
        prefix,
        n_simul=1,
        n_max=3,
        fmax=0.5,
        maxsteps=2,
        k=1.0,
        climb=False,
        method="improvedtangent",
        optimizer=FIRE,
        remove_rotation_and_translation=False,
        parallel=False,
        smooth_curve=False,
        interpolate_method="linear",
    )
    report = band.run()
    assert len(report.images) >= 2
    assert np.isfinite([float(image.get_potential_energy()) for image in report.images]).all()


class DoubleWell(Calculator):
    """E = (x^2 - 1)^2 + 2 y^2 + 2 z^2. Minima at x = -1, 1 and a saddle at the origin."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        x, y, z = (float(v) for v in atoms.positions[0])
        self.results = {
            "energy": (x * x - 1.0) ** 2 + 2.0 * y * y + 2.0 * z * z,
            "forces": np.array([[-4.0 * x * (x * x - 1.0), -4.0 * y, -4.0 * z]]),
        }


def _double_well_band(tmp_path, fmax, maxsteps):  # type: ignore[no-untyped-def]
    from ase.io import write

    prefix = str(tmp_path / "neb")
    for index, xyz in enumerate(([-1.1, 0.3, 0.0], [0.9, -0.2, 0.0])):
        atoms = Atoms("H", positions=[xyz])
        atoms.calc = DoubleWell()
        write(f"{prefix}{index:03d}.traj", atoms)

    def attach(images):  # type: ignore[no-untyped-def]
        for image in images:
            image.calc = DoubleWell()

    return AutoNEB(
        attach,
        prefix,
        n_simul=1,
        n_max=5,
        fmax=fmax,
        maxsteps=maxsteps,
        k=5.0,
        climb=True,
        method="improvedtangent",
        optimizer=FIRE,
        parallel=False,
        interpolate_method="linear",
    )


def test_autoneb_run_reports_a_converged_stage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    band = _double_well_band(tmp_path, fmax=[5e-2, 1e-3], maxsteps=[400, 400])
    report = band.run()
    assert report.status == 1
    assert report.converged
    assert 0.0 <= report.max_force <= 1e-3
    assert 0 < report.iteration <= 400
    assert report.n_images == 5 == len(report.images)
    assert report.positions.shape == (5, 3)
    assert report.surface_rows > 0
    assert report.images is band.all_images
    assert np.allclose(
        [image.positions[0] for image in report.images], report.positions, atol=1e-12
    )
    saddle = report.images[report.ci_index].positions[0]
    assert np.linalg.norm(saddle) < 0.05
    assert band.status == report.status
    assert band.max_force == report.max_force
    assert band.iteration == report.iteration
    assert band.surface_rows == report.surface_rows
    assert band.ci == report.ci_index


def test_autoneb_run_reports_an_exhausted_step_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    band = _double_well_band(tmp_path, fmax=[1e-12, 1e-12], maxsteps=[2, 2])
    report = band.run()
    assert report.status == 0
    assert not report.converged
    assert report.max_force > 1e-12
    assert report.iteration == 2
    assert band.iteration == report.iteration == 2
    assert band.status == 0
    assert band.max_force == report.max_force
    assert len(report.images) == report.n_images


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



def _flat_autoneb(prefix, images, **kwargs):  # type: ignore[no-untyped-def]
    from ase.io import write

    for index, atoms in enumerate(images):
        write(f"{prefix}{index:03d}.traj", atoms)

    def attach(images):  # type: ignore[no-untyped-def]
        for image in images:
            image.calc = Flat()

    options = {
        "n_simul": 1,
        "n_max": 3,
        "fmax": 0.5,
        "maxsteps": 2,
        "k": 1.0,
        "climb": False,
        "method": "improvedtangent",
        "optimizer": FIRE,
        "parallel": False,
        "interpolate_method": "linear",
    }
    options.update(kwargs)
    return AutoNEB(attach, prefix, **options)


def test_autoneb_refuses_endpoints_with_other_atoms(tmp_path) -> None:  # type: ignore[no-untyped-def]
    hydrogen = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    helium = Atoms("He", positions=[[1.0, 0.0, 0.0]])
    band = _flat_autoneb(str(tmp_path / "numbers"), [hydrogen, helium])
    with pytest.raises(ValueError, match="different orders"):
        band.run()
    assert not os.path.exists(str(tmp_path / "numbers002.traj"))


def test_autoneb_refuses_endpoints_with_another_cell_or_pbc(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cell = np.diag([4.0, 4.0, 4.0])
    start = Atoms("H", positions=[[0.5, 0.5, 0.5]], cell=cell, pbc=True)
    other_pbc = Atoms("H", positions=[[1.5, 0.5, 0.5]], cell=cell, pbc=[True, True, False])
    with pytest.raises(ValueError, match="boundary conditions"):
        _flat_autoneb(str(tmp_path / "pbc"), [start, other_pbc]).run()
    other_cell = Atoms("H", positions=[[1.5, 0.5, 0.5]], cell=1.1 * cell, pbc=True)
    with pytest.raises(NotImplementedError, match="Variable cell"):
        _flat_autoneb(str(tmp_path / "cell"), [start, other_cell]).run()
    longer = Atoms("H2", positions=[[1.5, 0.5, 0.5], [2.5, 0.5, 0.5]], cell=cell, pbc=True)
    with pytest.raises(ValueError, match="numbers of atoms"):
        _flat_autoneb(str(tmp_path / "count"), [start, longer]).run()


def test_autoneb_images_keep_the_atoms_of_their_endpoints(tmp_path) -> None:  # type: ignore[no-untyped-def]
    start = Atoms("HHe", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], tags=[1, 2])
    end = Atoms("HHe", positions=[[0.0, 1.0, 0.0], [1.5, 1.0, 0.0]], tags=[3, 4])
    report = _flat_autoneb(str(tmp_path / "band"), [start, end]).run()
    assert report.n_images == 3
    for image in report.images:
        assert list(image.get_atomic_numbers()) == [1, 2]
        assert list(image.get_chemical_symbols()) == ["H", "He"]
    assert list(report.images[0].get_tags()) == [1, 2]
    assert list(report.images[1].get_tags()) == [1, 2]
    assert list(report.images[-1].get_tags()) == [3, 4]

    from ase.io import read

    written = read(str(tmp_path / "band002.traj"))
    assert list(written.get_atomic_numbers()) == [1, 2]
    assert list(written.get_tags()) == [3, 4]


class _World:
    """A stand-in communicator: a rank, a size, and a counted barrier."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank = rank
        self.size = size
        self.barriers = 0

    def barrier(self) -> None:
        self.barriers += 1


def _two_hydrogens() -> list[Atoms]:
    return [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[1.0, 0.0, 0.0]]),
    ]


def test_autoneb_world_none_is_the_ase_world(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import ase.parallel

    band = _flat_autoneb(str(tmp_path / "default"), _two_hydrogens())
    assert band.world is ase.parallel.world


def test_autoneb_refuses_image_parallel_mpi(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(NotImplementedError, match="calling thread"):
        _flat_autoneb(str(tmp_path / "mpi"), _two_hydrogens(), parallel=True, world=_World(0, 2))


@pytest.mark.parametrize("rank", [0, 1])
def test_autoneb_writes_the_band_from_rank_zero_alone(tmp_path, rank: int) -> None:  # type: ignore[no-untyped-def]
    world = _World(rank, 2)
    prefix = str(tmp_path / "ranked")
    band = _flat_autoneb(prefix, _two_hydrogens(), world=world)
    before = os.path.getmtime(f"{prefix}001.traj")
    report = band.run()
    assert report.n_images == 3
    assert world.barriers == 1
    assert os.path.exists(f"{prefix}002.traj") == (rank == 0)
    if rank != 0:
        assert os.path.getmtime(f"{prefix}001.traj") == before


class Triangle(Calculator):
    """Harmonic springs between every pair of three atoms, so rigid motion costs nothing."""

    implemented_properties = ["energy", "forces"]
    rest = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.2}

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = atoms.positions
        energy = 0.0
        forces = np.zeros_like(positions)
        for (i, j), rest in self.rest.items():
            bond = positions[j] - positions[i]
            length = float(np.linalg.norm(bond))
            stretch = length - rest
            energy += 0.5 * stretch * stretch
            pull = stretch * bond / length
            forces[i] += pull
            forces[j] -= pull
        self.results = {"energy": energy, "forces": forces}


def _rotation(axis, angle):  # type: ignore[no-untyped-def]
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _turned_triangle_band(tmp_path, name, remove):  # type: ignore[no-untyped-def]
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.9, 0.0]])
    bent = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.7, 0.3]])
    turn = _rotation([0.3, 0.2, 1.0], np.radians(50.0))
    product = bent @ turn.T + np.array([2.0, -1.0, 0.5])
    images = [Atoms("H3", positions=reactant), Atoms("H3", positions=product)]
    from ase.io import write

    prefix = str(tmp_path / name)
    for index, atoms in enumerate(images):
        write(f"{prefix}{index:03d}.traj", atoms)

    def attach(images):  # type: ignore[no-untyped-def]
        for image in images:
            image.calc = Triangle()

    return AutoNEB(
        attach,
        prefix,
        n_simul=1,
        n_max=4,
        fmax=0.05,
        maxsteps=50,
        k=1.0,
        climb=False,
        method="improvedtangent",
        optimizer=FIRE,
        parallel=False,
        remove_rotation_and_translation=remove,
        interpolate_method="linear",
    )


def _ase_fit_moves(target, atoms) -> float:  # type: ignore[no-untyped-def]
    from ase.build import minimize_rotation_and_translation

    moved = atoms.copy()
    minimize_rotation_and_translation(target, moved)
    return float(np.abs(moved.positions - atoms.positions).max())


def test_kabsch_matches_the_ase_rigid_fit() -> None:
    from ase.build import minimize_rotation_and_translation

    from rgsaddle.ase import _kabsch

    rng = np.random.default_rng(11)
    target = rng.normal(size=(6, 3))
    turn = _rotation([1.0, -2.0, 0.5], 2.0)
    moving = (target + 0.05 * rng.normal(size=(6, 3))) @ turn.T + np.array([3.0, 1.0, -2.0])
    fitted = _kabsch(target, moving)
    reference = Atoms("H6", positions=moving)
    minimize_rotation_and_translation(Atoms("H6", positions=target), reference)
    assert np.allclose(fitted, reference.positions, atol=1e-8)


def test_autoneb_removes_rotation_as_well_as_translation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    report = _turned_triangle_band(tmp_path, "aligned", remove=True).run()
    images = report.images
    assert len(images) == 4
    for index in range(1, len(images)):
        assert _ase_fit_moves(images[index - 1], images[index]) < 1e-8, index
    # Each fit puts an image's centroid on its predecessor's, so the band
    # shares the reactant's centroid and the product's offset is gone.
    centre = images[0].positions.mean(axis=0)
    for image in images:
        assert np.allclose(image.positions.mean(axis=0), centre, atol=1e-8)

    kept = _turned_triangle_band(tmp_path, "kept", remove=False).run()
    assert _ase_fit_moves(kept.images[-2], kept.images[-1]) > 0.1
