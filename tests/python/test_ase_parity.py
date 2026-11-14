"""rgsaddle behind ASE's NEB and dimer calls, against ASE 3.29 on EMT.

Each comparison prints the barrier and the number of EMT force calls
for both codes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.mep import NEB as AseNEB
from ase.mep import dimer as ase_dimer
from ase.optimize import BFGS, FIRE

import rgsaddle.ase as rg

BARRIER_TOL = 1e-3


class Counted(EMT):  # type: ignore[misc]
    calls = 0

    def calculate(self, *args: Any, **kwargs: Any) -> None:
        type(self).calls += 1
        super().calculate(*args, **kwargs)


def _relax(atoms: Atoms, fmax: float = 1e-3) -> Atoms:
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=fmax)
    return atoms


def _adatom_minimum(symbol: str) -> Atoms:
    slab = fcc100("Al", size=(2, 2, 3), vacuum=6.0)
    add_adsorbate(slab, symbol, 1.7, "hollow")
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    return _relax(slab)


def adatom_hop(symbol: str, n_images: int = 7) -> list[Atoms]:
    """An adatom hops to the next hollow of Al(100), ASE's NEB tutorial."""
    initial = _adatom_minimum(symbol)
    final = initial.copy()
    final.positions[-1, 0] += initial.cell[0, 0] / 2
    _relax(final)
    return [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]


def vacancy_hop(n_images: int = 7) -> list[Atoms]:
    """A vacancy in periodic fcc Al swaps with a neighbour; the final state is wrapped."""
    crystal = bulk("Al", "fcc", a=4.05, cubic=True).repeat(2)
    distances = crystal.get_distances(0, range(1, len(crystal)), mic=True)
    neighbour = int(np.argmin(distances))
    site = crystal.positions[0].copy()
    del crystal[0]
    initial = _relax(crystal)
    final = initial.copy()
    final.positions[neighbour] = site
    _relax(final)
    final.wrap()
    return [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]


CASES = {
    "Al adatom on Al(100)": (lambda: adatom_hop("Al"), False),
    "Au adatom on Al(100)": (lambda: adatom_hop("Au"), False),
    "vacancy in bulk Al": (vacancy_hop, True),
}


def _attach(images: list[Atoms]) -> None:
    for atoms in images:
        atoms.calc = Counted()


def _barrier(images: list[Atoms]) -> float:
    energies = [atoms.get_potential_energy() for atoms in images]
    return float(max(energies) - energies[0])


def _ase_reference(images: list[Atoms], mic: bool, fmax: float) -> tuple[float, int]:
    _attach(images)
    neb = AseNEB(images, climb=True, method="improvedtangent")
    neb.interpolate(mic=mic)
    Counted.calls = 0
    assert FIRE(neb, logfile=None).run(fmax=fmax, steps=2000)
    calls = Counted.calls
    return _barrier(images), calls


def _report(capsys: pytest.CaptureFixture[str], line: str) -> None:
    with capsys.disabled():
        print(f"\n  {line}", end="")


@pytest.mark.parametrize("name", list(CASES))
def test_ci_neb_reaches_ases_barrier(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    make, mic = CASES[name]
    fmax = 0.01
    ase_barrier, ase_calls = _ase_reference(make(), mic, fmax)

    images = make()
    _attach(images)
    neb = rg.NEB(images, climb=True, method="improvedtangent")
    neb.interpolate(mic=mic)
    Counted.calls = 0
    assert rg.NEBOptimizer(neb, method="fire", logfile=None).run(fmax=fmax, steps=2000)
    calls = Counted.calls
    barrier = _barrier(images)

    _report(
        capsys,
        f"{name}: ase FIRE {ase_barrier:.6f} eV / {ase_calls} calls, "
        f"rgsaddle fire {barrier:.6f} eV / {calls} calls",
    )
    assert abs(barrier - ase_barrier) < BARRIER_TOL
    assert neb.get_residual() < fmax
    assert neb.imax == int(np.argmax([atoms.get_potential_energy() for atoms in images]))


@pytest.mark.parametrize("method", ["ode", "lbfgs"])
def test_other_band_steppers_reach_the_same_barrier(method: str, capsys: pytest.CaptureFixture[str]) -> None:
    ase_barrier, _ = _ase_reference(adatom_hop("Al"), False, 0.01)
    images = adatom_hop("Al")
    _attach(images)
    neb = rg.NEB(images, climb=True)
    neb.interpolate()
    Counted.calls = 0
    assert rg.NEBOptimizer(neb, method=method, logfile=None).run(fmax=0.01, steps=2000)
    _report(capsys, f"Al adatom, rgsaddle {method}: {_barrier(images):.6f} eV / {Counted.calls} calls")
    assert abs(_barrier(images) - ase_barrier) < BARRIER_TOL


def test_ase_fire_drives_the_rgsaddle_band(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ase_barrier, ase_calls = _ase_reference(adatom_hop("Al"), False, 0.01)
    images = adatom_hop("Al")
    _attach(images)
    neb = rg.NEB(images, climb=True)
    neb.interpolate()
    Counted.calls = 0
    path = tmp_path / "band.traj"
    with Trajectory(str(path), "w") as traj:
        opt = FIRE(neb, logfile=None, trajectory=traj)
        assert opt.run(fmax=0.01, steps=2000)
    _report(capsys, f"Al adatom, ASE FIRE on rgsaddle.ase.NEB: {_barrier(images):.6f} eV / {Counted.calls} calls")
    assert abs(_barrier(images) - ase_barrier) < BARRIER_TOL

    with Trajectory(str(path)) as written:
        frames = list(written)
    assert len(frames) % len(images) == 0
    last = frames[-len(images) :]
    for frame, atoms in zip(last, images):
        assert np.allclose(frame.positions, atoms.positions)
        assert frame.get_potential_energy() == pytest.approx(atoms.get_potential_energy())


def test_fixed_atoms_stay_put_and_feel_no_band_force() -> None:
    images = adatom_hop("Al")
    fixed = images[0].constraints[0].index
    start = [atoms.positions[fixed].copy() for atoms in images]
    _attach(images)
    neb = rg.NEB(images, climb=True)
    neb.interpolate()
    start = [atoms.positions[fixed].copy() for atoms in images]
    forces = neb.get_forces().reshape(len(images) - 2, -1, 3)
    assert np.all(forces[:, fixed] == 0.0)
    assert rg.NEBOptimizer(neb, method="fire", logfile=None).run(fmax=0.02, steps=2000)
    for atoms, before in zip(images, start):
        assert np.array_equal(atoms.positions[fixed], before)

    aligned = rg.NEB(adatom_hop("Al"), remove_rotation_and_translation=True)
    _attach(aligned.images)
    aligned.interpolate()
    forces = aligned.get_forces().reshape(len(aligned.images) - 2, -1, 3)
    assert np.all(forces[:, fixed] == 0.0)


def test_each_image_calculator_matches_one_shared_calculator() -> None:
    own = adatom_hop("Al")
    _attach(own)
    shared = adatom_hop("Al")
    calc = EMT()
    for atoms in shared:
        atoms.calc = calc
    with pytest.raises(ValueError, match="allow_shared_calculator"):
        rg.NEB(shared, climb=True).get_forces()

    bands = [rg.NEB(own, climb=True), rg.NEB(shared, climb=True, allow_shared_calculator=True)]
    for neb in bands:
        neb.interpolate()
    assert np.allclose(bands[0].get_forces(), bands[1].get_forces(), atol=1e-12)
    for neb in bands:
        assert rg.NEBOptimizer(neb, method="fire", logfile=None).run(fmax=0.01, steps=2000)
    assert np.allclose(own[3].positions, shared[3].positions, atol=1e-10)
    assert _barrier(own) == pytest.approx(_barrier(shared), abs=1e-10)


def test_spring_list_mic_and_ase_bookkeeping_match_ase() -> None:
    reference = vacancy_hop(5)
    ours = vacancy_hop(5)
    ks = [0.05, 0.1, 0.2, 0.1]
    a = AseNEB(reference, k=ks, climb=True, method="improvedtangent")
    b = rg.NEB(ours, k=ks, climb=True, method="improvedtangent")
    for neb, images in ((a, reference), (b, ours)):
        _attach(images)
        neb.interpolate(mic=True)
    assert np.allclose(a.get_forces(), b.get_forces(), atol=1e-10)
    assert b.imax == a.imax
    assert b.get_residual() == pytest.approx(a.get_residual(), rel=1e-10)
    assert np.allclose(b.real_forces, a.real_forces, atol=1e-10)


def test_idpp_interpolation_starts_off_the_straight_line() -> None:
    images = adatom_hop("Al", 5)
    neb = rg.NEB(images)
    neb.interpolate(method="idpp")
    straight = 0.5 * (images[0].positions + images[-1].positions)
    assert not np.allclose(images[2].positions, straight)
    fixed = images[0].constraints[0].index
    assert np.allclose(images[2].positions[fixed], images[0].positions[fixed])


# ASE's DimerControl keeps its parameters in a class-level dict, so a
# setting from an earlier test carries over unless every one is given.
ASE_DIMER_DEFAULTS: dict[str, Any] = {
    "eigenmode_method": "dimer",
    "f_rot_min": 0.1,
    "f_rot_max": 1.0,
    "max_num_rot": 1,
    "trial_angle": np.pi / 4.0,
    "trial_trans_step": 0.001,
    "maximum_translation": 0.1,
    "cg_translation": True,
    "use_central_forces": True,
    "dimer_separation": 0.0001,
    "extrapolate_forces": False,
    "gauss_std": 0.1,
    "order": 1,
    "displacement_center": None,
    "displacement_radius": None,
    "number_of_displacement_atoms": None,
}


def _dimer_search(module: Any) -> tuple[Atoms, Any, float, int]:
    atoms = _adatom_minimum("Al")
    BFGS(atoms, logfile=None).run(fmax=1e-4)
    e0 = atoms.get_potential_energy()
    atoms.calc = Counted()
    vector = np.zeros((len(atoms), 3))
    vector[-1] = [0.6, 0.1, 0.0]
    control = module.DimerControl(
        logfile=None,
        **{
            **ASE_DIMER_DEFAULTS,
            "initial_eigenmode_method": "displacement",
            "displacement_method": "vector",
            "mask": [atom.tag == 0 for atom in atoms],
        },
    )
    d_atoms = module.MinModeAtoms(atoms, control)
    d_atoms.displace(displacement_vector=vector)
    Counted.calls = 0
    assert module.MinModeTranslate(d_atoms, logfile=None).run(fmax=1e-3, steps=2000)
    return atoms, d_atoms, atoms.get_potential_energy() - e0, Counted.calls


def _hessian_eigenvalues(atoms: Atoms, step: float = 1e-3) -> np.ndarray:
    fixed = set(atoms.constraints[0].index)
    dofs = [(i, c) for i in range(len(atoms)) if i not in fixed for c in range(3)]
    x0 = atoms.get_positions()
    hessian = np.zeros((len(dofs), len(dofs)))
    for j, (i, c) in enumerate(dofs):
        columns = []
        for sign in (1.0, -1.0):
            x = x0.copy()
            x[i, c] += sign * step
            atoms.set_positions(x, apply_constraint=False)
            forces = atoms.get_forces(apply_constraint=False)
            columns.append(np.array([forces[a, b] for a, b in dofs]))
        hessian[:, j] = -(columns[0] - columns[1]) / (2 * step)
    atoms.set_positions(x0, apply_constraint=False)
    return np.linalg.eigvalsh(0.5 * (hessian + hessian.T))


def test_dimer_reaches_ases_saddle_with_one_negative_mode(capsys: pytest.CaptureFixture[str]) -> None:
    ase_atoms, _, ase_barrier, ase_calls = _dimer_search(ase_dimer)
    atoms, d_atoms, barrier, calls = _dimer_search(rg)
    _report(
        capsys,
        f"dimer, Al adatom: ase {ase_barrier:.6f} eV / {ase_calls} calls, "
        f"rgsaddle {barrier:.6f} eV / {calls} calls",
    )
    assert abs(barrier - ase_barrier) < BARRIER_TOL
    assert np.abs(atoms.positions - ase_atoms.positions).max() < 0.01
    assert d_atoms.get_curvature() < 0.0
    eigenvalues = _hessian_eigenvalues(atoms)
    assert int((eigenvalues < 0.0).sum()) == 1
    assert eigenvalues[0] == pytest.approx(d_atoms.get_curvature(), rel=0.05)
    fixed = atoms.constraints[0].index
    assert np.all(d_atoms.get_eigenmode()[fixed] == 0.0)

    guess = np.zeros((len(atoms), 3))
    guess[-1] = [0.0, 1.0, 0.0]
    guess[0] = [1.0, 0.0, 0.0]
    d_atoms.set_eigenmode(guess)
    assert np.allclose(d_atoms.get_eigenmode()[-1], [0.0, 1.0, 0.0])
    assert np.allclose(d_atoms._session.mode().reshape(-1, 3), d_atoms.get_eigenmode())


def test_importing_the_ase_module_does_not_import_ase() -> None:
    code = "import sys, rgsaddle.ase; print('ase' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
