"""ASE 3.29 ``NEBOptimizer`` on an ``rgsaddle.ase.NEB`` band.

``method='ODE'`` is Makri, Ortner, and Kermode's adaptive ODE12r step
(J. Chem. Phys. 150, 094109, 2019). ``method='static'`` is the fixed
step. The Al adatom hop runs that ODE solver on EMT both ways. The band
forces match at 1e-9 eV/Angstrom, and both solvers stop on the same
residual, so the barrier and the climbing-image energy are required to
agree within 1e-9 eV.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")

from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import NEB as AseNeb
from ase.mep.neb import NEBOptimizer as AseNebOptimizer
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from rgsaddle.ase import NEB, NEBOptimizer


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


class DoubleWell(Calculator):
    """V = (x^2 - 1)^2 + 5 y^2. The ends at x = +/-1 are minima."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):  # type: ignore[no-untyped-def]
        Calculator.calculate(self, atoms, properties, system_changes)
        x = float(atoms.positions[0, 0])
        y = float(atoms.positions[0, 1])
        forces = np.zeros_like(atoms.positions)
        forces[0, 0] = -4.0 * x * (x * x - 1.0)
        forces[0, 1] = -10.0 * y
        self.results = {"energy": (x * x - 1.0) ** 2 + 5.0 * y * y, "forces": forces}


def _chain(xs: list[float], calc: type[Calculator] = Bowl) -> list[Atoms]:
    images = []
    for x in xs:
        atoms = Atoms("H", positions=[[x, 0.15, 0.0]], cell=[8.0, 8.0, 8.0], pbc=False)
        atoms.calc = calc()
        images.append(atoms)
    return images


def test_keywords_and_the_ase_residual(tmp_path) -> None:  # type: ignore[no-untyped-def]
    images = _chain([-1.0, -0.3, 0.4, 1.0])
    neb = NEB(images, k=0.1, climb=False, method="improvedtangent")
    with pytest.raises(RuntimeError, match=r"get_residual\(\) called before get_forces"):
        neb.get_residual()
    opt = NEBOptimizer(neb, logfile=None)
    with pytest.raises(RuntimeError, match=r"get_residual\(\) called before get_forces"):
        opt.get_residual()

    with pytest.raises(ValueError, match="method must be one of"):
        NEBOptimizer(neb, method="bfgs", logfile=None)
    krylov = NEBOptimizer(neb, method="krylov", logfile=None)
    assert krylov.method == "krylov"
    with pytest.raises(ValueError, match=r"unknown method: krylov$"):
        krylov.run(steps=1)
    stored = NEBOptimizer(neb, method="ODE", logfile=None)
    assert stored.method == "ode"
    with pytest.raises(ValueError, match=r"unknown method: ode$"):
        stored.run(method="ODE", steps=1)
    with pytest.raises(ValueError, match=r"unknown method: ode$"):
        stored.run(method="krylov", steps=1)

    missing = tmp_path / "absent"
    NEBOptimizer(neb, restart=str(missing), logfile=None, master=True)
    present = tmp_path / "restart"
    present.write_text("not a restart", encoding="utf-8")
    with pytest.raises(NotImplementedError):
        NEBOptimizer(neb, restart=str(present), logfile=None)

    climb = NEB(_chain([-1.0, 0.2, 1.0]), k=0.1, climb=True, method="improvedtangent")
    climb.get_forces()
    with pytest.raises(ValueError):
        climb.get_residual()
    ase_climb = AseNeb(_chain([-1.0, 0.2, 1.0]), k=0.1, climb=True, method="improvedtangent")
    ase_climb.get_forces()
    with pytest.raises(ValueError):
        ase_climb.get_residual()
    idpp = NEB(_chain([-1.0, 0.0, 1.0]), k=0.1, climb=False, method="improvedtangent")
    idpp.interpolate(method="idpp")
    assert NEBOptimizer(idpp, method="static", alpha=0.01, logfile=None).run(fmax=1e9, steps=1) is True


@pytest.mark.parametrize("method", ["improvedtangent", "aseneb"])
def test_residual_matches_ase_329(method: str) -> None:
    ours = NEB(_chain([-1.0, -0.3, 0.4, 1.0]), k=[0.05, 0.2, 0.4], climb=False, method=method)
    ase = AseNeb(_chain([-1.0, -0.3, 0.4, 1.0]), k=[0.05, 0.2, 0.4], climb=False, method=method)
    ours.interpolate()
    ase.interpolate()
    np.testing.assert_allclose(ours.get_forces(), ase.get_forces(), atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(ours.get_residual(), ase.get_residual(), atol=1e-9, rtol=0.0)
    opt = NEBOptimizer(ours, logfile=None)
    assert opt.get_residual() == pytest.approx(ours.get_residual())


def test_static_step_alpha_logfile_and_observers() -> None:
    images = _chain([-0.4, 0.2, 0.9])
    neb = NEB(images, k=0.1, climb=False, method="improvedtangent")
    before = np.array(neb.get_positions(), copy=True)
    held = NEBOptimizer(neb, method="static", alpha=0.0, logfile=None)
    assert isinstance(held, Optimizer)
    assert type(held).__name__ == "NEBOptimizer"
    assert held.run(fmax=0.0, steps=2) is False
    np.testing.assert_allclose(neb.get_positions(), before, atol=0.0, rtol=0.0)

    moved = _chain([-0.4, 0.2, 0.9])
    band = NEB(moved, k=0.1, climb=False, method="aseneb")
    origin = np.array(band.get_positions(), copy=True)
    buf = io.StringIO()
    seen: list[int] = []
    with NEBOptimizer(band, method="STATIC", alpha=0.05, logfile=buf) as opt:
        opt.attach(lambda: seen.append(opt.nsteps))
        assert opt.method == "static"
        assert opt.alpha == 0.05
        assert opt.run(fmax=0.0, steps=2, method="static") is False
    assert seen == [0, 1]
    assert not np.allclose(band.get_positions(), origin)
    text = buf.getvalue()
    assert "NEBOptimizer[static]" in text
    assert "Step" in text


def test_ode_keywords_reach_ase_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(force, x0, **kwargs):  # type: ignore[no-untyped-def]
        captured["rtol"] = kwargs["rtol"]
        captured["C1"] = kwargs["C1"]
        captured["C2"] = kwargs["C2"]
        captured["verbose"] = kwargs["verbose"]
        captured["fmax"] = kwargs["fmax"]
        captured["steps"] = kwargs["steps"]
        captured["callback"] = kwargs["callback"]
        captured["residual"] = kwargs["residual"]
        return x0

    monkeypatch.setattr("ase.optimize.ode.ode12r", fake)
    neb = NEB(_chain([-0.2, 0.1, 0.6]), k=0.1, climb=False, method="improvedtangent")
    opt = NEBOptimizer(neb, method="ODE", verbose=2, rtol=0.3, C1=0.04, C2=3.0, logfile=None)
    assert opt.run(fmax=0.02, steps=7) is True
    assert captured == {
        "rtol": 0.3,
        "C1": 0.04,
        "C2": 3.0,
        "verbose": 2,
        "fmax": 0.02,
        "steps": 7,
        "callback": opt.callback,
        "residual": opt.get_residual,
    }


def test_ode_path_matches_ase_on_the_double_well() -> None:
    # Ends are the minima, so both bands climb. A bowl does not: its ends
    # lie above the interior, and this band then refuses the climb.
    xs = [-1.0, -0.4, 0.2, 0.7, 1.0]
    ours = NEB(_chain(xs, DoubleWell), k=0.1, climb=True, method="improvedtangent")
    ase = AseNeb(_chain(xs, DoubleWell), k=0.1, climb=True, method="improvedtangent")
    ours.interpolate()
    ase.interpolate()
    np.testing.assert_allclose(ours.get_forces(), ase.get_forces(), atol=1e-9, rtol=0.0)
    assert NEBOptimizer(ours, method="ODE", logfile=None).run(fmax=1e-6, steps=8) is True
    assert AseNebOptimizer(ase, method="ODE", logfile=None).run(fmax=1e-6, steps=8) is True
    np.testing.assert_allclose(ours.get_positions(), ase.get_positions(), atol=1e-8, rtol=0.0)


def _relaxed_al_hop():
    """Al adatom in an Al(100) hollow, and the hollow half a cell along x."""
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Al", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    slab.set_constraint(FixAtoms(mask=slab.get_tags() > 1))
    initial = slab.copy()
    initial.calc = EMT()
    final = slab.copy()
    final[-1].x += float(final.cell[0, 0]) / 2.0
    final.calc = EMT()
    assert BFGS(initial, logfile=None).run(fmax=0.05, steps=80)
    assert BFGS(final, logfile=None).run(fmax=0.05, steps=80)
    return initial, final


def _emt_band(initial: Atoms, final: Atoms) -> list[Atoms]:
    images = [initial.copy()]
    images[0].calc = EMT()
    for _ in range(3):
        image = initial.copy()
        image.calc = EMT()
        images.append(image)
    end = final.copy()
    end.calc = EMT()
    images.append(end)
    return images


def _barrier(images: list[Atoms]) -> tuple[float, float, int]:
    energies = np.array([float(image.get_potential_energy()) for image in images], dtype=np.float64)
    climbing = int(np.argmax(energies))
    return float(energies[climbing] - energies[0]), float(energies[climbing]), climbing


def test_al_adatom_hop_matches_ase_neboptimizer() -> None:
    initial, final = _relaxed_al_hop()
    ours = NEB(_emt_band(initial, final), k=0.1, climb=True, method="improvedtangent")
    ase = AseNeb(_emt_band(initial, final), k=0.1, climb=True, method="improvedtangent")
    ours.interpolate()
    ase.interpolate()
    np.testing.assert_allclose(ours.get_forces(), ase.get_forces(), atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(ours.get_residual(), ase.get_residual(), atol=1e-9, rtol=0.0)
    assert NEBOptimizer(ours, method="ODE", logfile=None).run(fmax=0.05, steps=300) is True
    assert AseNebOptimizer(ase, method="ODE", logfile=None).run(fmax=0.05, steps=300) is True
    assert float(ours.get_residual()) <= 0.05
    assert float(ase.get_residual()) <= 0.05
    our_barrier, our_climb, our_index = _barrier(ours.images)
    ase_barrier, ase_climb, ase_index = _barrier(ase.images)
    assert our_index == ase_index
    assert 0 < our_index < len(ours.images) - 1
    # 1e-9 eV: same ODE12r stepper. On this hop the energies reproduced
    # to about 1e-14 eV, far inside the 0.05 eV/Angstrom force stop.
    np.testing.assert_allclose(our_barrier, ase_barrier, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(our_climb, ase_climb, atol=1e-9, rtol=0.0)
