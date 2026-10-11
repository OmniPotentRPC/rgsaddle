"""SurfaceAdapter on the calling thread, in request order."""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from rgsaddle._core import ClaimSlot
from rgsaddle._core._testing import drive_eval
from rgsaddle.ase import SurfaceAdapter

# SHA-256 of energies, gradients, and raw forces for the fixture below.
# 3.14 and 3.14t must match this constant. The bytes do not depend on
# how many Python threads call an adapter of their own.
BYTES_SHA256 = "8a60b8fa36c70a2ae7d7069418500aeb12b3e61e6205b6e7e96f4357dd916b4e"


class LogCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(self, log: list[int], token: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.log = log
        self.token = token

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        assert atoms.info.get("image") is not None
        self.log.append(int(atoms.info["image"]))
        n = len(atoms)
        self.results["energy"] = float(atoms.positions[0, 0])
        forces = np.zeros((n, 3))
        forces[:, 0] = 1.0
        forces[:, 1] = float(self.token)
        self.results["forces"] = forces


class BoomCalc(Calculator):
    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(self, kind: type[BaseException], text: str) -> None:
        super().__init__()
        self.kind = kind
        self.text = text

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        raise self.kind(self.text)


class OverlapCalc(Calculator):
    """Fails if calculate is entered a second time before the first returns."""

    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.overlap = False

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active > 1:
                self.overlap = True
        try:
            time.sleep(0.05)
            assert atoms is not None
            n = len(atoms)
            self.results["energy"] = 0.0
            self.results["forces"] = np.zeros((n, 3))
        finally:
            with self._lock:
                self.active -= 1


def _atoms(calc: Calculator, image: int, x0: float = 0.0) -> Atoms:
    atoms = Atoms("Au2", positions=[[x0, 0.0, 0.0], [1.5, 0.0, 0.0]], pbc=False)
    atoms.info["image"] = image
    atoms.calc = calc
    return atoms


def test_shared_calculator_is_refused() -> None:
    calc = LogCalc([], 0)
    images = [_atoms(calc, 0, 0.0), _atoms(calc, 1, 1.0)]
    with pytest.raises(ValueError, match="share a calculator"):
        SurfaceAdapter(images, allow_shared_calculator=False)


def test_row_order_and_constraints() -> None:
    log: list[int] = []
    shared = LogCalc(log, 7)
    other = LogCalc(log, 3)
    images = [
        _atoms(shared, 0, 0.0),
        _atoms(other, 1, 0.0),
        _atoms(shared, 2, 0.0),
    ]
    images[0].set_constraint(FixAtoms(indices=[0]))
    adapter = SurfaceAdapter(images, allow_shared_calculator=True)
    ids = np.array([2, 0, 1], dtype=np.int64)
    assert adapter.owners(ids).dtype == np.uint64
    assert int(adapter.owners(ids)[0]) == int(adapter.owners(ids)[1])
    assert int(adapter.owners(ids)[0]) != int(adapter.owners(ids)[2])
    assert adapter.reentrant(ids).dtype == np.uint8
    assert adapter.reentrant(ids).tolist() == [0, 0, 0]
    x = np.stack([image.positions.reshape(-1).copy() for image in (images[2], images[0], images[1])])
    x[0, 0] = 4.0
    x[1, 0] = 5.0
    x[2, 0] = 6.0
    energies, gradients = adapter(ids, x)
    # The calling thread keeps the request order. It does not regroup.
    assert log == [2, 0, 1]
    assert energies.shape == (3,)
    assert gradients.shape == (3, 6)
    assert energies[0] == 4.0
    assert energies[1] == 5.0
    assert energies[2] == 6.0
    assert np.array_equal(images[2].positions, [[4.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    raw = adapter.raw_forces
    assert raw is not None
    # Image 0 is request row 1. Atom 0 is fixed, so its constrained
    # force is 0. Atom 1 keeps the calculator's y component, token 7.
    assert raw[1, 0] == 1.0
    assert gradients[1, 0] == 0.0
    assert gradients[1, 4] == -7.0
    # Unconstrained image 2 is request row 0.
    assert gradients[0, 0] == -1.0
    assert raw[0, 0] == 1.0
    assert gradients[0, 4] == -7.0


def test_exception_leaves_raw_forces_unset() -> None:
    ok = LogCalc([], 1)
    bad = BoomCalc(RuntimeError, "worker-boom")
    adapter = SurfaceAdapter([_atoms(ok, 0), _atoms(bad, 1)])
    ids = np.arange(2, dtype=np.int64)
    x = np.zeros((2, 6), dtype=np.float64)
    with pytest.raises(RuntimeError, match="worker-boom"):
        adapter(ids, x)
    assert adapter.raw_forces is None


def test_keyboard_interrupt() -> None:
    bad = BoomCalc(KeyboardInterrupt, "stop")
    adapter = SurfaceAdapter([_atoms(bad, 0)])
    ids = np.array([0], dtype=np.int64)
    x = np.zeros((1, 6), dtype=np.float64)
    with pytest.raises(KeyboardInterrupt):
        adapter(ids, x)
