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


def test_ids_and_x_reject_dtype_and_order() -> None:
    first = LogCalc([], 1)
    second = LogCalc([], 2)
    adapter = SurfaceAdapter([_atoms(first, 0), _atoms(second, 1)])
    ids = np.array([0, 1], dtype=np.int64)
    x = np.zeros((2, 6), dtype=np.float64)
    with pytest.raises(TypeError, match="ids"):
        adapter(np.array([0.9, 1.2]), x)
    with pytest.raises(TypeError, match="ids"):
        adapter.owners(np.array([0.0, 1.0]))
    with pytest.raises(TypeError, match="ids"):
        adapter.reentrant(np.arange(4, dtype=np.int64)[::2])
    with pytest.raises(TypeError, match="x"):
        adapter(ids, np.zeros((2, 6), dtype=np.float32))
    fortran = np.asfortranarray(np.zeros((2, 6), dtype=np.float64))
    with pytest.raises(TypeError, match="x"):
        adapter(ids, fortran)
    energies, _gradients = adapter(ids, x)
    assert energies.shape == (2,)


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


def _fixture_blob() -> bytes:
    calcs = [LogCalc([], i) for i in range(4)]
    images = [_atoms(calc, i, 0.0) for i, calc in enumerate(calcs)]
    adapter = SurfaceAdapter(images)
    ids = np.array([3, 1, 0, 2], dtype=np.int64)
    x = np.arange(4 * 6, dtype=np.float64).reshape(4, 6)
    energies, gradients = adapter(ids, x)
    assert adapter.raw_forces is not None
    return energies.tobytes() + gradients.tobytes() + adapter.raw_forces.tobytes()


def test_bytes_match_across_threads() -> None:
    assert hashlib.sha256(_fixture_blob()).hexdigest() == BYTES_SHA256
    for callers in (1, 2, 8):
        barrier = threading.Barrier(callers)
        found: list[str] = []
        lock = threading.Lock()

        def once() -> None:
            barrier.wait()
            digest = hashlib.sha256(_fixture_blob()).hexdigest()
            with lock:
                found.append(digest)

        threads = [threading.Thread(target=once) for _ in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert found == [BYTES_SHA256] * callers


def test_eight_threads_hammer_the_adapter() -> None:
    entered = threading.Event()
    release = threading.Event()
    state = threading.Lock()
    active = 0
    max_active = 0

    class HoldCalc(Calculator):
        implemented_properties = ["energy", "forces"]
        nolabel = True

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: list[str] = all_changes,
        ) -> None:
            nonlocal active, max_active
            super().calculate(atoms, properties, system_changes)
            with state:
                active += 1
                max_active = max(max_active, active)
                overlapped = active > 1
            try:
                assert not overlapped
                entered.set()
                assert release.wait(timeout=5)
                assert atoms is not None
                n = len(atoms)
                self.results["energy"] = 0.0
                self.results["forces"] = np.zeros((n, 3))
            finally:
                with state:
                    active -= 1

    adapter = SurfaceAdapter([_atoms(HoldCalc(), 0)])
    ids = np.array([0], dtype=np.int64)
    x = np.zeros((1, 6), dtype=np.float64)
    started = threading.Barrier(8)
    results: list[str] = []
    done = threading.Condition()

    def once() -> None:
        started.wait()
        try:
            adapter(ids, x)
            text = "ok"
        except RuntimeError as exc:
            text = str(exc)
        with done:
            results.append(text)
            done.notify_all()

    threads = [threading.Thread(target=once) for _ in range(8)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=5)
    with done:
        saw_contention = done.wait_for(
            lambda: sum(item == "SurfaceAdapter is in use by another thread" for item in results) >= 7,
            timeout=5,
        )
    release.set()
    for thread in threads:
        thread.join()
    assert saw_contention
    assert results.count("ok") == 1
    assert results.count("SurfaceAdapter is in use by another thread") == 7
    assert max_active == 1
    energies, gradients = adapter(ids, x)
    assert energies.shape == (1,)
    assert gradients.shape == (1, 6)


def test_shared_calculator_calculate_stays_serial() -> None:
    """One calculator is never inside calculate on two workers at once."""
    calc = OverlapCalc()
    calc.reentrant = False
    images = [_atoms(calc, 0), _atoms(calc, 1)]
    adapter = SurfaceAdapter(images, allow_shared_calculator=True)

    def callback(group_ids: np.ndarray, group_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energies = np.empty(group_ids.shape[0], dtype=np.float64)
        gradients = np.empty_like(group_x)
        for local, image_id in enumerate(group_ids.tolist()):
            atoms = images[int(image_id)]
            coords = np.array(group_x[local], dtype=np.float64, copy=True).reshape((-1, 3))
            atoms.set_positions(coords, apply_constraint=False)
            energies[local] = float(atoms.get_potential_energy())
            gradients[local] = -np.array(atoms.get_forces(), dtype=np.float64, copy=True).reshape(-1)
        return energies, gradients

    ids = np.array([1, 0], dtype=np.int64)
    x = np.zeros((2, 6), dtype=np.float64)
    owners = adapter.owners(ids)
    flags = adapter.reentrant(ids)
    assert flags.tolist() == [0, 0]
    assert owners.tolist() == [owners[0], owners[0]]
    energies, gradients = drive_eval(callback, ids, x, owners, flags, 8, ClaimSlot())
    assert energies.shape == (2,)
    assert gradients.shape == (2, 6)
    assert not calc.overlap
    assert calc.max_active == 1


def test_reentrant_owner_groups_run_concurrently() -> None:
    n = 4
    gate = threading.Barrier(n)
    state = threading.Lock()
    active = 0
    max_active = 0

    class GateCalc(Calculator):
        implemented_properties = ["energy", "forces"]
        nolabel = True

        def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list[str] | None = None,
            system_changes: list[str] = all_changes,
        ) -> None:
            nonlocal active, max_active
            super().calculate(atoms, properties, system_changes)
            with state:
                active += 1
                max_active = max(max_active, active)
            try:
                gate.wait(timeout=5)
                assert atoms is not None
                count = len(atoms)
                self.results["energy"] = 0.0
                self.results["forces"] = np.zeros((count, 3))
            finally:
                with state:
                    active -= 1
    calcs = [GateCalc() for _ in range(n)]
    for calc in calcs:
        calc.reentrant = True
    images = [_atoms(calc, i) for i, calc in enumerate(calcs)]
    adapter = SurfaceAdapter(images)
    ids = np.arange(n, dtype=np.int64)
    x = np.zeros((n, 6), dtype=np.float64)
    owners = adapter.owners(ids)
    flags = adapter.reentrant(ids)
    assert len(set(int(item) for item in owners.tolist())) == n
    assert flags.tolist() == [1] * n

    def callback(group_ids: np.ndarray, group_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energies = np.empty(group_ids.shape[0], dtype=np.float64)
        gradients = np.empty_like(group_x)
        for local, image_id in enumerate(group_ids.tolist()):
            atoms = images[int(image_id)]
            coords = np.array(group_x[local], dtype=np.float64, copy=True).reshape((-1, 3))
            atoms.set_positions(coords, apply_constraint=False)
            energies[local] = float(atoms.get_potential_energy())
            gradients[local] = -np.array(atoms.get_forces(), dtype=np.float64, copy=True).reshape(-1)
        return energies, gradients

    energies, gradients = drive_eval(callback, ids, x, owners, flags, n, ClaimSlot())
    assert energies.shape == (n,)
    assert gradients.shape == (n, 6)
    assert max_active == n


def test_shared_calculator_is_checked_at_call_time() -> None:
    first = LogCalc([], 1)
    second = LogCalc([], 2)
    images = [_atoms(first, 0), _atoms(second, 1)]
    adapter = SurfaceAdapter(images)
    images[1].calc = first
    ids = np.array([0, 1], dtype=np.int64)
    x = np.zeros((2, 6), dtype=np.float64)
    with pytest.raises(ValueError, match="share a calculator"):
        adapter(ids, x)
    with pytest.raises(ValueError, match="share a calculator"):
        adapter.owners(ids)
    with pytest.raises(ValueError, match="share a calculator"):
        adapter.reentrant(ids)
    shared = SurfaceAdapter([_atoms(first, 0), _atoms(first, 1)], allow_shared_calculator=True)
    energies, _gradients = shared(ids, x)
    assert energies.shape == (2,)


def test_claim_covers_raw_forces() -> None:
    calc = LogCalc([], 1)
    adapter = SurfaceAdapter([_atoms(calc, 0)])
    ids = np.array([0], dtype=np.int64)
    x = np.zeros((1, 6), dtype=np.float64)
    seen = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with adapter.claim():
            seen.set()
            assert release.wait(timeout=5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert seen.wait(timeout=5)
    try:
        adapter(ids, x)
    except RuntimeError as exc:
        assert str(exc) == "SurfaceAdapter is in use by another thread"
        assert adapter.raw_forces is None
    else:
        raise AssertionError("the call must see the in-use claim")
    finally:
        release.set()
        holder.join()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: adapter(ids, x)).result()
    assert adapter.raw_forces is not None
