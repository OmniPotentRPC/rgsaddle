"""F3 claim. A second caller gets RuntimeError and the flag is released."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from rgsaddle._core import ClaimSlot
from rgsaddle._core._testing import HoldGate, claim_hold_until, claim_work


def test_claim_released_across_calls() -> None:
    slot = ClaimSlot()
    assert claim_work(slot, 0) == 1
    assert claim_work(slot, 0) == 1


def test_claim_held_until_release() -> None:
    slot = ClaimSlot()
    gate = HoldGate()
    holder = threading.Thread(target=claim_hold_until, args=(slot, gate))
    holder.start()
    deadline = threading.Event()
    for _ in range(2000):
        if gate.held():
            break
        deadline.wait(0.001)
    assert gate.held()
    try:
        claim_work(slot, 0)
    except RuntimeError as exc:
        assert str(exc) == "SurfaceAdapter is in use by another thread"
    else:
        gate.release()
        holder.join()
        raise AssertionError("second caller must see the F3 error")
    gate.release()
    holder.join()
    assert claim_work(slot, 0) == 1


def test_eight_threads_raise_or_finish() -> None:
    slot = ClaimSlot()
    start = threading.Barrier(8)

    def once() -> str:
        start.wait()
        try:
            claim_work(slot, 50)
        except RuntimeError as exc:
            return str(exc)
        return "ok"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: once(), range(8)))
    assert "ok" in results
    for item in results:
        assert item == "ok" or item == "SurfaceAdapter is in use by another thread"
    assert claim_work(slot, 0) == 1
