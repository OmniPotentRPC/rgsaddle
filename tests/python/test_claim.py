"""A second caller gets RuntimeError and the flag is released."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rgsaddle._core import ClaimSlot
from rgsaddle._core._testing import HoldGate, claim_hold_until, claim_work


def test_claim_released_across_calls() -> None:
    slot = ClaimSlot()
    assert claim_work(slot, 0) == 1
    assert claim_work(slot, 0) == 1


def test_claim_context_manager() -> None:
    slot = ClaimSlot()
    with slot:
        try:
            claim_work(slot, 0)
        except RuntimeError as exc:
            assert str(exc) == "SurfaceAdapter is in use by another thread"
        else:
            raise AssertionError("the context manager must hold the claim")
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
        raise AssertionError("second caller must see the in-use error")
    gate.release()
    holder.join()
    assert claim_work(slot, 0) == 1


def test_exit_from_another_thread_keeps_the_claim() -> None:
    slot = ClaimSlot()
    entered = threading.Event()
    release = threading.Event()

    def worker() -> None:
        slot.__enter__()
        entered.set()
        release.wait()
        slot.__exit__(None, None, None)

    holder = threading.Thread(target=worker)
    holder.start()
    assert entered.wait(2.0)
    with pytest.raises(RuntimeError, match="does not hold the claim"):
        slot.__exit__(None, None, None)
    try:
        claim_work(slot, 0)
    except RuntimeError as exc:
        assert str(exc) == "SurfaceAdapter is in use by another thread"
    else:
        release.set()
        holder.join()
        raise AssertionError("a foreign exit must leave the claim held")
    release.set()
    holder.join()
    assert claim_work(slot, 0) == 1


def test_freeing_a_held_slot_releases_the_claim() -> None:
    slot = ClaimSlot()
    slot.__enter__()
    del slot
    assert claim_work(ClaimSlot(), 0) == 1


def test_freeing_a_held_slot_from_another_thread() -> None:
    held: list[ClaimSlot] = []

    def enter() -> None:
        slot = ClaimSlot()
        slot.__enter__()
        held.append(slot)

    worker = threading.Thread(target=enter)
    worker.start()
    worker.join()
    slot = held.pop()
    del slot
    assert claim_work(ClaimSlot(), 0) == 1


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
