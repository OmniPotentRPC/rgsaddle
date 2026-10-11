"""SIGINT during a slow callback, on the serial path and the parallel path."""

from __future__ import annotations

import _thread
import threading
import time

import numpy as np
import pytest

from rgsaddle._core import ClaimSlot
from rgsaddle._core._testing import drive_eval

# interrupt_main delivers SIGINT to the main thread. pytest-run-parallel
# must not move these tests onto a worker, and must not repeat them.
pytestmark = [pytest.mark.thread_unsafe, pytest.mark.iterations(1)]


def _slow(ids: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time.sleep(2.0)
    return np.zeros(ids.shape[0], dtype=np.float64), np.zeros_like(x)


def _interrupt_during(threads: int) -> None:
    slot = ClaimSlot()
    ids = np.arange(threads, dtype=np.int64)
    x = np.zeros((threads, 1), dtype=np.float64)
    owners = np.arange(threads, dtype=np.uint64)
    flags = np.zeros(threads, dtype=np.uint8)
    timer = threading.Timer(0.05, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            drive_eval(_slow, ids, x, owners, flags, threads, slot)
    finally:
        timer.cancel()


def test_sigint_on_the_serial_path() -> None:
    _interrupt_during(1)


def test_sigint_on_the_parallel_path() -> None:
    _interrupt_during(2)
