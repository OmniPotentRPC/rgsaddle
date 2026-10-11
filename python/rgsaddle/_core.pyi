"""rgsaddle nanobind surface: status mapping and the F5 eval loop."""

from collections.abc import Callable
from typing import Annotated

import numpy
from numpy.typing import NDArray


def version() -> str:
    """Package version string."""

class NumericalError(ArithmeticError):
    pass

class HoldGate:
    def __init__(self) -> None: ...

    def release(self) -> None:
        """Let the holding thread drop the claim."""

    def held(self) -> bool:
        """True once the holding thread owns the claim."""

class ClaimSlot:
    def __init__(self) -> None: ...

    def work(self, hold_ms: int = 0) -> int:
        """Claim SurfaceAdapter, hold, and release (F3, F4)."""

    def hold_until(self, gate: HoldGate) -> None:
        """Claim and wait, with the GIL released, until gate.release()."""

def last_error() -> str:
    """Thread-local last-error text (F2, F7)."""

def set_last_error(text: str) -> None:
    """Replace the thread-local last-error text."""

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value (F7). True on OK. False is not used: convergence is not a status code.
    """

def schedule_canonical(ids: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)], owners: Annotated[NDArray[numpy.uint64], dict(shape=(None,), order='C', device='cpu', writable=False)], reentrant: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], threads: int) -> str:
    """Canonical F5 schedule. The worker count is not part of the text (F8)."""

def echo_f64(values: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> NDArray[numpy.float64]:
    """Byte-copy a float64 array. -0.0, NaN and subnormals pass through (F6)."""

def drive_eval(callback: Callable, ids: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)], x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], owners: Annotated[NDArray[numpy.uint64], dict(shape=(None,), order='C', device='cpu', writable=False)], reentrant: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], threads: int, slot: ClaimSlot) -> tuple:
    """
    Fake stencil eval loop. Groups rows with the F5 scheduler and calls callback(ids, x) through the trampoline.
    """
