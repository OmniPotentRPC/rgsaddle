"""rgsaddle nanobind surface: status mapping, the ASE image callback, and Stencil."""

from __future__ import annotations

import collections.abc

import numpy
import numpy.typing



def version() -> str:
    """Package version string."""

RGSADDLE_FORCE_L2: int = 0

RGSADDLE_FORCE_LINF: int = 1

RGSADDLE_FORCE_MAX_ATOM: int = 2

RGSADDLE_MINMODE_DIMER: int = 0

RGSADDLE_MINMODE_LANCZOS: int = 1

RGSADDLE_IRC_GS2: int = 0

RGSADDLE_IRC_MOROKUMA: int = 1

class NumericalError(ArithmeticError):
    pass

class ClaimSlot:
    def __init__(self) -> None: ...

    def __enter__(self) -> ClaimSlot:
        """Claim SurfaceAdapter until __exit__."""

    def __exit__(self, exc_type: object | None, exc_value: object | None, traceback: object | None) -> bool:
        """Release the claim. Exceptions propagate."""

def last_error() -> str:
    """
    Thread-local slot reserved for the C ABI last-error text. Returns an empty string unless an error text was set. raise_status reads it and then clears it.
    """

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value. The message is the stored text, or the status name when that text is empty. The stored text is cleared. True on OK. False is not used: convergence is not a status code.
    """

class StencilReport:
    phase: int
    status: int
    max_force: float
    ci_index: int
    iteration: int
    surface_rows: int
    total_evaluations: int
    zoom_lo: int
    zoom_hi: int
    curvature: float

    def zoom_window(self) -> tuple[int, int] | None: ...

class StencilBuilder:
    def ci_neb(self) -> StencilBuilder: ...
    def plain(self) -> StencilBuilder: ...
    def zoom(self) -> StencilBuilder: ...
    def autoneb(self) -> StencilBuilder: ...
    def dimer(self) -> StencilBuilder: ...
    def lanczos(self) -> StencilBuilder: ...
    def energy_weighted(self, k_min: float, k_max: float) -> StencilBuilder: ...
    def threads(self, count: int) -> StencilBuilder: ...
    def preset(self, preset: int) -> StencilBuilder: ...
    def ends(self, ends: int) -> StencilBuilder: ...
    def ci_tol(self, tol: float) -> StencilBuilder: ...
    def zoom_stability(self, count: int) -> StencilBuilder: ...
    def zoom_force_abs(self, force: float) -> StencilBuilder: ...
    def set_i64(self, key: int, value: int) -> None: ...
    def set_f64(self, key: int, value: float) -> None: ...
    def build(self) -> Stencil: ...

class Stencil:
    @staticmethod
    def band(positions: numpy.typing.NDArray[numpy.float64]) -> StencilBuilder: ...
    @staticmethod
    def point(x0: numpy.typing.NDArray[numpy.float64]) -> StencilBuilder: ...
    def step(self, surface: collections.abc.Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> StencilReport: ...
    def run(self, surface: collections.abc.Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], max_steps: int) -> StencilReport: ...
    def positions(self) -> numpy.typing.NDArray[numpy.float64]: ...
