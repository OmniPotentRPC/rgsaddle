"""rgsaddle nanobind surface: status mapping and the ASE image callback."""

from collections.abc import Callable
from typing import Annotated

import numpy
import numpy.typing
from numpy.typing import NDArray


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

class StencilReport:
    @property
    def status(self) -> int:
        """0 while running, 1 when converged."""

    @property
    def max_force(self) -> float:
        """Maximum absolute force component at the step."""

    @property
    def curvature(self) -> float:
        """Curvature along the mode. NaN for a minimiser."""

    @property
    def iteration(self) -> int:
        """Steps taken."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations in this call."""

    def converged(self) -> bool:
        """True when status is converged."""

class Stencil:
    @staticmethod
    def point(x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], preset: str = 'dimer', mode: object | None = None, dr: float = 0.001, force_tol: float = 0.001, max_move: float = 0.2, rotation_tol: float = 0.0001, max_rotations: int = 20, krylov_dim: int = 12, threads: int = 1, art_threshold: float = 0.0, art_push: int = 20) -> Stencil:
        """
        Point search. dimer and lanczos follow MinModeSession. rotation_tol 1e-4 also stops the dimer after one rotation once the rotational force is under 1. rotation_angle_tol is the Rust default of five degrees (0 would disable that angle test) and is not a Python argument. A curvature that is not negative steps by max_move along the mode. Band preset names raise ValueError.
        """

    @staticmethod
    def band(x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], preset: str = 'ci-neb') -> None:
        """Band presets stay on feat/band-methods-zoom and raise ValueError."""

    def step(self, surface: Callable[..., StencilReport]) -> StencilReport:
        """
        One step. surface(x) returns (energy, gradient) with x shaped (rows, dof).
        """

    def run(self, surface: Callable[..., StencilReport], max_steps: int) -> StencilReport:
        """Loop of step until convergence or max_steps."""

    def positions(self) -> NDArray[numpy.float64]:
        """Copy of the current position."""

    def mode(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Copy of the mode, or None for a minimiser."""

    def curvature(self) -> float:
        """Curvature along the mode."""

class ConnectReport:
    @property
    def position(self) -> NDArray[numpy.float64]:
        """Copy of the returned geometry."""

    @property
    def energy(self) -> float:
        """Energy at that geometry."""

    @property
    def force_norm(self) -> float:
        """Euclidean norm of the Cartesian gradient."""

    @property
    def kind(self) -> str:
        """
        s for an accepted saddle, u for an upper bound, f when a segment could not be split.
        """

    @property
    def hessian_index(self) -> int:
        """
        Negative eigenvalues of the rigid-projected Hessian, or -1 when it was not computed.
        """

    @property
    def evaluations(self) -> int:
        """Surface evaluations, including a Hessian that decided acceptance."""

def connect(surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], left: numpy.typing.NDArray[numpy.float64], right: numpy.typing.NDArray[numpy.float64], force_tol: float = 0.001, max_calls: int = 250000, quench_nodes: bool = True) -> ConnectReport:
    """
    Double-ended connect. surface(x) returns (energy, gradient) for one image shaped (1, 3N).
    """

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value. The message is the stored text, or the status name when that text is empty. The stored text is cleared. True on OK. False is not used: convergence is not a status code.
    """
