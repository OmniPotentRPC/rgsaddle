"""
rgsaddle nanobind surface: status mapping, the ASE image callback, and Stencil.
"""

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

    @property
    def phase(self) -> int:
        """0 for a band or point step, 1 for a zoom pack, 2 for an insertion."""

    @property
    def ci_index(self) -> int:
        """Climbing image, or -1 when there is none."""

    @property
    def zoom_lo(self) -> int:
        """First row of the zoom window, or -1."""

    @property
    def zoom_hi(self) -> int:
        """Last row of the zoom window, or -1."""

    def converged(self) -> bool:
        """True when status is converged."""

class Stencil:
    @staticmethod
    def point(x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', writable=False)], preset: str = 'dimer', mode: object | None = None, dr: float = 0.001, force_tol: float = 0.001, max_move: float = 0.2, rotation_tol: float = 0.0001, max_rotations: int = 20, krylov_dim: int = 12, threads: int = 1, art_threshold: float = 0.0, art_push: int = 20) -> Stencil:
        """
        Point search. dimer and lanczos follow MinModeSession. A band preset name raises ValueError.
        """

    @staticmethod
    def band(x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], preset: str = 'ci-neb', k: float = 5.0, k_min: float = 0.972, k_max: float = 9.72, force_tol: float = 0.001, max_move: float = 0.2, threads: int = 1, spring: int = -1, zoom_stability: int = -1, zoom_alpha: float = 0.5, zoom_offset: int = -1, zoom_interp: int = -1, zoom_ends: int = -1, zoom_force: float = float('nan'), zoom_force_kind: int = -1, image_scale: float = float('nan'), band_slack: float = 10.0, ci_trigger: float = 0.5, ci_trigger_force: float = 0.0, n_simul: int = 3, n_max: int = 10, tangent: int = 1, zoom_max_iter: int = -1) -> Stencil:
        """
        Band search. neb, dneb, ci-neb, energy-weighted-neb, zoom-neb, autoneb, oci-neb, idpp, and s-idpp run. Zoom ends stay fixed unless zoom_ends is 1.
        """

    def step(self, surface: Callable[..., StencilReport]) -> StencilReport:
        """
        One step. surface(x) returns (energy, gradient) with x shaped (rows, dof).
        """

    def run(self, surface: Callable[..., StencilReport], max_steps: int) -> StencilReport:
        """Loop of step until convergence or max_steps."""

    def positions(self) -> object:
        """
        Copy of the current position. A point is one dimension. A band is (rows, dof).
        """

    def image_ids(self) -> list[int]:
        """Stable image ids. A point stencil raises ValueError."""

    def mode(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Copy of the mode, or None for a minimiser."""

    def curvature(self) -> float:
        """Curvature along the mode."""

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value. The message is the stored text, or the status name when that text is empty. The stored text is cleared. True on OK. False is not used: convergence is not a status code.
    """
