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

RGSADDLE_IRC_SELLA_SPHERE: int = 2

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

THERMODYNAMIC_DERIVATIVE: str = 'potential.full_minus_harmonic'

def hyperplane_thermodynamic_integration(sampler: Callable[..., numpy.typing.NDArray[numpy.float64]], point: numpy.typing.NDArray[numpy.float64], normal: numpy.typing.NDArray[numpy.float64], thermal: float = 0.02585, nodes: int = 8, thermalize: int = 500, stride: int = 5, samples: int = 2000, blocks: int = 20, seed: int = 1) -> tuple[float, float, int]:
    """
    Fixed-plane lambda TI through a Python sampler. The callback receives point, normal, lambda, thermal, seed, thermalize, stride and samples, and returns float64 C-contiguous V_full - V_harmonic observations.
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
        Point search. dimer and lanczos follow MinModeSession. The dimer keeps its Polak-Ribiere rotation history across translations. rotation_angle_tol is the Rust default of five degrees and is not a Python argument, so the dimer stops on that angle or on rotation_tol without inferring the classic torque window. At rotation_tol 1e-4 and looser, lanczos also stops at two Krylov vectors once the Ritz residual divided by the absolute Ritz value is under 1. A curvature that is not negative steps by max_move along the mode. Band preset names raise ValueError.
        """

    @staticmethod
    def band(x: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', writable=False)], preset: str = 'ci-neb') -> None:
        """Band presets are not implemented and raise ValueError."""

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> StencilReport:
        """
        One step. surface(x) returns (energy, gradient) with x shaped (rows, dof).
        """

    def run(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], max_steps: int) -> StencilReport:
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

class LinebOrbit:
    @property
    def positions(self) -> NDArray[numpy.float64]:
        """Measured Cartesian positions over one imaginary-time period."""

    @property
    def gradients(self) -> NDArray[numpy.float64]:
        """Cartesian energy gradients paired with the measured positions."""

    @property
    def max_abs_residual(self) -> float: ...

    @property
    def rms_residual(self) -> float: ...

    @property
    def max_bead(self) -> int: ...

    @property
    def max_coordinate(self) -> int: ...

class LinebResult:
    @property
    def orbit(self) -> LinebOrbit | None:
        """Bead observations when requested for a rate."""

    @property
    def positions(self) -> NDArray[numpy.float64]:
        """Optimized images shaped (n_images, 3 * n_atoms)."""

    @property
    def action(self) -> float:
        """Spline abbreviated action W."""

    @property
    def half_period(self) -> float:
        """Half period from the action energy derivative."""

    @property
    def half_period_dynamics(self) -> float:
        """Half period from constrained dynamics along the path."""

    @property
    def temperature(self) -> float:
        """Instanton temperature from half_period."""

    @property
    def temperature_dynamics(self) -> float:
        """Instanton temperature from half_period_dynamics."""

    @property
    def log_rate_over_reactant(self) -> float:
        """ln(k Z_r), or NaN when beads is 0."""

    @property
    def curvature_evaluations(self) -> int: ...

    @property
    def bead_evaluations(self) -> int: ...

    @property
    def image_hessian_step(self) -> float | None: ...

    @property
    def turning(self) -> object:
        """Measured return-turn agreement in the surface's units, or None."""

    @property
    def negative_modes(self) -> int | None:
        """
        Number of negative retained fluctuation eigenvalues, or None without a rate.
        """

    @property
    def modes(self) -> int | None:
        """
        Fluctuation eigenvalues retained after gauge projection, or None without a rate.
        """

    @property
    def iterations(self) -> int:
        """Optimizer steps taken."""

    @property
    def surface_calls(self) -> dict[str, int]:
        """
        Energy and gradient evaluations of individual configurations, by phase.
        """

    @property
    def optimizer_calls(self) -> int:
        """Surface evaluations inside the optimizer."""

    @property
    def measure_calls(self) -> int:
        """Surface evaluations of the action and period quadrature."""

    @property
    def dynamics_calls(self) -> int:
        """Surface evaluations of the constrained dynamics."""

    @property
    def rate_calls(self) -> int:
        """Surface evaluations of the harmonic rate, or 0 when beads is 0."""

def lineb_solve(surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], positions: numpy.typing.NDArray[numpy.float64], atom_masses: numpy.typing.NDArray[numpy.float64], energy: float, opt_tol: float, perp_tol: float, molecular: bool = False, max_steps: int = 1000, beads: int = 0, spring: float = 0.1, endpoint_scale: float = 0.2, time_step: float = 0.1, max_move: float = 0.2, turning_width: float = 0.001, hbar: float = 1.0, boltzmann: float = 1.0, quadrature_pairs: int = 4, endpoint_kappa: float | None = None, linear_seed: bool = False, endpoint_convergence: bool = False, projected_verlet: bool = False, return_orbit: bool = False, interpolated_force_clock: bool = False, interpolated_hessian: bool = False, image_hessian_step: float | None = None, batch_surface: bool = False) -> LinebResult:
    """
    Fixed-energy LI-NEB path and optional harmonic rate. positions is (n_images, 3N). atom_masses is one positive mass per atom; molecular=False expands them to coordinates, molecular=True aligns the free molecule. The default callback receives one image. With batch_surface=True, surface(x) receives (rows, 3N) and returns (rows,) energies and (rows, 3N) gradients in input order. Call counts count geometries. beads=0 skips the rate and leaves log_rate_over_reactant as NaN. A calculator exception propagates. An unconverged search raises with the C status.
    """

class SessionReport:
    @property
    def status(self) -> int:
        """0 while running, 1 when converged."""

    @property
    def max_force(self) -> float:
        """Largest force component, or NaN for an estimate."""

    @property
    def curvature(self) -> float:
        """Curvature along the mode; 0 on a band step."""

    @property
    def iteration(self) -> int:
        """Steps taken."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations in this call."""

    @property
    def ci_index(self) -> int:
        """Climbing image, or -1."""

    @property
    def rotations(self) -> int:
        """Dimer rotations or Lanczos actions."""

    @property
    def saddle_index(self) -> int:
        """Saddle index certified at the session's current position, or -1."""

class SaddleIndex:
    @property
    def certified(self) -> bool:
        """
        True when a certificate holds for the current point; otherwise the counts are the session's.
        """

    @property
    def index(self) -> int:
        """Curvatures under -negative_tol; 1 for a first-order saddle."""

    @property
    def converged(self) -> bool:
        """Every resolved pair met residual_tol, or the space filled."""

    @property
    def curvatures(self) -> NDArray[numpy.float64]:
        """Lowest curvatures resolved, ascending, energy per length squared."""

    @property
    def evaluations(self) -> int:
        """Surface evaluations of this certificate."""

    @property
    def total_evaluations(self) -> int:
        """
        Surface evaluations of every certificate and escape push of the session.
        """

    @property
    def escapes(self) -> int:
        """Escape pushes of the session."""

def saddle_index(surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], x: numpy.typing.NDArray[numpy.float64], gradient: numpy.typing.NDArray[numpy.float64] | None = None, mode: numpy.typing.NDArray[numpy.float64] | None = None, fixed: numpy.typing.NDArray[numpy.uint8] | None = None, lowest: int = 4, max_actions: int = 100, dr: float = 0.001, residual_tol: float = 0.01, negative_tol: float = 0.001) -> SaddleIndex:
    """
    Saddle index of x (ABI minor 40): the lowest curvatures on the free coordinates by block Lanczos on forward-difference Hessian actions, each one surface call. surface(x) is one image shaped (1, 3N). gradient at x saves one call; mode seeds the space; fixed is one uint8 per atom, 1 for a fixed atom.
    """

def project_band(positions: numpy.typing.NDArray[numpy.float64], energies: numpy.typing.NDArray[numpy.float64], gradients: numpy.typing.NDArray[numpy.float64], k: float = 0.1, climb: bool = False, projection: str = 'neb', tangent: str = 'improved', remove_translation: bool = False, force_tol: float = 0.001, fixed: numpy.typing.NDArray[numpy.uint8] | None = None, spring_ks: numpy.typing.NDArray[numpy.float64] | None = None, cell: numpy.typing.NDArray[numpy.float64] | None = None) -> tuple[numpy.typing.NDArray[numpy.float64], int]:
    """
    Project an evaluated band. Returns (forces, ci_index). forces is flat, one row per image including zero endpoints, in the ASE force convention. climb selects the interior image by the endpoint-slope rule on this call; a higher endpoint that slopes into the band does not climb. Units are the caller's. k is energy per length squared. spring_ks, when given, is one constant per spring (length images - 1) and replaces k. cell is the 3x3 row-major lattice; a zero row is not wrapped. fixed is one uint8 per atom, 1 for a fixed atom (ABI minor 39): its force rows are zero, tangents and springs use the free atoms, and the mean force stays. The endpoints must agree on every fixed atom.
    """

def dyneb_mask(positions: numpy.typing.NDArray[numpy.float64], energies: numpy.typing.NDArray[numpy.float64], forces: numpy.typing.NDArray[numpy.float64], fmax: float, scale_fmax: float) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.uint8]]:
    """
    ASE 3.29 DyNEB mask on a whole band (image-major positions, energies and projected forces). Returns (forces, frozen). An interior image whose largest per-atom force norm is strictly below fmax * (1 + distance * scale_fmax) is zeroed, except the highest-energy image. frozen is one uint8 per image, 1 when the optimizer would leave it in place.
    """

def idpp_path(reactant: numpy.typing.NDArray[numpy.float64], product: numpy.typing.NDArray[numpy.float64], n_images: int, cell: numpy.typing.NDArray[numpy.float64] | None = None, max_iterations: int = 5000, grad_tol: float = 0.001, max_move: float = 0.1) -> numpy.typing.NDArray[numpy.float64]:
    """
    Image-dependent pair-potential path, endpoints included, shape (n_images, 3N). Interior images are relaxed toward linearly interpolated minimum-image pair distances by L-BFGS. cell is a 3x3 row-major lattice, or None in free space. No surface is evaluated.
    """

def reparametrize(positions: numpy.typing.NDArray[numpy.float64], scheme: str = 'linear', anchor: int | None = None, cell: numpy.typing.NDArray[numpy.float64] | None = None) -> numpy.typing.NDArray[numpy.float64]:
    """
    Place the interior images of a band at equal arc length and return the new band. scheme 'linear' places them on the polyline; 'not-a-knot' (or 'spline') on the not-a-knot cubic spline through the images in their arc length (E, Ren and Vanden-Eijnden 2007). The first and last rows, and an interior anchor, do not move. cell is the 3x3 row-major cell; the chords are then minimum images. No surface is evaluated.
    """

def halve_raised_energy(previous: float, trial: float, step: float) -> tuple[bool, float]:
    """
    Compare a trial energy with the energy before the step. Returns (raised, new_step). A trial more than 1e-4 above previous halves step. No surface is evaluated.
    """

def trust_update(radius: float, predicted: float, actual: float, step_norm: float, sigma_inc: float, sigma_dec: float, rho_inc: float, rho_dec: float, delta_min: float) -> float:
    """
    Trust radius from a predicted energy change and the change that occurred. actual is the trial energy minus the energy before the step. A prediction below 1e-14 in absolute value returns radius. The remaining arguments are the Sella schedule. No surface is evaluated.
    """

class MinMode:
    def __init__(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], mode: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], kind: str = 'dimer', dr: float = 0.001, force_tol: float = 0.001, max_move: float = 0.2, rotation_tol: float = 0.0001, max_rotations: int = 20, krylov_dim: int = 12, rotation_stop_length: float = 0.0) -> None:
        """
        Dimer or Lanczos minimum-mode session translated by L-BFGS. Lengths are the caller's. A positive rotation_stop_length stops a rotation once |C| * rotation_stop_length * |sin w| is at most force_tol (Kastner and Sherwood 2008); 0 turns that stop off.
        """

    def estimate(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """Refresh the lowest mode. surface(x) is one image shaped (1, 3N)."""

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """
        Estimate the mode, invert the force, and take one translation. A converged step certifies the saddle index; with escape_attempts set, a step that pushes off a saddle of index two or more reports status 0.
        """

    def set_position(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], gradient: object | None = None) -> None:
        """
        Move to a new point. gradient, when given, skips the next centre evaluation.
        """

    def position(self) -> NDArray[numpy.float64]:
        """Copy of the current position."""

    def mode(self) -> NDArray[numpy.float64]:
        """Copy of the current mode."""

    def saddle_index(self) -> SaddleIndex:
        """
        Certificate of the current position (ABI minor 40). On by default; a push or a move drops it.
        """

    def set_saddle_check(self, enabled: bool = True, escape_attempts: int = 0, escape_push: float = 0.1, lowest: int = 4, max_actions: int = 100, dr: float = 0.001, residual_tol: float = 0.01, negative_tol: float = 0.001) -> None:
        """
        Replace the saddle check. enabled=False turns it off. escape_attempts above 0 pushes a converged point of index two or more by escape_push, the largest per-atom displacement, doubled on each later attempt, along its extra negative modes and searches again. The block size stays 2, finite differences stay forward, and the seed stays at the deterministic C default.
        """

class Band:
    def __init__(self, positions: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], k: float = 5.0, climb: bool = True, ci_after: float = 0.5, ci_after_rel: float = 0.0, projection: str = 'neb', tangent: str = 'improved', method: str = 'fire', memory: int = 20, force_tol: float = 0.001, max_move: float = 0.2, time_step: float = 1.0, remove_translation: bool = True, min_mode: bool = False, string: bool = False, reparametrization: str = 'not-a-knot', cell: object | None = None, fixed: object | None = None) -> None:
        """
        Stepping band session for NEB and the simplified string method (E, Ren and Vanden-Eijnden 2007). positions is (n_images >= 3, 3N). string=True evolves interior images under the full potential force and reparametrizes to equal arc length on 'not-a-knot' (or 'spline') or 'linear', anchoring an armed climbing image.
        """

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """
        One band or simplified-string step. surface(x) receives (rows, 3N) and returns (energies, gradients).
        """

    def run(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], max_steps: int = 1000) -> SessionReport:
        """Loop step until convergence or max_steps."""

    def positions(self) -> NDArray[numpy.float64]:
        """Copy of the current band positions, shaped (n_images, 3N)."""

    def set_positions(self, positions: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)]) -> None:
        """Replace the band positions; unchanged rows keep cached evaluations."""

    def evaluation(self) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """Cached (energies, gradients, projected) at the current band."""

    def set_string(self, enabled: bool = True, reparametrization: str = 'not-a-knot') -> None:
        """
        Enable or clear the simplified string method on the band (ABI minor 44).
        """

    def set_free_end(self, enabled: bool = True, side: str = 'last', target_energy: object | None = None, max_cuts: int = 2, cut_force: float = 0.0, energy_tol: float = 0.001, stability: int = 2, window_offset: int = 1, cubic: bool = True) -> None:
        """Enable or clear free-end adaptive NEB on the band (ABI minor 45)."""

    def set_refine(self, enabled: bool = True, threshold: float = 0.0, memory: int = 20) -> None:
        """
        Install or clear the per-step L-BFGS refine stepper (ABI minor 60). A threshold of 0 selects 0.5, in the host's force units.
        """

    def is_refining(self) -> bool:
        """True when the last first-order step used the refine stepper."""

    def target_energy_residual(self) -> float | None:
        """
        Signed target-energy residual E(x_end) - E_target from the last free-end step.
        """

    def free_end_cuts(self) -> int:
        """
        Number of adaptive cut-and-remesh refinements performed so far (ABI minor 45).
        """

    def set_fixed_atoms(self, fixed: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> None:
        """Replace the per-atom fixed mask (1 fixed, 0 free)."""

    def saddle_index(self) -> SaddleIndex:
        """
        Certificate of the climbing image at its current position (ABI minor 40).
        """

    def set_saddle_check(self, enabled: bool = True, escape_attempts: int = 0, escape_push: float = 0.1, lowest: int = 4, max_actions: int = 100, dr: float = 0.001, residual_tol: float = 0.01, negative_tol: float = 0.001) -> None:
        """Replace the climbing image's saddle check; enabled=False turns it off."""

    def reset(self) -> None:
        """Drop optimizer history, climbing state, and all cached surface values."""

    def restart(self) -> None:
        """
        Drop optimizer history and climbing state, keeping cached surface values.
        """

class GrowingString:
    def __init__(self, reactant: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], product: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], n_images: int = 11, frontier_tol: float = 0.2, min_steps_per_node: int = 1, max_steps_per_node: int = 0, join_tol: float | None = None, climb: bool = True, ci_after: float = 0.5, ci_after_rel: float = 0.0, method: str = 'fire', memory: int = 20, force_tol: float = 0.001, max_move: float = 0.2, remove_translation: bool = True, min_mode: bool = False, reparametrization: str = 'not-a-knot', cell: object | None = None, fixed: object | None = None) -> None:
        """
        Growing string session (Peters, Heyden, Bell and Chakraborty 2004, ABI minor 44).
        """

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """One growing-string or joined simplified-string step."""

    def run(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], max_steps: int = 1000) -> SessionReport:
        """Loop step until the fragments join and converge, or until max_steps."""

    def n_images(self) -> int:
        """
        Active image count (left_nodes + right_nodes while growing, n_images once joined).
        """

    def nodes_added(self) -> int:
        """
        Nodes the last step grew on the reactant and product fragments (0, 1, or 2).
        """

    def joined(self) -> bool:
        """
        True once the reactant and product fragments have joined into a StringSession.
        """

    def positions(self) -> NDArray[numpy.float64]:
        """Copy of the active positions, shaped (active_images, 3N)."""

    def ids(self) -> NDArray[numpy.uint64]:
        """Stable uint64 image IDs of the active rows."""

    def set_fixed_atoms(self, fixed: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> None:
        """Replace the per-atom fixed mask (1 fixed, 0 free)."""

    def saddle_index(self) -> SaddleIndex:
        """
        Certificate of the climbing image once joined and converged (ABI minor 44).
        """

class SellaSaddle:
    def __init__(self, x: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], masses: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)], order: int = 1, delta: float = 0.1, force_tol: float = 0.001) -> None:
        """
        Order-1 Sella saddle search. masses has one entry per atom, in the caller's mass unit. delta is the initial trust radius per free coordinate.
        """

    def step(self, surface: Callable[..., tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) -> SessionReport:
        """
        One partitioned-RFO trust step. surface(x) is one image shaped (1, 3N).
        """

    def position(self) -> NDArray[numpy.float64]:
        """Copy of the current Cartesian position."""

def raise_status(code: int) -> bool:
    """
    Map an rgsaddle_status_t value. The message is the stored text, or the status name when that text is empty. The stored text is cleared. True on OK. False is not used: convergence is not a status code.
    """
