# MIT License
# Copyright 2026 Rohit Goswami
"""Evaluate ASE images for C ABI request rows ``(ids, x)``.

ASE is imported on first use. ``import rgsaddle`` does not import it.

Calculator calls follow Hjorth Larsen et al., J. Phys.: Condens. Matter
29, 273002 (2017) (doi:10.1088/1361-648X/aa680e): ``get_forces()``
applies constraints, and ``get_forces(apply_constraint=False)`` is the
raw force.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from rgsaddle._core import ClaimSlot, GrowingString as CoreGrowingString, SaddleIndex

Float64 = NDArray[np.float64]
Int64 = NDArray[np.int64]
Uint64 = NDArray[np.uint64]
Uint8 = NDArray[np.uint8]


class _Claim:
    """Hold a claim. ``__exit__`` returns false so exceptions propagate."""

    def __init__(self, slot: ClaimSlot) -> None:
        self._slot = slot

    def __enter__(self) -> ClaimSlot:
        return self._slot.__enter__()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> Literal[False]:
        self._slot.__exit__(exc_type, exc_value, traceback)
        return False


class SurfaceAdapter:
    """Map ``list[Atoms]`` onto ``(ids, x) -> (energies, gradients)``.

    Image ids are the list positions at construction. ``x`` has shape
    ``(rows, 3 * n_atoms)``. Gradients are minus ``get_forces()`` after
    constraints. ``raw_forces`` keeps the unconstrained forces for the
    same rows, in request order.

    The adapter does not own a thread count. ``__call__`` evaluates the
    given rows on the calling thread, in the given order. ``owners`` and
    ``reentrant`` tell a caller which rows share a calculator. A second
    call raises ``RuntimeError`` while the first call still holds the
    claim.

    A shared calculator is refused at construction and again on
    ``__call__``, ``owners``, and ``reentrant``, using ``atoms.calc``
    identity.
    """

    def __init__(
        self,
        images: list[Any],
        *,
        allow_shared_calculator: bool = False,
    ) -> None:
        if len(images) == 0:
            raise ValueError("images must be non-empty")
        self.allow_shared_calculator = allow_shared_calculator
        self._images: dict[int, Any] = {}
        self._slot = ClaimSlot()
        self.raw_forces: Float64 | None = None
        natoms: int | None = None
        seen: dict[int, int] = {}
        for index, atoms in enumerate(images):
            calc = getattr(atoms, "calc", None)
            if calc is None:
                raise ValueError(f"image {index} has no calculator")
            count = len(atoms)
            if natoms is None:
                natoms = count
            elif count != natoms:
                raise ValueError(f"image {index}: expected {natoms} atoms, got {count}")
            owner = id(calc)
            if not allow_shared_calculator and owner in seen:
                raise ValueError(
                    f"images {seen[owner]} and {index} share a calculator; "
                    "pass allow_shared_calculator=True"
                )
            if owner not in seen:
                seen[owner] = index
            self._images[index] = atoms
        self._natoms = 0 if natoms is None else natoms

    def claim(self) -> ClaimSlot:
        """In-use claim. Hold it across the whole ``__call__``."""
        return self._slot

    def owners(self, ids: Int64) -> Uint64:
        """Calculator identity of each requested image, as ``uint64``."""
        ids_arr = self._row_ids(ids)
        out = np.empty(ids_arr.shape[0], dtype=np.uint64)
        for row, image_id in enumerate(ids_arr.tolist()):
            out[row] = np.uint64(id(self._atoms_for(int(image_id)).calc))
        return out

    def reentrant(self, ids: Int64) -> Uint8:
        """1 when the row's calculator sets ``reentrant`` true, else 0."""
        ids_arr = self._row_ids(ids)
        out = np.empty(ids_arr.shape[0], dtype=np.uint8)
        for row, image_id in enumerate(ids_arr.tolist()):
            calc = self._atoms_for(int(image_id)).calc
            out[row] = np.uint8(1 if getattr(calc, "reentrant", False) else 0)
        return out

    def __call__(self, ids: Int64, x: Float64) -> tuple[Float64, Float64]:
        """Evaluate the requested rows on this thread, in the given order.

        ``ids`` is int64, shape ``(rows,)``. ``x`` is float64,
        shape ``(rows, dof)``. A wrong dtype or order raises
        ``TypeError`` naming ``ids`` or ``x``. The result arrays are
        new copies. ``raw_forces`` is written before the claim drops.
        """
        ids_arr = self._row_ids(ids)
        if (
            not isinstance(x, np.ndarray)
            or x.dtype != np.dtype(np.float64)
            or not x.flags.c_contiguous
        ):
            raise TypeError("x: expected float64 C-contiguous")
        x_arr = x
        if x_arr.ndim != 2:
            raise ValueError(f"x: expected rank 2, got rank {x_arr.ndim}")
        n = int(ids_arr.shape[0])
        dof = self._natoms * 3
        if x_arr.shape != (n, dof):
            raise ValueError(f"x: expected shape ({n}, {dof}), got {tuple(int(v) for v in x_arr.shape)}")
        with _Claim(self.claim()):
            energies = np.empty(n, dtype=np.float64)
            gradients = np.empty((n, dof), dtype=np.float64)
            raw = np.empty((n, dof), dtype=np.float64)
            for row, image_id in enumerate(ids_arr.tolist()):
                energy, gradient, raw_force = self._eval_one(int(image_id), x_arr[row], dof)
                energies[row] = energy
                gradients[row] = gradient
                raw[row] = raw_force
            self.raw_forces = raw
            return energies, gradients

    def _check_shared_calculators(self) -> None:
        if self.allow_shared_calculator:
            return
        seen: dict[int, int] = {}
        for image_id, atoms in self._images.items():
            calc = getattr(atoms, "calc", None)
            if calc is None:
                raise ValueError(f"image {image_id} has no calculator")
            owner = id(calc)
            if owner in seen:
                raise ValueError(
                    f"images {seen[owner]} and {image_id} share a calculator; "
                    "pass allow_shared_calculator=True"
                )
            seen[owner] = image_id

    def _row_ids(self, ids: Int64) -> Int64:
        self._check_shared_calculators()
        if (
            not isinstance(ids, np.ndarray)
            or ids.dtype != np.dtype(np.int64)
            or not ids.flags.c_contiguous
        ):
            raise TypeError("ids: expected int64 C-contiguous")
        if ids.ndim != 1:
            raise ValueError(f"ids: expected rank 1, got rank {ids.ndim}")
        return ids

    def _atoms_for(self, image_id: int) -> Any:
        atoms = self._images.get(image_id)
        if atoms is None:
            raise ValueError(f"unknown image id {image_id}")
        return atoms

    def _eval_one(self, image_id: int, row: Float64, dof: int) -> tuple[float, Float64, Float64]:
        atoms = self._atoms_for(image_id)
        coords = np.array(row, dtype=np.float64, copy=True).reshape((self._natoms, 3))
        # The caller asked for this geometry. Force constraints are
        # applied below. Position constraints must not move the row
        # off ``x``.
        atoms.set_positions(coords, apply_constraint=False)
        energy = float(atoms.get_potential_energy())
        raw_force = np.array(atoms.get_forces(apply_constraint=False), dtype=np.float64, copy=True)
        constrained = np.array(atoms.get_forces(), dtype=np.float64, copy=True)
        gradient = -constrained.reshape(dof)
        return energy, gradient, raw_force.reshape(dof)


class Point:
    """Drive one ASE ``Atoms`` object with a point stencil.

    ``run`` constructs an ``ase.optimize.optimize.Optimizer`` whose
    ``step`` is one ``Stencil`` step, so the log, the trajectory, and
    the return value are ASE's. ASE is imported on first use.
    ``preset`` is ``dimer``, ``lanczos``, ``art``, ``eigen-follow``,
    or ``min``. Band presets are not implemented.
    """

    def __init__(self, atoms: Any, *, preset: str = "dimer", **options: Any) -> None:
        self.atoms = atoms
        self.preset = preset
        self.options = options
        self._stencil: Any = None
        self.mode: Float64 | None = None
        self.curvature = float("nan")
        self._observers: list[tuple[Any, int, tuple[Any, ...], dict[str, Any]]] = []

    def __ase_optimizable__(self) -> Any:
        return self.atoms.__ase_optimizable__()

    def attach(self, function: Any, interval: int = 1, *args: Any, **kwargs: Any) -> None:
        self._observers.append((function, interval, args, kwargs))

    def _surface(self, positions: Float64) -> tuple[Float64, Float64]:
        saved = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)
        count = len(self.atoms)
        try:
            self.atoms.set_positions(
                np.ascontiguousarray(positions, dtype=np.float64).reshape(count, 3),
                apply_constraint=False,
            )
            energy = float(self.atoms.get_potential_energy())
            gradient = -np.ascontiguousarray(self.atoms.get_forces(), dtype=np.float64).reshape(1, -1)
        finally:
            self.atoms.set_positions(saved, apply_constraint=False)
        return np.array([energy], dtype=np.float64), gradient

    def _ensure(self) -> Any:
        if self._stencil is None:
            from rgsaddle._core import Stencil

            coordinates = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
            self._stencil = Stencil.point(coordinates, preset=self.preset, **self.options)
        return self._stencil

    def run(
        self,
        fmax: float = 0.05,
        steps: int | None = None,
        *,
        logfile: Any = None,
        trajectory: Any = None,
    ) -> bool:
        from ase.optimize.optimize import DEFAULT_MAX_STEPS, Optimizer

        point = self

        class StencilDriver(Optimizer):
            def initialize(self) -> None:
                return None

            def read(self) -> None:
                return None

            def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
                search = point._ensure()
                report = search.step(point._surface)
                count = len(point.atoms)
                updated = np.asarray(search.positions(), dtype=np.float64).reshape(count, 3)
                point.atoms.set_positions(updated, apply_constraint=False)
                point.curvature = float(report.curvature)
                mode = search.mode()
                point.mode = None if mode is None else np.asarray(mode, dtype=np.float64)

        if steps is None:
            steps = DEFAULT_MAX_STEPS
        driver = StencilDriver(self.atoms, logfile=logfile, trajectory=trajectory)
        host: Any = driver
        for function, interval, args, kwargs in self._observers:
            host.attach(function, interval, *args, **kwargs)
        return bool(host.run(fmax=fmax, steps=steps))


def _image_positions(images: list[Any]) -> Float64:
    rows = [np.asarray(image.get_positions(), dtype=np.float64).reshape(-1) for image in images]
    return np.ascontiguousarray(np.stack(rows), dtype=np.float64)


def _fixed_atoms(images: list[Any]) -> Uint8 | None:
    """One byte per atom, 1 where ``FixAtoms`` holds it, or None.

    Every image must fix the same atoms. Other constraints stay with
    the calculator path: ``get_forces()`` applies them.
    """
    masks: list[Uint8] = []
    for image in images:
        mask = np.zeros(len(image), dtype=np.uint8)
        constraints = getattr(image, "constraints", None) or []
        if constraints:
            from ase.constraints import FixAtoms

            for constraint in constraints:
                if isinstance(constraint, FixAtoms):
                    mask[np.asarray(constraint.index, dtype=np.int64)] = 1
        masks.append(mask)
    if not masks or not masks[0].any():
        if any(mask.any() for mask in masks):
            raise ValueError("FixAtoms must fix the same atoms in every image")
        return None
    for index, mask in enumerate(masks[1:], start=1):
        if not np.array_equal(mask, masks[0]):
            raise ValueError(f"image {index} fixes other atoms than image 0")
    return masks[0]


def saddle_index(
    atoms: Any,
    mode: Any = None,
    *,
    lowest: int = 4,
    max_actions: int = 100,
    dr: float = 1e-3,
    residual_tol: float = 1e-2,
    negative_tol: float = 1e-3,
) -> SaddleIndex:
    """Saddle index of ``atoms`` on its calculator (rgsaddle ABI minor 40).

    The lowest curvatures on the coordinates ``FixAtoms`` leaves free
    come from block Lanczos on forward-difference Hessian actions of
    displacement ``dr``, one force call each, in a space seeded with
    ``mode`` when it is given and with pseudo-random vectors. ``index``
    counts the curvatures under ``-negative_tol``: 1 for a first-order
    saddle. ``atoms`` is left where it was.
    """
    from rgsaddle._core import saddle_index as certify

    if getattr(atoms, "calc", None) is None:
        raise ValueError("atoms has no calculator")
    count = len(atoms)
    saved = np.array(atoms.get_positions(), dtype=np.float64, copy=True)

    def surface(positions: Float64) -> tuple[Float64, Float64]:
        try:
            row = np.asarray(positions, dtype=np.float64).reshape(count, 3)
            atoms.set_positions(row, apply_constraint=False)
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces(apply_constraint=False)
            gradient = -np.ascontiguousarray(forces, dtype=np.float64).reshape(1, -1)
        finally:
            atoms.set_positions(saved, apply_constraint=False)
        return np.array([energy], dtype=np.float64), gradient

    seed = None if mode is None else np.ascontiguousarray(mode, dtype=np.float64).reshape(-1)
    return certify(
        surface,
        saved.reshape(-1),
        mode=seed,
        fixed=_fixed_atoms([atoms]),
        lowest=lowest,
        max_actions=max_actions,
        dr=dr,
        residual_tol=residual_tol,
        negative_tol=negative_tol,
    )


def _springs(k: Any, n_springs: int) -> tuple[list[float], float | None]:
    """One constant, or one per segment. A scalar also comes back as the scalar."""
    if isinstance(k, (str, bytes)):
        raise NotImplementedError("k must be a number or a sequence of springs")
    if np.ndim(k) == 0:
        try:
            scalar = float(k)
        except (TypeError, ValueError) as err:
            raise NotImplementedError("k must be a number or a sequence of springs") from err
        if not np.isfinite(scalar):
            raise ValueError("k must be finite")
        return [scalar] * max(n_springs, 0), scalar
    values = [float(v) for v in np.asarray(k, dtype=np.float64).reshape(-1)]
    if n_springs > 0 and len(values) != n_springs:
        raise ValueError(f"k must be one constant or {n_springs} springs, got {len(values)}")
    if any(not np.isfinite(v) for v in values):
        raise ValueError("k must be finite")
    return values, None


def _band_cell(images: list[Any]) -> Float64 | None:
    """3x3 lattice with non-periodic rows zeroed, or None in free space."""
    pbc = np.asarray(images[0].pbc, dtype=bool)
    if not bool(pbc.any()):
        return None
    cell = np.array(images[0].cell, dtype=np.float64, copy=True)
    for axis, flag in enumerate(pbc.tolist()):
        if not flag:
            cell[axis, :] = 0.0
    return np.ascontiguousarray(cell, dtype=np.float64)


def _check_images(images: list[Any]) -> None:
    if len(images) == 0:
        return
    numbers = np.asarray(images[0].get_atomic_numbers())
    pbc = np.asarray(images[0].pbc)
    cell = np.asarray(images[0].cell, dtype=np.float64)
    for image in images:
        if len(image) != len(images[0]):
            raise ValueError("Images have different numbers of atoms")
        if np.any(np.asarray(image.pbc) != pbc):
            raise ValueError("Images have different boundary conditions")
        if np.any(np.asarray(image.get_atomic_numbers()) != numbers):
            raise ValueError("Images have atoms in different orders")
        for flag, row, row0 in zip(np.asarray(image.pbc), np.asarray(image.cell), cell):
            if flag and np.any(np.abs(np.asarray(row) - row0) > 1e-8):
                raise NotImplementedError(
                    "Variable cell in periodic directions is not implemented yet for NEB"
                )


def _pin_fixed(images: list[Any]) -> None:
    fixed = _fixed_atoms(images)
    if fixed is None:
        return
    start = np.asarray(images[0].get_positions(), dtype=np.float64)
    for image in images[1:-1]:
        row = np.array(image.get_positions(), dtype=np.float64, copy=True)
        row[fixed == 1] = start[fixed == 1]
        image.set_positions(row, apply_constraint=False)


def interpolate(
    images: list[Any],
    mic: bool = False,
    interpolate_cell: bool = False,
    use_scaled_coord: bool = False,
    apply_constraint: bool | None = None,
) -> None:
    """Linear placement of the interior images, the ASE ``mep.interpolate`` call.

    ``mic`` uses the minimum image of the end-to-end displacement.
    ``apply_constraint=None`` raises ``RuntimeError`` when a constraint
    would move an image off that line, which is ASE's behaviour.
    """
    if len(images) < 2:
        return
    if use_scaled_coord:
        pos1 = np.asarray(images[0].get_scaled_positions(wrap=mic), dtype=np.float64)
        pos2 = np.asarray(images[-1].get_scaled_positions(wrap=mic), dtype=np.float64)
    else:
        pos1 = np.asarray(images[0].get_positions(), dtype=np.float64)
        pos2 = np.asarray(images[-1].get_positions(), dtype=np.float64)
    step = pos2 - pos1
    if not use_scaled_coord and mic:
        from ase.geometry import find_mic

        typed_find_mic = cast(Callable[[Any, Any, Any], tuple[Float64, Any]], find_mic)
        step = typed_find_mic(step, images[0].get_cell(), images[0].pbc)[0]
    step = step / (len(images) - 1.0)
    cell1 = None
    cell_step = None
    if interpolate_cell:
        cell1 = np.array(images[0].get_cell(), dtype=np.float64, copy=True)
        cell_step = (np.array(images[-1].get_cell(), dtype=np.float64) - cell1) / (len(images) - 1.0)
    for index in range(1, len(images) - 1):
        if interpolate_cell and cell1 is not None and cell_step is not None:
            images[index].set_cell(cell1 + index * cell_step)
        new_pos = pos1 + index * step
        if use_scaled_coord:
            images[index].set_scaled_positions(new_pos)
            continue
        if apply_constraint is None:
            unconstrained = images[index].copy()
            unconstrained.set_positions(new_pos, apply_constraint=False)
            images[index].set_positions(new_pos, apply_constraint=True)
            if not np.allclose(unconstrained.positions, images[index].positions):
                raise RuntimeError(
                    f"Constraints in image {index}\n"
                    "affect the interpolation results.\n"
                    "Please specify if you want to\n"
                    "apply or ignore the constraints\n"
                    "during the interpolation\n"
                    "with the apply_constraint argument."
                )
        else:
            images[index].set_positions(new_pos, apply_constraint=apply_constraint)


def _place_idpp(
    images: list[Any],
    *,
    mic: bool,
    max_iterations: int = 5000,
    grad_tol: float = 1e-3,
) -> None:
    """Interior images from ``rgsaddle.idpp_path``. Endpoints stay put."""
    from rgsaddle._core import idpp_path

    if mic and not bool(np.asarray(images[0].pbc, dtype=bool).all()):
        if bool(np.asarray(images[0].pbc, dtype=bool).any()):
            raise NotImplementedError("idpp mic on a partially periodic cell")
    cell = None
    if mic and bool(np.asarray(images[0].pbc, dtype=bool).all()):
        cell = np.ascontiguousarray(np.array(images[0].cell, dtype=np.float64), dtype=np.float64)
    start = np.ascontiguousarray(images[0].get_positions(), dtype=np.float64).reshape(-1)
    end = np.ascontiguousarray(images[-1].get_positions(), dtype=np.float64).reshape(-1)
    path = np.asarray(
        idpp_path(
            start,
            end,
            len(images),
            cell=cell,
            max_iterations=max_iterations,
            grad_tol=grad_tol,
        ),
        dtype=np.float64,
    )
    natoms = len(images[0])
    for index, image in enumerate(images[1:-1], start=1):
        image.set_positions(path[index].reshape(natoms, 3), apply_constraint=False)
    _pin_fixed(images)


def idpp_interpolate(
    images: Any,
    traj: str | None = "idpp.traj",
    log: str | None = "idpp.log",
    fmax: float = 0.1,
    optimizer: Any = None,
    mic: bool = False,
    steps: int = 100,
) -> None:
    """IDPP placement. ``images`` is a list or an object with ``.images``.

    The path is ``rgsaddle.idpp_path`` (L-BFGS on the pair potential).
    ``steps`` is that minimizer's iteration cap and ``fmax`` its gradient
    tolerance. ``traj`` and ``log`` raise ``NotImplementedError`` unless
    they are the ASE defaults, because this path writes no trajectory.
    ``optimizer`` may be ``MDMin`` or ``BFGS``. Either one runs this
    L-BFGS path. Any other optimizer raises ``NotImplementedError``.
    """
    if traj not in (None, "idpp.traj") or log not in (None, "idpp.log"):
        raise NotImplementedError("idpp_interpolate traj and log: the Rust path writes no trajectory")
    if optimizer is not None:
        from ase.optimize import BFGS, MDMin

        # The observable is the minimized IDPP path. MDMin is ASE's default
        # and BFGS is the other optimizer its tests pass. Both run
        # rgsaddle.idpp_path. Any other optimizer is named and refused.
        if optimizer not in (MDMin, BFGS):
            name = getattr(optimizer, "__name__", repr(optimizer))
            raise NotImplementedError(
                f"idpp_interpolate optimizer '{name}': the path is rgsaddle.idpp_path (L-BFGS)"
            )
    band = images.images if hasattr(images, "images") else images
    _place_idpp(band, mic=mic, max_iterations=max(int(steps), 1), grad_tol=float(fmax))


class _NebView:
    """ASE optimizable view of interior images."""

    def __init__(self, neb: NEB) -> None:
        self._neb = neb

    def get_x(self) -> Float64:
        return np.ascontiguousarray(self._neb.get_positions(), dtype=np.float64).reshape(-1)

    def set_x(self, x: Float64) -> None:
        self._neb.set_positions(np.asarray(x, dtype=np.float64))

    def get_gradient(self) -> Float64:
        return -np.ascontiguousarray(self._neb.get_forces(), dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        return float(self._neb.get_potential_energy())

    def ndofs(self) -> int:
        return int(self.get_x().shape[0])

    def gradient_norm(self, gradient: Float64) -> float:
        forces = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        return float(np.max(np.linalg.norm(forces, axis=1)))

    def converged(self, gradient: Float64, fmax: float) -> bool:
        return self.gradient_norm(gradient) < fmax

    def iterimages(self) -> Any:
        return self._neb.iterimages()


class NEB:
    """Nudged elastic band over a list of ASE ``Atoms``.

    The constructor follows ``ase.mep.NEB``: ``k`` is one spring or one
    per segment, ``climb`` turns the climbing image on for every force
    call, and ``method`` is ``improvedtangent`` (default), ``aseneb``,
    ``eb``, ``dneb``, ``spline``, or ``string``. ``precon`` raises
    ``NotImplementedError``, and so does any of ``dynamic_relaxation``,
    ``fmax``, or ``scale_fmax``. Those three belong to :class:`DyNEB`.

    ``spline`` is ASE's cubic-spline NEB. The tangent is the unit
    ``dx/ds`` of the not-a-knot cubic through the images, and the spring
    is the curvature term, with ``s`` the normalized cumulative chord
    length. ``string`` uses that tangent and no spring.
    ``set_positions`` then places the interior images at equal steps of
    the same parameter.

    ``get_positions`` and ``get_forces`` are the interior images only,
    shape ``((n_images - 2) * n_atoms, 3)``. ``get_potential_energy`` is
    the highest interior energy. ``ase.optimize`` drives the band through
    ``__ase_optimizable__``.

    ``climb=True`` does not climb an interior image when a higher
    endpoint slopes down into the band. ASE's ``NEB`` climbs the highest
    image on every call, including that slope.

    ``remove_rotation_and_translation=True`` aligns each free, non-periodic
    image to the previous one with ASE's rigid fit, and otherwise removes
    the mean Cartesian force. A fixed atom is not rotated.

    ``parallel=True`` with a single rank evaluates images on the calling
    thread. An MPI world (``world.size > 1``) raises ``NotImplementedError``.
    Shared calculators with ``parallel=True`` raise ``RuntimeError``.

    ``interpolate(method='linear')`` is the straight line, with ``mic``.
    ``method='idpp'`` then runs ``idpp_path``. ``FixAtoms`` stays on the
    first image. Lengths and energies are the calculator's.
    """

    def __init__(
        self,
        images: list[Any],
        k: Any = 0.1,
        climb: bool = False,
        parallel: bool = False,
        remove_rotation_and_translation: bool = False,
        world: Any = None,
        method: str | None = None,
        allow_shared_calculator: bool = False,
        precon: Any = None,
        **kwargs: Any,
    ) -> None:
        for keyword in ("dynamic_relaxation", "fmax", "scale_fmax"):
            if keyword in kwargs:
                raise NotImplementedError(f"{keyword}: dynamic NEB (DyNEB) is not implemented")
        if kwargs:
            name = next(iter(kwargs))
            raise NotImplementedError(f"unsupported NEB argument {name}")
        if precon is not None:
            raise NotImplementedError("precon: preconditioned NEB is not implemented")
        _check_images(images)
        self.images = images
        self.climb = bool(climb)
        self.parallel = bool(parallel)
        self.allow_shared_calculator = bool(allow_shared_calculator)
        self.remove_rotation_and_translation = bool(remove_rotation_and_translation)
        if world is None:
            import ase.parallel

            world = ase.parallel.world
        self.world = world
        if self.parallel and self.allow_shared_calculator:
            raise RuntimeError("Cannot use shared calculators in parallel in NEB.")
        if self.parallel and int(getattr(self.world, "size", 1)) > 1:
            raise NotImplementedError("parallel NEB over an MPI world (world.size > 1)")
        n_springs = max(len(images) - 1, 0)
        self.k, self._k_scalar = _springs(k, n_springs)
        if method is None:
            warnings.warn(
                "The default method has changed from 'aseneb' to 'improvedtangent'. "
                "Set method='improvedtangent' to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            method = "improvedtangent"
        forces = {
            "improvedtangent": ("neb", "improved"),
            "aseneb": ("neb", "improved"),
            "eb": ("eb", "improved"),
            "dneb": ("dneb", "improved"),
            "spline": ("neb", "spline"),
            "string": ("neb", "spline"),
        }
        if method not in forces:
            raise NotImplementedError(method)
        self.method = method
        self._projection, self._tangent = forces[method]
        self._adapter: SurfaceAdapter | None = None
        self.energies: Float64 | None = None
        self.emax = float("nan")
        self.imax: int | None = None
        self.ci: int | None = None
        self._real_forces: Float64 | None = None

    @property
    def natoms(self) -> int:
        return len(self.images[0])

    @property
    def nimages(self) -> int:
        return len(self.images)

    def __len__(self) -> int:
        if len(self.images) < 2:
            return 0
        return (len(self.images) - 2) * len(self.images[0])

    def __ase_optimizable__(self) -> _NebView:
        return _NebView(self)

    @staticmethod
    def freeze_results_on_image(atoms: Any, **results: Any) -> None:
        from ase.calculators.singlepoint import SinglePointCalculator

        make_calculator = cast(Callable[..., Any], SinglePointCalculator)
        atoms.calc = make_calculator(atoms, **results)

    def _refuse_shared(self) -> None:
        if self.allow_shared_calculator:
            return
        seen: dict[int, int] = {}
        for index, image in enumerate(self.images):
            calc = getattr(image, "calc", None)
            if calc is None:
                continue
            owner = id(calc)
            if owner in seen:
                raise ValueError(
                    "One or more NEB images share the same calculator. "
                    "Each image must have its own calculator."
                )
            seen[owner] = index

    def _adapter_for(self) -> SurfaceAdapter:
        self._refuse_shared()
        if self._adapter is None:
            self._adapter = SurfaceAdapter(
                self.images, allow_shared_calculator=self.allow_shared_calculator
            )
        return self._adapter

    def _can_align(self) -> bool:
        """Kabsch is safe only for a free, non-periodic band."""
        if not self.remove_rotation_and_translation or len(self.images) < 2:
            return False
        if _fixed_atoms(self.images) is not None:
            return False
        return not bool(np.asarray(self.images[0].pbc, dtype=bool).any())

    def _align_endpoints(self) -> None:
        """Move the product onto the reactant, as ASE's interpolate does."""
        if not self._can_align():
            return
        from ase.build import minimize_rotation_and_translation

        align = cast(Callable[[Any, Any], None], minimize_rotation_and_translation)
        align(self.images[0], self.images[-1])

    def _align(self) -> bool:
        """Align each image onto the previous one. True when that ran."""
        if not self._can_align():
            return False
        from ase.build import minimize_rotation_and_translation

        align = cast(Callable[[Any, Any], None], minimize_rotation_and_translation)
        for index in range(1, len(self.images)):
            align(self.images[index - 1], self.images[index])
        return True

    def interpolate(self, method: str = "linear", mic: bool = False, apply_constraint: bool | None = None) -> None:
        """Place interior images. ``method`` is ``linear`` or ``idpp``."""
        if method not in ("linear", "idpp"):
            raise NotImplementedError(f"interpolate method '{method}'")
        if len(self.images) < 3:
            raise ValueError("interpolation needs two endpoints and one interior image")
        if self.remove_rotation_and_translation:
            self._align_endpoints()
        interpolate(self.images, mic=mic, apply_constraint=apply_constraint)
        if method == "idpp":
            _place_idpp(self.images, mic=mic)

    def idpp_interpolate(
        self,
        traj: str | None = "idpp.traj",
        log: str | None = "idpp.log",
        fmax: float = 0.1,
        optimizer: Any = None,
        mic: bool = False,
        steps: int = 100,
    ) -> None:
        warnings.warn(
            "Please use NEB.interpolate(method='idpp') or idpp_interpolate",
            FutureWarning,
            stacklevel=2,
        )
        idpp_interpolate(self, traj=traj, log=log, fmax=fmax, optimizer=optimizer, mic=mic, steps=steps)

    def get_positions(self) -> Float64:
        blocks = [np.asarray(image.get_positions(), dtype=np.float64) for image in self.images[1:-1]]
        if not blocks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(blocks, axis=0)

    def set_positions(self, positions: Any, adjust_positions: bool = True) -> None:
        natoms = len(self.images[0])
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        expect = (len(self.images) - 2) * natoms
        count = len(array)
        if count != expect:
            raise ValueError(f"positions: expected {expect} atoms, got {count}")
        if adjust_positions and self.method == "string" and expect > 0:
            array = self._redistribute_string(array)
        snap = self.method == "string"
        for index, image in enumerate(self.images[1:-1]):
            image.set_positions(
                array[index * natoms : (index + 1) * natoms],
                apply_constraint=snap,
            )

    def _redistribute_string(self, interior: Float64) -> Float64:
        """ASE string step: equal chord-length parameter on the cubic spline."""
        from rgsaddle._core import spline_resample

        natoms = len(self.images[0])
        proposed = np.array(_image_positions(self.images), dtype=np.float64, copy=True)
        interior_rows = np.asarray(interior, dtype=np.float64).reshape(
            (len(self.images) - 2, natoms * 3)
        )
        proposed[1:-1] = interior_rows
        updated = spline_resample(
            np.ascontiguousarray(proposed, dtype=np.float64),
            cell=_band_cell(self.images),
        )
        rows = np.asarray(updated, dtype=np.float64)
        return np.ascontiguousarray(rows[1:-1].reshape(-1, 3), dtype=np.float64)

    def set_calculators(self, calculators: Any) -> None:
        """Attach one calculator, or one per image, or one per interior image."""
        if not isinstance(calculators, list):
            if self.allow_shared_calculator:
                calculators = [calculators] * self.nimages
            else:
                raise RuntimeError(
                    "Cannot set shared calculator to NEB with allow_shared_calculator=False"
                )
        n = len(calculators)
        if n == self.nimages:
            for index in range(self.nimages):
                self.images[index].calc = calculators[index]
        elif n == self.nimages - 2:
            for index in range(1, self.nimages - 1):
                self.images[index].calc = calculators[index - 1]
        else:
            raise RuntimeError(f"len(calculators)={n} does not fit to len(images)={self.nimages}")
        self._adapter = None

    def get_potential_energy(self) -> float:
        if self.energies is None:
            self.get_forces()
        return float(self.emax)

    def get_forces(self) -> Float64:
        from rgsaddle._core import project_band

        aligned = self._align()
        positions = _image_positions(self.images)
        ids = np.arange(positions.shape[0], dtype=np.int64)
        energies, gradients = self._adapter_for()(ids, positions)
        springs, scalar = self.k, self._k_scalar
        call: dict[str, Any] = {
            "climb": self.climb,
            "projection": self._projection,
            "tangent": self._tangent,
            "remove_translation": self.remove_rotation_and_translation and not aligned,
            "fixed": _fixed_atoms(self.images),
            "cell": _band_cell(self.images),
        }
        # The string method keeps the caller's k on the object and applies none of it.
        if self.method == "string":
            call["k"] = 0.0
        elif scalar is None:
            call["k"] = springs[0] if springs else 0.1
            call["spring_ks"] = np.ascontiguousarray(springs, dtype=np.float64)
        else:
            call["k"] = scalar
        flat, ci = project_band(
            positions,
            np.ascontiguousarray(energies, dtype=np.float64),
            np.ascontiguousarray(gradients, dtype=np.float64),
            **call,
        )
        count = len(self.images)
        natoms = len(self.images[0])
        forces = np.asarray(flat, dtype=np.float64).reshape(count, natoms, 3)
        self.energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        interior = self.energies[1:-1]
        if interior.size:
            self.imax = 1 + int(np.argmax(interior))
            self.emax = float(interior[self.imax - 1])
        else:
            self.imax = None
            self.emax = float("nan")
        self.ci = None if int(ci) < 0 else int(ci)
        raw = self._adapter_for().raw_forces
        if raw is not None:
            self._real_forces = np.asarray(raw, dtype=np.float64).reshape(count, natoms, 3)
        return np.ascontiguousarray(forces[1:-1].reshape(-1, 3), dtype=np.float64)

    def get_residual(self) -> float:
        """Largest per-atom force on the interior images."""
        forces = self.get_forces().reshape(-1, 3)
        if forces.size == 0:
            return 0.0
        return float(np.max(np.linalg.norm(forces, axis=1)))

    def iterimages(self) -> Any:
        for index, atoms in enumerate(self.images):
            if index == 0 or index == len(self.images) - 1 or self.energies is None or self._real_forces is None:
                yield atoms
            else:
                copied = atoms.copy()
                self.freeze_results_on_image(
                    copied,
                    energy=float(self.energies[index]),
                    forces=np.array(self._real_forces[index], dtype=np.float64, copy=True),
                )
                yield copied

    def saddle_index(self, **options: Any) -> SaddleIndex:
        """Saddle index of the climbing image, seeded with its tangent.

        Call it once the band has converged with ``climb=True``.
        ``options`` are the keywords of :func:`saddle_index`. A band
        whose start shares a symmetry with the surface can converge on a
        saddle of higher index; ``index`` above 1 says so.
        """
        if self.ci is None:
            self.get_forces()
        if self.ci is None:
            raise ValueError("no interior image climbs")
        positions = _image_positions(self.images)
        tangent = positions[self.ci + 1] - positions[self.ci - 1]
        return saddle_index(self.images[self.ci], tangent, **options)


class DyNEB(NEB):
    """ASE 3.29 dynamic NEB.

    ``dynamic_relaxation`` leaves an interior image where it is once the
    largest per-atom norm of its projected force is strictly below
    ``fmax``. ``scale_fmax`` scales that test by the Euclidean distance
    from the highest-energy image,
    ``fmax * (1 + distance * scale_fmax)``, with no minimum image. The
    highest-energy image keeps its force. A skipped image keeps the
    energy and the force from the last evaluation and still enters the
    spring and the tangent; a neighbour that pulls the projected force
    back over the threshold is evaluated again. ``fmax`` is the
    optimizer's ``fmax``.

    ``dynamic_relaxation=False`` is the ordinary band.
    ``scale_fmax`` with that flag raises ``ValueError``. ``parallel=True``
    raises ``ValueError`` while dynamic relaxation is on. The mask is
    ``rgsaddle._core.dyneb_mask``.
    """

    def __init__(
        self,
        images: list[Any],
        k: Any = 0.1,
        fmax: float = 0.05,
        climb: bool = False,
        parallel: bool = False,
        remove_rotation_and_translation: bool = False,
        world: Any = None,
        dynamic_relaxation: bool = True,
        scale_fmax: float = 0.0,
        method: str | None = None,
        allow_shared_calculator: bool = False,
        precon: Any = None,
    ) -> None:
        super().__init__(
            images,
            k=k,
            climb=climb,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world,
            method=method,
            allow_shared_calculator=allow_shared_calculator,
            precon=precon,
        )
        self.fmax = float(fmax)
        self.dynamic_relaxation = bool(dynamic_relaxation)
        self.scale_fmax = float(scale_fmax)
        if not self.dynamic_relaxation and self.scale_fmax:
            raise ValueError(
                "Scaled convergence criteria only implemented in series with dynamic relaxation."
            )

    def set_positions(self, positions: Any, adjust_positions: bool = True) -> None:
        """Move interior images whose force is still at or above ``fmax``.

        The band force is read once, on the coordinates this call
        started with, and every free image is moved after that. A force
        read inside the image loop would evaluate a band whose earlier
        images had already moved.
        """
        if not self.dynamic_relaxation:
            super().set_positions(positions, adjust_positions)
            return
        if self.parallel:
            raise ValueError(
                "Dynamic relaxation does not work efficiently when parallelizing over images. "
                "Try AutoNEB routine for freezing images in parallel."
            )
        natoms = len(self.images[0])
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        expect = (len(self.images) - 2) * natoms
        if len(array) != expect:
            raise ValueError(f"positions: expected {expect} atoms, got {len(array)}")
        if len(self.images) < 3:
            return
        forces = np.asarray(self.get_forces(), dtype=np.float64).reshape((-1, 3))
        n1 = 0
        for image in self.images[1:-1]:
            block = forces[n1 : n1 + natoms]
            image_fmax = float(np.sqrt((block**2).sum(axis=1)).max()) if block.size else 0.0
            if image_fmax >= self.fmax:
                image.set_positions(array[n1 : n1 + natoms], apply_constraint=False)
            n1 += natoms

    def get_forces(self) -> Float64:
        interior = super().get_forces()
        if not self.dynamic_relaxation:
            return interior
        from rgsaddle._core import dyneb_mask

        count = len(self.images)
        natoms = len(self.images[0])
        if count < 3 or self.energies is None:
            return interior
        positions = np.ascontiguousarray(_image_positions(self.images).reshape(-1), dtype=np.float64)
        energies = np.ascontiguousarray(self.energies, dtype=np.float64)
        full = np.zeros(count * natoms * 3, dtype=np.float64)
        full[natoms * 3 : (count - 1) * natoms * 3] = np.asarray(interior, dtype=np.float64).reshape(-1)
        masked, _frozen = dyneb_mask(
            positions,
            energies,
            full,
            self.fmax,
            self.scale_fmax,
        )
        masked = np.asarray(masked, dtype=np.float64).reshape(count, natoms, 3)
        return np.ascontiguousarray(masked[1:-1].reshape(-1, 3), dtype=np.float64)


class SingleCalculatorNEB(NEB):
    """``NEB(allow_shared_calculator=True)``.

    ASE deprecated this class in 3.23. Constructing it warns and sets
    the shared-calculator flag.
    """

    def __init__(self, images: list[Any], *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "Please use NEB(allow_shared_calculator=True) instead of SingleCalculatorNEB",
            FutureWarning,
            stacklevel=2,
        )
        kwargs["allow_shared_calculator"] = True
        super().__init__(images, *args, **kwargs)


def _effective_force(forces: Float64, mode: Float64, curvature: float) -> Float64:
    """ASE force after the minimum-mode translation of the gradient."""
    flat = np.asarray(forces, dtype=np.float64).reshape(-1)
    tau = np.asarray(mode, dtype=np.float64).reshape(-1)
    norm = float(np.dot(tau, tau) ** 0.5)
    if norm == 0.0:
        return flat.reshape(forces.shape)
    tau = tau / norm
    parallel = float(np.dot(flat, tau))
    if curvature < 0.0:
        eff = flat - 2.0 * parallel * tau
    else:
        eff = -parallel * tau
    return eff.reshape(forces.shape)


class DimerControl:
    """Parameters of ASE's dimer search.

    Unknown names raise ``ValueError``. ``logfile=None`` keeps the run
    quiet. An MPI communicator with ``size > 1`` raises
    ``NotImplementedError``.
    """

    _DEFAULTS: dict[str, Any] = {
        "eigenmode_method": "dimer",
        "f_rot_min": 0.1,
        "f_rot_max": 1.00,
        "max_num_rot": 1,
        "trial_angle": float(np.pi / 4.0),
        "trial_trans_step": 0.001,
        "maximum_translation": 0.1,
        "cg_translation": True,
        "use_central_forces": True,
        "dimer_separation": 0.0001,
        "initial_eigenmode_method": "gauss",
        "extrapolate_forces": False,
        "displacement_method": "gauss",
        "gauss_std": 0.1,
        "order": 1,
        "mask": None,
        "displacement_center": None,
        "displacement_radius": None,
        "number_of_displacement_atoms": None,
    }

    def __init__(
        self,
        logfile: Any = "-",
        eigenmode_logfile: Any = None,
        comm: Any = None,
        **kwargs: Any,
    ) -> None:
        if comm is not None and int(getattr(comm, "size", 1)) > 1:
            raise NotImplementedError("DimerControl comm with world.size > 1")
        self.parameters = dict(self._DEFAULTS)
        for key, value in kwargs.items():
            self.set_parameter(key, value, log=False)
        self.logfile = None if logfile in (None, "-") else logfile
        self.eigenmode_logfile = None if eigenmode_logfile in (None, "-") else eigenmode_logfile
        self.counters = {"forcecalls": 0, "rotcount": 0, "optcount": 0}

    def __enter__(self) -> DimerControl:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> Literal[False]:
        del exc_type, exc_value, traceback
        return False

    def set_parameter(self, parameter: str, value: Any, log: bool = True) -> None:
        del log
        if parameter not in self.parameters:
            raise ValueError(
                f"Invalid parameter >>{parameter}<< with value >>{value!s}<< in DimerControl"
            )
        self.parameters[parameter] = value

    def get_parameter(self, parameter: str) -> Any:
        if parameter not in self.parameters:
            raise ValueError(f"Invalid parameter >>{parameter}<<")
        return self.parameters[parameter]

    def get_logfile(self) -> Any:
        return self.logfile

    def get_eigenmode_logfile(self) -> Any:
        return self.eigenmode_logfile

    def get_counter(self, counter: str) -> int:
        return int(self.counters[counter])

    def increment_counter(self, counter: str) -> None:
        self.counters[counter] = int(self.counters[counter]) + 1


class _MinModeView:
    def __init__(self, search: MinModeAtoms) -> None:
        self._search = search

    def get_x(self) -> Float64:
        return np.ascontiguousarray(self._search.atoms.get_positions(), dtype=np.float64).reshape(-1)

    def set_x(self, x: Float64) -> None:
        count = len(self._search.atoms)
        self._search.atoms.set_positions(
            np.asarray(x, dtype=np.float64).reshape(count, 3), apply_constraint=False
        )
        self._search._invalidate()

    def get_gradient(self) -> Float64:
        return -np.ascontiguousarray(self._search.get_forces(), dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        return float(self._search.atoms.get_potential_energy())

    def ndofs(self) -> int:
        return 3 * len(self._search.atoms)

    def gradient_norm(self, gradient: Float64) -> float:
        forces = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        return float(np.max(np.linalg.norm(forces, axis=1)))

    def converged(self, gradient: Float64, fmax: float) -> bool:
        if not self.gradient_norm(gradient) < fmax:
            return False
        curvature = self._search.curvature
        return bool(np.isfinite(curvature) and curvature < 0.0)

    def iterimages(self) -> Any:
        yield self._search.atoms


class MinModeAtoms:
    """One ASE ``Atoms`` object driven by a dimer or Lanczos mode.

    ``get_forces`` estimates the lowest mode and returns the effective
    force. Positive curvature drops the force perpendicular to the mode.
    Negative curvature reflects the force along the mode.

    The ASE call is ``MinModeAtoms(atoms, control)`` with a
    ``DimerControl``. ``displace``, ``get_eigenmode``, ``get_curvature``
    and ``get_barrier_energy`` are that interface. ``mode=``, ``kind``,
    ``dr`` and ``max_rotations`` remain for a search that does not build
    a control object. ``order`` other than 1 and a non-dimer
    ``eigenmode_method`` raise ``NotImplementedError``.
    """

    def __init__(
        self,
        atoms: Any,
        control: Any = None,
        mode: Any = None,
        *,
        eigenmodes: Any = None,
        kind: str = "dimer",
        dr: float = 1e-3,
        max_rotations: int = 10,
        random_seed: int | None = None,
        comm: Any = None,
        **kwargs: Any,
    ) -> None:
        del random_seed
        if comm is not None and int(getattr(comm, "size", 1)) > 1:
            raise NotImplementedError("MinModeAtoms comm with world.size > 1")
        if control is not None and mode is None and not hasattr(control, "get_parameter"):
            mode = control
            control = None
        if getattr(atoms, "calc", None) is None:
            raise ValueError("atoms has no calculator")
        if control is None and kwargs:
            control = DimerControl(**kwargs)
        elif control is not None:
            for key, value in kwargs.items():
                control.set_parameter(key, value)
        self.atoms = atoms
        self.control = control
        self.kind = kind
        self.dr = float(dr)
        self.max_rotations = int(max_rotations)
        if control is not None:
            method_name = str(control.get_parameter("eigenmode_method")).lower()
            if method_name != "dimer":
                raise NotImplementedError(f"eigenmode_method '{method_name}'")
            if int(control.get_parameter("order")) != 1:
                raise NotImplementedError("order other than 1")
            self.kind = "dimer"
            self.dr = float(control.get_parameter("dimer_separation"))
            # ASE's max_num_rot counts Fourier rotations. This dimer takes
            # up to that many rotations, and at least a few, so one ASE
            # rotation does not leave the mode at its initial guess.
            self.max_rotations = max(4, int(control.get_parameter("max_num_rot")))
        self._origin = np.array(atoms.get_positions(), dtype=np.float64, copy=True)
        self._origin_energy = float(atoms.get_potential_energy())
        self._have_mode = False
        self._mode0 = np.zeros(len(atoms) * 3, dtype=np.float64)
        if eigenmodes is not None and mode is None:
            mode = eigenmodes[0]
        if mode is not None:
            self._mode0 = self._unit_mode(np.asarray(mode, dtype=np.float64).reshape(-1))
            self._have_mode = True
        self._session: Any = None
        self._session_pos: Float64 | None = None
        self.mode: Float64 | None = None
        self._eigenmode: Float64 | None = None
        self.curvature = float("nan")
        self.curvatures: list[float] | None = None
        self._cache_pos: Float64 | None = None
        self._cache_forces: Float64 | None = None
        self._cache_raw: Float64 | None = None

    def __len__(self) -> int:
        return len(self.atoms)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        if "original" in attr.split("_"):
            renamed = attr.replace("_original_", "_")
            if renamed == "get_positions":
                return lambda: self._origin.copy()
            if renamed == "get_potential_energy":
                return lambda: self._origin_energy
            return getattr(self._atoms_origin(), renamed)
        return getattr(self.atoms, attr)

    def _atoms_origin(self) -> Any:
        copied = self.atoms.copy()
        copied.set_positions(self._origin, apply_constraint=False)
        return copied

    def get_control(self) -> Any:
        return self.control

    def _invalidate(self) -> None:
        self._cache_pos = None

    def _unit_mode(self, flat: Float64) -> Float64:
        mode = np.asarray(flat, dtype=np.float64).reshape(-1).copy()
        fixed = _fixed_atoms([self.atoms])
        if fixed is not None:
            mode.reshape(-1, 3)[fixed == 1] = 0.0
        norm = float(np.dot(mode, mode) ** 0.5)
        if norm == 0.0:
            mode = np.zeros_like(mode)
            mode[0] = 1.0
            return np.array(mode, dtype=np.float64, copy=True)
        return np.array(mode / norm, dtype=np.float64, copy=True)

    def _initialize_eigenmode(self) -> None:
        if self._have_mode:
            return
        count = len(self.atoms)
        if self.control is None:
            guess = np.zeros(count * 3, dtype=np.float64)
            guess[0] = 1.0
        else:
            method = str(self.control.get_parameter("initial_eigenmode_method")).lower()
            pos = np.asarray(self.atoms.get_positions(), dtype=np.float64)
            if method == "displacement" and np.any(pos - self._origin):
                guess = (pos - self._origin).reshape(-1)
            elif method == "gauss":
                raise NotImplementedError("initial_eigenmode_method 'gauss'")
            else:
                raise NotImplementedError(
                    f"initial_eigenmode_method '{method}' needs a displacement from the original positions"
                )
        self._mode0 = self._unit_mode(np.asarray(guess, dtype=np.float64))
        self._have_mode = True
        self._session = None

    def __ase_optimizable__(self) -> _MinModeView:
        return _MinModeView(self)

    def _surface(self, positions: Float64) -> tuple[Float64, Float64]:
        saved = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)
        count = len(self.atoms)
        try:
            row = np.ascontiguousarray(positions, dtype=np.float64).reshape(count, 3)
            self.atoms.set_positions(row, apply_constraint=False)
            energy = float(self.atoms.get_potential_energy())
            gradient = -np.ascontiguousarray(self.atoms.get_forces(), dtype=np.float64).reshape(1, -1)
        finally:
            self.atoms.set_positions(saved, apply_constraint=False)
        return np.array([energy], dtype=np.float64), gradient

    def _ensure(self) -> Any:
        self._initialize_eigenmode()
        from rgsaddle._core import MinMode

        coordinates = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
        if self._session is None:
            self._session = MinMode(
                coordinates,
                np.ascontiguousarray(self._mode0, dtype=np.float64),
                kind=self.kind,
                dr=self.dr,
                max_rotations=self.max_rotations,
            )
            self._session_pos = coordinates.copy()
            return self._session
        if self._session_pos is None or not np.array_equal(coordinates, self._session_pos):
            self._session.set_position(coordinates)
            self._session_pos = coordinates.copy()
        return self._session

    def _remember(self, raw: Float64, effective: Float64, mode: Float64, curvature: float) -> None:
        self._cache_raw = raw
        self._cache_forces = effective
        self._cache_pos = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)
        self.mode = mode
        self._eigenmode = mode.reshape(len(self.atoms), 3)
        self.curvature = curvature
        self.curvatures = [curvature]

    def _estimate(self) -> None:
        search = self._ensure()
        report = search.estimate(self._surface)
        curvature = float(report.curvature)
        mode = self._unit_mode(np.asarray(search.mode(), dtype=np.float64))
        # The curvature and the effective force are even in the mode.
        # Keep the sign of the guess so a displacement stays a displacement.
        if float(np.dot(mode, self._mode0)) < 0.0:
            mode = -mode
        raw = np.array(self.atoms.get_forces(), dtype=np.float64, copy=True)
        self._remember(raw, _effective_force(raw, mode, curvature), mode, curvature)
        if self.control is not None:
            self.control.increment_counter("forcecalls")
            self.control.increment_counter("optcount")

    def get_forces(self, real: bool = False, pos: Any = None, **kwargs: Any) -> Float64:
        del kwargs
        if pos is not None:
            raw = self._forces_at(pos)
            if real:
                return raw
            if self._eigenmode is None:
                self._estimate()
            mode = self._eigenmode
            assert mode is not None
            return _effective_force(raw, mode.reshape(-1), self.curvature)
        current = np.asarray(self.atoms.get_positions(), dtype=np.float64)
        if (
            self._cache_pos is not None
            and self._cache_forces is not None
            and self._cache_raw is not None
            and np.array_equal(current, self._cache_pos)
        ):
            return self._cache_raw if real else self._cache_forces
        self._estimate()
        assert self._cache_raw is not None and self._cache_forces is not None
        return self._cache_raw if real else self._cache_forces

    def _forces_at(self, pos: Any) -> Float64:
        saved = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)
        try:
            self.atoms.set_positions(
                np.asarray(pos, dtype=np.float64).reshape(len(self.atoms), 3),
                apply_constraint=False,
            )
            return np.array(self.atoms.get_forces(), dtype=np.float64, copy=True)
        finally:
            self.atoms.set_positions(saved, apply_constraint=False)

    def get_projected_forces(self, pos: Any = None) -> Float64:
        return self.get_forces(real=False, pos=pos)

    def get_potential_energy(self) -> float:
        return float(self.atoms.get_potential_energy())

    def get_barrier_energy(self) -> float:
        return float(self.get_potential_energy() - self._origin_energy)

    def get_curvature(self, order: Any = "max") -> float:
        if self.curvatures is None:
            self.get_forces()
        assert self.curvatures is not None
        if order == "max":
            return float(max(self.curvatures))
        if int(order) != 1:
            raise NotImplementedError("curvature order other than 1")
        return float(self.curvatures[0])

    def get_eigenmode(self, order: int = 1) -> Float64:
        if order != 1:
            raise NotImplementedError("eigenmode order other than 1")
        if self._eigenmode is None:
            self.get_forces()
        assert self._eigenmode is not None
        return self._eigenmode

    def displace(
        self,
        displacement_vector: Any = None,
        mask: Any = None,
        method: str | None = None,
        displacement_center: Any = None,
        radius: float | None = None,
        number_of_atoms: int | None = None,
        gauss_std: float | None = None,
        mic: bool = True,
        log: bool = True,
    ) -> None:
        """Move atoms. ``method='vector'`` applies ``displacement_vector``."""
        del log
        refused = [
            name
            for name, value in (
                ("displacement_center", displacement_center),
                ("radius", radius),
                ("number_of_atoms", number_of_atoms),
                ("gauss_std", gauss_std),
            )
            if value is not None
        ]
        if refused:
            raise NotImplementedError("displace " + ", ".join(refused))
        if mic is not True:
            raise NotImplementedError("displace mic")
        if self.control is not None:
            if mask is None:
                mask = self.control.get_parameter("mask")
            if method is None:
                method = self.control.get_parameter("displacement_method")
        if method is None:
            method = "vector" if displacement_vector is not None else "gauss"
        method_name = str(method).lower()
        if displacement_vector is not None and method_name != "vector":
            raise ValueError(
                f"displacement_vector was supplied but a different method ('{method}') was chosen."
            )
        if displacement_vector is None and method_name == "vector":
            raise ValueError(f"A displacement_vector must be supplied when using method = '{method}'.")
        if method_name != "vector":
            raise NotImplementedError(f"displacement_method '{method}'")
        vector = np.array(displacement_vector, dtype=np.float64, copy=True)
        if mask is not None:
            selected = np.asarray(mask, dtype=bool)
            vector[~selected] = 0.0
        self.atoms.set_positions(self.atoms.get_positions() + vector)
        self._have_mode = False
        self._session = None
        self._eigenmode = None
        self._invalidate()


class MinModeTranslate:
    """ASE's minimum-mode translation, on ``rgsaddle``'s mode and force.

    The trial step and the conjugate-gradient direction are ASE's.
    The mode and the curvature come from ``MinModeAtoms.get_forces``.
    """

    def __init__(self, dimeratoms: MinModeAtoms, logfile: Any = "-", trajectory: Any = None) -> None:
        from ase.optimize.optimize import Optimizer

        translate = self

        class _Driver(Optimizer):
            def initialize(self) -> None:
                return None

            def read(self) -> None:
                return None

            def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
                translate._step()

        self.dimeratoms = dimeratoms
        self.control = dimeratoms.get_control()
        if self.control is None:
            raise ValueError("MinModeTranslate needs a MinModeAtoms object with a control")
        self.cg_on = bool(self.control.get_parameter("cg_translation"))
        self.trial_step = float(self.control.get_parameter("trial_trans_step"))
        self.max_step = float(self.control.get_parameter("maximum_translation"))
        self._cg_init = True
        self._direction_old: Float64 | None = None
        self._cg_direction: Float64 | None = None
        self._driver = _Driver(cast(Any, dimeratoms), logfile=logfile, trajectory=trajectory)

    def __enter__(self) -> MinModeTranslate:
        enter: Callable[[], Any] = self._driver.__enter__
        enter()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> Any:
        exit_driver: Callable[[object, object, object], Any] = self._driver.__exit__
        return exit_driver(exc_type, exc_value, traceback)

    def run(self, fmax: float = 0.05, steps: int | None = None) -> bool:
        run_driver: Callable[..., Any] = self._driver.run
        if steps is None:
            return bool(run_driver(fmax=fmax))
        return bool(run_driver(fmax=fmax, steps=steps))

    def _cg(self, direction: Float64) -> Float64:
        if self._cg_init or self._direction_old is None or self._cg_direction is None:
            self._cg_init = False
            self._direction_old = np.array(direction, dtype=np.float64, copy=True)
            self._cg_direction = np.array(direction, dtype=np.float64, copy=True)
            return np.array(self._cg_direction, dtype=np.float64, copy=True)
        old_norm = float(np.dot(self._direction_old.ravel(), self._direction_old.ravel()))
        beta = 0.0
        if old_norm != 0.0:
            beta = float(
                np.dot(direction.ravel(), (direction - self._direction_old).ravel()) / old_norm
            )
        if beta < 0.0:
            beta = 0.0
        self._cg_direction = direction + self._cg_direction * beta
        self._direction_old = np.array(direction, dtype=np.float64, copy=True)
        return np.array(self._cg_direction, dtype=np.float64, copy=True)

    def _step(self) -> None:
        atoms = self.dimeratoms
        forces = np.array(atoms.get_forces(), dtype=np.float64, copy=True)
        positions = np.array(atoms.get_positions(), dtype=np.float64, copy=True)
        curvature = float(atoms.get_curvature())
        direction = forces.copy()
        if self.cg_on:
            direction = self._cg(direction)
        if not np.isfinite(direction).all():
            raise RuntimeError("Dimer calculation diverged")
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return
        direction = direction / length
        if curvature > 0.0:
            step = direction * self.max_step
        else:
            trial = positions + direction * self.trial_step
            trial_force = np.asarray(atoms.get_projected_forces(trial), dtype=np.float64).reshape(-1)
            along = direction.reshape(-1)
            current = forces.reshape(-1)
            midpoint = float(np.dot(trial_force + current, along) / 2.0)
            stiffness = float(np.dot(trial_force - current, along) / self.trial_step)
            if stiffness == 0.0 or not np.isfinite(stiffness):
                step = direction * self.max_step
            else:
                coeff = -midpoint / stiffness + self.trial_step / 2.0
                step = coeff * direction
                if float(np.linalg.norm(step)) > self.max_step or not np.isfinite(step).all():
                    step = direction * self.max_step
        atoms.set_positions(positions + step)


class _SellaView:
    def __init__(self, opt: Sella) -> None:
        self._opt = opt

    def get_x(self) -> Float64:
        return np.ascontiguousarray(self._opt.atoms.get_positions(), dtype=np.float64).reshape(-1)

    def set_x(self, x: Float64) -> None:
        count = len(self._opt.atoms)
        self._opt.atoms.set_positions(
            np.asarray(x, dtype=np.float64).reshape(count, 3), apply_constraint=False
        )

    def get_gradient(self) -> Float64:
        forces = np.ascontiguousarray(self._opt.atoms.get_forces(), dtype=np.float64)
        return -forces.reshape(-1)

    def get_value(self) -> float:
        return float(self._opt.atoms.get_potential_energy())

    def ndofs(self) -> int:
        return 3 * len(self._opt.atoms)


class Sella:
    """Order-1 saddle optimizer with ASE's ``run(fmax, steps)`` loop.

    One ``step`` is a partitioned rational-function trust step on the
    calculator's gradient. The trust radius starts at ``delta`` per
    free coordinate. Masses come from ``atoms.get_masses``.
    """

    def __init__(self, atoms: Any, order: int = 1, delta: float = 0.1) -> None:
        if getattr(atoms, "calc", None) is None:
            raise ValueError("atoms has no calculator")
        self.atoms = atoms
        self.order = int(order)
        self.delta = float(delta)
        self._search: Any = None
        self._observers: list[tuple[Any, int, tuple[Any, ...], dict[str, Any]]] = []

    def __ase_optimizable__(self) -> _SellaView:
        return _SellaView(self)

    def attach(self, function: Any, interval: int = 1, *args: Any, **kwargs: Any) -> None:
        self._observers.append((function, interval, args, kwargs))

    def _surface(self, positions: Float64) -> tuple[Float64, Float64]:
        count = len(self.atoms)
        self.atoms.set_positions(
            np.ascontiguousarray(positions, dtype=np.float64).reshape(count, 3),
            apply_constraint=False,
        )
        energy = float(self.atoms.get_potential_energy())
        gradient = -np.ascontiguousarray(self.atoms.get_forces(), dtype=np.float64).reshape(1, -1)
        return np.array([energy], dtype=np.float64), gradient

    def _ensure(self, force_tol: float) -> Any:
        if self._search is None:
            from rgsaddle._core import SellaSaddle

            coordinates = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
            masses = np.ascontiguousarray(self.atoms.get_masses(), dtype=np.float64)
            self._search = SellaSaddle(
                coordinates, masses, order=self.order, delta=self.delta, force_tol=force_tol
            )
        return self._search

    def run(
        self,
        fmax: float = 0.05,
        steps: int | None = None,
        *,
        logfile: Any = None,
        trajectory: Any = None,
    ) -> bool:
        from ase.optimize.optimize import DEFAULT_MAX_STEPS, Optimizer

        opt = self

        class SellaDriver(Optimizer):
            def initialize(self) -> None:
                return None

            def read(self) -> None:
                return None

            def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
                search = opt._ensure(fmax)
                search.step(opt._surface)
                count = len(opt.atoms)
                updated = np.asarray(search.position(), dtype=np.float64).reshape(count, 3)
                opt.atoms.set_positions(updated, apply_constraint=False)

        if steps is None:
            steps = DEFAULT_MAX_STEPS
        driver = SellaDriver(self.atoms, logfile=logfile, trajectory=trajectory)
        host: Any = driver
        for function, interval, args, kwargs in self._observers:
            host.attach(function, interval, *args, **kwargs)
        return bool(host.run(fmax=fmax, steps=steps))


def _periodic_cell(images: list[Any]) -> Float64 | None:
    if not images:
        return None
    pbc = np.asarray(images[0].pbc, dtype=bool)
    if not pbc.any():
        return None
    raw = np.asarray(images[0].cell.array, dtype=np.float64).copy()
    for axis in range(3):
        if not pbc[axis]:
            raw[axis, :] = 0.0
    return np.ascontiguousarray(raw, dtype=np.float64)


class _StringView:
    """ASE optimizable view of a SimplifiedString band."""

    def __init__(self, string: SimplifiedString) -> None:
        self._string = string

    def get_x(self) -> Float64:
        return np.ascontiguousarray(self._string.get_positions(), dtype=np.float64).reshape(-1)

    def set_x(self, x: Float64) -> None:
        self._string.set_positions(np.asarray(x, dtype=np.float64))

    def get_gradient(self) -> Float64:
        if self._string._projected is not None:
            return -np.ascontiguousarray(self._string._projected, dtype=np.float64).reshape(-1)
        return -np.ascontiguousarray(self._string.get_forces(), dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        return float(self._string.get_potential_energy())

    def ndofs(self) -> int:
        return int(self.get_x().shape[0])

    def gradient_norm(self, gradient: Float64) -> float:
        if gradient.size == 0:
            return 0.0
        return float(np.max(np.abs(np.asarray(gradient, dtype=np.float64))))

    def converged(self, gradient: Float64, fmax: float) -> bool:
        if self._string._report is not None:
            return bool(self._string._report.status == 1)
        if not self._string._driving:
            return self.gradient_norm(gradient) < fmax
        return False

    def iterimages(self) -> list[Any]:
        return list(self._string.images)


class SimplifiedString:
    """Simplified string method over a list of ASE ``Atoms``.

    E, Ren, and Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007),
    doi:10.1063/1.2720838. Interior images evolve under the full
    potential force (no spring, no tangent projection) through an rgmin
    stepper and are placed at equal arc length after each step on the
    not-a-knot cubic spline (``reparametrization='not-a-knot'``, the
    default) or on the polyline (``'linear'``). With ``climb=True`` the
    climbing image reflects the potential force along the tangent and
    stays anchored during reparametrization.

    ``run(fmax, steps)`` drives the Rust :class:`rgsaddle.Band` session
    under an ASE ``Optimizer`` loop so ``logfile``, ``trajectory``, and
    attached observers run per step.
    """

    def __init__(
        self,
        images: list[Any],
        *,
        climb: bool = False,
        ci_after: float = 0.5,
        ci_after_rel: float = 0.0,
        reparametrization: str = "not-a-knot",
        method: str = "fire",
        memory: int = 20,
        max_move: float = 0.2,
        time_step: float = 1.0,
        remove_rotation_and_translation: bool = False,
        min_mode: bool = False,
        allow_shared_calculator: bool = False,
    ) -> None:
        if len(images) < 3:
            raise ValueError("a string needs two endpoints and at least one interior image")
        if reparametrization not in ("linear", "not-a-knot", "spline"):
            raise ValueError("reparametrization must be linear, not-a-knot, or spline")
        self.images = images
        self.climb = bool(climb)
        self.ci_after = float(ci_after)
        self.ci_after_rel = float(ci_after_rel)
        self.reparametrization = reparametrization
        self.method = method
        self.memory = int(memory)
        self.max_move = float(max_move)
        self.time_step = float(time_step)
        self.remove_rotation_and_translation = bool(remove_rotation_and_translation)
        self.min_mode = bool(min_mode)
        self._adapter = SurfaceAdapter(images, allow_shared_calculator=allow_shared_calculator)
        self.energies: Float64 | None = None
        self.emax = float("nan")
        self.ci: int | None = None
        self._band: Any = None
        self._report: Any = None
        self._projected: Float64 | None = None
        self._driving = False
        self._observers: list[tuple[Any, int, tuple[Any, ...], dict[str, Any]]] = []

    def __len__(self) -> int:
        if len(self.images) < 2:
            return 0
        return (len(self.images) - 2) * len(self.images[0])

    def __ase_optimizable__(self) -> _StringView:
        return _StringView(self)

    def attach(self, function: Any, interval: int = 1, *args: Any, **kwargs: Any) -> None:
        self._observers.append((function, interval, args, kwargs))

    def interpolate(
        self, method: str = "linear", mic: bool = False, apply_constraint: bool = True
    ) -> None:
        """Place the interior images on the straight line between the ends."""
        del apply_constraint
        if method != "linear":
            raise ValueError("interpolate linearly here; idpp_path builds an IDPP start in Rust")
        if mic:
            raise ValueError("unwrap the images first; interpolation here is Cartesian")
        start = np.asarray(self.images[0].get_positions(), dtype=np.float64)
        end = np.asarray(self.images[-1].get_positions(), dtype=np.float64)
        fixed = _fixed_atoms(self.images)
        last = len(self.images) - 1
        for index, image in enumerate(self.images[1:-1], start=1):
            weight = index / last
            row = (1.0 - weight) * start + weight * end
            if fixed is not None:
                row[fixed == 1] = start[fixed == 1]
            image.set_positions(row, apply_constraint=False)
        self._band = None
        self._report = None
        self._projected = None

    def get_positions(self) -> Float64:
        blocks = [
            np.asarray(image.get_positions(), dtype=np.float64) for image in self.images[1:-1]
        ]
        if not blocks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(blocks, axis=0)

    def set_positions(self, positions: Any) -> None:
        from rgsaddle._core import reparametrize

        natoms = len(self.images[0])
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        expect = (len(self.images) - 2) * natoms
        if len(array) != expect:
            raise ValueError(f"positions: expected {expect} atoms, got {len(array)}")
        whole = _image_positions(self.images)
        whole[1:-1] = array.reshape((len(self.images) - 2, natoms * 3))
        placed = reparametrize(
            whole,
            scheme=self.reparametrization,
            anchor=self.ci,
            cell=_periodic_cell(self.images),
        )
        fixed = _fixed_atoms(self.images)
        first = np.asarray(self.images[0].get_positions(), dtype=np.float64)
        for index, image in enumerate(self.images[1:-1], start=1):
            coords = np.asarray(placed[index], dtype=np.float64).reshape((natoms, 3)).copy()
            if fixed is not None:
                coords[fixed == 1] = first[fixed == 1]
            image.set_positions(coords, apply_constraint=False)
        self._band = None
        self._report = None
        self._projected = None

    def get_potential_energy(self) -> float:
        if self.energies is None:
            self.get_forces()
        return float(self.emax)

    def get_forces(self) -> Float64:
        from rgsaddle._core import project_band

        positions = _image_positions(self.images)
        ids = np.arange(positions.shape[0], dtype=np.int64)
        energies, gradients = self._adapter(ids, positions)
        flat, ci = project_band(
            positions,
            np.ascontiguousarray(energies, dtype=np.float64),
            np.ascontiguousarray(gradients, dtype=np.float64),
            k=0.0,
            climb=self.climb,
            projection="string",
            tangent="improved",
            remove_translation=self.remove_rotation_and_translation,
            fixed=_fixed_atoms(self.images),
        )
        count = len(self.images)
        natoms = len(self.images[0])
        forces = np.asarray(flat, dtype=np.float64).reshape(count, natoms, 3)
        self.energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        interior = self.energies[1:-1]
        self.emax = float(np.max(interior)) if interior.size else float("nan")
        self.ci = None if int(ci) < 0 else int(ci)
        return np.ascontiguousarray(forces[1:-1].reshape(-1, 3), dtype=np.float64)

    def _surface(self, x: Float64) -> tuple[Float64, Float64]:
        rows = int(x.shape[0])
        count = len(self.images)
        if rows == count:
            ids = np.arange(count, dtype=np.int64)
        elif rows == count - 2:
            ids = np.arange(1, count - 1, dtype=np.int64)
        elif rows == 1:
            target = 1 if self.ci is None else self.ci
            ids = np.array([target], dtype=np.int64)
        else:
            raise ValueError(f"unexpected band request row count {rows} for {count} images")
        return self._adapter(ids, np.ascontiguousarray(x, dtype=np.float64))

    def _ensure(self, force_tol: float) -> Any:
        if self._band is None:
            from rgsaddle._core import Band

            self._band = Band(
                _image_positions(self.images),
                climb=self.climb,
                ci_after=self.ci_after,
                ci_after_rel=self.ci_after_rel,
                method=self.method,
                memory=self.memory,
                force_tol=force_tol,
                max_move=self.max_move,
                time_step=self.time_step,
                remove_translation=self.remove_rotation_and_translation,
                min_mode=self.min_mode,
                string=True,
                reparametrization=self.reparametrization,
                cell=_periodic_cell(self.images),
                fixed=_fixed_atoms(self.images),
            )
            if not self.climb:
                self._band.set_saddle_check(enabled=False)
        return self._band

    def run(
        self,
        fmax: float = 0.05,
        steps: int | None = None,
        *,
        logfile: Any = None,
        trajectory: Any = None,
    ) -> bool:
        from ase.optimize.optimize import DEFAULT_MAX_STEPS, Optimizer

        opt = self
        self._band = None
        self._report = None
        self._driving = True

        class StringDriver(Optimizer):
            def initialize(self) -> None:
                return None

            def read(self) -> None:
                return None

            def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
                band = opt._ensure(fmax)
                report = band.step(opt._surface)
                natoms = len(opt.images[0])
                updated = np.asarray(band.positions(), dtype=np.float64).reshape(
                    len(opt.images), natoms, 3
                )
                for idx in range(1, len(opt.images) - 1):
                    opt.images[idx].set_positions(updated[idx], apply_constraint=False)
                energies, _, projected = band.evaluation()
                opt.energies = np.asarray(energies, dtype=np.float64).reshape(-1)
                interior = opt.energies[1:-1]
                opt.emax = float(np.max(interior)) if interior.size else float("nan")
                opt.ci = None if int(report.ci_index) < 0 else int(report.ci_index)
                opt._projected = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
                opt._report = report

        if steps is None:
            steps = DEFAULT_MAX_STEPS
        try:
            target: Any = self
            driver = StringDriver(target, logfile=logfile, trajectory=trajectory)
            host: Any = driver
            for function, interval, args, kwargs in self._observers:
                host.attach(function, interval, *args, **kwargs)
            return bool(host.run(fmax=fmax, steps=steps))
        finally:
            self._driving = False

    def saddle_index(self, **options: Any) -> SaddleIndex:
        """Saddle index of the climbing image, seeded with its tangent."""
        if self.ci is None:
            self.get_forces()
        if self.ci is None:
            raise ValueError("no interior image climbs")
        positions = _image_positions(self.images)
        tangent = positions[self.ci + 1] - positions[self.ci - 1]
        return saddle_index(self.images[self.ci], tangent, **options)


class GrowingString:
    """Growing string method over a list of ASE ``Atoms``.

    Peters, Heyden, Bell, and Chakraborty, J. Chem. Phys. 120, 7877
    (2004), doi:10.1063/1.1691018, handing off upon join to the
    simplified string method of E, Ren, and Vanden-Eijnden, J. Chem.
    Phys. 126, 164103 (2007), doi:10.1063/1.2720838.
    """

    def __init__(
        self,
        images: list[Any],
        *,
        frontier_tol: float = 0.2,
        min_steps_per_node: int = 1,
        max_steps_per_node: int = 0,
        join_tol: float | None = None,
        climb: bool = True,
        ci_after: float = 0.5,
        ci_after_rel: float = 0.0,
        reparametrization: str = "not-a-knot",
        method: str = "fire",
        memory: int = 20,
        max_move: float = 0.2,
        remove_rotation_and_translation: bool = False,
        min_mode: bool = False,
        allow_shared_calculator: bool = False,
    ) -> None:
        if len(images) < 4:
            raise ValueError("a growing string needs at least four image slots")
        if reparametrization not in ("linear", "not-a-knot", "spline"):
            raise ValueError("reparametrization must be linear, not-a-knot, or spline")
        self.images = images
        self.frontier_tol = float(frontier_tol)
        self.min_steps_per_node = int(min_steps_per_node)
        self.max_steps_per_node = int(max_steps_per_node)
        self.join_tol = None if join_tol is None else float(join_tol)
        self.climb = bool(climb)
        self.ci_after = float(ci_after)
        self.ci_after_rel = float(ci_after_rel)
        self.reparametrization = reparametrization
        self.method = method
        self.memory = int(memory)
        self.max_move = float(max_move)
        self.remove_rotation_and_translation = bool(remove_rotation_and_translation)
        self.min_mode = bool(min_mode)
        self._adapter = SurfaceAdapter(images, allow_shared_calculator=allow_shared_calculator)
        self.ci: int | None = None
        self._session: CoreGrowingString | None = None
        self._report: Any = None
        self._endpoints: tuple[Float64, Float64] | None = None
        self._observers: list[tuple[Any, int, tuple[Any, ...], dict[str, Any]]] = []

    def attach(self, function: Any, interval: int = 1, *args: Any, **kwargs: Any) -> None:
        self._observers.append((function, interval, args, kwargs))

    def _restore_endpoints(self) -> None:
        # Image ids run 0..n-1, so the adapter evaluates the last list slot
        # as an interior node. The endpoints must not keep that geometry.
        if self._endpoints is None:
            return
        reactant, product = self._endpoints
        self.images[0].set_positions(reactant, apply_constraint=False)
        self.images[-1].set_positions(product, apply_constraint=False)

    def _surface(self, x: Float64) -> tuple[Float64, Float64]:
        assert self._session is not None
        rows = int(x.shape[0])
        active_ids = np.asarray(self._session.ids(), dtype=np.int64)
        if rows == int(active_ids.shape[0]):
            ids = active_ids
        elif rows == int(active_ids.shape[0]) - 2:
            ids = active_ids[1:-1]
        elif rows == 1:
            target = 1 if self.ci is None else int(active_ids[self.ci])
            ids = np.array([target], dtype=np.int64)
        else:
            ids = np.arange(rows, dtype=np.int64)
        return self._adapter(
            np.ascontiguousarray(ids, dtype=np.int64),
            np.ascontiguousarray(x, dtype=np.float64),
        )

    def _ensure(self, force_tol: float) -> CoreGrowingString:
        if self._session is None:
            reactant = np.ascontiguousarray(
                self.images[0].get_positions(), dtype=np.float64
            ).reshape(-1)
            product = np.ascontiguousarray(
                self.images[-1].get_positions(), dtype=np.float64
            ).reshape(-1)
            self._endpoints = (
                np.array(self.images[0].get_positions(), dtype=np.float64, copy=True),
                np.array(self.images[-1].get_positions(), dtype=np.float64, copy=True),
            )
            self._session = CoreGrowingString(
                reactant,
                product,
                n_images=len(self.images),
                frontier_tol=self.frontier_tol,
                min_steps_per_node=self.min_steps_per_node,
                max_steps_per_node=self.max_steps_per_node,
                join_tol=self.join_tol,
                climb=self.climb,
                ci_after=self.ci_after,
                ci_after_rel=self.ci_after_rel,
                method=self.method,
                memory=self.memory,
                force_tol=force_tol,
                max_move=self.max_move,
                remove_translation=self.remove_rotation_and_translation,
                min_mode=self.min_mode,
                reparametrization=self.reparametrization,
                cell=_periodic_cell(self.images),
                fixed=_fixed_atoms(self.images),
            )
        return self._session

    def run(
        self,
        fmax: float = 0.05,
        steps: int = 500,
    ) -> bool:
        session = self._ensure(fmax)
        natoms = len(self.images[0])
        try:
            for _ in range(steps):
                report = session.step(self._surface)
                self._restore_endpoints()
                self._report = report
                self.ci = None if int(report.ci_index) < 0 else int(report.ci_index)
                if session.joined():
                    updated = np.asarray(session.positions(), dtype=np.float64).reshape(
                        len(self.images), natoms, 3
                    )
                    for idx in range(1, len(self.images) - 1):
                        self.images[idx].set_positions(updated[idx], apply_constraint=False)
                for function, interval, args, kwargs in self._observers:
                    if int(report.iteration) % max(int(interval), 1) == 0:
                        function(*args, **kwargs)
                if int(report.status) == 1:
                    return True
            return False
        finally:
            self._restore_endpoints()

    def saddle_index(self) -> SaddleIndex:
        if self._session is None:
            raise ValueError("growing string has not run")
        return self._session.saddle_index()
