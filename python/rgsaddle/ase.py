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

from typing import Any, Literal

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

    def iterimages(self) -> list[Any]:
        return list(self._neb.images)


class NEB:
    """Nudged elastic band over a list of ASE ``Atoms``.

    The constructor matches ``ase.mep.NEB``: ``images`` carry the
    calculators, ``k`` is the spring constant, and ``climb`` turns the
    climbing image on for every force call.     ``get_positions`` and
    ``get_forces`` are the interior images only, shape
    ``((n_images - 2) * n_atoms, 3)``, which is what ASE's optimizer
    reads. ``get_potential_energy`` is the highest interior energy,
    the value ASE's NEB reports. A higher fixed endpoint does not
    replace it.

    ``climb=True`` does not climb an interior image when a higher
    endpoint slopes down into the band. ASE's ``NEB`` climbs the
    highest image on every call, including that slope. The force is
    then the reflected true force, and the image is driven up a slope
    that has no saddle on it.

    ``method`` is ``improvedtangent`` (default), ``aseneb``, ``eb``, or
    ``dneb``. The tangents are unit vectors, so ``aseneb`` and
    ``improvedtangent`` project the same force. ``spline`` and
    ``string`` are not this class. ``remove_rotation_and_translation``
    removes the mean Cartesian force. It does not rotate the images:
    a Kabsch step would move the geometry the calculator sees, and the
    unit tangent already fixes the path direction.

    ``FixAtoms`` on the images is the band's fixed-atom mask (rgsaddle
    ABI minor 39): those atoms get zero force rows, tangents and
    springs use the free atoms, and the mean force stays even with
    ``remove_rotation_and_translation``. Every image must fix the same
    atoms, and the endpoints must agree on where they sit.

    Lengths and energies are the calculator's.
    """

    def __init__(
        self,
        images: list[Any],
        k: float = 0.1,
        climb: bool = False,
        parallel: bool = False,
        remove_rotation_and_translation: bool = False,
        method: str | None = None,
        allow_shared_calculator: bool = False,
        precon: Any = None,
        **kwargs: Any,
    ) -> None:
        if parallel:
            raise ValueError("this NEB evaluates images on the calling thread")
        if precon is not None:
            raise ValueError("preconditioning belongs on BandConfig.exp_precon")
        if kwargs:
            name = next(iter(kwargs))
            raise ValueError(f"unsupported NEB argument {name}")
        self.images = images
        self.k = float(k)
        self.climb = bool(climb)
        self.remove_rotation_and_translation = bool(remove_rotation_and_translation)
        chosen = "improvedtangent" if method is None else method
        if chosen in ("improvedtangent", "aseneb"):
            self._projection = "neb"
            self._tangent = "improved"
        elif chosen == "eb":
            self._projection = "eb"
            self._tangent = "improved"
        elif chosen == "dneb":
            self._projection = "dneb"
            self._tangent = "improved"
        else:
            raise ValueError(
                "method must be improvedtangent, aseneb, eb, or dneb; "
                "use SimplifiedString for the simplified string method"
            )
        self.method = chosen
        self._adapter = SurfaceAdapter(images, allow_shared_calculator=allow_shared_calculator)
        self.energies: Float64 | None = None
        self.emax = float("nan")
        self.ci: int | None = None

    def __len__(self) -> int:
        if len(self.images) < 2:
            return 0
        return (len(self.images) - 2) * len(self.images[0])

    def __ase_optimizable__(self) -> _NebView:
        return _NebView(self)

    def interpolate(self, method: str = "linear", mic: bool = False, apply_constraint: bool = True) -> None:
        """Place the interior images on the straight line between the ends.

        ``mic=True`` is refused: pass images the host has already
        unwrapped. A ``Cell`` on the Rust band applies the minimum
        image to force differences, not to this placement.
        """
        del apply_constraint
        if method != "linear":
            raise ValueError("interpolate linearly here; idpp_path builds an IDPP start in Rust")
        if mic:
            raise ValueError("unwrap the images first; interpolation here is Cartesian")
        if len(self.images) < 3:
            raise ValueError("interpolation needs two endpoints and one interior image")
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

    def get_positions(self) -> Float64:
        blocks = [
            np.asarray(image.get_positions(), dtype=np.float64) for image in self.images[1:-1]
        ]
        if not blocks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(blocks, axis=0)

    def set_positions(self, positions: Any) -> None:
        natoms = len(self.images[0])
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        expect = (len(self.images) - 2) * natoms
        count = len(array)
        if count != expect:
            raise ValueError(f"positions: expected {expect} atoms, got {count}")
        for index, image in enumerate(self.images[1:-1]):
            image.set_positions(array[index * natoms : (index + 1) * natoms], apply_constraint=False)

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
            k=self.k,
            climb=self.climb,
            projection=self._projection,
            tangent=self._tangent,
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
        self._search._moved()

    def get_gradient(self) -> Float64:
        return -np.ascontiguousarray(self._search.get_forces(), dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        return float(self._search.atoms.get_potential_energy())

    def ndofs(self) -> int:
        return 3 * len(self._search.atoms)


class MinModeAtoms:
    """One ASE ``Atoms`` object driven by a dimer or Lanczos mode.

    ``get_forces`` estimates the lowest mode and returns the effective
    force an ASE optimizer should follow. Positive curvature drops the
    force perpendicular to the mode. Negative curvature reflects the
    force along the mode. The atoms object itself is not copied.
    """

    def __init__(
        self,
        atoms: Any,
        mode: Any = None,
        *,
        kind: str = "dimer",
        dr: float = 1e-3,
        max_rotations: int = 10,
    ) -> None:
        if getattr(atoms, "calc", None) is None:
            raise ValueError("atoms has no calculator")
        self.atoms = atoms
        self.kind = kind
        self.dr = float(dr)
        self.max_rotations = int(max_rotations)
        count = len(atoms)
        if mode is None:
            guess = np.zeros(count * 3, dtype=np.float64)
            guess[0] = 1.0
        else:
            guess = np.ascontiguousarray(mode, dtype=np.float64).reshape(-1)
        self._mode0 = guess
        self._session: Any = None
        self.mode: Float64 | None = None
        self.curvature = float("nan")

    def _moved(self) -> None:
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
        if self._session is None:
            from rgsaddle._core import MinMode

            coordinates = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
            self._session = MinMode(
                coordinates,
                self._mode0,
                kind=self.kind,
                dr=self.dr,
                max_rotations=self.max_rotations,
            )
        return self._session

    def get_forces(self) -> Float64:
        search = self._ensure()
        report = search.estimate(self._surface)
        self.curvature = float(report.curvature)
        self.mode = np.asarray(search.mode(), dtype=np.float64)
        forces = np.array(self.atoms.get_forces(), dtype=np.float64, copy=True)
        return _effective_force(forces, self.mode, self.curvature)


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
