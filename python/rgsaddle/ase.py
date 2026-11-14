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

import importlib
import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rgsaddle._core import ClaimSlot

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
    or ``min``. Band presets stay on ``feat/band-methods-zoom``.
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


def _ase(module: str, name: str) -> Any:
    """An ASE function, imported on first use. ASE's functions are untyped."""
    return getattr(importlib.import_module(module), name)


def _image_positions(images: list[Any]) -> Float64:
    rows = [np.asarray(image.get_positions(), dtype=np.float64).reshape(-1) for image in images]
    return np.ascontiguousarray(np.stack(rows), dtype=np.float64)


def _max_atom_norm(gradient: Any) -> float:
    """ASE's ``fmax`` measure: the largest per-atom force norm."""
    forces = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
    if forces.shape[0] == 0:
        return 0.0
    return float(np.sqrt((forces * forces).sum(axis=1).max()))


# rgsaddle has no ASE ``aseneb`` tangent or ASE ``eb`` spring. The names
# stay accepted and map to the nearest rgsaddle band, with a warning.
_NEB_METHODS: dict[str, tuple[str, str, str | None]] = {
    "improvedtangent": ("neb", "improved", None),
    "aseneb": (
        "neb",
        "improved",
        "rgsaddle has no equivalent of ASE's aseneb tangent and spring; "
        "this band uses the improved tangent (method='improvedtangent')",
    ),
    "eb": (
        "eb",
        "improved",
        "method='eb' is rgsaddle's plain elastic band (full spring at zero rest "
        "length, improved tangent), not ASE's FullSpringMethod",
    ),
    "dneb": ("dneb", "improved", None),
}


class _NebView:
    """ASE optimizable view of interior images."""

    def __init__(self, neb: NEB) -> None:
        self._neb = neb

    def get_x(self) -> Float64:
        return np.ascontiguousarray(self._neb.get_positions(), dtype=np.float64).reshape(-1)

    def set_x(self, x: Float64) -> None:
        self._neb.set_positions(np.asarray(x, dtype=np.float64).reshape(-1, 3))

    def get_gradient(self) -> Float64:
        return -np.ascontiguousarray(self._neb.get_forces(), dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        return float(self._neb.get_potential_energy())

    def ndofs(self) -> int:
        return 3 * len(self._neb)

    def gradient_norm(self, gradient: Float64) -> float:
        return _max_atom_norm(gradient)

    def converged(self, gradient: Float64, fmax: float) -> bool:
        return self.gradient_norm(gradient) < fmax

    def iterimages(self) -> Any:
        return self._neb.iterimages()


class NEB:
    """Nudged elastic band over a list of ASE ``Atoms``.

    The constructor is ``ase.mep.NEB``'s: ``images`` carry the
    calculators, ``k`` is one spring constant or one per segment, and
    ``climb`` climbs the highest interior image on every force call.
    ``get_positions`` and ``get_forces`` are the interior images only,
    shape ``((n_images - 2) * n_atoms, 3)``, which is what an ASE
    optimizer reads. ``get_potential_energy`` is the highest interior
    energy. After a force call ``energies``, ``real_forces``, ``imax``,
    ``emax`` and ``residuals`` hold what ASE's NEB holds, and ``ci`` is
    the image that climbed, or None.

    Spring vectors between neighbours use the minimum image of the
    first image's cell along its periodic axes, as ASE does.
    ``set_positions`` applies the image constraints. Gradients are
    minus ``Atoms.get_forces()``, so a ``FixAtoms`` atom feels no force.

    ``climb=True`` does not climb an interior image when a higher
    endpoint slopes down into the band. ASE's ``NEB`` climbs the
    highest image on every call, including that slope. The force is
    then the reflected true force, and the image is driven up a slope
    that has no saddle on it.

    ``method`` is ``improvedtangent`` (default), ``dneb``, ``aseneb``
    or ``eb``. ``aseneb`` and ``eb`` have no rgsaddle equivalent and
    warn: ``aseneb`` uses the improved tangent, ``eb`` rgsaddle's plain
    elastic band. ``spline`` and ``string`` are refused.
    ``remove_rotation_and_translation`` aligns neighbouring images with
    ``ase.build.minimize_rotation_and_translation`` before each force
    call, as ASE does. One rank only: ``parallel`` is refused and
    ``world`` must have size 1.

    Lengths and energies are the calculator's.
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
        if isinstance(world, str):
            # Before 0.3.3 the sixth positional argument was method.
            if method is not None:
                raise TypeError("method was given twice")
            warnings.warn(
                "the sixth positional argument of NEB is world, as in ase.mep.NEB; "
                "pass method by keyword",
                DeprecationWarning,
                stacklevel=2,
            )
            method, world = world, None
        if parallel:
            raise ValueError("this NEB evaluates images on the calling thread")
        if world is not None and int(getattr(world, "size", 1)) != 1:
            raise ValueError("this NEB runs on one rank; world.size must be 1")
        if precon is not None:
            raise ValueError("preconditioning belongs on BandConfig.exp_precon")
        if kwargs:
            name = next(iter(kwargs))
            raise ValueError(f"unsupported NEB argument {name}")
        if len(images) == 0:
            raise ValueError("images must be non-empty")
        first = images[0]
        for image in images:
            if len(image) != len(first):
                raise ValueError("Images have different numbers of atoms")
            if np.any(np.asarray(image.pbc) != np.asarray(first.pbc)):
                raise ValueError("Images have different boundary conditions")
            if np.any(image.get_atomic_numbers() != first.get_atomic_numbers()):
                raise ValueError("Images have atoms in different orders")
            for periodic, row, row0 in zip(image.pbc, image.cell, first.cell):
                if periodic and np.any(np.abs(np.asarray(row) - np.asarray(row0)) > 1e-8):
                    raise NotImplementedError(
                        "Variable cell in periodic directions is not implemented for NEB"
                    )
        chosen = "improvedtangent" if method is None else method
        if chosen not in _NEB_METHODS:
            raise ValueError(
                "method must be improvedtangent, dneb, aseneb, or eb; "
                "spline and string are not implemented"
            )
        projection, tangent, note = _NEB_METHODS[chosen]
        if note is not None:
            warnings.warn(note, UserWarning, stacklevel=2)
        self._projection = projection
        self._tangent = tangent
        self.method = chosen
        self.images = images
        self.climb = bool(climb)
        self.parallel = False
        self.world = world
        self.precon = None
        self.allow_shared_calculator = bool(allow_shared_calculator)
        self.remove_rotation_and_translation = bool(remove_rotation_and_translation)
        springs = len(images) - 1
        if isinstance(k, (int, float, np.integer, np.floating)):
            self.k: list[float] = [float(k)] * max(springs, 0)
        else:
            self.k = [float(value) for value in k]
            if len(self.k) != springs:
                raise ValueError(f"k: expected {springs} spring constants, got {len(self.k)}")
        self._adapter: SurfaceAdapter | None = None
        self.energies: Float64 | None = None
        self.real_forces: Float64 | None = None
        self.residuals: list[float] | None = None
        self.emax = float("nan")
        self.imax: int | None = None
        self.ci: int | None = None

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
    def freeze_results_on_image(atoms: Any, **results_to_include: Any) -> None:
        atoms.calc = _ase("ase.calculators.singlepoint", "SinglePointCalculator")(
            atoms=atoms, **results_to_include
        )

    def _surface_adapter(self) -> SurfaceAdapter:
        if self._adapter is None:
            self._adapter = SurfaceAdapter(
                self.images, allow_shared_calculator=self.allow_shared_calculator
            )
        return self._adapter

    def _springs(self) -> tuple[float, Float64 | None]:
        ks = np.asarray(self.k, dtype=np.float64)
        if ks.size == 0 or np.all(ks == ks[0]):
            return float(ks[0]) if ks.size else 0.1, None
        return float(ks[0]), np.ascontiguousarray(ks)

    def _unwrapped(self, raw: Float64) -> tuple[Float64, Float64]:
        """The band with each image moved to the minimum image of the one before.

        Returns ``(band, shifts)`` with ``band = raw + shifts``. The shifts
        are lattice vectors, so energies and forces do not change.
        """
        pbc = np.asarray(self.images[0].pbc, dtype=bool)
        if not pbc.any():
            return raw, np.zeros_like(raw)
        find_mic = _ase("ase.geometry", "find_mic")

        cell = self.images[0].cell
        natoms = self.natoms
        band = raw.copy()
        for index in range(1, raw.shape[0]):
            step = raw[index].reshape(natoms, 3) - raw[index - 1].reshape(natoms, 3)
            mic, _ = find_mic(step, cell, pbc)
            band[index] = band[index - 1] + np.asarray(mic, dtype=np.float64).reshape(-1)
        return band, band - raw

    def _idpp_cell(self) -> Float64 | None:
        """The cell for minimum-image IDPP pair distances.

        rgsaddle's IDPP wraps all three lattice vectors. Along a vacuum
        axis that matches ASE only while the atoms span less than half
        the cell height there, which is checked.
        """
        pbc = np.asarray(self.images[0].pbc, dtype=bool)
        if not pbc.any():
            return None
        cell = np.array(self.images[0].cell, dtype=np.float64)
        if abs(float(np.linalg.det(cell))) < 1e-12:
            raise ValueError("idpp with mic=True needs a cell with three lattice vectors")
        if not pbc.all():
            inverse = np.linalg.inv(cell)
            for image in (self.images[0], self.images[-1]):
                scaled = np.asarray(image.get_positions(), dtype=np.float64) @ inverse
                span = scaled.max(axis=0) - scaled.min(axis=0)
                if np.any(span[~pbc] >= 0.5):
                    warnings.warn(
                        "the atoms span more than half the cell along a non-periodic axis; "
                        "rgsaddle's IDPP wraps that axis and ASE's does not",
                        UserWarning,
                        stacklevel=3,
                    )
                    break
        return np.ascontiguousarray(cell)

    def _record(self, energies: Float64, raw_forces: Float64 | None, ci: int) -> None:
        count = len(self.images)
        natoms = self.natoms
        self.energies = np.array(energies, dtype=np.float64).reshape(-1)
        real = np.zeros((count, natoms, 3), dtype=np.float64)
        if raw_forces is not None:
            rows = np.asarray(raw_forces, dtype=np.float64).reshape(count, natoms, 3)
            real[1:-1] = rows[1:-1]
        self.real_forces = real
        interior = self.energies[1:-1]
        if interior.size:
            self.imax = 1 + int(np.argsort(interior)[-1])
            self.emax = float(self.energies[self.imax])
        else:
            self.imax = None
            self.emax = float("nan")
        self.ci = None if int(ci) < 0 else int(ci)

    def _residuals(self, interior: Float64) -> None:
        rows = np.asarray(interior, dtype=np.float64).reshape(len(self.images) - 2, -1)
        self.residuals = [
            float(np.abs(row).max()) for index, row in enumerate(rows, start=1) if index != self.ci
        ]

    def interpolate(
        self, method: str = "linear", mic: bool = False, apply_constraint: bool | None = None
    ) -> None:
        """Place the interior images between the ends, as ``ase.mep.NEB.interpolate``.

        ``linear`` is ASE's straight line, with ASE's ``mic`` and
        ``apply_constraint`` handling. ``idpp`` then relaxes the interior
        with rgsaddle's image dependent pair potential (``idpp_path``),
        pair distances taken at the minimum image of a periodic cell.
        """
        if method not in ("linear", "idpp"):
            raise ValueError("method must be linear or idpp")
        if len(self.images) < 3:
            raise ValueError("interpolation needs two endpoints and one interior image")
        if self.remove_rotation_and_translation:
            _ase("ase.build", "minimize_rotation_and_translation")(self.images[0], self.images[-1])
        _ase("ase.mep.neb", "interpolate")(self.images, mic=mic, apply_constraint=apply_constraint)
        if method == "idpp":
            self._idpp(mic=mic, apply_constraint=apply_constraint is not False)

    def idpp_interpolate(
        self,
        traj: Any = "idpp.traj",
        log: Any = "idpp.log",
        fmax: float = 0.1,
        optimizer: Any = None,
        mic: bool = False,
        steps: int = 100,
    ) -> None:
        """ASE's deprecated call. Use ``interpolate(method='idpp')``.

        The path is rgsaddle's ``idpp_path`` from the two endpoints,
        stopped at an IDPP gradient norm of ``fmax`` per image or after
        ``steps`` iterations. ``traj``, ``log`` and ``optimizer`` are
        not used, and no file is written.
        """
        del traj, log, optimizer
        warnings.warn(
            "Please use NEB's interpolate(method='idpp') method",
            FutureWarning,
            stacklevel=2,
        )
        self._idpp(mic=mic, grad_tol=float(fmax), max_iterations=max(int(steps), 1))

    def _idpp(
        self,
        *,
        mic: bool,
        apply_constraint: bool = True,
        grad_tol: float = 1e-3,
        max_iterations: int = 5000,
    ) -> None:
        from rgsaddle._core import idpp_path

        if len(self.images) < 3:
            raise ValueError("interpolation needs two endpoints and one interior image")
        start = np.ascontiguousarray(self.images[0].get_positions(), dtype=np.float64).reshape(-1)
        end = np.ascontiguousarray(self.images[-1].get_positions(), dtype=np.float64).reshape(-1)
        if mic:
            step, _ = _ase("ase.geometry", "find_mic")(
                (end - start).reshape(-1, 3), self.images[0].cell, self.images[0].pbc
            )
            end = np.ascontiguousarray(start + np.asarray(step, dtype=np.float64).reshape(-1))
        path = idpp_path(
            start,
            end,
            len(self.images),
            cell=self._idpp_cell() if mic else None,
            max_iterations=max_iterations,
            grad_tol=grad_tol,
        )
        natoms = self.natoms
        for index, image in enumerate(self.images[1:-1], start=1):
            image.set_positions(
                np.asarray(path[index], dtype=np.float64).reshape(natoms, 3),
                apply_constraint=apply_constraint,
            )

    def get_positions(self) -> Float64:
        blocks = [
            np.asarray(image.get_positions(), dtype=np.float64) for image in self.images[1:-1]
        ]
        if not blocks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(blocks, axis=0)

    def set_positions(self, positions: Any, adjust_positions: bool = True) -> None:
        """Write the interior images. ``Atoms.set_positions`` applies their constraints."""
        del adjust_positions
        natoms = len(self.images[0])
        array = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
        expect = (len(self.images) - 2) * natoms
        if len(array) != expect:
            raise ValueError(f"positions: expected {expect} atoms, got {len(array)}")
        for index, image in enumerate(self.images[1:-1]):
            image.set_positions(array[index * natoms : (index + 1) * natoms])

    def _place(self, band: Float64) -> None:
        """Write session rows to the interior images exactly, without constraints."""
        natoms = self.natoms
        for index, image in enumerate(self.images[1:-1], start=1):
            image.set_positions(
                np.asarray(band[index], dtype=np.float64).reshape(natoms, 3),
                apply_constraint=False,
            )

    def get_potential_energy(self) -> float:
        """The highest interior energy, as ASE's NEB reports."""
        if self.energies is None:
            self.get_forces()
        return float(self.emax)

    def get_residual(self) -> float:
        """Largest force component over the images that do not climb, as ASE."""
        if self.residuals is None:
            raise RuntimeError("get_residual() called before get_forces()")
        return float(np.max(self.residuals)) if self.residuals else 0.0

    def get_forces(self) -> Float64:
        from rgsaddle._core import project_band

        if len(self.images) < 3:
            raise ValueError("a band needs two endpoints and one interior image")
        if self.remove_rotation_and_translation:
            minimize_rotation_and_translation = _ase("ase.build", "minimize_rotation_and_translation")

            for index in range(1, len(self.images)):
                minimize_rotation_and_translation(self.images[index - 1], self.images[index])
        adapter = self._surface_adapter()
        raw = _image_positions(self.images)
        ids = np.arange(raw.shape[0], dtype=np.int64)
        energies, gradients = adapter(ids, raw)
        band, _ = self._unwrapped(raw)
        k, ks = self._springs()
        flat, ci = project_band(
            np.ascontiguousarray(band, dtype=np.float64),
            np.ascontiguousarray(energies, dtype=np.float64),
            np.ascontiguousarray(gradients, dtype=np.float64),
            k=k,
            climb=self.climb,
            projection=self._projection,
            tangent=self._tangent,
            remove_translation=False,
            ks=ks,
        )
        count = len(self.images)
        natoms = self.natoms
        forces = np.asarray(flat, dtype=np.float64).reshape(count, natoms, 3)
        self._record(np.asarray(energies, dtype=np.float64), adapter.raw_forces, int(ci))
        interior = np.ascontiguousarray(forces[1:-1].reshape(-1, 3), dtype=np.float64)
        self._residuals(interior)
        return interior

    def set_calculators(self, calculators: Any) -> None:
        """Attach calculators, as ``ase.mep.NEB.set_calculators``."""
        if not isinstance(calculators, list):
            if self.allow_shared_calculator:
                calculators = [calculators] * len(self.images)
            else:
                raise RuntimeError(
                    "Cannot set shared calculator to NEB with allow_shared_calculator=False"
                )
        count = len(calculators)
        if count == len(self.images):
            for image, calc in zip(self.images, calculators):
                image.calc = calc
        elif count == len(self.images) - 2:
            for image, calc in zip(self.images[1:-1], calculators):
                image.calc = calc
        else:
            raise RuntimeError(
                f"len(calculators)={count} does not fit to len(images)={len(self.images)}"
            )
        self._adapter = None

    def iterimages(self) -> Any:
        """Images for a trajectory: interior copies carry the energy and raw force."""
        for index, atoms in enumerate(self.images):
            if (
                index == 0
                or index == len(self.images) - 1
                or self.energies is None
                or self.real_forces is None
            ):
                yield atoms
            else:
                frozen = atoms.copy()
                self.freeze_results_on_image(
                    frozen, energy=self.energies[index], forces=self.real_forces[index]
                )
                yield frozen


class _BandView(_NebView):
    """The band force from rgsaddle's session at the accepted band."""

    def __init__(self, neb: NEB, driver: Any) -> None:
        super().__init__(neb)
        self._driver = driver

    def get_gradient(self) -> Float64:
        projected = self._driver._current_projection()
        if projected is None:
            return super().get_gradient()
        return -np.ascontiguousarray(projected, dtype=np.float64).reshape(-1)

    def gradient_norm(self, gradient: Any) -> float:
        return float(np.abs(np.asarray(gradient, dtype=np.float64)).max())


_BAND_METHODS = {"ode": "ode12r", "ode12r": "ode12r", "fire": "fire", "lbfgs": "lbfgs",
                 "quickmin": "quickmin", "rtr": "rtr"}
_STALLED = -13


def _build_neb_optimizer() -> type:
    from ase.optimize.optimize import Optimizer

    class NEBOptimizer(Optimizer):
        """rgsaddle's band optimizer behind ``ase.mep.NEBOptimizer``'s call.

        ``neb`` is an ``rgsaddle.ase.NEB``. Each ``step`` is one step of
        rgsaddle's band session (``rgsaddle._core.Band``) on the same
        images, springs, tangent and projection. ``run(fmax, steps)``,
        the log, the trajectory and ``attach`` are ASE's. ``fmax`` bounds
        the largest component of the band force, the measure of ASE's
        ``NEBOptimizer`` (``NEB.get_residual``), except that the climbing
        image is held to it as well.

        ``method`` is ``ode`` (ODE12r, ASE's default name), ``fire``,
        ``lbfgs``, ``quickmin`` or ``rtr``. With ``climb`` the session
        arms the climbing image once the largest force is under
        ``ci_after`` (eOn's ``ci_after``), or under ``ci_after_rel``
        times the first one; ``ci_after=float('inf')`` climbs from the
        start, as ASE does. ``min_mode=True`` hands an armed climbing
        image to the OCI-NEB refinement. ``alpha``, ``verbose``,
        ``rtol``, ``C1`` and ``C2`` are ASE's ODE12r settings and are
        stored, not used.
        """

        def __init__(
            self,
            neb: NEB,
            restart: Any = None,
            logfile: Any = "-",
            trajectory: Any = None,
            master: Any = None,
            append_trajectory: bool = False,
            method: str = "ODE",
            alpha: float = 0.01,
            verbose: int = 0,
            rtol: float = 0.1,
            C1: float = 1e-2,
            C2: float = 2.0,
            *,
            max_move: float = 0.2,
            memory: int = 0,
            ci_after: float = 0.5,
            ci_after_rel: float = 0.0,
            min_mode: bool = False,
        ) -> None:
            if not isinstance(neb, NEB):
                raise TypeError("NEBOptimizer drives an rgsaddle.ase.NEB")
            if neb.remove_rotation_and_translation:
                raise ValueError(
                    "the rgsaddle band session does not rotate images; drive a band with "
                    "remove_rotation_and_translation=True with an ASE optimizer"
                )
            chosen = _BAND_METHODS.get(method.lower())
            if chosen is None:
                raise ValueError(
                    "method must be ode, fire, lbfgs, quickmin, or rtr; "
                    "ASE's static and krylov steppers have no rgsaddle equivalent"
                )
            self.neb = neb
            self.method = chosen
            self.alpha = alpha
            self.verbose = verbose
            self.rtol = rtol
            self.C1 = C1
            self.C2 = C2
            self.max_move = float(max_move)
            self.memory = int(memory)
            self.ci_after = float(ci_after)
            self.ci_after_rel = float(ci_after_rel)
            self.min_mode = bool(min_mode)
            self.report: Any = None
            self._session: Any = None
            self._force_tol: float | None = None
            self._shifts: Float64 | None = None
            self._projected: Float64 | None = None
            self._accepted: Float64 | None = None
            self._raw: dict[int, list[tuple[Float64, Float64]]] = {}
            atoms: Any = neb
            super().__init__(
                atoms=atoms,
                restart=restart,
                logfile=logfile,
                trajectory=trajectory,
                append_trajectory=append_trajectory,
                master=master,
            )
            self.optimizable = _BandView(neb, self)

        def initialize(self) -> None:
            return None

        def read(self) -> None:
            raise NotImplementedError("NEBOptimizer does not read a restart file")

        def todict(self) -> dict[str, Any]:
            base: Any = super()
            description: dict[str, Any] = base.todict()
            description["method"] = self.method
            return description

        def _current_projection(self) -> Float64 | None:
            if self._projected is None or self._accepted is None:
                return None
            if not np.array_equal(_image_positions(self.neb.images), self._accepted):
                return None
            return self._projected

        def _ensure(self) -> Any:
            driver: Any = self
            fmax = float(driver.fmax)
            if self._session is not None and self._force_tol == fmax:
                return self._session
            from rgsaddle._core import Band

            neb = self.neb
            band, shifts = neb._unwrapped(_image_positions(neb.images))
            k, ks = neb._springs()
            self._session = Band(
                np.ascontiguousarray(band, dtype=np.float64),
                k=k,
                ks=ks,
                climb=neb.climb,
                ci_after=self.ci_after,
                ci_after_rel=self.ci_after_rel,
                projection=neb._projection,
                tangent=neb._tangent,
                method=self.method,
                force_tol=fmax,
                max_move=self.max_move,
                memory=self.memory,
                min_mode=self.min_mode,
                force_gate="linf",
            )
            self._force_tol = fmax
            self._shifts = shifts
            self._raw = {}
            return self._session

        def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
            session = self._ensure()
            neb = self.neb
            adapter = neb._surface_adapter()
            shifts = self._shifts
            assert shifts is not None

            def surface(ids: Int64, x: Float64) -> tuple[Float64, Float64]:
                rows = np.ascontiguousarray(x - shifts[ids], dtype=np.float64)
                energies, gradients = adapter(ids, rows)
                raw = adapter.raw_forces
                assert raw is not None
                for row, image in enumerate(ids.tolist()):
                    kept = self._raw.setdefault(int(image), [])
                    kept.append((x[row].copy(), raw[row].copy()))
                    del kept[:-3]
                return energies, gradients

            report = session.step(surface)
            if report.status == _STALLED:
                session.restart()
            self.report = report
            band = np.asarray(session.positions(), dtype=np.float64)
            neb._place(band - shifts)
            self._accepted = _image_positions(neb.images)
            evaluation = session.evaluation()
            if evaluation is None:
                self._projected = None
                return
            energies, _, projected = evaluation
            raw = np.zeros_like(band)
            for image in range(1, band.shape[0] - 1):
                for where, force in reversed(self._raw.get(image, [])):
                    if np.array_equal(where, band[image]):
                        raw[image] = force
                        break
            neb._record(np.asarray(energies, dtype=np.float64), raw, int(report.ci_index))
            self._projected = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
            neb._residuals(self._projected)

    NEBOptimizer.__qualname__ = "NEBOptimizer"
    return NEBOptimizer


def _constrained_direction(atoms: Any, vector: Any) -> Float64:
    """``vector`` with each constraint's force projection applied, as ASE masks a force."""
    direction = np.array(vector, dtype=np.float64, copy=True).reshape(-1, 3)
    for constraint in getattr(atoms, "constraints", []) or []:
        adjust = getattr(constraint, "adjust_forces", None)
        if adjust is not None:
            adjust(atoms, direction)
    return direction


def _unit(vector: Any) -> Float64:
    flat = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(flat))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("the eigenmode guess is zero on the free coordinates")
    return flat / norm


_DIMER_DEFAULTS: dict[str, Any] = {
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

# ASE parameters the rgsaddle session has no knob for. rgsaddle's
# MinModeTranslate steps with its own L-BFGS translation.
_DIMER_UNUSED = ("trial_angle", "trial_trans_step", "cg_translation", "extrapolate_forces")

# Before 0.3.3 MinModeAtoms took mode=, kind=, dr= and max_rotations=
# with these defaults.
_LEGACY_DIMER = {
    "dimer_separation": 1e-3,
    "max_num_rot": 10,
    "f_rot_min": 1e-4,
    "f_rot_max": 1.0,
    "maximum_translation": 0.2,
}


class DimerControl:
    """Dimer parameters under ``ase.mep.dimer.DimerControl``'s names and defaults.

    ``eigenmode_method`` is ``dimer`` or ``lanczos``. The session reads
    ``dimer_separation`` (half the dimer length), ``f_rot_min`` (the
    rotational force under which rotation stops), ``f_rot_max`` (above
    ``f_rot_min``: stop after one rotation once the rotational force is
    under 1, rgsaddle's fixed window), ``max_num_rot``,
    ``maximum_translation`` and ``use_central_forces`` (forward
    differences when true). ``displace`` reads the displacement keys.
    ``trial_angle``, ``trial_trans_step``, ``cg_translation`` and
    ``extrapolate_forces`` are stored and not used. ``order`` must be 1.
    Nothing is written to ``logfile``.
    """

    parameters: dict[str, Any]

    def __init__(
        self, logfile: Any = "-", eigenmode_logfile: Any = None, comm: Any = None, **kwargs: Any
    ) -> None:
        del comm
        self.parameters = dict(_DIMER_DEFAULTS)
        self.explicit: set[str] = set()
        for key, value in kwargs.items():
            self.set_parameter(key, value, log=False)
        self.logfile = logfile
        self.eigenmode_logfile = eigenmode_logfile
        self.counters = {"forcecalls": 0, "rotcount": 0, "optcount": 0}

    def set_parameter(self, parameter: str, value: Any, log: bool = True) -> None:
        del log
        if parameter not in self.parameters:
            raise ValueError(f"Invalid parameter >>{parameter}<< with value >>{value!s}<<")
        if parameter == "eigenmode_method" and str(value).lower() not in ("dimer", "lanczos"):
            raise ValueError("eigenmode_method must be dimer or lanczos")
        if parameter == "order" and int(value) != 1:
            raise ValueError("order must be 1; the index-k step has no Python session")
        self.parameters[parameter] = value
        self.explicit.add(parameter)

    def get_parameter(self, parameter: str) -> Any:
        if parameter not in self.parameters:
            raise ValueError(f"Invalid parameter >>{parameter}<<")
        return self.parameters[parameter]

    def get_logfile(self) -> Any:
        return self.logfile

    def get_eigenmode_logfile(self) -> Any:
        return self.eigenmode_logfile

    def get_counter(self, counter: str) -> int:
        return self.counters[counter]

    def increment_counter(self, counter: str) -> None:
        self.counters[counter] += 1

    def reset_counter(self, counter: str) -> None:
        self.counters[counter] = 0

    def reset_all_counters(self) -> None:
        for key in self.counters:
            self.counters[key] = 0


class _MinModeView:
    """ASE optimizable over the effective force; converged needs negative curvature."""

    def __init__(self, search: MinModeAtoms) -> None:
        self._search = search

    def get_x(self) -> Float64:
        return np.ascontiguousarray(self._search.atoms.get_positions(), dtype=np.float64).reshape(-1)

    def set_x(self, x: Float64) -> None:
        self._search.atoms.set_positions(np.asarray(x, dtype=np.float64).reshape(-1, 3))

    def get_gradient(self) -> Float64:
        return -np.ascontiguousarray(self._search.get_forces(), dtype=np.float64).reshape(-1)

    def get_value(self) -> float:
        return float(self._search.get_potential_energy())

    def ndofs(self) -> int:
        return 3 * len(self._search.atoms)

    def gradient_norm(self, gradient: Float64) -> float:
        return _max_atom_norm(gradient)

    def converged(self, gradient: Float64, fmax: float) -> bool:
        return self.gradient_norm(gradient) < fmax and self._search.get_curvature() < 0.0

    def iterimages(self) -> Any:
        return self._search.atoms.iterimages()


class MinModeAtoms:
    """One ASE ``Atoms`` object driven along its lowest mode, as ``ase.mep.dimer.MinModeAtoms``.

    ``MinModeAtoms(atoms, control, eigenmodes=None, random_seed=None)``
    takes a ``DimerControl`` (this module's or ASE's). The mode comes
    from one rgsaddle ``MinMode`` session that stays with the object, so
    the previous mode seeds the next rotation after a translation.
    ``get_forces()`` is the effective force: above the well (positive
    curvature) the force along the mode, reversed; below it the force
    with its component along the mode reflected. ``get_forces(real=True)``
    is ``Atoms.get_forces()``. A mode guess is projected onto the
    coordinates the constraints leave free. Other attributes, and
    ``get_original_*`` on the state at construction, pass through to the
    atoms, as in ASE.

    Before 0.3.3 the call was ``MinModeAtoms(atoms, mode, kind=, dr=,
    max_rotations=)``. That call still works: a mode in the second
    position warns, and without a control the old defaults hold
    (separation 1e-3, ten rotations, rotational tolerance 1e-4,
    translation cap 0.2). ``mode`` and ``curvature`` stay as the flat
    mode and the latest curvature.
    """

    def __init__(
        self,
        atoms: Any,
        control: Any = None,
        eigenmodes: Any = None,
        random_seed: int | None = None,
        comm: Any = None,
        *,
        mode: Any = None,
        kind: str | None = None,
        dr: float | None = None,
        max_rotations: int | None = None,
        **kwargs: Any,
    ) -> None:
        del comm
        if getattr(atoms, "calc", None) is None:
            raise ValueError("atoms has no calculator")
        if control is not None and not hasattr(control, "get_parameter"):
            warnings.warn(
                "the second argument of MinModeAtoms is a DimerControl, as in ASE; "
                "pass the mode as mode=",
                DeprecationWarning,
                stacklevel=2,
            )
            if mode is not None:
                raise TypeError("mode was given twice")
            mode, control = control, None
        logfile = kwargs.pop("logfile", None)
        mlogfile = kwargs.pop("eigenmode_logfile", None)
        if control is None:
            settings = dict(_LEGACY_DIMER)
            settings.update(kwargs)
            control = DimerControl(logfile=logfile, eigenmode_logfile=mlogfile, **settings)
            self._legacy = True
        else:
            for key, value in kwargs.items():
                control.set_parameter(key, value)
            self._legacy = False
        if kind is not None:
            control.set_parameter("eigenmode_method", kind)
        if dr is not None:
            control.set_parameter("dimer_separation", float(dr))
        if max_rotations is not None:
            control.set_parameter("max_num_rot", int(max_rotations))
        if int(control.get_parameter("order")) != 1:
            raise ValueError("order must be 1; the index-k step has no Python session")
        unused = sorted(
            name
            for name in set(getattr(control, "explicit", ())) & set(_DIMER_UNUSED)
            if control.get_parameter(name) != _DIMER_DEFAULTS[name]
        )
        if unused:
            warnings.warn(
                "rgsaddle's minimum-mode session does not read " + ", ".join(unused),
                UserWarning,
                stacklevel=2,
            )
        self.minmode_init = True
        self.atoms = atoms
        self.control = control
        self.order = 1
        if mode is not None:
            eigenmodes = [np.asarray(mode, dtype=np.float64).reshape(-1, 3)]
        self.eigenmodes: list[Float64] | None = (
            None if eigenmodes is None else [np.asarray(m, dtype=np.float64) for m in eigenmodes]
        )
        self.curvatures = [100.0]
        self.random_state = np.random.RandomState(random_seed)
        self.atoms0 = atoms.copy()
        self.save_original_forces()
        self.logfile = control.get_logfile()
        self.mlogfile = control.get_eigenmode_logfile()
        self.mode: Float64 | None = None
        self.curvature = float("nan")
        self.rotation_required = True
        self._session: Any = None
        self._force_tol = 0.05 if not self._legacy else 1e-3
        self._checked: Float64 | None = None
        self.forces0: Float64 | None = None
        self.energy0 = float("nan")

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") or attr in ("atoms", "atoms0"):
            raise AttributeError(attr)
        if "original" in attr.split("_"):
            return getattr(self.__dict__["atoms0"], attr.replace("_original_", "_"))
        return getattr(self.__dict__["atoms"], attr)

    def __len__(self) -> int:
        return len(self.atoms)

    def __ase_optimizable__(self) -> _MinModeView:
        return _MinModeView(self)

    @property
    def kind(self) -> str:
        return str(self.control.get_parameter("eigenmode_method")).lower()

    @property
    def dr(self) -> float:
        return float(self.control.get_parameter("dimer_separation"))

    @property
    def max_rotations(self) -> int:
        return int(self.control.get_parameter("max_num_rot"))

    def save_original_forces(self, force_calculation: bool = False) -> None:
        """Keep the energy and forces of the starting state on ``atoms0``."""
        calc = self.atoms.calc
        required = getattr(calc, "calculation_required", None)
        if (required is not None and not required(self.atoms, ["energy", "forces"])) or (
            force_calculation
        ):
            self.atoms0.calc = _ase("ase.calculators.singlepoint", "SinglePointCalculator")(
                self.atoms0,
                energy=self.atoms.get_potential_energy(),
                forces=self.atoms.get_forces(),
            )

    def get_control(self) -> Any:
        return self.control

    def get_atoms(self) -> Any:
        return self.atoms

    def set_atoms(self, atoms: Any) -> None:
        self.atoms = atoms
        self._session = None
        self._checked = None

    def get_curvature(self, order: Any = "max") -> float:
        if order == "max":
            return float(max(self.curvatures))
        return float(self.curvatures[int(order) - 1])

    def get_eigenmode(self, order: int = 1) -> Float64:
        if self.eigenmodes is None:
            raise RuntimeError("no eigenmode yet; call get_forces() first")
        return self.eigenmodes[order - 1]

    def set_eigenmode(self, eigenmode: Any, order: int = 1) -> None:
        mode = _unit(_constrained_direction(self.atoms, eigenmode))
        if self.eigenmodes is None:
            self.eigenmodes = [mode]
        else:
            self.eigenmodes[order - 1] = mode
        if self._session is not None:
            self._session.set_mode(np.ascontiguousarray(mode.reshape(-1)))
        self.rotation_required = True

    def set_curvature(self, curvature: float, order: int = 1) -> None:
        self.curvatures[order - 1] = float(curvature)

    def calculation_required(self) -> bool:
        if self.minmode_init or self._checked is None:
            return True
        return not np.array_equal(self._checked, self.atoms.get_positions())

    def initialize_eigenmodes(
        self, method: str | None = None, eigenmodes: Any = None, gauss_std: float | None = None
    ) -> None:
        """The first mode guess, as ASE makes it, projected onto the free coordinates."""
        if eigenmodes is None:
            pos = np.asarray(self.atoms.get_positions(), dtype=np.float64)
            old = np.asarray(self.atoms0.get_positions(), dtype=np.float64)
            chosen = method or str(self.control.get_parameter("initial_eigenmode_method"))
            if self._legacy and method is None:
                free = np.abs(_constrained_direction(self.atoms, np.ones_like(pos))).sum(axis=1)
                guess = np.zeros_like(pos)
                guess[int(np.argmax(free > 0.0)), 0] = 1.0
            elif chosen.lower() == "displacement" and (pos - old).any():
                guess = pos - old
            elif chosen.lower() == "gauss":
                self.displace(log=False, gauss_std=gauss_std, method="gauss")
                guess = np.asarray(self.atoms.get_positions(), dtype=np.float64) - pos
                self.atoms.set_positions(pos, apply_constraint=False)
            else:
                raise NotImplementedError(
                    "initial_eigenmode must use either 'gauss' or 'displacement', if the "
                    "latter is used the atoms must have moved away from the original "
                    f"positions. You have requested '{chosen}'."
                )
            eigenmodes = [guess]
        self.eigenmodes = [_unit(_constrained_direction(self.atoms, eigenmodes[0]))]

    def calculate_real_forces_and_energies(self, **kwargs: Any) -> None:
        if self.minmode_init:
            self.minmode_init = False
            self.initialize_eigenmodes(eigenmodes=self.eigenmodes)
        self.rotation_required = True
        self.forces0 = np.array(self.atoms.get_forces(**kwargs), dtype=np.float64, copy=True)
        self.energy0 = float(self.atoms.get_potential_energy())
        self.control.increment_counter("forcecalls")
        self._checked = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)

    def get_potential_energy(self) -> float:
        if self.calculation_required():
            self.calculate_real_forces_and_energies()
        return self.energy0

    def get_forces(self, real: bool = False, pos: Any = None, **kwargs: Any) -> Float64:
        """The effective force, or with ``real=True`` the force of ``Atoms``."""
        if self.calculation_required() and pos is None:
            self.calculate_real_forces_and_energies(**kwargs)
        if real and pos is None:
            assert self.forces0 is not None
            return np.array(self.forces0, dtype=np.float64, copy=True)
        if real:
            old = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)
            self.atoms.set_positions(pos)
            forces = np.array(self.atoms.get_forces(), dtype=np.float64, copy=True)
            self.control.increment_counter("forcecalls")
            self.atoms.set_positions(old)
            return forces
        if self.rotation_required:
            self.find_eigenmodes(order=self.order)
            self.rotation_required = False
            self.control.increment_counter("optcount")
        return self.get_projected_forces()

    def _session_options(self) -> dict[str, Any]:
        control = self.control
        f_min = float(control.get_parameter("f_rot_min"))
        f_max = float(control.get_parameter("f_rot_max"))
        return {
            "kind": str(control.get_parameter("eigenmode_method")).lower(),
            "dr": float(control.get_parameter("dimer_separation")),
            "force_tol": float(self._force_tol),
            "max_move": float(control.get_parameter("maximum_translation")),
            "rotation_tol": f_min,
            "max_rotations": int(control.get_parameter("max_num_rot")),
            "method": "lbfgs",
            "difference": "forward" if control.get_parameter("use_central_forces") else "central",
            "force_gate": "max_atom",
            "torque_window": 1 if f_max > f_min else 0,
        }

    def _point(self, positions: Float64) -> tuple[Float64, Float64]:
        """Session surface. The atoms keep the last geometry so its result stays cached."""
        row = np.ascontiguousarray(positions, dtype=np.float64).reshape(len(self.atoms), 3)
        self.atoms.set_positions(row, apply_constraint=False)
        energy = float(self.atoms.get_potential_energy())
        gradient = -np.ascontiguousarray(self.atoms.get_forces(), dtype=np.float64).reshape(1, -1)
        self.control.increment_counter("forcecalls")
        return np.array([energy], dtype=np.float64), gradient

    def _sync(self, search: Any) -> None:
        self.mode = np.asarray(search.mode(), dtype=np.float64)
        self.eigenmodes = [self.mode.reshape(-1, 3)]

    def _ensure_session(self) -> Any:
        if self._session is None:
            from rgsaddle._core import MinMode

            assert self.eigenmodes is not None
            x = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
            seed = np.ascontiguousarray(self.eigenmodes[0], dtype=np.float64).reshape(-1)
            self._session = MinMode(x, seed, **self._session_options())
        return self._session

    def find_eigenmodes(self, order: int = 1) -> None:
        """Rotate rgsaddle's session into the lowest mode at the current point."""
        del order
        assert self.forces0 is not None
        search = self._ensure_session()
        here = np.array(self.atoms.get_positions(), dtype=np.float64, copy=True)
        search.set_position(
            np.ascontiguousarray(here.reshape(-1)),
            np.ascontiguousarray(-self.forces0.reshape(-1)),
        )
        report = search.estimate(self._point)
        self.atoms.set_positions(here, apply_constraint=False)
        self._checked = here
        self.curvature = float(report.curvature)
        self.curvatures = [self.curvature]
        self.control.counters["rotcount"] = int(report.rotations)
        self._sync(search)

    def get_projected_forces(self, pos: Any = None) -> Float64:
        """ASE's projection: reverse along the mode above the well, reflect below it."""
        if pos is not None:
            forces = self.get_forces(real=True, pos=pos).copy()
        else:
            assert self.forces0 is not None
            forces = self.forces0.copy()
        assert self.eigenmodes is not None
        mode = self.eigenmodes[0]
        parallel = float(np.vdot(forces, mode)) * mode
        if self.get_curvature(order=1) > 0.0:
            return np.asarray(-parallel, dtype=np.float64)
        return np.asarray(forces - 2.0 * parallel, dtype=np.float64)

    def _set_force_tol(self, fmax: float) -> None:
        if float(fmax) != self._force_tol:
            self._force_tol = float(fmax)
            if self._session is not None:
                self._session = None
                self.rotation_required = True

    def translate(self) -> Any:
        """One translation of rgsaddle's session from the current point."""
        self.get_forces()
        search = self._ensure_session()
        report = search.step(self._point)
        x = np.asarray(search.position(), dtype=np.float64).reshape(len(self.atoms), 3)
        self.atoms.set_positions(x, apply_constraint=False)
        self._sync(search)
        return report

    def restore_original_positions(self) -> None:
        self.atoms.set_positions(self.atoms0.get_positions())

    def get_barrier_energy(self) -> float:
        """Energy now minus the energy of the starting state."""
        try:
            original = float(self.atoms0.get_potential_energy())
        except RuntimeError:
            warnings.warn(
                "The potential energy is not available, without further calculations, "
                "most likely at the original state.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return self.get_potential_energy() - original

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
        """Move the atoms away from their current position, as ASE's ``displace``."""
        del log
        control = self.control
        count = len(self.atoms)
        if mask is None:
            mask = control.get_parameter("mask")
        if method is None:
            method = str(control.get_parameter("displacement_method"))
        if gauss_std is None:
            gauss_std = control.get_parameter("gauss_std")
        if displacement_center is None:
            displacement_center = control.get_parameter("displacement_center")
        if radius is None:
            radius = control.get_parameter("displacement_radius")
        if number_of_atoms is None:
            number_of_atoms = control.get_parameter("number_of_displacement_atoms")
        if displacement_vector is not None and method.lower() != "vector":
            raise ValueError(
                f"displacement_vector was supplied but a different method ('{method}') was chosen."
            )
        if displacement_vector is None and method.lower() == "vector":
            raise ValueError(f"A displacement_vector must be supplied when using method = '{method}'.")
        if displacement_center is not None and radius is None and number_of_atoms is None:
            raise ValueError(
                "When displacement_center is chosen, either radius or number_of_atoms must be supplied."
            )
        positions = np.asarray(self.atoms.get_positions(), dtype=np.float64)
        centre_mask: list[bool] | None = None
        if displacement_center is not None:
            centre = displacement_center
            if isinstance(centre, (int, np.integer)):
                index = int(centre) % count
                distances = [
                    (k, float(self.atoms.get_distance(k, index, mic=mic))) for k in range(count)
                ]
            elif len(centre) == 3 and all(isinstance(c, float) for c in centre):
                distances = [
                    (k, float(np.linalg.norm(positions[k] - np.asarray(centre)))) for k in range(count)
                ]
            else:
                raise ValueError(
                    "displacement_center must be either the number of an atom in "
                    "MinModeAtoms object or a 3D position (3-tuple of floats)."
                )
            near = [d < radius for _, d in distances] if radius is not None else [True] * count
            if number_of_atoms is not None:
                nearest = [k for k, _ in sorted(distances, key=lambda item: item[1])][:number_of_atoms]
                closest = [k in nearest for k in range(count)]
            else:
                closest = [True] * count
            centre_mask = [near[k] and closest[k] for k in range(count)]
        if mask is None:
            mask = [True] * count
            if centre_mask is None:
                warnings.warn(
                    "It was not possible to figure out which atoms to displace, "
                    "Will try to displace all atoms.",
                    UserWarning,
                    stacklevel=2,
                )
        if centre_mask is not None:
            mask = [bool(mask[k]) and centre_mask[k] for k in range(count)]
        if displacement_vector is None:
            if method.lower() != "gauss":
                raise ValueError(f"Invalid displacement method >>{method}<<")
            std = gauss_std or control.get_parameter("gauss_std")
            vector = np.zeros((count, 3), dtype=np.float64)
            for k in range(count):
                if mask[k]:
                    vector[k] = [self.random_state.normal(0.0, std) for _ in range(3)]
        else:
            vector = np.array(displacement_vector, dtype=np.float64, copy=True).reshape(count, 3)
        for k in range(count):
            if not mask[k]:
                vector[k] = 0.0
        self.atoms.set_positions(positions + vector)

    def summarize(self) -> None:
        """Print the barrier, curvature, steps and force calls, as ASE does."""
        import sys

        out = self.logfile if hasattr(self.logfile, "write") else sys.stdout
        label = "MINMODE:SUMMARY: "
        text = label + "-------------------------\n"
        text += label + "Barrier: %16.4f\n" % self.get_barrier_energy()
        text += label + "Curvature: %14.4f\n" % self.get_curvature()
        text += label + "Optimizer steps: %8i\n" % self.control.get_counter("optcount")
        text += label + "Forcecalls: %13i\n" % self.control.get_counter("forcecalls")
        text += label + "-------------------------\n"
        out.write(text)


def _build_min_mode_translate() -> type:
    from ase.optimize.optimize import DEFAULT_MAX_STEPS, Optimizer

    class MinModeTranslate(Optimizer):
        """rgsaddle's minimum-mode translation behind ``ase.mep.dimer.MinModeTranslate``'s call.

        Each ``step`` is one step of the ``MinModeAtoms`` session: the
        mode found at this point, then rgsaddle's L-BFGS translation on
        the effective force, or one ``maximum_translation`` step along
        the mode while the curvature is not negative. ``run(fmax)``
        stops when the largest per-atom effective force is under
        ``fmax`` and the curvature is negative, as ASE's does.
        """

        def __init__(
            self,
            dimeratoms: MinModeAtoms,
            logfile: Any = "-",
            trajectory: Any = None,
            **kwargs: Any,
        ) -> None:
            if not isinstance(dimeratoms, MinModeAtoms):
                raise TypeError("MinModeTranslate drives an rgsaddle.ase.MinModeAtoms")
            self.dimeratoms = dimeratoms
            self.control = dimeratoms.get_control()
            self.report: Any = None
            atoms: Any = dimeratoms
            super().__init__(atoms, None, logfile, trajectory, **kwargs)

        def initialize(self) -> None:
            return None

        def read(self) -> None:
            raise NotImplementedError("MinModeTranslate does not read a restart file")

        def irun(self, fmax: float = 0.05, steps: int = DEFAULT_MAX_STEPS) -> Any:  # type: ignore[override]
            self.dimeratoms._set_force_tol(fmax)
            base: Any = super()
            return base.irun(fmax=fmax, steps=steps)

        def run(self, fmax: float = 0.05, steps: int = DEFAULT_MAX_STEPS) -> bool:  # type: ignore[override]
            self.dimeratoms._set_force_tol(fmax)
            base: Any = super()
            return bool(base.run(fmax=fmax, steps=steps))

        def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
            self.report = self.dimeratoms.translate()

    MinModeTranslate.__qualname__ = "MinModeTranslate"
    return MinModeTranslate


def __getattr__(name: str) -> Any:
    # ASE is imported when one of these classes is first used.
    if name == "NEBOptimizer":
        built = _build_neb_optimizer()
    elif name == "MinModeTranslate":
        built = _build_min_mode_translate()
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = built
    return built


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
