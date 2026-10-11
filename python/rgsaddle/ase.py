# MIT License
# Copyright 2026 Rohit Goswami
"""ASE image list as the stencil surface callback.

ASE is imported on first use. ``import rgsaddle`` does not import it.

The saddle step that will consume this callback is the one in
Goswami, Gunde, and Jónsson, Front. Chem. 2026
(doi:10.3389/fchem.2026.1807063). Calculator calls follow
Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017)
(doi:10.1088/1361-648X/aa680e): ``get_forces()`` applies constraints,
and ``get_forces(apply_constraint=False)`` is the raw force.
"""

from __future__ import annotations

import threading
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rgsaddle._core import ClaimSlot, drive_eval

Float64 = NDArray[np.float64]
Int64 = NDArray[np.int64]


class SurfaceAdapter:
    """Map ``list[Atoms]`` onto ``(ids, x) -> (energies, gradients)``.

    Image ids are the list positions at construction. ``x`` has shape
    ``(rows, 3 * n_atoms)``. Gradients are minus ``get_forces()`` after
    constraints. ``raw_forces`` keeps the unconstrained forces for the
    same rows, in request order.

    Rows that share one calculator object run on one worker in ascending
    image id (F5). Distinct calculators run concurrently up to
    ``threads``. A second call on the same adapter raises
    ``RuntimeError`` while the first call still holds the claim (F3).
    On a GIL build the Python callbacks serialise. ``threads`` still
    selects the worker count. It does not change the bytes that come back.
    """

    def __init__(
        self,
        images: list[Any],
        *,
        allow_shared_calculator: bool = False,
        threads: int = 1,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be >= 1")
        if len(images) == 0:
            raise ValueError("images must be non-empty")
        self.threads = threads
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

    def __call__(self, ids: Int64, x: Float64) -> tuple[Float64, Float64]:
        """Evaluate the requested rows.

        ``ids`` is int64, shape ``(rows,)``. ``x`` is float64,
        shape ``(rows, dof)``. The result arrays are new copies (F6).
        """
        ids_arr = np.ascontiguousarray(ids, dtype=np.int64)
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if ids_arr.ndim != 1:
            raise ValueError(f"ids: expected rank 1, got rank {ids_arr.ndim}")
        if x_arr.ndim != 2:
            raise ValueError(f"x: expected rank 2, got rank {x_arr.ndim}")
        n = int(ids_arr.shape[0])
        dof = self._natoms * 3
        if x_arr.shape != (n, dof):
            raise ValueError(f"x: expected shape ({n}, {dof}), got {tuple(int(v) for v in x_arr.shape)}")
        owners = np.empty(n, dtype=np.uint64)
        reentrant = np.empty(n, dtype=np.uint8)
        for row, image_id in enumerate(ids_arr.tolist()):
            atoms = self._images.get(int(image_id))
            if atoms is None:
                raise ValueError(f"unknown image id {int(image_id)}")
            calc = atoms.calc
            owners[row] = np.uint64(id(calc))
            reentrant[row] = np.uint8(1 if getattr(calc, "reentrant", False) else 0)
        raw = np.empty((n, dof), dtype=np.float64)
        placement = {int(image_id): row for row, image_id in enumerate(ids_arr.tolist())}
        write_lock = threading.Lock()

        def on_group(group_ids: Int64, group_x: Float64) -> tuple[Float64, Float64]:
            energies, gradients, raw_group = self._eval_group(group_ids, group_x)
            with write_lock:
                for local, image_id in enumerate(group_ids.tolist()):
                    raw[placement[int(image_id)]] = raw_group[local]
            return energies, gradients

        out = drive_eval(on_group, ids_arr, x_arr, owners, reentrant, self.threads, self._slot)
        energies = cast(Float64, out[0])
        gradients = cast(Float64, out[1])
        self.raw_forces = np.array(raw, dtype=np.float64, copy=True)
        return energies, gradients

    def _eval_group(self, ids: Int64, x: Float64) -> tuple[Float64, Float64, Float64]:
        """Evaluate one calculator's rows, already in ascending id order."""
        n = int(ids.shape[0])
        dof = int(x.shape[1])
        energies = np.empty(n, dtype=np.float64)
        gradients = np.empty((n, dof), dtype=np.float64)
        raw = np.empty((n, dof), dtype=np.float64)
        for local, image_id in enumerate(ids.tolist()):
            atoms = self._images[int(image_id)]
            coords = np.array(x[local], dtype=np.float64, copy=True).reshape((self._natoms, 3))
            # The stencil asked for this geometry. Force constraints are
            # applied below. Position constraints must not move the row
            # off ``x``.
            atoms.set_positions(coords, apply_constraint=False)
            energies[local] = float(atoms.get_potential_energy())
            raw_force = np.array(atoms.get_forces(apply_constraint=False), dtype=np.float64, copy=True)
            constrained = np.array(atoms.get_forces(), dtype=np.float64, copy=True)
            raw[local] = raw_force.reshape(dof)
            gradients[local] = -constrained.reshape(dof)
        return energies, gradients, raw
