# MIT License
# Copyright 2026 Rohit Goswami
"""ASE optimizer classes that need ASE at import time.

``rgsaddle.ase`` loads this module on first use of ``MinModeTranslate``
or ``MinModeAtoms.__ase_optimizable__``, so ``import rgsaddle.ase`` does
not import ASE.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import numpy as np
from ase.optimize.optimize import OptimizableAtoms, Optimizer
from numpy.typing import NDArray

from rgsaddle.ase import MinModeAtoms

Float64 = NDArray[np.float64]


class _MinModeView(OptimizableAtoms):
    """ASE optimizable view of a ``MinModeAtoms``.

    The gradient is the negative effective force. With a ``DimerControl``
    attached, convergence also needs a negative curvature.
    """

    def __init__(self, search: MinModeAtoms) -> None:
        self.atoms = search.atoms
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

    def gradient_norm(self, gradient: Float64) -> float:
        forces = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        return float(np.sqrt((forces * forces).sum(axis=1)).max())

    def converged(self, gradient: Float64, fmax: float) -> bool:
        forces_ok = self.gradient_norm(gradient) < fmax
        if self._search.control is None:
            return forces_ok
        return forces_ok and self._search.curvature < 0.0


class MinModeTranslate(Optimizer):
    """One Rust minimum-mode step per optimizer iteration.

    The constructor matches ``ase.mep.dimer.MinModeTranslate``:
    ``dimeratoms`` is a ``MinModeAtoms``, then the logfile and the
    trajectory. Each step is one translation of the core's minimum-mode
    session on the effective force, with the optimizer that the
    ``MinModeAtoms`` names in ``method`` (L-BFGS by default, FIRE with
    ``method="fire"``). It is not ASE's Newton line search. Convergence
    is the effective force under ``fmax`` and, when a ``DimerControl`` is
    attached, a negative curvature.
    """

    def __init__(
        self,
        dimeratoms: MinModeAtoms,
        logfile: IO[Any] | str | Path | None = "-",
        trajectory: str | Path | None = None,
    ) -> None:
        if not isinstance(dimeratoms, MinModeAtoms):
            raise TypeError("MinModeTranslate drives a MinModeAtoms object")
        Optimizer.__init__(self, dimeratoms.atoms, logfile=logfile, trajectory=trajectory)
        self.optimizable = dimeratoms.__ase_optimizable__()
        self.dimeratoms = dimeratoms
        self.control = dimeratoms.get_control()

    def initialize(self) -> None:
        return None

    def read(self) -> None:
        return None

    def step(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        fmax = self.fmax
        self.dimeratoms.translate(0.05 if fmax is None else float(fmax))
