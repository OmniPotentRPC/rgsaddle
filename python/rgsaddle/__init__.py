# MIT License
# Copyright 2026 Rohit Goswami
"""Status codes and enum controls for the ASE image surface.

The extension module is ``rgsaddle._core``. Importing this package does
not import ASE. Use ``rgsaddle.ase`` for the image adapter.
"""

from __future__ import annotations

from enum import IntEnum

from rgsaddle._core import (
    Band as Band,
    ConnectReport as ConnectReport,
    GrowingString as GrowingString,
    LinebOrbit as LinebOrbit,
    LinebResult as LinebResult,
    NumericalError as NumericalError,
    SaddleIndex as SaddleIndex,
    SessionReport as SessionReport,
    Stencil as Stencil,
    StencilReport as StencilReport,
    VineyardReport as VineyardReport,
    atomic_mass as atomic_mass,
    connect as connect,
    THERMODYNAMIC_DERIVATIVE as THERMODYNAMIC_DERIVATIVE,
    hyperplane_thermodynamic_integration as hyperplane_thermodynamic_integration,
    idpp_path as idpp_path,
    last_error as last_error,
    lineb_solve as lineb_solve,
    raise_status as raise_status,
    reparametrize as reparametrize,
    saddle_index as saddle_index,
    version as _version,
    vineyard_prefactor_fd as vineyard_prefactor_fd,
)

__version__ = _version()

# Names and values match the previous module.


class ForceGate(IntEnum):
    """eOn / gpr force norm. The wire values are the discriminants."""

    L2_NORM = 0
    LINF_NORM = 1
    MAX_FORCE_ON_ATOM = 2


class HessUpdate(IntEnum):
    """Sella Hessian update."""

    BFGS = 0
    TS_BFGS = 1


class IrcKind(IntEnum):
    """Outer IRC increment."""

    GS2 = 0
    MOROKUMA = 1
    SELLA_SPHERE = 2


class MinModeKind(IntEnum):
    """Python lowest-mode estimators; index-k LOBPCG is C++/C/Rust-only."""

    DIMER = 0
    LANCZOS = 1


class EigenDevice(IntEnum):
    """Dense eigen device. DLPK is the vecops waist."""

    HOST = 0
    DLPK = 1


class ExpandKind(IntEnum):
    """Sella subspace expansion."""

    LANCZOS = 0
    GD = 1
    JD0 = 2
    JD0_ALT = 3
    MJD0 = 4
    MJD0_ALT = 5


class CellChart(IntEnum):
    """How the nine lattice numbers enter the packed PES."""

    ENTRIES = 0
    LOG_DEFORM = 1


class RestrictedKind(IntEnum):
    """Sella restricted step."""

    TRUST_REGION = 0
    MAX_INTERNAL_STEP = 1
    RESTRICTED_ATOMIC_STEP = 2


__all__ = [
    "Band",
    "CellChart",
    "ConnectReport",
    "EigenDevice",
    "ExpandKind",
    "ForceGate",
    "GrowingString",
    "HessUpdate",
    "IrcKind",
    "LinebOrbit",
    "LinebResult",
    "MinModeKind",
    "NumericalError",
    "RestrictedKind",
    "SaddleIndex",
    "SessionReport",
    "Stencil",
    "StencilReport",
    "THERMODYNAMIC_DERIVATIVE",
    "VineyardReport",
    "__version__",
    "atomic_mass",
    "connect",
    "hyperplane_thermodynamic_integration",
    "idpp_path",
    "last_error",
    "lineb_solve",
    "raise_status",
    "reparametrize",
    "saddle_index",
    "vineyard_prefactor_fd",
]
