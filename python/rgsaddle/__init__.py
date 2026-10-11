# MIT License
# Copyright 2026 Rohit Goswami
"""Status codes and enum controls for the ASE image surface.

The extension module is ``rgsaddle._core``. Importing this package does
not import ASE. Use ``rgsaddle.ase`` for the image adapter.
"""

from __future__ import annotations

from enum import IntEnum

from rgsaddle._core import (
    NumericalError as NumericalError,
    Stencil as Stencil,
    StencilReport as StencilReport,
    last_error as last_error,
    raise_status as raise_status,
    version as _version,
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


class MinModeKind(IntEnum):
    """Lowest-mode estimator."""

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


class StencilPreset(IntEnum):
    """Named stencil recipe. Values match the C preset ordinals."""

    CI_NEB = 0
    EON_ZOOM = 1
    ORCA_ZOOM_NEB_TS = 2
    AUTONEB = 3
    DIMER = 4
    LANCZOS = 5
    ART = 6
    PLAIN = 7


class ZoomEnds(IntEnum):
    """Window ends. Fixed is the eOn default."""

    FIXED = 0
    FREE_PERP = 1


class BandMethod(IntEnum):
    """Band method ordinals. Point methods continue the same sequence."""

    PLAIN = 0
    CI_NEB = 1
    AUTONEB = 2
    ZOOM = 3
    DIMER = 4
    LANCZOS = 5
    ART = 6


class SpringModel(IntEnum):
    """Spring law. Energy-weighted is off unless this value is set."""

    UNIFORM = 0
    ENERGY_WEIGHTED = 1
    ONSAGER_MACHLUP = 2
    EXPLICIT = 3


class ZoomWindowMode(IntEnum):
    """How the zoom window is chosen."""

    AUTO = 0
    MANUAL = 1


class Interp(IntEnum):
    """Pack inside the zoom window."""

    LINEAR = 0
    CUBIC = 1


class Refine(IntEnum):
    """What follows the band. OCI and eigenvector following are refused here."""

    NONE = 0
    OCINEB = 1
    EIGENVECTOR_FOLLOWING = 2


class StencilPhase(IntEnum):
    """Which stencil step just ran."""

    BAND = 0
    ZOOM = 1
    INSERT = 2
    ALIGN = 3
    MINMODE = 4
    EF = 5
    ROTATE = 6
    TRANSLATE = 7


__all__ = [
    "BandMethod",
    "CellChart",
    "EigenDevice",
    "ExpandKind",
    "ForceGate",
    "HessUpdate",
    "Interp",
    "IrcKind",
    "MinModeKind",
    "NumericalError",
    "Refine",
    "RestrictedKind",
    "SpringModel",
    "Stencil",
    "StencilPhase",
    "StencilPreset",
    "StencilReport",
    "ZoomEnds",
    "ZoomWindowMode",
    "__version__",
    "last_error",
    "raise_status",
]
