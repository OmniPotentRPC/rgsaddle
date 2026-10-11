"""Enum names and values match the PyO3 module on main.

The table is the module built from acf239252b82741c6a635b8096c5bc47c8b70eeb.
"""

from __future__ import annotations

import enum
import subprocess
import sys

import rgsaddle
from rgsaddle import (
    CellChart,
    EigenDevice,
    ExpandKind,
    ForceGate,
    HessUpdate,
    IrcKind,
    MinModeKind,
    RestrictedKind,
)

# member name -> discriminant, in declaration order.
TABLE: dict[type[enum.IntEnum], list[tuple[str, int]]] = {
    ForceGate: [("L2_NORM", 0), ("LINF_NORM", 1), ("MAX_FORCE_ON_ATOM", 2)],
    HessUpdate: [("BFGS", 0), ("TS_BFGS", 1)],
    IrcKind: [("GS2", 0), ("MOROKUMA", 1)],
    MinModeKind: [("DIMER", 0), ("LANCZOS", 1)],
    EigenDevice: [("HOST", 0), ("DLPK", 1)],
    ExpandKind: [
        ("LANCZOS", 0),
        ("GD", 1),
        ("JD0", 2),
        ("JD0_ALT", 3),
        ("MJD0", 4),
        ("MJD0_ALT", 5),
    ],
    CellChart: [("ENTRIES", 0), ("LOG_DEFORM", 1)],
    RestrictedKind: [
        ("TRUST_REGION", 0),
        ("MAX_INTERNAL_STEP", 1),
        ("RESTRICTED_ATOMIC_STEP", 2),
    ],
}


def test_version() -> None:
    assert rgsaddle.__version__ == "0.2.2"
    from rgsaddle import _core

    assert _core.version() == rgsaddle.__version__
    assert _core.__version__ == rgsaddle.__version__


def test_names_and_values() -> None:
    for cls, members in TABLE.items():
        assert issubclass(cls, enum.IntEnum)
        assert [(item.name, int(item)) for item in cls] == members
        for name, value in members:
            assert getattr(cls, name) == value
            assert cls(value) is getattr(cls, name)


def test_unknown_ordinal_is_refused() -> None:
    for cls in TABLE:
        try:
            cls(99)
        except ValueError:
            continue
        raise AssertionError(f"{cls.__name__}(99) must raise")


def test_import_does_not_pull_ase() -> None:
    script = "import sys\nimport rgsaddle\nraise SystemExit(0 if 'ase' not in sys.modules else 1)\n"
    run = subprocess.run([sys.executable, "-c", script], check=False)
    assert run.returncode == 0
