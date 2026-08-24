"""Maturin-generated ForceGate is a real enum, not a ctypes int pile."""

from __future__ import annotations

import enum

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


def test_force_gate_is_an_enum():
    assert issubclass(ForceGate, enum.IntEnum)
    assert ForceGate.L2_NORM == 0
    assert ForceGate.LINF_NORM == 1
    assert ForceGate.MAX_FORCE_ON_ATOM == 2
    assert int(ForceGate.MAX_FORCE_ON_ATOM) == 2
    assert ForceGate(2) is ForceGate.MAX_FORCE_ON_ATOM


def test_unknown_ordinal_is_refused():
    try:
        ForceGate(99)
    except ValueError:
        return
    raise AssertionError("ForceGate(99) must raise")


def test_hess_update_is_an_enum():
    assert issubclass(HessUpdate, enum.IntEnum)
    assert HessUpdate.BFGS == 0
    assert HessUpdate.TS_BFGS == 1
    assert HessUpdate(1) is HessUpdate.TS_BFGS


def test_irc_and_minmode_match_the_c_wire():
    assert issubclass(IrcKind, enum.IntEnum)
    assert IrcKind.GS2 == 0
    assert IrcKind.MOROKUMA == 1
    assert MinModeKind.DIMER == 0
    assert MinModeKind.LANCZOS == 1
    try:
        IrcKind(99)
    except ValueError:
        return
    raise AssertionError("IrcKind(99) must raise")


def test_eigen_device_is_an_enum():
    assert issubclass(EigenDevice, enum.IntEnum)
    assert EigenDevice.HOST == 0
    assert EigenDevice.DLPK == 1


def test_expand_kind_is_an_enum():
    assert issubclass(ExpandKind, enum.IntEnum)
    assert ExpandKind.LANCZOS == 0
    assert ExpandKind.GD == 1
    assert ExpandKind.JD0 == 2
    assert ExpandKind.JD0_ALT == 3
    assert ExpandKind.MJD0 == 4
    assert ExpandKind.MJD0_ALT == 5
    assert ExpandKind(2) is ExpandKind.JD0


def test_restricted_kind_is_an_enum():
    assert issubclass(RestrictedKind, enum.IntEnum)
    assert RestrictedKind.TRUST_REGION == 0
    assert RestrictedKind.MAX_INTERNAL_STEP == 1
    assert RestrictedKind(1) is RestrictedKind.MAX_INTERNAL_STEP


def test_cell_chart_is_an_enum():
    assert issubclass(CellChart, enum.IntEnum)
    assert CellChart.ENTRIES == 0
    assert CellChart.LOG_DEFORM == 1
    assert CellChart(1) is CellChart.LOG_DEFORM
