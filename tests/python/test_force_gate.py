"""Maturin-generated ForceGate is a real enum, not a ctypes int pile."""

from __future__ import annotations

import enum

from rgsaddle import ForceGate


def test_force_gate_is_an_enum():
    assert issubclass(ForceGate, enum.Enum)
    assert ForceGate.L2_NORM == 0
    assert ForceGate.LINF_NORM == 1
    assert ForceGate.MAX_FORCE_ON_ATOM == 2
    assert int(ForceGate.MAX_FORCE_ON_ATOM) == 2
    assert ForceGate(2) == ForceGate.MAX_FORCE_ON_ATOM


def test_unknown_ordinal_is_refused():
    try:
        ForceGate(99)
    except ValueError:
        return
    raise AssertionError("ForceGate(99) must raise")
