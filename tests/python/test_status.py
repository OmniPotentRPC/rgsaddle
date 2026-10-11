"""F7 mapping of rgsaddle_status_t."""

from __future__ import annotations

import pytest

import rgsaddle
from rgsaddle import NumericalError
from rgsaddle._core._testing import set_last_error

# Codes from include/rgsaddle.h. Convergence is a run report, not a status.
OK = 0
NULL_SESSION = -1
NULL_BAND = -2
NULL_SURFACE = -3
NULL_REPORT = -4
SHAPE = -5
SURFACE_FAILED = -6
NON_FINITE = -7
SOLVER = -8
ABI_MISMATCH = -9
ALLOC = -10
INVALID_PARAMETER = -11
NO_EVALUATION = -12

VALUE = {SHAPE, INVALID_PARAMETER}
NUMERICAL = {NON_FINITE}
RUNTIME = {
    NULL_SESSION,
    NULL_BAND,
    NULL_SURFACE,
    NULL_REPORT,
    SURFACE_FAILED,
    SOLVER,
    ABI_MISMATCH,
    ALLOC,
    NO_EVALUATION,
    99,
}

NAMES = {
    OK: "OK",
    NULL_SESSION: "NULL_SESSION",
    NULL_BAND: "NULL_BAND",
    NULL_SURFACE: "NULL_SURFACE",
    NULL_REPORT: "NULL_REPORT",
    SHAPE: "SHAPE",
    SURFACE_FAILED: "SURFACE_FAILED",
    NON_FINITE: "NON_FINITE",
    SOLVER: "SOLVER",
    ABI_MISMATCH: "ABI_MISMATCH",
    ALLOC: "ALLOC",
    INVALID_PARAMETER: "INVALID_PARAMETER",
    NO_EVALUATION: "NO_EVALUATION",
    99: "UNKNOWN",
}


def test_ok_returns_true() -> None:
    assert rgsaddle.raise_status(OK) is True
    assert rgsaddle.last_error() == ""


def test_value_and_numerical_and_runtime() -> None:
    for code in sorted(VALUE):
        with pytest.raises(ValueError, match=f"^{NAMES[code]}$"):
            rgsaddle.raise_status(code)
        assert rgsaddle.last_error() == ""
    for code in NUMERICAL:
        with pytest.raises(NumericalError, match=f"^{NAMES[code]}$"):
            rgsaddle.raise_status(code)
        assert issubclass(NumericalError, ArithmeticError)
        assert rgsaddle.last_error() == ""
    for code in sorted(RUNTIME):
        with pytest.raises(RuntimeError, match=f"^{NAMES[code]}$"):
            rgsaddle.raise_status(code)
        assert rgsaddle.last_error() == ""


def test_back_to_back_status_uses_each_name() -> None:
    with pytest.raises(ValueError, match="^SHAPE$"):
        rgsaddle.raise_status(SHAPE)
    assert rgsaddle.last_error() == ""
    with pytest.raises(NumericalError, match="^NON_FINITE$"):
        rgsaddle.raise_status(NON_FINITE)
    assert rgsaddle.last_error() == ""
    assert rgsaddle.raise_status(OK) is True
    assert rgsaddle.last_error() == ""


def test_last_error_is_the_message() -> None:
    set_last_error("singular update")
    with pytest.raises(NumericalError, match="^singular update$"):
        rgsaddle.raise_status(NON_FINITE)
    assert rgsaddle.last_error() == ""
    with pytest.raises(NumericalError, match="^NON_FINITE$"):
        rgsaddle.raise_status(NON_FINITE)
    assert rgsaddle.last_error() == ""


def test_surface_failed_without_a_python_error_is_runtime() -> None:
    set_last_error("callback failed")
    with pytest.raises(RuntimeError, match="^callback failed$"):
        rgsaddle.raise_status(SURFACE_FAILED)
    assert rgsaddle.last_error() == ""
