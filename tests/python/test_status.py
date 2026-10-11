"""F7 mapping of rgsaddle_status_t."""

from __future__ import annotations

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


def test_ok_returns_true() -> None:
    set_last_error("")
    assert rgsaddle.raise_status(OK) is True


def test_value_and_numerical_and_runtime() -> None:
    for code in sorted(VALUE):
        set_last_error("")
        try:
            rgsaddle.raise_status(code)
        except ValueError as exc:
            assert rgsaddle.last_error() == str(exc)
            assert rgsaddle.last_error() != ""
        else:
            raise AssertionError(f"{code} must be ValueError")
    for code in NUMERICAL:
        set_last_error("")
        try:
            rgsaddle.raise_status(code)
        except NumericalError as exc:
            assert issubclass(NumericalError, ArithmeticError)
            assert str(exc) == "NON_FINITE"
            assert rgsaddle.last_error() == "NON_FINITE"
        else:
            raise AssertionError("NON_FINITE must be NumericalError")
    for code in RUNTIME:
        set_last_error("")
        try:
            rgsaddle.raise_status(code)
        except RuntimeError as exc:
            assert rgsaddle.last_error() == str(exc)
        else:
            raise AssertionError(f"{code} must be RuntimeError")


def test_last_error_is_the_message() -> None:
    set_last_error("singular update")
    try:
        rgsaddle.raise_status(NON_FINITE)
    except NumericalError as exc:
        assert str(exc) == "singular update"
    else:
        raise AssertionError("expected NumericalError")
    assert rgsaddle.last_error() == "singular update"


def test_surface_failed_without_a_python_error_is_runtime() -> None:
    set_last_error("callback failed")
    try:
        rgsaddle.raise_status(SURFACE_FAILED)
    except RuntimeError as exc:
        assert str(exc) == "callback failed"
    else:
        raise AssertionError("expected RuntimeError")
