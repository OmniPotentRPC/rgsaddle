"""F6 round trips and the F7 status table."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import rgsaddle
from rgsaddle import NumericalError
from rgsaddle._core import ClaimSlot, drive_eval, echo_f64

VALUES = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.just(-0.0),
    st.just(math.nan),
    st.just(math.ldexp(1.0, -1074)),
    st.just(-math.ldexp(1.0, -1074)),
)


@settings(max_examples=30)
@given(values=st.lists(VALUES, min_size=0, max_size=8))
def test_echo_round_trip(values: list[float]) -> None:
    src = np.ascontiguousarray(values, dtype=np.float64)
    out = np.asarray(echo_f64(src))
    assert out.dtype == np.float64
    assert out.flags.c_contiguous
    assert out.shape == src.shape
    assert out.tobytes() == src.tobytes()


def test_echo_rejects_dtype_and_order() -> None:
    with pytest_raises_type():
        echo_f64(np.array([1, 2, 3], dtype=np.int64))
    fortran = np.asfortranarray(np.ones((2, 2), dtype=np.float64))
    with pytest_raises_type():
        echo_f64(fortran)


def test_drive_eval_shape() -> None:
    slot = ClaimSlot()

    def callback(ids: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(ids.shape[0]), np.zeros_like(x)

    ids = np.array([0], dtype=np.int64)
    x = np.zeros((2, 3), dtype=np.float64)
    owners = np.array([1], dtype=np.uint64)
    flags = np.array([0], dtype=np.uint8)
    try:
        drive_eval(callback, ids, x, owners, flags, 1, slot)
    except ValueError as exc:
        assert "x:" in str(exc)
        return
    raise AssertionError("shape mismatch must raise")


@settings(max_examples=20)
@given(code=st.integers(min_value=-20, max_value=20))
def test_status_class(code: int) -> None:
    rgsaddle.set_last_error("")
    if code == 0:
        assert rgsaddle.raise_status(code) is True
        return
    expected: type[BaseException]
    if code in (-5, -11):
        expected = ValueError
    elif code == -7:
        expected = NumericalError
    else:
        expected = RuntimeError
    try:
        rgsaddle.raise_status(code)
    except expected:
        return
    raise AssertionError(f"{code} did not raise {expected.__name__}")


class _TypeErrorBox:
    pass


def pytest_raises_type():
    import pytest

    return pytest.raises(TypeError)
