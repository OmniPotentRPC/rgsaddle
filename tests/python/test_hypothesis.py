"""F6 round trips and the F7 status table."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import rgsaddle
from rgsaddle import NumericalError
from rgsaddle._core import ClaimSlot
from rgsaddle._core._testing import drive_eval, echo_f64, set_last_error

VALUES = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.just(-0.0),
    st.just(math.nan),
    st.just(math.ldexp(1.0, -1074)),
    st.just(-math.ldexp(1.0, -1074)),
)


def _callback(ids: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(ids.shape[0], dtype=np.float64), np.zeros(x.shape, dtype=np.float64)


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
    with pytest.raises(TypeError):
        echo_f64(np.array([1, 2, 3], dtype=np.int64))
    fortran = np.asfortranarray(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(TypeError):
        echo_f64(fortran)


def test_drive_eval_shape() -> None:
    slot = ClaimSlot()
    ids = np.array([0], dtype=np.int64)
    x = np.zeros((2, 3), dtype=np.float64)
    owners = np.array([1], dtype=np.uint64)
    flags = np.array([0], dtype=np.uint8)
    with pytest.raises(ValueError, match=r"x:") as exc:
        drive_eval(_callback, ids, x, owners, flags, 1, slot)
    text = str(exc.value)
    assert "(1, 3)" in text
    assert "(2, 3)" in text


def test_callback_dtype_and_order_name_the_array() -> None:
    slot = ClaimSlot()
    ids = np.array([0, 1], dtype=np.int64)
    x = np.zeros((2, 3), dtype=np.float64)
    # One owner keeps both rows in a single callback, so the gradient
    # matrix has two columns of extent and Fortran order is visible.
    owners = np.array([7, 7], dtype=np.uint64)
    flags = np.zeros(2, dtype=np.uint8)

    def bad_energy(group_ids: np.ndarray, group_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(group_ids.shape[0], dtype=np.float32), np.zeros_like(group_x)

    def bad_order(group_ids: np.ndarray, group_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gradients = np.asfortranarray(np.zeros_like(group_x))
        return np.zeros(group_ids.shape[0], dtype=np.float64), gradients

    with pytest.raises(TypeError, match="energies"):
        drive_eval(bad_energy, ids, x, owners, flags, 1, slot)
    with pytest.raises(TypeError, match="gradients"):
        drive_eval(bad_order, ids, x, owners, flags, 1, slot)


@settings(max_examples=20)
@given(values=st.lists(st.integers(min_value=0, max_value=6), min_size=1, max_size=4))
def test_int32_ids_raise_type_error(values: list[int]) -> None:
    ids = np.array(values, dtype=np.int32)
    x = np.zeros((len(values), 1), dtype=np.float64)
    owners = np.zeros(len(values), dtype=np.uint64)
    flags = np.zeros(len(values), dtype=np.uint8)
    with pytest.raises(TypeError):
        drive_eval(_callback, ids, x, owners, flags, 1, ClaimSlot())


@settings(max_examples=20)
@given(
    rows=st.integers(min_value=0, max_value=4),
    cols=st.integers(min_value=1, max_value=4),
    bad_rows=st.integers(min_value=0, max_value=6),
)
def test_shape_error_names_both_shapes(rows: int, cols: int, bad_rows: int) -> None:
    if bad_rows == rows:
        return
    ids = np.arange(rows, dtype=np.int64)
    x = np.zeros((bad_rows, cols), dtype=np.float64)
    owners = np.arange(rows, dtype=np.uint64)
    flags = np.zeros(rows, dtype=np.uint8)
    with pytest.raises(ValueError) as exc:
        drive_eval(_callback, ids, x, owners, flags, 1, ClaimSlot())
    text = str(exc.value)
    assert f"({rows}, {cols})" in text
    assert f"({bad_rows}, {cols})" in text


@settings(max_examples=20)
@given(code=st.integers(min_value=-20, max_value=20))
def test_status_class(code: int) -> None:
    set_last_error("")
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
    with pytest.raises(expected):
        rgsaddle.raise_status(code)
