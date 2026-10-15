"""Index-1 Newton on a quadratic saddle, without the extension module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _tsopt():
    path = Path(__file__).resolve().parents[2] / "python" / "rgsaddle" / "tsopt.py"
    spec = importlib.util.spec_from_file_location("rgsaddle.tsopt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rgsaddle.tsopt"] = module
    spec.loader.exec_module(module)
    return module


def test_quadratic_saddle_is_index_1():
    tsopt = _tsopt()

    def energy_grad(x):
        return float(-x[0] * x[0] + x[1] * x[1]), np.array([-2.0 * x[0], 2.0 * x[1]])

    def projected(_x):
        return np.eye(2), np.diag([-2.0, 2.0])

    result = tsopt.refine(
        np.array([0.4, -0.3]),
        energy_grad,
        projected,
        kind="default",
        cap=40,
    )
    assert result.converged
    assert result.index == 1
    assert result.gradient_norm < 1e-3
    assert result.calls == result.force_calls + result.hessian_calls
    assert abs(result.x[0]) < 1e-3
    assert abs(result.x[1]) < 1e-3


def test_cap_counts_only_evaluations_that_ran():
    tsopt = _tsopt()
    seen = {"force": 0, "hessian": 0}

    def energy_grad(x):
        seen["force"] += 1
        return float(x[0] * x[0] + x[1] * x[1]), np.array([2.0 * x[0], 2.0 * x[1]])

    def projected(_x):
        seen["hessian"] += 1
        return np.eye(2), np.eye(2) * 2.0

    result = tsopt.refine(
        np.array([0.4, -0.2]),
        energy_grad,
        projected,
        kind="default",
        cap=6,
    )
    assert result.force_calls == seen["force"]
    assert result.hessian_calls == seen["hessian"]
    assert result.calls == seen["force"] + seen["hessian"]
    assert result.calls <= 6
    assert not result.converged


def test_soft_mode_follow_finds_the_quartic_saddle():
    tsopt = _tsopt()

    def energy_grad(x):
        # Minima at x=±1. The index-1 point is the origin.
        energy = (1.0 - x[0] * x[0]) ** 2 + x[1] * x[1]
        grad = np.array([-4.0 * x[0] * (1.0 - x[0] * x[0]), 2.0 * x[1]])
        return float(energy), grad

    def projected(x):
        hessian = np.diag([-4.0 + 12.0 * x[0] * x[0], 2.0])
        return np.eye(2), hessian

    result = tsopt.follow_soft_mode(
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        energy_grad,
        projected,
        max_steps=40,
    )
    assert result.converged
    assert result.index == 1
    assert result.gradient_norm < 1e-3
    assert result.calls == result.force_calls + result.hessian_calls
    assert abs(result.x[0]) < 1e-2
    assert abs(result.x[1]) < 1e-2
