"""The ASE drop-in example reaches the barrier ASE's own NEB reaches."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# RGSADDLE_REQUIRE_ASE=1 turns the skip into a collection error.
if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "ase_neb.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("ase_neb_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # tests/examples_ci.rs lists examples/, so leave no __pycache__ there.
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = keep
    return module


def test_drop_in_neb_matches_ase_on_the_adatom_hop() -> None:
    example = _load()
    ase_height, ase_steps = example.barrier(example.AseNEB)
    height, steps = example.barrier(example.RgsaddleNEB)
    assert height == pytest.approx(ase_height, abs=1e-6)
    assert height == pytest.approx(0.3748, abs=5e-4)
    assert abs(steps - ase_steps) <= 2
    assert steps < 500
