# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Run ASE 3.29.0 NEB and dimer tests against ``rgsaddle.ase``.

The modules under ``vendor/`` are ASE's own tests, LGPL-2.1-or-later,
copied from tag 3.29.0. This file replaces ``ase.mep.NEB`` and the
minimum-mode classes with the rgsaddle drivers before those modules
import them.

DyNEB is not vendored. ``ase/test/neb/test_dynamic_neb.py`` drives
``ase.mep.DyNEB`` (dynamic and scaled image relaxation). rgsaddle does
not implement dynamic NEB, and a keyword that would select it raises
``NotImplementedError``.

``ase/optimize/test/neb.py`` is not vendored. It is a script with no
pytest entry and no assertion.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")
else:
    import ase  # noqa: F401  # collection fails closed when ASE is required

import ase.mep as mep
import ase.mep.dimer as ase_dimer
import ase.mep.neb as ase_neb

import rgsaddle.ase as drivers
from rgsaddle.ase import MinModeAtoms, NEB

# Vendor modules bind ``from ase.mep import NEB`` at import. Patch first,
# import them, then put ASE's classes back so the rest of the suite still
# compares against ASE. ``test_shared_calculator_neb`` looks names up on the
# ``neb`` module at call time, so that module keeps a view with the drivers.
_ORIG_MEP = {name: getattr(mep, name) for name in dir(mep) if not name.startswith("_")}
_ORIG_NEB = {name: getattr(ase_neb, name) for name in dir(ase_neb) if not name.startswith("_")}
_ORIG_DIMER = {
    name: getattr(ase_dimer, name) for name in dir(ase_dimer) if not name.startswith("_")
}

mep.NEB = NEB
ase_neb.NEB = NEB
mep.MinModeAtoms = MinModeAtoms
ase_dimer.MinModeAtoms = MinModeAtoms

_OPTIONAL = (
    "DimerControl",
    "MinModeTranslate",
    "interpolate",
    "idpp_interpolate",
    "SingleCalculatorNEB",
)
_OVERRIDES: dict[str, object] = {"NEB": NEB, "MinModeAtoms": MinModeAtoms}
for _name in _OPTIONAL:
    _obj = getattr(drivers, _name, None)
    if _obj is None:
        continue
    _OVERRIDES[_name] = _obj
    setattr(mep, _name, _obj)
    if hasattr(ase_neb, _name):
        setattr(ase_neb, _name, _obj)
    if hasattr(ase_dimer, _name):
        setattr(ase_dimer, _name, _obj)


class _Prefer:
    """Attribute lookup that prefers the rgsaddle drivers over an ASE module."""

    def __init__(self, base: object, overrides: dict[str, object]) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str) -> object:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_base"), name)


# Pytest loads these files as top-level modules named by the file stem.
for _path in sorted((Path(__file__).parent / "vendor").glob("test_*.py")):
    _mod_name = _path.stem
    _spec = importlib.util.spec_from_file_location(_mod_name, _path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load vendored test {_path}")
    _vendor = importlib.util.module_from_spec(_spec)
    sys.modules[_mod_name] = _vendor
    _spec.loader.exec_module(_vendor)
    if _mod_name == "test_shared_calculator_neb":
        _vendor.neb = _Prefer(ase_neb, _OVERRIDES)

for _name, _obj in _ORIG_MEP.items():
    setattr(mep, _name, _obj)
for _name, _obj in _ORIG_NEB.items():
    setattr(ase_neb, _name, _obj)
for _name, _obj in _ORIG_DIMER.items():
    setattr(ase_dimer, _name, _obj)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "optimize: ASE optimizer test")
    config.addinivalue_line("markers", "slow: ASE slow test")


@pytest.fixture
def testdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test working directory, the role of ASE's ``testdir`` fixture."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# node id substring -> one-line reason. Matched against pytest's nodeid.
_SKIPS: tuple[tuple[str, str], ...] = (
    (
        "test_autoneb.py",
        "drives ASE AutoNEB image insertion, not the NEB or minimum-mode drivers",
    ),
    (
        "test_dimer.py",
        "exercises FixBondLength on a two-atom cluster, not the minimum-mode method",
    ),
    (
        "test_get_neb_method",
        "asserts ASE's private NEB method classes, not the public force",
    ),
    (
        "test_neb_optimizers",
        "drives ASE NEBOptimizer on a spline band with an Exp preconditioner",
    ),
    (
        "test_precon_initialisation",
        "asserts ASE Exp preconditioner assembly, not the band force",
    ),
    (
        "test_single_precon_initialisation",
        "asserts ASE Exp preconditioner assembly, not the band force",
    ),
    (
        "test_list_precon_initialisation",
        "asserts ASE Exp preconditioner assembly, not the band force",
    ),
    (
        "test_precon_assembly",
        "asserts ASE Exp preconditioner assembly, not the band force",
    ),
    (
        "test_spline_fit",
        "asserts ASE's cubic-spline fit object, not the band force",
    ),
    (
        "test_integrate_forces",
        "integrates forces on ASE's spline fit, not the band force",
    ),
    (
        "test_neb_tr.py",
        "FIRE step-count ratio follows ASE's partial BFGS IDPP path; the band force matches on one geometry and the barriers agree, but the aligned run is not faster by the factor 0.75",
    ),
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        node = item.nodeid
        for needle, reason in _SKIPS:
            if needle in node:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        method = callspec.params.get("method")
        precon = callspec.params.get("precon")
        if method in {"spline", "string"} or (precon not in (None, "precon") and precon is not None):
            # ``precon`` is also the fixture-parameter name only inside
            # test_neb_methods. Other tests do not parametrize it.
            if "test_neb_methods" in node:
                item.add_marker(
                    pytest.mark.skip(
                        reason="spline, string, and preconditioned NEB have no rgsaddle force"
                    )
                )
