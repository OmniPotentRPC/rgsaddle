"""Default Gaussian dimer displacement against ASE 3.29.0.

ASE 3.29.0 ``DimerControl`` defaults ``displacement_method`` and
``initial_eigenmode_method`` to ``gauss``. ``displace`` draws
``RandomState(seed).normal(0, gauss_std)`` one coordinate at a time,
atom then x, y, z. ``get_potential_energy`` builds the initial mode
from one further draw and restores the positions before any rotation.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")

from ase import Atoms
from ase.calculators.emt import EMT
from ase.mep.dimer import DimerControl as AseControl
from ase.mep.dimer import MinModeAtoms as AseMinMode

from rgsaddle.ase import DimerControl, MinModeAtoms

# Positions, energies and modes are the same RandomState draw. 1e-9 is
# the repeatability bar and the agreement bar with ASE 3.29.0.
_TOLERANCE = 1e-9

# ASE stores DimerControl.parameters on the class. Another test can
# replace the gauss default, so each comparison sets it and puts it back.
# The class dict is process-wide, so the checks stay on one thread.
pytestmark = [pytest.mark.thread_unsafe, pytest.mark.iterations(1)]

_GAUSS = {
    "displacement_method": "gauss",
    "initial_eigenmode_method": "gauss",
}


def _cluster() -> Atoms:
    atoms = Atoms(
        "Au4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.4, 0.0],
            [1.4, 0.8, 2.2],
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )
    atoms.calc = EMT()
    return atoms


def _quiet_gauss(search: MinModeAtoms | AseMinMode, **displace: object) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        search.displace(**displace)
        energy = float(search.get_potential_energy())
        mode = np.asarray(search.get_eigenmode(), dtype=np.float64)
        forces = np.asarray(search.atoms.get_forces(), dtype=np.float64)
        positions = np.asarray(search.get_positions(), dtype=np.float64)
    return positions, energy, forces, mode


def _pair(
    seed: int,
    *,
    displace: dict[str, object] | None = None,
) -> tuple[tuple[np.ndarray, float, np.ndarray, np.ndarray], tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    displace = {} if displace is None else displace
    saved = dict(AseControl.parameters)
    try:
        with DimerControl(logfile=None, eigenmode_logfile=None, **_GAUSS) as control:
            ours = MinModeAtoms(_cluster(), control, random_seed=seed)
            ours_result = _quiet_gauss(ours, **displace)
        with AseControl(logfile=None, eigenmode_logfile=None, **_GAUSS) as control:
            reference = AseMinMode(_cluster(), control, random_seed=seed)
            reference_result = _quiet_gauss(reference, **displace)
    finally:
        AseControl.parameters.clear()
        AseControl.parameters.update(saved)
    return ours_result, reference_result


def _assert_close(
    ours: tuple[np.ndarray, float, np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, float, np.ndarray, np.ndarray],
) -> None:
    positions, energy, forces, mode = ours
    ref_positions, ref_energy, ref_forces, ref_mode = reference
    assert positions.shape == ref_positions.shape
    assert mode.shape == ref_mode.shape == positions.shape
    assert np.max(np.abs(positions - ref_positions)) < _TOLERANCE
    assert abs(energy - ref_energy) < _TOLERANCE
    assert np.max(np.abs(forces - ref_forces)) < _TOLERANCE
    assert np.max(np.abs(mode - ref_mode)) < _TOLERANCE
    assert abs(float(np.linalg.norm(mode)) - 1.0) < _TOLERANCE


def test_default_gauss_displacement_matches_ase_329() -> None:
    ours, reference = _pair(7)
    _assert_close(ours, reference)
    # The draw moved the cluster. A no-op would also match a no-op.
    assert np.max(np.abs(ours[0] - _cluster().get_positions())) > 1e-3


def test_gauss_draw_repeats_to_1e_9() -> None:
    first, _reference = _pair(11)
    second, _again = _pair(11)
    _assert_close(first, second)


def test_mask_and_center_match_ase_329() -> None:
    masked, masked_ref = _pair(5, displace={"mask": [False, False, False, True], "gauss_std": 0.2})
    _assert_close(masked, masked_ref)
    start = _cluster().get_positions()
    # Only the last atom is free under that mask.
    assert np.max(np.abs(masked[0][:3] - start[:3])) < _TOLERANCE
    assert np.max(np.abs(masked[0][3] - start[3])) > 1e-3

    centred, centred_ref = _pair(
        9,
        displace={"displacement_center": 0, "radius": 3.0, "number_of_atoms": 2, "gauss_std": 0.15},
    )
    _assert_close(centred, centred_ref)
    moved = np.linalg.norm(centred[0] - start, axis=1)
    assert moved[0] > 1e-3
    assert moved[3] > 1e-3
    assert moved[1] < _TOLERANCE
    assert moved[2] < _TOLERANCE
