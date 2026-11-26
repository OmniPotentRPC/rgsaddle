"""ASE 3.29 spline and string NEB forces on an EMT Au adatom hop.

``method='spline'`` takes the unit tangent and the curvature spring from
the not-a-knot cubic through the images. ``method='string'`` uses that
tangent with no spring and redistributes the interior images. One
``get_forces`` call is deterministic, so the per-image forces match at
1e-9. The converged energies use 1e-5 eV: both optimizers stop on the
force residual (0.05 eV/Angstrom), not on the energy.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("RGSADDLE_REQUIRE_ASE") != "1":
    pytest.importorskip("ase")

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import NEB as AseNeb
from ase.optimize import BFGS

from rgsaddle.ase import NEB


def _endpoints():
    """Relaxed Al(100) hollow and the hollow one cell step along x."""
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    slab.set_constraint(FixAtoms(mask=slab.get_tags() > 1))
    initial = slab.copy()
    initial.calc = EMT()
    final = slab.copy()
    final[-1].x += float(final.cell[0, 0]) / 2.0
    final.calc = EMT()
    # Relax the ends so the hop is a barrier between minima, as in ASE's NEB fixture.
    assert BFGS(initial, logfile=None).run(fmax=0.05, steps=80)
    assert BFGS(final, logfile=None).run(fmax=0.05, steps=80)
    return initial, final


def _band(initial, final):
    images = []
    for _ in range(3):
        image = initial.copy()
        image.calc = EMT()
        images.append(image)
    images.insert(0, initial.copy())
    images[0].calc = EMT()
    end = final.copy()
    end.calc = EMT()
    images.append(end)
    return images


def _energies(images) -> np.ndarray:
    return np.array([float(image.get_potential_energy()) for image in images], dtype=np.float64)


@pytest.mark.parametrize("method", ["spline", "string"])
def test_method_matches_ase_329_emt(method: str) -> None:
    initial, final = _endpoints()
    ase_images = _band(initial, final)
    our_images = _band(initial, final)
    ase = AseNeb(ase_images, k=0.1, climb=False, method=method)
    ours = NEB(our_images, k=0.1, climb=False, method=method)
    ase.interpolate()
    ours.interpolate()
    np.testing.assert_allclose(
        np.stack([image.positions for image in ours.images]),
        np.stack([image.positions for image in ase.images]),
        atol=1e-12,
        rtol=0.0,
    )
    # One call, before any redistribution. The spline arithmetic is deterministic.
    np.testing.assert_allclose(ours.get_forces(), ase.get_forces(), atol=1e-9, rtol=0.0)

    if method == "string":
        nudged = np.array(ase.get_positions(), dtype=np.float64, copy=True)
        nudged[0, 0] += 0.03
        ase.set_positions(nudged)
        ours.set_positions(np.array(nudged, copy=True))
        np.testing.assert_allclose(ours.get_positions(), ase.get_positions(), atol=1e-9, rtol=0.0)
        # Restore the interpolated band for the optimizer comparison.
        ase_images = _band(initial, final)
        our_images = _band(initial, final)
        ase = AseNeb(ase_images, k=0.1, climb=False, method=method)
        ours = NEB(our_images, k=0.1, climb=False, method=method)
        ase.interpolate()
        ours.interpolate()

    assert BFGS(ase, logfile=None).run(fmax=0.05, steps=200)
    assert BFGS(ours, logfile=None).run(fmax=0.05, steps=200)
    # 1e-5 eV, not 1e-9: both runs stop when the largest force is under
    # 0.05 eV/Angstrom, so the energy is not fixed more tightly than that
    # residual along the path.
    np.testing.assert_allclose(_energies(ours.images), _energies(ase.images), atol=1e-5, rtol=0.0)
