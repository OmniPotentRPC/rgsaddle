# Vendored from ASE 3.29.0 (ase/test/neb/test_neb.py).
# ASE is free software under the GNU Lesser General Public License,
# version 2.1 or later (LGPL-2.1-or-later). This copy keeps that notice.
# Upstream: https://gitlab.com/ase/ase/-/blob/3.29.0/ase/test/neb/test_neb.py

"""Tests for NEB."""

import numpy as np
import pytest

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import NEB
from ase.optimize import BFGS


@pytest.fixture(name='neb')
def fixture_neb():
    """NEB Fixture."""
    atoms_ref = fcc100('Al', size=(2, 2, 3))
    add_adsorbate(atoms_ref, 'Au', 1.7, 'hollow')
    atoms_ref.center(axis=2, vacuum=4.0)

    # Fix second and third layers:
    mask = atoms_ref.get_tags() > 1
    atoms_ref.set_constraint(FixAtoms(mask=mask))

    initial = atoms_ref.copy()
    initial.calc = EMT()
    with BFGS(initial) as opt:
        opt.run(fmax=0.05)

    final = atoms_ref.copy()
    final[-1].x += final.get_cell()[0, 0] / 2
    final.calc = EMT()
    with BFGS(final) as opt:
        opt.run(fmax=0.05)

    images = [initial]
    for _ in range(3):
        image = initial.copy()
        image.calc = EMT()
        images.append(image)
    images.append(final)

    neb = NEB(images, method='improvedtangent')
    neb.interpolate()
    with BFGS(neb) as opt:
        opt.run(fmax=0.05)

    return neb


def test_unconstrained_forces(neb: NEB) -> None:
    """Test if unconstrained forces are different from constrained forces."""
    n = len(neb)
    for i, atoms in enumerate(neb.iterimages()):
        if i in {0, n - 1}:
            continue
        assert atoms.constraints  # not empty
        forces_unconstrained = atoms.get_forces(apply_constraint=False)
        forces_constrained = atoms.get_forces(apply_constraint=True)
        assert not np.allclose(forces_unconstrained, forces_constrained)
