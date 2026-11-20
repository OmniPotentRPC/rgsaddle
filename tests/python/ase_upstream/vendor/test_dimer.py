# Vendored from ASE 3.29.0 (ase/test/test_dimer.py).
# ASE is free software under the GNU Lesser General Public License,
# version 2.1 or later (LGPL-2.1-or-later). This copy keeps that notice.
# Upstream: https://gitlab.com/ase/ase/-/blob/3.29.0/ase/test/test_dimer.py

from ase import Atom, Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixBondLength


def test_dimer():
    dimer = Atoms(
        [Atom('X', (0, 0, 0)), Atom('X', (0, 0, 1))],
        calculator=LennardJones(),
        constraint=FixBondLength(0, 1),
    )
    print(dimer.get_forces())
    print(dimer.positions)
    dimer.positions[:] += 0.1
    print(dimer.positions)
    dimer.positions[:, 2] += 5.1
    print(dimer.positions)
    dimer.positions[:] = [(1, 2, 3), (4, 5, 6)]
    print(dimer.positions)
    dimer.set_positions([(1, 2, 3), (4, 5, 6.2)])
    print(dimer.positions)
