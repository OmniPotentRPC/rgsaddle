"""An Au adatom hopping on Al(100): ASE's NEB and the rgsaddle drop-in.

This is ASE's own NEB tutorial system with the EMT calculator. The two
runs differ in one import. Run it from a checkout after
``pip install 'rgsaddle[ase]'``::

    python3 examples/ase_neb.py
"""

from __future__ import annotations

from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import NEB as AseNEB
from ase.optimize import FIRE

from rgsaddle.ase import NEB as RgsaddleNEB


def hollow_to_hollow() -> tuple[Atoms, Atoms]:
    """Relaxed endpoints of a hop to the next hollow site, lower layers fixed."""
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))

    initial = slab.copy()
    initial.calc = EMT()
    FIRE(initial, logfile=None).run(fmax=0.05)

    final = slab.copy()
    final.positions[-1, 0] += final.cell[0, 0] / 2
    final.calc = EMT()
    FIRE(final, logfile=None).run(fmax=0.05)
    return initial, final


def barrier(neb_class: type, interior: int = 5) -> tuple[float, int]:
    """Climb the hop with ``neb_class`` and return the barrier in eV and the steps."""
    initial, final = hollow_to_hollow()
    images = [initial]
    for _ in range(interior):
        image = initial.copy()
        image.calc = EMT()
        images.append(image)
    images.append(final)

    neb = neb_class(images, climb=True, method="improvedtangent")
    neb.interpolate()
    optimizer = FIRE(neb, logfile=None)
    optimizer.run(fmax=0.05, steps=500)
    energies = [image.get_potential_energy() for image in images]
    return max(energies) - energies[0], optimizer.nsteps


def main() -> None:
    for name, neb_class in (("ase.mep.NEB", AseNEB), ("rgsaddle.ase.NEB", RgsaddleNEB)):
        height, steps = barrier(neb_class)
        print(f"{name:<18} barrier {height:.4f} eV in {steps} steps")


if __name__ == "__main__":
    main()
