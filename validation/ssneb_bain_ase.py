"""ASE Frechet-cell NEB on the switched Lennard-Jones Bain path.

The pair is the same one as tests/ssneb_bain.rs and
validation/bain_lj_reference.py: epsilon = 1 eV, sigma = 1 Angstrom,
C^2 quintic switch over [2.0, 2.2] Angstrom. Both ends are the
two-atom body-centred tetragonal cell. Every image is a
FrechetCellFilter of a copy of the bcc cell, so the NEB coordinates
are the undeformed positions plus the log strain. Periodic flags are
cleared on those copies because ASE's NEB refuses a cell that changes
in a periodic direction; the calculator still sums the lattice.

Run: python validation/ssneb_bain_ase.py
"""

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.mep import NEB
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator, all_changes

RC = 2.2
WIDTH = 0.2
A_BCC = 1.228441371
A_FCC = 1.100691058
C_FCC = 1.556612222
E_BCC = -7.4148256256


def pair(r):
    sr6 = (1.0 / r) ** 6
    v = 4.0 * (sr6 * sr6 - sr6)
    dv = 4.0 * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r
    t = np.clip((r - (RC - WIDTH)) / WIDTH, 0.0, 1.0)
    s = 1.0 + t**3 * (-10.0 + t * (15.0 - 6.0 * t))
    ds = -30.0 * t**2 * (1.0 - t) ** 2 / WIDTH
    return v * s, dv * s + v * ds


def lj(x, h):
    x = np.asarray(x, float).reshape(-1, 3)
    h = np.asarray(h, float).reshape(3, 3)
    n = len(x)
    inv = np.linalg.inv(h)
    reach = [
        int(np.ceil(RC * np.linalg.norm(inv[:, k]) + 0.5)) for k in range(3)
    ]
    energy = 0.0
    grad = np.zeros((n, 3))
    vir = np.zeros((3, 3))
    for i in range(n):
        for j in range(i, n):
            d0 = x[i] - x[j]
            f = d0 @ inv
            d0 = d0 - np.round(f) @ h
            weight = 0.5 if i == j else 1.0
            for a in range(-reach[0], reach[0] + 1):
                for b in range(-reach[1], reach[1] + 1):
                    for c in range(-reach[2], reach[2] + 1):
                        if i == j and a == b == c == 0:
                            continue
                        d = d0 + a * h[0] + b * h[1] + c * h[2]
                        r2 = float(d @ d)
                        if r2 >= RC * RC:
                            continue
                        r = np.sqrt(r2)
                        phi, dphi = pair(r)
                        energy += weight * phi
                        g = weight * dphi / r
                        grad[i] += g * d
                        grad[j] -= g * d
                        vir += g * np.outer(d, d)
    vol = abs(np.linalg.det(h))
    stress = vir / vol
    return energy, grad, stress


class LJ(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        LJ.calls += 1
        energy, forces, stress = lj(atoms.positions, atoms.cell.array)
        voigt = np.array(
            [stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]]
        )
        self.results = {"energy": energy, "forces": forces, "stress": voigt}


def bct(a, c):
    atoms = Atoms(
        "Si2",
        positions=[[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * c]],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, c]],
        pbc=False,
    )
    atoms.calc = LJ()
    return atoms


class BandImage:
    """FrechetCellFilter with the attributes ASE's NEB reads off an Atoms."""

    def __init__(self, filt):
        self.filt = filt
        self.pbc = np.zeros(3, dtype=bool)

    @property
    def cell(self):
        return self.filt.atoms.cell

    @property
    def calc(self):
        return self.filt.atoms.calc

    def get_atomic_numbers(self):
        return self.filt.atoms.get_atomic_numbers()

    def __len__(self):
        return len(self.filt)

    def get_positions(self):
        return self.filt.get_positions()

    def set_positions(self, pos):
        self.filt.set_positions(pos)

    def get_forces(self, **kwargs):
        return self.filt.get_forces(**kwargs)

    def get_potential_energy(self, **kwargs):
        return self.filt.atoms.get_potential_energy(**kwargs)


def main():
    bcc = bct(A_BCC, A_BCC)
    fcc = bct(A_FCC, C_FCC)
    n = 9
    images = []
    for _ in range(n):
        image = bcc.copy()
        image.calc = LJ()
        images.append(BandImage(FrechetCellFilter(image)))
    images[-1].filt.atoms.set_cell(fcc.cell, scale_atoms=False)
    images[-1].filt.atoms.set_positions(fcc.positions)
    target = images[-1].get_positions().copy()
    images[-1].set_positions(target)
    start = images[0].get_positions()
    for i in range(1, n - 1):
        t = i / (n - 1)
        images[i].set_positions((1.0 - t) * start + t * target)
    initial = [img.filt.atoms.get_potential_energy() for img in images]
    print(
        "initial E - E_bcc",
        " ".join(f"{e / 2.0 - E_BCC:.6f}" for e in initial),
    )
    neb = NEB(images, k=0.1, climb=True, method="improvedtangent")
    LJ.calls = 0
    opt = FIRE(neb, logfile=None, maxstep=0.02)
    opt.run(fmax=5e-3, steps=600)
    energies = [img.filt.atoms.get_potential_energy() for img in images]
    ratios = [img.filt.atoms.cell[2, 2] / img.filt.atoms.cell[0, 0] for img in images]
    ci = int(np.argmax(energies))
    barrier = energies[ci] / 2.0 - E_BCC
    print(f"ase steps {opt.nsteps} calculator_calls {LJ.calls}")
    print(f"ase climbing image {ci} c/a {ratios[ci]:.8f} barrier {barrier:.10f} eV/atom")
    print("ase energies per atom", " ".join(f"{e / 2.0 - E_BCC:.6f}" for e in energies))


if __name__ == "__main__":
    main()
