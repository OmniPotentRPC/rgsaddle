"""ASE AutoNEB on Müller-Brown, z held at 0. Prints one JSON object."""
import json
import os
import re
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixCartesian
from ase.io import write
from ase.mep import AutoNEB
from ase.optimize import FIRE

A = np.array([-200.0, -100.0, -170.0, 15.0])
AA = np.array([-1.0, -1.0, -6.5, 0.7])
B = np.array([0.0, 0.0, 11.0, 0.6])
C = np.array([-10.0, -10.0, -6.5, 0.7])
X0 = np.array([1.0, 0.0, -0.5, -1.0])
Y0 = np.array([0.0, 0.5, 1.5, 1.0])


def muller_brown(x, y):
    energy = 0.0
    gx = 0.0
    gy = 0.0
    for i in range(4):
        dx = x - X0[i]
        dy = y - Y0[i]
        exponent = AA[i] * dx * dx + B[i] * dx * dy + C[i] * dy * dy
        weight = A[i] * np.exp(exponent)
        px = 2.0 * AA[i] * dx + B[i] * dy
        py = B[i] * dx + 2.0 * C[i] * dy
        energy += weight
        gx += weight * px
        gy += weight * py
    return float(energy), float(gx), float(gy)


class MB(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        pos = atoms.get_positions()
        x, y = float(pos[0, 0]), float(pos[0, 1])
        energy, gx, gy = muller_brown(x, y)
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = -gx
        forces[0, 1] = -gy
        self.results = {"energy": energy, "forces": forces}


class TightFIRE(FIRE):
    def __init__(self, atoms, **kwargs):
        kwargs.setdefault("maxstep", 0.05)
        super().__init__(atoms, **kwargs)


def attach(images):
    for atoms in images:
        atoms.calc = MB()
        atoms.set_constraint(FixCartesian(0, mask=[False, False, True]))


def main():
    start = np.array([-0.05001082, 0.46669410, 0.0])
    end = np.array([0.62349940, 0.02803776, 0.0])
    root = tempfile.mkdtemp(prefix="ase-autoneb-")
    prefix = os.path.join(root, "neb")
    for index, xyz in enumerate((start, end)):
        atoms = Atoms("H", positions=[xyz])
        attach([atoms])
        atoms.get_potential_energy()
        write(f"{prefix}{index:03d}.traj", atoms)
    log_path = os.path.join(root, "log.txt")
    saved = sys.stdout
    sys.stdout = open(log_path, "w")
    try:
        neb = AutoNEB(
            attach,
            prefix,
            n_simul=3,
            n_max=7,
            fmax=[0.5, 0.05],
            maxsteps=[25, 80],
            k=5.0,
            climb=True,
            method="improvedtangent",
            optimizer=TightFIRE,
            remove_rotation_and_translation=False,
            space_energy_ratio=0.5,
            parallel=False,
            smooth_curve=False,
            interpolate_method="linear",
        )
        neb.run()
    finally:
        sys.stdout.close()
        sys.stdout = saved
    text = open(log_path).read()
    geometric = [int(m) for m in re.findall(r"Max length between images is at\s+(\d+)", text)]
    dynamic = [int(m) for m in re.findall(r"Adding image between (\d+) and", text)]
    images = neb.all_images
    energies = []
    positions = []
    for atoms in images:
        atoms.calc = MB()
        energies.append(float(atoms.get_potential_energy()))
        positions.append([float(v) for v in atoms.positions[0]])
    interior = list(range(1, len(energies) - 1))
    ci = max(interior, key=lambda i: energies[i])
    print(
        json.dumps(
            {
                "geometric": geometric,
                "dynamic": dynamic,
                "ci_index": ci,
                "ci": positions[ci],
                "energies": energies,
                "positions": positions,
                "log": text[-2000:],
            }
        )
    )


if __name__ == "__main__":
    main()
