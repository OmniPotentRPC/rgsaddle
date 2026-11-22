#!/usr/bin/env python3
"""Record Sella's internal-coordinate counts for a fixed set of molecules.

Sella is an oracle here, never a dependency of the crate: the output
`tests/fixtures/sella_topology.txt` holds the geometries, Sella's
bonds, and the number of angles, dihedrals and dummy atoms it builds, so
the Rust test can compare without Python. Line format:

    sella <version>
    molecule <name>
    numbers <Z> ...
    positions <x y z per atom, flattened>
    bonds <i-j> ...
    counts <nbonds> <nangles> <ndihedrals> <ndummies>

    python scripts/gen_sella_topology_fixture.py > tests/fixtures/sella_topology.txt
"""
import json
import sys

from ase.build import molecule
from sella.internal import Internals

MOLECULES = [
    "H2O", "NH3", "CH4", "H2O2", "C2H6", "C2H4", "CH3OH", "CH3SH",
    "CO2", "HCN", "C2H2", "CH3CH2OH", "C6H6", "HCOOH", "CH3CHO",
]


def sella_version():
    try:
        from importlib.metadata import version

        return version("sella")
    except Exception:
        return "unknown"


def record(name):
    atoms = molecule(name)
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    counts = {k: len(v) for k, v in internals.internals.items()}
    bonds = sorted(sorted(int(i) for i in b.indices) for b in internals.internals["bonds"])
    return {
        "name": name,
        "numbers": [int(z) for z in atoms.get_atomic_numbers()],
        "positions": [[float(c) for c in p] for p in atoms.get_positions()],
        "bonds": bonds,
        "nbonds": counts["bonds"],
        "nangles": counts["angles"],
        "ndihedrals": counts["dihedrals"],
        "ndummies": int(internals.ndummies),
    }


def main():
    out = [f"sella {sella_version()}"]
    for name in MOLECULES:
        r = record(name)
        out.append(f"molecule {r['name']}")
        out.append("numbers " + " ".join(str(z) for z in r["numbers"]))
        out.append("positions " + " ".join(repr(c) for p in r["positions"] for c in p))
        out.append("bonds " + " ".join(f"{a}-{b}" for a, b in r["bonds"]))
        out.append(
            f"counts {r['nbonds']} {r['nangles']} {r['ndihedrals']} {r['ndummies']}"
        )
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
