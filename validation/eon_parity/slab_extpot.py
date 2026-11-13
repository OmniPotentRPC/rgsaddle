#!/usr/bin/env python3
"""Adatom double well used by the frozen-slab eOn rows.

Energy is (x^2 - 1)^2 + 8 y^2 + 8 (z - 1.2)^2 on the fourth atom.
The saddle-search con files store the raw coordinates, so SHIFT is
(0, 0, 0) for those rows. The NEB con files add 10 to x, and that run
set SHIFT to (10, 0, 0), so x = -1 is not wrapped into the cell.
"""
from pathlib import Path

SHIFT = (0.0, 0.0, 0.0)
rows = [
    line.split()
    for line in Path("from_eon_to_extpot").read_text().splitlines()
    if line.strip()
]
pos = [
    [float(row[1]) - SHIFT[0], float(row[2]) - SHIFT[1], float(row[3]) - SHIFT[2]]
    for row in rows[3:]
]
ax, ay, az = pos[3]
energy = (ax * ax - 1.0) ** 2 + 8.0 * ay * ay + 8.0 * (az - 1.2) ** 2
forces = [[0.0, 0.0, 0.0] for _ in pos]
forces[3] = [-4.0 * ax * (ax * ax - 1.0), -16.0 * ay, -16.0 * (az - 1.2)]
Path("from_extpot_to_eon").write_text(
    f"{energy:.16e}\n"
    + "\n".join(" ".join(f"{value:.16e}" for value in force) for force in forces)
    + "\n"
)
