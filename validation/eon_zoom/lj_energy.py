#!/usr/bin/env python3
"""ASE Lennard-Jones energy of the eOn neb_lj13 frames.

eOn 3.6.0 (doi:10.5281/zenodo.23166734) was not executed.
The frames are client/unit_tests/data/systems/neb_lj13 at 2f84d583.
epsilon = 1, sigma = 1, rc = 10, ro = 0, smooth = False.
"""

from __future__ import annotations

import json
import pathlib
import sys


def parse(path: pathlib.Path) -> tuple[float, list[tuple[float, float, float]]]:
    lines = path.read_text().splitlines()
    stored = json.loads(lines[1])["energy"]
    start = next(i for i, line in enumerate(lines) if line.startswith("Coordinates")) + 1
    coords = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return stored, coords


def energy(coords: list[tuple[float, float, float]]) -> float:
    total = 0.0
    rc2 = 100.0
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            r2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
            if r2 >= rc2:
                continue
            s6 = r2 ** -3
            total += 4.0 * (s6 * s6 - s6)
    return total


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent / "neb_lj13"
    worst = 0.0
    for name in ("reactant", "product", "saddle"):
        stored, coords = parse(root / f"{name}.con")
        got = energy(coords)
        delta = abs(got - stored)
        worst = max(worst, delta)
        print(f"{name} stored {stored:.12f} computed {got:.12f} delta {delta:.3e}")
    if worst >= 1e-4:
        print("outside 1e-4", file=sys.stderr)
        return 1
    print("eOn binary not run; Pt heptamer not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
