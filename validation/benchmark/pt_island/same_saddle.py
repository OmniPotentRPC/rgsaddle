"""Number the saddles that ``modes.py`` runs converge to.

usage: same_saddle.py OUTPUT.csv LABEL=MODES.csv ...

Each MODES.csv is the OUTPUT of one ``modes.py`` run, whose ``.npz``
sits beside it. Two converged points are one saddle when their barriers
agree to 0.01 eV and no atom differs by more than 0.1 A under the
minimum image, the repeat test of optbench's process table. Saddles are
numbered in the order the runs, as given, first reach them. OUTPUT has
one row per run and converged search: the label, the search, the
barrier and the saddle number. Two runs reach the same saddle from a
start when they give its search the same number.
"""

import csv
import sys

import numpy as np

BARRIER_TOL = 0.01
ATOM_TOL = 0.1


def main():
    from optbench.connection import per_atom_max

    out = sys.argv[1]
    saddles = []
    rows = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        points = np.load(path.removesuffix(".csv") + ".npz")
        lengths = points["lengths"]
        for search, x, barrier in zip(points["search"], points["x"], points["barrier"]):
            x = x.reshape(-1, 3)
            number = next(
                (
                    index
                    for index, (other, other_barrier) in enumerate(saddles)
                    if abs(barrier - other_barrier) <= BARRIER_TOL and per_atom_max(x, other, lengths) <= ATOM_TOL
                ),
                None,
            )
            if number is None:
                number = len(saddles)
                saddles.append((x, barrier))
            rows.append((label, int(search), f"{float(barrier):.8g}", number))
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("label", "search", "barrier", "saddle"))
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
