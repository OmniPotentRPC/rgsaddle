"""Replay the pinned pt-island misses and check what the push-off sees.

usage: pinned.py

OPTBENCH_ROOT and RGSADDLE_LIB as for ``pushoff.py``. Replays dimer
FIRE searches 1, 2, 5 and 18 and rotation-skip search 62 of the
catalog (seed 20261016) and prints each in the miss classes of the
campaign. It exits nonzero unless every search converges to an index-1
saddle, the mode handed to the push-off is a unit vector within 0.99 of
the lowest Hessian eigenvector and connects exactly when that
eigenvector does, and FIRE search 1 reaches a connected saddle under
1.5 eV. rgsaddle 0.2.2 and 8fa0eb15 fail the index on rotation-skip 62
(an index-3 point 16 A out) and the connection on FIRE 1, whose saddle
borders two other states; the dimer rotation history kept across
translations (116b92e) passes.
"""

import os
import sys

PINNED = (("dimer-fire", (1, 2, 5, 18)), ("dimer-rotskip", (62,)))


def miss_class(row):
    if not row["converged"]:
        return "unconverged"
    if row["hessian_index"] != 1:
        return "wrong_stationary"
    if not row["connected_session_mode"]:
        return "disconnected"
    if row["barrier"] >= 1.5:
        return "over_1.5_eV"
    return "connected"


def main():
    from pushoff import replay

    failures = []
    print("method search class barrier_eV saddle_A minus_A plus_A overlap")
    for method, searches in PINNED:
        for row in replay(f"rgsaddle/saddle-{method}", set(searches)):
            label = miss_class(row)
            print(
                method,
                row["search"],
                label,
                f"{row['barrier']:.3f}",
                f"{row['largest_atom_saddle']:.2f}",
                f"{row.get('largest_atom_minus', float('nan')):.2f}",
                f"{row.get('largest_atom_plus', float('nan')):.2f}",
                f"{row.get('overlap_with_hessian_mode', float('nan')):.4f}",
            )
            where = f"{method} {row['search']}"
            if label in ("unconverged", "wrong_stationary"):
                failures.append(f"{where}: {label}")
                continue
            if abs(row["mode_norm"] - 1.0) > 1e-9 or row["overlap_with_hessian_mode"] < 0.99:
                failures.append(f"{where}: push-off mode off the lowest eigenvector")
            if row["connected_session_mode"] != row["connected_hessian_mode"]:
                failures.append(f"{where}: push-off verdict differs from the eigenvector's")
            if (method, row["search"]) == ("dimer-fire", 1) and label != "connected":
                failures.append(f"{where}: {label}")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if "OPTBENCH_ROOT" not in os.environ or "RGSADDLE_LIB" not in os.environ:
        print(__doc__.splitlines()[2])
    else:
        main()
