"""Check the mode the pt-island driver pushes off along.

usage: pushoff.py TEMPLATE SEARCHES OUTPUT.csv

OPTBENCH_ROOT is an optbench checkout and RGSADDLE_LIB the librgsaddle
to test. The first SEARCHES draws of the catalog (seed 20261016) are
replayed with ``rgsaddle_minmode_step`` as ``optbench.rgsaddle_minmode``
climbs them: the same start, random mode, configuration, L2 force gate
and 1000-step limit. At a converged point the session mode the driver
reads with ``rgsaddle_minmode_mode`` is compared with the lowest
eigenvector of the driver's central-difference Hessian, and the endpoint
minimizations (the driver's L-BFGS from 0.2 on either side, connection
at distance 0.1) are run along each of the two. ``largest_atom_*`` are
the largest per-atom distances from the minimized reactant.
"""

import csv
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

FIELDS = (
    "search",
    "converged",
    "steps",
    "climb_calls",
    "barrier",
    "largest_atom_saddle",
    "hessian_index",
    "mode_norm",
    "overlap_with_hessian_mode",
    "connected_session_mode",
    "connected_hessian_mode",
    "largest_atom_minus",
    "largest_atom_plus",
)


def largest_atom(delta):
    return float(np.max(np.linalg.norm(delta.reshape(-1, 3), axis=1)))


def replay(name, wanted):
    """Yield the row of each catalog search in ``wanted``, in order."""
    from optbench import rgsaddle_minmode as M
    from optbench.campaign import load_template, resolve_problem
    from optbench.connection import connected as linked
    from optbench.connection import periodic_lengths
    from optbench.displacement import displace, neighbours
    from optbench.surfaces import load_con, potential_from_problem

    root = Path(os.environ["OPTBENCH_ROOT"])
    template = load_template(root, name)
    problem = resolve_problem(root, "saddle-search/pt-island")
    geometries = problem["input"]["geometries"]
    positions, fixed, cell = load_con(M._stored(root, next(p for p in geometries if p.endswith(".con"))))
    free = np.flatnonzero(~fixed)
    evaluate = M._bind(potential_from_problem(problem), positions, free, cell)
    saddle = M._section(template, "SaddleSearch")
    optimizer = M._section(template, "Optimizer")
    minimum, _ = M._lbfgs_minimize(M.HarnessSurface(evaluate), positions[free].reshape(-1).copy(), 1e-3, 1000)
    emin = evaluate(minimum)[0]
    lengths = periodic_lengths(cell, problem["potential"]["pbc"])
    reactant = positions.copy()
    reactant[free] = minimum.reshape(-1, 3)
    listed = list(range(7))
    near = neighbours(reactant, lengths, 3.3, listed)
    rng = np.random.default_rng(M.CATALOG_SEED + 1)
    lib = M._load()
    dof = minimum.size
    pointer = ctypes.POINTER(ctypes.c_double)
    for search in range(1, max(wanted) + 1):
        _epicenter, shift, seed = displace(rng, ~fixed, listed, near, 0.1)
        if search not in wanted:
            continue
        x0 = np.ascontiguousarray(minimum + shift[free].reshape(-1))
        kick = np.ascontiguousarray(seed[free].reshape(-1))
        cfg = M._minmode_config(lib, saddle.get("min_mode_method", "dimer"), optimizer["opt_method"], saddle, optimizer, 1e-3)
        session = lib.rgsaddle_minmode_create(ctypes.byref(cfg), dof // 3, x0.ctypes.data_as(pointer), kick.ctypes.data_as(pointer))
        lib.rgsaddle_minmode_set_force_gate(session, M.GATE_L2)
        surface = M.HarnessSurface(evaluate)
        callback = M._callback(lib, surface, dof)
        converged = False
        steps = 0
        while steps < int(saddle["max_iterations"]):
            report = M._blank_report(lib)
            rc = lib.rgsaddle_minmode_step(session, callback, None, ctypes.byref(report))
            steps += 1
            if rc != 0:
                break
            if report.status == 1:
                converged = True
                break
        x = M._position(lib, lib.rgsaddle_minmode_position, session, dof)
        mode = M._mode(lib, session, dof)
        lib.rgsaddle_minmode_free(session)
        row = {
            "search": search,
            "converged": converged,
            "steps": steps,
            "climb_calls": surface.calls,
            "barrier": evaluate(x)[0] - emin,
            "largest_atom_saddle": largest_atom(x - minimum),
            "mode_norm": float(np.linalg.norm(mode)),
        }
        if converged:
            values, vectors = M._hessian_spectrum(M.HarnessSurface(evaluate), x)
            exact = vectors[:, 0]
            row["hessian_index"] = int(sum(value < -1e-6 for value in values))
            row["overlap_with_hessian_mode"] = float(abs(np.dot(M._unit(mode), exact)))
            for label, axis in (("session_mode", M._unit(mode)), ("hessian_mode", exact)):
                judge = M.HarnessSurface(evaluate)
                minus, minus_ok = M._lbfgs_minimize(judge, x - 0.2 * axis, 1e-3, 1000)
                plus, plus_ok = M._lbfgs_minimize(judge, x + 0.2 * axis, 1e-3, 1000)
                row[f"connected_{label}"] = bool(minus_ok and plus_ok and linked(minimum, plus, minus, lengths, 0.1))
                if label == "session_mode":
                    row["largest_atom_minus"] = largest_atom(minus - minimum)
                    row["largest_atom_plus"] = largest_atom(plus - minimum)
        yield row


def text(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return "" if value is None else value


def main():
    name, searches, output = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for row in replay(name, set(range(1, searches + 1))):
            writer.writerow([text(row.get(field)) for field in FIELDS])
            handle.flush()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
