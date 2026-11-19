"""Follow the session mode through pt-island catalog searches.

usage: modes.py TEMPLATE SEARCHES OUTPUT.csv

OPTBENCH_ROOT is an optbench checkout and RGSADDLE_LIB the librgsaddle
to test. TEMPLATE is an optbench template name or a template file. The
first SEARCHES draws of the catalog (seed 20261016) are climbed through
``rgsaddle_minmode_step`` exactly as ``optbench.rgsaddle_minmode`` climbs
them: the same start, random mode, configuration, L2 force gate and
``max_iterations`` limit, stopping on a converged step, a failed step or
a step that spends no force call.

``actions`` counts the surface calls of the mode estimates and
``actions_1`` those of the first. ``start_mode_overlap_1`` compares the
mode of the first step with the lowest eigenvector of the
central-difference Hessian at the start. ``first_negative_step`` is the
first step whose curvature is negative; ``energy_negative`` is the
energy over the minimized reactant there and ``largest_atom_negative``
the largest atom displacement from it. A converged point gets the
Hessian index, the barrier, the overlap of the session mode with the
lowest Hessian eigenvector there, the share of the mode's squared norm
on the seven island atoms and the largest atom displacement. Next to
OUTPUT, an ``.npz`` keeps every converged point (``search``, ``x``,
``barrier``) for ``same_saddle.py``.
"""

import csv
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ISLAND = 7

FIELDS = (
    "search",
    "epicenter",
    "converged",
    "steps",
    "climb_calls",
    "actions",
    "actions_1",
    "start_mode_overlap_1",
    "first_negative_step",
    "energy_negative",
    "largest_atom_negative",
    "hessian_index",
    "barrier",
    "saddle_mode_overlap",
    "island_weight_saddle",
    "largest_atom_saddle",
)


def largest_atom(delta):
    return float(np.max(np.linalg.norm(delta.reshape(-1, 3), axis=1)))


def island_weight(vector):
    from optbench import rgsaddle_minmode as M

    unit = M._unit(vector)
    return float(np.dot(unit[: 3 * ISLAND], unit[: 3 * ISLAND]))


def text(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else value


def main():
    from optbench import rgsaddle_minmode as M
    from optbench.campaign import load_template, resolve_problem
    from optbench.displacement import displace, neighbours
    from optbench.surfaces import load_con, potential_from_problem

    name, searches, output = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    root = Path(os.environ["OPTBENCH_ROOT"])
    template = load_template(root, name)
    problem = resolve_problem(root, "saddle-search/pt-island")
    geometries = problem["input"]["geometries"]
    positions, fixed, cell = load_con(M._stored(root, next(p for p in geometries if p.endswith(".con"))))
    free = np.flatnonzero(~fixed)
    if list(free[:ISLAND]) != list(range(ISLAND)):
        raise RuntimeError("the island is not the first seven free atoms")
    evaluate = M._bind(potential_from_problem(problem), positions, free, cell)
    saddle = M._section(template, "SaddleSearch")
    optimizer = M._section(template, "Optimizer")
    minimum, _ = M._lbfgs_minimize(M.HarnessSurface(evaluate), positions[free].reshape(-1).copy(), 1e-3, 1000)
    emin = evaluate(minimum)[0]
    lengths = M.periodic_lengths(cell, problem["potential"]["pbc"])
    reactant = positions.copy()
    reactant[free] = minimum.reshape(-1, 3)
    listed = list(range(ISLAND))
    near = neighbours(reactant, lengths, 3.3, listed)
    rng = np.random.default_rng(M.CATALOG_SEED + 1)
    lib = M._load()
    dof = minimum.size
    pointer = ctypes.POINTER(ctypes.c_double)
    points = {"search": [], "x": [], "barrier": []}
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for search in range(1, searches + 1):
            epicenter, shift, seed = displace(rng, ~fixed, listed, near, 0.1)
            x0 = np.ascontiguousarray(minimum + shift[free].reshape(-1))
            kick = np.ascontiguousarray(seed[free].reshape(-1))
            _, vectors = M._hessian_spectrum(M.HarnessSurface(evaluate), x0)
            start_mode = vectors[:, 0]
            cfg = M._minmode_config(
                lib, saddle.get("min_mode_method", "dimer"), optimizer["opt_method"], saddle, optimizer, 1e-3
            )
            session = lib.rgsaddle_minmode_create(ctypes.byref(cfg), dof // 3, x0.ctypes.data_as(pointer), kick.ctypes.data_as(pointer))
            if not session:
                raise RuntimeError("rgsaddle_minmode_create failed")
            lib.rgsaddle_minmode_set_force_gate(session, M.GATE_L2)
            surface = M.HarnessSurface(evaluate)
            callback = M._callback(lib, surface, dof)
            row = {"search": search, "epicenter": epicenter}
            converged = False
            steps = 0
            actions = 0
            while steps < int(saddle["max_iterations"]):
                before = surface.calls
                report = M._blank_report(lib)
                rc = lib.rgsaddle_minmode_step(session, callback, None, ctypes.byref(report))
                steps += 1
                if rc != 0:
                    break
                actions += int(report.rotations)
                if steps == 1:
                    mode = M._mode(lib, session, dof)
                    row["actions_1"] = int(report.rotations)
                    row["start_mode_overlap_1"] = float(abs(np.dot(M._unit(mode), start_mode)))
                if report.curvature < 0.0 and "first_negative_step" not in row:
                    x = M._position(lib, lib.rgsaddle_minmode_position, session, dof)
                    row["first_negative_step"] = steps
                    row["energy_negative"] = evaluate(x)[0] - emin
                    row["largest_atom_negative"] = largest_atom(x - minimum)
                if report.status == 1:
                    converged = True
                    break
                if surface.calls == before:
                    break
            x = M._position(lib, lib.rgsaddle_minmode_position, session, dof)
            mode = M._mode(lib, session, dof)
            lib.rgsaddle_minmode_free(session)
            row.update(converged=converged, steps=steps, climb_calls=surface.calls, actions=actions)
            if converged:
                values, vectors = M._hessian_spectrum(M.HarnessSurface(evaluate), x)
                row["hessian_index"] = int(sum(value < -1e-6 for value in values))
                row["barrier"] = evaluate(x)[0] - emin
                row["saddle_mode_overlap"] = float(abs(np.dot(M._unit(mode), vectors[:, 0])))
                row["island_weight_saddle"] = island_weight(mode)
                row["largest_atom_saddle"] = largest_atom(x - minimum)
                points["search"].append(search)
                points["x"].append(x)
                points["barrier"].append(row["barrier"])
            writer.writerow([text(row.get(field)) for field in FIELDS])
            handle.flush()
    np.savez_compressed(
        output.with_suffix(".npz"),
        search=np.asarray(points["search"], dtype=np.int64),
        x=np.asarray(points["x"], dtype=np.float64).reshape(-1, dof),
        barrier=np.asarray(points["barrier"], dtype=np.float64),
        minimum=minimum,
        lengths=np.asarray(lengths, dtype=np.float64),
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
