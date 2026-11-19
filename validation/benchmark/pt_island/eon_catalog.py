"""eOn saddle searches from the starts of the rgsaddle pt-island campaign.

usage: eon_catalog.py FLAVOUR MAX_SEARCHES OUTDIR

FLAVOUR is ``classic`` (the dimer of optbench's eon/develop template:
improved false, steepest-descent rotation), ``improved`` (improved
dimer, conjugate-gradient rotation) or ``lanczos``. OPTBENCH_ROOT is an
optbench checkout and EONCLIENT the eonclient binary.

Search ``k`` displaces the minimized reactant with the ``k``-th draw of
``optbench.displacement.displace`` from seed 20261016, the draw that
search ``k`` of ``campaign.py`` uses, and writes it with the random mode
as ``displacement.con`` and ``direction.dat`` for an eOn
``saddle_search`` job. eOn climbs on its own ``morse_pt``. The saddle it
returns is judged on the optbench surface as the driver judges an
rgsaddle saddle: L-BFGS endpoint minimizations from 0.2 along the
saddle mode, the connection and repeat tests at distance 0.1 and energy
0.01, and the barrier against 1.5 eV. The Hessian index is not checked.
The run stops at 27 unique connected saddles below 1.5 eV.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CONFIG = """[Main]
job = saddle_search
finite_difference = 0.001
random_seed = 1

[Potential]
potential = morse_pt

[Optimizer]
opt_method = lbfgs
convergence_metric = norm
converged_force = 0.001
max_move = 0.2
max_iterations = 1000
lbfgs_memory = 20

[Saddle Search]
min_mode_method = {method}
max_energy = 5
max_iterations = 1000
converged_force = 0.001

[Dimer]
improved = {improved}
opt_method = {rotation}
converged_angle = 5
rotations_max = 20
rotations_min = 1
"""


def write_con(template, path, coords):
    lines = template.splitlines()
    k = 0
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 5 and parts[3] in ("0", "1") and k < len(coords):
            x, y, z = coords[k]
            lines[i] = f"{x:.12f}\t{y:.12f}\t{z:.12f}\t{parts[3]}\t{parts[4]}"
            k += 1
    if k != len(coords):
        raise RuntimeError(f"{path}: wrote {k} of {len(coords)} atoms")
    path.write_text("\n".join(lines) + "\n")


def main():
    from optbench import rgsaddle_minmode as M
    from optbench.campaign import resolve_problem
    from optbench.connection import ProcessTable, periodic_lengths
    from optbench.connection import connected as linked
    from optbench.displacement import displace, neighbours
    from optbench.surfaces import load_con, potential_from_problem

    flavour, limit, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["OPTBENCH_ROOT"])
    eonclient = os.environ["EONCLIENT"]
    problem = resolve_problem(root, "saddle-search/pt-island")
    geometries = problem["input"]["geometries"]
    con_path = M._stored(root, next(p for p in geometries if p.endswith(".con")))
    template = con_path.read_text()
    positions, fixed, cell = load_con(con_path)
    free = np.flatnonzero(~fixed)
    evaluate = M._bind(potential_from_problem(problem), positions, free, cell)
    minimum, _ = M._lbfgs_minimize(M.HarnessSurface(evaluate), positions[free].reshape(-1).copy(), 1e-3, 1000)
    emin = evaluate(minimum)[0]
    lengths = periodic_lengths(cell, problem["potential"]["pbc"])
    reactant = positions.copy()
    reactant[free] = minimum.reshape(-1, 3)
    listed = list(range(7))
    near = neighbours(reactant, lengths, 3.3, listed)
    rng = np.random.default_rng(M.CATALOG_SEED + 1)
    config = CONFIG.format(
        method="lanczos" if flavour == "lanczos" else "dimer",
        improved="true" if flavour == "improved" else "false",
        rotation="cg" if flavour == "improved" else "sd",
    )
    found = ProcessTable(lengths, 0.1, 0.01)
    good = 0
    with (out / "searches.jsonl").open("w") as log:
        for search in range(1, limit + 1):
            epicenter, shift, seed = displace(rng, ~fixed, listed, near, 0.1)
            work = out / f"s{search}"
            work.mkdir(exist_ok=True)
            write_con(template, work / "pos.con", reactant)
            write_con(template, work / "displacement.con", reactant + shift)
            np.savetxt(work / "direction.dat", seed, fmt="%.12f")
            (work / "config.ini").write_text(config)
            started = time.time()
            subprocess.run([eonclient], cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            wall = time.time() - started
            results = {}
            for line in (work / "results.dat").read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    results[parts[1]] = parts[0]
            row = {
                "search": search,
                "epicenter": epicenter,
                "termination_reason": int(results.get("termination_reason", -1)),
                "saddle_calls": int(results.get("force_calls_saddle", 0)),
                "eon_seconds": wall,
            }
            if row["termination_reason"] == 0 and (work / "saddle.con").is_file():
                point, _, _ = load_con(work / "saddle.con")
                x = point[free].reshape(-1)
                mode = np.loadtxt(work / "mode.dat").reshape(-1, 3)[free].reshape(-1)
                mode /= np.linalg.norm(mode)
                energy, gradient = evaluate(x)
                barrier = energy - emin
                judge = M.HarnessSurface(evaluate)
                minus, minus_ok = M._lbfgs_minimize(judge, x - 0.2 * mode, 1e-3, 1000)
                plus, plus_ok = M._lbfgs_minimize(judge, x + 0.2 * mode, 1e-3, 1000)
                link = bool(minus_ok and plus_ok and linked(minimum, plus, minus, lengths, 0.1))
                novel = link and not found.repeat(x, barrier)
                if novel:
                    found.add(x, barrier)
                success = bool(novel and barrier < 1.5)
                good += success
                row.update(
                    barrier=barrier,
                    harness_force_norm=float(np.linalg.norm(gradient)),
                    connected=link,
                    novel=bool(novel),
                    success=success,
                    endpoint_calls=judge.calls,
                    largest_atom_displacement=float(np.max(np.linalg.norm((x - minimum).reshape(-1, 3), axis=1))),
                )
            row["good_so_far"] = good
            shutil.rmtree(work)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if good >= 27:
                break


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
