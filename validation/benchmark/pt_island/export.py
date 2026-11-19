"""CSV tables from the pt-island campaign records.

usage: export.py RUNDIR EON_JSONL OUTDIR [HORIZON]

RUNDIR holds ``<method>.starts.jsonl`` and ``<method>.summary.json``
from ``campaign.py`` for the four methods. EON_JSONL is the
``searches.jsonl`` of ``eon_catalog.py``. OUTDIR receives one
``searches_<method>.csv`` per method, ``searches_eon_<flavour>.csv`` and
``summary.csv``. The eOn row of ``summary.csv`` counts its first
HORIZON searches, the rgsaddle campaigns' length, when HORIZON is set;
``searches_eon_<flavour>.csv`` keeps every search. EON_FLAVOUR is
the ``eon_catalog.py`` flavour of EON_JSONL, classic unless set.
RGSADDLE_COMMIT names the commit of a RUNDIR whose summaries do not
record one.

``outcome`` splits optbench's failure class. The driver files a
converged index-1 saddle that is a repeat, does not connect to the
reactant, or lies 1.5 eV or more above it under ``divergence``, the
same class as a climb that blows up; ``outcome`` names which it was,
taking the first of disconnected, over_1.5_eV and duplicate that
applies, for rgsaddle and eOn alike. ``wrong_stationary`` is a
converged point whose Hessian index is not one.
"""

import csv
import json
import os
import statistics
import sys
from pathlib import Path

METHODS = ("dimer-fire", "dimer-lbfgs", "dimer-rotskip", "lanczos")
SEARCH_FIELDS = (
    "search",
    "epicenter",
    "displaced_atoms",
    "success",
    "failure",
    "outcome",
    "converged",
    "steps",
    "rotations",
    "climb_calls",
    "endpoint_calls",
    "force_calls",
    "curvature",
    "hessian_index",
    "lowest_eigenvalue",
    "barrier",
    "connected_so_far",
)


def outcome(row):
    reasons = row.get("reasons") or []
    if row["success"]:
        return "new_saddle"
    if row.get("hessian_index") is None:
        return "no_index1_point"
    if "hessian_index" in reasons:
        return "wrong_stationary"
    if "bad_minima" in reasons:
        return "endpoint_unconverged"
    if "not_connected" in reasons:
        return "disconnected"
    if "barrier" in reasons:
        return "over_1.5_eV"
    if "duplicate" in reasons:
        return "duplicate"
    return "+".join(reasons) or "unknown"


def number(value, digits=8):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def main():
    rundir, eon_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for method in METHODS:
        rows = [json.loads(line) for line in (rundir / f"{method}.starts.jsonl").read_text().splitlines() if line.strip()]
        meta = json.loads((rundir / f"{method}.summary.json").read_text())
        connected = 0
        tally = {}
        with (out / f"searches_{method}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(SEARCH_FIELDS)
            for index, row in enumerate(rows, start=1):
                connected += bool(row["success"])
                label = outcome(row)
                tally[label] = tally.get(label, 0) + 1
                climb = int(row["force_calls"]) - int(row["endpoint_calls"])
                writer.writerow(
                    [
                        index,
                        row["epicenter"],
                        row["displaced_atoms"],
                        number(bool(row["success"])),
                        row["failure"] or "",
                        label,
                        number(bool(row["converged"])),
                        row["steps"],
                        row["rotations"],
                        climb,
                        row["endpoint_calls"],
                        row["force_calls"],
                        number(row.get("curvature")),
                        number(row.get("hessian_index")),
                        number(row.get("lowest_eigenvalue")),
                        number(row.get("barrier")),
                        connected,
                    ]
                )
        calls = [int(row["force_calls"]) for row in rows]
        climbs = [int(row["force_calls"]) - int(row["endpoint_calls"]) for row in rows]
        summary_rows.append(
            {
                "code": "rgsaddle",
                "method": method,
                "commit": meta.get("rgsaddle_commit") or os.environ["RGSADDLE_COMMIT"],
                "searches": meta["total_searches"],
                "connected_below_1.5": meta["connected"],
                "goal": meta["goal"],
                "reached": number(bool(meta["reached"])),
                "unique_connected_saddles": meta["unique_saddles"],
                "fraction_connected": number(meta["fraction_connected"], 6),
                "campaign_force_calls": meta["force_calls"],
                "mean_force_calls_per_search": number(statistics.fmean(calls), 6),
                "mean_climb_calls_per_search": number(statistics.fmean(climbs), 6),
                "wall_seconds": number(meta["wall_seconds"], 6),
                "disconnected": tally.get("disconnected", 0),
                "over_1.5_eV": tally.get("over_1.5_eV", 0),
                "duplicate": tally.get("duplicate", 0),
                "wrong_stationary": tally.get("wrong_stationary", 0),
                "endpoint_unconverged": tally.get("endpoint_unconverged", 0),
                "no_index1_point": tally.get("no_index1_point", 0),
                "high_energy_abort": 0,
            }
        )
    eon_all = [json.loads(line) for line in eon_path.read_text().splitlines() if line.strip()]
    eon = eon_all[: int(sys.argv[4])] if len(sys.argv) > 4 else eon_all
    fields = (
        "search",
        "epicenter",
        "termination_reason",
        "saddle_calls",
        "endpoint_calls",
        "barrier",
        "connected",
        "novel",
        "success",
        "harness_force_norm",
        "largest_atom_displacement",
        "good_so_far",
    )
    tally = {}
    flavour = os.environ.get("EON_FLAVOUR", "classic")
    with (out / f"searches_eon_{flavour}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for row in eon_all:
            writer.writerow([number(row.get(name)) for name in fields])
        for row in eon:
            if row["termination_reason"] == 3:
                label = "high_energy_abort"
            elif row["termination_reason"] != 0:
                label = "no_index1_point"
            elif row["success"]:
                label = "new_saddle"
            elif not row["connected"]:
                label = "disconnected"
            elif row["barrier"] >= 1.5:
                label = "over_1.5_eV"
            else:
                label = "duplicate"
            tally[label] = tally.get(label, 0) + 1
    calls = [row["saddle_calls"] + row.get("endpoint_calls", 0) for row in eon]
    summary_rows.append(
        {
            "code": "eOn 3.5.0",
            "method": f"{flavour}-lbfgs",
            "commit": "",
            "searches": len(eon),
            "connected_below_1.5": eon[-1]["good_so_far"],
            "goal": 27,
            "reached": number(eon[-1]["good_so_far"] >= 27),
            "unique_connected_saddles": sum(bool(row.get("novel")) for row in eon),
            "fraction_connected": number(sum(row.get("connected", True) is not False for row in eon) / len(eon), 6),
            "campaign_force_calls": sum(calls),
            "mean_force_calls_per_search": number(statistics.fmean(calls), 6),
            "mean_climb_calls_per_search": number(statistics.fmean(row["saddle_calls"] for row in eon), 6),
            "wall_seconds": number(sum(row["eon_seconds"] for row in eon), 6),
            "disconnected": tally.get("disconnected", 0),
            "over_1.5_eV": tally.get("over_1.5_eV", 0),
            "duplicate": tally.get("duplicate", 0),
            "wrong_stationary": 0,
            "endpoint_unconverged": 0,
            "no_index1_point": tally.get("no_index1_point", 0),
            "high_energy_abort": tally.get("high_energy_abort", 0),
        }
    )
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
