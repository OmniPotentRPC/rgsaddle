"""Matched-horizon tables from pt-island campaign runs.

usage: proxy.py OUTDIR HORIZONS ROWS LABEL=RUN ...

RUN is the OUTPUT of one ``campaign.py`` run, whose ``.starts.jsonl``
and ``.summary.json`` sit beside it. A run still going has neither and
is read from the driver's checkpoint, ``.header.json`` and the finished
lines of ``.partial.jsonl``; RGSADDLE_COMMIT names its commit, its goal
is the catalog's 27 and it has no wall time. HORIZONS is a comma-separated list
of search counts, ``all`` for the whole run; OUTDIR's ``summary.csv``
has one row per run and horizon that the run reaches, with the counts
of its first that many searches, ``export.py``'s outcome classes, and
the commit the run recorded. A horizon at which the run ends is left to
its ``all`` row when HORIZONS lists ``all``. ``unique_connected_saddles`` counts the
distinct connected saddles of any barrier, as the driver's
``unique_saddles`` does, and ``campaign_force_calls`` includes the
reactant minimization. ``wall_seconds`` is the run's and is filled on
``all`` rows only. ``searches_sha256`` is the SHA-256 of those searches
in the columns of ``export.py``, header included, so two runs that
agree on every search have the same digest.

``searches_<LABEL>.csv`` holds the first ROWS searches of each run in
those columns, every search for ``all``, and ``saddles_<LABEL>.csv``
every search of the run that found a connected saddle no earlier search
had found, of any barrier.
"""

import csv
import hashlib
import io
import json
import os
import statistics
import sys
from pathlib import Path

from export import SEARCH_FIELDS, number, outcome

GOAL = 27
RESUME_ONLY = ("rng_state", "saddle_x", "climb_calls")
CLASSES = (
    "new_saddle",
    "duplicate",
    "disconnected",
    "over_1.5_eV",
    "wrong_stationary",
    "endpoint_unconverged",
    "no_index1_point",
)


def novel(row):
    """A connected saddle no earlier search found, of any barrier."""
    return bool(row["success"]) or list(row.get("reasons") or []) == ["barrier"]


def load(run):
    """The searches and summary of a run, or of its checkpoint while it runs."""
    if run.with_suffix(".starts.jsonl").is_file():
        rows = [json.loads(text) for text in run.with_suffix(".starts.jsonl").read_text().splitlines() if text.strip()]
        return rows, json.loads(run.with_suffix(".summary.json").read_text())
    header = json.loads(run.with_name(run.stem + ".header.json").read_text())
    # The driver may be writing the last line.
    finished = run.with_name(run.stem + ".partial.jsonl").read_text().split("\n")[:-1]
    rows = [
        {key: value for key, value in json.loads(text).items() if key not in RESUME_ONLY}
        for text in finished
        if text.strip()
    ]
    meta = {
        "template": header["template"],
        "rgsaddle_commit": os.environ["RGSADDLE_COMMIT"],
        "goal": GOAL,
        "minimize_calls": header["minimize_calls"],
        "wall_seconds": None,
    }
    return rows, meta


def line(fields):
    handle = io.StringIO()
    csv.writer(handle, lineterminator="\n").writerow(fields)
    return handle.getvalue()


def search_lines(rows):
    connected = 0
    lines = []
    for index, row in enumerate(rows, start=1):
        connected += bool(row["success"])
        lines.append(
            line(
                [
                    index,
                    row["epicenter"],
                    row["displaced_atoms"],
                    number(bool(row["success"])),
                    row["failure"] or "",
                    outcome(row),
                    number(bool(row["converged"])),
                    row["steps"],
                    row["rotations"],
                    int(row["force_calls"]) - int(row["endpoint_calls"]),
                    row["endpoint_calls"],
                    row["force_calls"],
                    number(row.get("curvature")),
                    number(row.get("hessian_index")),
                    number(row.get("lowest_eigenvalue")),
                    number(row.get("barrier")),
                    connected,
                ]
            )
        )
    return lines


def main():
    out, horizons, limit = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    header = line(SEARCH_FIELDS)
    summary = []
    for arg in sys.argv[4:]:
        label, run = arg.split("=", 1)
        rows, meta = load(Path(run))
        lines = search_lines(rows)
        kept = lines if limit == "all" else lines[: int(limit)]
        (out / f"searches_{label}.csv").write_text(header + "".join(kept))
        found = [text for text, row in zip(lines, rows) if novel(row)]
        (out / f"saddles_{label}.csv").write_text(header + "".join(found))
        for horizon in horizons:
            whole = horizon == "all"
            if not whole and (int(horizon) > len(rows) or (int(horizon) == len(rows) and "all" in horizons)):
                continue
            count = len(rows) if whole else int(horizon)
            head = rows[:count]
            tally = {name: 0 for name in CLASSES}
            for row in head:
                if outcome(row) not in tally:
                    raise SystemExit(f"{label}: {outcome(row)} has no column")
                tally[outcome(row)] += 1
            calls = [int(row["force_calls"]) for row in head]
            climbs = [int(row["force_calls"]) - int(row["endpoint_calls"]) for row in head]
            connected = sum(bool(row["success"]) for row in head)
            digest = hashlib.sha256((header + "".join(lines[:count])).encode()).hexdigest()
            summary.append(
                {
                    "label": label,
                    "template": meta["template"],
                    "commit": meta["rgsaddle_commit"],
                    "horizon": horizon,
                    "searches": len(head),
                    "connected_below_1.5": connected,
                    "goal": meta["goal"],
                    "reached": number(connected >= int(meta["goal"])),
                    "unique_connected_saddles": sum(novel(row) for row in head),
                    "campaign_force_calls": int(meta["minimize_calls"]) + sum(calls),
                    "mean_force_calls_per_search": number(statistics.fmean(calls), 6),
                    "mean_climb_calls_per_search": number(statistics.fmean(climbs), 6),
                    "wall_seconds": number(meta["wall_seconds"], 6) if whole else "",
                    **{name: tally[name] for name in CLASSES[1:]},
                    "searches_sha256": digest,
                }
            )
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
