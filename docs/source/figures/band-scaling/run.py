# Copyright (c) rgsaddle developers
"""Paired timings of examples/band_scaling.rs on two builds.

Build the example on the baseline and on the change, then run from
the repository root:

    python docs/source/figures/band-scaling/run.py \
        <baseline-binary> <change-binary> <out-dir> [blocks]

A baseline without BandSession::set_image_threads runs at one image
thread only. Every point runs in blocks of four serial runs, baseline,
change, change, baseline (ABBA), so a drift of the machine during a
block falls on both builds alike. Each run's line goes to runs.log
and runs.csv in <out-dir>. A run whose hash differs from the
baseline's stops the script: both builds must take the same steps.
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

BAND_STEPS = 60
RTR_STEPS = 100
MINMODE_STEPS = 300
THREADS = range(1, 9)
DOUBLINGS = (1, 2, 4, 8)


def points():
    """(series, point, baseline arguments, change arguments)."""
    for atoms in (64, 512):
        base = ["band", atoms, 16, 1, BAND_STEPS]
        for t in THREADS:
            yield f"strong-{atoms}", t, base, ["band", atoms, 16, t, BAND_STEPS]
    for t in DOUBLINGS:
        yield (
            "weak-images",
            t,
            ["band", 256, 2 * t, 1, BAND_STEPS],
            ["band", 256, 2 * t, t, BAND_STEPS],
        )
    for t in DOUBLINGS:
        atoms = round(128 * math.sqrt(t))
        yield (
            "weak-atoms",
            t,
            ["band", atoms, 8, 1, BAND_STEPS],
            ["band", atoms, 8, t, BAND_STEPS],
        )
    for atoms in (38, 128, 256):
        args = ["band", atoms, 8, 1, RTR_STEPS, "rtr"]
        yield "rtr", atoms, args, args
    for kind in ("dimer", "lanczos"):
        for atoms in (38, 128, 256):
            args = ["minmode", atoms, kind, MINMODE_STEPS]
            yield kind, atoms, args, args


def run(binary: str, args: list) -> dict:
    line = subprocess.run(
        [binary, *map(str, args)], check=True, capture_output=True, text=True
    ).stdout.strip()
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    fields["line"] = line
    return fields


def main() -> None:
    baseline, change, out = sys.argv[1:4]
    blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["series", "point", "block", "slot", "build", "steps", "rows",
               "evaluations", "seconds", "hash", "args"]
    with open(out / "runs.csv", "w", newline="") as table, \
            open(out / "runs.log", "w") as log:
        writer = csv.DictWriter(table, columns)
        writer.writeheader()
        for series, point, base_args, change_args in points():
            expected = None
            for block in range(blocks):
                order = [("baseline", baseline, base_args),
                         ("change", change, change_args),
                         ("change", change, change_args),
                         ("baseline", baseline, base_args)]
                for slot, (build, binary, args) in enumerate(order):
                    fields = run(binary, args)
                    expected = expected or fields["hash"]
                    if fields["hash"] != expected:
                        sys.exit(f"{series} {point}: {fields['line']}")
                    log.write(f"{series} {point} {block} {slot} {build} "
                              f"{fields['line']}\n")
                    writer.writerow({
                        "series": series, "point": point, "block": block,
                        "slot": slot, "build": build,
                        "steps": fields["steps"],
                        "rows": fields.get("rows", ""),
                        "evaluations": fields.get("evaluations", ""),
                        "seconds": fields["seconds"], "hash": fields["hash"],
                        "args": " ".join(map(str, args)),
                    })
                table.flush()
                log.flush()
            print(series, point, "done", flush=True)


if __name__ == "__main__":
    main()
