"""One rgsaddle template on optbench saddle-search/pt-island.

usage: campaign.py TEMPLATE OUTPUT [MAX_SEARCHES]

OPTBENCH_ROOT is an optbench checkout, RGSADDLE_LIB the librgsaddle
built with ``cargo build --release --features capi`` and RGSADDLE_SRC
the rgsaddle checkout it was built from. The driver is
``optbench.rgsaddle_minmode.run_template`` at repetition 1, so the
catalog is seed 20261016 and the stop is 27 connected saddles below
1.5 eV or ``max_searches``. MAX_SEARCHES overrides the template's
``[ini.Search] max_searches``. Next to OUTPUT the driver writes the
per-search ``.starts.jsonl``; this script adds ``.summary.json`` with
the repetition summary, the wall time and the two commits. A run that
is stopped resumes from the driver's checkpoint; ``.wall`` keeps the
seconds of the earlier segments, and the summary reports their sum.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

def head(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def main():
    from optbench.campaign import load_template, resolve_problem
    from optbench.rgsaddle_minmode import run_template

    root = Path(os.environ["OPTBENCH_ROOT"])
    name, output = sys.argv[1], Path(sys.argv[2])
    template = load_template(root, name)
    rgsaddle = head(os.environ["RGSADDLE_SRC"])
    # The template pins the commit it was recorded with; this run names its own.
    template["commit"] = rgsaddle
    template["code_version"] = f"({rgsaddle[:8]})"
    if len(sys.argv) > 3:
        template["ini"]["Search"]["max_searches"] = int(sys.argv[3])
    problem = resolve_problem(root, "saddle-search/pt-island")
    ledger = output.with_suffix(".wall")
    earlier = float(ledger.read_text()) if ledger.is_file() else 0.0
    started = time.time()
    try:
        result = run_template(root, problem, template, output=output, repetition=1)
    finally:
        ledger.write_text(f"{earlier + time.time() - started}\n")
    summary = dict(result["repetitions"][0])
    summary["wall_seconds"] = float(ledger.read_text())
    summary["template"] = name
    summary["rgsaddle_commit"] = rgsaddle
    summary["optbench_commit"] = head(root)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.splitlines()[2])
    else:
        main()
