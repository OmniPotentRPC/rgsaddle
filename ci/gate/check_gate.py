#!/usr/bin/env python3
"""Fail when the committed gate files differ from a fresh export of `plan.ncl`."""

from __future__ import annotations

import argparse
import difflib
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = pathlib.Path("ci/gate/plan.ncl")
OUTPUTS = (
    ("workflow", "yaml", pathlib.Path(".github/workflows/gate.yml")),
    ("plan", "json", pathlib.Path("ci/gate/plan.json")),
)


def drift(root: pathlib.Path, nickel: str = "nickel") -> list[str]:
    """Return a unified diff for every committed output that is not current."""
    diffs: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        for field, fmt, relative in OUTPUTS:
            fresh = pathlib.Path(scratch) / relative.name
            subprocess.run(
                [
                    nickel,
                    "export",
                    "--format",
                    fmt,
                    "--field",
                    field,
                    str(root / SOURCE),
                    "-o",
                    str(fresh),
                ],
                check=True,
            )
            committed = root / relative
            actual = (
                committed.read_text(encoding="utf-8").splitlines(keepends=True)
                if committed.is_file()
                else []
            )
            expected = fresh.read_text(encoding="utf-8").splitlines(keepends=True)
            if actual != expected:
                diffs.append(
                    "".join(
                        difflib.unified_diff(
                            actual,
                            expected,
                            fromfile=f"{relative} (committed)",
                            tofile=f"{relative} (from {SOURCE})",
                        )
                    )
                )
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    try:
        diffs = drift(args.root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"could not export {SOURCE}: {error}", file=sys.stderr)
        return 1
    if diffs:
        sys.stderr.write("".join(diffs))
        print(
            f"generated gate files differ from {SOURCE}; "
            "edit the source and run pixi run gen-gate",
            file=sys.stderr,
        )
        return 1
    print("gate files ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
