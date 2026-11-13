#!/usr/bin/env python3
"""Fail when a tracked file names a machine-local path."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Absolute prefixes only. A relative segment such as out/home/ is not one.
ABS = re.compile(r"(?<![A-Za-z0-9_])/(?:tmp|home|opt)/")


def tracked() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [line for line in out.splitlines() if line]


def main() -> int:
    errors: list[str] = []
    for rel in tracked():
        path = ROOT / rel
        if not path.is_file():
            continue
        if path.suffix in {".png", ".pdf", ".woff", ".woff2"}:
            continue
        if rel == "docs/check_paths.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if ABS.search(line):
                errors.append(f"{rel}:{number}")
    if errors:
        print("local paths:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print("paths ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
