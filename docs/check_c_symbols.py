#!/usr/bin/env python3
"""Fail when a declared C function is missing from the generated index."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import render_c_symbols  # noqa: E402


def main() -> int:
    header = render_c_symbols.HEADER.read_text()
    pairs = render_c_symbols.entries(header)
    page = render_c_symbols.OUT.read_text()
    missing = [name for name, _ in pairs if f"={name}=" not in page]
    if missing:
        print("C index is missing:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    fresh = render_c_symbols.render(pairs)
    if fresh != page:
        print("C index is stale; run docs/render_c_symbols.py", file=sys.stderr)
        return 1
    print(f"c symbols ok ({len(pairs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
