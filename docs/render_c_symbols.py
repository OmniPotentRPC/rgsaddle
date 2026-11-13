#!/usr/bin/env python3
"""Write one reference line per C function declared in include/rgsaddle.h.

The how-to stays the narrative. This page is the checklist: every
declared rgsaddle_* function, then the first sentence of the comment
above it. Units are the caller's unless that sentence names them.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "rgsaddle.h"
OUT = ROOT / "docs" / "orgmode" / "reference" / "c-symbols.org"

DECL = re.compile(r"^[\w\s\*]+\b(rgsaddle_[A-Za-z0-9_]+)\s*\(", re.M)


def first_sentence(comment: str) -> str:
    text = re.sub(r"/\*+|\*/", " ", comment)
    text = re.sub(r"^\s*\*", " ", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Declared in include/rgsaddle.h."
    stop = text.find(". ")
    if stop == -1:
        return text if text.endswith(".") else text + "."
    return text[: stop + 1]


def entries(header: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in DECL.finditer(header):
        name = match.group(1)
        if name in seen or name.endswith("_t"):
            continue
        seen.add(name)
        before = header[: match.start()]
        comment = ""
        end = before.rfind("*/")
        if end != -1 and before[end + 2 :].strip() == "":
            start = before.rfind("/*", 0, end)
            if start != -1:
                comment = before[start : end + 2]
        found.append((name, first_sentence(comment)))
    return found


def render(pairs: list[tuple[str, str]]) -> str:
    lines = [
        "#+TITLE: C function index",
        "#+OPTIONS: toc:nil num:nil",
        "",
        "Every function declared in =include/rgsaddle.h=.",
        "The sentence is the comment above the declaration.",
        "Parameters are that declaration.",
        "Units are the caller's unless the sentence names them.",
        "The narrative for a session is the C and C++ how-to, not this list.",
        "",
    ]
    for name, sentence in pairs:
        lines.append(f"- ={name}=. Purpose: {sentence} Parameters: the declaration. Units: the caller's.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    pairs = entries(HEADER.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(pairs))
    print(f"wrote {len(pairs)} C functions")


if __name__ == "__main__":
    main()
