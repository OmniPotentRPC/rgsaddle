#!/usr/bin/env python3
"""Write one reference line per C symbol declared in include/rgsaddle.h.

The how-to stays the narrative. This page is the checklist: every
declared C function, typedef, macro constant, and enum constant, with
the first sentence of the comment above it. Units are the caller's
unless that sentence names them.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "rgsaddle.h"
OUT = ROOT / "docs" / "orgmode" / "reference" / "c-symbols.org"

DECL = re.compile(r"^[ \t]*\w[\w \t\*\n]*?\b((?:rgsaddle|xts)_[A-Za-z0-9_]+)\s*\(", re.M)
MACRO = re.compile(r"^[ \t]*#\s*define\s+((?:RGSADDLE|XTS)_[A-Za-z0-9_]+)\b", re.M)
OPAQUE = re.compile(r"^[ \t]*typedef\s+struct\s+\w+\s+(\w+)\s*;", re.M)
BLOCK_TYPEDEF = re.compile(r"^[ \t]*typedef\s+(struct|enum)\s*\{([^}]*)\}\s*(\w+)\s*;", re.M | re.S)
FN_TYPEDEF = re.compile(r"^[ \t]*typedef\s+[^;]*\(\s*\*\s*(\w+)\s*\)\s*\([^;]*\)\s*;", re.M | re.S)


FEATURE_TAG = re.compile(r"Requires\s+the\s+`(\w+)`\s+feature", re.S)


def required_features(comment: str) -> set[str]:
    """Cargo features a declaration's comment says the library needs for it."""
    flat = re.sub(r"/\*+|\*/|^\s*\*", " ", comment, flags=re.M)
    return set(FEATURE_TAG.findall(flat))


def first_sentence(comment: str, default: str = "Declared in include/rgsaddle.h.") -> str:
    text = re.sub(r"/\*+|\*/", " ", comment)
    text = re.sub(r"^\s*\*", " ", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return default
    stop = text.find(". ")
    if stop == -1:
        return text if text.endswith(".") else text + "."
    return text[: stop + 1]


def preceding_comment(header: str, pos: int) -> str:
    before = header[:pos]
    end = before.rfind("*/")
    if end != -1 and before[end + 2 :].strip() == "":
        start = before.rfind("/*", 0, end)
        if start != -1:
            return before[start : end + 2]
    return ""


def _blank_comments(header: str) -> str:
    return re.sub(
        r"/\*.*?\*/",
        lambda m: "".join("\n" if ch == "\n" else " " for ch in m.group(0)),
        header,
        flags=re.S,
    )


def entries(header: str) -> list[tuple[str, str]]:
    blanked = _blank_comments(header)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in DECL.finditer(blanked):
        name = match.group(1)
        if name in seen or name.endswith("_t") or name.endswith("_fn"):
            continue
        seen.add(name)
        comment = preceding_comment(header, match.start())
        sentence = first_sentence(comment)
        for feature in sorted(required_features(comment)):
            note = f"Requires the ={feature}= feature."
            if note not in sentence:
                sentence = f"{sentence} {note}"
        found.append((name, sentence))
    return found


def type_entries(header: str) -> list[tuple[str, str]]:
    blanked = _blank_comments(header)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in OPAQUE.finditer(blanked):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            comment = preceding_comment(header, match.start())
            if "handle" not in comment.lower() and "stencil" not in comment.lower():
                comment = ""
            found.append(
                (name, first_sentence(comment, "Opaque session handle in include/rgsaddle.h."))
            )
    for match in BLOCK_TYPEDEF.finditer(blanked):
        kind, _body, name = match.group(1), match.group(2), match.group(3)
        if name not in seen:
            seen.add(name)
            comment = preceding_comment(header, match.start())
            found.append(
                (name, first_sentence(comment, f"C {kind} typedef in include/rgsaddle.h."))
            )
    for match in FN_TYPEDEF.finditer(blanked):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            comment = preceding_comment(header, match.start())
            found.append(
                (name, first_sentence(comment, "Callback typedef in include/rgsaddle.h."))
            )
    return found


def constant_entries(header: str) -> list[tuple[str, str]]:
    blanked = _blank_comments(header)
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in MACRO.finditer(blanked):
        name = match.group(1)
        if name == "RGSADDLE_H" or name in seen:
            continue
        seen.add(name)
        comment = preceding_comment(header, match.start())
        found.append((name, first_sentence(comment, "Macro constant in include/rgsaddle.h.")))
    for match in BLOCK_TYPEDEF.finditer(blanked):
        kind, body, enum_name = match.group(1), match.group(2), match.group(3)
        if kind != "enum":
            continue
        for ematch in re.finditer(r"\b(RGSADDLE_[A-Z0-9_]+)\b", body):
            cname = ematch.group(1)
            if cname not in seen:
                seen.add(cname)
                found.append((cname, f"Enumerator of ={enum_name}=."))
    return found


def all_symbols(header: str) -> list[tuple[str, str]]:
    return entries(header) + type_entries(header) + constant_entries(header)


def render(
    pairs: list[tuple[str, str]],
    types: list[tuple[str, str]] | None = None,
    constants: list[tuple[str, str]] | None = None,
) -> str:
    if types is None or constants is None:
        header_text = HEADER.read_text()
        if types is None:
            types = type_entries(header_text)
        if constants is None:
            constants = constant_entries(header_text)
    lines = [
        "#+TITLE: C function index",
        "#+OPTIONS: toc:nil num:nil",
        "",
        "Every function, type, macro, and enum constant declared in =include/rgsaddle.h=.",
        "The sentence is the comment above the declaration.",
        "Parameters are that declaration.",
        "Units are the caller's unless the sentence names them.",
        "The narrative for a session is the C and C++ how-to, not this list.",
        "",
        "* Functions",
        "",
    ]
    for name, sentence in pairs:
        lines.append(
            f"- ={name}=. Purpose: {sentence} Parameters: the declaration. Units: the caller's."
        )
    lines.extend(["", "* Types and callbacks", ""])
    for name, sentence in types:
        lines.append(f"- ={name}=. Purpose: {sentence}")
    lines.extend(["", "* Macros and enum constants", ""])
    for name, sentence in constants:
        lines.append(f"- ={name}=. Purpose: {sentence}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    header = HEADER.read_text()
    pairs = entries(header)
    types = type_entries(header)
    constants = constant_entries(header)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(pairs, types, constants))
    total = len(pairs) + len(types) + len(constants)
    print(
        f"wrote {len(pairs)} C functions, {len(types)} C types, "
        f"{len(constants)} C constants ({total} total)"
    )


if __name__ == "__main__":
    main()
