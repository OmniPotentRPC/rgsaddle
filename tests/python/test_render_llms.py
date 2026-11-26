"""Org verbatim and code spans survive the guide renderer."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "docs" / "render_llms.py"
_SPEC = importlib.util.spec_from_file_location("render_llms", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("docs/render_llms.py could not be loaded")
render_llms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_llms)

# Org emphasis post-marker characters, from org-element's closing regexp:
# whitespace, or one of ! " ' ) , - . : ; ? [ \ }
_POST = frozenset("!\"'),-.:;?[\\}")
_PRE = frozenset("-('\"{")
_PUB_TYPE = re.compile(
    r"^pub (?:struct|enum|trait|type) ([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
_CURVATURE = (
    "if the rejected curvature is finite and non-negative",
    "a finite non-negative rejected curvature",
    "When the restarts are spent and the rejected curvature is finite and non-negative",
)


def _space(char: str) -> bool:
    return char.isspace()


def _opens(text: str, index: int) -> bool:
    if index + 1 >= len(text) or _space(text[index + 1]):
        return False
    if index == 0:
        return True
    prev = text[index - 1]
    return _space(prev) or prev in _PRE


def _post(text: str, index: int) -> bool:
    if index + 1 >= len(text):
        return True
    char = text[index + 1]
    return _space(char) or char in _POST


def _mono_line(text: str) -> list[str]:
    found: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        if text[index] in "=~" and _opens(text, index):
            mark = text[index]
            close = None
            for cand in range(index + 2, size):
                if text[cand] == mark and not _space(text[cand - 1]) and _post(text, cand):
                    close = cand
                    break
            if close is not None:
                found.append(text[index + 1 : close])
                index = close + 1
                continue
        index += 1
    return found


def mono_contents(text: str) -> list[str]:
    """Inner text of each Org =verbatim= and ~code~ span on one line.

    The closer is the first marker preceded by non-whitespace and followed
    by whitespace, punctuation, or the end of the line. The bytes between
    the markers are the span. A table row is split into cells first, so the
    end of a cell is a border. The separator is not punctuation.
    """
    if text.lstrip(" \t").startswith("|"):
        found: list[str] = []
        for cell in text.split("|"):
            found.extend(_mono_line(cell))
        return found
    return _mono_line(text)


def _pub_types(*paths: Path) -> set[str]:
    names: set[str] = set()
    for path in paths:
        files = path.rglob("*.rs") if path.is_dir() else [path]
        for file in files:
            names.update(_PUB_TYPE.findall(file.read_text(encoding="utf-8")))
    return names


def _reexported_types(path: Path) -> set[str]:
    names: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for block in re.finditer(r"pub use [^;]+;", text, flags=re.S):
        for name in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", block.group(0)):
            if not name.isupper():
                names.add(name)
    return names


def _missing(names: set[str], page: str) -> list[str]:
    absent = []
    for name in sorted(names):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", page) is None:
            absent.append(name)
    return absent


@pytest.mark.parametrize(
    ("src", "want"),
    [
        ("=thermal = 0.02585=", "`thermal = 0.02585`"),
        ("=nodes >= 2=", "`nodes >` 2="),
        ("| =nodes >= 2=| =kept=|", "| `nodes >` 2=| `kept`|"),
    ],
)
def test_org_mono_unit_cases(src: str, want: str) -> None:
    assert render_llms.clean_inline_org(src) == want


def test_code_span_keeps_comparison() -> None:
    assert render_llms.clean_inline_org("~nodes >= 2~") == "`nodes >= 2`"


def test_adjacent_verbatim_spans_stay_separate() -> None:
    assert render_llms.clean_inline_org("=foo= and =bar=") == "`foo` and `bar`"


def test_pipe_outside_a_table_does_not_close_verbatim() -> None:
    assert render_llms.clean_inline_org("=foo=| bar") == "=foo=| bar"


def test_pafi_table_cells_close_at_the_separator() -> None:
    """Rows 83, 84, and 89 of the PAFI howto have no space before `|`."""
    org = (_REPO / "docs" / "orgmode" / "howto" / "pafi.org").read_text(encoding="utf-8")
    lines = org.splitlines()
    want = {
        83: (
            "| `trajectories`| 1                   | independent trajectories per plane; "
            "with more than one, the error is the spread of their means, as in the reference code |"
        ),
        84: (
            "| `integrator`  | `LeimkuhlerMatthews`| `PafiIntegrator::EulerMaruyama` "
            "is the direct step of the paper's Eq. (9)                |"
        ),
        89: (
            "| `target_error`| `None`              | stop when the barrier error is at or "
            "below this; `None` runs one pass                    |"
        ),
    }
    for number, expected in want.items():
        assert render_llms.clean_inline_org(lines[number - 1]) == expected


@pytest.mark.thread_unsafe
def test_every_org_mono_span_is_inline_code() -> None:
    seen: list[str] = []
    real = render_llms.clean_inline_org

    def spy(text: str) -> str:
        seen.append(text)
        return real(text)

    original = render_llms.clean_inline_org
    render_llms.clean_inline_org = spy
    try:
        full = render_llms.render_llms_full_txt()
    finally:
        render_llms.clean_inline_org = original
    missing: list[str] = []
    for line in seen:
        for content in mono_contents(line):
            needle = f"`{content}`"
            if needle not in full:
                missing.append(content)
    assert missing == []
    for clause in _CURVATURE:
        assert clause in full


def test_pafi_and_ohtst_pages_name_every_public_type() -> None:
    pafi_names = _pub_types(_REPO / "src" / "pafi.rs")
    pafi_names.add("arc_at_time")
    ohtst_names = _pub_types(_REPO / "src" / "ohtst")
    ohtst_names.update(_reexported_types(_REPO / "src" / "ohtst" / "mod.rs"))
    pafi_page = (_REPO / "docs" / "orgmode" / "reference" / "pafi.org").read_text(encoding="utf-8")
    ohtst_page = (_REPO / "docs" / "orgmode" / "reference" / "ohtst.org").read_text(encoding="utf-8")
    assert _missing(pafi_names, pafi_page) == []
    assert _missing(ohtst_names, ohtst_page) == []
