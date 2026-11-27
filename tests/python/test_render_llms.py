"""Org verbatim and code spans survive the guide renderer."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "docs" / "render_llms.py"
if not _SCRIPT.is_file():
    pytest.skip(
        "docs/render_llms.py is not available in this tree",
        allow_module_level=True,
    )
_SPEC = importlib.util.spec_from_file_location("render_llms", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("docs/render_llms.py could not be loaded")
render_llms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_llms)

_PUB_TYPE = re.compile(
    r"^pub (?:struct|enum|trait|type) ([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_CURVATURE = (
    "if the rejected curvature is finite and non-negative",
    "a finite non-negative rejected curvature",
    "When the restarts are spent and the rejected curvature is finite and non-negative",
)


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
    for block in re.finditer(r"pub use [^;]+;", text, flags=re.DOTALL):
        for name in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", block.group(0)):
            if not name.isupper():
                names.add(name)
    return names


def _missing(names: set[str], page: str) -> list[str]:
    absent = []
    for name in sorted(names):
        if (
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", page)
            is None
        ):
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
    org = (_REPO / "docs" / "orgmode" / "howto" / "pafi.org").read_text(
        encoding="utf-8"
    )
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


_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`]*`")
_LATEX = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
_SLASH = re.compile(r"=/=")
_BESIDE = re.compile(r"(?:=`|`=)")
_VERBATIM = re.compile(r"=[A-Za-z_][A-Za-z0-9_]*=")
_STRAY = re.compile(r"\)=\s|[\"']=")
_CMP = re.compile(r"[<>]`\s+\S*=")


def _guide_problems(text: str) -> list[str]:
    """Report broken verbatim left in rendered Markdown.

    Fenced blocks are ignored. The checks read the finished text: a
    slash join, an equals sign against a backtick, a comparison that
    closed early, a tight ``=name=`` span outside inline code, or a
    closer left after a quote or parenthesis.
    """
    body = _FENCE.sub("", text)
    problems: list[str] = []
    for number, line in enumerate(body.splitlines(), 1):
        if _SLASH.search(line):
            problems.append(f"{number}: collapsed slash span")
        if _BESIDE.search(line):
            problems.append(f"{number}: equals beside a backtick")
        if _CMP.search(line):
            problems.append(f"{number}: comparison split out of a code span")
        prose = _LATEX.sub("", _INLINE.sub("", line))
        match = _VERBATIM.search(prose)
        if match is not None:
            problems.append(f"{number}: verbatim left in prose ({match.group(0)})")
        if _STRAY.search(prose):
            problems.append(f"{number}: stray closer outside a code span")
    return problems


def test_rendered_guides_keep_verbatim_inside_code() -> None:
    problems: list[str] = []
    for name in ("llms-full.txt", "llms.txt"):
        text = (_REPO / name).read_text(encoding="utf-8")
        problems.extend(f"{name}:{item}" for item in _guide_problems(text))
    assert problems == []
    full = (_REPO / "llms-full.txt").read_text(encoding="utf-8")
    for clause in _CURVATURE:
        assert clause in full


def test_pafi_and_ohtst_pages_name_every_public_type() -> None:
    pafi_names = _pub_types(_REPO / "src" / "pafi.rs")
    pafi_names.add("arc_at_time")
    ohtst_names = _pub_types(_REPO / "src" / "ohtst")
    ohtst_names.update(_reexported_types(_REPO / "src" / "ohtst" / "mod.rs"))
    pafi_page = (_REPO / "docs" / "orgmode" / "reference" / "pafi.org").read_text(
        encoding="utf-8"
    )
    ohtst_page = (_REPO / "docs" / "orgmode" / "reference" / "ohtst.org").read_text(
        encoding="utf-8"
    )
    assert _missing(pafi_names, pafi_page) == []
    assert _missing(ohtst_names, ohtst_page) == []
