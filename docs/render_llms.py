#!/usr/bin/env python3
"""Render llms.txt and llms-full.txt at the repository root from docs/orgmode."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG_DIR = ROOT / "docs" / "orgmode"
LLMS_OUT = ROOT / "llms.txt"
LLMS_FULL_OUT = ROOT / "llms-full.txt"
SITE_URL = "https://omnipotentrpc.github.io/rgsaddle"

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Core guides",
        [
            ("index.org", "Overview, installation, and host-controlled stepping loop"),
            ("getting-started.org", "First band session on an analytic double-well surface"),
            ("howto.org", "Task index for bands, minimum-mode searches, C ABI, and rates"),
            ("reference.org", "Exhaustive inventory of Rust modules, types, functions, configs, C ABI, and Python bindings"),
            ("explanation.org", "Mathematical and numerical design of band, string, and minimum-mode methods"),
            ("seat.org", "Role of rgsaddle in the leidarljos and rgpot stack"),
        ],
    ),
    (
        "Tutorials",
        [
            ("tutorials/band.org", "Nudged elastic band and OCI-NEB on the Mueller-Brown surface"),
            ("tutorials/dimer.org", "Dimer and Lanczos minimum-mode searches"),
            ("tutorials/sella.org", "Order-1 saddle optimization with Sella's P-RFO trust step"),
            ("tutorials/quickstart.org", "Intrinsic reaction coordinate (IRC) forward and reverse branches"),
            (
                "tutorials/pafi.org",
                "PAFI free-energy profile and rate on an analytic surface, in Rust and Python",
            ),
            (
                "tutorials/oh-tst.org",
                "OH-TST from a dimer seed through the optimal plane, transmission, and a rate, in Rust and Python",
            ),
        ],
    ),
    (
        "How-to guides",
        [
            ("howto/band.org", "NEB, climbing image, OCI-NEB, simplified and growing string, free-end adaptive NEB, IDPP, AFIR, and connect"),
            ("howto/autoneb.org", "AutoNEB window insertion and spring scaling"),
            (
                "howto/line-neb.org",
                "Fixed-energy line-integral NEB, endpoint policy, and motion along the path",
            ),
            ("howto/minmode.org", "Dimer, Lanczos, soft-mode seeding, surrogate Hessian actions, saddle index, and ARTn protocol"),
            ("howto/art.org", "Phased Activation-Relaxation Technique nouveau (ArtSession)"),
            ("howto/direction-cover.org", "Signed per-atom direction cover and arm picker for saddle searches from a minimum (DirectionCover)"),
            ("howto/indexk.org", "Index-k minimum-mode searches with LOBPCG and block Lanczos"),
            ("howto/landscape.org", "Downward pathway maps from a certified saddle to minima"),
            ("howto/solid-dimer.org", "Variable-cell solid-state dimer on atoms and strain"),
            ("howto/ssneb.org", "Generalized solid-state NEB on the joint atom-strain chart"),
            ("howto/strain-rate.org", "Elastic dipoles, multipoles, FMM fields, and strain response of a hop"),
            ("howto/defect-pair-rates.org", "Defect-pair hop rates, PAFI free energy, Bennett-Chandler kappa, and pair networks"),
            ("howto/npt-lattice.org", "Isotropic NPT lattice sampling with blocking standard errors"),
            (
                "howto/pafi.org",
                "Projected average force integration: a free-energy profile along a path at finite temperature",
            ),
            ("howto/oh-tst.org", "Optimal hyperplanar transition state theory (OH-TST) and reversible work"),
            ("howto/lammps-hyperplane.org", "Plane-constrained BAOAB sampling with LAMMPS through lammpc"),
            (
                "howto/rgpot.org",
                "Wrap an rgpot potential as a PointSurface and run relaxation, a dimer search, and PAFI",
            ),
            (
                "howto/pafi-ohtst-restarting.org",
                "Continue a PAFI error target, or extend an OH-TST plane sample from the last configuration",
            ),
            (
                "howto/pafi-ohtst-parallel-sampling.org",
                "PAFI samples its planes together; each OH-TST call keeps one sequential chain",
            ),
            (
                "howto/pafi-ohtst-checking-convergence.org",
                "Read PAFI effective-sample diagnostics and OH-TST quadrature node counts",
            ),
            ("howto/sparse-hessian.org", "Direct Curtis-Powell-Reid sparse Hessian recovery by distance-2 colouring"),
            ("howto/sella.org", "P-RFO, restricted P-RFO, Index1Session, Sella sessions, internal coordinates, cell charts, and SAMD"),
            ("howto/irc.org", "Intrinsic reaction coordinate (IRC) integration with SellaSphere, GS2, and Morokuma steps"),
            (
                "howto/geodesic.org",
                "Shorten an initial path in a redundant coordinate chart before any potential call",
            ),
            ("howto/lineb.org", "Fixed-energy tunneling paths, imaginary-time replicas, fluctuation spectra, and thermal rates"),
            ("howto/path-initialization.org", "Choosing linear, IDPP, or TRICS path initializers across NEB, AutoNEB, and C"),
            ("howto/io.org", "Reading and writing CON frames and restart bands"),
            ("howto/c-abi.org", "Stepping bands, minimum-mode sessions, Sella, IRC, and strings from C and C++"),
            ("howto/meson.org", "Consuming rgsaddle headers from Meson wrap-git or pkg-config"),
            ("howto/python.org", "Python nanobind surface, SurfaceAdapter, and ASE drop-in classes"),
            ("howto/migrate.org", "Migrating NEB, dimer, and Sella workflows from ASE or eOn"),
            ("howto/stencil.org", "Unified point stencil builder across Rust, C, C++, and Python"),
            ("howto/formal.org", "Lean 4 formal proofs and MPFR differential verification"),
        ],
    ),
    (
        "Explanation and reference",
        [
            ("explanation/mep.org", "Mass-weighted intrinsic reaction coordinates and unweighted minimum energy paths"),
            (
                "explanation/ohtst-pafi-exact.org",
                "Deterministic integrals used to check sampled PAFI and OH-TST estimates",
            ),
            ("explanation/one-stencil.org", "Householder reflection shared by climbing-image NEB and minimum-mode searches"),
            ("explanation/hyperplane-methods.org", "PAFI, OH-TST, and constrained-ensemble samplers on one geometric footing"),
            ("explanation/symmetry-regions.org", "Crystal symmetry operations, path multiplicity, and hard Voronoi regions"),
            ("explanation/trics.org", "Translation-rotation internal coordinates, stability diagnostic maze, and segment splicing"),
            (
                "explanation/pafi-ohtst.org",
                "Transition state theory shared by PAFI, OH-TST, and harmonic TST",
            ),
            (
                "explanation/elastic-dipole.org",
                "Finite-cell elastic dipole correction for periodic images, and the benchmark row labels",
            ),
            ("reference/architecture.org", "Crate architecture and module boundaries"),
            ("reference/lammps-hyperplane.org", "LAMMPS hyperplane sampler contract and C ABI hooks"),
            ("reference/bibliography.org", "Literature bibliography"),
            ("reference/c-symbols.org", "Complete C ABI symbol index for include/rgsaddle.h"),
            (
                "reference/pafi.org",
                "Public PAFI configuration, result types, and the functions that connect them",
            ),
            ("reference/ohtst.org", "Public OH-TST configuration and result types"),
            (
                "explanation/cyclic-block.org",
                "Rotation fallback, singular criterion, symmetric part, and residual norm of the cyclic block factor",
            ),
            (
                "reference/block-cyclic.org",
                "Cyclic block-tridiagonal factor, inertia, and the solve residual",
            ),
            ("contributing/index.org", "Build, test, MSRV, Python stub, documentation, and release workflow"),
        ],
    ),
]


def parse_title(org_text: str, fallback: str) -> str:
    for line in org_text.splitlines():
        match = re.match(r"^#\+title:\s*(.+)$", line, flags=re.I)
        if match:
            return match.group(1).strip()
    return fallback


def org_to_markdown(org_text: str, org_path: Path) -> str:
    lines = org_text.splitlines()
    out: list[str] = []
    in_export = False
    in_src = False

    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("#+title:") or lower.startswith("#+options:") or lower.startswith("#+startup:"):
            continue
        if lower.startswith("#+attr_rst:") or lower.startswith("#+caption:"):
            continue
        if lower.startswith("#+begin_export"):
            in_export = True
            continue
        if lower.startswith("#+end_export"):
            in_export = False
            continue
        if in_export:
            continue
        if lower.startswith("#+include:"):
            inc_match = re.search(r'"([^"]+)"\s+src\s+(\w+)(?:\s+:lines\s+"(\d+)-(\d+)")?', line)
            if inc_match:
                rel_inc, lang, start_s, end_s = inc_match.groups()
                inc_path = (org_path.parent / rel_inc).resolve()
                if inc_path.is_file():
                    inc_lines = inc_path.read_text(encoding="utf-8").splitlines()
                    if start_s and end_s:
                        start_i = max(int(start_s) - 1, 0)
                        end_i = int(end_s)
                        inc_lines = inc_lines[start_i:end_i]
                    out.append(f"```{lang}")
                    out.extend(inc_lines)
                    out.append("```")
            continue
        if lower.startswith("#+begin_src"):
            parts = line.strip().split()
            lang = parts[1] if len(parts) > 1 else ""
            out.append(f"```{lang}")
            in_src = True
            continue
        if lower.startswith("#+end_src"):
            out.append("```")
            in_src = False
            continue
        if lower.startswith("#+begin_example"):
            out.append("```text")
            in_src = True
            continue
        if lower.startswith("#+end_example"):
            out.append("```")
            in_src = False
            continue
        if in_src:
            out.append(line)
            continue

        heading = re.match(r"^(\*+)\s+(.+)$", line)
        if heading:
            level = min(len(heading.group(1)) + 1, 6)
            title = clean_inline_org(heading.group(2))
            out.append(f"{'#' * level} {title}")
            continue

        if re.match(r"^\|[-+]+\|$", line.strip()):
            cols = line.strip().strip("|").split("+")
            out.append("| " + " | ".join("---" for _ in cols) + " |")
            continue

        if line.startswith(": "):
            out.append("    " + line[2:])
            continue

        out.append(clean_inline_org(line))

    text = "\n".join(out).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text + "\n"


# Closing punctuation of an Org emphasis marker. org-element matches
# whitespace or one of these, or the end of the line:
#   ! " ' ) , - . : ; ? [ \ }
# A table separator is absent on purpose. Org splits the row into cells
# before the scan, and the end of a cell is the border.
_ORG_POST = frozenset("!\"'),-.:;?[\\}")
# Characters that may precede an opening marker, besides whitespace
# and the start of the line.
_ORG_PRE = frozenset("-('\"{")


def _org_opens(text: str, index: int) -> bool:
    if index + 1 >= len(text) or text[index + 1].isspace():
        return False
    if index == 0:
        return True
    prev = text[index - 1]
    return prev.isspace() or prev in _ORG_PRE


def _org_post(text: str, index: int) -> bool:
    if index + 1 >= len(text):
        return True
    char = text[index + 1]
    return char.isspace() or char in _ORG_POST


def _replace_org_mono_span(text: str) -> str:
    """Turn Org verbatim and code markers into Markdown inline code.

    A closing ``=`` or ``~`` is the first marker of the same kind that
    is preceded by non-whitespace and followed by whitespace, punctuation,
    or the end of the text. Bytes between the markers are copied unchanged.
    """
    out: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        if text[index] in "=~" and _org_opens(text, index):
            mark = text[index]
            close = None
            for cand in range(index + 2, size):
                if (
                    text[cand] == mark
                    and not text[cand - 1].isspace()
                    and _org_post(text, cand)
                ):
                    close = cand
                    break
            if close is not None:
                out.append("`")
                out.append(text[index + 1 : close])
                out.append("`")
                index = close + 1
                continue
        out.append(text[index])
        index += 1
    return "".join(out)


def _replace_org_mono(text: str) -> str:
    """Scan ``text`` for verbatim and code spans.

    Org splits a table row into cells before the emphasis scan, so each
    cell is scanned on its own. A closer that sits against the separator
    is then at the end of the cell. The separator is not punctuation on
    any other line.
    """
    if text.lstrip(" \t").startswith("|"):
        return "|".join(_replace_org_mono_span(cell) for cell in text.split("|"))
    return _replace_org_mono_span(text)


def clean_inline_org(text: str) -> str:
    text = re.sub(r"@@rst::doc:`([^`<]+?)\s*<([^>]+)>`@@", r"[\1](\2.html)", text)
    text = re.sub(r"@@rst::doc:`([^`]+)`@@", r"`\1`", text)
    text = re.sub(r"@@rst::cite:`([^`]+)`@@", r"[\1]", text)
    text = re.sub(r"@@rst:([^@]+)@@", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\[([^\]]+)\]\]", r"[\2](\1)", text)
    text = re.sub(r"\[\[file:([^\]]+)\]\]", r"(\1)", text)
    return _replace_org_mono(text)


def render_llms_txt() -> str:
    lines = [
        "# rgsaddle",
        "",
        "> Band, minimum-mode, and index-1 Newton saddle mechanics in Rust, C, C++, and Python. Every session takes one solver step on an assembled force or Hessian model while the host owns the outer loop.",
        "",
        "rgsaddle is system-agnostic and unit-agnostic: surfaces supply energies, gradients, and optional cell stresses or directional Hessian actions in the caller's units.",
        "",
        "- Documentation site: https://omnipotentrpc.github.io/rgsaddle/",
        "- Full documentation: https://omnipotentrpc.github.io/rgsaddle/llms-full.txt (repo root `llms-full.txt`)",
        "- Rust API docs: https://docs.rs/rgsaddle",
        "- Repository: https://github.com/OmniPotentRPC/rgsaddle",
        "",
    ]
    for section_title, entries in SECTIONS:
        lines.append(f"## {section_title}")
        lines.append("")
        for rel_org, summary in entries:
            org_path = ORG_DIR / rel_org
            title = parse_title(org_path.read_text(encoding="utf-8"), rel_org)
            html_rel = rel_org[:-4] + ".html"
            url = f"{SITE_URL}/{html_rel}"
            lines.append(f"- [{title}]({url}): {summary} (`docs/orgmode/{rel_org}`)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_llms_full_txt() -> str:
    chunks = [
        "# rgsaddle: Complete Documentation Corpus",
        "",
        "> Generated from `docs/orgmode/*.org` by `docs/render_llms.py`. Edit the Org sources and rerun `python3 docs/render_llms.py`.",
        "",
    ]
    for _section_title, entries in SECTIONS:
        for rel_org, summary in entries:
            org_path = ORG_DIR / rel_org
            raw = org_path.read_text(encoding="utf-8")
            title = parse_title(raw, rel_org)
            body = org_to_markdown(raw, org_path)
            chunks.append(f"# {title} (`docs/orgmode/{rel_org}`)")
            chunks.append("")
            chunks.append(f"> {summary}")
            chunks.append("")
            chunks.append(body.rstrip())
            chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def listed_guides() -> set[str]:
    return {rel for _title, entries in SECTIONS for rel, _summary in entries}


def unlisted_guides() -> list[str]:
    """Org pages under docs/orgmode that SECTIONS does not name.

    changelog.org is the towncrier sink, not a guide.
    """
    listed = listed_guides()
    missing: list[str] = []
    for path in sorted(ORG_DIR.rglob("*.org")):
        rel = path.relative_to(ORG_DIR).as_posix()
        if rel == "changelog.org" or rel in listed:
            continue
        missing.append(rel)
    return missing


def main_check_only() -> int:
    missing = unlisted_guides()
    if missing:
        print(
            "Org guides missing from the llms page list:\n" + "\n".join(missing),
            file=sys.stderr,
        )
        return 1
    expected_llms = render_llms_txt()
    expected_full = render_llms_full_txt()
    if not LLMS_OUT.is_file() or LLMS_OUT.read_text(encoding="utf-8") != expected_llms:
        print("llms.txt is missing or stale; run python3 docs/render_llms.py", file=sys.stderr)
        return 1
    if not LLMS_FULL_OUT.is_file() or LLMS_FULL_OUT.read_text(encoding="utf-8") != expected_full:
        print(
            "llms-full.txt is missing or stale; run python3 docs/render_llms.py",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that llms.txt and llms-full.txt match the current Org sources.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write llms.txt and llms-full.txt from the Org sources.",
    )
    args = parser.parse_args()

    if args.check and args.write:
        parser.error("pass only one of --check and --write")

    if args.check:
        rc = main_check_only()
        if rc == 0:
            print("llms.txt and llms-full.txt ok")
        return rc

    if unlisted_guides():
        return main_check_only()

    expected_llms = render_llms_txt()
    expected_full = render_llms_full_txt()

    LLMS_OUT.write_text(expected_llms, encoding="utf-8")
    LLMS_FULL_OUT.write_text(expected_full, encoding="utf-8")
    print(
        f"wrote {LLMS_OUT.name} ({len(expected_llms)} bytes) and "
        f"{LLMS_FULL_OUT.name} ({len(expected_full)} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
