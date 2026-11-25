#!/usr/bin/env python3
"""Check every cited DOI against the registry and verify .bib titles."""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "rgsaddle-documentation-link-check/0.6 (mailto:rgoswami@ieee.org)"
DOI_URL = re.compile(
    r"https://doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9%]+)",
    re.IGNORECASE,
)
DOI_TEXT = re.compile(
    r"\bdoi:(10\.[-._;()/:<>A-Z0-9]+)",
    re.IGNORECASE,
)
DOI_FIELD = re.compile(r"^\s*doi\s*=\s*[{\"']([^}\"']+)", re.IGNORECASE | re.MULTILINE)
BIB_ENTRY = re.compile(r"(?ms)^@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)(?=^@\w+\s*\{|\Z)")
BIB_FIELD = re.compile(r"(?m)^\s*([A-Za-z0-9_-]+)\s*=\s*")
DOCUMENTATION_SUFFIXES = {
    ".adoc",
    ".asciidoc",
    ".bib",
    ".c",
    ".cc",
    ".cpp",
    ".csv",
    ".h",
    ".hpp",
    ".html",
    ".json",
    ".lean",
    ".md",
    ".mdx",
    ".org",
    ".py",
    ".pyi",
    ".rs",
    ".rst",
    ".sh",
    ".sollya",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

LATEX_DROP_MACROS = re.compile(
    r"\\(?:"
    r"emph|textit|textbf|textrm|textsc|textsf|texttt|textup|textnormal|text|"
    r"mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|"
    r"bar|overline|underline|tilde|widetilde|hat|widehat|acute|grave|"
    r"ddot|dot|check|breve|vec|"
    r"to|rightarrow|leftarrow|leftrightarrow|longrightarrow|mapsto|"
    r"pm|mp|times|cdot|sim|approx|leq|geq|neq|infty|partial|nabla|"
    r"left|right|big|Big|bigg|Bigg"
    r")\b"
)
LATEX_ACCENT = re.compile(r"\\(?:['`^\"~=.])\{?([A-Za-z])\}?|\\[uvHckbd](?:\{([A-Za-z])\}|\s+([A-Za-z]))")
LATEX_SPECIAL = {
    r"\o": "o",
    r"\O": "O",
    r"\l": "l",
    r"\L": "L",
    r"\aa": "a",
    r"\AA": "A",
    r"\ae": "ae",
    r"\AE": "AE",
    r"\oe": "oe",
    r"\OE": "OE",
    r"\ss": "ss",
}
UNICODE_FOLD = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "ł": "l",
        "Ł": "l",
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
        "ß": "ss",
    }
)
GREEK_NAMES = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ϵ": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ϑ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ς": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "ϕ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}


def normalize_title(raw: str) -> str:
    """Fold case, accents, LaTeX/HTML markup, and punctuation for comparison."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", "", text)
    for macro, replacement in LATEX_SPECIAL.items():
        text = re.sub(re.escape(macro) + r"(?![A-Za-z])", replacement, text)
    text = LATEX_ACCENT.sub(
        lambda m: m.group(1) or m.group(2) or m.group(3) or "",
        text,
    )
    text = LATEX_DROP_MACROS.sub(" ", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = re.sub(r"\\.", " ", text)
    text = text.translate(UNICODE_FOLD)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    for letter, name in GREEK_NAMES.items():
        text = text.replace(letter, name)
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_bib_entries(path: Path) -> list[tuple[str, dict[str, str]]]:
    """Return `(cite_key, lowercased_fields)` records from a BibTeX file."""
    text = path.read_text(encoding="utf-8")
    entries: list[tuple[str, dict[str, str]]] = []
    for match in BIB_ENTRY.finditer(text):
        entry_type, key, body = match.group(1), match.group(2), match.group(3)
        if entry_type.lower() in {"comment", "preamble", "string"}:
            continue
        fields: dict[str, str] = {}
        pos = 0
        end = len(body)
        while pos < end:
            field_match = BIB_FIELD.search(body, pos)
            if field_match is None:
                break
            name = field_match.group(1).lower()
            val_start = field_match.end()
            if val_start >= end:
                break
            opener = body[val_start]
            if opener == "{":
                depth = 1
                cursor = val_start + 1
                while cursor < end and depth > 0:
                    if body[cursor] == "{":
                        depth += 1
                    elif body[cursor] == "}":
                        depth -= 1
                    cursor += 1
                fields[name] = body[val_start + 1 : cursor - 1].strip()
                pos = cursor
            elif opener == '"':
                cursor = val_start + 1
                while cursor < end and body[cursor] != '"':
                    if body[cursor] == "\\":
                        cursor += 1
                    cursor += 1
                fields[name] = body[val_start + 1 : cursor].strip()
                pos = cursor + 1
            else:
                bare = re.match(r"[^,\n}]+", body[val_start:])
                token = bare.group(0).strip() if bare else ""
                fields[name] = token
                pos = val_start + (bare.end() if bare else 1)
        entries.append((key, fields))
    return entries


def tracked_documentation_files() -> list[Path]:
    """Return tracked files whose extensions can contain documentation."""
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return sorted(
        ROOT / raw.decode("utf-8")
        for raw in output.split(b"\0")
        if raw and Path(raw.decode("utf-8")).suffix.lower() in DOCUMENTATION_SUFFIXES
    )


def bib_files() -> list[Path]:
    """Return tracked BibTeX files under `docs/`."""
    return [
        path
        for path in tracked_documentation_files()
        if path.suffix.lower() == ".bib" and path.is_relative_to(ROOT / "docs")
    ]


def normalize_doi_url(raw: str) -> str:
    """Decode a DOI URL target and remove surrounding prose punctuation."""
    doi = urllib.parse.unquote(raw).rstrip(".,;")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    return doi


def cited_dois(only_bibs: list[Path] | None = None) -> dict[str, list[str]]:
    """Return normalized DOI strings and the files that cite them."""
    citations: dict[str, list[str]] = {}
    if only_bibs is not None:
        files = only_bibs
    else:
        files = tracked_documentation_files()
    for path in files:
        text = path.read_text(encoding="utf-8")
        url_dois = [normalize_doi_url(match.group(1)) for match in DOI_URL.finditer(text)]
        without_urls = DOI_URL.sub("", text)
        text_dois = [
            match.group(1).rstrip(".,;:)")
            for match in DOI_TEXT.finditer(without_urls)
        ]
        field_dois = [match.group(1) for match in DOI_FIELD.finditer(without_urls)]
        dois = url_dois + text_dois + field_dois
        try:
            label = str(path.resolve().relative_to(ROOT))
        except ValueError:
            label = str(path)
        for doi in dois:
            citations.setdefault(doi.lower(), []).append(label)
    return citations


def fetch_json(url: str) -> tuple[int, dict]:
    """Fetch JSON from *url* with retries on transient errors and rate limits."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return 404, {}
            last_error = f"HTTP {error.code}"
            retry_after = error.headers.get("Retry-After") if error.headers else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * (attempt + 1)
            if error.code in {408, 429, 500, 502, 503, 504}:
                time.sleep(delay)
                continue
            raise RuntimeError(last_error) from error
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
            time.sleep(2**attempt)
    raise RuntimeError(last_error)


def registry_check(doi: str) -> str | None:
    """Return an error when the DOI handle registry does not know *doi*."""
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/")
    try:
        status, payload = fetch_json(url)
    except RuntimeError as error:
        return str(error)
    if status == 200 and payload.get("responseCode") == 1 and payload.get("values"):
        return None
    return f"registry response {payload.get('responseCode', status)}"


def fetch_registered_titles(doi: str) -> tuple[str, list[str]]:
    """Return `(source, candidate_titles)` from Crossref, or DataCite on 404."""
    cr_url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    status, payload = fetch_json(cr_url)
    if status == 200:
        message = payload.get("message") or {}
        titles = [t for t in (message.get("title") or []) if t]
        subtitles = [s for s in (message.get("subtitle") or []) if s]
        candidates = list(titles)
        if titles and subtitles:
            candidates.append(f"{titles[0]}: {subtitles[0]}")
        if not candidates:
            raise RuntimeError("Crossref record has no title")
        return "Crossref", candidates

    dc_url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="/")
    dc_status, dc_payload = fetch_json(dc_url)
    if dc_status == 200:
        attrs = (dc_payload.get("data") or {}).get("attributes") or {}
        candidates = [
            item["title"]
            for item in (attrs.get("titles") or [])
            if isinstance(item, dict) and item.get("title")
        ]
        if not candidates:
            raise RuntimeError("DataCite record has no title")
        return "DataCite", candidates

    raise RuntimeError("DOI not found in Crossref or DataCite")


def check_bib_titles(paths: list[Path]) -> tuple[int, list[str]]:
    """Compare every DOI-bearing `.bib` entry's title against Crossref/DataCite."""
    records: list[tuple[str, str, str, str]] = []
    errors: list[str] = []
    for path in paths:
        try:
            label = str(path.resolve().relative_to(ROOT))
        except ValueError:
            label = str(path)
        for key, fields in parse_bib_entries(path):
            doi = fields.get("doi", "").strip()
            if not doi:
                continue
            title = fields.get("title", "").strip()
            if not title:
                errors.append(f"{label}:{key} ({doi}): missing title field")
                continue
            records.append((label, key, doi, title))

    unique_dois = sorted({doi.lower() for _, _, doi, _ in records})

    def lookup(doi: str) -> tuple[str, list[str] | None, str | None]:
        try:
            source, titles = fetch_registered_titles(doi)
            return source, titles, None
        except RuntimeError as error:
            return "", None, str(error)

    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = dict(zip(unique_dois, pool.map(lookup, unique_dois)))

    for label, key, doi, bib_title in records:
        source, candidates, err = fetched[doi.lower()]
        if err is not None or not candidates:
            errors.append(f"{label}:{key} ({doi}): {err}")
            continue
        bib_norm = normalize_title(bib_title)
        cand_norms = [normalize_title(candidate) for candidate in candidates]
        if not bib_norm or bib_norm not in cand_norms:
            clean_cr = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(candidates[0]))).strip()
            errors.append(
                f"{label}:{key} ({doi}): title mismatch\n"
                f"  bib: {bib_title}\n"
                f"  {source}: {clean_cr}"
            )
    return len(records), errors


def main(argv: list[str]) -> int:
    only_bibs = [Path(arg) for arg in argv] if argv else None
    citations = cited_dois(only_bibs)
    with ThreadPoolExecutor(max_workers=12) as pool:
        errors = dict(zip(citations, pool.map(registry_check, citations)))
    failed = {doi: error for doi, error in errors.items() if error is not None}
    for doi, error in sorted(failed.items()):
        sources = ", ".join(sorted(set(citations[doi])))
        print(f"{doi}: {error} ({sources})", file=sys.stderr)

    target_bibs = only_bibs if only_bibs is not None else bib_files()
    checked_titles, title_errors = check_bib_titles(target_bibs)
    for message in title_errors:
        print(message, file=sys.stderr)

    if failed or title_errors:
        return 1
    print(f"doi links ok ({len(citations)}), bib titles ok ({checked_titles})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
