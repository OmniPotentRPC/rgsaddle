#!/usr/bin/env python3
"""Check cited DOIs, bibliography titles and authors, and nearby names."""

from __future__ import annotations

import fnmatch
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


CITE_RE = re.compile(r"cite:`([^`]+)`")
DOI_THEN_CITE_RE = re.compile(
    r"(?:doi:|https://doi\.org/)(10\.\d{4,9}/[^\s,;@]+?)[.,;)\s]*"
    r"@@rst::cite:`([^`]+)`@@",
    flags=re.IGNORECASE,
)
PROSE_DOI_RE = re.compile(r"\b(10\.\d{4,9}/\S+)", flags=re.IGNORECASE)
FOLD_LETTERS = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "ł": "l",
        "Ł": "L",
        "ß": "ss",
        "đ": "d",
    }
)


def prose_roots() -> tuple[Path, Path]:
    """Return the Rust sources and the org guides scanned for nearby authors."""
    return (ROOT / "src", ROOT / "docs" / "orgmode")


def canonicalize_doi(raw: str) -> str | None:
    """Return a lowercase DOI, or None when *raw* is not a DOI."""
    doi = raw.strip().strip("`~\"'")
    doi = doi.removeprefix("doi:").removeprefix("DOI:")
    while doi and not doi[-1].isalnum():
        doi = doi[:-1]
    while doi and not doi[0].isalnum():
        doi = doi[1:]
    if not doi.lower().startswith("10.") or "/" not in doi:
        return None
    return doi.lower()


def references_bib() -> dict[str, dict[str, str]]:
    """Return cite key to fields for `docs/source/references.bib`."""
    path = ROOT / "docs" / "source" / "references.bib"
    return {key: fields for key, fields in parse_bib_entries(path)}


def check_cites(records: dict[str, dict[str, str]]) -> list[str]:
    """Fail when an org cite key is missing or disagrees with the printed DOI."""
    errors: list[str] = []
    for path in sorted((ROOT / "docs" / "orgmode").rglob("*.org")):
        try:
            label = str(path.resolve().relative_to(ROOT))
        except ValueError:
            label = str(path)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in CITE_RE.finditer(line):
                for key in match.group(1).split(","):
                    key = key.strip()
                    if key and key not in records:
                        errors.append(f"{label}:{line_no}: cite key {key} is not in the bibliography")
            for match in DOI_THEN_CITE_RE.finditer(line):
                doi = canonicalize_doi(match.group(1))
                if doi is None:
                    continue
                for key in (item.strip() for item in match.group(2).split(",")):
                    if not key:
                        continue
                    bib_doi = records.get(key, {}).get("doi")
                    if bib_doi is not None and bib_doi.lower() != doi:
                        errors.append(
                            f"{label}:{line_no}: text cites {doi} beside {key}, "
                            f"but the bibliography has {bib_doi}"
                        )
    return errors


def name_tokens(text: str) -> list[str]:
    """Fold a name to ASCII tokens. Undecodable characters become `?` wildcards."""
    text = re.sub(r"[\u2010-\u2015]", " ", text)
    text = re.sub(r"\\(ae|AE|ss|[oOlL])(?![A-Za-z])", r"\1", text)
    text = re.sub(r"\\[`'^\"~=.cvuHkr]\s*\{?([A-Za-z])\}?", r"\1", text)
    text = text.replace("{", "").replace("}", "").replace("\ufffd", "?")
    text = unicodedata.normalize("NFKD", text.translate(FOLD_LETTERS)).casefold()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.encode("ascii", "replace").decode()
    return [tok for tok in re.split(r"[^0-9a-z?]+", text) if tok]


def tokens_match(local: list[str], remote: list[str]) -> bool:
    """Every token of the shorter name must match a distinct token of the longer.

    A token matches when it is equal, or when one side is a bare initial.
    """
    short, long_ = sorted((local, remote), key=len)
    unused = list(long_)
    pending: list[str] = []
    for tok in short:
        for other in unused:
            if fnmatch.fnmatchcase(tok, other) or fnmatch.fnmatchcase(other, tok):
                unused.remove(other)
                break
        else:
            pending.append(tok)
    for tok in pending:
        for other in unused:
            if (len(tok) == 1 or len(other) == 1) and tok[0] == other[0]:
                unused.remove(other)
                break
        else:
            return False
    return True


def bib_authors(author_field: str) -> list[list[str]]:
    """Split a BibTeX author field into token lists, one person at a time."""
    return [name_tokens(name) for name in re.split(r"\s+and\s+", author_field)]


def doi_registry(doi: str) -> str | None:
    """Return the registration agency name, lowercased, when doi.org knows it."""
    url = "https://doi.org/doiRA/" + urllib.parse.quote(doi, safe="/:")
    try:
        status, payload = fetch_json(url)
    except RuntimeError:
        return None
    if status != 200 or not isinstance(payload, list) or not payload:
        return None
    row = payload[0]
    if not isinstance(row, dict):
        return None
    agency = row.get("RA")
    if isinstance(agency, str) and agency:
        return agency.lower()
    return None


def crossref_authors(doi: str) -> list[list[str]] | None:
    """Return Crossref author token lists, or None when the record has none."""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        status, payload = fetch_json(url)
    except RuntimeError:
        return None
    if status != 200:
        return None
    message = payload.get("message") if isinstance(payload, dict) else None
    rows = message.get("author") if isinstance(message, dict) else None
    if not isinstance(rows, list):
        return None
    parsed: list[list[str]] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        given = row.get("given", "")
        family = row.get("family", "")
        if not isinstance(given, str):
            given = ""
        if not isinstance(family, str):
            family = ""
        parsed.append(name_tokens(f"{given} {family}"))
    return parsed


def check_authors(records: dict[str, dict[str, str]]) -> list[str]:
    """Compare bibliography authors with Crossref, name by name and in order."""
    errors: list[str] = []
    registry: dict[str, str | None] = {}
    for key, fields in records.items():
        doi, author = fields.get("doi"), fields.get("author")
        if not doi or not author:
            continue
        doi_norm = doi.lower()
        if doi_norm not in registry:
            registry[doi_norm] = doi_registry(doi_norm)
        if registry[doi_norm] != "crossref":
            continue
        remote = crossref_authors(doi_norm)
        if not remote:
            continue
        local = bib_authors(author)
        if len(local) != len(remote) or not all(
            tokens_match(left, right) for left, right in zip(local, remote)
        ):
            errors.append(
                f"{key}: author mismatch for {doi_norm}\n"
                f"  bib:      {author}\n"
                f"  crossref: {'; '.join(' '.join(person) for person in remote)}"
            )
    return errors


def prose_files() -> list[Path]:
    """Return Rust and org files whose nearby prose must name a cited author."""
    files: list[Path] = []
    for root in prose_roots():
        files.extend(sorted(root.rglob("*.rs")))
        files.extend(sorted(root.rglob("*.org")))
    return files


def check_prose_authors() -> list[str]:
    """Fail when prose cites a Crossref DOI without naming its first author."""
    errors: list[str] = []
    files = prose_files()
    hits: dict[str, list[tuple[str, int]]] = {}
    texts: dict[str, list[str]] = {}
    for path in files:
        try:
            label = str(path.resolve().relative_to(ROOT))
        except ValueError:
            label = str(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        texts[label] = lines
        for line_no, line in enumerate(lines, start=1):
            for match in PROSE_DOI_RE.finditer(line):
                doi = canonicalize_doi(match.group(1))
                if doi is None:
                    continue
                hits.setdefault(doi, []).append((label, line_no))

    unique = sorted(hits)
    with ThreadPoolExecutor(max_workers=8) as pool:
        agencies = dict(zip(unique, pool.map(doi_registry, unique)))
    crossref_dois = [doi for doi in unique if agencies.get(doi) == "crossref"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        authors = dict(zip(crossref_dois, pool.map(crossref_authors, crossref_dois)))

    for doi in crossref_dois:
        people = authors.get(doi)
        if not people or not people[0]:
            continue
        family = people[0][-1].replace("?", "*")
        for label, line_no in hits[doi]:
            lines = texts[label]
            window = " ".join(lines[max(0, line_no - 7) : line_no + 1])
            if any(
                fnmatch.fnmatchcase(tok, family) or fnmatch.fnmatchcase(family, tok)
                for tok in name_tokens(window)
            ):
                continue
            errors.append(
                f"{label}:{line_no}: {doi} is by {' '.join(people[0])} "
                "but the nearby text does not name that author"
            )
    return errors


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

    alignment: list[str] = []
    if only_bibs is None:
        records = references_bib()
        alignment.extend(check_cites(records))
        alignment.extend(check_authors(records))
        alignment.extend(check_prose_authors())
        for message in alignment:
            print(message, file=sys.stderr)

    if failed or title_errors or alignment:
        return 1
    if only_bibs is None:
        print(
            f"doi links ok ({len(citations)}), bib titles ok ({checked_titles}), "
            "cite keys, authors, and nearby names ok"
        )
    else:
        print(f"doi links ok ({len(citations)}), bib titles ok ({checked_titles})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
