#!/usr/bin/env python3
"""Validate DOI references in public Rust and docs sources."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOI_RE = re.compile(r"\b(10\.\d{4,9}/\S+)", flags=re.I)
SCAN_DIRS = (ROOT / "src", ROOT / "docs" / "orgmode")
SCAN_FILES = (
    ROOT / "docs" / "source" / "references.bib",
    ROOT / "llms.txt",
    ROOT / "llms-full.txt",
)


def fetch_json(url: str) -> object:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "rgsaddle-doi-checker/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def doi_resolves(doi: str) -> tuple[bool, str]:
    req = urllib.request.Request(
        f"https://doi.org/{urllib.parse.quote(doi, safe='/:')}",
        method="GET",
        headers={
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "rgsaddle-doi-checker/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True, "ok"
    except urllib.error.HTTPError as err:
        return False, f"registry response {err.code}"
    except urllib.error.URLError as err:
        return False, f"registry request failed: {err.reason}"


def doi_registry(doi: str) -> str | None:
    try:
        data = fetch_json(f"https://doi.org/doiRA/{urllib.parse.quote(doi, safe='/:')}")
    except urllib.error.HTTPError:
        return None
    if not isinstance(data, list) or not data:
        return None
    row = data[0]
    if not isinstance(row, dict):
        return None
    agency = row.get("RA")
    if isinstance(agency, str) and agency:
        return agency.lower()
    return None


def crossref_title(doi: str) -> tuple[bool, str]:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/:')}"
    try:
        data = fetch_json(url)
    except urllib.error.HTTPError as err:
        return False, f"crossref response {err.code}"
    except urllib.error.URLError as err:
        return False, f"crossref request failed: {err.reason}"
    if not isinstance(data, dict):
        return False, "crossref malformed response"
    msg = data.get("message")
    if not isinstance(msg, dict):
        return False, "crossref malformed response"
    titles = msg.get("title")
    if not isinstance(titles, list) or not titles or not isinstance(titles[0], str):
        return False, "crossref missing title"
    return True, titles[0]


def normalize_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().casefold()
    text = re.sub(r"[^0-9a-z]+", "", text)
    return text


def extract_bib_entries(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text(encoding="utf-8")
    out: list[tuple[str, str, str]] = []
    for m in re.finditer(r"@[\w]+\{([^,]+),", text):
        key = m.group(1).strip()
        start = m.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        body = text[start : pos - 1]
        doi = extract_bib_field(body, "doi")
        title = extract_bib_field(body, "title")
        if doi and title:
            out.append((key, doi, title))
    return out


def extract_bib_field(body: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*([{{\"])", body, flags=re.I)
    if not m:
        return None
    opener = m.group(1)
    pos = m.end()
    if opener == '"':
        end = body.find('"', pos)
        if end == -1:
            return None
        return body[pos:end].strip()
    depth = 1
    start = pos
    while pos < len(body) and depth > 0:
        ch = body[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return body[start : pos - 1].strip()


def collect_dois() -> tuple[dict[str, list[str]], list[tuple[str, str, str]]]:
    hits: dict[str, list[str]] = {}
    files: list[Path] = []
    for root in SCAN_DIRS:
        files.extend(sorted(root.rglob("*.rs")))
        files.extend(sorted(root.rglob("*.org")))
    files.extend([path for path in SCAN_FILES if path.is_file()])

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for ln, line in enumerate(text.splitlines(), start=1):
            for match in DOI_RE.finditer(line):
                doi = canonicalize_doi(match.group(1))
                if doi is None:
                    continue
                hits.setdefault(doi, []).append(f"{rel}:{ln}")
    bib_entries = extract_bib_entries(ROOT / "docs" / "source" / "references.bib")
    return hits, bib_entries


def canonicalize_doi(raw: str) -> str | None:
    doi = raw.strip().strip("`~\"'")
    doi = doi.removeprefix("doi:").removeprefix("DOI:")
    while doi and not doi[-1].isalnum():
        doi = doi[:-1]
    while doi and not doi[0].isalnum():
        doi = doi[1:]
    if not doi.lower().startswith("10."):
        return None
    if "/" not in doi:
        return None
    return doi.lower()


def main() -> int:
    doi_hits, bib_entries = collect_dois()
    errors: list[str] = []
    registry_cache: dict[str, str | None] = {}
    crossref_cache: dict[str, tuple[bool, str]] = {}

    for doi in sorted(doi_hits):
        ok, detail = doi_resolves(doi)
        if not ok:
            for loc in doi_hits[doi]:
                errors.append(f"{doi}: {detail} ({loc})")
            continue
        if doi not in registry_cache:
            registry_cache[doi] = doi_registry(doi)
        if registry_cache[doi] == "crossref":
            crossref_cache[doi] = crossref_title(doi)
            ok_x, detail_x = crossref_cache[doi]
            if not ok_x:
                for loc in doi_hits[doi]:
                    errors.append(f"{doi}: {detail_x} ({loc})")

    for key, doi, title in bib_entries:
        doi_norm = doi.lower()
        if doi_norm not in crossref_cache:
            if doi_norm not in registry_cache:
                registry_cache[doi_norm] = doi_registry(doi_norm)
            if registry_cache[doi_norm] != "crossref":
                continue
            crossref_cache[doi_norm] = crossref_title(doi_norm)
        ok, remote = crossref_cache[doi_norm]
        if not ok:
            continue
        if normalize_title(title) != normalize_title(remote):
            errors.append(
                f"{key}: title mismatch for {doi_norm}\n"
                f"  bib:      {title}\n"
                f"  crossref: {remote}"
            )

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"doi check ok ({len(doi_hits)} unique dois)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
