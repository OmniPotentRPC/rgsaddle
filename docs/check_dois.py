#!/usr/bin/env python3
"""Check every DOI cited by the public documentation against the registry."""

from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOI_URL = re.compile(r"https://doi\.org/([^\s>)\]]+)", re.IGNORECASE)
DOI_TEXT = re.compile(
    r"\bdoi:(10\.[-._;()/:<>A-Z0-9]+)",
    re.IGNORECASE,
)
DOI_FIELD = re.compile(r"^\s*doi\s*=\s*[{\"']([^}\"']+)", re.IGNORECASE | re.MULTILINE)
ENTRY_START = re.compile(r"@\w+\s*[{(]\s*([^,\s]+)\s*,", re.IGNORECASE)
SUFFIXES = {".bib", ".h", ".md", ".org", ".rs"}


def cited_dois() -> dict[str, list[str]]:
    """Return normalized DOI strings and the files that cite them."""
    citations: dict[str, list[str]] = {}
    roots = [ROOT / "README.md", ROOT / "docs", ROOT / "include", ROOT / "src"]
    files = [roots[0]]
    for directory in roots[1:]:
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in SUFFIXES
            and "build" not in path.parts
            and (path.suffix == ".bib" or "source" not in path.parts[:-1])
        )
    for path in files:
        text = path.read_text()
        url_dois = [
            urllib.parse.unquote(match.group(1)).rstrip(".,;")
            for match in DOI_URL.finditer(text)
        ]
        without_urls = DOI_URL.sub("", text)
        text_dois = [
            match.group(1).rstrip(".,;:)")
            for match in DOI_TEXT.finditer(without_urls)
        ]
        field_dois = [match.group(1) for match in DOI_FIELD.finditer(without_urls)]
        dois = url_dois + text_dois + field_dois
        for doi in dois:
            citations.setdefault(doi.lower(), []).append(str(path.relative_to(ROOT)))
    return citations


def registry_check(doi: str) -> str | None:
    """Return an error when the DOI handle registry does not know *doi*."""
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rgsaddle-documentation-link-check/0.6"},
    )
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if payload.get("responseCode") == 1 and payload.get("values"):
                return None
            return f"registry response {payload.get('responseCode')}"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
            time.sleep(2**attempt)
    return last_error


def bib_entries() -> list[tuple[str, str, str]]:
    """Return bibliography keys, titles, and normalized DOI strings."""
    text = (ROOT / "docs" / "source" / "references.bib").read_text()
    entries: list[tuple[str, str, str]] = []
    for match in ENTRY_START.finditer(text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        body = text[start : index - 1]
        title = bib_field(body, "title")
        doi = bib_field(body, "doi")
        if title is not None and doi is not None:
            entries.append((match.group(1), title, doi.lower()))
    return entries


def bib_field(body: str, name: str) -> str | None:
    """Read one braced or quoted field from a BibTeX entry body."""
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*=\s*([{{\"])", body)
    if match is None:
        return None
    opener = match.group(1)
    closer = "}" if opener == "{" else '"'
    depth = 1
    index = match.end()
    start = index
    while index < len(body):
        char = body[index]
        if opener == "{" and char == "{":
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return body[start:index].strip()
        index += 1
    return None


def metadata_title(doi: str) -> tuple[str | None, str | None]:
    """Return a title from Crossref or DataCite, and an error if both fail."""
    quoted = urllib.parse.quote(doi, safe="")
    requests = [
        ("Crossref", f"https://api.crossref.org/works/{quoted}"),
        ("DataCite", f"https://api.datacite.org/dois/{quoted}"),
    ]
    errors = []
    for registry, url in requests:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "rgsaddle-documentation-link-check/0.6 "
                    "(mailto:rgoswami@ieee.org)"
                )
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                if registry == "Crossref":
                    titles = payload.get("message", {}).get("title", [])
                else:
                    titles = payload.get("data", {}).get("attributes", {}).get("titles", [])
                    titles = [item.get("title") for item in titles if item.get("title")]
                if titles:
                    return titles[0], None
                errors.append(f"{registry}: no title")
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 2:
                    errors.append(f"{registry}: {error}")
                    break
                time.sleep(2**attempt)
            except (OSError, ValueError, urllib.error.URLError) as error:
                if attempt == 2:
                    errors.append(f"{registry}: {error}")
                    break
                time.sleep(2**attempt)
    return None, "; ".join(errors)


def normalized_title(title: str) -> str:
    """Normalize registry HTML and BibTeX markup for title comparison."""
    title = html.unescape(re.sub(r"<[^>]+>", "", title))
    title = re.sub(r"\\[`'\"^~=.uvHckbdtr]\s*{?([A-Za-z])}?", r"\1", title)
    title = re.sub(r"\\[A-Za-z]+\s*{([^{}]*)}", r"\1", title)
    title = title.replace("{", "").replace("}", "")
    title = unicodedata.normalize("NFKD", title).casefold()
    return "".join(char for char in title if char.isalnum())


def title_check(entry: tuple[str, str, str]) -> str | None:
    """Return an error when a bibliography title differs from its registry."""
    key, title, doi = entry
    registered, error = metadata_title(doi)
    if registered is None:
        return f"{key}: {error}"
    if normalized_title(title) != normalized_title(registered):
        return f"{key}: {title!r} != {registered!r}"
    return None


def main() -> int:
    citations = cited_dois()
    with ThreadPoolExecutor(max_workers=12) as pool:
        errors = dict(zip(citations, pool.map(registry_check, citations)))
    failed = {doi: error for doi, error in errors.items() if error is not None}
    if failed:
        for doi, error in sorted(failed.items()):
            sources = ", ".join(sorted(set(citations[doi])))
            print(f"{doi}: {error} ({sources})", file=sys.stderr)
        return 1
    entries = bib_entries()
    # Crossref's public API rate limits short bursts even with a polite
    # identifying user agent. Keep title lookups serial and let retries
    # handle transient throttling.
    with ThreadPoolExecutor(max_workers=1) as pool:
        title_errors = [error for error in pool.map(title_check, entries) if error is not None]
    if title_errors:
        for error in title_errors:
            print(error, file=sys.stderr)
        return 1
    print(f"doi links ok ({len(citations)}); bibliography titles ok ({len(entries)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
