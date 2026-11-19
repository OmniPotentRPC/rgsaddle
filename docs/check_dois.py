#!/usr/bin/env python3
"""Check every DOI cited by the public documentation against the registry."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOI_URL = re.compile(r"https://doi\.org/([^\s>)\]\"'}]+)", re.IGNORECASE)
DOI_TEXT = re.compile(
    r"\bdoi:(10\.[-._;()/:<>A-Z0-9]+)",
    re.IGNORECASE,
)
DOI_FIELD = re.compile(r"^\s*doi\s*=\s*[{\"']([^}\"']+)", re.IGNORECASE | re.MULTILINE)
SUFFIXES = {
    ".bib",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".lean",
    ".md",
    ".org",
    ".py",
    ".pyi",
    ".rs",
}


def tracked_sources() -> list[Path]:
    """Return the tracked text files that may cite a paper."""
    listing = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    own = Path(__file__).resolve()
    return [
        ROOT / name
        for name in listing.splitlines()
        if Path(name).suffix in SUFFIXES
        and (ROOT / name).is_file()
        and (ROOT / name).resolve() != own
    ]


def cited_dois() -> dict[str, list[str]]:
    """Return normalized DOI strings and the files that cite them."""
    citations: dict[str, list[str]] = {}
    files = tracked_sources()
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
        headers={"User-Agent": "rgsaddle-documentation-link-check/1.0"},
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
            if attempt < 2:
                time.sleep(2**attempt)
    return last_error


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
    print(f"doi links ok ({len(citations)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
