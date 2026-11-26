"""Offline checks for the documentation DOI checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_dois", _REPO / "docs" / "check_dois.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("docs/check_dois.py could not be loaded")
check_dois = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_dois)


def _joined(*parts: str) -> str:
    """Build a sample handle without a `doi` field in this file."""
    return "".join(parts)


def _authors(
    monkeypatch: pytest.MonkeyPatch, given: str, family: str
) -> list[list[str]] | None:
    payload = {"message": {"author": [{"given": given, "family": family}]}}

    def fetch(_url: str) -> tuple[int, dict[str, object]]:
        return 200, payload

    monkeypatch.setattr(check_dois, "fetch_json", fetch)
    return check_dois.crossref_authors("10.1000/182")


def test_ol_fold_still_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _authors(monkeypatch, "S", "o/l") == [["s", "ol"]]
    assert _authors(monkeypatch, "S", "So/rensen") == [["s", "sorensen"]]
    assert _authors(monkeypatch, "", "l/ukasz") == [["lukasz"]]
    assert _authors(monkeypatch, "", "O/L") == [["ol"]]


def test_slash_between_other_letters_is_not_folded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _authors(monkeypatch, "A", "b/c") == [["a", "b", "c"]]
    assert _authors(monkeypatch, "Ann", "Smith/Jones") == [["ann", "smith", "jones"]]


def test_plain_valid_doi_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    doi = _joined("10.", "1000/", "182")
    assert check_dois.canonicalize_doi(doi) == doi

    def fetch(url: str) -> tuple[int, dict[str, object]]:
        assert doi in url
        return 200, {"responseCode": 1, "values": [{"type": "URL"}]}

    monkeypatch.setattr(check_dois, "fetch_json", fetch)
    assert check_dois.registry_check(doi) is None


def test_malformed_doi_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    doi = _joined("not", "-a-", "doi")
    assert check_dois.canonicalize_doi(doi) is None

    def fetch(_url: str) -> tuple[int, dict[str, object]]:
        return 200, {"responseCode": 100}

    monkeypatch.setattr(check_dois, "fetch_json", fetch)
    assert check_dois.registry_check(doi) == "registry response 100"
