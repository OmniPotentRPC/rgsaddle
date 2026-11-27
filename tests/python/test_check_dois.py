"""Offline checks for the documentation DOI checker."""

from __future__ import annotations

import email.utils
import importlib.util
import io
import json
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "docs" / "check_dois.py"
if not _SCRIPT.is_file():
    pytest.skip(
        "docs/check_dois.py is not available in this tree",
        allow_module_level=True,
    )
_SPEC = importlib.util.spec_from_file_location("check_dois", _SCRIPT)
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
    assert _authors(monkeypatch, "So/ren", "Larsen") == [["soren", "larsen"]]


def test_slash_between_other_letters_is_not_folded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _authors(monkeypatch, "A", "b/c") == [["a", "b", "c"]]
    assert _authors(monkeypatch, "Ann", "Smith/Jones") == [["ann", "smith", "jones"]]
    assert _authors(monkeypatch, "b/c", "Ann") == [["b", "c", "ann"]]


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


def _sample_doi() -> str:
    return _joined("10.", "1000/", "182")


def _sample_bib(tmp_path: Path) -> Path:
    doi = _sample_doi()
    path = tmp_path / "sample.bib"
    path.write_text(
        "@article{sample,\n  title = {Sample title},\n  doi = {" + doi + "},\n}\n",
        encoding="utf-8",
    )
    return path


def _url_of(request: object) -> str:
    full = getattr(request, "full_url", None)
    return full if isinstance(full, str) else str(request)


def _http_error(url: str, code: int, retry_after: str | None = None) -> HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(url, code, "error", headers, io.BytesIO(b""))


class _JSONResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.status = status
        self._data = json.dumps(payload).encode()

    def read(self, amt: int = -1) -> bytes:
        if amt is None or amt < 0:
            chunk = self._data
            self._data = b""
            return chunk
        chunk = self._data[:amt]
        self._data = self._data[amt:]
        return chunk

    def __enter__(self) -> "_JSONResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _handle_ok() -> _JSONResponse:
    return _JSONResponse({"responseCode": 1, "values": [{"type": "URL"}]})


def _crossref_ok() -> _JSONResponse:
    return _JSONResponse({"message": {"title": ["Sample title"]}})


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    urlopen: object,
) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(check_dois.time, "sleep", lambda delay: delays.append(float(delay)))
    monkeypatch.setattr(check_dois, "_pace_doi", lambda _url: None)
    monkeypatch.setattr(check_dois.urllib.request, "urlopen", urlopen)
    return delays


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_then_success(
    status: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doi_calls = 0

    def urlopen(request: object, timeout: int = 30) -> _JSONResponse:
        del timeout
        url = _url_of(request)
        nonlocal doi_calls
        host = check_dois.urllib.parse.urlsplit(url).hostname or ""
        if host.endswith("doi.org"):
            doi_calls += 1
            if doi_calls == 1:
                raise _http_error(url, status, retry_after="1")
            return _handle_ok()
        return _crossref_ok()

    delays = _install_transport(monkeypatch, urlopen)
    code = check_dois.main([str(_sample_bib(tmp_path))])
    captured = capsys.readouterr()
    assert code == 0
    assert delays == [1.0]
    assert "warning:" not in captured.err
    assert "doi links ok" in captured.out
    assert "rate-limit" not in captured.out


@pytest.mark.parametrize("status", [429, 503])
def test_persistent_rate_limit_warns(
    status: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = {"doi": 0, "other": 0}

    def urlopen(request: object, timeout: int = 30) -> _JSONResponse:
        del timeout
        url = _url_of(request)
        host = check_dois.urllib.parse.urlsplit(url).hostname or ""
        if host.endswith("doi.org"):
            calls["doi"] += 1
        else:
            calls["other"] += 1
        raise _http_error(url, status)

    delays = _install_transport(monkeypatch, urlopen)
    code = check_dois.main([str(_sample_bib(tmp_path))])
    captured = capsys.readouterr()
    expected = [
        check_dois.RATE_LIMIT_BASE_S * (2**attempt)
        for attempt in range(check_dois.FETCH_ATTEMPTS - 1)
    ]
    assert code == 0
    assert delays == expected + expected
    assert sum(expected) <= check_dois.RATE_LIMIT_BUDGET_S
    assert max(delays) <= check_dois.RATE_LIMIT_BUDGET_S
    assert calls == {
        "doi": check_dois.FETCH_ATTEMPTS,
        "other": check_dois.FETCH_ATTEMPTS,
    }
    assert "warning:" in captured.err
    assert f"HTTP {status} (rate limited)" in captured.err
    assert "with rate-limit warnings" in captured.out


@pytest.mark.parametrize("status", [429, 503])
def test_long_retry_after_warns_without_stalling(
    status: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def urlopen(request: object, timeout: int = 30) -> _JSONResponse:
        del timeout
        nonlocal calls
        calls += 1
        raise _http_error(_url_of(request), status, retry_after="120")

    delays = _install_transport(monkeypatch, urlopen)
    code = check_dois.main([str(_sample_bib(tmp_path))])
    captured = capsys.readouterr()
    assert code == 0
    assert delays == []
    assert calls == 2
    assert "warning:" in captured.err


def test_retry_after_http_date_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    moment = datetime.now(timezone.utc) + timedelta(seconds=3)
    header = email.utils.formatdate(moment.timestamp(), usegmt=True)
    doi_calls = 0

    def urlopen(request: object, timeout: int = 30) -> _JSONResponse:
        del timeout
        url = _url_of(request)
        nonlocal doi_calls
        host = check_dois.urllib.parse.urlsplit(url).hostname or ""
        if host.endswith("doi.org"):
            doi_calls += 1
            if doi_calls == 1:
                raise _http_error(url, 429, retry_after=header)
            return _handle_ok()
        return _crossref_ok()

    delays = _install_transport(monkeypatch, urlopen)
    code = check_dois.main([str(_sample_bib(tmp_path))])
    captured = capsys.readouterr()
    assert code == 0
    assert delays[0] == pytest.approx(3.0, abs=0.5)
    assert delays[0] <= check_dois.RATE_LIMIT_BUDGET_S
    assert "warning:" not in captured.err


def test_doi_lookups_are_spaced(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr(check_dois.time, "sleep", lambda delay: delays.append(float(delay)))
    doi = _sample_doi()
    handle = "https://doi.org/api/handles/" + doi
    with check_dois._doi_pace_lock:
        check_dois._doi_not_before = check_dois.time.monotonic() + check_dois.DOI_LOOKUP_GAP_S
    check_dois._pace_doi(handle)
    assert len(delays) == 1
    assert delays[0] == pytest.approx(check_dois.DOI_LOOKUP_GAP_S, abs=0.05)
    delays.clear()
    check_dois._pace_doi("https://api.crossref.org/works/" + doi)
    assert delays == []


def test_missing_handle_still_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doi_calls = 0

    def urlopen(request: object, timeout: int = 30) -> _JSONResponse:
        del timeout
        nonlocal doi_calls
        url = _url_of(request)
        host = check_dois.urllib.parse.urlsplit(url).hostname or ""
        if host.endswith("doi.org"):
            doi_calls += 1
        raise _http_error(url, 404)

    delays = _install_transport(monkeypatch, urlopen)
    code = check_dois.main([str(_sample_bib(tmp_path))])
    captured = capsys.readouterr()
    assert code != 0
    assert doi_calls == 1
    assert delays == []
    assert "404" in captured.err
    assert "warning:" not in captured.err


def test_other_http_error_still_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def urlopen(request: object, timeout: int = 30) -> _JSONResponse:
        del timeout
        raise _http_error(_url_of(request), 500)

    delays = _install_transport(monkeypatch, urlopen)
    code = check_dois.main([str(_sample_bib(tmp_path))])
    captured = capsys.readouterr()
    assert code != 0
    assert delays[0] == pytest.approx(1.5)
    assert "HTTP 500" in captured.err
    assert "warning:" not in captured.err
