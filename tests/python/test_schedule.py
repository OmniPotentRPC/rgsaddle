"""Owner schedule. The worker count is not an input."""

from __future__ import annotations

import hashlib

import numpy as np

from rgsaddle._core._testing import schedule_canonical

IDS = np.array([0, 2, 1, 4, 3, 5], dtype=np.int64)
OWNERS = np.array([10, 10, 20, 10, 30, 20], dtype=np.uint64)
FLAGS = np.array([0, 0, 1, 0, 0, 1], dtype=np.uint8)
WORKERS = (1, 2, 3, 8)


def test_schedule_hash_matches_across_workers(capsys: object) -> None:
    hashes = []
    texts = []
    for workers in WORKERS:
        text = schedule_canonical(IDS, OWNERS, FLAGS, workers)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes.append(digest)
        texts.append(text)
        print(f"schedule_sha256 workers={workers} {digest}")
    assert len(set(hashes)) == 1
    assert len(set(texts)) == 1
    assert texts[0] == "10 0 0 2 4\n20 1 1 5\n30 0 3\n"


def test_schedule_rejects_zero_workers() -> None:
    try:
        schedule_canonical(IDS, OWNERS, FLAGS, 0)
    except ValueError:
        return
    raise AssertionError("threads 0 must raise")


def test_schedule_rejects_duplicate_row_id() -> None:
    ids = np.array([0, 0], dtype=np.int64)
    owners = np.array([1, 2], dtype=np.uint64)
    flags = np.array([0, 0], dtype=np.uint8)
    try:
        schedule_canonical(ids, owners, flags, 1)
    except ValueError as exc:
        assert "duplicate row_id" in str(exc)
        assert "rgbind" not in str(exc)
        assert "::" not in str(exc)
        return
    raise AssertionError("duplicate row id must raise")
