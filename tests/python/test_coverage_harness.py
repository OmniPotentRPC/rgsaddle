"""Saddle-coverage harness checks that do not need a potential."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "bench" / "harness.py"
if not HARNESS_PATH.is_file():
    pytest.skip(
        f"bench/harness.py is not in this source tree ({HARNESS_PATH})",
        allow_module_level=True,
    )

SPEC = importlib.util.spec_from_file_location("saddle_coverage", HARNESS_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load the coverage harness at {HARNESS_PATH}")
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_write_table_keeps_the_build_section(tmp_path):
    payload = {
        "summary": [],
        "systems": {},
        "comparisons": [],
        "runs": [],
        "shas": {
            "rgsaddle": "abc12345",
            "rgpot": "def12345",
            "eon": "ghi12345",
        },
        "versions": {
            "ase": "3.29.0",
            "pyeonclient": "0.4.2",
            "rgsaddle_abi": "1.42",
            "numpy": "2.5.5",
        },
        "cpu_model": "Example CPU",
        "threads": 1,
        "rounds": 12,
    }
    path = tmp_path / "results.md"
    harness.write_table(str(path), payload)
    text = path.read_text(encoding="utf-8")
    assert "## Build" in text
    assert "Example CPU" in text
    assert "3.29.0" in text
    assert "0.4.2" in text
    assert "abc12345" in text
    harness.write_table(str(path), payload)
    rewritten = path.read_text(encoding="utf-8")
    assert rewritten.count("## Build") == 1
    assert "Example CPU" in rewritten


def test_coverage_record_stores_the_cpu_and_versions():
    class Args:
        shas = {"rgpot": "a" * 8, "eon": "b" * 8, "rgsaddle": "c" * 8}
        rounds = 12
        threads = 1

    recorded = {"ase": "3.29.0", "numpy": "2.5.5", "pyeonclient": "0.4.2"}
    payload = harness.coverage_record(Args, recorded, {"runs": []})
    assert payload["versions"]["ase"] == "3.29.0"
    assert payload["versions"]["pyeonclient"] == "0.4.2"
    assert payload["versions"]["eon_project"] == harness.EON_PROJECT_VERSION
    assert payload["cpu_model"]
    assert not str(payload["cpu_model"]).startswith("/")


def _cluster(numbers):
    return {
        "fixed": np.zeros(len(numbers), dtype=np.uint8),
        "periodic": False,
        "numbers": list(numbers),
        "cell": np.eye(3) * 40.0,
    }


def test_rotated_copy_scores_about_zero():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.2],
            [0.4, 0.9, 0.1],
            [0.5, 0.3, 1.0],
            [0.6, 0.4, 1.6],
        ],
        dtype=np.float64,
    )
    angle = 1.15
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    trial = reference @ rotation.T + np.array([2.5, -1.2, 0.7])
    system = _cluster([29, 29, 29, 29, 1])
    assert harness.rmsd(system, reference, trial) < 1e-8
    swapped = trial.copy()
    swapped[[0, 2]] = swapped[[2, 0]]
    assert harness.rmsd(system, reference, swapped) < 1e-8


def test_shared_reactant_does_not_share_a_minmode_start():
    reactant = np.zeros((6, 3), dtype=np.float64)
    reactant[1, 0] = 1.2
    reactant[2, 1] = 0.8
    fixed = np.zeros(6, dtype=np.uint8)
    fixed[0] = 1
    hop = {"name": "pt_hop", "reactant": reactant, "fixed": fixed}
    exchange = {
        "name": "pt_exchange",
        "reactant": reactant.copy(),
        "fixed": fixed.copy(),
    }
    hop_position, hop_mode = harness.displace(hop, 3)
    exchange_position, exchange_mode = harness.displace(exchange, 3)
    assert not np.allclose(hop_position, exchange_position)
    assert not np.allclose(hop_mode, exchange_mode)
    again, again_mode = harness.displace(hop, 3)
    assert np.allclose(hop_position, again)
    assert np.allclose(hop_mode, again_mode)


def test_dimer_and_lanczos_are_different_kinds():
    dimer = harness.minmode_config(0, 0.01)
    lanczos = harness.minmode_config(1, 0.01)
    assert dimer.kind == 0
    assert lanczos.kind == 1
    assert dimer.kind != lanczos.kind


def test_locate_eonclib_beside_the_package(tmp_path):
    package = tmp_path / "pyeonclient"
    package.mkdir()
    library = package / "libeonclib.so"
    library.write_bytes(b"\x7fELF")
    found = harness.locate_eonclib([str(tmp_path)], str(tmp_path / "absent"))
    assert found == str(library)


def test_missing_eonclib_explains_the_preload(tmp_path, monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    with pytest.raises(SystemExit) as caught:
        harness.preload_eonclib([str(tmp_path)], str(tmp_path))
    message = str(caught.value)
    assert "libeonclib" in message
    assert "LD_LIBRARY_PATH" in message
    assert "pyeonclient" in message
