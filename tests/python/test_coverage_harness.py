"""Saddle-coverage harness checks that do not need a potential."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "saddle_coverage", ROOT / "bench" / "harness.py"
)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_write_table_keeps_the_build_section(tmp_path):
    payload = {
        "summary": [],
        "comparisons": [],
        "systems": {
            "pt_exchange": {"reference": {"usable": True, "index": 1, "barrier": 4.569}},
        },
        "runs": [
            {
                "system": "pt_exchange",
                "method": "ase-neb",
                "images": 7,
                "round": 0,
                "status": "max_iterations",
                "success": False,
                "wall_s": 1.0,
                "outside_s": 0.5,
                "force_calls": 10,
                "barrier": None,
                "rmsd": 5.0,
                "energy_error": 1.0,
            },
            {
                "system": "pt_exchange",
                "method": "eon-dimer",
                "images": None,
                "round": 0,
                "status": "GOOD",
                "success": False,
                "wall_s": 1.0,
                "outside_s": 0.5,
                "force_calls": 10,
                "barrier": None,
                "rmsd": 1.0,
                "energy_error": 1.0,
            },
        ],
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
    assert "| source |" in text
    assert "shas.rgsaddle" in text
    assert "versions.rgsaddle_abi" in text
    assert "runs[*].rgsaddle_revision" not in text
    assert "runs[*].rgsaddle_abi" not in text
    assert "next run" in text
    assert "unverified (" in text
    for run in payload["runs"]:
        assert "rgsaddle_revision" not in run
        assert "rgsaddle_abi" not in run
    harness.write_table(str(path), payload)
    rewritten = path.read_text(encoding="utf-8")
    assert rewritten.count("## Build") == 1
    assert "Example CPU" in rewritten


def test_coverage_record_stores_the_cpu_and_versions():
    class Args:
        shas = {"rgpot": "a" * 8, "eon": "b" * 8, "rgsaddle": "c" * 8}
        rounds = 12
        threads = 1

    recorded = {
        "ase": "3.29.0",
        "numpy": "2.5.5",
        "pyeonclient": "0.4.2",
        "rgsaddle_abi": "1.43",
    }
    payload = harness.coverage_record(
        Args,
        recorded,
        {
            "runs": [
                {"system": "pt_hop", "method": "ase-neb", "images": 7},
                {"system": "pt_hop", "method": "eon-dimer", "images": None},
            ]
        },
    )
    assert payload["versions"]["ase"] == "3.29.0"
    assert payload["versions"]["pyeonclient"] == "0.4.2"
    assert payload["versions"]["eon_project"] == harness.EON_PROJECT_VERSION
    assert payload["cpu_model"]
    assert not str(payload["cpu_model"]).startswith("/")
    assert payload["build"]
    for run in payload["runs"]:
        assert run["rgsaddle_revision"]
        assert run["rgsaddle_abi"]


def test_fresh_neb_rows_use_the_live_revision():
    class Args:
        shas = {"rgpot": "a" * 8, "eon": "b" * 8, "rgsaddle": "c" * 8}
        rounds = 12
        threads = 1

    payload = harness.coverage_record(
        Args,
        {"rgsaddle_abi": "1.43"},
        {
            "runs": [
                {"system": "pt_exchange", "method": "ase-neb", "images": 7},
            ]
        },
    )
    assert payload["runs"][0]["rgsaddle_revision"] == "c" * 8
    assert payload["runs"][0]["rgsaddle_abi"] == "1.43"


def test_finalize_results_corrects_summary_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_load_external_rmsd_corrections", lambda: [])
    payload = {
        "runs": [
            {
                "system": "pt_exchange",
                "method": "ase-neb",
                "images": 7,
                "round": 0,
                "wall_s": 1.0,
                "outside_s": 0.5,
                "force_calls": 10,
                "success": False,
                "rmsd": 5.0,
                "energy_error": 1.0,
            }
        ],
        "rmsd_corrections": [
            {
                "system": "pt_exchange",
                "method": "ase-neb",
                "images": 7,
                "round": 0,
                "rmsd_old": 5.0,
                "rmsd_new": 2.0,
                "cause": "test correction",
            }
        ],
    }
    harness.finalize_results(payload)
    path = tmp_path / "results.json"
    harness.dump_json(str(path), payload)
    written = path.read_text(encoding="utf-8")
    assert payload["summary"][0]["rmsd_median_angstrom"] == 2.0
    assert '"rmsd_corrected": 2.0' in written
    assert '"rmsd_median_angstrom": 2.0' in written


def _measured_run(**overrides):
    row = {
        "system": "pt_hop",
        "method": "ase-neb",
        "images": 7,
        "round": 0,
        "status": "max_iterations",
        "success": False,
        "wall_s": 1.0,
        "outside_s": 0.4,
        "force_calls": 10,
        "barrier": None,
        "rmsd": 1.0,
        "energy_error": 0.2,
    }
    row.update(overrides)
    return row


def test_write_table_recomputes_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_load_external_rmsd_corrections", lambda: [])
    payload = {
        "summary": [
            {
                "system": "pt_hop",
                "method": "ase-neb",
                "images": 7,
                "rmsd_median_angstrom": 8.5,
            }
        ],
        "comparisons": [
            {
                "system": "pt_hop",
                "images": 7,
                "a": "stale",
                "b": "row",
                "wall_p": 0.5,
                "outside_p": 0.5,
                "calls_p": 0.5,
                "barrier_p": 0.5,
                "success_p": 0.5,
            }
        ],
        "systems": {"pt_hop": {"reference": {"usable": True, "index": 1, "barrier": 0.09}}},
        "runs": [
            _measured_run(round=0, rmsd=1.0),
            _measured_run(round=1, rmsd=3.0),
        ],
        "rmsd_corrections": [
            {
                "system": "pt_hop",
                "method": "ase-neb",
                "images": 7,
                "round": 0,
                "rmsd_old": 1.0,
                "rmsd_new": 4.0,
                "cause": "unit correction",
            },
            {
                "system": "pt_hop",
                "method": "ase-neb",
                "images": 7,
                "round": 1,
                "rmsd_old": 3.0,
                "rmsd_new": 6.0,
                "cause": "unit correction",
            },
        ],
        "shas": {"rgsaddle": "abc12345", "rgpot": "def12345", "eon": "ghi12345"},
        "versions": {"rgsaddle_abi": "1.43", "ase": "3.29.0"},
        "cpu_model": "Example CPU",
        "threads": 1,
        "rounds": 2,
    }
    path = tmp_path / "results.md"
    harness.write_table(str(path), payload)
    text = path.read_text(encoding="utf-8")
    assert payload["summary"][0]["rmsd_median_angstrom"] == 5.0
    assert payload["comparisons"] == []
    assert "8.5" not in text
    assert "stale" not in text
    assert "| 5 |" in text


def test_matching_rmsd_overlay_is_a_consistency_check(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_load_external_rmsd_corrections", lambda: [])
    payload = {
        "summary": [],
        "comparisons": [],
        "systems": {"pt_hop": {"reference": {"usable": True, "index": 1, "barrier": 0.09}}},
        "runs": [_measured_run(rmsd=1.25)],
        "rmsd_corrections": [
            {
                "system": "pt_hop",
                "method": "ase-neb",
                "images": 7,
                "round": 0,
                "rmsd_old": 1.25,
                "rmsd_new": 1.25,
                "cause": "same value",
            }
        ],
        "shas": {"rgsaddle": "abc12345"},
        "versions": {"rgsaddle_abi": "1.43"},
        "threads": 1,
        "rounds": 1,
    }
    path = tmp_path / "results.md"
    harness.write_table(str(path), payload)
    text = path.read_text(encoding="utf-8")
    assert "## RMSD consistency check" in text
    assert "## RMSD corrections" not in text
    assert "summary is unchanged" in text
    assert payload["summary"][0]["rmsd_median_angstrom"] == 1.25


def test_malformed_rmsd_corrections_raise(tmp_path, monkeypatch):
    path = tmp_path / "corrections.json"
    monkeypatch.setattr(harness, "RMSD_CORRECTIONS_PATH", str(path))
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed RMSD corrections JSON"):
        harness._load_external_rmsd_corrections()
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected an object with a list of rows"):
        harness._load_external_rmsd_corrections()
    monkeypatch.setattr(harness, "RMSD_CORRECTIONS_PATH", str(tmp_path / "absent.json"))
    assert harness._load_external_rmsd_corrections() == []


def test_lockfile_reader_falls_back_to_tomli(monkeypatch, tmp_path):
    import sys
    import types

    seen = {}

    def loads(text):
        seen["text"] = text
        return {
            "package": [
                {"name": "minimage", "version": "9.9.9", "source": "registry+example"},
            ]
        }

    fake = types.ModuleType("tomli")
    fake.loads = loads
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", fake)
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        'package = [{ name = "minimage", version = "1.2.3", source = "registry+example" }]\n',
        encoding="utf-8",
    )
    found = harness._cargo_lock_packages(str(lock))
    assert seen["text"].startswith("package")
    assert found["minimage"][0]["version"] == "9.9.9"


def test_validate_payload_rejects_missing_cell_revision():
    payload = {
        "runs": [{"system": "pt_hop", "method": "ase-neb", "images": 7}],
        "build": [],
    }
    with pytest.raises(RuntimeError, match="missing per-cell revision metadata"):
        harness.validate_payload(payload)


def test_validate_payload_rejects_build_with_too_few_revisions():
    payload = {
        "runs": [
            {
                "system": "pt_hop",
                "method": "ase-neb",
                "images": 7,
                "rgsaddle_revision": "aaa",
                "rgsaddle_abi": "1.42",
            },
            {
                "system": "pt_hop",
                "method": "eon-dimer",
                "images": None,
                "rgsaddle_revision": "bbb",
                "rgsaddle_abi": "1.43",
            },
        ],
        "build": [
            {
                "component": "rgsaddle-revision",
                "value": "aaa, ABI 1.42",
                "source": "fixture cell metadata",
                "revision_sha": "aaa",
                "revision_abi": "1.42",
            }
        ],
    }
    with pytest.raises(RuntimeError, match="fewer than cell revisions"):
        harness.validate_payload(payload)


def test_validate_payload_rejects_a_bare_build_literal():
    payload = {
        "runs": [
            {
                "system": "pt_hop",
                "method": "ase-neb",
                "images": 7,
                "rgsaddle_revision": "aaa",
                "rgsaddle_abi": "1.42",
            }
        ],
        "build": ["0.3.11"],
    }
    with pytest.raises(RuntimeError, match="bare Build value"):
        harness.validate_payload(payload)


def test_build_entries_record_their_source():
    payload = {
        "runs": [
            {
                "system": "pt_exchange",
                "method": "ase-neb",
                "images": 7,
                "rgsaddle_revision": "aaa",
                "rgsaddle_abi": "1.42",
            }
        ],
        "shas": {"rgsaddle": "bbb", "rgpot": "ccc", "eon": "ddd"},
        "versions": {"rgpot": "3.7.0", "eon_project": "3.6.0"},
    }
    rows = harness.derive_build_entries(payload)
    assert rows
    for row in rows:
        assert row["source"]
    assert not any(
        row["component"] == "linkcell (eOn-side)" and row["value"] == "0.3.11"
        for row in rows
    )


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
