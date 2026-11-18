"""The Sella topology fixture and the covalent radii table stay honest.

Sella is an oracle for these checks only; the crate never imports it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ase")
sella = pytest.importorskip("sella")

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
FIXTURE = ROOT / "tests" / "fixtures" / "sella_topology.txt"


def run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_recorded_fixture_is_what_this_sella_builds():
    done = run("gen_sella_topology_fixture.py")
    assert done.returncode == 0, done.stderr
    recorded = FIXTURE.read_text().splitlines()
    fresh = done.stdout.splitlines()
    assert recorded[1:] == fresh[1:]
    assert recorded[0].startswith("sella ")


def test_the_fixture_header_names_a_sella_release():
    from importlib.metadata import version

    assert FIXTURE.read_text().splitlines()[0] == f"sella {version('sella')}"


def test_the_covalent_radii_table_agrees_with_the_independent_sources():
    done = run("check_covalent_radii.py")
    assert done.returncode == 0, done.stdout + done.stderr
