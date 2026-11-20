"""Skip the coverage harness only when its file is absent.

The wheel smoke copies ``tests/python`` and not ``bench/``. A missing
harness is that layout. A harness that is present and cannot be loaded
is a real collection failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "test_coverage_harness.py"
REASON = "bench/harness.py is not in this source tree"


def _project(tmp_path: Path, harness: str | None) -> Path:
    root = tmp_path / "project"
    dest = root / "tests" / "python"
    dest.mkdir(parents=True)
    shutil.copy(SOURCE, dest / SOURCE.name)
    if harness is not None:
        bench = root / "bench"
        bench.mkdir()
        (bench / "harness.py").write_text(harness, encoding="utf-8")
    return root


def _pytest(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/python/test_coverage_harness.py",
            "-rs",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_missing_harness_skips_with_a_reason(tmp_path: Path) -> None:
    completed = _pytest(_project(tmp_path, None))
    output = completed.stdout + completed.stderr
    # Exit 5 is pytest's "no tests ran" status for a module-level skip
    # with nothing else in the session. It is not a collection error.
    assert completed.returncode in (0, 5), output
    assert "ERROR" not in output
    assert REASON in output


def test_unreadable_harness_is_not_skipped(tmp_path: Path) -> None:
    completed = _pytest(_project(tmp_path, "def this is not python\n"))
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert REASON not in output
