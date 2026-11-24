"""Self tests for the gate plan, its pixi environments, and `run.py`."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "ci" / "gate"
PLATFORMS = ("linux-64", "linux-aarch64")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, GATE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = load_module("run")


def locked_packages(lock: str) -> dict[tuple[str, str], set[str]]:
    """Map (environment, platform) to the conda package names it locks."""
    packages: dict[tuple[str, str], set[str]] = {}
    environment = platform = None
    in_environments = False
    for line in lock.splitlines():
        if line == "environments:":
            in_environments = True
            continue
        if in_environments and line and not line.startswith(" "):
            break
        if not in_environments:
            continue
        if match := re.fullmatch(r"  ([A-Za-z0-9_-]+):", line):
            environment, platform = match.group(1), None
        elif match := re.fullmatch(r"      ([a-z0-9-]+):", line):
            platform = match.group(1)
            packages[(environment, platform)] = set()
        elif platform and (match := re.match(r"      - conda: \S+/([^/]+)\.conda$", line)):
            name = match.group(1).rsplit("-", 2)[0]
            packages[(environment, platform)].add(name)
    return packages


def gate_environments() -> set[str]:
    plan = json.loads((GATE_DIR / "plan.json").read_text(encoding="utf-8"))
    environments = set()
    for command in plan["legs"]["pixi"]["commands"]:
        argv = command["argv"]
        environments.add(argv[argv.index("-e") + 1] if "-e" in argv else "default")
    return environments


def test_capnp_is_a_default_dependency():
    manifest = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    assert "capnproto" in manifest["dependencies"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_every_gate_environment_locks_capnp(platform):
    packages = locked_packages((ROOT / "pixi.lock").read_text(encoding="utf-8"))
    environments = gate_environments()
    assert environments == {"default", "python"}
    for environment in sorted(environments):
        assert "capnproto" in packages[(environment, platform)], environment


def write_executable(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def fake_eessi(tmp_path, monkeypatch):
    """An EESSI tree and an `eb-local` that only the caller's PATH holds."""
    eessi = tmp_path / "eessi"
    init = eessi / "versions" / "2026.06" / "init" / "lmod" / "bash"
    init.parent.mkdir(parents=True)
    init.write_text(
        textwrap.dedent(
            """
            : "${EESSI_NO_MODULE_PURGE_ON_INIT}"
            if [ -n "$PS1" ]; then :; fi
            module() {
              case "$*" in
                "-t avail Rust")
                  echo "Rust/1.88.0-GCCcore-14.3.0"
                  echo "Rust/1.94.1-GCCcore-15.2.0"
                  echo "Rust/1.95.0-GCCcore-15.2.0" ;;
                "-t list") echo "Rust/1.94.1-GCCcore-15.2.0" ;;
              esac
            }
            """
        ),
        encoding="utf-8",
    )
    write_executable(
        tmp_path / "tools" / "eb-local",
        """
        #!/bin/sh
        cat <<'SETUP'
        rustc() { gate-fake-toolchain rustc "$@"; }
        cargo() { gate-fake-toolchain cargo "$@"; }
        gcc() { gate-fake-toolchain gcc "$@"; }
        SETUP
        """,
    )
    write_executable(
        tmp_path / "tools" / "gate-fake-toolchain",
        """
        #!/bin/sh
        case "$1 $2" in
          "cargo test")
            echo "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 2 filtered out; finished in 0.01s" ;;
          "rustc --version") echo "rustc 1.91.0" ;;
          "cargo --version") echo "cargo 1.91.0" ;;
          "gcc --version") echo "gcc (GCC) 15.2.0" ;;
        esac
        """,
    )
    home = tmp_path / "home"
    home.mkdir()
    for profile in (".bash_profile", ".profile", ".bashrc"):
        (home / profile).write_text("export PATH=/usr/bin:/bin\n", encoding="utf-8")
    monkeypatch.setattr(run, "EESSI_ROOT", eessi)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{tmp_path / 'tools'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("BASH_ENV", raising=False)
    plan = json.loads((GATE_DIR / "plan.json").read_text(encoding="utf-8"))
    return plan


def test_shell_argv_is_not_a_login_shell():
    argv = run.shell_argv("true")
    assert argv[0] == "bash"
    assert not {"-l", "-lc", "--login"} & set(argv)


def test_eessi_leg_keeps_the_callers_path(fake_eessi, monkeypatch):
    scripts = []
    capture = run.run_capture

    def recording_capture(argv, **kwargs):
        scripts.append(argv[-1])
        return capture(argv, **kwargs)

    monkeypatch.setattr(run, "run_capture", recording_capture)
    leg = run.eessi_leg(fake_eessi)
    assert leg.status == "passed", leg.reason
    outputs = "\n".join(result.output for result in leg.commands)
    assert "command not found" not in outputs
    assert "Rust module: Rust/1.94.1-GCCcore-15.2.0" in leg.artefacts
    assert "module list: Rust/1.94.1-GCCcore-15.2.0" in leg.artefacts
    eb_local = shlex.quote(shutil.which("eb-local"))
    assert all(
        f"$({eb_local} dev Rust/1.94.1-GCCcore-15.2.0)" in script
        for script in scripts[2:]
    )


def test_eessi_leg_reports_init_failure(fake_eessi):
    init = (
        run.EESSI_ROOT
        / "versions"
        / fake_eessi["eessi"]["stack_version"]
        / "init"
        / "lmod"
        / "bash"
    )
    init.write_text("echo broken init >&2\nfalse\n", encoding="utf-8")
    leg = run.eessi_leg(fake_eessi)
    assert leg.status == "failed"
    assert leg.reason == "EESSI initialization failed with exit 1"
    assert "EESSI init: failed with exit 1" in leg.artefacts


def test_eessi_leg_refuses_an_unpinned_architecture(fake_eessi, monkeypatch):
    monkeypatch.setattr(run.platform, "machine", lambda: "riscv64")
    leg = run.eessi_leg(fake_eessi)
    assert leg.status == "failed"
    assert leg.reason == "no Rust module is pinned for architecture riscv64"


def test_eessi_leg_refuses_a_missing_pinned_module(fake_eessi):
    fake_eessi["eessi"]["rust_modules"][run.platform.machine()] = (
        "Rust/1.93.0-GCCcore-15.2.0"
    )
    leg = run.eessi_leg(fake_eessi)
    assert leg.status == "failed"
    assert leg.reason == (
        "pinned Rust module Rust/1.93.0-GCCcore-15.2.0 "
        f"is unavailable on {run.platform.machine()}"
    )


def test_eessi_leg_fails_closed_when_eb_local_fails(fake_eessi, tmp_path):
    write_executable(
        tmp_path / "tools" / "eb-local",
        """
        #!/bin/sh
        echo 'eb-local: no such module' >&2
        exit 1
        """,
    )
    leg = run.eessi_leg(fake_eessi)
    assert leg.status == "failed"
    assert leg.commands and all(result.returncode != 0 for result in leg.commands)
    outputs = "\n".join(result.output for result in leg.commands)
    assert "eb-local" in outputs
    assert "host toolchain" in outputs


check_gate = load_module("check_gate")
needs_nickel = pytest.mark.skipif(
    shutil.which("nickel") is None, reason="nickel is not on PATH"
)


@pytest.fixture
def gate_tree(tmp_path):
    for relative in (check_gate.SOURCE, *(output for _, _, output in check_gate.OUTPUTS)):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


@needs_nickel
def test_committed_gate_files_are_current():
    assert check_gate.drift(ROOT) == []


@needs_nickel
def test_check_gate_rejects_a_workflow_only_edit(gate_tree):
    workflow = gate_tree / ".github" / "workflows" / "gate.yml"
    edited = workflow.read_text(encoding="utf-8").replace(
        "fail-fast: false", "fail-fast: true"
    )
    workflow.write_text(edited, encoding="utf-8")
    diffs = check_gate.drift(gate_tree)
    assert len(diffs) == 1 and "-      fail-fast: true" in diffs[0]
    assert workflow.read_text(encoding="utf-8") == edited


@needs_nickel
def test_check_gate_rejects_a_source_only_edit(gate_tree):
    source = gate_tree / check_gate.SOURCE
    source.write_text(
        source.read_text(encoding="utf-8").replace("max_cpus = 8", "max_cpus = 4"),
        encoding="utf-8",
    )
    diffs = check_gate.drift(gate_tree)
    assert len(diffs) == 1 and '+  "max_cpus": 4,' in diffs[0]


@needs_nickel
def test_check_gate_rejects_a_missing_output(gate_tree):
    (gate_tree / "ci" / "gate" / "plan.json").unlink()
    assert len(check_gate.drift(gate_tree)) == 1


@needs_nickel
def test_check_gate_exits_nonzero_on_drift(gate_tree):
    workflow = gate_tree / ".github" / "workflows" / "gate.yml"
    workflow.write_text(workflow.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(GATE_DIR / "check_gate.py"), "--root", str(gate_tree)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "+# edit" not in completed.stderr and "-# edit" in completed.stderr


CARGO_OUTPUT = """
     Running unittests src/lib.rs (target/debug/deps/rgsaddle-0123456789abcdef)
test result: ok. 1000 passed; 0 failed; 30 ignored; 0 measured; 0 filtered out; finished in 41.20s

     Running tests/band.rs (target/debug/deps/band-0123456789abcdef)
test result: ok. 40 passed; 0 failed; 8 ignored; 0 measured; 0 filtered out; finished in 2.03s

   Doc-tests rgsaddle
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.51s
"""


def test_cargo_counts_keep_ignored_tests():
    counts = run.parse_test_counts(CARGO_OUTPUT)
    assert counts == [run.TestCounts("cargo", 1046, 0, 38, 0)]
    assert str(counts[0]) == "cargo: 1046 passed, 0 failed, 38 ignored, 0 skipped"


def test_cargo_counts_failures_and_filtered_tests():
    output = (
        "test result: FAILED. 4 passed; 1 failed; 2 ignored; 0 measured; "
        "7 filtered out; finished in 0.10s\n"
    )
    assert run.cargo_counts(output) == run.TestCounts("cargo", 4, 1, 2, 7)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "======= 120 passed, 3 skipped, 2 xfailed, 1 warning in 12.30s =======\n",
            run.TestCounts("pytest", 120, 0, 2, 3),
        ),
        ("5 passed, 2 deselected in 0.40s\n", run.TestCounts("pytest", 5, 0, 0, 2)),
        (
            "=== 2 failed, 7 passed, 1 skipped, 1 xpassed, 3 errors in 63.00s (0:01:03) ===\n",
            run.TestCounts("pytest", 8, 5, 0, 1),
        ),
        ("============ no tests ran in 0.01s ============\n", run.TestCounts("pytest", 0, 0, 0, 0)),
    ],
)
def test_pytest_counts(output, expected):
    assert run.parse_test_counts(output) == [expected]


def test_pytest_counts_read_the_last_summary_only():
    output = (
        "SKIPPED [1] tests/python/test_ase_against.py:12: ase is missing\n"
        "==== 1 passed in 0.01s ====\n"
        "==== 9 passed, 1 skipped in 1.00s ====\n"
    )
    assert run.pytest_counts(output) == run.TestCounts("pytest", 9, 0, 0, 1)


def test_meson_counts():
    output = textwrap.dedent(
        """
        Ok:                 3
        Expected Fail:      1
        Fail:               0
        Unexpected Pass:    0
        Skipped:            2
        Timeout:            0
        """
    )
    assert run.parse_test_counts(output) == [run.TestCounts("Meson", 3, 0, 1, 2)]


def test_meson_counts_unexpected_passes_and_timeouts_fail():
    output = "Ok: 1\nFail: 1\nUnexpected Pass: 1\nTimeout: 2\n"
    assert run.meson_counts(output) == run.TestCounts("Meson", 1, 4, 0, 0)


def test_no_counts_without_a_summary():
    assert run.parse_test_counts("Compiling rgsaddle v0.5.0\nFinished\n") == []


def test_a_test_command_with_failures_does_not_pass():
    result = run.CommandResult(
        name="tests",
        kind="test",
        argv=["true"],
        returncode=0,
        output="",
        tests=[run.TestCounts("cargo", 3, 1, 0, 0)],
    )
    assert not result.passed


def test_eessi_leg_reports_all_four_counts(fake_eessi):
    leg = run.eessi_leg(fake_eessi)
    assert "test summary: cargo: 3 passed, 0 failed, 1 ignored, 2 skipped" in leg.artefacts
