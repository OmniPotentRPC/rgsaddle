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
            module() {
              case "$*" in
                "-t avail Rust")
                  echo "Rust/1.88.0-GCCcore-14.3.0"
                  echo "Rust/1.91.0-GCCcore-15.2.0" ;;
                "-t list") echo "Rust/1.91.0-GCCcore-15.2.0" ;;
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
    assert "Rust module: Rust/1.91.0-GCCcore-15.2.0" in leg.artefacts
    assert "module list: Rust/1.91.0-GCCcore-15.2.0" in leg.artefacts
    eb_local = shlex.quote(shutil.which("eb-local"))
    assert all(f"$({eb_local} dev " in script for script in scripts[1:])


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
