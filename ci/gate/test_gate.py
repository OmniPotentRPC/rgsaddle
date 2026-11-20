"""Self tests for the gate plan, its pixi environments, and `run.py`."""

from __future__ import annotations

import json
import pathlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "ci" / "gate"
PLATFORMS = ("linux-64", "linux-aarch64")


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
