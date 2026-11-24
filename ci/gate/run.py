#!/usr/bin/env python3
"""Run the committed rgsaddle gate plan and write its Markdown report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field


ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "ci" / "gate" / "plan.json"
EESSI_ROOT = pathlib.Path("/cvmfs/software.eessi.io")
FORBIDDEN_OUTPUT = re.compile(
    r"(?m)(?:^|\s)(?:error(?:\[|:|\s)|FAILED(?:\s|$)|panicked(?:\s|$))"
)


@dataclass
class CommandResult:
    name: str
    kind: str
    argv: list[str]
    returncode: int
    output: str
    tests: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return FORBIDDEN_OUTPUT.search(self.output) is None

    @property
    def passed(self) -> bool:
        counts_present = self.kind != "test" or bool(self.tests)
        return self.returncode == 0 and self.clean and counts_present


@dataclass
class LegResult:
    name: str
    status: str
    reason: str
    commands: list[CommandResult] = field(default_factory=list)
    artefacts: list[str] = field(default_factory=list)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def run_capture(
    argv: list[str], *, env: dict[str, str] | None = None
) -> tuple[int, str]:
    resolved = list(argv)
    if resolved and resolved[0] == "pixi" and shutil.which("pixi") is None:
        user_pixi = pathlib.Path.home() / ".pixi" / "bin" / "pixi"
        if user_pixi.is_file():
            resolved[0] = str(user_pixi)
    print(f"+ {shlex.join(argv)}", flush=True)
    completed = subprocess.run(
        resolved,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    return completed.returncode, completed.stdout


def test_counts(output: str) -> list[str]:
    counts: list[str] = []
    cargo_matches = re.findall(
        r"test result: [^.]+\.\s+(\d+) passed;\s+(\d+) failed", output
    )
    if cargo_matches:
        passed = sum(int(item[0]) for item in cargo_matches)
        failed = sum(int(item[1]) for item in cargo_matches)
        counts.append(f"cargo: {passed} passed, {failed} failed")
    pytest_matches = re.findall(r"(?m)(\d+) passed(?:,| in)", output)
    if pytest_matches:
        counts.append(f"pytest: {pytest_matches[-1]} passed")
    meson_ok = re.findall(r"(?m)^Ok:\s+(\d+)\s*$", output)
    meson_failed = re.findall(r"(?m)^Fail:\s+(\d+)\s*$", output)
    if meson_ok and meson_failed:
        counts.append(f"Meson: {meson_ok[-1]} passed, {meson_failed[-1]} failed")
    return counts


def shell_argv(script: str) -> list[str]:
    # A login shell rereads the profile files, which reset PATH and drop
    # tools such as eb-local that the caller put on it.
    return ["bash", "--noprofile", "--norc", "-c", script]


def execute_commands(
    commands: list[dict[str, object]], prefix: str = ""
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for command in commands:
        argv = [str(value) for value in command["argv"]]
        actual = shell_argv(f"{prefix}\n{shlex.join(argv)}") if prefix else argv
        returncode, output = run_capture(actual)
        kind = str(command["kind"])
        results.append(
            CommandResult(
                name=str(command["name"]),
                kind=kind,
                argv=argv,
                returncode=returncode,
                output=output,
                tests=test_counts(output) if kind == "test" else [],
            )
        )
    return results


def version_line(argv: list[str]) -> str:
    returncode, output = run_capture(argv)
    if returncode != 0:
        return f"{shlex.join(argv)} exited {returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else f"{shlex.join(argv)} produced no version"


def pixi_leg(plan: dict[str, object]) -> LegResult:
    commands = plan["legs"]["pixi"]["commands"]
    results = execute_commands(commands)
    failures = [result.name for result in results if not result.passed]
    reason = (
        "all pixi build and test checks passed"
        if not failures
        else "failed checks: " + ", ".join(failures)
    )
    summaries = [
        count
        for result in results
        for count in result.tests
    ]
    artefacts = [
        f"pixi version: {version_line(['pixi', '--version'])}",
        f"Rust toolchain: {version_line(['pixi', 'run', 'rustc', '--version'])}",
        f"Cargo toolchain: {version_line(['pixi', 'run', 'cargo', '--version'])}",
        f"Python version: {version_line(['pixi', 'run', '-e', 'python', 'python', '--version'])}",
        "module list: not used by the pixi leg",
        "test summary: " + ("; ".join(summaries) if summaries else "none found"),
    ]
    return LegResult(
        name="pixi",
        status="passed" if not failures else "failed",
        reason=reason,
        commands=results,
        artefacts=artefacts,
    )


def select_rust_module(output: str, prefix: str, suffix: str) -> str | None:
    candidates = {
        match.rstrip(":")
        for match in re.findall(r"Rust/[A-Za-z0-9.+_-]+", output)
        if match.startswith(prefix) and match.endswith(suffix)
    }

    def version_key(module: str) -> tuple[int, ...]:
        version = module.removeprefix(prefix).removesuffix(suffix)
        return tuple(int(part) for part in re.findall(r"\d+", version))

    return max(candidates, key=version_key) if candidates else None


def eessi_leg(plan: dict[str, object]) -> LegResult:
    config = plan["eessi"]
    stack = str(config["stack_version"])
    init = EESSI_ROOT / "versions" / stack / "init" / "lmod" / "bash"
    modules = EESSI_ROOT / "init" / "modules"
    eb_local = shutil.which("eb-local")
    missing = []
    if not init.is_file():
        missing.append(str(init))
    if eb_local is None:
        missing.append("eb-local")
    if missing:
        return LegResult(
            name="eessi",
            status="unavailable",
            reason="EESSI not available: missing " + ", ".join(missing),
            artefacts=[
                f"EESSI stack: {stack}",
                f"compiler: {config['compiler']}",
                "module list: unavailable",
                "test summary: not run",
            ],
        )

    module_probe = (
        f"source {shlex.quote(str(init))}\n"
        f"module use {shlex.quote(str(modules))}\n"
        "module -t avail Rust 2>&1"
    )
    probe_code, module_output = run_capture(shell_argv(module_probe))
    rust_module = select_rust_module(
        module_output,
        str(config["rust_module_prefix"]),
        str(config["rust_module_suffix"]),
    )
    if probe_code != 0 or rust_module is None:
        return LegResult(
            name="eessi",
            status="failed",
            reason=(
                "module avail Rust did not find a module built with "
                f"{config['compiler']}"
            ),
            artefacts=[
                f"EESSI stack: {stack}",
                f"compiler: {config['compiler']}",
                "module list: no matching Rust module",
                "test summary: not run",
            ],
        )

    setup = (
        f"source {shlex.quote(str(init))}\n"
        f"module use {shlex.quote(str(modules))}\n"
        "unset LD_LIBRARY_PATH\n"
        f"export EESSI_VERSION={shlex.quote(stack)}\n"
        f'eval "$({shlex.quote(eb_local)} dev {shlex.quote(rust_module)})"'
    )
    commands = plan["legs"]["eessi"]["commands"]
    results = execute_commands(commands, prefix=setup)
    list_code, loaded_modules = run_capture(
        shell_argv(f"{setup}\nmodule -t list 2>&1")
    )
    module_list = "; ".join(
        line.strip()
        for line in loaded_modules.splitlines()
        if re.fullmatch(r"[A-Za-z0-9+_.-]+/[A-Za-z0-9+_.-]+", line.strip())
    )
    version_code, version_output = run_capture(
        shell_argv(f"{setup}\nrustc --version\ncargo --version\ngcc --version")
    )
    tool_versions = [
        line.strip()
        for line in version_output.splitlines()
        if line.startswith(("rustc ", "cargo ", "gcc "))
    ]
    failures = [result.name for result in results if not result.passed]
    if list_code != 0:
        failures.append("module list")
    if version_code != 0 or len(tool_versions) != 3:
        failures.append("toolchain versions")
    summaries = [
        count
        for result in results
        for count in result.tests
    ]
    reason = (
        "all EESSI build and test checks passed"
        if not failures
        else "failed checks: " + ", ".join(failures)
    )
    return LegResult(
        name="eessi",
        status="passed" if not failures else "failed",
        reason=reason,
        commands=results,
        artefacts=[
            f"EESSI stack: {stack}",
            f"compiler: {config['compiler']}",
            f"Rust module: {rust_module}",
            "toolchain versions: " + "; ".join(tool_versions),
            f"module list: {module_list or 'empty'}",
            "test summary: " + ("; ".join(summaries) if summaries else "none found"),
        ],
    )


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def limit_cpu_affinity(max_cpus: int) -> int:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return os.cpu_count() or 1
    available = sorted(os.sched_getaffinity(0))
    selected = available[:max_cpus]
    os.sched_setaffinity(0, selected)
    return len(selected)


def repository_name() -> str:
    remote = git_output("remote", "get-url", "origin")
    remote = re.sub(r"(?<=://)[^/@]+@", "", remote)
    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    return remote or ROOT.name


def details_block(legs: list[LegResult], success: bool) -> str:
    mood = ":grin: SUCCESS" if success else ":cry: FAILURE"
    lines = [
        f"<details><summary>{mood} _(click triangle for details)_</summary>",
        "",
        "_Details_",
        "",
    ]
    for leg in legs:
        status_mark = ":white_check_mark:" if leg.status == "passed" else ":x:"
        lines.append(f"- {status_mark} {leg.name}: {leg.reason}")
        for result in leg.commands:
            exit_mark = ":white_check_mark:" if result.returncode == 0 else ":x:"
            clean_mark = ":white_check_mark:" if result.clean else ":x:"
            lines.append(
                f"  - {exit_mark} {result.name}: exit {result.returncode}"
            )
            lines.append(
                "  - "
                f"{clean_mark} no message matching `error|FAILED|panicked`"
            )
            if result.kind == "test":
                count_mark = ":white_check_mark:" if result.tests else ":x:"
                summary = "; ".join(result.tests) if result.tests else "none"
                lines.append(f"  - {count_mark} test counts found: {summary}")
    lines.extend(["", "_Artefacts_", ""])
    for leg in legs:
        for artefact in leg.artefacts:
            lines.append(f"- {leg.name}: {artefact}")
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def markdown_cell(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", "<br>")


def make_report(
    legs: list[LegResult],
    submitted: str,
    started: str,
    finished: str,
    cpu_count: int,
) -> tuple[str, int]:
    failures = [leg for leg in legs if leg.status == "failed"]
    unavailable = [leg for leg in legs if leg.status == "unavailable"]
    success = not failures and not unavailable
    if failures:
        reason = "; ".join(leg.reason for leg in failures)
        exit_code = 1
    elif unavailable:
        reason = "; ".join(leg.reason for leg in unavailable)
        exit_code = 2
    else:
        reason = "all selected gate legs passed"
        exit_code = 0
    os_release = platform.platform()
    selected = ", ".join(leg.name for leg in legs)
    details = markdown_cell(details_block(legs, success))
    lines = [
        "# rgsaddle build and test gate",
        "",
        f"Instance / host: {os.environ.get('RUNNER_NAME', 'local')} / {socket.gethostname()}",
        f"Repository: {repository_name()}",
        f"Building on: {os_release} ({cpu_count} CPUs)",
        f"Building for arch: {platform.machine()}",
        f"Job dir: {ROOT}",
        f"Commit SHA: {git_output('rev-parse', 'HEAD')}",
        "",
        "|date|job status|comment|",
        "|---|---|---|",
        f"|{submitted}|submitted|selected legs: {selected}|",
        f"|{started}|running|executed the committed gate plan|",
        f"|{finished}|finished|{details}|",
        f"|{finished}|test-result|_Reason_: {markdown_cell(reason)}|",
        "",
    ]
    return "\n".join(lines), exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=("eessi", "pixi", "all"), required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument(
        "--gist",
        action="store_true",
        help="create a secret gist when GH_TOKEN is set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submitted = utc_now()
    with PLAN_PATH.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    cpu_count = limit_cpu_affinity(int(plan["max_cpus"]))
    started = utc_now()
    selected = ["eessi", "pixi"] if args.leg == "all" else [args.leg]
    legs = [
        eessi_leg(plan) if name == "eessi" else pixi_leg(plan)
        for name in selected
    ]
    finished = utc_now()
    report, exit_code = make_report(
        legs, submitted, started, finished, cpu_count
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote {args.report}")
    if args.gist and os.environ.get("GH_TOKEN"):
        gist_code, gist_output = run_capture(
            [
                "gh",
                "gist",
                "create",
                "--secret",
                "--desc",
                "rgsaddle build and test gate",
                str(args.report),
            ]
        )
        if gist_code != 0:
            return gist_code
        urls = re.findall(r"https://gist\.github\.com/\S+", gist_output)
        if urls:
            print(urls[-1])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
