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
    tests: list[TestCounts] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return FORBIDDEN_OUTPUT.search(self.output) is None

    @property
    def passed(self) -> bool:
        counts_present = self.kind != "test" or bool(self.tests)
        no_failures = all(counts.failed == 0 for counts in self.tests)
        return self.returncode == 0 and self.clean and counts_present and no_failures


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


@dataclass(frozen=True)
class TestCounts:
    tool: str
    passed: int
    failed: int
    ignored: int
    skipped: int

    def __str__(self) -> str:
        return (
            f"{self.tool}: {self.passed} passed, {self.failed} failed, "
            f"{self.ignored} ignored, {self.skipped} skipped"
        )


CARGO_RESULT = re.compile(
    r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"\d+ measured; (\d+) filtered out"
)
PYTEST_SUMMARY = re.compile(
    r"(?m)^=*\s*((?:\d+ [a-z]+, )*\d+ [a-z]+|no tests ran) in [\d.]+s\b"
)
PYTEST_OUTCOME = re.compile(r"(\d+) ([a-z]+)")
MESON_LABELS = ("Ok", "Expected Fail", "Fail", "Unexpected Pass", "Skipped", "Timeout")


def cargo_counts(output: str) -> TestCounts | None:
    """Sum every libtest summary; filtered out tests count as skipped."""
    matches = CARGO_RESULT.findall(output)
    if not matches:
        return None
    passed, failed, ignored, skipped = (
        sum(int(match[index]) for match in matches) for index in range(4)
    )
    return TestCounts("cargo", passed, failed, ignored, skipped)


def pytest_counts(output: str) -> TestCounts | None:
    """Read the last summary; errors fail, xfailed is ignored, deselected skipped."""
    summaries = PYTEST_SUMMARY.findall(output)
    if not summaries:
        return None
    outcomes: dict[str, int] = {}
    for number, outcome in PYTEST_OUTCOME.findall(summaries[-1]):
        outcomes[outcome] = outcomes.get(outcome, 0) + int(number)
    return TestCounts(
        "pytest",
        outcomes.get("passed", 0) + outcomes.get("xpassed", 0),
        outcomes.get("failed", 0) + outcomes.get("error", 0) + outcomes.get("errors", 0),
        outcomes.get("xfailed", 0),
        outcomes.get("skipped", 0) + outcomes.get("deselected", 0),
    )


def meson_counts(output: str) -> TestCounts | None:
    """Read the last summary; unexpected passes and timeouts fail."""
    values: dict[str, int] = {}
    for label in MESON_LABELS:
        found = re.findall(rf"(?m)^{label}:\s+(\d+)\s*$", output)
        if found:
            values[label] = int(found[-1])
    if "Ok" not in values or "Fail" not in values:
        return None
    return TestCounts(
        "Meson",
        values["Ok"],
        values["Fail"] + values.get("Unexpected Pass", 0) + values.get("Timeout", 0),
        values.get("Expected Fail", 0),
        values.get("Skipped", 0),
    )


def parse_test_counts(output: str) -> list[TestCounts]:
    parsed = (cargo_counts(output), pytest_counts(output), meson_counts(output))
    return [counts for counts in parsed if counts is not None]


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
                tests=parse_test_counts(output) if kind == "test" else [],
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
    summaries = [str(counts) for result in results for counts in result.tests]
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


def available_rust_modules(output: str) -> set[str]:
    return {
        match.rstrip(":")
        for match in re.findall(r"Rust/[A-Za-z0-9.+_-]+", output)
    }


def eessi_init_script(init: pathlib.Path) -> str:
    # EESSI's init reads optional shell variables such as PS1. Keep nounset
    # disabled only while sourcing it, then restore strict handling.
    return (
        "set -eo pipefail\n"
        "set +u\n"
        f"source {shlex.quote(str(init))}\n"
        "set -u"
    )


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

    arch = platform.machine()
    rust_modules = config["rust_modules"]
    rust_module = rust_modules.get(arch)
    if rust_module is None:
        return LegResult(
            name="eessi",
            status="failed",
            reason=f"no Rust module is pinned for architecture {arch}",
            artefacts=[
                f"EESSI stack: {stack}",
                f"compiler: {config['compiler']}",
                f"architecture: {arch}",
                "module list: not queried",
                "test summary: not run",
            ],
        )

    init_script = eessi_init_script(init)
    init_code, _ = run_capture(shell_argv(init_script))
    if init_code != 0:
        return LegResult(
            name="eessi",
            status="failed",
            reason=f"EESSI initialization failed with exit {init_code}",
            artefacts=[
                f"EESSI stack: {stack}",
                f"compiler: {config['compiler']}",
                f"Rust module: {rust_module}",
                f"EESSI init: failed with exit {init_code}",
                "module list: not queried",
                "test summary: not run",
            ],
        )

    module_probe = (
        f"{init_script}\n"
        f"module use {shlex.quote(str(modules))}\n"
        "module -t avail Rust 2>&1"
    )
    probe_code, module_output = run_capture(shell_argv(module_probe))
    if probe_code != 0:
        return LegResult(
            name="eessi",
            status="failed",
            reason=f"module avail Rust failed with exit {probe_code}",
            artefacts=[
                f"EESSI stack: {stack}",
                f"compiler: {config['compiler']}",
                f"Rust module: {rust_module}",
                "module list: query failed",
                "test summary: not run",
            ],
        )
    if rust_module not in available_rust_modules(module_output):
        return LegResult(
            name="eessi",
            status="failed",
            reason=f"pinned Rust module {rust_module} is unavailable on {arch}",
            artefacts=[
                f"EESSI stack: {stack}",
                f"compiler: {config['compiler']}",
                f"Rust module: {rust_module}",
                "module list: pinned module unavailable",
                "test summary: not run",
            ],
        )

    # Capture eb-local first: a failure inside eval "$(...)" is masked and
    # would leave the host toolchain in use.
    eb_command = f"{shlex.quote(eb_local)} dev {shlex.quote(rust_module)}"
    setup = (
        f"{init_script}\n"
        f"module use {shlex.quote(str(modules))}\n"
        "unset LD_LIBRARY_PATH\n"
        f"export EESSI_VERSION={shlex.quote(stack)}\n"
        f"eb_env=$({eb_command}) || "
        "{ echo 'eb-local failed; refusing to use the host toolchain' >&2; exit 1; }\n"
        'eval "$eb_env"'
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
    summaries = [str(counts) for result in results for counts in result.tests]
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
                summary = "; ".join(map(str, result.tests)) if result.tests else "none"
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
