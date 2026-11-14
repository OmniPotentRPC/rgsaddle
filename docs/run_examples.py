#!/usr/bin/env python3
"""Run every code block of the narrative docs and the README.

A Rust block is a complete program. It is built as an example of a
scratch crate under ``target/docs-examples`` that depends on this
checkout, and run. A Python block runs under the current interpreter.
An ``sh`` block runs under ``bash -euo pipefail``. A ``console`` block
runs the lines that start with ``$ ``. All of them run from the
repository root, except a block marked ``:dir new-crate``, which runs
in a crate that ``cargo new`` has just made.

When an example block follows a source block with only blank lines
between them, every non-blank line of the example block must appear in
the block's standard output. Times such as ``finished in 0.72s`` are
not compared.

A ``toml`` block is a dependency table. Each version must equal the
requirement in ``Cargo.toml``, or the package version for rgsaddle.

An Org ``#+INCLUDE`` must name a file in the repository and a line
range inside it.

A block that does not run carries a reason: ``:skip "reason"`` on the
Org ``#+begin_src`` line, or ``<!-- docs-example: skip "reason" -->``
on the line before a README fence. A README fence is the output of the
block above it when the line before it is
``<!-- docs-example: output -->``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "docs" / "orgmode"
SCRATCH = ROOT / "target" / "docs-examples"
GENERATED = {"changelog.org", "c-symbols.org"}
RUN = {"rust", "python", "sh", "console"}
TIME = re.compile(r"finished in [0-9.]+m?s")
TIMEOUT = 3600


@dataclass
class Block:
    page: Path
    line: int
    lang: str
    code: str
    skip: str | None = None
    expected: list[str] = field(default_factory=list)
    new_crate: bool = False

    @property
    def name(self) -> str:
        return f"{self.page.relative_to(ROOT)}:{self.line}"

    @property
    def slug(self) -> str:
        stem = re.sub(r"[^a-z0-9]+", "_", str(self.page.relative_to(ROOT)).lower())
        return f"{stem}_{self.line}"


def header_args(text: str) -> dict[str, str]:
    args: dict[str, str] = {}
    for key, quoted, bare in re.findall(r':([\w-]+)\s+(?:"([^"]*)"|(\S+))', text):
        args[key] = quoted or bare
    return args


def org_blocks(path: Path) -> tuple[list[Block], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    errors: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        include = re.match(r'\s*#\+INCLUDE:\s*"([^"]+)"(.*)$', line, re.I)
        if include:
            errors.extend(check_include(path, i + 1, include.group(1), include.group(2)))
        start = re.match(r"\s*#\+begin_src\s+(\S+)(.*)$", line, re.I)
        if not start:
            i += 1
            continue
        lang = start.group(1).lower()
        args = header_args(start.group(2))
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not re.match(r"\s*#\+end_src", lines[j], re.I):
            body.append(lines[j])
            j += 1
        block = Block(path, i + 1, lang, "\n".join(body) + "\n", args.get("skip"))
        if "skip" in args and not args["skip"].strip():
            errors.append(f"{block.name}: :skip needs a reason")
        if args.get("dir") == "new-crate":
            block.new_crate = True
        elif "dir" in args:
            errors.append(f"{block.name}: :dir takes new-crate")
        k = j + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k < len(lines) and re.match(r"\s*#\+begin_example", lines[k], re.I):
            k += 1
            while k < len(lines) and not re.match(r"\s*#\+end_example", lines[k], re.I):
                if lines[k].strip():
                    block.expected.append(lines[k].strip())
                k += 1
        blocks.append(block)
        i = j + 1
    return blocks, errors


def check_include(page: Path, number: int, target: str, rest: str) -> list[str]:
    where = f"{page.relative_to(ROOT)}:{number}"
    path = (page.parent / target).resolve()
    if not path.is_file():
        return [f"{where}: #+INCLUDE names a missing file {target}"]
    span = re.search(r':lines\s+"(\d*)-(\d*)"', rest)
    if not span:
        return []
    count = len(path.read_text(encoding="utf-8").splitlines())
    first = int(span.group(1) or 1)
    last = int(span.group(2) or count + 1) - 1
    if not 1 <= first <= last <= count:
        return [f"{where}: #+INCLUDE lines {span.group(1)}-{span.group(2)} fall outside {target} ({count} lines)"]
    return []


def readme_blocks(path: Path) -> tuple[list[Block], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    errors: list[str] = []
    i = 0
    while i < len(lines):
        start = re.match(r"```(\S*)\s*$", lines[i])
        if not start:
            i += 1
            continue
        before = lines[i - 1].strip() if i else ""
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```"):
            body.append(lines[j])
            j += 1
        if before == "<!-- docs-example: output -->":
            if not blocks:
                errors.append(f"{path.name}:{i + 1}: output block with no block above it")
            else:
                blocks[-1].expected = [b.strip() for b in body if b.strip()]
        else:
            marker = re.match(r'<!-- docs-example: skip "([^"]+)" -->$', before)
            lang = {"": "text", "sh": "sh", "bash": "sh"}.get(start.group(1), start.group(1))
            blocks.append(Block(path, i + 1, lang, "\n".join(body) + "\n", marker.group(1) if marker else None))
        i = j + 1
    return blocks, errors


def collect() -> tuple[list[Block], list[str]]:
    blocks: list[Block] = []
    errors: list[str] = []
    pages = sorted(p for p in ORG.rglob("*.org") if p.name not in GENERATED)
    for page in pages:
        found, bad = org_blocks(page)
        blocks += found
        errors += bad
    found, bad = readme_blocks(ROOT / "README.md")
    blocks += found
    errors += bad
    return blocks, errors


def cargo_manifest() -> dict:
    return tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))


def check_toml(block: Block, manifest: dict) -> list[str]:
    try:
        table = tomllib.loads(block.code)
    except tomllib.TOMLDecodeError as err:
        return [f"{block.name}: {err}"]
    errors = []
    ours = manifest["dependencies"]
    for name, spec in table.get("dependencies", {}).items():
        given = spec if isinstance(spec, str) else spec.get("version")
        if name == "rgsaddle":
            want = manifest["package"]["version"]
        elif name in ours:
            want = ours[name] if isinstance(ours[name], str) else ours[name].get("version")
        else:
            errors.append(f"{block.name}: {name} is not a dependency of rgsaddle")
            continue
        if given != want:
            errors.append(f"{block.name}: {name} = {given!r}, Cargo.toml has {want!r}")
    return errors


def compare(block: Block, out: str) -> list[str]:
    have = [TIME.sub("finished in", line.strip()) for line in out.splitlines()]
    missing = [line for line in block.expected if TIME.sub("finished in", line) not in have]
    if missing:
        return [f"{block.name}: output lacks\n    " + "\n    ".join(missing) + "\n  got\n    " + "\n    ".join(have[-40:])]
    return []


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=TIMEOUT, check=False)


def failure(block: Block, proc: subprocess.CompletedProcess[str]) -> str:
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-40:]
    return f"{block.name}: exit {proc.returncode}\n    " + "\n    ".join(tail)


def rust_crate(blocks: list[Block], manifest: dict) -> Path:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "examples").mkdir(parents=True)
    ndarray = manifest["dependencies"]["ndarray"]
    (SCRATCH / "Cargo.toml").write_text(
        "[package]\n"
        'name = "docs-examples"\n'
        'version = "0.0.0"\n'
        'edition = "2024"\n'
        "publish = false\n\n"
        "[dependencies]\n"
        'rgsaddle = { path = "../.." }\n'
        f'ndarray = "{ndarray}"\n\n'
        "[workspace]\n",
        encoding="utf-8",
    )
    (SCRATCH / "src").mkdir()
    (SCRATCH / "src" / "lib.rs").write_text("", encoding="utf-8")
    shutil.copy(ROOT / "Cargo.lock", SCRATCH / "Cargo.lock")
    for block in blocks:
        (SCRATCH / "examples" / f"{block.slug}.rs").write_text(block.code, encoding="utf-8")
    return SCRATCH


def run_rust(blocks: list[Block], manifest: dict) -> list[str]:
    if not blocks:
        return []
    crate = rust_crate(blocks, manifest)
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target"))
    errors = []
    for block in blocks:
        missing = "fn main" not in block.code
        if missing:
            errors.append(f"{block.name}: a Rust block is a complete program with fn main")
            continue
        print(f"rust    {block.name}", flush=True)
        proc = run(["cargo", "run", "--quiet", "--example", block.slug], crate, env)
        if proc.returncode != 0:
            errors.append(failure(block, proc))
            continue
        errors += compare(block, proc.stdout)
    return errors


def shell_lines(block: Block) -> str:
    if block.lang != "console":
        return block.code
    commands: list[str] = []
    continued = False
    for line in block.code.splitlines():
        if line.startswith("$ "):
            commands.append(line[2:])
        elif continued:
            commands[-1] += "\n" + line
        continued = bool(commands) and commands[-1].rstrip().endswith("\\")
    return "\n".join(commands) + "\n"


def run_other(block: Block) -> list[str]:
    print(f"{block.lang:<7} {block.name}", flush=True)
    cwd = ROOT
    if block.new_crate:
        cwd = SCRATCH / "new-crate" / block.slug
        if cwd.exists():
            shutil.rmtree(cwd)
        cwd.parent.mkdir(parents=True, exist_ok=True)
        made = run(["cargo", "new", "--quiet", "--vcs", "none", "--name", "host", str(cwd)], ROOT)
        if made.returncode != 0:
            return [failure(block, made)]
    if block.lang == "python":
        proc = run([sys.executable, "-c", block.code], cwd)
    else:
        proc = run(["bash", "-euo", "pipefail", "-c", shell_lines(block)], cwd)
    if proc.returncode != 0:
        return [failure(block, proc)]
    return compare(block, proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list the blocks and what happens to each")
    parser.add_argument("--lang", action="append", help="run only these languages")
    parser.add_argument("--page", action="append", help="run only blocks whose page path contains this text")
    args = parser.parse_args()

    blocks, errors = collect()
    manifest = cargo_manifest()
    chosen = [
        b
        for b in blocks
        if (not args.lang or b.lang in args.lang) and (not args.page or any(p in b.name for p in args.page))
    ]
    if args.list:
        for b in chosen:
            state = f"skip: {b.skip}" if b.skip else ("check" if b.lang == "toml" else "run" if b.lang in RUN else "text")
            print(f"{b.name:<45} {b.lang:<8} {state}")
        return 0

    skipped = [b for b in chosen if b.skip]
    live = [b for b in chosen if not b.skip]
    for b in live:
        if b.lang not in RUN | {"toml", "text"}:
            errors.append(f'{b.name}: {b.lang} blocks do not run; add :skip "reason"')
    for b in live:
        if b.lang == "toml":
            errors += check_toml(b, manifest)
    errors += run_rust([b for b in live if b.lang == "rust"], manifest)
    for b in live:
        if b.lang in RUN - {"rust"}:
            errors += run_other(b)

    counts = {lang: sum(1 for b in live if b.lang == lang) for lang in sorted(RUN | {"toml"})}
    print("\nskipped:")
    for b in skipped:
        print(f"  {b.name} ({b.lang}): {b.skip}")
    ran = ", ".join(f"{n} {lang}" for lang, n in counts.items())
    if errors:
        print("\n" + "\n".join(errors), file=sys.stderr)
        print(f"\ndocs examples: {len(errors)} failed; {ran}; {len(skipped)} skipped", file=sys.stderr)
        return 1
    print(f"\ndocs examples ok: {ran}; {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
