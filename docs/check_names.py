#!/usr/bin/env python3
"""Fail when a docs page names a symbol or a file that does not exist.

The names are the inline code of the Org pages (``~x~`` and ``=x=``)
and of the README (backticks), outside source and example blocks. Each
one resolves against:

- the rustdoc HTML of rgsaddle and rgmin under ``target/doc``
  (``cargo doc --no-deps --all-features -p rgsaddle`` and
  ``cargo doc --no-deps -p rgmin``). ``A::b`` needs ``b`` to be a field,
  method, variant or associated item of ``A``, or an item of module
  ``A``. The fields of ``A { x, .. }`` must be fields of ``A``;
- the declarations of ``include/rgsaddle.h`` and the C++ headers for
  ``rgsaddle_*`` and ``RGSADDLE_*``;
- the Python sources and the stub under ``python/rgsaddle`` for
  ``rgsaddle.*`` and ``_core.*``. The keyword arguments of a call must
  be parameters of the function;
- the repository tree for a path such as ``tests/double_well.rs``.

A bare name must exist in one of those, or in ``docs/known_names.txt``,
which lists the names that belong to other projects.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "docs" / "orgmode"
DOC = ROOT / "target" / "doc"
GENERATED = {"changelog.org", "c-symbols.org"}
EXTENSIONS = {".rs", ".py", ".c", ".h", ".hpp", ".cpp", ".org", ".toml", ".json", ".csv", ".sh", ".el", ".md", ".sollya", ".dot", ".png", ".pyi", ".yml", ".txt", ".lock", ".cfg"}
IDENT = re.compile(r"[A-Za-z_]\w*")
RUST_PATH = re.compile(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+")
DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
FILE = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+/?")


class RustIndex:
    """Items, modules and members of the documented crates."""

    def __init__(self) -> None:
        self.items: dict[str, set[str]] = defaultdict(set)
        self.members: dict[str, set[str]] = defaultdict(set)
        self.fields: dict[str, set[str]] = defaultdict(set)
        self.modules: dict[str, set[str]] = defaultdict(set)
        self.crates: set[str] = set()

    def load(self, crate: str) -> None:
        base = DOC / crate
        if not (base / "index.html").is_file():
            raise SystemExit(f"{base}/index.html is missing; run cargo doc first")
        self.crates.add(crate)
        for page in base.rglob("*.html"):
            rel = page.relative_to(DOC)
            module = "::".join(rel.parts[:-1])
            if page.name == "index.html":
                self.modules[rel.parts[-2]].add(module)
                continue
            found = re.fullmatch(r"(struct|enum|fn|trait|constant|type|macro|static|union|traitalias)\.(\w+)\.html", page.name)
            if not found:
                continue
            name = found.group(2)
            self.items[name].add(module)
            text = page.read_text(encoding="utf-8", errors="replace")
            for kind, member, field in re.findall(
                r'id="(method|tymethod|structfield|variant|associatedconstant|associatedtype)\.(\w+)(?:\.field\.(\w+))?"', text
            ):
                self.members[name].add(member)
                if kind == "structfield":
                    self.fields[name].add(member)
                if field:
                    self.fields[f"{name}::{member}"].add(field)

    def names(self) -> set[str]:
        out = set(self.items) | set(self.modules)
        for members in self.members.values():
            out |= members
        for fields in self.fields.values():
            out |= fields
        return out

    def module_items(self, module: str) -> set[str]:
        out = set()
        for path in self.modules.get(module, ()):
            folder = DOC / path.replace("::", "/")
            for page in folder.glob("*.html"):
                found = re.fullmatch(r"\w+\.(\w+)\.html", page.name)
                if found:
                    out.add(found.group(1))
            out |= {p.name for p in folder.iterdir() if (p / "index.html").is_file()}
        return out

    def resolve(self, path: str) -> str | None:
        """None when the path resolves, otherwise the reason."""
        parts = path.split("::")
        head, rest = parts[0], parts[1:]
        if not rest:
            return None if head in self.items or head in self.modules else "unknown"
        if head in self.crates:
            if not rest:
                return None
            if rest[0] in self.items or rest[0] in self.modules:
                return self.resolve("::".join(rest))
            return f"{rest[0]} is not in {head}"
        if head in self.items:
            member = rest[0]
            if member not in self.members[head]:
                return f"{head} has no member {member}"
            if len(rest) > 1:
                return f"{path} nests past a member"
            return None
        if head in self.modules:
            inside = self.module_items(head)
            if rest[0] not in inside:
                return f"module {head} has no item {rest[0]}"
            if len(rest) > 1:
                return self.resolve("::".join(rest))
            return None
        return "unknown"


class PythonIndex:
    """Modules, classes, functions and parameters of the package."""

    def __init__(self) -> None:
        self.modules: dict[str, dict[str, object]] = {}
        self.names: set[str] = set()

    def load(self) -> None:
        base = ROOT / "python" / "rgsaddle"
        for module, file in (("rgsaddle._core", "_core.pyi"), ("rgsaddle.ase", "ase.py"), ("rgsaddle", "__init__.py")):
            tree = ast.parse((base / file).read_text(encoding="utf-8"))
            self.modules[module] = self.scan(tree.body, module)
        self.modules["rgsaddle"]["ase"] = self.modules["rgsaddle.ase"]
        self.modules["rgsaddle"]["_core"] = self.modules["rgsaddle._core"]

    def scan(self, body: list[ast.stmt], module: str) -> dict[str, object]:
        space: dict[str, object] = {}
        for node in body:
            if isinstance(node, ast.ClassDef):
                members: dict[str, object] = {}
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        members[sub.name] = self.params(sub)
                        for inner in ast.walk(sub):
                            if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) and inner.value.id == "self":
                                members.setdefault(inner.attr, None)
                    elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                        members[sub.target.id] = None
                    elif isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if isinstance(target, ast.Name):
                                members[target.id] = None
                init = members.get("__init__")
                space[node.name] = {"members": members, "params": init}
                self.names.add(node.name)
                self.names |= set(members)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                space[node.name] = {"members": {}, "params": self.params(node)}
                self.names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        space[target.id] = None
                        self.names.add(target.id)
            elif isinstance(node, ast.ImportFrom) and node.module == "rgsaddle._core":
                for alias in node.names:
                    space[alias.asname or alias.name] = ("rgsaddle._core", alias.name)
        return space

    def params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        args = node.args
        names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        self.names |= set(names)
        return [n for n in names if n not in ("self", "cls")]

    def lookup(self, dotted: str) -> tuple[object, str | None]:
        parts = dotted.split(".")
        if parts[0] == "_core":
            parts = ["rgsaddle", "_core", *parts[1:]]
        if parts[0] != "rgsaddle":
            return None, "unknown"
        node: object = self.modules["rgsaddle"]
        for i, part in enumerate(parts[1:], start=1):
            if isinstance(node, tuple):
                node = self.modules[node[0]].get(node[1])
            if isinstance(node, dict) and "members" in node:
                members = node["members"]
                if part not in members:
                    return None, f"{'.'.join(parts[:i])} has no attribute {part}"
                node = {"members": {}, "params": members[part]} if isinstance(members[part], list) else members[part]
            elif isinstance(node, dict):
                if part not in node:
                    return None, f"{'.'.join(parts[:i])} has no attribute {part}"
                node = node[part]
            else:
                return None, f"{'.'.join(parts[:i])} has no attribute {part}"
        if isinstance(node, tuple):
            node = self.modules[node[0]].get(node[1])
        return node, None


EXTERNAL_MODULES = (
    "builtins",
    "enum",
    "numpy",
    "ase",
    "ase.mep",
    "ase.mep.autoneb",
    "ase.mep.dimer",
    "ase.optimize",
    "ase.optimize.optimize",
    "ase.calculators.calculator",
)


class ExternalPython:
    """Names of the Python modules the pages compare against, by import."""

    def __init__(self) -> None:
        import importlib
        import inspect

        self.modules: dict[str, object] = {}
        self.names: set[str] = set()
        for name in EXTERNAL_MODULES:
            try:
                module = importlib.import_module(name)
            except ImportError as err:
                raise SystemExit(f"check_names needs {name}: {err}") from err
            self.modules[name] = module
            for attr in dir(module):
                if attr.startswith("__"):
                    continue
                self.names.add(attr)
                value = getattr(module, attr, None)
                if inspect.isclass(value):
                    self.names |= {a for a in dir(value) if not a.startswith("__")}
                    try:
                        self.names |= set(inspect.signature(value.__init__).parameters)
                    except (TypeError, ValueError):
                        pass

    def resolve(self, dotted: str) -> bool:
        import ase

        parts = dotted.split(".")
        if parts[0] in ("atoms", "Atoms"):
            node: object = ase.Atoms
            parts = parts[1:]
        else:
            node = None
            for i in range(len(parts), 0, -1):
                module = self.modules.get(".".join(parts[:i]))
                if module is not None:
                    node, parts = module, parts[i:]
                    break
            if node is None:
                return False
        for part in parts:
            if not hasattr(node, part):
                return False
            node = getattr(node, part)
        return True


def c_names() -> tuple[set[str], set[str]]:
    c: set[str] = set()
    cpp: set[str] = set()
    header = (ROOT / "include" / "rgsaddle.h").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
    c |= set(re.findall(r"\b(?:rgsaddle|RGSADDLE)_\w+", code))
    c |= set(IDENT.findall(code))
    for hpp in (ROOT / "include" / "rgsaddle").glob("*.hpp"):
        text = re.sub(r"//[^\n]*", "", hpp.read_text(encoding="utf-8"))
        cpp |= set(re.findall(r"\b(?:class|struct)\s+(\w+)", text))
        cpp |= set(re.findall(r"\b(\w+)\s*\(", text))
    return c, cpp


def known_names() -> tuple[set[str], set[str]]:
    names: set[str] = set()
    prefixes: set[str] = set()
    for line in (ROOT / "docs" / "known_names.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith("*"):
            prefixes.add(line[:-1])
        else:
            names.add(line)
    return names, prefixes


def strip_blocks(text: str, markdown: bool) -> list[str]:
    lines = text.splitlines()
    out = []
    inside = False
    for line in lines:
        if markdown:
            if line.startswith("```"):
                inside = not inside
                out.append("")
                continue
        else:
            if re.match(r"\s*#\+begin_(src|example|export)", line, re.I):
                inside = True
            if inside:
                if re.match(r"\s*#\+end_(src|example|export)", line, re.I):
                    inside = False
                out.append("")
                continue
            if re.match(r"\s*#\+", line):
                out.append("")
                continue
        out.append("" if inside else line)
    return out


def inline_code(line: str, markdown: bool) -> list[str]:
    if markdown:
        return re.findall(r"`([^`]+)`", line)
    pattern = r"(?:(?<=[\s(\[\"'/])|^)([~=])([^\s~=](?:[^\n]*?[^\s])?)\1(?=[\s).,;:!?'\"\]/-]|$)"
    return [m.group(2) for m in re.finditer(pattern, line)]


def pages() -> list[Path]:
    found = sorted(p for p in ORG.rglob("*.org") if p.name not in GENERATED)
    return [*found, ROOT / "README.md"]


SKIP_DIRS = {".git", "target", ".lake", "node_modules", "build", "__pycache__"}


def tracked_basenames() -> set[str]:
    import os

    names = set()
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        names |= set(files) | set(dirs)
    return names


BASENAMES: set[str] = set()


def repo_file(token: str) -> bool:
    clean = token.rstrip("/")
    if (ROOT / clean).exists():
        return True
    if (ORG / f"{clean}.org").is_file():
        return True
    if "/" not in clean:
        if not BASENAMES:
            BASENAMES.update(tracked_basenames())
        return clean in BASENAMES
    return False


def path_like(token: str) -> bool:
    """A token with a slash that names a file rather than a ratio."""
    if not FILE.fullmatch(token) or token.startswith("10."):
        return False
    first = token.split("/", 1)[0]
    return Path(token.rstrip("/")).suffix in EXTENSIONS or (ROOT / first).is_dir() or (ORG / first).is_dir()


class Checker:
    def __init__(self) -> None:
        self.rust = RustIndex()
        self.rust.load("rgsaddle")
        self.rust.load("rgmin")
        self.python = PythonIndex()
        self.python.load()
        self.c, self.cpp = c_names()
        self.external = ExternalPython()
        self.allow, self.allow_prefix = known_names()
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        self.features = set(manifest["features"])
        towncrier = tomllib.loads((ROOT / "towncrier.toml").read_text(encoding="utf-8"))
        self.features |= {kind["directory"] for kind in towncrier["tool"]["towncrier"]["type"]}
        top = {p.name for p in ROOT.iterdir()}
        self.bare = (
            self.rust.names() | self.python.names | self.c | self.cpp | self.features | top
            | self.external.names | self.allow
        )
        self.used: set[str] = set()

    def allowed(self, token: str) -> bool:
        if token in self.allow:
            self.used.add(token)
            return True
        for prefix in self.allow_prefix:
            if token.startswith(prefix):
                self.used.add(prefix + "*")
                return True
        return False

    def check(self, token: str) -> str | None:
        token = token.strip()
        if token.endswith("()"):
            token = token[:-2]
        if self.allowed(token):
            return None
        head = re.match(r"[\w:.]+", token)
        if not head:
            return None
        name = head.group(0)
        tail = token[len(name):]
        if "/" in token:
            if path_like(token):
                return None if repo_file(token) or self.allowed(token) else "no such file"
            return None
        if tail and not tail.startswith(("(", " {", "{")):
            return None
        if RUST_PATH.fullmatch(name):
            return self.rust_path(name, tail)
        if DOTTED.fullmatch(name):
            return self.dotted(name, tail)
        if IDENT.fullmatch(name):
            return self.ident(name, tail)
        return None

    def rust_path(self, name: str, tail: str) -> str | None:
        if name.startswith("rgsaddle::") and name.split("::", 1)[1] in self.cpp:
            return None
        reason = self.rust.resolve(name)
        if reason == "unknown":
            return None if self.allowed(name) or self.allowed(name.split("::")[0] + "::*") else f"{name} is not a known path"
        if reason:
            return reason
        return self.literal_fields(name, tail)

    def literal_fields(self, name: str, tail: str) -> str | None:
        if not tail.lstrip().startswith("{"):
            return None
        fields = self.rust.fields.get(name) or self.rust.fields.get(name.split("::")[-1], set())
        for part in tail.strip()[1:-1].split(","):
            field = part.split(":")[0].strip()
            if field and not field.startswith("..") and IDENT.fullmatch(field) and field not in fields:
                return f"{name} has no field {field}"
        return None

    def dotted(self, name: str, tail: str) -> str | None:
        first = name.split(".")[0]
        if Path(name).suffix in EXTENSIONS:
            return None if repo_file(name) or self.allowed(name) else "no such file"
        if first in ("rgsaddle", "_core"):
            node, reason = self.python.lookup(name)
            if reason:
                return reason
            if tail.startswith("("):
                return self.call(name, node, tail)
            return None
        if first.startswith("rgsaddle_") or first == "version":
            return None if all(part in self.c for part in name.split(".")) else f"{name} is not a C field"
        member = name.split(".")[1]
        python_class = self.python_class(first)
        if first in self.rust.items or python_class is not None:
            if member in self.rust.members.get(first, ()) or (python_class and member in python_class):
                return None
            return f"{first} has no member {member}"
        if self.external.resolve(name):
            return None
        return None if self.allowed(name) else f"{name} is not a known name"

    def python_class(self, name: str) -> dict[str, object] | None:
        for space in self.python.modules.values():
            node = space.get(name)
            if isinstance(node, dict) and "members" in node:
                return node["members"]
        return None

    def call(self, name: str, node: object, tail: str) -> str | None:
        params = node.get("params") if isinstance(node, dict) else None
        if not isinstance(params, list):
            return None
        try:
            expr = ast.parse(f"f{tail}", mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(expr, ast.Call):
            return None
        for keyword in expr.keywords:
            if keyword.arg and keyword.arg not in params:
                return f"{name} has no parameter {keyword.arg}"
        if len(expr.args) > len(params):
            return f"{name} takes {len(params)} arguments"
        return None

    def ident(self, name: str, tail: str) -> str | None:
        if name.startswith(("rgsaddle_", "RGSADDLE_")):
            return None if name in self.c else f"{name} is not in include/rgsaddle.h"
        if re.fullmatch(r"_[A-Z][A-Z0-9_]*", name):
            return None if any(c.endswith(name) for c in self.c) else f"no RGSADDLE_ name ends in {name}"
        if name in self.rust.items:
            return self.literal_fields(name, tail)
        if len(name) <= 2:
            return None
        if name in self.bare:
            if name in self.allow:
                self.used.add(name)
            return None
        if tail.startswith("(") and name in self.python.names:
            return None
        return f"{name} is not a known name"


def main() -> int:
    checker = Checker()
    errors = []
    count = 0
    for page in pages():
        markdown = page.suffix == ".md"
        for number, line in enumerate(strip_blocks(page.read_text(encoding="utf-8"), markdown), start=1):
            for token in inline_code(line, markdown):
                count += 1
                reason = checker.check(token)
                if reason:
                    errors.append(f"{page.relative_to(ROOT)}:{number}: {token}: {reason}")
    listed = checker.allow | {p + "*" for p in checker.allow_prefix}
    unused = sorted(listed - checker.used)
    if unused:
        errors.append("docs/known_names.txt lists names no page uses: " + ", ".join(unused))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"\nnames: {len(errors)} errors in {count} inline names", file=sys.stderr)
        return 1
    print(f"names ok ({count} inline names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
