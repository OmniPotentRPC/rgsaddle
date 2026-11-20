#!/usr/bin/env python3
"""Fail when any public Rust, C ABI, or Python item is missing from docs/orgmode, or when C symbols or llms.txt are stale."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import render_c_symbols  # noqa: E402
import render_llms  # noqa: E402

RUST_TOKEN_RE = re.compile(
    r"//[^\n]*|"
    r"/\*[\s\S]*?\*/|"
    r"\bb?r(#+)\"[\s\S]*?\"\1|"
    r"\bb?r\"[\s\S]*?\"|"
    r"b?\"(?:\\[\s\S]|[^\"\\])*\"|"
    r"b?\x27(?:\\.|[^\x27\\\n])\x27"
)


def strip_rust_comments_and_tests(text: str) -> str:
    cleaned = RUST_TOKEN_RE.sub(
        lambda m: "\n" * m.group(0).count("\n"),
        text,
    )
    while True:
        match = re.search(r"#\s*\[\s*cfg\s*\([^)]*\btest\b[^)]*\)\s*\]", cleaned)
        if not match:
            break
        pos = match.end()
        while pos < len(cleaned) and cleaned[pos] not in "{;":
            pos += 1
        if pos < len(cleaned) and cleaned[pos] == "{":
            depth = 1
            pos += 1
            while pos < len(cleaned) and depth > 0:
                if cleaned[pos] == "{":
                    depth += 1
                elif cleaned[pos] == "}":
                    depth -= 1
                pos += 1
        elif pos < len(cleaned) and cleaned[pos] == ";":
            pos += 1
        cleaned = cleaned[: match.start()] + cleaned[pos:]
    return cleaned


def parse_pub_uses(txt: str) -> tuple[set[str], list[str]]:
    names: set[str] = set()
    globs: list[str] = []
    for match in re.finditer(r"(?<!\()\bpub\s+use\s+([^;]+);", txt, flags=re.S):
        stmt = match.group(1)
        for glob_mod in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*::\s*\*", stmt):
            globs.append(glob_mod)
        for orig, alias in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)", stmt
        ):
            names.add(orig)
            names.add(alias)
        stripped = re.sub(r"\s+as\s+[A-Za-z_][A-Za-z0-9_]*", "", stmt)
        for ident in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)(?!\s*::)", stripped):
            if ident not in ("self", "super", "crate"):
                names.add(ident)
    return names, globs


def scan_file_items(
    txt: str, allowed_names: set[str] | None = None
) -> tuple[list[str], list[str], list[str], list[tuple[str, str, str]]]:
    depths = [0] * (len(txt) + 1)
    depth = 0
    for idx, ch in enumerate(txt):
        depths[idx] = depth
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    types: list[str] = []
    fns: list[str] = []
    consts: list[str] = []
    cfields: list[tuple[str, str, str]] = []
    for match in re.finditer(
        r"(?<!\()\bpub\s+(?:struct|enum|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*)", txt
    ):
        if depths[match.start()] == 0:
            name = match.group(1)
            if allowed_names is None or name in allowed_names:
                types.append(name)
    for match in re.finditer(
        r"(?<!\()\bpub\s+(?:const\s+|async\s+|unsafe\s+)*fn\s+([A-Za-z_][A-Za-z0-9_]*)", txt
    ):
        if depths[match.start()] == 0:
            name = match.group(1)
            if allowed_names is None or name in allowed_names:
                fns.append(name)
    for match in re.finditer(
        r"(?<!\()\bpub\s+(?:const|static)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", txt
    ):
        if depths[match.start()] == 0:
            name = match.group(1)
            if allowed_names is None or name in allowed_names:
                consts.append(name)
    public_type_names = set(types)
    for match in re.finditer(
        r"with_setters!\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{([^}]*)\}\s*\)", txt, flags=re.S
    ):
        sname = match.group(1)
        if sname in public_type_names:
            for fmatch in re.finditer(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*=>\s*([A-Za-z_][A-Za-z0-9_]*)\s*:",
                match.group(2),
            ):
                cfields.append((sname, fmatch.group(2), fmatch.group(1)))
    for imatch in re.finditer(
        r"\bimpl(?:\s*<[^>]*>)?\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*<[^>]*>)?\s*\{", txt
    ):
        sname = imatch.group(1)
        if sname not in public_type_names:
            continue
        start = imatch.end()
        d2 = 1
        pos = start
        while pos < len(txt) and d2 > 0:
            if txt[pos] == "{":
                d2 += 1
            elif txt[pos] == "}":
                d2 -= 1
            pos += 1
        ibody = txt[start : pos - 1]
        for wmatch in re.finditer(
            r"\bpub\s+(?:const\s+)?fn\s+(with_[A-Za-z0-9_]+)\b", ibody
        ):
            wname = wmatch.group(1)
            cfields.append((sname, wname[5:], wname))
    return types, fns, consts, cfields


def collect_rust_inventory() -> tuple[list[str], list[str], list[str], list[str], list[tuple[str, str, str]]]:
    lib_src = strip_rust_comments_and_tests((ROOT / "src" / "lib.rs").read_text(encoding="utf-8"))
    root_reexports, _ = parse_pub_uses(lib_src)
    pub_mods = sorted(
        m
        for m in re.findall(r"(?<!\()\bpub\s+mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", lib_src)
        if not m.startswith("capi")
    )
    types: set[str] = {"VocnFound", "VocnPrimitive"}
    fns: set[str] = set()
    consts: set[str] = set()
    config_fields: dict[tuple[str, str], str] = {}

    priv_root_mods = re.findall(r"(?<!pub\s)\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", lib_src)
    for pm in priv_root_mods:
        if pm in ("config", "kani_proofs", "differential"):
            continue
        fpath = ROOT / "src" / f"{pm}.rs"
        if fpath.is_file():
            txt = strip_rust_comments_and_tests(fpath.read_text(encoding="utf-8"))
            t_list, f_list, c_list, cf_list = scan_file_items(txt, root_reexports)
            types.update(t_list)
            fns.update(f_list)
            consts.update(c_list)
            for sname, fname, setter in cf_list:
                config_fields[(sname, setter)] = fname

    for mod in pub_mods:
        f1 = ROOT / "src" / f"{mod}.rs"
        f2 = ROOT / "src" / mod / "mod.rs"
        target = f1 if f1.is_file() else f2
        txt = strip_rust_comments_and_tests(target.read_text(encoding="utf-8"))
        mod_reexports, mod_globs = parse_pub_uses(txt)
        allowed_in_subs = mod_reexports | root_reexports
        t_list, f_list, c_list, cf_list = scan_file_items(txt, None)
        types.update(t_list)
        fns.update(f_list)
        consts.update(c_list)
        for sname, fname, setter in cf_list:
            config_fields[(sname, setter)] = fname

        sub_dir = ROOT / "src" / mod
        for sub in re.findall(r"(?<!pub\s)\bmod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", txt):
            sf = sub_dir / f"{sub}.rs"
            if sf.is_file():
                stxt = strip_rust_comments_and_tests(sf.read_text(encoding="utf-8"))
                allow = None if sub in mod_globs else allowed_in_subs
                t_list, f_list, c_list, cf_list = scan_file_items(stxt, allow)
                types.update(t_list)
                fns.update(f_list)
                consts.update(c_list)
                for sname, fname, setter in cf_list:
                    config_fields[(sname, setter)] = fname

    cfg_items = [
        (sname, fname, setter) for (sname, setter), fname in sorted(config_fields.items())
    ]
    return pub_mods, sorted(types), sorted(fns), sorted(consts), cfg_items


CFG_ATTR_RE = re.compile(r"#\s*\[\s*cfg\s*\((.*)\)\s*\]\s*$")
FEATURE_RE = re.compile(r"feature\s*=\s*\"([A-Za-z0-9_-]+)\"")
EXPORT_RE = re.compile(r"#\s*\[\s*(?:unsafe\s*\(\s*)?no_mangle\s*\)?\s*\]")
MOD_RE = re.compile(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")


def cfg_features(attrs: list[str]) -> set[str]:
    """Cargo features, other than capi, that the cfg attributes in `attrs` require."""
    found: set[str] = set()
    for attr in attrs:
        match = CFG_ATTR_RE.match(attr.strip())
        if match and "not(" not in match.group(1) and "test" not in match.group(1):
            found.update(FEATURE_RE.findall(match.group(1)))
    return found - {"capi"}


def attributes_above(lines: list[str], index: int) -> list[str]:
    """Attribute, doc and comment lines directly above `lines[index]`."""
    attrs: list[str] = []
    pos = index - 1
    while pos >= 0 and lines[pos].lstrip().startswith(("#[", "//")):
        attrs.append(lines[pos])
        pos -= 1
    return attrs


def module_gates() -> dict[Path, set[str]]:
    """Features each capi source file needs because of how its module is declared."""
    gates: dict[Path, set[str]] = {}
    src = ROOT / "src"
    roots = [src / "lib.rs", src / "capi.rs"]
    for root in roots:
        lines = root.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = MOD_RE.match(line)
            if not match:
                continue
            name = match.group(1)
            if not name.startswith("capi") and root.name == "lib.rs":
                continue
            features = cfg_features(attributes_above(lines, index))
            candidates = [src / f"{name}.rs", src / root.stem / f"{name}.rs"]
            for path in candidates:
                if path.is_file():
                    gates[path] = features
    return gates


def collect_c_exports() -> dict[str, set[str]]:
    """Every exported C symbol, with the features that gate it."""
    gates = module_gates()
    exports: dict[str, set[str]] = {}
    sources = [ROOT / "src" / "capi.rs", *sorted((ROOT / "src").glob("capi_*.rs"))]
    sources += sorted((ROOT / "src" / "capi").glob("*.rs"))
    for path in sources:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not EXPORT_RE.search(line):
                continue
            for follow in lines[index + 1 :]:
                match = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", follow)
                if match:
                    break
            else:
                continue
            features = cfg_features(attributes_above(lines, index))
            # Attributes may sit above the doc comment or between it and the fn.
            pos = index + 1
            while pos < len(lines) and not re.search(r"\bfn\s", lines[pos]):
                features |= cfg_features([lines[pos]])
                pos += 1
            exports[match.group(1)] = features | gates.get(path, set())
    return exports


def check_c_feature_gating(header: str) -> list[str]:
    """Header declarations against the symbols the Rust sources export.

    Every declared function must be exported and the other way round, and a
    function that the library exports only under a Cargo feature must say so
    in its header comment with "Requires the `feature` feature."
    """
    exports = collect_c_exports()
    blanked = render_c_symbols._blank_comments(header)
    declared: dict[str, set[str]] = {}
    for match in render_c_symbols.DECL.finditer(blanked):
        name = match.group(1)
        if name.endswith("_t") or name.endswith("_fn") or name in declared:
            continue
        comment = render_c_symbols.preceding_comment(header, match.start())
        declared[name] = render_c_symbols.required_features(comment)
    errors: list[str] = []
    for name in sorted(set(declared) - set(exports)):
        errors.append(f"C symbol {name} is declared in include/rgsaddle.h but not exported")
    for name in sorted(set(exports) - set(declared)):
        errors.append(f"C symbol {name} is exported but not declared in include/rgsaddle.h")
    for name in sorted(set(declared) & set(exports)):
        if declared[name] != exports[name]:
            said = ", ".join(sorted(declared[name])) or "none"
            gated = ", ".join(sorted(exports[name])) or "none"
            errors.append(
                f"C symbol {name}: the header requires feature {said}, the sources gate it on {gated}"
            )
    return errors


def collect_python_inventory() -> list[str]:
    init_tree = ast.parse((ROOT / "python" / "rgsaddle" / "__init__.py").read_text(encoding="utf-8"))
    core_tree = ast.parse((ROOT / "python" / "rgsaddle" / "_core.pyi").read_text(encoding="utf-8"))
    ase_tree = ast.parse((ROOT / "python" / "rgsaddle" / "ase.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in init_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
    for node in core_tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                    names.add(sub.name)
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    if not sub.target.id.startswith("_"):
                        names.add(sub.target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for node in ase_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.add(node.name)
    return sorted(names)


def extract_documented_identifiers() -> set[str]:
    documented: set[str] = set()
    org_dir = ROOT / "docs" / "orgmode"
    for path in sorted(org_dir.rglob("*.org")):
        if path.name == "changelog.org":
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"[=~]([^=~\n]+)[=~]", text):
            for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", match.group(1)):
                documented.add(ident)
        for match in re.finditer(r"#\+begin_src\b.*?#\+end_src", text, flags=re.S | re.I):
            for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", match.group(0)):
                documented.add(ident)
    return documented


def main() -> int:
    header = render_c_symbols.HEADER.read_text(encoding="utf-8")
    pairs = render_c_symbols.entries(header)
    types = render_c_symbols.type_entries(header)
    constants = render_c_symbols.constant_entries(header)
    all_c = pairs + types + constants
    page = render_c_symbols.OUT.read_text(encoding="utf-8")
    missing_c = [name for name, _ in all_c if f"={name}=" not in page]
    if missing_c:
        print("C index is missing:\n" + "\n".join(missing_c), file=sys.stderr)
        return 1
    fresh = render_c_symbols.render(pairs, types, constants)
    if fresh != page:
        print("C index is stale; run python3 docs/render_c_symbols.py", file=sys.stderr)
        return 1

    gating = check_c_feature_gating(header)
    if gating:
        print("C feature gating:\n" + "\n".join(gating), file=sys.stderr)
        return 1

    documented = extract_documented_identifiers()
    pub_mods, pub_types, pub_fns, pub_consts, cfg_fields = collect_rust_inventory()
    py_items = collect_python_inventory()

    errors: list[str] = []
    for mod in pub_mods:
        if mod not in documented:
            errors.append(f"Rust pub mod {mod}")
    for tname in pub_types:
        if tname not in documented:
            errors.append(f"Rust pub type {tname}")
    for fname in pub_fns:
        if fname not in documented:
            errors.append(f"Rust pub fn {fname}")
    for cname in pub_consts:
        if cname not in documented:
            errors.append(f"Rust pub const {cname}")
    for sname, fname, setter in cfg_fields:
        if fname not in documented and setter not in documented:
            errors.append(f"Rust config field {sname}.{fname} ({setter})")
    for py_name in py_items:
        if py_name not in documented:
            errors.append(f"Python binding {py_name}")

    if errors:
        print(
            "Public items missing from docs/orgmode:\n" + "\n".join(errors),
            file=sys.stderr,
        )
        return 1

    if render_llms.main_check_only() != 0:
        return 1

    print(
        f"c symbols ok ({len(all_c)}: {len(pairs)} functions, {len(types)} types, {len(constants)} constants); "
        f"rust items ok ({len(pub_mods)} mods, {len(pub_types)} types, {len(pub_fns)} fns, "
        f"{len(pub_consts)} consts, {len(cfg_fields)} config builders); "
        f"python bindings ok ({len(py_items)}); llms.txt ok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
