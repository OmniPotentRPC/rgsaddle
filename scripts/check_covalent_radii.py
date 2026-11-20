#!/usr/bin/env python3
"""Check src/covalent.rs against independent sources of Cordero et al. 2008.

Cordero, Gomez, Platero-Prats, Reves, Echeverria, Cremades, Barragan,
Alvarez, "Covalent radii revisited", Dalton Trans. (2008) 2832-2838,
doi:10.1039/B801115J, Table 2.

The Rust table must equal ase.data.covalent_radii for Z = 1..96 (ASE
takes the same table and Sella reads ASE), and, when the optional
`mendeleev` package is installed, its `covalent_radius_cordero` for every
element Table 2 does not split. Table 2 splits carbon (sp3, sp2, sp) and
manganese, iron and cobalt (low spin, high spin); for those the Rust
value must be the one Sella takes, and it must be one of the published
entries.

    python scripts/check_covalent_radii.py
"""
import re
import sys
from pathlib import Path

from ase.data import covalent_radii

SPLIT_ENTRIES_PM = {
    6: {"sp3": 76, "sp2": 73, "sp": 69},
    25: {"low spin": 139, "high spin": 161},
    26: {"low spin": 132, "high spin": 152},
    27: {"low spin": 126, "high spin": 150},
}
SELLA_CHOICE_PM = {6: 76, 25: 139, 26: 132, 27: 126}


def rust_table():
    text = (Path(__file__).resolve().parent.parent / "src" / "covalent.rs").read_text()
    start = text.index("CORDERO_2008_PM: [u16; 96] = [")
    body = text[start : text.index("];", start)]
    body = re.sub(r"//[^\n]*", "", body).split("=", 1)[1]
    return [int(v) for v in re.findall(r"\b\d+\b", body)]


def main():
    table = rust_table()
    failures = []
    if len(table) != 96:
        failures.append(f"table has {len(table)} entries, expected 96")
    try:
        from mendeleev import element
    except ImportError:
        element = None
    for z in range(1, min(len(table), 96) + 1):
        ours = table[z - 1]
        ase = round(covalent_radii[z] * 100)
        if ours != ase:
            failures.append(f"Z={z}: rust {ours} pm, ASE {ase} pm")
        if z in SPLIT_ENTRIES_PM:
            if ours != SELLA_CHOICE_PM[z]:
                failures.append(f"Z={z}: rust {ours} pm, Sella takes {SELLA_CHOICE_PM[z]} pm")
            if ours not in SPLIT_ENTRIES_PM[z].values():
                failures.append(f"Z={z}: {ours} pm is not a Table 2 entry")
        elif element is not None:
            other = element(z).covalent_radius_cordero
            if other is None or round(other) != ours:
                failures.append(f"Z={z}: rust {ours} pm, mendeleev {other} pm")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"covalent radii: {len(table)} entries agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
