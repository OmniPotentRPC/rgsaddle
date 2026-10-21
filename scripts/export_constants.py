#!/usr/bin/env python3
"""Check the Rust literals named by the stencil contracts.

The Lean model treats a tangent as normalised when it is nonzero.
Rust uses a finite floor. This script refuses a drift in those floors.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLES = {
    "tangent_norm_floor": ("src/stencil/triad.rs", "norm > 1e-10"),
    "dneb_norm_floor": ("src/projection.rs", "spring_perp_norm > 1e-10"),
    "mode_norm_floor": ("src/minmode.rs", "norm > 1e-14"),
    "default_dr": ("src/minmode.rs", "dr: 1e-3"),
    "mode_reset_angle": ("src/minmode.rs", "5.0 * PI / 180.0"),
}


def snapshot():
    out = {}
    for name, (path, needle) in NEEDLES.items():
        text = (ROOT / path).read_text()
        if needle not in text:
            raise SystemExit(f"{path} no longer contains {needle}")
        out[name] = {"path": path, "literal": needle}
    return out


def main():
    if "--check" not in sys.argv:
        raise SystemExit("usage: export_constants.py --check")
    path = ROOT / "scripts" / "constants.json"
    found = snapshot()
    recorded = json.loads(path.read_text())
    if json.dumps(recorded, sort_keys=True) != json.dumps(found, sort_keys=True):
        raise SystemExit(
            "scripts/constants.json does not match the Rust sources:\n"
            + json.dumps(found, indent=2)
        )
    print("export_constants: floors match scripts/constants.json")


if __name__ == "__main__":
    main()
