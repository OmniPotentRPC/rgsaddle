"""A missing side gradient extrapolated as g- = 2 g0 - g+ makes the
central curvature equal the forward curvature with delta = DeltaR / 2.

Goswami, Gunde, Jonsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Eq. 2.
src/minmode.rs uses dr as that half-separation under FiniteDifference::Forward.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

import sympy as sp

n = 3
g0, gp, gm = sp.Matrix(sp.symbols(f"g0:{n}")), sp.Matrix(sp.symbols(f"gp:{n}")), None
N = sp.Matrix(sp.symbols("N0:3"))
Delta = sp.symbols("DeltaR", positive=True)
gm = 2 * g0 - gp
central = (gp - gm).dot(N) / Delta
forward = (gp - g0).dot(N) / (Delta / 2)
assert sp.simplify(central - forward) == 0

out = {
    "identity": "g- = 2 g0 - g+ implies (g+ - g-)·N / DeltaR = (g+ - g0)·N / (DeltaR/2)",
    "citation": "doi:10.3389/fchem.2026.1807063",
}
path = Path(__file__).resolve().parents[1] / "golden" / "extrapolated.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print("extrapolated central/forward: ok")
print(f"central - forward = {sp.simplify(central - forward)}")
print(f"wrote {path}")
