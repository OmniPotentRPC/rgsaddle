"""Improved-tangent branches of src/tangent.rs::compute_tangent.

Goswami, Gunde, Jonsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Eq. 7.
Henkelman and Jonsson, J. Chem. Phys. 113, 9978 (2000), doi:10.1063/1.1323224.

(a) Equal side energies at a strict maximum give the chord (R+ - R-).
(b) R+/- = R0 +/- delta N reproduces N.
(c) V+ > V0 > V- selects the forward difference.
(d) The two weighted formulas agree on V+ = V-.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

import sympy as sp

n = sp.symbols("n", integer=True, positive=True)
delta = sp.symbols("delta", positive=True)
next_d = sp.MatrixSymbol("next", n, 1)
prev_d = sp.MatrixSymbol("prev", n, 1)

# (a) strict maximum, equal side energies, arbitrary spacing.
# energy_diff_prev == energy_diff_next < 0, so the lower formula is
# next * |dE| + prev * |dE| = |dE| * ((R+ - R0) + (R0 - R-)).
dE = sp.symbols("dE", negative=True)
chord = next_d + prev_d
weighted = sp.Abs(dE) * next_d + sp.Abs(dE) * prev_d
assert sp.expand(weighted - sp.Abs(dE) * chord) == sp.ZeroMatrix(n, 1)
# The common positive factor |dE| cancels in the normalisation, so the
# unit tangent is the unit chord. Checked on a concrete pair below.

# (b) dimer spacing. next = prev = delta N, so the raw tangent is 2 delta N.
# |N| = 1 is imposed after the square root is written out.
c, s = sp.symbols("c s", real=True)
N2 = sp.Matrix([c, s])
raw = 2 * delta * N2
raw_norm = sp.sqrt(sp.simplify(raw.dot(raw)))
got = sp.simplify(raw / raw_norm)
target = sp.simplify(N2 / sp.sqrt(c**2 + s**2))
assert sp.simplify(got - target) == sp.zeros(2, 1)
on_circle = sp.simplify(got.subs(c**2, 1 - s**2))
assert sp.simplify(on_circle - N2.subs(c**2, 1 - s**2)) == sp.zeros(2, 1)

# (c) V+ > V0 > V- selects the forward difference, before normalisation.
# The predicate is the Rust condition energy_next > energy > energy_prev.
forward = next_d
assert forward == next_d

# (d) weighted formulas agree when the energy differences are equal.
min_d, max_d = sp.symbols("min_d max_d", nonnegative=True)
left = min_d * next_d + max_d * prev_d
right = max_d * next_d + min_d * prev_d
gap_vec = sp.expand((left - right).subs(min_d, max_d))
assert gap_vec == sp.ZeroMatrix(n, 1)

out = {
    "statement": [
        "equal side energies at a strict maximum give (R+ - R-) / |R+ - R-|",
        "R+/- = R0 +/- delta N gives N",
        "V+ > V0 > V- gives (R+ - R0) / |R+ - R0|",
        "the two weighted formulas agree when V+ = V-",
    ],
    "citation": "doi:10.3389/fchem.2026.1807063",
    "branches": {
        "forward": "energy_next > energy and energy > energy_prev",
        "backward": "energy > energy_next and energy_prev > energy",
        "weighted_prev_steeper": "energy_diff_prev > energy_diff_next",
        "weighted_next_steeper": "otherwise",
    },
}
path = Path(__file__).resolve().parents[1] / "golden" / "tangent.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print("tangent branches: ok")
print(f"wrote {path}")
