"""Energy-weighted springs.

Asgeirsson, Bjornsson, Jonsson et al.? The design in this crate follows
Asgeirsson and others, J. Chem. Theory Comput. 17, 4929 (2021),
doi:10.1021/acs.jctc.1c00462, as specified for the band:

E_i = max(E[i], E[i+1]), E_ref is the higher band end,
alpha = clamp((E_max - E_i) / (E_max - E_ref), 0, 1),
k_i = (1 - alpha) k_max + alpha k_min when E_i > E_ref, else k_min.

When E_max = E_ref the quotient is undefined. On the physical domain
E_i <= E_max the comparison E_i > E_ref is false, so k_i = k_min without
a division. That is the value the Rust spring must use. A uniform
constant reduces the parallel force to Eq. 6 of
doi:10.3389/fchem.2026.1807063.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

import sympy as sp

e, e_max, e_ref, k_min, k_max = sp.symbols(
    "E_i E_max E_ref k_min k_max", real=True
)
alpha = sp.Min(1, sp.Max(0, (e_max - e) / (e_max - e_ref)))
k_hi = (1 - alpha) * k_max + alpha * k_min


def k_of(energy, assume_span=True):
    if assume_span:
        return sp.Piecewise((k_min, energy <= e_ref), (k_hi, True))
    return k_hi


# Range, for k_min <= k_max and E_ref < E_i < E_max.
assumptions = sp.Q.positive(e_max - e_ref) & sp.Q.nonnegative(k_max - k_min)
# alpha decreases from 1 at E_ref to 0 at E_max, so k increases from k_min to k_max.
alpha_ref = sp.simplify(alpha.subs(e, e_ref))
alpha_max = sp.simplify(alpha.subs(e, e_max))
assert alpha_ref == 1
assert alpha_max == 0
k_at_ref = sp.simplify(k_hi.subs(e, e_ref))
k_at_max = sp.simplify(k_hi.subs(e, e_max))
assert k_at_ref == k_min
assert k_at_max == k_max

# Derivative on the open interval is (k_max - k_min) / (E_max - E_ref) >= 0.
alpha_open = (e_max - e) / (e_max - e_ref)
k_open = (1 - alpha_open) * k_max + alpha_open * k_min
slope = sp.simplify(sp.diff(k_open, e))
assert slope == (k_max - k_min) / (e_max - e_ref)

# Uniform constants: k_min = k_max = k gives k everywhere, and the parallel
# force k (dist_next - dist_prev) tau is Eq. 6.
k, dist_next, dist_prev = sp.symbols("k dist_next dist_prev", real=True)
tau = sp.Matrix(sp.symbols("tau0:3"))
parallel = k * (dist_next - dist_prev) * tau
eq6 = k * (dist_next - dist_prev) * tau
gap = sp.simplify(parallel - eq6)
assert gap.shape == (3, 1)
assert all(entry == 0 for entry in gap)

# Continuous extension. For E_i <= E_ref the piecewise value is already k_min.
# The limit of k_open as E_i -> E_ref from above is k_min.
limit_ref = sp.limit(k_open, e, e_ref)
assert sp.simplify(limit_ref - k_min) == 0
# When E_max = E_ref the domain E_i <= E_max forces E_i <= E_ref, so the
# guard "do not divide; use k_min" matches that limit. The path E_i = E_max
# is a single point there and is the same point.
guard = "Rust must not divide when E_max == E_ref; k_i = k_min"

out = {
    "k_at_E_max": "k_max",
    "k_at_E_ref": "k_min",
    "slope": str(slope),
    "uniform_reduces_to_eq6": True,
    "equal_ends_extension": "k_min",
    "rust_guard": guard,
    "citation": [
        "doi:10.1021/acs.jctc.1c00462",
        "doi:10.3389/fchem.2026.1807063",
    ],
}
path = Path(__file__).resolve().parents[1] / "golden" / "energy_weighted.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print("energy-weighted springs: ok")
print(guard)
print(f"wrote {path}")
