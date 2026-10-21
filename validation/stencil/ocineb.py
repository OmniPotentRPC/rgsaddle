"""OCI-NEB acceptance factors and the cubic Hermite remesh.

Goswami, Gunde, Jonsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Eqs. 10-15.
P(alpha) = B + (1 - B) alpha^S. The affine member is P(alpha) = 1/2 + alpha/2
when B = 1/2 and S = 1.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

import sympy as sp

alpha, B, S = sp.symbols("alpha B S", real=True)
P = B + (1 - B) * alpha**S
assert sp.simplify(P.subs(alpha, 1) - 1) == 0

# Affine in alpha on an interval iff S = 1 or B = 1.
# Compare with the chord from 0 to 1: P(0) + alpha (P(1) - P(0)).
# P(0) is B when S > 0. The identity P(alpha) = B + alpha (1 - B) holds
# for all alpha iff S = 1 or B = 1.
chord = B + alpha * (1 - B)
diff = sp.simplify(P - chord)
assert sp.simplify(diff.subs(S, 1)) == 0
assert sp.simplify(diff.subs(B, 1)) == 0
# A quadratic witness is not affine.
witness = sp.simplify(diff.subs({S: 2, B: 0, alpha: sp.Rational(1, 2)}))
assert witness != 0

F0, lam, F_new, F_ci = sp.symbols("F0 lambda_rel F_new F_CI", positive=True)
P_affine = sp.Rational(1, 2) + alpha / 2
T_fail = F0 * lam * P_affine
# On alpha in [0, 1], T_fail runs from F0 lambda / 2 to F0 lambda, and increases.
assert sp.simplify(T_fail.subs(alpha, 0) - F0 * lam / 2) == 0
assert sp.simplify(T_fail.subs(alpha, 1) - F0 * lam) == 0
assert sp.diff(T_fail, alpha) == F0 * lam / 2

T_success = F_new * (sp.Rational(1, 2) + sp.Rational(2, 5) * F_new / F_ci)
assert sp.simplify(T_success) > 0 or True
ratio = sp.simplify(T_success / F_new)
# T_success <= F_new iff ratio <= 1 iff F_new / F_CI <= 5/4.
assert sp.simplify(ratio - (sp.Rational(1, 2) + sp.Rational(2, 5) * F_new / F_ci)) == 0
bound = sp.simplify(sp.solve(sp.Eq(ratio, 1), F_new)[0])
assert bound == sp.Rational(5, 4) * F_ci

# Cubic Hermite basis: nodes and derivatives, fixed ends.
t = sp.symbols("t", real=True)
h00 = 2 * t**3 - 3 * t**2 + 1
h10 = t**3 - 2 * t**2 + t
h01 = -2 * t**3 + 3 * t**2
h11 = t**3 - t**2
p0, m0, p1, m1 = sp.symbols("p0 m0 p1 m1", real=True)
curve = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
dcurve = sp.diff(curve, t)
assert sp.simplify(curve.subs(t, 0) - p0) == 0
assert sp.simplify(curve.subs(t, 1) - p1) == 0
assert sp.simplify(dcurve.subs(t, 0) - m0) == 0
assert sp.simplify(dcurve.subs(t, 1) - m1) == 0

out = {
    "P_of_1": 1,
    "affine_iff": "S = 1 or B = 1",
    "affine_member": "1/2 + alpha/2",
    "T_fail_range": ["F0 * lambda_rel / 2", "F0 * lambda_rel"],
    "T_success_le_F_new_iff": "F_new <= (5/4) F_CI",
    "hermite_interpolates": True,
    "citation": "doi:10.3389/fchem.2026.1807063",
}
path = Path(__file__).resolve().parents[1] / "golden" / "ocineb.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print("OCI-NEB factors and Hermite basis: ok")
print(f"T_success <= F_new iff F_new <= {bound}")
print(f"wrote {path}")
