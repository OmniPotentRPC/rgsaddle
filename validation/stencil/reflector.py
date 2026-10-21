"""Householder reflection T(v) = I - 2 v v^T.

Goswami, Gunde, Jonsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Eqs. 3 and 9.
The climbing image with a zero DNEB term is T(tau) F.
For negative curvature, -translated_gradient(g, N) = T(N) (-g).
The ascend branch -(F · v) v equals T(v) F exactly when F is parallel to v.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

import sympy as sp

# Concrete 3-vector with |v| = 1 carries the algebraic identities that
# need a numeric dimension (determinant, spectrum, orthogonality).
c, s, t = sp.symbols("c s t", real=True)
vv = sp.Matrix([c, s, t])
TT = sp.eye(3) - 2 * vv * vv.T
q = c**2 + s**2 + t**2
unit = {q: 1}


def sphere(expr):
    """Reduce an identity that holds once |v| = 1."""
    return sp.simplify(sp.simplify(sp.expand(expr)).subs(unit))


assert sphere(TT - TT.T) == sp.zeros(3)
assert sphere(TT.T * TT - sp.eye(3)) == sp.zeros(3)
assert sphere(TT * TT - sp.eye(3)) == sp.zeros(3)
det = sp.simplify(TT.det())
assert sp.expand(det - (1 - 2 * q)) == 0
assert sp.simplify(det.subs(c**2, 1 - s**2 - t**2)) == -1
# Spectrum of a Householder reflection: -1 along v and +1 on the plane.
lam = sp.symbols("lam")
char = sp.simplify(TT.charpoly(lam).as_expr())
target = (lam - 1) ** 2 * (lam + 2 * q - 1)
assert sp.expand(char - target) == 0
on_sphere = sp.factor(sp.simplify(char.subs(c**2, 1 - s**2 - t**2)))
assert on_sphere == (lam - 1) ** 2 * (lam + 1)

FF = sp.Matrix(sp.symbols("F0:3"))
climb = FF - 2 * (FF.dot(vv)) * vv
assert sphere(TT * FF - climb) == sp.zeros(3, 1)

# -translated_gradient for C < 0, with force F = -g, is T(v) F.
g = -FF
translated = g - 2 * (g.dot(vv)) * vv
assert sphere(-translated - TT * FF) == sp.zeros(3, 1)

# T(v) F = 0 iff F = 0 when |v| = 1.
# If T F = 0 then F = 2 (F·v) v, so F·v = 2 (F·v), so F·v = 0, so F = 0.
a = sp.symbols("a", real=True)
parallel = a * vv
assert sphere(TT * parallel + parallel) == sp.zeros(3, 1)
perp = sp.Matrix([s, -c, 0])
assert sp.simplify(perp.dot(vv)) == 0
assert sphere(TT * perp - perp) == sp.zeros(3, 1)

# Ascend branch -(F·v) v equals T F iff the perpendicular component vanishes.
ascend = lambda M: -(M.dot(vv)) * vv
diff_parallel = sphere(ascend(parallel) - TT * parallel)
assert diff_parallel == sp.zeros(3, 1)
diff_perp = sphere(ascend(perp) - TT * perp)
assert diff_perp == -perp
# They agree on a general vector iff that vector is parallel to v.
parallel_gen = sp.Matrix([a * c, a * s, a * t])
assert sphere(ascend(parallel_gen) - TT * parallel_gen) == sp.zeros(3, 1)
# A perpendicular witness is not parallel, and the residual is that witness.
assert diff_perp != sp.zeros(3, 1)

out = {
    "symmetric": True,
    "orthogonal": True,
    "involutive": True,
    "det": -1,
    "spectrum": [-1, 1, 1],
    "climbing_image_force_zero_dneb": "T(tau) F",
    "neg_translated_gradient_negative_curvature": "T(N) (-g)",
    "reflect_zero_iff_zero": True,
    "ascend_equals_reflect_iff": "F is parallel to v",
    "citation": "doi:10.3389/fchem.2026.1807063",
}
path = Path(__file__).resolve().parents[1] / "golden" / "reflector.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print("reflector: ok")
print(f"wrote {path}")
