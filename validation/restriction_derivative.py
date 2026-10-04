"""Exact norm derivative, scale invariance, and zero-step right derivative."""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

alpha, sy, dx, dy = sp.symbols("alpha sy dx dy", real=True)
sx, scale = sp.symbols("sx scale", positive=True)
path_norm = sp.sqrt((sx + alpha * dx)**2 + (sy + alpha * dy)**2)
expected = (sx * dx + sy * dy)/sp.sqrt(sx**2 + sy**2)
assert sp.simplify(sp.diff(path_norm, alpha).subs(alpha, 0) - expected) == 0
scaled = expected.subs({sx: sx * scale, sy: sy * scale})
assert sp.simplify(scaled - expected) == 0
derivative_scaled = expected.subs({dx: dx * scale, dy: dy * scale})
assert sp.simplify(derivative_scaled - scale * expected) == 0

t = sp.symbols("t", positive=True)
right_quotient = sp.sqrt((t * dx)**2 + (t * dy)**2)/t
assert sp.simplify(right_quotient - sp.sqrt(dx**2 + dy**2)) == 0

tiny = sp.Rational(1, 10**20)
exact = sp.Integer(5)
floored = (25 * tiny)/sp.Rational(1, 10**12)
assert floored == sp.Rational(1, 4_000_000)
assert floored != exact
print("EXACT_NONZERO_NORM_DERIVATIVE")
print("EXACT_SCALE_INVARIANCE")
print("EXACT_ZERO_STEP_RIGHT_DERIVATIVE")
print("REJECTED_DENOMINATOR_FLOOR")
