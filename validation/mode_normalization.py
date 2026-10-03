"""Exact normalization and reflection identities for host mode seeds.

Run: uv run --with sympy python validation/mode_normalization.py
The square norm is positive. Floating-point bounds have their own check
in mode_normalization.sollya; Rust and C tests exercise the entry points.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

q = sp.symbols("q", positive=True)
a, b, c = sp.symbols("a b c", real=True)
v = sp.Matrix([a, b, c])
unit = v / sp.sqrt(q)
unit_square = sp.expand(unit.dot(unit)).subs(c**2, q - a**2 - b**2)
assert sp.simplify(unit_square - 1) == 0

f = sp.Matrix(sp.symbols("f0:3", real=True))
reflected = f - 2 * unit.dot(f) * unit
norm_change = sp.expand(reflected.dot(reflected) - f.dot(f))
assert sp.rem(sp.together(norm_change * q**2), a**2 + b**2 + c**2 - q, c) == 0

# A nonunit seed changes the reflected force norm even in one dimension.
s, force = sp.symbols("s force", real=True)
wrong = force - 2 * s**2 * force
assert sp.expand(wrong**2 - force**2 - 4 * s**2 * (s**2 - 1) * force**2) == 0
assert (wrong**2 - force**2).subs({s: sp.Rational(1, 2), force: 1}) == -sp.Rational(3, 4)
print("mode normalization and reflection identities: ok")
