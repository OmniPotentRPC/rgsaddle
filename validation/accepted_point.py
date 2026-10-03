"""Exact polynomial witness for curvature at the accepted point."""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as s

a, x, y, z, h = s.symbols("a x y z h", real=True)
energy = a*x**3/3 - x**2/2 + (1-a)*x + y*y + z*z
gradient = s.Matrix([s.diff(energy, q) for q in (x, y, z)])
hessian = s.hessian(energy, (x, y, z))
mode = s.Matrix([1, 0, 0])
opening = {x: -1, y: 0, z: 0}
trial = s.Matrix([-1, 0, 0]) - (s.eye(3) - 2*mode*mode.T)*gradient.subs(opening)
assert trial == s.Matrix([1, 0, 0])
assert gradient.subs({x: 1, y: 0, z: 0}) == s.zeros(3, 1)
forward = s.cancel((gradient[0].subs(x, x+h)-gradient[0])/h)
assert s.simplify(forward - (2*a*x - 1 + a*h)) == 0
for coefficient, start, accepted in [(1, -3, 1), (-1, 1, -3)]:
    assert hessian[0, 0].subs({a: coefficient, x: -1}) == start
    assert hessian[0, 0].subs({a: coefficient, x: 1}) == accepted
    estimate = forward.subs({a: coefficient, x: 1, h: s.Rational(1, 1000)})
    assert s.sign(estimate) == s.sign(accepted)
    print(f"CERTIFIED_ACCEPTED_CURVATURE {coefficient} {estimate}")
# Along this diagonal cubic, one action completes either rotation method.
# The callbacks are centre, opening probe, accepted centre, accepted probe.
print("EXPECTED_CALLBACKS 4")
