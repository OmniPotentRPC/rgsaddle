"""Why rotate_dimer measures instead of extrapolating far past its trial angle.

Run: uv run --no-project --with sympy python validation/dimer_extrapolation.py

With a forward-difference action h(v) = (g(x + dr v) - g(x)) / dr the
dimer sees h(v) = H v + dr/2 T[v, v] + O(dr^2), T the third derivative.
Its Rayleigh quotient C(phi) = n(phi) . h(n(phi)) along
n(phi) = n cos(phi) + theta sin(phi) then has the slope

    C'(0) = 2 theta.H n + 3 dr/2 T[n, n, theta],

while the fit in src/minmode.rs::rotate_dimer takes C'(0) = 2 h.theta
= 2 theta.H n + dr T[n, n, theta]. The mismatch delta = dr/2 T[n,n,theta]
enters the fitted amplitude as

    a1 = (C(0) - C(phi1) + b1 sin(2 phi1)) / (1 - cos(2 phi1)) ~ a1_true - delta / (2 phi1),

so a trial angle phi1 clamped near 1e-4 rad turns delta ~ 1e-2 into an
amplitude error ~ 50 and a fitted step to a far angle. The actions
extrapolated to that angle carry the error multiplied by
|sin(phi1 - phi)| / sin(phi1). On V = (x^2 - 1)^2 + 5 (y^2 + z^2) + x y / 2
at (0.85, 0.05, -0.04), dr = 1e-3, the extrapolated dimer reported a
curvature of -14.07 where the Hessian spectrum is [4.62, 10.05]
(tests/minmode_switch.rs). With the trial angle on its floor
MIN_TRIAL_ANGLE = 1e-4 and an extrapolation weight past
MAX_EXTRAPOLATION = 8 the rotated dimer is measured, and a measurement
above the better of the two measured dimers ends the rotation on that
dimer.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

phi, dr, phi1 = sp.symbols("phi dr phi1", positive=True)
# Two-dimensional slice: n = e1, theta = e2; H and a cubic surface term.
h11, h12, h22 = sp.symbols("h11 h12 h22", real=True)
t111, t112, t122, t222 = sp.symbols("t111 t112 t122 t222", real=True)
x, y = sp.symbols("x y", real=True)
E = (h11 * x**2 / 2 + h12 * x * y + h22 * y**2 / 2
     + t111 * x**3 / 6 + t112 * x**2 * y / 2 + t122 * x * y**2 / 2 + t222 * y**3 / 6)
grad = sp.Matrix([sp.diff(E, x), sp.diff(E, y)])


def action(v):
    shifted = grad.subs({x: dr * v[0], y: dr * v[1]}, simultaneous=True)
    return (shifted - grad.subs({x: 0, y: 0})) / dr


n = sp.Matrix([sp.cos(phi), sp.sin(phi)])
C = sp.expand((n.T * action(n))[0])
slope = sp.diff(C, phi).subs(phi, 0)
theta = sp.Matrix([0, 1])
b1 = (action(sp.Matrix([1, 0])).T * theta)[0]
mismatch = sp.simplify(slope - 2 * b1)
# T[n, n, theta] with n = e1, theta = e2 is t112.
assert sp.simplify(mismatch - dr * t112 / 2) == 0
print("forward-difference slope mismatch C'(0) - 2 h.theta = dr/2 T[n,n,theta]: ok")

# Amplitude error of the three-point fit as phi1 -> 0.
C0 = C.subs(phi, 0)
C1 = C.subs(phi, phi1)
a1_fit = (C0 - C1 + b1 * sp.sin(2 * phi1)) / (1 - sp.cos(2 * phi1))
# Exact quadratic part of the model: a1 = (h11 - h22)/2 at dr = 0.
lead = sp.series(sp.simplify(a1_fit - (h11 - h22) / 2), phi1, 0, 1).removeO()
assert sp.simplify(lead.coeff(phi1, -1) + dr * t112 / 4) == 0
print("fitted a1 error ~ -delta / (2 phi1) with delta = dr/2 T[n,n,theta]: ok")

# Extrapolation weights of the measured actions to angle phi.
p = sp.symbols("p", real=True)
w_old = sp.sin(phi1 - p) / sp.sin(phi1)
w_trial = sp.sin(p) / sp.sin(phi1)
for value in (sp.Rational(1, 10**4),):
    weight = sp.N(sp.Abs(w_old.subs({phi1: value, p: sp.pi / 2})), 6)
    assert weight > 9999
print("weights at phi1 = 1e-4, phi = pi/2: ~1e4, past MAX_EXTRAPOLATION = 8: ok")
