"""Symbolic checks for the modified Newton dimer rotation in src/minmode.rs.

Run: uv run --with sympy python validation/dimer_rotation.py

On a quadratic surface g(x) = g0 + H (x - x0), the dimer gradient at
x0 + dr n(phi), n(phi) = n cos(phi) + theta sin(phi), and the curvature
C(phi) = (g(phi) - g0) . n(phi) / dr = n(phi)^T H n(phi) satisfy:

1. C(phi) = a0/2 + a1 cos(2 phi) + b1 sin(2 phi), b1 = theta^T H n.
2. a1 from C(0), C'(0) = 2 b1 and C(phi1):
   a1 = (C(0) - C(phi1) + b1 sin(2 phi1)) / (1 - cos(2 phi1)).
3. The minimum sits at phi* = atan(b1 / a1) / 2 or that plus pi/2.
4. g(phi) = g0 + sin(phi1 - phi)/sin(phi1) (g1 - g0)
          + sin(phi)/sin(phi1) (g(phi1) - g0)          (exact),
   and the g0 weight 1 - w_old - w_trial equals Heyden's
   1 - cos(phi) - sin(phi) tan(phi1/2).
"""

import sympy as sp

phi, phi1, dr = sp.symbols("phi phi1 dr", real=True)
# A generic symmetric 3x3 Hessian and an orthonormal pair (n, theta).
h = sp.symbols("h0:6", real=True)
H = sp.Matrix([[h[0], h[1], h[2]], [h[1], h[3], h[4]], [h[2], h[4], h[5]]])
n = sp.Matrix([1, 0, 0])
theta = sp.Matrix([0, 1, 0])
g0 = sp.Matrix(sp.symbols("g0:3", real=True))


def n_of(p):
    return n * sp.cos(p) + theta * sp.sin(p)


def g_of(p):
    return g0 + dr * H * n_of(p)


def curvature(p):
    return ((g_of(p) - g0).T * n_of(p))[0] / dr


C = sp.simplify(curvature(phi))
b1 = (theta.T * H * n)[0]
C0 = sp.simplify(curvature(0))
a1_true = sp.simplify((n.T * H * n)[0] - (theta.T * H * theta)[0]) / 2
a0_true = sp.simplify((n.T * H * n)[0] + (theta.T * H * theta)[0])
fourier = a0_true / 2 + a1_true * sp.cos(2 * phi) + b1 * sp.sin(2 * phi)
assert sp.simplify(sp.expand_trig(C - fourier)) == 0, "1: Fourier form"

dC0 = sp.diff(C, phi).subs(phi, 0)
assert sp.simplify(dC0 - 2 * b1) == 0, "1: C'(0) = 2 b1"

a1_fit = (C0 - curvature(phi1) + b1 * sp.sin(2 * phi1)) / (1 - sp.cos(2 * phi1))
assert sp.simplify(sp.expand_trig(a1_fit - a1_true)) == 0, "2: a1 from one trial"
assert sp.simplify(2 * (C0 - a1_true) - a0_true) == 0, "2: a0 = 2 (C0 - a1)"

phi_star = sp.atan(b1 / a1_true) / 2
stationary = sp.diff(fourier, phi).subs(phi, phi_star)
assert sp.simplify(sp.expand_trig(stationary)) == 0, "3: stationary angle"

w_old = sp.sin(phi1 - phi) / sp.sin(phi1)
w_trial = sp.sin(phi) / sp.sin(phi1)
g_extrap = g0 + w_old * (g_of(0) - g0) + w_trial * (g_of(phi1) - g0)
diff = sp.simplify(sp.expand_trig(g_extrap - g_of(phi)))
assert diff == sp.zeros(3, 1), "4: extrapolated gradient"
# tan(phi1/2) written as (1 - cos phi1) / sin phi1, the half-angle form.
heyden = 1 - sp.cos(phi) - sp.sin(phi) * (1 - sp.cos(phi1)) / sp.sin(phi1)
assert sp.simplify(sp.expand_trig((1 - w_old - w_trial) - heyden)) == 0, "4: Heyden weight"

# Numeric spot check of the closed-form step on a random Hessian.
vals = {h[0]: 2.0, h[1]: -1.3, h[2]: 0.4, h[3]: -0.7, h[4]: 0.9, h[5]: 3.1}
f = sp.lambdify(phi, fourier.subs(vals))
p_star = float(phi_star.subs(vals))
cands = [p_star, p_star + 1.5707963267948966]
best = min(cands, key=f)
grid = min(float(f(k * 3.14159265358979 / 2000)) for k in range(-1000, 1000))
assert abs(f(best) - grid) < 1e-5, (f(best), grid)
print("dimer rotation identities: ok")
print(f"  spot check: C(phi*) = {float(f(best)):.9f}, grid minimum {grid:.9f}")
