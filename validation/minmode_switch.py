"""The switched min-mode force on a quadratic model.

Run: uv run --with sympy python validation/minmode_switch.py

`translated_gradient` in src/minmode.rs hands the translation solver

    g_eff = g - 2 (g.tau) tau    when the curvature along tau is negative,
    g_eff = -(g.tau) tau         when it is not,

the negative of the force F_eff = F - 2 (F.tau) tau, resp. -(F.tau) tau,
with F = -g. Model: E(d) = 1/2 d' H d about a stationary point, H
symmetric, tau = e1 an exact unit eigenvector with eigenvalue lam,
d = a e1 + w with w orthogonal to e1, and H = diag(lam, k2, k3).

Minimum basin (lam > 0, k > 0):
  * the switched force is lam a e1: no perpendicular part, and the true
    energy rises along it at the rate (F.tau)^2 = lam^2 a^2;
  * the reflected force adds -K w, so its energy rate
    lam^2 a^2 - |K w|^2 is negative wherever the perpendicular force
    dominates: the walker slides back down the valley while it climbs;
  * under the switched flow a(t) = a0 exp(lam t) and w stays fixed.
Saddle region (lam < 0):
  * the reflected force -R H d decays every coordinate,
    d/dt |d|^2 = 2 (lam a^2 - w'Kw) < 0, so the walker converges on the
    saddle (formal: minmode_descends_to_saddle); the switch keeps it.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

a, w2, w3, t = sp.symbols("a w2 w3 t", real=True)
lam, k2, k3 = sp.symbols("lam k2 k3", real=True)
a0, w20, w30 = sp.symbols("a0 w20 w30", real=True)

tau = sp.Matrix([1, 0, 0])
d = sp.Matrix([a, w2, w3])
H = sp.diag(lam, k2, k3)
g = H * d
F = -g
F_par = (F.dot(tau)) * tau


def switched(curvature_negative):
    """F_eff of src/minmode.rs::translated_gradient, negated."""
    if curvature_negative:
        return F - 2 * F_par
    return -F_par


def perp(v):
    return v - v.dot(tau) * tau


# Both branches reverse the mode component.
for branch in (True, False):
    assert sp.simplify(switched(branch).dot(tau) + F.dot(tau)) == 0
# The nonnegative branch drops the perpendicular force; the negative one keeps it.
assert perp(switched(False)) == sp.zeros(3, 1)
assert sp.simplify(perp(switched(True)) - perp(F)) == sp.zeros(3, 1)
print("branches: mode component reversed; perpendicular dropped (curvature >= 0) or kept: ok")

# Energy rate dE/dt = g . F_eff along each branch.
rate_switched = sp.expand(g.dot(switched(False)))
rate_reflected = sp.expand(g.dot(switched(True)))
assert sp.simplify(rate_switched - lam**2 * a**2) == 0
assert sp.simplify(rate_reflected - (lam**2 * a**2 - k2**2 * w2**2 - k3**2 * w3**2)) == 0
# The general identity behind it: g . (-(F.tau) tau) = (F.tau)^2 >= 0.
assert sp.simplify(rate_switched - F.dot(tau) ** 2) == 0
print("energy rate: switched (F.tau)^2 = lam^2 a^2 >= 0; reflected lam^2 a^2 - |K w|^2: ok")

# Minimum basin: a concrete point where the reflected force descends.
basin = {lam: 1, k2: 4, k3: 9, a: sp.Rational(1, 10), w2: sp.Rational(1, 5), w3: 0}
assert rate_reflected.subs(basin) < 0 < rate_switched.subs(basin)
print(f"basin lam=1, k=(4,9), d=(0.1,0.2,0): reflected rate {rate_reflected.subs(basin)}, "
      f"switched rate {rate_switched.subs(basin)}: ok")

# Switched flow from the basin: d' = F_eff solves to a0 exp(lam t), w fixed.
sol_switched = sp.Matrix([a0 * sp.exp(lam * t), w20, w30])
rhs = switched(False).subs({a: sol_switched[0], w2: w20, w3: w30})
assert sp.simplify(sol_switched.diff(t) - rhs) == sp.zeros(3, 1)
assert sol_switched.subs(t, 0) == sp.Matrix([a0, w20, w30])
# Reflected flow from the same point: a grows the same way, w relaxes to the valley floor.
sol_reflected = sp.Matrix([a0 * sp.exp(lam * t), w20 * sp.exp(-k2 * t), w30 * sp.exp(-k3 * t)])
rhs = switched(True).subs({a: sol_reflected[0], w2: sol_reflected[1], w3: sol_reflected[2]})
assert sp.simplify(sol_reflected.diff(t) - rhs) == sp.zeros(3, 1)
print("basin flows: switched a0 e^(lam t) with w fixed; reflected relaxes w to 0: ok")

# Saddle region: lam < 0, k > 0. The reflected flow contracts |d|^2.
L, K2, K3 = sp.symbols("L K2 K3", positive=True)
saddle = {lam: -L, k2: K2, k3: K3}
contract = sp.expand(2 * d.dot(switched(True)).subs(saddle))
assert sp.simplify(contract - (-2 * L * a**2 - 2 * K2 * w2**2 - 2 * K3 * w3**2)) == 0
sol_saddle = sp.Matrix([a0 * sp.exp(-L * t), w20 * sp.exp(-K2 * t), w30 * sp.exp(-K3 * t)])
rhs = switched(True).subs(saddle).subs({a: sol_saddle[0], w2: sol_saddle[1], w3: sol_saddle[2]})
assert sp.simplify(sol_saddle.diff(t) - rhs) == sp.zeros(3, 1)
assert all(sp.limit(c, t, sp.oo) == 0 for c in sol_saddle)
print("saddle region: reflected flow decays every coordinate, d|d|^2/dt < 0: ok")

# Fixed-step map with step alpha (the translation solver's first step):
# switched in the basin keeps w exactly, a grows by (1 + alpha lam).
alpha = sp.symbols("alpha", positive=True)
step = d + alpha * switched(False)
assert sp.simplify(step - sp.Matrix([(1 + alpha * lam) * a, w2, w3])) == sp.zeros(3, 1)
print("fixed-step map: switched step leaves w unchanged, a -> (1 + alpha lam) a: ok")
