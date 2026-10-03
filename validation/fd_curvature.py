"""Error terms of the finite-difference curvature in src/minmode.rs.

Run: uv run --with sympy python validation/fd_curvature.py

The dimer and the Lanczos actions estimate H v from gradients. With
g(x + d v) = g + d H v + d^2/2 T[v,v] + d^3/6 Q[v,v,v] + O(d^4):

  forward  C_f(d) = v.(g(x + d v) - g(x)) / d
                  = v.H.v + d/2 T[v,v,v] + d^2/6 Q[v,v,v,v] + O(d^3)
  central  C_c(d) = v.(g(x + d v) - g(x - d v)) / (2 d)
                  = v.H.v + d^2/6 Q[v,v,v,v] + O(d^4)

A gradient carrying absolute noise eps per evaluation adds at most
2 eps / d (forward) or eps / d (central). Minimizing the sum:

  forward  d* = 2 sqrt(eps / |T|),        E* = 2 sqrt(eps |T|)
  central  d* = (3 eps / |Q|)^(1/3),      E* = 3^(2/3) |Q|^(1/3) eps^(2/3) / 2

The forward form costs one gradient per action (the centre is known),
the central form two. rgsaddle uses the forward form; the table below
gives d* for a Morse Pt pair (eOn morse_pt constants) at the noise of
an analytic potential and of an SCF, so a host picks `dr` by its noise.
fd_curvature.sollya checks the same optima and the floating-point cost
of forming x + d v.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

d, eps = sp.symbols("d epsilon", positive=True)
T, Q, H = sp.symbols("T Q H", real=True)
s = sp.symbols("s", real=True)

# One-dimensional model along the unit direction v: f(s) = g.v at x + s v.
g_line = H * s + T * s**2 / 2 + Q * s**3 / 6
forward = sp.series((g_line.subs(s, d) - g_line.subs(s, 0)) / d, d, 0, 3).removeO()
central = sp.series((g_line.subs(s, d) - g_line.subs(s, -d)) / (2 * d), d, 0, 4).removeO()
assert sp.simplify(forward - (H + d * T / 2 + d**2 * Q / 6)) == 0
assert sp.simplify(central - (H + d**2 * Q / 6)) == 0

Ta, Qa = sp.symbols("T_abs Q_abs", positive=True)
E_f = d * Ta / 2 + 2 * eps / d
E_c = d**2 * Qa / 6 + eps / d
d_f = sp.solve(sp.diff(E_f, d), d)[0]
d_c = sp.solve(sp.diff(E_c, d), d)[0]
assert sp.simplify(d_f - 2 * sp.sqrt(eps / Ta)) == 0
assert sp.simplify(d_c - (3 * eps / Qa) ** sp.Rational(1, 3)) == 0
E_f_star = sp.simplify(E_f.subs(d, d_f))
E_c_star = sp.simplify(E_c.subs(d, d_c))
assert sp.simplify(E_f_star - 2 * sp.sqrt(eps * Ta)) == 0
print("series and optima: ok")
print(f"  forward: d* = {d_f}, E* = {E_f_star}")
print(f"  central: d* = {d_c}, E* = {E_c_star}")

# Morse V(r) = D (exp(-2 a (r - r0)) - 2 exp(-a (r - r0))), eOn morse_pt.
r, D, a, r0 = sp.symbols("r D a r0", positive=True)
V = D * (sp.exp(-2 * a * (r - r0)) - 2 * sp.exp(-a * (r - r0)))
pt = {D: sp.Float("0.7102"), a: sp.Float("1.6047"), r0: sp.Float("2.8970")}
V2 = sp.diff(V, r, 2).subs(r, r0).subs(pt)
V3 = sp.diff(V, r, 3).subs(r, r0).subs(pt)
V4 = sp.diff(V, r, 4).subs(r, r0).subs(pt)
assert sp.simplify(sp.diff(V, r, 3).subs(r, r0) + 6 * D * a**3) == 0
print(f"Morse Pt at r0: V'' = {float(V2):.4f} eV/A^2, V''' = {float(V3):.4f} eV/A^3, V'''' = {float(V4):.4f} eV/A^4")

print(f"{'eps (eV/A)':>12} {'d* fwd (A)':>12} {'E* fwd':>10} {'E(1e-3) fwd':>12} {'d* ctr (A)':>12} {'E* ctr':>10}")
for e in [1e-14, 1e-10, 1e-6, 1e-4, 1e-3]:
    sub = {eps: e, Ta: abs(float(V3)), Qa: abs(float(V4))}
    print(
        f"{e:12.0e} {float(d_f.subs(sub)):12.3e} {float(E_f_star.subs(sub)):10.2e}"
        f" {float(E_f.subs(sub).subs(d, 1e-3)):12.2e}"
        f" {float(d_c.subs(sub)):12.3e} {float(E_c_star.subs(sub)):10.2e}"
    )
