"""How far the inverted-force operator moves when the mode turns.

Run: uv run --with sympy python validation/mode_reset.py

The min-mode translation steps on R g with R = I - 2 tau tau'. When the
mode turns by theta inside the plane of tau and a unit u orthogonal to
it, tau' = tau cos(theta) + u sin(theta), and

    R' - R = 2 (tau tau' - tau' tau'')   has spectral norm 2 sin(theta),

so the effective Hessian R H R moves by at most 4 sin(theta) |H|. This
is the basis of MODE_RESET_ANGLE in src/minmode.rs: at eOn's converged
angle, 5 degrees, the bound is 0.349 |H|.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

th = sp.symbols("theta", real=True)
tau = sp.Matrix([1, 0])
tau2 = sp.Matrix([sp.cos(th), sp.sin(th)])
R = sp.eye(2) - 2 * tau * tau.T
R2 = sp.eye(2) - 2 * tau2 * tau2.T
D = sp.simplify(R2 - R)
eig = [sp.simplify(e) for e in (D.T * D).eigenvals()]
assert all(sp.simplify(e - 4 * sp.sin(th) ** 2) == 0 for e in eig), eig
# Orthogonal complement of the plane is untouched, so the 2x2 block is the norm.
print("|R' - R|_2 = 2 |sin(theta)|: ok")
bound = 4 * sp.sin(sp.pi * 5 / 180)
print(f"4 sin(5 deg) = {float(bound):.4f}")
assert abs(float(bound) - 0.3486) < 1e-3
