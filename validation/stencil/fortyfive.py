"""Frozen-axis stability of the reflected Hessian.

Goswami, Gunde, Jonsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Sec. 2.3.1.
In two dimensions, H = diag(-a, b) with a, b > 0 and unit v = (cos theta, sin theta),
det(T(v) H) = a b and tr(T(v) H) = (a + b) cos 2 theta.
The eigenvalues have positive real part iff |v_1| > 1/sqrt(2).

For n = 3..6 a multistart search minimises the smallest real part of the
spectrum of T(v) H subject to |v · e_min| > 1/sqrt(2). A negative value is a
counterexample to the same threshold. The search reports the best candidate.
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution

a, b, theta = sp.symbols("a b theta", real=True)
c = sp.cos(theta)
s = sp.sin(theta)
T = sp.Matrix([[1 - 2 * c**2, -2 * c * s], [-2 * c * s, 1 - 2 * s**2]])
H = sp.Matrix([[-a, 0], [0, b]])
M = sp.simplify(T * H)
tr = sp.simplify(M.trace())
det = sp.simplify(M.det())
assert tr == (a + b) * sp.cos(2 * theta)
assert det == a * b
# Positive real parts iff trace > 0 and det > 0 (Routh-Hurwitz, 2 by 2).
# det = a b > 0 when a, b > 0. trace > 0 iff cos 2 theta > 0 iff |cos theta| > 1/sqrt(2).
cos2 = sp.cos(2 * theta)
assert sp.simplify(cos2 - (2 * c**2 - 1)) == 0

INV_SQRT2 = 1.0 / np.sqrt(2.0)


def pack_v(raw, n):
    """Unit vector with |v_0| strictly above 1/sqrt(2)."""
    # raw[0] in (0, 1) maps to |v_0| in (1/sqrt(2), 1].
    v0 = np.sqrt(0.5 + 0.5 * (1.0 / (1.0 + np.exp(-raw[0]))))
    rest = np.zeros(n - 1)
    if n > 1:
        angles = raw[1:n]
        # hypersphere for the remaining mass
        mass = np.sqrt(max(0.0, 1.0 - v0 * v0))
        if n == 2:
            rest[0] = mass
        else:
            coords = np.ones(n - 1)
            phi = np.arctan(angles)
            s = 1.0
            for i, p in enumerate(phi[:-1] if n > 3 else phi):
                coords[i] = s * np.cos(p) if n > 3 else s
                s *= np.sin(p) if n > 3 else 1.0
            if n == 3:
                coords[0] = np.cos(phi[0])
                coords[1] = np.sin(phi[0])
            else:
                coords[-1] = s
            rest = mass * coords
    v = np.concatenate([[v0], rest])
    v /= np.linalg.norm(v)
    return v


def min_real_part(x, n):
    log_a = x[0]
    log_b = x[1:n]
    a_eig = np.exp(np.clip(log_a, -8, 4))
    b_eig = np.exp(np.clip(log_b, -8, 4))
    lam = np.concatenate([[-a_eig], b_eig])
    v = pack_v(x[n:], n)
    # Numerical guard: reject vectors that fell back onto the threshold.
    if abs(v[0]) <= INV_SQRT2 + 1e-6:
        return 0.0
    Tm = np.eye(n) - 2.0 * np.outer(v, v)
    M = Tm @ np.diag(lam)
    ev = np.linalg.eigvals(M)
    return float(np.min(np.real(ev)))


def search(n, seed):
    # variables: log a, log b_1..b_{n-1}, v raw (n,)
    dim = n + n
    bounds = [(-4.0, 3.0)] * n + [(-3.0, 3.0)] * n
    result = differential_evolution(
        lambda x: min_real_part(x, n),
        bounds,
        seed=seed,
        popsize=8,
        mutation=0.5,
        recombination=0.7,
        atol=1e-8,
        workers=1,
        updating="immediate",
        polish=True,
    )
    x = result.x
    value = float(result.fun)
    a_eig = float(np.exp(np.clip(x[0], -8, 4)))
    b_eig = [float(np.exp(np.clip(t, -8, 4))) for t in x[1:n]]
    v = pack_v(x[n:], n)
    lam = np.concatenate([[-a_eig], b_eig])
    Tm = np.eye(n) - 2.0 * np.outer(v, v)
    ev = np.linalg.eigvals(Tm @ np.diag(lam))
    return {
        "n": n,
        "seed": seed,
        "min_real_part": value,
        "eigenvalues_real": [float(z) for z in np.real(ev)],
        "eigenvalues_imag": [float(z) for z in np.imag(ev)],
        "spectrum": [-a_eig, *b_eig],
        "v": [float(t) for t in v],
        "abs_v_dot_emin": float(abs(v[0])),
        "threshold": INV_SQRT2,
        "counterexample": bool(value < -1e-6 and abs(v[0]) > INV_SQRT2 + 1e-4),
    }


candidates = [search(n, seed=1000 + n) for n in range(3, 7)]
any_counter = any(c["counterexample"] for c in candidates)
print("2-D det/trace: ok")
print("counterexample search, best candidate per dimension:")
for c in candidates:
    print(json.dumps(c))
print("counterexample_found", any_counter)

out = {
    "two_d": {
        "det": "a*b",
        "trace": "(a+b) cos 2 theta",
        "positive_real_part_iff": "|v_1| > 1/sqrt(2)",
    },
    "search": candidates,
    "counterexample_found": any_counter,
    "n_d_status": "open" if not any_counter else "counterexample",
    "citation": "doi:10.3389/fchem.2026.1807063",
}
path = Path(__file__).resolve().parents[1] / "golden" / "fortyfive.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {path}")
