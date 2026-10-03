"""The Ritz residual bound behind the Lanczos exit in src/minmode.rs.

Run: uv run --with sympy python validation/lanczos_residual.py

After j + 1 Lanczos steps, A Q_j = Q_j T_j + beta_j q_{j+1} e_j^T. For a
Ritz pair (theta, s) of T_j and y = Q_j s,

    A y - theta y = beta_j q_{j+1} (e_j . s),   so   |A y - theta y| = beta_j |s_j|.

The rotational force of the dimer is the same quantity, |H v - (v.H v) v|
for unit v, which is why rgsaddle stops Lanczos on beta_j |s_j| <=
rotation_tol. This script checks the tridiagonal Ritz pairs
symbolically and the identity numerically, at 50 digits, on a random
symmetric matrix through every Krylov dimension.
"""

import random

import mpmath as mp
import sympy as sp

mp.mp.dps = 50

# Symbolic: a 2-step relation with generic entries.
a0, a1, b0, b1 = sp.symbols("a0 a1 b0 b1", real=True)
T = sp.Matrix([[a0, b0], [b0, a1]])
lam = sp.symbols("lam")
for theta in sp.solve((T - lam * sp.eye(2)).det(), lam):
    s = (T - theta * sp.eye(2)).nullspace()[0]
    s = s / sp.sqrt((s.T * s)[0])
    # Residual of T s - theta s vanishes; the Lanczos residual is the
    # coupling to the next vector, b1 * s[1].
    assert sp.simplify(((T - theta * sp.eye(2)) * s).norm()) == 0
print("tridiagonal Ritz pairs: ok")

random.seed(7)
n = 8
A = mp.matrix(n, n)
for i in range(n):
    for j in range(i, n):
        v = mp.mpf(random.uniform(-1, 1))
        A[i, j] = v
        A[j, i] = v
q = [mp.matrix([mp.mpf(random.uniform(-1, 1)) for _ in range(n)])]
q[0] = q[0] / mp.norm(q[0])
alpha, beta = [], []
worst = mp.mpf(0)
for j in range(n - 1):
    w = A * q[j]
    a = (q[j].T * w)[0]
    alpha.append(a)
    w = w - a * q[j]
    if j > 0:
        w = w - beta[j - 1] * q[j - 1]
    for qi in q:
        w = w - (qi.T * w)[0] * qi
    b = mp.norm(w)
    k = len(alpha)
    Tm = mp.matrix(k, k)
    for i in range(k):
        Tm[i, i] = alpha[i]
        if i + 1 < k:
            Tm[i, i + 1] = beta[i]
            Tm[i + 1, i] = beta[i]
    evals, evecs = mp.eigsy(Tm)
    low = min(range(k), key=lambda i: evals[i])
    s = evecs[:, low]
    y = mp.matrix(n, 1)
    for i in range(k):
        y += s[i] * q[i]
    true_res = mp.norm(A * y - evals[low] * y)
    bound = b * abs(s[k - 1])
    worst = max(worst, abs(true_res - bound))
    beta.append(b)
    q.append(w / b)
assert worst < mp.mpf("1e-40"), worst
print(f"Ritz residual equals beta_j |s_j| through dimension {n - 1}: max gap {mp.nstr(worst, 3)}")
