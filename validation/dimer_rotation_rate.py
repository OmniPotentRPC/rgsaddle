"""Rotation counts of three dimer rotation schemes on an exact quadratic.

Run: uv run --with numpy python validation/dimer_rotation_rate.py

H = diag(-1, 0.5, 1, 2, 4, 8), so the Hessian action is exact and only
the rotation scheme differs. Each rotation costs one gradient in every
scheme. Columns: rotations to bring the rotational force |H n - C n|
under the tolerance (2000 = did not converge).

- fixed: the steepest rotation with the fixed trust scale
  min(|F|/(|C|+|F|), 0.5) that src/minmode.rs used before the modified
  Newton step;
- newton-sd: the modified Newton in-plane step along the rotational
  force;
- newton-cg: the same step along Polak-Ribiere conjugate directions
  (src/minmode.rs).
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import numpy as np

H = np.diag([-1.0, 0.5, 1.0, 2.0, 4.0, 8.0])


def newton(seed, cg, tol, maxit=2000):
    n = seed / np.linalg.norm(seed)
    h = H @ n
    c = h @ n
    f_prev = d_prev = None
    it = 0
    while True:
        f = -(h - c * n)
        fn = np.linalg.norm(f)
        if fn <= tol or it >= maxit:
            return it, c
        d = f.copy()
        if cg and f_prev is not None:
            d = d + max(0.0, (f - f_prev) @ f / (f_prev @ f_prev)) * d_prev
        d = d - (d @ n) * n
        dn = np.linalg.norm(d)
        th = d / dn
        b1 = h @ th
        phi1 = np.clip(0.5 * np.arctan(abs(b1) / (abs(c) + 1e-300)), 1e-4, np.pi / 4)
        n1 = n * np.cos(phi1) + th * np.sin(phi1)
        c1 = (H @ n1) @ n1
        it += 1
        a1 = (c - c1 + b1 * np.sin(2 * phi1)) / (1 - np.cos(2 * phi1))
        a0 = 2 * (c - a1)

        def model(p):
            return 0.5 * a0 + a1 * np.cos(2 * p) + b1 * np.sin(2 * p)

        phi = 0.5 * np.arctan(b1 / a1)
        if model(phi) > model(phi + np.pi / 2):
            phi += np.pi / 2
        if phi > np.pi / 2:
            phi -= np.pi
        s, co = np.sin(phi), np.cos(phi)
        th_rot = th * co - n * s
        n = n * co + th * s
        n /= np.linalg.norm(n)
        h = H @ n
        c = h @ n
        f_prev, d_prev = f, th_rot * dn


def fixed(seed, tol, maxit=2000):
    t = seed / np.linalg.norm(seed)
    r = 0
    while True:
        hv = H @ t
        c = hv @ t
        p = hv - t * c
        pn = np.linalg.norm(p)
        if pn <= tol or r >= maxit:
            return r, c
        t = t - (p / pn) * min(pn / (abs(c) + pn), 0.5)
        t /= np.linalg.norm(t)
        r += 1


seeds = [np.array([0.3, 0.5, 0.4, 0.4, 0.4, 0.4])]
rng = np.random.default_rng(1)
seeds += [rng.normal(size=6) for _ in range(3)]
print(f"{'seed':>5} {'tol':>7} {'fixed':>6} {'newton-sd':>10} {'newton-cg':>10}")
for k, seed in enumerate(seeds):
    for tol in (1e-2, 1e-4, 1e-6):
        print(
            f"{k:5d} {tol:7.0e} {fixed(seed, tol)[0]:6d}"
            f" {newton(seed, False, tol)[0]:10d} {newton(seed, True, tol)[0]:10d}"
        )
it, c = newton(seeds[0], True, 1e-6)
assert abs(c + 1.0) < 1e-10 and it == 16, (it, c)
