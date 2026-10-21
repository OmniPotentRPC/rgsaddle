"""The Jacobian and the strain-stress pairing of the solid-state band.

Run: uv run --no-project --with sympy python validation/ssneb_jacobian.py

src/solid_state.rs (Sheppard, Xiao, Chemelewski, Johnson, Henkelman,
J. Chem. Phys. 136, 074103 (2012), doi:10.1063/1.3684549) appends to
the atomic displacement the strain averaged in the two cells,

    eps_J = J/2 (h_A^-1 + h_B^-1)(h_B - h_A),

and to the atomic force the cell force F_c = -(V/J) sigma, with
J = (V/N)^(1/3) sqrt(N) and sigma = (1/V) dE/deps for the right strain
h <- h (I + eps) at fixed fractional coordinates. Checked here:

1. Units: J is a length, eps_J a length, F_c an energy per length,
   and F_c : eps_J an energy, as F . dr is.
2. Pairing: on a Bravais-lattice pair energy, -dE/d(J eps_ij) along a
   lower-triangular strain is the (i, j) entry of F_c, so the joint
   force is the negative joint gradient on the cell block as it is on
   the atoms.
3. Rotations: E is invariant under h <- h R, so V sigma is symmetric and
   a first-order rotation (antisymmetric eps) costs no energy; the
   lower triangle carries every strain and clearing the upper one
   removes only the rotation.
4. Scale: J eps equals the root-sum-square change of the N
   nearest-neighbour vectors under a uniform strain eps, sqrt(N) eps a
   with a = (V/N)^(1/3), so a unit of joint strain weighs what a unit
   of atomic displacement does.
5. The averaged strain is antisymmetric in the two cells and equals
   J eps to first order for h_B = h_A (I + eps).
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

# 1. Units: scale length by l and energy by e and read off the degrees.
l, e = sp.symbols("l e", positive=True)
V, N, sigma, eps = sp.symbols("V N sigma eps", positive=True)
J = (V / N) ** sp.Rational(1, 3) * sp.sqrt(N)
dims = {V: V * l**3, sigma: sigma * e / l**3}
J_s = J.subs(dims)
assert sp.simplify(J_s / J - l) == 0
eps_J = J * eps
assert sp.simplify(eps_J.subs(dims) / eps_J - l) == 0
F_c = V * sigma / J
assert sp.simplify(F_c.subs(dims) / F_c - e / l) == 0
work = F_c * eps_J
assert sp.simplify(work - V * sigma * eps) == 0
assert sp.simplify(work.subs(dims) / work - e) == 0
print("units: J and J eps are lengths, (V/J) sigma is a force, their product V sigma eps is work: ok")

# 2. Pairing on a lattice pair energy. Lower-triangular cell, harmonic
# pair term about r0 over the nearest lattice translations, evaluated at
# a rational sample cell; every derivative is exact and compared to 40
# digits.
h = sp.Matrix(3, 3, lambda r, c: sp.Symbol(f"h{r}{c}", real=True) if c <= r else 0)
r0, k = sp.symbols("r0 k", positive=True)
shells = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, -1, 0)]
sample = {h[0, 0]: sp.Rational(3), h[1, 0]: sp.Rational(1, 2), h[1, 1]: sp.Rational(29, 10),
          h[2, 0]: sp.Rational(1, 5), h[2, 1]: sp.Rational(-3, 10), h[2, 2]: sp.Rational(31, 10),
          r0: sp.Rational(28, 10), k: 2}


def energy(cell):
    total = 0
    for n in shells:
        v = sp.Matrix([n]) * cell
        r = sp.sqrt((v * v.T)[0])
        total += k / 2 * (r - r0) ** 2
    return total


h_num = h.subs(sample)
volume = h_num.det()
strain = sp.Matrix(3, 3, lambda r, c: sp.Symbol(f"e{r}{c}", real=True))
E_strained = energy(h_num * (sp.eye(3) + strain)).subs(sample)
zero = {strain[r, c]: 0 for r in range(3) for c in range(3)}
dE = sp.Matrix(3, 3, lambda r, c: sp.diff(E_strained, strain[r, c]).subs(zero))
cauchy = dE / volume
Jsym = sp.Symbol("J", positive=True)
J_num = sp.Rational(7, 3)
cell_force = -(volume / J_num) * cauchy
# Joint coordinate q = J eps on the lower triangle: -dE/dq_ij = F_c[i, j].
q = sp.Symbol("q")
for r in range(3):
    for c in range(r + 1):
        along = sp.zeros(3, 3)
        along[r, c] = q / J_num
        Eq = energy(h_num * (sp.eye(3) + along)).subs(sample)
        lhs = -sp.diff(Eq, q).subs(q, 0)
        assert abs(sp.N(lhs - cell_force[r, c], 50)) < 1e-40, (r, c)
print("pairing: -dE/d(J eps_ij) = -(V/J) sigma_ij on every lower-triangle entry: ok")

# 3. Rotation invariance makes V sigma symmetric; an antisymmetric strain is free.
dE_num = dE.applyfunc(lambda v: sp.N(v, 50))
assert max(abs(v) for v in (dE_num - dE_num.T)) < 1e-40
w = sp.Symbol("w")
for (a, b) in [(0, 1), (0, 2), (1, 2)]:
    skew = sp.zeros(3, 3)
    skew[a, b], skew[b, a] = w, -w
    rate = sp.diff(energy(h_num * (sp.eye(3) + skew)).subs(sample), w).subs(w, 0)
    assert abs(sp.N(rate, 50)) < 1e-40, (a, b)
print("rotations: V sigma symmetric and antisymmetric strains free on the sample cell: ok")

# 4. Scale: a cubic cell of N = n^3 atoms at spacing a = (V/N)^(1/3).
# A uniform strain eps changes every nearest-neighbour vector by eps a;
# over N atoms the root-sum-square of that change is sqrt(N) eps a.
a, n_side = sp.symbols("a n", positive=True)
N_atoms = n_side**3
V_cube = (n_side * a) ** 3
J_cube = (V_cube / N_atoms) ** sp.Rational(1, 3) * sp.sqrt(N_atoms)
neighbour_rss = sp.sqrt(N_atoms * (eps * a) ** 2)
assert sp.simplify(J_cube * eps - neighbour_rss) == 0
print("scale: J eps is the root-sum-square neighbour displacement sqrt(N) eps (V/N)^(1/3): ok")

# 5. Averaged strain: antisymmetric in the cells, J eps to first order.
hA = sp.Matrix(3, 3, lambda r, c: sp.Symbol(f"a{r}{c}", real=True) if c <= r else 0)
hB = sp.Matrix(3, 3, lambda r, c: sp.Symbol(f"b{r}{c}", real=True) if c <= r else 0)


def averaged(x, y):
    return Jsym / 2 * (x.inv() + y.inv()) * (y - x)


assert sp.simplify(averaged(hA, hB) + averaged(hB, hA)) == sp.zeros(3, 3)
t = sp.Symbol("t")
small = sp.Matrix(3, 3, lambda r, c: sp.Symbol(f"d{r}{c}", real=True) if c <= r else 0)
series = averaged(hA, hA * (sp.eye(3) + t * small))
first = sp.simplify(series.diff(t).subs(t, 0))
assert sp.simplify(first - Jsym * small) == sp.zeros(3, 3)
print("averaged strain: antisymmetric in the two cells, J eps to first order: ok")
