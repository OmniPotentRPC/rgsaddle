"""Strain parametrisation and Jacobian of the Bain-path band.

Run: uv run --no-project --with sympy python validation/ssneb_bain_cell.py

The band of tests/c/solid_bain_lj.c carries an nx x ny x nz supercell
of the two-atom body-centred tetragonal cell diag(a, a, c), N = 2 nx
ny nz atoms. Endpoint bcc is diag(a0, a0, a0); endpoint fcc is
diag(a1, a1, sqrt(2) a1). Checked here:

1. Parametrisation. The cell h(q, v) = diag(nx a, ny a, nz q a) with
   a = (2 v / q)^(1/3) has volume N v, and its c/a is recovered from
   the cell as q = (h22 / nz) / (h00 / nx).
2. The linear interpolation of the two endpoint cells, the initial
   band of rgsaddle and eOn alike, stays diagonal with h00 / nx =
   h11 / ny, and its c/a rises monotonically from 1 to sqrt(2).
3. The right strain h_A^-1 (h_B - h_A) between two such cells, and the
   averaged strain of the joint metric, are diagonal: no shear and no
   rotation enters the cell block. The test builds every image from the
   same fractional coordinates, so the atomic block of the joint
   displacement starts at zero, and the inversion symmetry of each site
   keeps the atomic force at zero.
4. J = (Vbar / N)^(1/3) sqrt(N), with Vbar the mean endpoint volume,
   equals vbar^(1/3) sqrt(N) for the mean endpoint volume per atom
   vbar; the C test compares rgsaddle_band_solid_jacobian against it.
5. Supercell scaling. With E = N e(eps) and sigma = (1/v) de/deps, the
   cell force -(V/J) sigma and the joint strain J eps both scale as
   sqrt(N), and so does a uniform spring force k |J eps|. The
   stationary band in (q, v) does not depend on N; only a force
   tolerance does, by sqrt(N).
"""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

import sympy as sp

nx, ny, nz = sp.symbols("n_x n_y n_z", positive=True, integer=True)
q, v = sp.symbols("q v", positive=True)
N = 2 * nx * ny * nz

# 1. Parametrisation.
a = (2 * v / q) ** sp.Rational(1, 3)
h = sp.diag(nx * a, ny * a, nz * q * a)
assert sp.simplify(h.det() - N * v) == 0
assert sp.simplify((h[2, 2] / nz) / (h[0, 0] / nx) - q) == 0
assert sp.simplify(h[0, 0] / nx - h[1, 1] / ny) == 0
print("parametrisation: det h = N v and c/a = (h22/nz)/(h00/nx) = q: ok")

# 2. Linear interpolation of the endpoint cells.
a0, a1, t = sp.symbols("a0 a1 t", positive=True)
h0 = sp.diag(nx * a0, ny * a0, nz * a0)
h1 = sp.diag(nx * a1, ny * a1, nz * sp.sqrt(2) * a1)
ht = (1 - t) * h0 + t * h1
assert all(ht[i, j] == 0 for i in range(3) for j in range(3) if i != j)
assert sp.simplify(ht[0, 0] / nx - ht[1, 1] / ny) == 0
q_t = sp.simplify((ht[2, 2] / nz) / (ht[0, 0] / nx))
assert sp.simplify(q_t.subs(t, 0) - 1) == 0
assert sp.simplify(q_t.subs(t, 1) - sp.sqrt(2)) == 0
dq_dt = sp.factor(sp.diff(q_t, t))
# dq/dt = (sqrt(2) - 1) a0 a1 / ((1 - t) a0 + t a1)^2 > 0 for positive lattice constants.
assert sp.simplify(dq_dt - (sp.sqrt(2) - 1) * a0 * a1 / ((1 - t) * a0 + t * a1) ** 2) == 0
print("interpolation: diagonal, tetragonal, c/a from 1 to sqrt(2) with dq/dt > 0: ok")

# 3. Strains between two tetragonal cells.
qa, va, qb, vb, J = sp.symbols("q_a v_a q_b v_b J", positive=True)
hA = h.subs({q: qa, v: va})
hB = h.subs({q: qb, v: vb})
right = sp.simplify(hA.inv() * (hB - hA))
joint = sp.simplify(J / 2 * (hA.inv() + hB.inv()) * (hB - hA))
for m in (right, joint):
    assert all(sp.simplify(m[i, j]) == 0 for i in range(3) for j in range(3) if i != j)
assert sp.simplify(joint[0, 0] - joint[1, 1]) == 0
print("strain: right and averaged strains are diagonal: ok")

# 4. Jacobian.
vbar = sp.symbols("vbar", positive=True)
Vbar = N * vbar
J_cell = (Vbar / N) ** sp.Rational(1, 3) * sp.sqrt(N)
assert sp.simplify(J_cell - vbar ** sp.Rational(1, 3) * sp.sqrt(N)) == 0
for reps, atoms in (((1, 1, 1), 2), ((5, 5, 4), 200)):
    value = J_cell.subs({nx: reps[0], ny: reps[1], nz: reps[2]})
    assert sp.simplify(value - vbar ** sp.Rational(1, 3) * sp.sqrt(atoms)) == 0
print("jacobian: J = vbar^(1/3) sqrt(N); 2 atoms sqrt(2) vbar^(1/3), 200 atoms 10 sqrt(2) vbar^(1/3): ok")

# 5. Scaling with the supercell.
eps, k = sp.symbols("epsilon k", positive=True)
e = sp.Function("e")
n = sp.symbols("N", positive=True)
Jn = vbar ** sp.Rational(1, 3) * sp.sqrt(n)
sigma = sp.diff(e(eps), eps) / v  # (1/V) dE/deps with E = N e, V = N v
cell_force = -(n * v / Jn) * sigma
joint_strain = Jn * eps
spring = k * joint_strain
for quantity in (cell_force, joint_strain, spring):
    assert sp.simplify(quantity.subs(n, 100 * n) / quantity - 10) == 0
print("scaling: cell force, joint strain and spring force all scale as sqrt(N): ok")
