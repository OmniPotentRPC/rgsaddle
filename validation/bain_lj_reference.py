"""Reference Bain path of a switched Lennard-Jones crystal.

Run: uv run --no-project --with numpy --with scipy python validation/bain_lj_reference.py

The case behind tests/c/solid_bain_lj.c. Lennard-Jones with
epsilon = 1 eV and sigma = 1 Angstrom, unshifted, times rgpot's C^2
quintic switch S(t) = 1 - 10 t^3 + 15 t^4 - 6 t^5 over [2.0, 2.2]
Angstrom (rgpot LJConfig cutoff 2.2, switch_width 0.2).

A body-centred tetragonal Bravais lattice, conventional cell
diag(a, a, c) with atoms at (0, 0, 0) and (1/2, 1/2, 1/2), is bcc at
q = c/a = 1 and fcc at q = sqrt(2). Every atom sits on an inversion
centre, so the atomic forces vanish at any cell and the relaxed
energy at a fixed q is a one-dimensional minimum over the volume per
atom v:

    E*(q) = min_v e(q, v),   a = (2 v / q)^(1/3),   c = q a.

The saddle of e(q, v) on the tetragonal path is the maximum of E*(q),
since both partial derivatives vanish there. The scan evaluates E* on
a uniform grid of spacing DQ between the two minima and refines the
maximum by a bounded search. The script checks, and prints:

1. The minima: bcc at q = 1 and fcc at q = sqrt(2), each a minimum of
   E* and with a positive-definite Hessian over the six cell strains.
2. One interior maximum of E* on the grid. Its strain Hessian has
   exactly one negative eigenvalue: an index-1 saddle in cell space.
3. Both minima are dynamically stable: the lowest phonon eigenvalue
   on a 12^3 grid of wave vectors is positive.
4. Along the whole relaxed path, the 5 x 5 x 4 supercell used by the
   eOn test keeps every width above twice the cutoff, so a
   minimum-image pair list sees every pair inside the cutoff once.
5. The analytic Cauchy stress matches a central difference of the
   energy along each lower-triangular strain.

The barrier from this scan, per atom, is the reference for the band
tests; the scan resolution DQ is their tolerance on c/a, and the
energy change over one grid step at the saddle is their tolerance on
the barrier.

bcc is metastable here because of the truncation: with a cutoff of
2.5 Angstrom or longer the bcc lattice of this pair potential is
unstable along the Bain path and is not a minimum.
"""

if not __debug__:
    raise SystemExit("The reference checks require Python assertions; disable optimization.")

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1.0
SIG = 1.0
RC = 2.2
WIDTH = 0.2
RON = RC - WIDTH
DQ = 2.5e-4
SUPERCELL = (5, 5, 4)

_n = np.arange(-4, 5)
_grid = np.array(np.meshgrid(_n, _n, _n, indexing="ij")).reshape(3, -1).T.astype(float)
SITES = np.vstack([_grid, _grid + 0.5])
SITES = SITES[np.any(SITES != 0.0, axis=1)]


def pair(r):
    """phi, phi', phi'' of the switched pair term."""
    sr6 = (SIG / r) ** 6
    v = 4.0 * EPS * (sr6 * sr6 - sr6)
    v1 = 4.0 * EPS * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r
    v2 = 4.0 * EPS * (156.0 * sr6 * sr6 - 42.0 * sr6) / (r * r)
    t = np.clip((r - RON) / WIDTH, 0.0, 1.0)
    s = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
    s1 = -30.0 * t**2 * (1.0 - t) ** 2 / WIDTH
    s2 = (-60.0 * t + 180.0 * t**2 - 120.0 * t**3) / WIDTH**2
    return v * s, v1 * s + v * s1, v2 * s + 2.0 * v1 * s1 + v * s2


def lattice(h):
    """Neighbour vectors inside the cutoff for rows-are-vectors cell h."""
    vec = SITES @ h
    r = np.linalg.norm(vec, axis=1)
    keep = r < RC
    return vec[keep], r[keep]


def energy(h):
    """Energy per atom."""
    _, r = lattice(h)
    return 0.5 * pair(r)[0].sum()


def stress(h):
    """Cauchy stress (1/V) dE/deps for h <- h (I + eps), eV/Angstrom^3."""
    vec, r = lattice(h)
    _, d1, _ = pair(r)
    volume_per_atom = abs(np.linalg.det(h)) / 2.0
    return 0.5 * np.einsum("k,ki,kj->ij", d1 / r, vec, vec) / volume_per_atom


def cell(q, v):
    a = (2.0 * v / q) ** (1.0 / 3.0)
    return np.diag([a, a, q * a])


def relaxed(q):
    """E*(q) and the volume per atom that attains it."""
    res = minimize_scalar(
        lambda v: energy(cell(q, v)), bounds=(0.80, 1.10), method="bounded", options={"xatol": 1e-12}
    )
    return res.fun, res.x


def strain_hessian(h, step=1e-4):
    """Hessian of the energy per atom over the six symmetric strains."""

    def at(e6):
        e = np.array([[e6[0], e6[5], e6[4]], [e6[5], e6[1], e6[3]], [e6[4], e6[3], e6[2]]])
        return energy(h @ (np.eye(3) + e))

    hess = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            ei = np.zeros(6)
            ej = np.zeros(6)
            ei[i] = step
            ej[j] = step
            hess[i, j] = (at(ei + ej) - at(ei - ej) - at(ej - ei) + at(-ei - ej)) / (4.0 * step * step)
    return np.linalg.eigvalsh(0.5 * (hess + hess.T))


def lowest_phonon(h, nk=12):
    """Lowest eigenvalue of the dynamical matrix over an nk^3 grid, k = 0 excluded."""
    vec, r = lattice(h)
    _, d1, d2 = pair(r)
    u = vec / r[:, None]
    blocks = (d2 - d1 / r)[:, None, None] * u[:, :, None] * u[:, None, :] + (d1 / r)[:, None, None] * np.eye(3)
    recip = 2.0 * np.pi * np.linalg.inv(h).T
    f = np.arange(nk) / nk
    ks = np.array(np.meshgrid(f, f, f, indexing="ij")).reshape(3, -1).T @ recip
    lowest = np.inf
    for k in ks[1:]:
        dyn = (blocks * (1.0 - np.cos(vec @ k))[:, None, None]).sum(axis=0)
        lowest = min(lowest, np.linalg.eigvalsh(dyn)[0])
    return lowest


# 5. Stress against the energy, at a strained triclinic sample cell.
sample = cell(1.09, 0.93) @ np.array([[1.0, 0.0, 0.0], [0.03, 1.0, 0.0], [-0.02, 0.01, 1.0]])
sigma = stress(sample)
volume_per_atom = abs(np.linalg.det(sample)) / 2.0
for i, j in [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)]:
    eps = np.zeros((3, 3))
    eps[i, j] = 1e-6
    numeric = (energy(sample @ (np.eye(3) + eps)) - energy(sample @ (np.eye(3) - eps))) / 2e-6
    assert abs(numeric / volume_per_atom - sigma[i, j]) < 1e-6, (i, j, numeric / volume_per_atom, sigma[i, j])
print("stress: (1/V) dE/deps matches the pair virial on a triclinic cell to 1e-6 eV/A^3: ok")

# 1. Minima.
e_bcc, v_bcc = relaxed(1.0)
e_fcc, v_fcc = relaxed(np.sqrt(2.0))
for q0 in (1.0, np.sqrt(2.0)):
    e0 = relaxed(q0)[0]
    assert relaxed(q0 - 1e-3)[0] > e0 and relaxed(q0 + 1e-3)[0] > e0, q0
assert strain_hessian(cell(1.0, v_bcc)).min() > 0.0
assert strain_hessian(cell(np.sqrt(2.0), v_fcc)).min() > 0.0

# 2. Grid and saddle.
qs = np.arange(1.0, np.sqrt(2.0), DQ)
es = np.array([relaxed(q)[0] for q in qs])
interior = [k for k in range(1, len(qs) - 1) if es[k] > es[k - 1] and es[k] > es[k + 1]]
assert len(interior) == 1, interior
k = interior[0]
q_grid, e_grid = qs[k], es[k]
res = minimize_scalar(
    lambda q: -relaxed(q)[0], bounds=(qs[k - 1], qs[k + 1]), method="bounded", options={"xatol": 1e-10}
)
q_saddle = res.x
e_saddle, v_saddle = relaxed(q_saddle)
assert abs(q_saddle - q_grid) <= 0.5 * DQ + 1e-9
assert e_saddle >= e_grid
saddle_eigs = strain_hessian(cell(q_saddle, v_saddle))
assert (saddle_eigs < 0.0).sum() == 1, saddle_eigs
energy_step = e_saddle - min(relaxed(q_saddle - DQ)[0], relaxed(q_saddle + DQ)[0])

# 3. Phonons.
phonon_bcc = lowest_phonon(cell(1.0, v_bcc))
phonon_fcc = lowest_phonon(cell(np.sqrt(2.0), v_fcc))
assert phonon_bcc > 0.0 and phonon_fcc > 0.0

# 4. Supercell widths along the relaxed path.
nx, ny, nz = SUPERCELL
width = min(min(nx, ny) * cell(q, relaxed(q)[1])[0, 0] for q in qs[::40])
width = min(width, nz * cell(1.0, v_bcc)[2, 2])
assert width > 2.0 * RC, width


def lattice_constants(q, v):
    h = cell(q, v)
    return h[0, 0], h[2, 2]


a_bcc, _ = lattice_constants(1.0, v_bcc)
a_fcc, c_fcc = lattice_constants(np.sqrt(2.0), v_fcc)
a_sad, c_sad = lattice_constants(q_saddle, v_saddle)
print()
print("| point  | c/a        | v (A^3/atom) | a (A)       | c (A)       | E (eV/atom)   |")
print("|--------+------------+--------------+-------------+-------------+---------------|")
print(f"| bcc    | 1          | {v_bcc:.10f} | {a_bcc:.9f} | {a_bcc:.9f} | {e_bcc:.10f} |")
print(f"| saddle | {q_saddle:.8f} | {v_saddle:.10f} | {a_sad:.9f} | {c_sad:.9f} | {e_saddle:.10f} |")
print(f"| fcc    | sqrt(2)    | {v_fcc:.10f} | {a_fcc:.9f} | {c_fcc:.9f} | {e_fcc:.10f} |")
print()
print(f"barrier E_saddle - E_bcc     {e_saddle - e_bcc:.10f} eV/atom")
print(f"reaction E_fcc - E_bcc       {e_fcc - e_bcc:.10f} eV/atom")
print(f"grid maximum                 c/a = {q_grid:.6f}, E - E_bcc = {e_grid - e_bcc:.10f} eV/atom")
print(f"scan resolution              DQ = {DQ:g} over {len(qs)} points")
print(f"energy over one step at max  {energy_step:.3e} eV/atom")
print(f"strain Hessian (eV/atom)     bcc min {strain_hessian(cell(1.0, v_bcc)).min():.4f}")
print(f"                             saddle {np.array2string(saddle_eigs, precision=4)}")
print(f"                             fcc min {strain_hessian(cell(np.sqrt(2.0), v_fcc)).min():.4f}")
print(f"lowest phonon (eV/A^2)       bcc {phonon_bcc:.4f}, fcc {phonon_fcc:.4f}")
print(f"narrowest {nx}x{ny}x{nz} width     {width:.4f} A > 2 rc = {2 * RC} A")
