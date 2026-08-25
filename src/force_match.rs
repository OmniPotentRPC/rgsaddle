//! Sella `force_match.pyx`: pair force-field seed Hessian, in Rust.
//!
//! Linear coefficients are least-squares against the Cartesian force.
//! Nonlinear `rho` / `r0` follow Sella: joint `brute` (`Ns=10` when
//! `nnonlin < 5`) then L-BFGS-B with `dFnonlin`. The hot path is the
//! pair force / Hessian kernels (`lj` / `buck` / `morse` / `bond`);
//! there is no Python optimizer. Reductions go through
//! [`rgmin::vecops`] so `par` applies. A host that needs the increment
//! on a set calls [`force_match_on`] (`project` / `retract` /
//! `transport`).

use ndarray::{Array1, Array2, Array3, ArrayView1};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot, nrm2};

use crate::eigensolve::solve_dense;
use crate::error::SaddleError;

/// Sella default Buckingham / Morse `rho`.
const DEFAULT_RHO: f64 = 2.5;

/// Pair term in `force_match.pyx` `types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairKind {
    /// Lennard-Jones `C6` / `C12`.
    Lj,
    /// Buckingham `A`, `C6`, `rho`.
    Buckingham,
    /// Morse `A`, `B`, `rho`.
    Morse,
    /// Harmonic bond `K` at `r0`.
    Bond,
}

/// Options for the full Cython matcher.
#[derive(Clone, Debug)]
pub struct ForceMatchOpts {
    /// Sella `types`. Default is `buck` then `bond`.
    pub kinds: Vec<PairKind>,
    /// Bond cutoff scale. Sella uses `1.5 * (rcov_i + rcov_j)`.
    pub bond_scale: f64,
    /// vdW cutoff as a multiple of the shortest pair distance. Sella uses 3.
    pub rcut_factor: f64,
    /// Lattice translations. Empty / identity is the isolated molecule.
    pub tvecs: Vec<[f64; 3]>,
}

impl Default for ForceMatchOpts {
    fn default() -> Self {
        Self {
            kinds: vec![PairKind::Buckingham, PairKind::Bond],
            bond_scale: 1.5,
            rcut_factor: 3.0,
            tvecs: vec![[0.0, 0.0, 0.0]],
        }
    }
}

/// Fitted seed Hessian plus linear / nonlinear coefficients.
#[derive(Clone, Debug)]
pub struct ForceMatchReport {
    pub hessian: Array2<f64>,
    pub linpars: Array1<f64>,
    pub nonlinpars: Array1<f64>,
}

/// ASE `covalent_radii` missing placeholder (Å).
const CORDERO_MISSING: f64 = 0.2;

/// ASE Cordero covalent radii (Å), Z = 0..=96 (`ase.data.covalent_radii`).
const CORDERO: [f64; 97] = [
    0.20, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07,
    1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22,
    1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45,
    1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,
    1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36,
    1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80,
    1.69,
];

/// ASE Cordero covalent radius (Å) by Z. Missing entries are 0.2.
pub fn covalent_radius(z: u8) -> f64 {
    CORDERO.get(z as usize).copied().unwrap_or(CORDERO_MISSING)
}

/// Pairs with `r < scale * (rcov_i + rcov_j)`.
pub fn covalent_pairs(
    x: ArrayView1<f64>,
    z: &[u8],
    scale: f64,
) -> Result<Vec<[usize; 2]>, SaddleError> {
    if x.len() % 3 != 0 || x.len() / 3 != z.len() {
        return Err(SaddleError::Shape(
            "force_match frame is not 3N with matching Z".into(),
        ));
    }
    let n = z.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = [
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ];
            let r2 = dot3(dx);
            if r2 <= f64::MIN_POSITIVE {
                continue;
            }
            let r = r2.sqrt();
            let cut = scale * (covalent_radius(z[i]) + covalent_radius(z[j]));
            if r <= cut {
                pairs.push([i, j]);
            }
        }
    }
    Ok(pairs)
}

/// Fit pair spring constants so `F_ff ≈ -g`.
///
/// Pair `i-j` contributes `k (r - r0) u` on `i` and the opposite on
/// `j` (`V = 1/2 k (r-r0)^2`), with `r0 = rcov_i + rcov_j`.
pub fn fit_bond_ks(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    z: &[u8],
    pairs: &[[usize; 2]],
) -> Result<Array1<f64>, SaddleError> {
    if g.len() != x.len() {
        return Err(SaddleError::Shape(
            "force_match gradient must match the 3N frame".into(),
        ));
    }
    let nlin = pairs.len();
    let ndof = x.len();
    let mut jac = Array2::<f64>::zeros((ndof, nlin));
    for (p, pair) in pairs.iter().enumerate() {
        let (col, _) = pair_force_col(x, z, *pair)?;
        for i in 0..ndof {
            jac[(i, p)] = col[i];
        }
    }
    Ok(fit_linear(&jac, force_target(g)))
}

/// Analytic pair-harmonic Hessian at the current geometry.
pub fn bond_hessian(
    x: ArrayView1<f64>,
    z: &[u8],
    pairs: &[[usize; 2]],
    ks: ArrayView1<f64>,
) -> Result<Array2<f64>, SaddleError> {
    if ks.len() != pairs.len() {
        return Err(SaddleError::Shape(
            "force_match k vector must match the pair list".into(),
        ));
    }
    let mut h = Array2::<f64>::zeros((x.len(), x.len()));
    for (p, pair) in pairs.iter().enumerate() {
        let i = pair[0];
        let j = pair[1];
        let (r, dx) = pair_disp(x, i, j, [0.0, 0.0, 0.0])?;
        let r0 = covalent_radius(z[i]) + covalent_radius(z[j]);
        let stretch = (r - r0) / r;
        accumulate_pair(&mut h, i, j, dx, ks[p] * stretch, ks[p] * r0 / (r * r * r));
    }
    Ok(h)
}

/// Fit bond `k` and return the seed Hessian in one call.
pub fn force_match_hessian(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    z: &[u8],
    scale: f64,
) -> Result<(Array2<f64>, Array1<f64>, Vec<[usize; 2]>), SaddleError> {
    let pairs = covalent_pairs(x, z, scale)?;
    if pairs.is_empty() {
        return Ok((Array2::eye(x.len()), Array1::zeros(0), pairs));
    }
    let ks = fit_bond_ks(x, g, z, &pairs)?;
    let h = bond_hessian(x, z, &pairs, ks.view())?;
    Ok((h, ks, pairs))
}

/// Full Sella matcher. Default `kinds` is Buckingham then bond.
pub fn force_match(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    z: &[u8],
    opts: &ForceMatchOpts,
) -> Result<ForceMatchReport, SaddleError> {
    if x.len() % 3 != 0 || x.len() / 3 != z.len() || g.len() != x.len() {
        return Err(SaddleError::Shape(
            "force_match frame is not 3N with matching Z and gradient".into(),
        ));
    }
    let set = collect_pairs(x, z, opts)?;
    let nlin = set.nlin();
    let nnonlin = set.nnonlin();
    if nlin == 0 {
        return Ok(ForceMatchReport {
            hessian: Array2::eye(x.len()),
            linpars: Array1::zeros(0),
            nonlinpars: Array1::zeros(0),
        });
    }
    let ftrue = force_target(g);
    let (guess, lo, hi) = nonlin_bounds(&set);
    let nonlin = if nnonlin == 0 {
        Array1::zeros(0)
    } else {
        refine_nonlin(&set, x.len(), &ftrue, guess, &lo, &hi)?
    };
    let (linpars, _, _) = jac_and_fit(&set, x.len(), nonlin.view(), &ftrue)?;
    let hessian = assemble_hess(&set, x.len(), linpars.view(), nonlin.view())?;
    Ok(ForceMatchReport {
        hessian,
        linpars,
        nonlinpars: nonlin,
    })
}

/// Project a force-match direction onto `T_x`.
pub fn project_force_match<M: Manifold>(man: &M, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    man.project(x, v)
}

/// Retract a force-match increment onto the set.
pub fn retract_force_match<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    v: &Array1<f64>,
    eta: f64,
) -> Array1<f64> {
    let vn = nrm2(v.view());
    if vn < 1e-12 {
        return x.clone();
    }
    let mut s = Array1::zeros(v.len());
    axpy(eta / vn, v.view(), &mut s);
    let s = man.project(x, &s);
    man.retract(x, &s)
}

/// Vector transport of a force-match increment from `x_from` to `x_to`.
pub fn transport_force_match<M: Manifold>(
    man: &M,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    man.transport(x_from, x_to, v)
}

/// Seed Hessian plus a retracted increment that stays on `man`.
pub fn force_match_on<M: Manifold>(
    man: &M,
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    z: &[u8],
    opts: &ForceMatchOpts,
) -> Result<(Array1<f64>, ForceMatchReport), SaddleError> {
    let report = force_match(x, g, z, opts)?;
    let x0 = x.to_owned();
    let force = force_target(g);
    let step = newton_increment(&report.hessian, &force);
    let v = project_force_match(man, &x0, &step);
    let y = retract_force_match(man, &x0, &v, 1e-4);
    let _ = transport_force_match(man, &x0, &y, &v);
    Ok((y, report))
}

/// Regularized Newton increment `H s ≈ F` for the seed Hessian.
fn newton_increment(h: &Array2<f64>, f: &Array1<f64>) -> Array1<f64> {
    let n = f.len();
    if h.nrows() != n || h.ncols() != n {
        return f.clone();
    }
    let mut a = h.clone();
    for i in 0..n {
        a[(i, i)] += 1e-8;
    }
    solve_dense(a.view(), f.view()).unwrap_or_else(|_| f.clone())
}

struct Pair {
    i: usize,
    j: usize,
    xij: [f64; 3],
}

struct Interaction {
    pairs: Vec<Pair>,
    r0: f64,
}

struct PairSet {
    lj: Vec<Interaction>,
    buck: Vec<Interaction>,
    morse: Vec<Interaction>,
    bond: Vec<Interaction>,
}

impl PairSet {
    fn nlin(&self) -> usize {
        2 * self.lj.len() + 2 * self.buck.len() + 2 * self.morse.len() + self.bond.len()
    }

    fn nnonlin(&self) -> usize {
        self.buck.len() + self.morse.len() + self.bond.len()
    }
}

fn collect_pairs(
    x: ArrayView1<f64>,
    z: &[u8],
    opts: &ForceMatchOpts,
) -> Result<PairSet, SaddleError> {
    let n = z.len();
    let tvecs = if opts.tvecs.is_empty() {
        vec![[0.0, 0.0, 0.0]]
    } else {
        opts.tvecs.clone()
    };
    let mut rmin = f64::INFINITY;
    for i in 0..n {
        for j in i..n {
            let rij = [
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ];
            for &tv in &tvecs {
                if i == j && tv[0] == 0.0 && tv[1] == 0.0 && tv[2] == 0.0 {
                    continue;
                }
                let xij = [rij[0] + tv[0], rij[1] + tv[1], rij[2] + tv[2]];
                let r2 = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2];
                if r2 > f64::MIN_POSITIVE {
                    rmin = rmin.min(r2.sqrt());
                }
            }
        }
    }
    if !rmin.is_finite() {
        return Ok(PairSet {
            lj: Vec::new(),
            buck: Vec::new(),
            morse: Vec::new(),
            bond: Vec::new(),
        });
    }
    let rcut2 = (opts.rcut_factor * rmin).powi(2);
    let do_lj = opts.kinds.contains(&PairKind::Lj);
    let do_buck = opts.kinds.contains(&PairKind::Buckingham);
    let do_morse = opts.kinds.contains(&PairKind::Morse);
    let do_bond = opts.kinds.contains(&PairKind::Bond);

    let mut lj = Vec::<((u8, u8), Interaction)>::new();
    let mut buck = Vec::<((u8, u8), Interaction)>::new();
    let mut morse = Vec::<((u8, u8), Interaction)>::new();
    let mut bond = Vec::<((u8, u8), Interaction)>::new();

    for i in 0..n {
        for j in i..n {
            let rij = [
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ];
            for &tv in &tvecs {
                if i == j && tv[0] == 0.0 && tv[1] == 0.0 && tv[2] == 0.0 {
                    continue;
                }
                let xij = [rij[0] + tv[0], rij[1] + tv[1], rij[2] + tv[2]];
                let r2 = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2];
                if r2 > rcut2 || r2 <= f64::MIN_POSITIVE {
                    continue;
                }
                let r = r2.sqrt();
                let key = elem_key(z[i], z[j]);
                let pair = Pair { i, j, xij };
                if do_lj {
                    push_pair(&mut lj, key, pair.i, pair.j, pair.xij, 0.0);
                }
                if do_buck {
                    push_pair(&mut buck, key, pair.i, pair.j, pair.xij, 0.0);
                }
                if do_morse {
                    push_pair(&mut morse, key, pair.i, pair.j, pair.xij, 0.0);
                }
                if do_bond {
                    let rcov = covalent_radius(key.0) + covalent_radius(key.1);
                    if r <= opts.bond_scale * rcov {
                        push_pair(&mut bond, key, pair.i, pair.j, pair.xij, rcov);
                    }
                }
            }
        }
    }
    Ok(PairSet {
        lj: take_inter(lj),
        buck: take_inter(buck),
        morse: take_inter(morse),
        bond: take_inter(bond),
    })
}

fn elem_key(zi: u8, zj: u8) -> (u8, u8) {
    if zi <= zj { (zi, zj) } else { (zj, zi) }
}

fn push_pair(
    buckets: &mut Vec<((u8, u8), Interaction)>,
    key: (u8, u8),
    i: usize,
    j: usize,
    xij: [f64; 3],
    r0: f64,
) {
    let pair = Pair { i, j, xij };
    match buckets.iter_mut().find(|(k, _)| *k == key) {
        Some((_, inter)) => inter.pairs.push(pair),
        None => buckets.push((
            key,
            Interaction {
                pairs: vec![pair],
                r0,
            },
        )),
    }
}

fn take_inter(buckets: Vec<((u8, u8), Interaction)>) -> Vec<Interaction> {
    buckets.into_iter().map(|(_, inter)| inter).collect()
}

fn nonlin_bounds(set: &PairSet) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = set.nnonlin();
    let mut x0 = Array1::zeros(n);
    let mut lo = Array1::zeros(n);
    let mut hi = Array1::zeros(n);
    let mut k = 0;
    for _ in &set.buck {
        x0[k] = DEFAULT_RHO;
        lo[k] = 0.1;
        hi[k] = 10.0;
        k += 1;
    }
    for _ in &set.morse {
        x0[k] = DEFAULT_RHO;
        lo[k] = 0.1;
        hi[k] = 10.0;
        k += 1;
    }
    for inter in &set.bond {
        x0[k] = inter.r0;
        lo[k] = 0.5 * inter.r0;
        hi[k] = 2.0 * inter.r0;
        k += 1;
    }
    (x0, lo, hi)
}

fn lbfgs_bounds(set: &PairSet) -> (Array1<f64>, Array1<f64>) {
    let n = set.nnonlin();
    let mut lo = Array1::zeros(n);
    let hi = Array1::from_elem(n, f64::INFINITY);
    let mut k = 0;
    for _ in &set.buck {
        lo[k] = 0.0;
        k += 1;
    }
    for _ in &set.morse {
        lo[k] = 1.0;
        k += 1;
    }
    for _ in &set.bond {
        lo[k] = 0.0;
        k += 1;
    }
    (lo, hi)
}

fn refine_nonlin(
    set: &PairSet,
    ndof: usize,
    ftrue: &Array1<f64>,
    guess: Array1<f64>,
    lo: &Array1<f64>,
    hi: &Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let n = guess.len();
    if n == 0 {
        return Ok(guess);
    }
    let x0 = if n < 5 {
        brute_nonlin(set, ndof, ftrue, lo, hi)?
    } else {
        guess
    };
    let (blo, bhi) = lbfgs_bounds(set);
    lbfgs_b(set, ndof, ftrue, x0, &blo, &bhi)
}

fn brute_nonlin(
    set: &PairSet,
    ndof: usize,
    ftrue: &Array1<f64>,
    lo: &Array1<f64>,
    hi: &Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    const NS: usize = 10;
    let n = lo.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut idx = vec![0usize; n];
    let mut best_x = lo.clone();
    let mut best = f64::INFINITY;
    loop {
        let mut x = Array1::zeros(n);
        for i in 0..n {
            let frac = idx[i] as f64 / (NS - 1) as f64;
            x[i] = lo[i] + frac * (hi[i] - lo[i]);
        }
        let chi = chisq(set, ndof, ftrue, x.view())?;
        if chi < best {
            best = chi;
            best_x = x;
        }
        let mut k = 0;
        while k < n {
            idx[k] += 1;
            if idx[k] < NS {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
        if k == n {
            break;
        }
    }
    Ok(best_x)
}

fn lbfgs_b(
    set: &PairSet,
    ndof: usize,
    ftrue: &Array1<f64>,
    mut x: Array1<f64>,
    lo: &Array1<f64>,
    hi: &Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    const M: usize = 6;
    const MAXITER: usize = 200;
    const GTOL: f64 = 1e-10;
    const FTOL: f64 = 1e-8;
    clip_box(&mut x, lo, hi);
    let n = x.len();
    let (mut f, mut g) = objective_grad(set, ndof, ftrue, x.view())?;
    let mut s_hist: Vec<Array1<f64>> = Vec::new();
    let mut y_hist: Vec<Array1<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    for _ in 0..MAXITER {
        let pg = projected_grad(x.view(), g.view(), lo, hi);
        if nrm2(pg.view()) < GTOL {
            break;
        }
        let mut dir = lbfgs_direction(&g, &s_hist, &y_hist, &rho_hist);
        for i in 0..n {
            if x[i] <= lo[i] && dir[i] < 0.0 {
                dir[i] = 0.0;
            }
            if x[i] >= hi[i] && dir[i] > 0.0 {
                dir[i] = 0.0;
            }
        }
        let dn = nrm2(dir.view());
        if dn < 1e-18 {
            break;
        }
        let mut alpha = 1.0;
        let mut accepted = false;
        let mut x_new = x.clone();
        let mut f_new = f;
        let mut g_new = g.clone();
        for _ in 0..30 {
            x_new = &x + &(alpha * &dir);
            clip_box(&mut x_new, lo, hi);
            let (fnv, gnv) = objective_grad(set, ndof, ftrue, x_new.view())?;
            let dx = &x_new - &x;
            let armijo = f + 1e-4 * alpha * dot(g.view(), dx.view());
            if fnv <= armijo {
                f_new = fnv;
                g_new = gnv;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        if (f - f_new).abs() <= FTOL * (1.0 + f.abs()) {
            x = x_new;
            break;
        }
        let s = &x_new - &x;
        let y = &g_new - &g;
        let sy = dot(s.view(), y.view());
        if sy > 1e-16 {
            if s_hist.len() == M {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }
        x = x_new;
        f = f_new;
        g = g_new;
    }
    Ok(x)
}

fn clip_box(x: &mut Array1<f64>, lo: &Array1<f64>, hi: &Array1<f64>) {
    for i in 0..x.len() {
        if x[i] < lo[i] {
            x[i] = lo[i];
        } else if x[i] > hi[i] {
            x[i] = hi[i];
        }
    }
}

fn projected_grad(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    lo: &Array1<f64>,
    hi: &Array1<f64>,
) -> Array1<f64> {
    let mut pg = g.to_owned();
    for i in 0..x.len() {
        if x[i] <= lo[i] {
            pg[i] = pg[i].min(0.0);
        } else if x[i] >= hi[i] {
            pg[i] = pg[i].max(0.0);
        }
    }
    pg
}

fn lbfgs_direction(
    g: &Array1<f64>,
    s_hist: &[Array1<f64>],
    y_hist: &[Array1<f64>],
    rho_hist: &[f64],
) -> Array1<f64> {
    let mut q = g.clone();
    let m = s_hist.len();
    let mut alpha = vec![0.0; m];
    for i in (0..m).rev() {
        alpha[i] = rho_hist[i] * dot(s_hist[i].view(), q.view());
        axpy(-alpha[i], y_hist[i].view(), &mut q);
    }
    if m > 0 {
        let yy = dot(y_hist[m - 1].view(), y_hist[m - 1].view());
        if yy > 1e-18 {
            let gamma = dot(s_hist[m - 1].view(), y_hist[m - 1].view()) / yy;
            q.mapv_inplace(|v| v * gamma);
        }
    }
    for i in 0..m {
        let beta = rho_hist[i] * dot(y_hist[i].view(), q.view());
        axpy(alpha[i] - beta, s_hist[i].view(), &mut q);
    }
    q.mapv_inplace(|v| -v);
    q
}

fn chisq(
    set: &PairSet,
    ndof: usize,
    ftrue: &Array1<f64>,
    pars: ArrayView1<f64>,
) -> Result<f64, SaddleError> {
    let (chi, _) = objective_grad(set, ndof, ftrue, pars)?;
    Ok(chi)
}

fn objective_grad(
    set: &PairSet,
    ndof: usize,
    ftrue: &Array1<f64>,
    pars: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let (lin, jac, dfn) = jac_and_fit(set, ndof, pars, ftrue)?;
    let mut pred = Array1::zeros(ndof);
    for p in 0..lin.len() {
        axpy(lin[p], jac.column(p), &mut pred);
    }
    let mut d = pred;
    axpy(-1.0, ftrue.view(), &mut d);
    let chi = dot(d.view(), d.view());
    let nnonlin = pars.len();
    let nlin = lin.len();
    let mut dchi = Array1::zeros(nnonlin);
    for q in 0..nnonlin {
        let mut acc = 0.0;
        for l in 0..nlin {
            let mut col = Array1::zeros(ndof);
            for i in 0..ndof {
                col[i] = dfn[(i, q, l)];
            }
            acc += lin[l] * dot(d.view(), col.view());
        }
        dchi[q] = 2.0 * acc;
    }
    Ok((chi, dchi))
}

fn jac_and_fit(
    set: &PairSet,
    ndof: usize,
    pars: ArrayView1<f64>,
    ftrue: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>, Array3<f64>), SaddleError> {
    let nlin = set.nlin();
    let nnonlin = set.nnonlin();
    let mut jac = Array2::<f64>::zeros((ndof, nlin));
    let mut dfn = Array3::<f64>::zeros((ndof, nnonlin, nlin));
    let mut linstart = 0;
    let mut nonlinstart = 0;
    for inter in &set.lj {
        for p in &inter.pairs {
            lj_force(p, linstart, &mut jac);
        }
        linstart += 2;
    }
    for inter in &set.buck {
        let rho = pars[nonlinstart];
        for p in &inter.pairs {
            buck_force(p, linstart, nonlinstart, rho, &mut jac, &mut dfn);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.morse {
        let rho = pars[nonlinstart];
        for p in &inter.pairs {
            morse_force(p, linstart, nonlinstart, rho, &mut jac, &mut dfn);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.bond {
        let r0 = pars[nonlinstart];
        for p in &inter.pairs {
            bond_force(p, linstart, nonlinstart, r0, &mut jac, &mut dfn);
        }
        linstart += 1;
        nonlinstart += 1;
    }
    Ok((fit_linear(&jac, ftrue.clone()), jac, dfn))
}

fn assemble_hess(
    set: &PairSet,
    ndof: usize,
    lin: ArrayView1<f64>,
    nonlin: ArrayView1<f64>,
) -> Result<Array2<f64>, SaddleError> {
    let mut h = Array2::<f64>::zeros((ndof, ndof));
    let mut linstart = 0;
    let mut nonlinstart = 0;
    for inter in &set.lj {
        let c6 = lin[linstart];
        let c12 = lin[linstart + 1];
        for p in &inter.pairs {
            lj_hess(p, c6, c12, &mut h);
        }
        linstart += 2;
    }
    for inter in &set.buck {
        let a = lin[linstart];
        let c6 = lin[linstart + 1];
        let rho = nonlin[nonlinstart];
        for p in &inter.pairs {
            buck_hess(p, a, c6, rho, &mut h);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.morse {
        let a = lin[linstart];
        let b = lin[linstart + 1];
        let rho = nonlin[nonlinstart];
        for p in &inter.pairs {
            morse_hess(p, a, b, rho, &mut h);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.bond {
        let k = lin[linstart];
        let r0 = nonlin[nonlinstart];
        for p in &inter.pairs {
            sella_bond_hess(p, k, r0, &mut h);
        }
        linstart += 1;
        nonlinstart += 1;
    }
    Ok(h)
}

fn add_col(jac: &mut Array2<f64>, p: &Pair, scale: f64, col: usize) {
    for a in 0..3 {
        jac[(3 * p.i + a, col)] += scale * p.xij[a];
        jac[(3 * p.j + a, col)] -= scale * p.xij[a];
    }
}

fn add_dnonlin(dfn: &mut Array3<f64>, p: &Pair, scale: f64, q: usize, col: usize) {
    for a in 0..3 {
        dfn[(3 * p.i + a, q, col)] += scale * p.xij[a];
        dfn[(3 * p.j + a, q, col)] -= scale * p.xij[a];
    }
}

fn lj_force(p: &Pair, col: usize, jac: &mut Array2<f64>) {
    let r2 = dot3(p.xij);
    let r8 = r2.powi(4);
    let r14 = r8 * r2 * r2 * r2;
    add_col(jac, p, 6.0 / r8, col);
    add_col(jac, p, -12.0 / r14, col + 1);
}

fn buck_force(
    p: &Pair,
    col: usize,
    q: usize,
    rho: f64,
    jac: &mut Array2<f64>,
    dfn: &mut Array3<f64>,
) {
    let r2 = dot3(p.xij);
    let r = r2.sqrt();
    let r8 = r2.powi(4);
    let expterm = (-rho * r).exp();
    add_col(jac, p, -rho * expterm / r, col);
    add_col(jac, p, 6.0 / r8, col + 1);
    add_dnonlin(dfn, p, rho * expterm - expterm / r, q, col);
}

fn morse_force(
    p: &Pair,
    col: usize,
    q: usize,
    rho: f64,
    jac: &mut Array2<f64>,
    dfn: &mut Array3<f64>,
) {
    let r = dot3(p.xij).sqrt();
    let expterm = (-rho * r).exp();
    let exp2 = expterm * expterm;
    add_col(jac, p, -2.0 * rho * exp2 / r, col);
    add_col(jac, p, rho * expterm / r, col + 1);
    add_dnonlin(dfn, p, 2.0 * exp2 * (2.0 * rho * r - 1.0) / r, q, col);
    add_dnonlin(dfn, p, expterm * (1.0 - rho * r) / r, q, col + 1);
}

fn bond_force(
    p: &Pair,
    col: usize,
    q: usize,
    r0: f64,
    jac: &mut Array2<f64>,
    dfn: &mut Array3<f64>,
) {
    let r = dot3(p.xij).sqrt();
    add_col(jac, p, 2.0 * (r - r0) / r, col);
    add_dnonlin(dfn, p, -2.0 / r, q, col);
}

fn lj_hess(p: &Pair, c6: f64, c12: f64, h: &mut Array2<f64>) {
    let r2 = dot3(p.xij);
    let r8 = r2.powi(4);
    let r10 = r8 * r2;
    let r14 = r10 * r2 * r2;
    let r16 = r8 * r8;
    let diag = -12.0 * c12 / r14 + 6.0 * c6 / r8;
    let rest = 168.0 * c12 / r16 - 48.0 * c6 / r10;
    accumulate_pair(h, p.i, p.j, p.xij, diag, rest);
}

fn buck_hess(p: &Pair, a: f64, c6: f64, b: f64, h: &mut Array2<f64>) {
    let r2 = dot3(p.xij);
    let r = r2.sqrt();
    let r8 = r2.powi(4);
    let r10 = r8 * r2;
    let expterm = (-b * r).exp();
    let diag = 6.0 * c6 / r8 - a * b * expterm / r;
    let rest = -48.0 * c6 / r10 + a * b * expterm / (r * r * r) + a * b * b * expterm / r2;
    accumulate_pair(h, p.i, p.j, p.xij, diag, rest);
}

fn morse_hess(p: &Pair, a: f64, b: f64, rho: f64, h: &mut Array2<f64>) {
    let r2 = dot3(p.xij);
    let r = r2.sqrt();
    let expterm = (-rho * r).exp();
    let exp2 = expterm * expterm;
    let diag = (b * expterm - 2.0 * a * exp2) / r;
    let rest =
        rho * ((2.0 * a * exp2 - b * expterm) / r + rho * (4.0 * a * exp2 - b * expterm)) / r2;
    accumulate_pair(h, p.i, p.j, p.xij, diag, rest);
}

fn sella_bond_hess(p: &Pair, k: f64, r0: f64, h: &mut Array2<f64>) {
    let r2 = dot3(p.xij);
    let r = r2.sqrt();
    let diag = 2.0 * k * (r - r0) / r;
    let rest = 2.0 * k * r0 / (r2 * r);
    accumulate_pair(h, p.i, p.j, p.xij, diag, rest);
}

fn accumulate_pair(h: &mut Array2<f64>, i: usize, j: usize, xij: [f64; 3], diag: f64, rest: f64) {
    for a in 0..3 {
        for b in 0..3 {
            let hab = if a == b { diag } else { 0.0 } + rest * xij[a] * xij[b];
            h[(3 * i + a, 3 * i + b)] += hab;
            h[(3 * j + a, 3 * j + b)] += hab;
            h[(3 * i + a, 3 * j + b)] -= hab;
            h[(3 * j + a, 3 * i + b)] -= hab;
        }
    }
}

fn pair_force_col(
    x: ArrayView1<f64>,
    z: &[u8],
    pair: [usize; 2],
) -> Result<(Array1<f64>, f64), SaddleError> {
    let i = pair[0];
    let j = pair[1];
    let (r, dx) = pair_disp(x, i, j, [0.0, 0.0, 0.0])?;
    let r0 = covalent_radius(z[i]) + covalent_radius(z[j]);
    let factor = r - r0;
    let mut col = Array1::zeros(x.len());
    for a in 0..3 {
        let u = dx[a] / r;
        col[3 * i + a] = factor * u;
        col[3 * j + a] = -factor * u;
    }
    Ok((col, r))
}

fn pair_disp(
    x: ArrayView1<f64>,
    i: usize,
    j: usize,
    tvec: [f64; 3],
) -> Result<(f64, [f64; 3]), SaddleError> {
    let dx = [
        x[3 * j] - x[3 * i] + tvec[0],
        x[3 * j + 1] - x[3 * i + 1] + tvec[1],
        x[3 * j + 2] - x[3 * i + 2] + tvec[2],
    ];
    let r2 = dot3(dx);
    if r2 <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("force_match bond"));
    }
    Ok((r2.sqrt(), dx))
}

fn dot3(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn force_target(g: ArrayView1<f64>) -> Array1<f64> {
    let mut target = g.to_owned();
    target.mapv_inplace(|v| -v);
    target
}

fn fit_linear(j: &Array2<f64>, target: Array1<f64>) -> Array1<f64> {
    let nlin = j.ncols();
    let mut a = Array2::<f64>::zeros((nlin, nlin));
    let mut rhs = Array1::zeros(nlin);
    for i in 0..nlin {
        let ji = j.column(i);
        for k in 0..nlin {
            a[(i, k)] = dot(ji, j.column(k));
        }
        a[(i, i)] += 1e-12;
        rhs[i] = dot(ji, target.view());
    }
    solve_spd(&a, &rhs)
}

fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut acc = a[(i, j)];
            for k in 0..j {
                acc -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, i)] = acc.max(1e-30).sqrt();
            } else {
                l[(i, j)] = acc / l[(j, j)];
            }
        }
    }
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[(i, k)] * y[k];
        }
        y[i] = acc / l[(i, i)];
    }
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[(k, i)] * x[k];
        }
        x[i] = acc / l[(i, i)];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraints;
    use crate::internal::pack_cart;
    use ndarray::{Array1, Array2};

    #[test]
    fn water_pairs_are_the_two_oh_bonds() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let z = [8u8, 1, 1];
        let pairs = covalent_pairs(x.view(), &z, 1.3).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&[0, 1]));
        assert!(pairs.contains(&[0, 2]));
    }

    #[test]
    fn fitted_hessian_is_symmetric_and_finite() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0]);
        let g = Array1::from(vec![-0.2, 0.0, 0.0, 0.2, 0.0, 0.0]);
        let z = [6u8, 6];
        let (h, ks, pairs) = force_match_hessian(x.view(), g.view(), &z, 1.5).unwrap();
        assert_eq!(pairs.len(), 1);
        assert!(ks[0].is_finite());
        for i in 0..6 {
            for j in 0..6 {
                assert!((h[(i, j)] - h[(j, i)]).abs() < 1e-12);
                assert!(h[(i, j)].is_finite());
            }
        }
        let (col, _) = pair_force_col(x.view(), &z, pairs[0]).unwrap();
        let mut pred = Array1::zeros(6);
        axpy(ks[0], col.view(), &mut pred);
        let target = force_target(g.view());
        let mut d = pred;
        axpy(-1.0, target.view(), &mut d);
        let err = nrm2(d.view());
        assert!(err < 1e-8, "LS residual {err}");
    }

    #[test]
    fn physical_spring_k_and_bond_curvature_are_positive() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0]);
        let z = [6u8, 6];
        let r0 = covalent_radius(6) + covalent_radius(6);
        let fi = 1.0 * (1.1 - r0);
        let g = Array1::from(vec![-fi, 0.0, 0.0, fi, 0.0, 0.0]);
        let (h, ks, pairs) = force_match_hessian(x.view(), g.view(), &z, 1.5).unwrap();
        assert_eq!(pairs.len(), 1);
        assert!((ks[0] - 1.0).abs() < 1e-8, "k={}", ks[0]);
        assert!(h[(0, 0)] > 0.0, "H_ii along the bond {}", h[(0, 0)]);
    }

    #[test]
    fn default_buck_bond_hessian_is_symmetric_and_finite() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let g = Array1::from(vec![0.1, 0.0, 0.0, -0.05, 0.02, 0.0, -0.05, -0.02, 0.0]);
        let z = [8u8, 1, 1];
        let report = force_match(x.view(), g.view(), &z, &ForceMatchOpts::default()).unwrap();
        assert_eq!(report.hessian.nrows(), 9);
        for i in 0..9 {
            for j in 0..9 {
                assert!((report.hessian[(i, j)] - report.hessian[(j, i)]).abs() < 1e-10);
                assert!(report.hessian[(i, j)].is_finite());
            }
        }
        assert!(report.linpars.iter().all(|v| v.is_finite()));
        assert!(report.nonlinpars.iter().all(|v| v.is_finite()));
        assert!(!report.nonlinpars.is_empty());
    }

    #[test]
    fn lj_only_is_linear_and_finite() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0]);
        let g = Array1::from(vec![-0.1, 0.0, 0.0, 0.1, 0.0, 0.0]);
        let z = [10u8, 10];
        let opts = ForceMatchOpts {
            kinds: vec![PairKind::Lj],
            ..ForceMatchOpts::default()
        };
        let report = force_match(x.view(), g.view(), &z, &opts).unwrap();
        assert_eq!(report.nonlinpars.len(), 0);
        assert_eq!(report.linpars.len(), 2);
        assert!(report.linpars.iter().all(|v| v.is_finite()));
        assert!(report.hessian.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn morse_arm_hessian_is_finite() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![-0.3, 0.0, 0.0, 0.3, 0.0, 0.0]);
        let z = [6u8, 8];
        let opts = ForceMatchOpts {
            kinds: vec![PairKind::Morse],
            ..ForceMatchOpts::default()
        };
        let report = force_match(x.view(), g.view(), &z, &opts).unwrap();
        assert_eq!(report.nonlinpars.len(), 1);
        assert!(report.nonlinpars[0] >= 1.0);
        assert!(report.hessian.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lj_kernel_matches_sella_update_hess() {
        let mut h = Array2::<f64>::zeros((6, 6));
        let xij = [1.0_f64, 0.0, 0.0];
        let c6 = 1.0_f64;
        let c12 = 0.0_f64;
        let r2 = 1.0_f64;
        let r8 = r2.powi(4);
        let r10 = r8 * r2;
        let r14 = r10 * r2 * r2;
        let r16 = r8 * r8;
        let diag = -12.0 * c12 / r14 + 6.0 * c6 / r8;
        let rest = 168.0 * c12 / r16 - 48.0 * c6 / r10;
        accumulate_pair(&mut h, 0, 1, xij, diag, rest);
        assert!((diag - 6.0).abs() < 1e-14);
        assert!((rest + 48.0).abs() < 1e-14);
        assert!((h[(0, 0)] - (diag + rest)).abs() < 1e-12);
        assert!((h[(1, 1)] - diag).abs() < 1e-12);
        assert!((h[(0, 3)] + h[(0, 0)]).abs() < 1e-12);
    }

    #[test]
    fn force_match_on_com_stays_on_the_set() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        assert!(cons.residual_norm(x.view()).unwrap() < 1e-14);
        let g = Array1::from(vec![0.1, 0.0, 0.0, -0.05, 0.02, 0.0, -0.05, -0.02, 0.0]);
        let z = [8u8, 1, 1];
        let (y, report) =
            force_match_on(&cons, x.view(), g.view(), &z, &ForceMatchOpts::default()).unwrap();
        let res = cons.residual_norm(y.view()).unwrap();
        assert!(res < 1e-10, "force_match retract left the COM set: {res}");
        let v = Array1::from_elem(9, 0.2);
        let pv = project_force_match(&cons, &x, &v);
        let y2 = retract_force_match(&cons, &x, &pv, 1e-4);
        assert!(cons.residual_norm(y2.view()).unwrap() < 1e-10);
        let tv = transport_force_match(&cons, &x, &y2, &pv);
        assert_eq!(tv.len(), 9);
        assert!(report.hessian.iter().all(|v| v.is_finite()));
        let force = force_target(g.view());
        let step = newton_increment(&report.hessian, &force);
        let mut hs = Array1::zeros(9);
        for i in 0..9 {
            for j in 0..9 {
                hs[i] += report.hessian[(i, j)] * step[j];
            }
        }
        let mut resid = hs;
        axpy(-1.0, force.view(), &mut resid);
        assert!(
            nrm2(resid.view()) < 1e-4,
            "Newton increment does not use the seed Hessian"
        );
    }

    #[test]
    fn covalent_radius_matches_ase_cordero() {
        assert!((covalent_radius(21) - 1.70).abs() < 1e-12);
        assert!((covalent_radius(27) - 1.26).abs() < 1e-12);
        assert!((covalent_radius(96) - 1.69).abs() < 1e-12);
        assert!((covalent_radius(97) - 0.20).abs() < 1e-12);
        assert!((covalent_radius(6) - 0.76).abs() < 1e-12);
    }

    #[test]
    fn cutoff_equality_is_a_bond() {
        let r0 = covalent_radius(6) + covalent_radius(6);
        let r = 1.5 * r0;
        let x = Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]);
        let z = [6u8, 6];
        let pairs = covalent_pairs(x.view(), &z, 1.5).unwrap();
        assert_eq!(pairs, vec![[0, 1]]);
        let opts = ForceMatchOpts {
            kinds: vec![PairKind::Bond],
            bond_scale: 1.5,
            rcut_factor: 10.0,
            tvecs: vec![[0.0, 0.0, 0.0]],
        };
        let set = collect_pairs(x.view(), &z, &opts).unwrap();
        assert_eq!(set.bond.len(), 1);
    }

    #[test]
    fn one_atom_force_match_is_identity() {
        let x = Array1::from(vec![0.0, 0.0, 0.0]);
        let g = Array1::from(vec![0.1, 0.0, 0.0]);
        let z = [6u8];
        let report = force_match(x.view(), g.view(), &z, &ForceMatchOpts::default()).unwrap();
        assert_eq!(report.linpars.len(), 0);
        assert_eq!(report.hessian, Array2::eye(3));
    }

    #[test]
    fn rcut_uses_closest_tvec_image() {
        let x_far = Array1::from(vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        let x_near = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let g = Array1::from(vec![-0.1, 0.0, 0.0, 0.1, 0.0, 0.0]);
        let z = [10u8, 10];
        let far = ForceMatchOpts {
            kinds: vec![PairKind::Lj],
            tvecs: vec![[0.0, 0.0, 0.0], [-9.0, 0.0, 0.0]],
            ..ForceMatchOpts::default()
        };
        let near = ForceMatchOpts {
            kinds: vec![PairKind::Lj],
            tvecs: vec![[0.0, 0.0, 0.0]],
            ..ForceMatchOpts::default()
        };
        let h_far = force_match(x_far.view(), g.view(), &z, &far).unwrap();
        let h_near = force_match(x_near.view(), g.view(), &z, &near).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                assert!((h_far.hessian[(i, j)] - h_near.hessian[(i, j)]).abs() < 1e-10);
            }
        }
    }
}
