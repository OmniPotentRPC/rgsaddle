//! Sella `force_match.pyx`: pair force-field seed Hessian, in Rust.
//!
//! The Cython matcher fits Buckingham / Morse / LJ / bond terms to a
//! residual force (`types` default `buck` + `bond`). Linear
//! coefficients are a least-squares fit; nonlinear `rho` / `r0` are
//! boxed and refined. The bond arm ([`force_match_hessian`]) is the
//! covalent-pair shortcut: one `k_ij` at fixed `r0`. There is no
//! Python hot path. Reductions go through [`rgmin::vecops`] so `par`
//! applies. A host that needs the Newton increment on a set calls
//! [`force_match_on`] (`project` / `retract` / `transport`).

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot, nrm2};

use crate::error::SaddleError;

/// Default covalent radii (Å), Cordero table by Z when known, else 0.70.
pub fn covalent_radius(z: u8) -> f64 {
    match z {
        1 => 0.31,
        2 => 0.28,
        3 => 1.28,
        4 => 0.96,
        5 => 0.84,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        10 => 0.58,
        11 => 1.66,
        12 => 1.41,
        13 => 1.21,
        14 => 1.11,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        18 => 1.06,
        19 => 2.03,
        20 => 1.76,
        21 => 1.70,
        22 => 1.60,
        23 => 1.53,
        24 => 1.39,
        25 => 1.39,
        26 => 1.32,
        27 => 1.26,
        28 => 1.24,
        29 => 1.32,
        30 => 1.22,
        31 => 1.22,
        32 => 1.20,
        33 => 1.19,
        34 => 1.20,
        35 => 1.20,
        36 => 1.16,
        _ => 0.70,
    }
}

/// Pair potential kinds in Sella `force_match(atoms, types=...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairKind {
    Lj,
    Buck,
    Morse,
    Bond,
}

/// Sella default `types=['buck', 'bond']`.
pub const DEFAULT_KINDS: &[PairKind] = &[PairKind::Buck, PairKind::Bond];

/// Options for the full Cython matcher.
#[derive(Clone, Debug)]
pub struct ForceMatchOpts {
    /// Sella `types`. Default is `buck` then `bond`.
    pub kinds: Vec<PairKind>,
    /// Bond cutoff scale. Sella uses `1.5 * (rcov_i + rcov_j)`.
    pub bond_scale: f64,
    /// Cutoff as a multiple of the shortest pair distance. Sella uses 3.
    pub rcut_factor: f64,
    /// Optional lattice translations. Empty / `None` is the isolated molecule.
    pub tvecs: Vec<[f64; 3]>,
}

impl Default for ForceMatchOpts {
    fn default() -> Self {
        Self {
            kinds: DEFAULT_KINDS.to_vec(),
            bond_scale: 1.5,
            rcut_factor: 3.0,
            tvecs: vec![[0.0, 0.0, 0.0]],
        }
    }
}

/// Fitted seed Hessian plus the linear / nonlinear coefficients.
#[derive(Clone, Debug)]
pub struct ForceMatchReport {
    pub hessian: Array2<f64>,
    pub linpars: Array1<f64>,
    pub nonlinpars: Array1<f64>,
}

/// Pairs with `r < scale * (rcov_i + rcov_j)`.
pub fn covalent_pairs(x: ArrayView1<f64>, z: &[u8], scale: f64) -> Result<Vec<[usize; 2]>, SaddleError> {
    if x.len() % 3 != 0 || x.len() / 3 != z.len() {
        return Err(SaddleError::Shape(
            "force_match frame is not 3N with matching Z".into(),
        ));
    }
    let n = z.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * j] - x[3 * i];
            let dy = x[3 * j + 1] - x[3 * i + 1];
            let dz = x[3 * j + 2] - x[3 * i + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let cut = scale * (covalent_radius(z[i]) + covalent_radius(z[j]));
            if r > 1e-12 && r < cut {
                pairs.push([i, j]);
            }
        }
    }
    Ok(pairs)
}

/// Fit pair spring constants so `F_ff ≈ -g` in the least-squares sense.
///
/// Pair `i-j` contributes `k (r - r0) u` on `j` and the opposite on
/// `i`, with `r0 = rcov_i + rcov_j`.
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
    let mut j = Array2::<f64>::zeros((ndof, nlin));
    for (p, pair) in pairs.iter().enumerate() {
        let (col, _) = pair_force_col(x, z, *pair)?;
        for i in 0..ndof {
            j[(i, p)] = col[i];
        }
    }
    // Normal equations (J^T J) k = J^T (-g) through vecops.
    let mut a = Array2::<f64>::zeros((nlin, nlin));
    let mut rhs = Array1::zeros(nlin);
    let mut target = g.to_owned();
    target.mapv_inplace(|v| -v);
    for i in 0..nlin {
        let ji = j.column(i);
        for k in 0..nlin {
            a[(i, k)] = dot(ji, j.column(k));
        }
        a[(i, i)] += 1e-12;
        rhs[i] = dot(ji, target.view());
    }
    Ok(solve_spd(&a, &rhs))
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
    let n = x.len();
    let mut h = Array2::<f64>::zeros((n, n));
    for (p, pair) in pairs.iter().enumerate() {
        let i = pair[0];
        let j = pair[1];
        let dx = [
            x[3 * j] - x[3 * i],
            x[3 * j + 1] - x[3 * i + 1],
            x[3 * j + 2] - x[3 * i + 2],
        ];
        let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        if r2 <= f64::MIN_POSITIVE {
            return Err(SaddleError::NonFinite("force_match bond"));
        }
        let r = r2.sqrt();
        let r0 = covalent_radius(z[i]) + covalent_radius(z[j]);
        let k = ks[p];
        let u = [dx[0] / r, dx[1] / r, dx[2] / r];
        // H_ii = k [ uu^T + (r-r0)/r (I - uu^T) ]
        let stretch = (r - r0) / r;
        for a in 0..3 {
            for b in 0..3 {
                let hab = k * (u[a] * u[b] + stretch * ((a == b) as i32 as f64 - u[a] * u[b]));
                h[(3 * i + a, 3 * i + b)] += hab;
                h[(3 * j + a, 3 * j + b)] += hab;
                h[(3 * i + a, 3 * j + b)] -= hab;
                h[(3 * j + a, 3 * i + b)] -= hab;
            }
        }
    }
    Ok(h)
}

/// Fit `k` and return the seed Hessian in one call (bond arm).
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

/// Full Sella matcher: `types` default `buck` + `bond`.
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
    let nlin = 2 * set.lj.len() + 2 * set.buck.len() + 2 * set.morse.len() + set.bond.len();
    let nnonlin = set.buck.len() + set.morse.len() + set.bond.len();
    if nlin == 0 {
        return Ok(ForceMatchReport {
            hessian: Array2::eye(x.len()),
            linpars: Array1::zeros(0),
            nonlinpars: Array1::zeros(0),
        });
    }
    let (x0, lo, hi) = nonlin_guess(&set);
    let ftrue = {
        let mut t = g.to_owned();
        t.mapv_inplace(|v| -v);
        t
    };
    let nonlin = if nnonlin == 0 {
        Array1::zeros(0)
    } else {
        refine_nonlin(&set, x.len(), nlin, &ftrue, x0, &lo, &hi)?
    };
    let (linpars, _, _) = fill_jac_and_fit(&set, x.len(), nlin, nnonlin, nonlin.view(), &ftrue)?;
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
    let mut hg = Array1::zeros(x.len());
    for i in 0..x.len() {
        hg[i] = dot(report.hessian.row(i), g);
    }
    let x0 = x.to_owned();
    let v = project_force_match(man, &x0, &hg);
    let y = retract_force_match(man, &x0, &v, 1e-4);
    let _ = transport_force_match(man, &x0, &y, &v);
    Ok((y, report))
}

struct Pair {
    i: usize,
    j: usize,
    xij: [f64; 3],
}

struct Interaction {
    pairs: Vec<Pair>,
    rcov: f64,
}

struct PairSet {
    lj: Vec<Interaction>,
    buck: Vec<Interaction>,
    morse: Vec<Interaction>,
    bond: Vec<Interaction>,
}

fn collect_pairs(x: ArrayView1<f64>, z: &[u8], opts: &ForceMatchOpts) -> Result<PairSet, SaddleError> {
    let n = z.len();
    let tvecs = if opts.tvecs.is_empty() {
        vec![[0.0, 0.0, 0.0]]
    } else {
        opts.tvecs.clone()
    };
    let mut rmin = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * j] - x[3 * i];
            let dy = x[3 * j + 1] - x[3 * i + 1];
            let dz = x[3 * j + 2] - x[3 * i + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > 1e-12 && r < rmin {
                rmin = r;
            }
        }
    }
    if !rmin.is_finite() {
        return Err(SaddleError::NonFinite("force_match rmin"));
    }
    let rcut = opts.rcut_factor * rmin;
    let rcut2 = rcut * rcut;
    let do_lj = opts.kinds.contains(&PairKind::Lj);
    let do_buck = opts.kinds.contains(&PairKind::Buck);
    let do_morse = opts.kinds.contains(&PairKind::Morse);
    let do_bond = opts.kinds.contains(&PairKind::Bond);

    let mut lj_map: Vec<((u8, u8), Interaction)> = Vec::new();
    let mut buck_map: Vec<((u8, u8), Interaction)> = Vec::new();
    let mut morse_map: Vec<((u8, u8), Interaction)> = Vec::new();
    let mut bond_map: Vec<((u8, u8), Interaction)> = Vec::new();

    for i in 0..n {
        for j in i..n {
            let rij = [
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ];
            for t in &tvecs {
                if i == j && t[0] == 0.0 && t[1] == 0.0 && t[2] == 0.0 {
                    continue;
                }
                let xij = [rij[0] + t[0], rij[1] + t[1], rij[2] + t[2]];
                let dij2 = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2];
                if dij2 > rcut2 {
                    continue;
                }
                let dij = dij2.sqrt();
                let eij = if z[i] <= z[j] {
                    (z[i], z[j])
                } else {
                    (z[j], z[i])
                };
                let pair = Pair { i, j, xij };
                if do_lj {
                    push_pair(&mut lj_map, eij, pair.i, pair.j, pair.xij, 0.0);
                }
                if do_buck {
                    push_pair(&mut buck_map, eij, pair.i, pair.j, pair.xij, 0.0);
                }
                if do_morse {
                    push_pair(&mut morse_map, eij, pair.i, pair.j, pair.xij, 0.0);
                }
                if do_bond {
                    let rcov = covalent_radius(z[i]) + covalent_radius(z[j]);
                    if dij <= opts.bond_scale * rcov {
                        push_pair(&mut bond_map, eij, pair.i, pair.j, pair.xij, rcov);
                    }
                }
            }
        }
    }
    Ok(PairSet {
        lj: take_ints(lj_map),
        buck: take_ints(buck_map),
        morse: take_ints(morse_map),
        bond: take_ints(bond_map),
    })
}

fn push_pair(
    map: &mut Vec<((u8, u8), Interaction)>,
    eij: (u8, u8),
    i: usize,
    j: usize,
    xij: [f64; 3],
    rcov: f64,
) {
    if let Some((_, slot)) = map.iter_mut().find(|(k, _)| *k == eij) {
        slot.pairs.push(Pair { i, j, xij });
    } else {
        map.push((
            eij,
            Interaction {
                pairs: vec![Pair { i, j, xij }],
                rcov,
            },
        ));
    }
}

fn take_ints(map: Vec<((u8, u8), Interaction)>) -> Vec<Interaction> {
    map.into_iter().map(|(_, v)| v).collect()
}

fn nonlin_guess(set: &PairSet) -> (Array1<f64>, Vec<f64>, Vec<f64>) {
    let n = set.buck.len() + set.morse.len() + set.bond.len();
    let mut x0 = Array1::zeros(n);
    let mut lo = vec![0.0; n];
    let mut hi = vec![f64::INFINITY; n];
    let mut p = 0;
    for _ in &set.buck {
        x0[p] = 2.5;
        lo[p] = 0.0;
        hi[p] = 10.0;
        p += 1;
    }
    for _ in &set.morse {
        x0[p] = 2.5;
        lo[p] = 1.0;
        hi[p] = 10.0;
        p += 1;
    }
    for b in &set.bond {
        x0[p] = b.rcov.max(1e-3);
        lo[p] = 0.5 * b.rcov;
        hi[p] = 2.0 * b.rcov.max(1e-3);
        p += 1;
    }
    (x0, lo, hi)
}

fn refine_nonlin(
    set: &PairSet,
    ndof: usize,
    nlin: usize,
    ftrue: &Array1<f64>,
    x0: Array1<f64>,
    lo: &[f64],
    hi: &[f64],
) -> Result<Array1<f64>, SaddleError> {
    let nnonlin = x0.len();
    let mut best = x0.clone();
    let (mut best_f, _) = objective(set, ndof, nlin, nnonlin, best.view(), ftrue)?;
    if nnonlin < 5 {
        let ns = 6usize;
        let ngrid = ns.pow(nnonlin as u32);
        for idx in 0..ngrid {
            let mut trial = Array1::zeros(nnonlin);
            let mut rest = idx;
            for d in 0..nnonlin {
                let t = (rest % ns) as f64 / (ns as f64 - 1.0);
                rest /= ns;
                let h = if hi[d].is_finite() { hi[d] } else { lo[d] + 10.0 };
                trial[d] = lo[d] + t * (h - lo[d]);
            }
            let (f, _) = objective(set, ndof, nlin, nnonlin, trial.view(), ftrue)?;
            if f < best_f {
                best_f = f;
                best = trial;
            }
        }
    }
    // Projected Armijo steps on the boxed nonlinear parameters.
    for _ in 0..32 {
        let (f, grad) = objective(set, ndof, nlin, nnonlin, best.view(), ftrue)?;
        let gn = nrm2(grad.view());
        if gn < 1e-8 {
            break;
        }
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..12 {
            let mut trial = best.clone();
            axpy(-step, grad.view(), &mut trial);
            for d in 0..nnonlin {
                trial[d] = trial[d].clamp(lo[d], hi[d]);
            }
            let (ft, _) = objective(set, ndof, nlin, nnonlin, trial.view(), ftrue)?;
            if ft <= f - 1e-4 * step * gn * gn {
                best = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Ok(best)
}

fn objective(
    set: &PairSet,
    ndof: usize,
    nlin: usize,
    nnonlin: usize,
    pars: ArrayView1<f64>,
    ftrue: &Array1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let (linpars, dflin, dfnonlin) = fill_jac_and_fit(set, ndof, nlin, nnonlin, pars, ftrue)?;
    let mut fapprox = Array1::zeros(ndof);
    for k in 0..nlin {
        axpy(linpars[k], dflin.column(k), &mut fapprox);
    }
    let mut df = fapprox;
    axpy(-1.0, ftrue.view(), &mut df);
    let chisq = dot(df.view(), df.view());
    let mut dchisq = Array1::zeros(nnonlin);
    for p in 0..nnonlin {
        let mut acc = 0.0;
        for l in 0..nlin {
            let col = dfnonlin.slice(ndarray::s![.., p, l]);
            acc += linpars[l] * dot(df.view(), col);
        }
        dchisq[p] = 2.0 * acc;
    }
    Ok((chisq, dchisq))
}

fn fill_jac(
    set: &PairSet,
    ndof: usize,
    nlin: usize,
    nnonlin: usize,
    pars: ArrayView1<f64>,
) -> Result<(Array2<f64>, ndarray::Array3<f64>), SaddleError> {
    let natoms = ndof / 3;
    let mut dflin = Array2::<f64>::zeros((ndof, nlin));
    let mut dfnonlin = ndarray::Array3::<f64>::zeros((ndof, nnonlin, nlin));
    let mut linstart = 0;
    let mut nonlinstart = 0;
    for inter in &set.lj {
        for p in &inter.pairs {
            lj_jac(p, linstart, natoms, &mut dflin);
        }
        linstart += 2;
    }
    for inter in &set.buck {
        let rho = pars[nonlinstart];
        for p in &inter.pairs {
            buck_jac(p, linstart, nonlinstart, rho, natoms, &mut dflin, &mut dfnonlin);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.morse {
        let rho = pars[nonlinstart];
        for p in &inter.pairs {
            morse_jac(p, linstart, nonlinstart, rho, natoms, &mut dflin, &mut dfnonlin);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.bond {
        let r0 = pars[nonlinstart];
        for p in &inter.pairs {
            bond_jac(p, linstart, nonlinstart, r0, natoms, &mut dflin, &mut dfnonlin);
        }
        linstart += 1;
        nonlinstart += 1;
    }
    Ok((dflin, dfnonlin))
}

fn fill_jac_and_fit(
    set: &PairSet,
    ndof: usize,
    nlin: usize,
    nnonlin: usize,
    pars: ArrayView1<f64>,
    ftrue: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>, ndarray::Array3<f64>), SaddleError> {
    let (dflin, dfnonlin) = fill_jac(set, ndof, nlin, nnonlin, pars)?;
    let mut a = Array2::<f64>::zeros((nlin, nlin));
    let mut rhs = Array1::zeros(nlin);
    for i in 0..nlin {
        let ji = dflin.column(i);
        for k in 0..nlin {
            a[(i, k)] = dot(ji, dflin.column(k));
        }
        a[(i, i)] += 1e-12;
        rhs[i] = dot(ji, ftrue.view());
    }
    let linpars = solve_spd(&a, &rhs);
    Ok((linpars, dflin, dfnonlin))
}

fn pair_force_col(
    x: ArrayView1<f64>,
    z: &[u8],
    pair: [usize; 2],
) -> Result<(Array1<f64>, f64), SaddleError> {
    let i = pair[0];
    let j = pair[1];
    let dx = [
        x[3 * j] - x[3 * i],
        x[3 * j + 1] - x[3 * i + 1],
        x[3 * j + 2] - x[3 * i + 2],
    ];
    let r = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
    if r <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("force_match bond"));
    }
    let r0 = covalent_radius(z[i]) + covalent_radius(z[j]);
    let factor = r - r0;
    let mut col = Array1::zeros(x.len());
    for a in 0..3 {
        let u = dx[a] / r;
        col[3 * i + a] = -factor * u;
        col[3 * j + a] = factor * u;
    }
    Ok((col, r))
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

fn slot(atom: usize, k: usize, natoms: usize) -> usize {
    let _ = natoms;
    3 * atom + k
}

fn lj_jac(p: &Pair, linstart: usize, natoms: usize, dflin: &mut Array2<f64>) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij8 = dij2 * dij2 * dij2 * dij2;
    let dij14 = dij8 * dij2 * dij2 * dij2;
    let f6 = 6.0 / dij8;
    let f12 = -12.0 / dij14;
    for k in 0..3 {
        let si = slot(p.i, k, natoms);
        let sj = slot(p.j, k, natoms);
        dflin[(si, linstart)] += f6 * p.xij[k];
        dflin[(si, linstart + 1)] += f12 * p.xij[k];
        dflin[(sj, linstart)] -= f6 * p.xij[k];
        dflin[(sj, linstart + 1)] -= f12 * p.xij[k];
    }
}

fn buck_jac(
    p: &Pair,
    linstart: usize,
    nonlinstart: usize,
    b: f64,
    natoms: usize,
    dflin: &mut Array2<f64>,
    dfnonlin: &mut ndarray::Array3<f64>,
) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij = dij2.sqrt();
    let dij8 = dij2 * dij2 * dij2 * dij2;
    let expterm = (-b * dij).exp();
    let fexp = -b * expterm / dij;
    let db = b * expterm - expterm / dij;
    let f6 = 6.0 / dij8;
    for k in 0..3 {
        let si = slot(p.i, k, natoms);
        let sj = slot(p.j, k, natoms);
        dflin[(si, linstart)] += fexp * p.xij[k];
        dflin[(si, linstart + 1)] += f6 * p.xij[k];
        dflin[(sj, linstart)] -= fexp * p.xij[k];
        dflin[(sj, linstart + 1)] -= f6 * p.xij[k];
        dfnonlin[(si, nonlinstart, linstart)] += db * p.xij[k];
        dfnonlin[(sj, nonlinstart, linstart)] -= db * p.xij[k];
    }
}

fn morse_jac(
    p: &Pair,
    linstart: usize,
    nonlinstart: usize,
    rho: f64,
    natoms: usize,
    dflin: &mut Array2<f64>,
    dfnonlin: &mut ndarray::Array3<f64>,
) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij = dij2.sqrt();
    let expterm = (-rho * dij).exp();
    let expterm2 = expterm * expterm;
    let f_rep = -2.0 * rho * expterm2 / dij;
    let f_att = rho * expterm / dij;
    let drho_rep = 2.0 * expterm2 * (2.0 * rho * dij - 1.0) / dij;
    let drho_att = expterm * (1.0 - rho * dij) / dij;
    for k in 0..3 {
        let si = slot(p.i, k, natoms);
        let sj = slot(p.j, k, natoms);
        dflin[(si, linstart)] += f_rep * p.xij[k];
        dflin[(si, linstart + 1)] += f_att * p.xij[k];
        dflin[(sj, linstart)] -= f_rep * p.xij[k];
        dflin[(sj, linstart + 1)] -= f_att * p.xij[k];
        dfnonlin[(si, nonlinstart, linstart)] += drho_rep * p.xij[k];
        dfnonlin[(si, nonlinstart, linstart + 1)] += drho_att * p.xij[k];
        dfnonlin[(sj, nonlinstart, linstart)] -= drho_rep * p.xij[k];
        dfnonlin[(sj, nonlinstart, linstart + 1)] -= drho_att * p.xij[k];
    }
}

fn bond_jac(
    p: &Pair,
    linstart: usize,
    nonlinstart: usize,
    r0: f64,
    natoms: usize,
    dflin: &mut Array2<f64>,
    dfnonlin: &mut ndarray::Array3<f64>,
) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij = dij2.sqrt();
    let fbond = 2.0 * (dij - r0) / dij;
    let dr0 = -2.0 / dij;
    for k in 0..3 {
        let si = slot(p.i, k, natoms);
        let sj = slot(p.j, k, natoms);
        dflin[(si, linstart)] += fbond * p.xij[k];
        dflin[(sj, linstart)] -= fbond * p.xij[k];
        dfnonlin[(si, nonlinstart, linstart)] += dr0 * p.xij[k];
        dfnonlin[(sj, nonlinstart, linstart)] -= dr0 * p.xij[k];
    }
}

fn assemble_hess(
    set: &PairSet,
    ndof: usize,
    linpars: ArrayView1<f64>,
    nonlin: ArrayView1<f64>,
) -> Result<Array2<f64>, SaddleError> {
    let natoms = ndof / 3;
    let mut hess = Array2::<f64>::zeros((ndof, ndof));
    let mut linstart = 0;
    let mut nonlinstart = 0;
    for inter in &set.lj {
        let c6 = linpars[linstart];
        let c12 = linpars[linstart + 1];
        for p in &inter.pairs {
            lj_hess(p, c6, c12, natoms, &mut hess);
        }
        linstart += 2;
    }
    for inter in &set.buck {
        let a = linpars[linstart];
        let c6 = linpars[linstart + 1];
        let rho = nonlin[nonlinstart];
        for p in &inter.pairs {
            buck_hess(p, a, c6, rho, natoms, &mut hess);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.morse {
        let a = linpars[linstart];
        let b = linpars[linstart + 1];
        let rho = nonlin[nonlinstart];
        for p in &inter.pairs {
            morse_hess(p, a, b, rho, natoms, &mut hess);
        }
        linstart += 2;
        nonlinstart += 1;
    }
    for inter in &set.bond {
        let k = linpars[linstart];
        let r0 = nonlin[nonlinstart];
        for p in &inter.pairs {
            sella_bond_hess(p, k, r0, natoms, &mut hess);
        }
        linstart += 1;
        nonlinstart += 1;
    }
    Ok(hess)
}

fn update_hess(p: &Pair, natoms: usize, hess: &mut Array2<f64>, diag: f64, rest: f64) {
    let _ = natoms;
    for k in 0..3 {
        let hessterm = diag + rest * p.xij[k] * p.xij[k];
        add_pair(hess, p.i, k, p.j, k, hessterm);
        for a in (k + 1)..3 {
            let hessterm = rest * p.xij[k] * p.xij[a];
            add_pair(hess, p.i, k, p.j, a, hessterm);
            add_pair(hess, p.i, a, p.j, k, hessterm);
        }
    }
}

fn add_pair(hess: &mut Array2<f64>, i: usize, k: usize, j: usize, a: usize, term: f64) {
    let ik = 3 * i + k;
    let ja = 3 * j + a;
    let ia = 3 * i + a;
    let jk = 3 * j + k;
    hess[(ik, ia)] += term;
    hess[(ik, ja)] -= term;
    hess[(jk, ia)] -= term;
    hess[(jk, ja)] += term;
}

fn lj_hess(p: &Pair, c6: f64, c12: f64, natoms: usize, hess: &mut Array2<f64>) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij8 = dij2 * dij2 * dij2 * dij2;
    let dij10 = dij8 * dij2;
    let dij14 = dij10 * dij2 * dij2;
    let dij16 = dij8 * dij8;
    let diag = -12.0 * c12 / dij14 + 6.0 * c6 / dij8;
    let rest = 168.0 * c12 / dij16 - 48.0 * c6 / dij10;
    update_hess(p, natoms, hess, diag, rest);
}

fn buck_hess(p: &Pair, a: f64, c6: f64, b: f64, natoms: usize, hess: &mut Array2<f64>) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij = dij2.sqrt();
    let dij3 = dij * dij2;
    let dij8 = dij2 * dij2 * dij2 * dij2;
    let dij10 = dij8 * dij2;
    let expterm = (-b * dij).exp();
    let diag = 6.0 * c6 / dij8 - a * b * expterm / dij;
    let rest = -48.0 * c6 / dij10 + a * b * expterm / dij3 + a * b * b * expterm / dij2;
    update_hess(p, natoms, hess, diag, rest);
}

fn morse_hess(p: &Pair, a: f64, b: f64, rho: f64, natoms: usize, hess: &mut Array2<f64>) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij = dij2.sqrt();
    let expterm = (-rho * dij).exp();
    let expterm2 = expterm * expterm;
    let diag = (b * expterm - 2.0 * a * expterm2) / dij;
    let rest = rho
        * ((2.0 * a * expterm2 - b * expterm) / dij + rho * (4.0 * a * expterm2 - b * expterm))
        / dij2;
    update_hess(p, natoms, hess, diag, rest);
}

fn sella_bond_hess(p: &Pair, k: f64, r0: f64, natoms: usize, hess: &mut Array2<f64>) {
    let dij2 = p.xij[0] * p.xij[0] + p.xij[1] * p.xij[1] + p.xij[2] * p.xij[2];
    let dij = dij2.sqrt();
    let dij3 = dij2 * dij;
    let diag = 2.0 * k * (dij - r0) / dij;
    let rest = 2.0 * k * r0 / dij3;
    update_hess(p, natoms, hess, diag, rest);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraints;
    use crate::internal::pack_cart;
    use ndarray::Array1;

    #[test]
    fn water_pairs_are_the_two_oh_bonds() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, // O
            0.96, 0.0, 0.0, // H
            -0.24, 0.93, 0.0, // H
        ]);
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
        // Residual force along the bond should be smaller than |g|.
        let (col, _) = pair_force_col(x.view(), &z, pairs[0]).unwrap();
        let mut pred = Array1::zeros(6);
        axpy(ks[0], col.view(), &mut pred);
        let mut target = g.clone();
        target.mapv_inplace(|v| -v);
        let mut d = pred.clone();
        axpy(-1.0, target.view(), &mut d);
        let err = nrm2(d.view());
        assert!(err < 1e-8, "LS residual {err}");
    }

    #[test]
    fn default_buck_bond_hessian_is_symmetric_and_finite() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0,
        ]);
        let g = Array1::from(vec![0.1, 0.0, 0.0, -0.05, 0.02, 0.0, -0.05, -0.02, 0.0]);
        let z = [8u8, 1, 1];
        let report = force_match(x.view(), g.view(), &z, &ForceMatchOpts::default()).unwrap();
        let n = 9;
        assert_eq!(report.hessian.nrows(), n);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (report.hessian[(i, j)] - report.hessian[(j, i)]).abs() < 1e-10,
                    "H[{i},{j}]"
                );
                assert!(report.hessian[(i, j)].is_finite());
            }
        }
        assert!(report.linpars.iter().all(|v| v.is_finite()));
        assert!(report.nonlinpars.iter().all(|v| v.is_finite()));
        assert!(!report.nonlinpars.is_empty(), "default types include bond/buck");
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
    fn force_match_on_com_stays_on_the_set() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        assert!(cons.residual_norm(x.view()).unwrap() < 1e-14);
        let g = Array1::from(vec![0.1, 0.0, 0.0, -0.05, 0.02, 0.0, -0.05, -0.02, 0.0]);
        let z = [8u8, 1, 1];
        let (y, report) = force_match_on(&cons, x.view(), g.view(), &z, &ForceMatchOpts::default())
            .unwrap();
        let res = cons.residual_norm(y.view()).unwrap();
        assert!(res < 1e-10, "force_match retract left the COM set: {res}");
        let v = Array1::from_elem(9, 0.2);
        let pv = project_force_match(&cons, &x, &v);
        let y2 = retract_force_match(&cons, &x, &pv, 1e-4);
        assert!(cons.residual_norm(y2.view()).unwrap() < 1e-10);
        let tv = transport_force_match(&cons, &x, &y2, &pv);
        assert_eq!(tv.len(), 9);
        assert!(report.hessian.iter().all(|v| v.is_finite()));
    }
}
