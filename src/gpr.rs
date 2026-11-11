//! Gaussian process with derivative observations for a saddle search.
//!
//! Goswami, Masterov, Kamath, Peña-Torres, and Jónsson, J. Chem. Theory
//! Comput. 21, 7935 (2025), doi:10.1021/acs.jctc.5c00866 (arXiv:2505.12519),
//! and the C++ `gpridge` dimer, build this surrogate and run the minimum-mode
//! search on it. The kernel and the derivative blocks follow Koistinen,
//! Ásgeirsson, Vehtari, and Jónsson, J. Chem. Theory Comput. 16, 499
//! (2020), doi:10.1021/acs.jctc.9b01038, and Koistinen, Dagbjartsdóttir,
//! Ásgeirsson, Vehtari, and Jónsson, J. Chem. Phys. 147, 152720 (2017),
//! doi:10.1063/1.4986787. A model surface with no pair distances uses the
//! squared exponential on the coordinates (Denzel and Kästner, J. Chem.
//! Theory Comput. 14, 5777 (2018), doi:10.1021/acs.jctc.8b00708).
//!
//! A geometry of `N` free atoms is compared through
//! `φ_p = 1 / r_p`. The covariance is
//! `σ_f² exp(-½ Σ_p ((φ_p - φ_p') / l)²)`. Energies and Cartesian
//! gradients are observed together, so the training matrix is the
//! Jacobian of that kernel. A constant mean is profiled out of the
//! marginal likelihood. The signal variance carries the Student-t
//! hyperprior used by `gpr_optim` (degrees of freedom 28, as in Goswami
//! and Jónsson) and the logarithmic barrier that keeps `σ_f²` below 2
//! in standardized units.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;

const PRIOR_NU: f64 = 28.0;
const PRIOR_S2: f64 = 1.0;
/// Upper bound on the signal variance in standardized units. The 2026
/// barrier used 2; that ceiling flattened the surrogate on these
/// surfaces, so the marginal likelihood may use a wider range.
const SIGNAL_VARIANCE_CEILING: f64 = 1.0e4;
const NOISE: f64 = 1e-8;

/// One electronic-structure evaluation: energy and its Cartesian gradient.
#[derive(Clone, Debug)]
pub struct Observation {
    pub position: Array1<f64>,
    pub energy: f64,
    pub gradient: Array1<f64>,
}

/// Covariance geometry.
#[derive(Clone, Debug)]
pub enum KernelKind {
    /// Squared exponential on the coordinates. Müller-Brown and any
    /// other surface that is not a set of interatomic distances.
    Cartesian,
    /// Squared exponential on inverse distances. `cell` is an
    /// orthorhombic minimum image; `fixed` atoms (flat xyz) enter the
    /// distances and not the gradient.
    InverseDistance {
        cell: Option<[f64; 3]>,
        fixed: Option<Array1<f64>>,
    },
}

/// How many coordinate sweeps the marginal-likelihood fit may take.
#[derive(Clone, Debug)]
pub struct FitOptions {
    pub passes: usize,
    pub noise: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            passes: 8,
            noise: NOISE,
        }
    }
}

#[derive(Clone, Debug)]
struct Features {
    phi: Vec<f64>,
    jac_ptr: Vec<usize>,
    jac_col: Vec<usize>,
    jac_val: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct Layout {
    dim: usize,
    n_free: usize,
    fixed: Vec<f64>,
    cell: Option<[f64; 3]>,
    pairs: Vec<(usize, usize)>,
    cartesian: bool,
}

impl Layout {
    pub(crate) fn new(dim: usize, kind: &KernelKind) -> Result<Self, SaddleError> {
        if dim == 0 {
            return Err(SaddleError::Invalid(
                "a Gaussian process needs a positive dimension".into(),
            ));
        }
        match kind {
            KernelKind::Cartesian => Ok(Self {
                dim,
                n_free: 0,
                fixed: Vec::new(),
                cell: None,
                pairs: Vec::new(),
                cartesian: true,
            }),
            KernelKind::InverseDistance { cell, fixed } => {
                if !dim.is_multiple_of(3) {
                    return Err(SaddleError::Invalid(
                        "inverse-distance kernel needs Cartesian xyz coordinates".into(),
                    ));
                }
                let fixed = match fixed {
                    Some(values) => {
                        if !values.len().is_multiple_of(3) || !values.iter().all(|v| v.is_finite())
                        {
                            return Err(SaddleError::Invalid(
                                "fixed atoms must be finite xyz coordinates".into(),
                            ));
                        }
                        values.to_vec()
                    }
                    None => Vec::new(),
                };
                if let Some(lengths) = cell
                    && lengths.iter().any(|l| !(l.is_finite() && *l > 0.0))
                {
                    return Err(SaddleError::Invalid(
                        "cell lengths must be positive and finite".into(),
                    ));
                }
                let n_free = dim / 3;
                let n_fixed = fixed.len() / 3;
                let mut pairs = Vec::new();
                for i in 0..n_free {
                    for j in (i + 1)..n_free {
                        pairs.push((i, j));
                    }
                    for j in 0..n_fixed {
                        pairs.push((i, n_free + j));
                    }
                }
                // A single free atom with no environment has no distances.
                // The caller treats that as a Cartesian surface.
                Ok(Self {
                    dim,
                    n_free,
                    fixed,
                    cell: *cell,
                    pairs,
                    cartesian: false,
                })
            }
        }
    }

    pub(crate) fn cartesian(&self) -> bool {
        self.cartesian
    }

    pub(crate) fn has_pairs(&self) -> bool {
        !self.pairs.is_empty()
    }

    fn xyz(&self, free: &[f64], atom: usize) -> [f64; 3] {
        if atom < self.n_free {
            let i = 3 * atom;
            [free[i], free[i + 1], free[i + 2]]
        } else {
            let i = 3 * (atom - self.n_free);
            [self.fixed[i], self.fixed[i + 1], self.fixed[i + 2]]
        }
    }

    fn separation(&self, free: &[f64], i: usize, j: usize) -> [f64; 4] {
        let a = self.xyz(free, i);
        let b = self.xyz(free, j);
        let mut d = [0.0; 3];
        let mut r2 = 0.0;
        for k in 0..3 {
            d[k] = a[k] - b[k];
            if let Some(cell) = self.cell
                && cell[k] > 0.0
            {
                d[k] -= cell[k] * (d[k] / cell[k]).round();
            }
            r2 += d[k] * d[k];
        }
        let r = r2.sqrt().max(1e-8);
        [d[0], d[1], d[2], r]
    }

    pub(crate) fn distance(&self, free: &[f64], i: usize, j: usize) -> f64 {
        self.separation(free, i, j)[3]
    }

    /// Shortest pair distance, for the nearest-neighbour step cap.
    pub(crate) fn minimum_distance(&self, free: &[f64]) -> Option<f64> {
        if self.pairs.is_empty() {
            return None;
        }
        let mut best = f64::INFINITY;
        for &(i, j) in &self.pairs {
            best = best.min(self.distance(free, i, j));
        }
        Some(best)
    }

    /// 1D max-log gap, `max |log(r'/r)|`.
    ///
    /// Koistinen's window `2/3 < r'/r < 3/2` is this gap bounded by
    /// `ln(1.5)`. Goswami and Jónsson, Eq. 7, write the same quantity
    /// `max |log(r_ij(x') / r_ij(x))|` and note that it is the log-ratio
    /// of the inverse distances. A Cartesian surface has no pairs; the
    /// gap is then the Euclidean distance, in the same role.
    pub(crate) fn gap(&self, a: &[f64], b: &[f64]) -> f64 {
        if self.cartesian || self.pairs.is_empty() {
            let mut sum = 0.0;
            for i in 0..self.dim {
                let d = a[i] - b[i];
                sum += d * d;
            }
            return sum.sqrt();
        }
        let mut worst = 0.0_f64;
        for &(i, j) in &self.pairs {
            let ra = self.distance(a, i, j).max(1e-8);
            let rb = self.distance(b, i, j).max(1e-8);
            worst = worst.max((rb / ra).ln().abs());
        }
        worst
    }

    fn features(&self, free: &[f64]) -> Features {
        if self.cartesian || self.pairs.is_empty() {
            let n = free.len();
            let mut jac_ptr = Vec::with_capacity(n + 1);
            let mut jac_col = Vec::with_capacity(n);
            let mut jac_val = Vec::with_capacity(n);
            jac_ptr.push(0);
            for i in 0..n {
                jac_col.push(i);
                jac_val.push(1.0);
                jac_ptr.push(i + 1);
            }
            return Features {
                phi: free.to_vec(),
                jac_ptr,
                jac_col,
                jac_val,
            };
        }
        let n = self.pairs.len();
        let mut phi = Vec::with_capacity(n);
        let mut jac_ptr = Vec::with_capacity(n + 1);
        let mut jac_col = Vec::with_capacity(n * 6);
        let mut jac_val = Vec::with_capacity(n * 6);
        jac_ptr.push(0);
        for &(i, j) in &self.pairs {
            let sep = self.separation(free, i, j);
            let r = sep[3];
            phi.push(1.0 / r);
            let inv_r3 = 1.0 / (r * r * r);
            if i < self.n_free {
                for (k, component) in sep.iter().enumerate().take(3) {
                    jac_col.push(3 * i + k);
                    jac_val.push(-component * inv_r3);
                }
            }
            if j < self.n_free {
                for (k, component) in sep.iter().enumerate().take(3) {
                    jac_col.push(3 * j + k);
                    jac_val.push(component * inv_r3);
                }
            }
            jac_ptr.push(jac_col.len());
        }
        Features {
            phi,
            jac_ptr,
            jac_col,
            jac_val,
        }
    }
}

/// Largest absolute log-ratio of pair distances (of inverse distances).
pub fn max_log_pair_distance(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    cell: Option<[f64; 3]>,
    fixed: Option<ArrayView1<f64>>,
) -> Result<f64, SaddleError> {
    if x.len() != y.len() {
        return Err(SaddleError::Shape(
            "the two geometries must share a length".into(),
        ));
    }
    let fixed = fixed.map(|v| v.to_owned());
    let layout = Layout::new(x.len(), &KernelKind::InverseDistance { cell, fixed })?;
    Ok(layout.gap(x.as_slice().unwrap(), y.as_slice().unwrap()))
}

fn cov_block(a: &Features, b: &Features, amplitude2: f64, length: f64, dim: usize) -> Vec<f64> {
    let w = dim + 1;
    let mut block = vec![0.0; w * w];
    let nfeat = a.phi.len();
    let mut r2 = 0.0;
    let mut diff = vec![0.0; nfeat];
    for (slot, (left, right)) in diff.iter_mut().zip(a.phi.iter().zip(b.phi.iter())) {
        *slot = left - right;
        r2 += *slot * *slot;
    }
    let l2 = length * length;
    let k = amplitude2 * (-0.5 * r2 / l2).exp();
    block[0] = k;
    let mut u = vec![0.0; dim];
    let mut v = vec![0.0; dim];
    for (p, delta) in diff.iter().enumerate() {
        for t in a.jac_ptr[p]..a.jac_ptr[p + 1] {
            u[a.jac_col[t]] += delta * a.jac_val[t];
        }
        for t in b.jac_ptr[p]..b.jac_ptr[p + 1] {
            v[b.jac_col[t]] += delta * b.jac_val[t];
        }
    }
    for bcol in 0..dim {
        block[1 + bcol] = k * v[bcol] / l2;
    }
    for arow in 0..dim {
        block[(1 + arow) * w] = -k * u[arow] / l2;
    }
    let l4 = l2 * l2;
    for arow in 0..dim {
        for bcol in 0..dim {
            block[(1 + arow) * w + 1 + bcol] = -k * u[arow] * v[bcol] / l4;
        }
    }
    let gscale = k / l2;
    for p in 0..nfeat {
        for ta in a.jac_ptr[p]..a.jac_ptr[p + 1] {
            for tb in b.jac_ptr[p]..b.jac_ptr[p + 1] {
                let row = 1 + a.jac_col[ta];
                let col = 1 + b.jac_col[tb];
                block[row * w + col] += gscale * a.jac_val[ta] * b.jac_val[tb];
            }
        }
    }
    block
}

/// Joint covariance of `(energy, gradient)` at `x` with the same
/// observation at `y`. Row and column 0 are the energy. No noise.
pub fn observation_covariance(
    kind: &KernelKind,
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    amplitude2: f64,
    length: f64,
) -> Result<Array2<f64>, SaddleError> {
    if x.len() != y.len() {
        return Err(SaddleError::Shape(
            "covariance geometries must share a length".into(),
        ));
    }
    if !(amplitude2.is_finite() && amplitude2 > 0.0 && length.is_finite() && length > 0.0) {
        return Err(SaddleError::Invalid(
            "amplitude and length scale must be positive and finite".into(),
        ));
    }
    let layout = Layout::new(x.len(), kind)?;
    let fa = layout.features(x.as_slice().unwrap());
    let fb = layout.features(y.as_slice().unwrap());
    let block = cov_block(&fa, &fb, amplitude2, length, x.len());
    let w = x.len() + 1;
    Ok(Array2::from_shape_vec((w, w), block).expect("covariance block is square"))
}

struct Prepared {
    layout: Layout,
    feats: Vec<Features>,
    /// Standardized targets, energy slots still carrying the raw
    /// (mean-centered) value. The profiled mean is removed in the solve.
    y: Vec<f64>,
    mean: f64,
    scale: f64,
    dim: usize,
    noise: f64,
}

fn prepare(data: &[Observation], kind: &KernelKind, noise: f64) -> Result<Prepared, SaddleError> {
    if data.is_empty() {
        return Err(SaddleError::Invalid(
            "the Gaussian process needs at least one observation".into(),
        ));
    }
    let dim = data[0].position.len();
    let layout = Layout::new(dim, kind)?;
    if !noise.is_finite() || noise < 0.0 {
        return Err(SaddleError::Invalid(
            "observation noise must be finite and non-negative".into(),
        ));
    }
    let mut mean = 0.0;
    let mut g2 = 0.0;
    let mut ng = 0usize;
    for obs in data {
        if obs.position.len() != dim || obs.gradient.len() != dim {
            return Err(SaddleError::Shape(
                "observations must share the position length".into(),
            ));
        }
        if !obs.energy.is_finite()
            || !obs.position.iter().all(|v| v.is_finite())
            || !obs.gradient.iter().all(|v| v.is_finite())
        {
            return Err(SaddleError::NonFinite("Gaussian-process observation"));
        }
        mean += obs.energy;
        g2 += obs.gradient.dot(&obs.gradient);
        ng += dim;
    }
    mean /= data.len() as f64;
    let scale = (g2 / ng.max(1) as f64).sqrt().max(1e-6);
    let w = dim + 1;
    let mut y = vec![0.0; data.len() * w];
    let mut feats = Vec::with_capacity(data.len());
    for (i, obs) in data.iter().enumerate() {
        y[i * w] = (obs.energy - mean) / scale;
        for a in 0..dim {
            y[i * w + 1 + a] = obs.gradient[a] / scale;
        }
        feats.push(layout.features(obs.position.as_slice().unwrap()));
    }
    Ok(Prepared {
        layout,
        feats,
        y,
        mean,
        scale,
        dim,
        noise,
    })
}

struct Solved {
    alpha: Vec<f64>,
    constant: f64,
    evidence: f64,
}

fn assemble(prepared: &Prepared, amplitude2: f64, length: f64, jitter: f64) -> Option<Vec<f64>> {
    let dim = prepared.dim;
    let w = dim + 1;
    let n_obs = prepared.feats.len();
    let n = n_obs * w;
    let mut matrix = vec![0.0; n * n];
    for i in 0..n_obs {
        for j in i..n_obs {
            let block = cov_block(
                &prepared.feats[i],
                &prepared.feats[j],
                amplitude2,
                length,
                dim,
            );
            for a in 0..w {
                for b in 0..w {
                    let value = block[a * w + b];
                    matrix[(i * w + a) * n + (j * w + b)] = value;
                    matrix[(j * w + b) * n + (i * w + a)] = value;
                }
            }
        }
    }
    for i in 0..n {
        matrix[i * n + i] += jitter;
    }
    Some(matrix)
}

fn solve_spd(matrix: &[f64], rhs: &[&[f64]], n: usize) -> Option<(Vec<Vec<f64>>, f64)> {
    #[cfg(feature = "faer")]
    {
        solve_spd_faer(matrix, rhs, n)
    }
    #[cfg(not(feature = "faer"))]
    {
        solve_spd_rust(matrix, rhs, n)
    }
}

#[cfg(feature = "faer")]
fn solve_spd_faer(matrix: &[f64], rhs: &[&[f64]], n: usize) -> Option<(Vec<Vec<f64>>, f64)> {
    use faer::linalg::solvers::Solve;
    let mut mat = faer::Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            mat[(i, j)] = matrix[i * n + j];
        }
    }
    let chol = mat.llt(faer::Side::Lower).ok()?;
    let mut logdet = 0.0;
    {
        let lower = chol.L();
        for i in 0..n {
            let diag = lower[(i, i)];
            if !(diag.is_finite() && diag > 0.0) {
                return None;
            }
            logdet += 2.0 * diag.ln();
        }
    }
    let mut solved = Vec::with_capacity(rhs.len());
    for column in rhs {
        let mut right = faer::Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            right[(i, 0)] = column[i];
        }
        chol.solve_in_place(&mut right);
        solved.push((0..n).map(|i| right[(i, 0)]).collect());
    }
    Some((solved, logdet))
}

#[cfg(not(feature = "faer"))]
fn solve_spd_rust(matrix: &[f64], rhs: &[&[f64]], n: usize) -> Option<(Vec<Vec<f64>>, f64)> {
    let mut lower = matrix.to_vec();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = lower[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if j == i {
                if !(sum.is_finite() && sum > 0.0) {
                    return None;
                }
                lower[i * n + i] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    let mut logdet = 0.0;
    for i in 0..n {
        logdet += 2.0 * lower[i * n + i].ln();
    }
    let mut solved = Vec::with_capacity(rhs.len());
    for column in rhs {
        let mut z = vec![0.0; n];
        for i in 0..n {
            let mut sum = column[i];
            for k in 0..i {
                sum -= lower[i * n + k] * z[k];
            }
            z[i] = sum / lower[i * n + i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = z[i];
            for k in (i + 1)..n {
                sum -= lower[k * n + i] * x[k];
            }
            x[i] = sum / lower[i * n + i];
        }
        solved.push(x);
    }
    Some((solved, logdet))
}

fn penalty(amplitude2: f64, n_obs: usize) -> f64 {
    if !(amplitude2.is_finite() && amplitude2 > 0.0) {
        return f64::NEG_INFINITY;
    }
    let log_s = amplitude2.ln();
    let prior = -(PRIOR_NU / 2.0 + 1.0) * log_s - PRIOR_NU * PRIOR_S2 / (2.0 * amplitude2);
    let gap = SIGNAL_VARIANCE_CEILING.ln() - log_s;
    if gap <= 1e-8 {
        return f64::NEG_INFINITY;
    }
    let mu = (1e-4 + 1e-3 * n_obs as f64).min(0.5);
    prior - mu * gap.ln()
}

fn solve_posterior(prepared: &Prepared, amplitude2: f64, length: f64) -> Option<Solved> {
    if !(amplitude2.is_finite() && amplitude2 > 0.0 && length.is_finite() && length > 0.0) {
        return None;
    }
    let dim = prepared.dim;
    let w = dim + 1;
    let n_obs = prepared.feats.len();
    let n = n_obs * w;
    let mut jitter = prepared.noise.max(1e-12);
    for _ in 0..6 {
        let matrix = assemble(prepared, amplitude2, length, jitter)?;
        let mut one = vec![0.0; n];
        for i in 0..n_obs {
            one[i * w] = 1.0;
        }
        let Some((solved, logdet)) = solve_spd(&matrix, &[&prepared.y, &one], n) else {
            jitter *= 10.0;
            continue;
        };
        let alpha0 = &solved[0];
        let z = &solved[1];
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n_obs {
            num += alpha0[i * w];
            den += z[i * w];
        }
        let constant = if den.abs() > 1e-18 { num / den } else { 0.0 };
        if !constant.is_finite() {
            jitter *= 10.0;
            continue;
        }
        let mut alpha = alpha0.clone();
        for i in 0..n {
            alpha[i] -= constant * z[i];
        }
        let mut dot = 0.0;
        for i in 0..n {
            let residual = prepared.y[i] - constant * one[i];
            dot += residual * alpha[i];
        }
        if !dot.is_finite() || !logdet.is_finite() {
            jitter *= 10.0;
            continue;
        }
        let log_ml =
            -0.5 * dot - 0.5 * logdet - 0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();
        let evidence = log_ml + penalty(amplitude2, n_obs);
        if !evidence.is_finite() {
            jitter *= 10.0;
            continue;
        }
        return Some(Solved {
            alpha,
            constant,
            evidence,
        });
    }
    None
}

fn initial_length(feats: &[Features]) -> f64 {
    if feats.len() >= 2 {
        let mut dists = Vec::new();
        for window in feats.windows(2) {
            let mut r2 = 0.0;
            for (left, right) in window[0].phi.iter().zip(window[1].phi.iter()) {
                let d = left - right;
                r2 += d * d;
            }
            dists.push(r2.sqrt());
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        return dists[dists.len() / 2].max(1e-2);
    }
    let phi = &feats[0].phi;
    if phi.is_empty() {
        return 1.0;
    }
    let rms = (phi.iter().map(|p| p * p).sum::<f64>() / phi.len() as f64).sqrt();
    rms.max(0.2)
}

/// Fitted inverse-distance (or Cartesian) process.
#[derive(Clone, Debug)]
pub struct GaussianProcess {
    layout: Layout,
    feats: Vec<Features>,
    beta: Vec<Vec<f64>>,
    alpha_e: Vec<f64>,
    mean: f64,
    scale: f64,
    constant: f64,
    amplitude2: f64,
    length: f64,
    evidence: f64,
}

impl GaussianProcess {
    pub fn fit(
        data: &[Observation],
        kind: &KernelKind,
        options: &FitOptions,
    ) -> Result<Self, SaddleError> {
        let prepared = prepare(data, kind, options.noise)?;
        let mut amplitude2 = 1.0;
        let mut length = initial_length(&prepared.feats);
        let mut best = solve_posterior(&prepared, amplitude2, length)
            .map(|s| s.evidence)
            .unwrap_or(f64::NEG_INFINITY);
        let mut sweeps = 0usize;
        loop {
            if sweeps >= options.passes.max(1) + 4 {
                break;
            }
            sweeps += 1;
            let mut improved = false;
            for &(factor_a, factor_l) in &[
                (0.5, 1.0),
                (0.75, 1.0),
                (1.25, 1.0),
                (2.0, 1.0),
                (1.0, 0.5),
                (1.0, 0.75),
                (1.0, 1.25),
                (1.0, 2.0),
            ] {
                let trial_a = amplitude2 * factor_a;
                let trial_l = length * factor_l;
                let Some(solved) = solve_posterior(&prepared, trial_a, trial_l) else {
                    continue;
                };
                if solved.evidence > best + 1e-8 {
                    best = solved.evidence;
                    amplitude2 = trial_a;
                    length = trial_l;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        let solved = solve_posterior(&prepared, amplitude2, length).ok_or_else(|| {
            SaddleError::Solver("Gaussian-process covariance was not positive definite".into())
        })?;
        Ok(Self::from_solved(prepared, solved, amplitude2, length))
    }

    fn from_solved(prepared: Prepared, solved: Solved, amplitude2: f64, length: f64) -> Self {
        let dim = prepared.dim;
        let w = dim + 1;
        let mut alpha_e = Vec::with_capacity(prepared.feats.len());
        let mut beta = Vec::with_capacity(prepared.feats.len());
        for (i, feat) in prepared.feats.iter().enumerate() {
            alpha_e.push(solved.alpha[i * w]);
            let mut weights = vec![0.0; feat.phi.len()];
            for (p, weight) in weights.iter_mut().enumerate() {
                let mut acc = 0.0;
                for t in feat.jac_ptr[p]..feat.jac_ptr[p + 1] {
                    let component = solved.alpha[i * w + 1 + feat.jac_col[t]];
                    acc += feat.jac_val[t] * component;
                }
                *weight = acc;
            }
            beta.push(weights);
        }
        Self {
            layout: prepared.layout,
            feats: prepared.feats,
            beta,
            alpha_e,
            mean: prepared.mean,
            scale: prepared.scale,
            constant: solved.constant,
            amplitude2,
            length,
            evidence: solved.evidence,
        }
    }

    pub fn amplitude2(&self) -> f64 {
        self.amplitude2
    }

    pub fn length_scale(&self) -> f64 {
        self.length
    }

    pub fn evidence(&self) -> f64 {
        self.evidence
    }

    /// Predictive energy and Cartesian gradient of the posterior mean.
    pub fn predict(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if x.len() != self.layout.dim {
            return Err(SaddleError::Shape(
                "prediction point must match the training dimension".into(),
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("Gaussian-process query"));
        }
        let feat = self.layout.features(x.as_slice().unwrap());
        let l2 = self.length * self.length;
        let l4 = l2 * l2;
        let mut energy = self.constant;
        let mut gradient = vec![0.0; self.layout.dim];
        for (obs, (cached, beta)) in self.feats.iter().zip(self.beta.iter()).enumerate() {
            let nfeat = feat.phi.len();
            let mut r2 = 0.0;
            let mut s = 0.0;
            let mut diff = vec![0.0; nfeat];
            for p in 0..nfeat {
                diff[p] = feat.phi[p] - cached.phi[p];
                r2 += diff[p] * diff[p];
                s += diff[p] * beta[p];
            }
            let k = self.amplitude2 * (-0.5 * r2 / l2).exp();
            energy += k * self.alpha_e[obs] + (k / l2) * s;
            let mut u = vec![0.0; self.layout.dim];
            let mut projected = vec![0.0; self.layout.dim];
            for p in 0..nfeat {
                for t in feat.jac_ptr[p]..feat.jac_ptr[p + 1] {
                    let col = feat.jac_col[t];
                    let val = feat.jac_val[t];
                    u[col] += diff[p] * val;
                    projected[col] += val * beta[p];
                }
            }
            for a in 0..self.layout.dim {
                gradient[a] += -(k * u[a] / l2) * self.alpha_e[obs] - (k * u[a] / l4) * s
                    + (k / l2) * projected[a];
            }
        }
        if !energy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("Gaussian-process prediction"));
        }
        let energy = self.mean + self.scale * energy;
        for value in &mut gradient {
            *value *= self.scale;
        }
        Ok((energy, Array1::from(gradient)))
    }
}

/// Marginal likelihood plus the signal-variance prior and barrier, at a
/// fixed amplitude and length. The fitter maximises this.
pub fn fitting_evidence(
    data: &[Observation],
    kind: &KernelKind,
    amplitude2: f64,
    length: f64,
) -> Result<f64, SaddleError> {
    let prepared = prepare(data, kind, NOISE)?;
    Ok(solve_posterior(&prepared, amplitude2, length)
        .map(|s| s.evidence)
        .unwrap_or(f64::NEG_INFINITY))
}
