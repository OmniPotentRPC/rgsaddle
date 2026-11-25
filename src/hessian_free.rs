//! Matrix-free finite-difference Hessian actions, Lanczos lowest eigenpairs,
//! and stochastic Lanczos quadrature log-determinant ratios.
//!
//! Prefactor and saddle-index workflows only need Hessian-vector products
//! `H v` (or mass-weighted products `M^{-1/2} H M^{-1/2} v`) rather than
//! assembling a dense `3N x 3N` matrix:
//!
//! - [`lanczos_lowest_modes`] extracts the `k` lowest Ritz pairs by
//!   full-reorthogonalization Lanczos;
//! - [`slq_logdet_ratio`] estimates `logdet H_a - logdet H_b` using common
//!   Rademacher probe vectors across both operators (Ubaru, Chen, and Saad,
//!   SIAM J. Matrix Anal. Appl. 38, 1075 (2017));
//! - [`DeflatedAction`] projects out an orthonormal subspace and pins those
//!   directions to unit eigenvalue `+1.0`, computing `logdet'` without changing
//!   the positive-mode contributions.

use ndarray::{Array1, Array2, ArrayView1};

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::{FiniteDifference, PointSurface};

/// Matrix-free symmetric linear operator `y = H x`.
pub trait HessianAction {
    /// Dimension of the coordinate vector.
    fn dim(&self) -> usize;

    /// Apply the operator to `x`, returning `H x`.
    fn apply(&mut self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>;
}

/// Finite-difference Hessian-vector product over a gradient or force closure.
///
/// With `FiniteDifference::Central`, each application uses two evaluations
/// (`(g(x0 + h v) - g(x0 - h v)) / (2 h)`). With `FiniteDifference::Forward`,
/// the reference gradient `g(x0)` is cached on the first application and
/// subsequent products cost one evaluation. When per-coordinate masses are
/// attached via [`Self::with_masses_amu`], [`HessianAction::apply`] evaluates
/// the mass-weighted operator `M^{-1/2} H M^{-1/2} v`.
pub struct FdHessianAction<F> {
    x0: Array1<f64>,
    eval_grad: F,
    g0: Option<Array1<f64>>,
    inv_sqrt_masses: Option<Array1<f64>>,
    step: f64,
    difference: FiniteDifference,
    eval_count: usize,
}

impl<F> FdHessianAction<F>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    /// Construct an operator from a Cartesian gradient callback `g(x) = dE/dx`.
    pub fn new(
        x0: Array1<f64>,
        eval_grad: F,
        step: f64,
        difference: FiniteDifference,
    ) -> Result<Self, SaddleError> {
        if !step.is_finite() || step <= 0.0 {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        if x0.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("x0"));
        }
        Ok(Self {
            x0,
            eval_grad,
            g0: None,
            inv_sqrt_masses: None,
            step,
            difference,
            eval_count: 0,
        })
    }

    /// Attach per-coordinate atomic masses in unified atomic mass units (`u`)
    /// so [`HessianAction::apply`] returns `M^{-1/2} H M^{-1/2} x`.
    pub fn with_masses_amu(mut self, dof_masses_amu: &[f64]) -> Result<Self, SaddleError> {
        if dof_masses_amu.len() != self.x0.len() {
            return Err(SaddleError::Shape(format!(
                "mass length {} does not match dimension {}",
                dof_masses_amu.len(),
                self.x0.len()
            )));
        }
        let mut inv_sqrt = Array1::zeros(dof_masses_amu.len());
        for (i, &m) in dof_masses_amu.iter().enumerate() {
            if !m.is_finite() || m <= 0.0 {
                return Err(SaddleError::Invalid(format!(
                    "coordinate mass at index {i} must be positive and finite, got {m}"
                )));
            }
            inv_sqrt[i] = 1.0 / m.sqrt();
        }
        self.inv_sqrt_masses = Some(inv_sqrt);
        Ok(self)
    }

    /// Number of callback evaluations performed so far.
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Finite-difference step size.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// Finite-difference scheme.
    pub fn difference(&self) -> FiniteDifference {
        self.difference
    }

    fn call_grad(&mut self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        call_grad_with(&mut self.eval_grad, &mut self.eval_count, self.x0.len(), x)
    }
}

fn call_grad_with<F>(
    eval_grad: &mut F,
    eval_count: &mut usize,
    dim: usize,
    x: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let g = eval_grad(x)?;
    *eval_count += 1;
    if g.len() != dim {
        return Err(SaddleError::Shape(format!(
            "gradient length {} does not match dimension {dim}",
            g.len()
        )));
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("gradient"));
    }
    Ok(g)
}

/// Build an [`FdHessianAction`] from a Cartesian force callback `F(x) = -dE/dx`.
#[allow(clippy::type_complexity)]
pub fn fd_hessian_from_forces<F>(
    x0: Array1<f64>,
    mut eval_forces: F,
    step: f64,
    difference: FiniteDifference,
) -> Result<
    FdHessianAction<impl FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>>,
    SaddleError,
>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    FdHessianAction::new(x0, move |x| eval_forces(x).map(|f| -f), step, difference)
}

/// Build an [`FdHessianAction`] from a [`PointSurface`].
#[allow(clippy::type_complexity)]
pub fn fd_hessian_from_surface<S: PointSurface>(
    surface: &S,
    x0: Array1<f64>,
    step: f64,
    difference: FiniteDifference,
) -> Result<
    FdHessianAction<impl FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> + '_>,
    SaddleError,
> {
    FdHessianAction::new(
        x0,
        move |x| surface.eval(x).map(|(_, g)| g),
        step,
        difference,
    )
}

impl<F> HessianAction for FdHessianAction<F>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    fn dim(&self) -> usize {
        self.x0.len()
    }

    fn apply(&mut self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let n = self.x0.len();
        if x.len() != n {
            return Err(SaddleError::Shape(format!(
                "HessianAction input length {} does not match dimension {n}",
                x.len()
            )));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("HessianAction input"));
        }
        let u = match &self.inv_sqrt_masses {
            Some(inv_sqrt) => &x * inv_sqrt,
            None => x.to_owned(),
        };
        let norm = u.dot(&u).sqrt();
        if norm == 0.0 {
            return Ok(Array1::zeros(n));
        }
        let h = self.step / norm;
        let xp = &self.x0 + &(&u * h);
        let gp = self.call_grad(xp.view())?;
        let mut hv = match self.difference {
            FiniteDifference::Forward => {
                if self.g0.is_none() {
                    let g0 = call_grad_with(
                        &mut self.eval_grad,
                        &mut self.eval_count,
                        n,
                        self.x0.view(),
                    )?;
                    self.g0 = Some(g0);
                }
                let g0 = self.g0.as_ref().expect("g0 initialized");
                (&gp - g0) / h
            }
            FiniteDifference::Central => {
                let xm = &self.x0 - &(&u * h);
                let gm = self.call_grad(xm.view())?;
                (&gp - &gm) / (2.0 * h)
            }
        };
        if let Some(inv_sqrt) = &self.inv_sqrt_masses {
            hv *= inv_sqrt;
        }
        Ok(hv)
    }
}

/// Orthonormal basis of the numerical span of `vectors` in `R^dim`.
///
/// Each vector is pre-scaled by its maximum absolute entry before normalizing
/// so extreme magnitudes do not underflow or overflow. Columns with singular
/// values below `f64::EPSILON * max(dim, k) * sigma_max` are dropped.
pub fn orthonormal_basis(vectors: &[Array1<f64>], dim: usize) -> Result<Array2<f64>, SaddleError> {
    let mut cols: Vec<Array1<f64>> = Vec::with_capacity(vectors.len());
    for (idx, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            return Err(SaddleError::Shape(format!(
                "deflation vector {idx} has length {}, expected {dim}",
                v.len()
            )));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(SaddleError::NonFinite("deflation vector"));
        }
        let big = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if big == 0.0 {
            continue;
        }
        let mut w = v / big;
        let norm = w.dot(&w).sqrt();
        if norm > 0.0 {
            w /= norm;
            cols.push(w);
        }
    }
    if cols.is_empty() || dim == 0 {
        return Ok(Array2::zeros((dim, 0)));
    }
    let mut gram = Array2::<f64>::zeros((cols.len(), cols.len()));
    for i in 0..cols.len() {
        for j in i..cols.len() {
            let dot = cols[i].dot(&cols[j]);
            gram[(i, j)] = dot;
            gram[(j, i)] = dot;
        }
    }
    let (evals, evecs) = exact_eigh(gram.view())?;
    let sigma_max = evals
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(0.0)
        .sqrt();
    if sigma_max == 0.0 {
        return Ok(Array2::zeros((dim, 0)));
    }
    let tol = f64::EPSILON * dim.max(cols.len()) as f64 * sigma_max;
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for (j, &lambda) in evals.iter().enumerate().rev() {
        if lambda <= 0.0 || lambda.sqrt() <= tol {
            continue;
        }
        let mut q = Array1::<f64>::zeros(dim);
        for (i, col) in cols.iter().enumerate() {
            q.scaled_add(evecs[(i, j)], col);
        }
        for prev in &kept {
            let c = q.dot(prev);
            q.scaled_add(-c, prev);
        }
        let qnorm = q.dot(&q).sqrt();
        if qnorm > 1.0e-12 {
            q /= qnorm;
            kept.push(q);
        }
    }
    let mut basis = Array2::zeros((dim, kept.len()));
    for (j, q) in kept.iter().enumerate() {
        basis.column_mut(j).assign(q);
    }
    Ok(basis)
}

/// Symmetric operator wrapper that projects out an orthonormal subspace and
/// pins those directions to eigenvalue `+1.0` so they contribute `ln 1 = 0`
/// to log-determinant estimates.
pub struct DeflatedAction<A: HessianAction> {
    inner: A,
    basis: Array2<f64>,
}

impl<A: HessianAction> DeflatedAction<A> {
    /// Wrap `inner`, deflating the span of `directions`.
    pub fn new(inner: A, directions: &[Array1<f64>]) -> Result<Self, SaddleError> {
        let dim = inner.dim();
        let basis = orthonormal_basis(directions, dim)?;
        Ok(Self { inner, basis })
    }

    /// Borrow the underlying operator.
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Mutably borrow the underlying operator.
    pub fn inner_mut(&mut self) -> &mut A {
        &mut self.inner
    }

    /// Number of orthonormal deflated directions.
    pub fn rank(&self) -> usize {
        self.basis.ncols()
    }
}

impl<A: HessianAction> HessianAction for DeflatedAction<A> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn apply(&mut self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let n = self.inner.dim();
        if x.len() != n {
            return Err(SaddleError::Shape(format!(
                "DeflatedAction input length {} does not match dimension {n}",
                x.len()
            )));
        }
        if self.basis.ncols() == 0 {
            return self.inner.apply(x);
        }
        let coeffs = self.basis.t().dot(&x);
        let px = &x - &self.basis.dot(&coeffs);
        let raw_y = self.inner.apply(px.view())?;
        let y_coeffs = self.basis.t().dot(&raw_y);
        let mut y = &raw_y - &self.basis.dot(&y_coeffs);
        y += &self.basis.dot(&coeffs);
        Ok(y)
    }
}

/// Lowest Ritz eigenpairs extracted by full-reorthogonalization Lanczos.
#[derive(Clone, Debug)]
pub struct LanczosSpectrum {
    /// Ascending Ritz eigenvalues of length `k`.
    pub eigenvalues: Array1<f64>,
    /// Corresponding Ritz vectors of shape `(dim, k)` (columns).
    pub eigenvectors: Array2<f64>,
    /// Residual norms `||H v_i - theta_i v_i||` of length `k`.
    pub residuals: Array1<f64>,
    /// Operator applications consumed.
    pub n_applies: usize,
}

fn orthonormalize_against(v: &mut Array1<f64>, basis: &[Array1<f64>]) -> f64 {
    for _ in 0..2 {
        for b in basis {
            let c = v.dot(b);
            v.scaled_add(-c, b);
        }
    }
    let norm = v.dot(v).sqrt();
    if norm > 0.0 {
        *v /= norm;
    }
    norm
}

fn rademacher_unit(n: usize, seed: u64) -> Array1<f64> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut v = Array1::<f64>::zeros(n);
    for entry in &mut v {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        *entry = if z & 1 == 0 { 1.0 } else { -1.0 };
    }
    v / (n as f64).sqrt()
}

/// Extract the `k` lowest eigenpairs of a symmetric operator via Lanczos with
/// full two-pass Gram-Schmidt reorthogonalization.
pub fn lanczos_lowest_modes(
    op: &mut dyn HessianAction,
    k: usize,
    max_iters: usize,
    tol: f64,
    seed: u64,
) -> Result<LanczosSpectrum, SaddleError> {
    let n = op.dim();
    if k == 0 || n == 0 || k > n {
        return Err(SaddleError::Shape(format!(
            "cannot extract {k} lowest modes from dimension {n}"
        )));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(SaddleError::Invalid(format!(
            "Lanczos tolerance must be finite and non-negative, got {tol}"
        )));
    }
    let m_max = max_iters.min(n).max(k + 2).min(n);
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(m_max);
    let mut hv_cache: Vec<Array1<f64>> = Vec::with_capacity(m_max);
    let mut v = rademacher_unit(n, seed);
    orthonormalize_against(&mut v, &basis);
    let mut n_applies = 0usize;
    let mut best: Option<LanczosSpectrum> = None;

    while basis.len() < m_max {
        let hv = op.apply(v.view())?;
        if hv.len() != n {
            return Err(SaddleError::Shape(format!(
                "operator returned length {}, expected {n}",
                hv.len()
            )));
        }
        if hv.iter().any(|x| !x.is_finite()) {
            return Err(SaddleError::NonFinite("HessianAction output"));
        }
        n_applies += 1;
        basis.push(v);
        hv_cache.push(hv.clone());

        let mut w = hv;
        let bn = orthonormalize_against(&mut w, &basis);

        let m = basis.len();
        if m >= k {
            let mut t = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    t[(i, j)] = basis[i].dot(&hv_cache[j]);
                }
            }
            let (evals, evecs) = exact_eigh(t.view())?;
            let mut eigenvalues = Array1::<f64>::zeros(k);
            let mut eigenvectors = Array2::<f64>::zeros((n, k));
            let mut residuals = Array1::<f64>::zeros(k);

            for col in 0..k {
                let theta = evals[col];
                let mut ritz = Array1::<f64>::zeros(n);
                let mut hres = Array1::<f64>::zeros(n);
                for i in 0..m {
                    let c = evecs[(i, col)];
                    ritz.scaled_add(c, &basis[i]);
                    hres.scaled_add(c, &hv_cache[i]);
                }
                let diff = &hres - &(&ritz * theta);
                let res = diff.dot(&diff).sqrt();
                eigenvalues[col] = theta;
                eigenvectors.column_mut(col).assign(&ritz);
                residuals[col] = res;
            }
            let converged = residuals.iter().all(|&r| r < tol);
            best = Some(LanczosSpectrum {
                eigenvalues,
                eigenvectors,
                residuals,
                n_applies,
            });
            if converged {
                break;
            }
        }

        if bn < 1.0e-14 {
            break;
        }
        v = w;
    }

    best.ok_or_else(|| SaddleError::Solver("Lanczos did not reach minimum subspace rank".into()))
}

/// Stochastic Lanczos quadrature estimate of `logdet H_a - logdet H_b`.
#[derive(Clone, Debug)]
pub struct SlqLogDetRatio {
    /// Mean estimate of `logdet H_a - logdet H_b`.
    pub delta_logdet: f64,
    /// Per-probe paired difference estimates.
    pub per_probe: Vec<f64>,
    /// Standard error over probes.
    pub std_err: f64,
    /// Total operator applications consumed across both operators.
    pub n_applies: usize,
}

fn slq_probe(
    op: &mut dyn HessianAction,
    z: &Array1<f64>,
    m: usize,
    applies: &mut usize,
) -> Result<f64, SaddleError> {
    let n = op.dim();
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alphas = Vec::with_capacity(m);
    let mut betas = Vec::with_capacity(m);
    let mut v = z.clone();
    orthonormalize_against(&mut v, &basis);
    for _ in 0..m.min(n) {
        let hv = op.apply(v.view())?;
        if hv.len() != n {
            return Err(SaddleError::Shape(format!(
                "operator returned length {}, expected {n}",
                hv.len()
            )));
        }
        if hv.iter().any(|x| !x.is_finite()) {
            return Err(SaddleError::NonFinite("HessianAction output"));
        }
        *applies += 1;
        let alpha = v.dot(&hv);
        alphas.push(alpha);
        basis.push(v);
        let mut w = hv;
        let beta = orthonormalize_against(&mut w, &basis);
        if beta < 1.0e-14 {
            break;
        }
        betas.push(beta);
        v = w;
    }
    let msz = alphas.len();
    let mut t = Array2::<f64>::zeros((msz, msz));
    for i in 0..msz {
        t[(i, i)] = alphas[i];
        if i + 1 < msz {
            t[(i, i + 1)] = betas[i];
            t[(i + 1, i)] = betas[i];
        }
    }
    let (evals, evecs) = exact_eigh(t.view())?;
    let mut acc = 0.0;
    for i in 0..msz {
        let theta = evals[i];
        if theta <= 0.0 {
            return Err(SaddleError::Solver(format!(
                "non-positive Ritz value {theta} in SLQ quadrature"
            )));
        }
        let w = evecs[(0, i)];
        acc += w * w * theta.ln();
    }
    Ok(n as f64 * acc)
}

/// Estimate `logdet H_a - logdet H_b` by stochastic Lanczos quadrature with
/// common Rademacher probes at both operators.
///
/// Both operators must be positive definite on their subspace (deflate any
/// unstable or rigid modes first with [`DeflatedAction`]).
pub fn slq_logdet_ratio(
    op_a: &mut dyn HessianAction,
    op_b: &mut dyn HessianAction,
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Result<SlqLogDetRatio, SaddleError> {
    let n = op_a.dim();
    if n == 0 || n != op_b.dim() {
        return Err(SaddleError::Shape(format!(
            "operator dimensions ({n}, {}) must match and be positive",
            op_b.dim()
        )));
    }
    if n_probes == 0 || lanczos_steps == 0 {
        return Err(SaddleError::Invalid(
            "n_probes and lanczos_steps must be positive".into(),
        ));
    }
    let mut per_probe = Vec::with_capacity(n_probes);
    let mut applies = 0usize;
    for p in 0..n_probes {
        let z = rademacher_unit(n, seed.wrapping_add((p as u64).wrapping_mul(0x9E37)));
        let la = slq_probe(op_a, &z, lanczos_steps, &mut applies)?;
        let lb = slq_probe(op_b, &z, lanczos_steps, &mut applies)?;
        per_probe.push(la - lb);
    }
    let mean = per_probe.iter().sum::<f64>() / n_probes as f64;
    let var =
        per_probe.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_probes.max(2) - 1) as f64;
    Ok(SlqLogDetRatio {
        delta_logdet: mean,
        per_probe,
        std_err: (var / n_probes as f64).sqrt(),
        n_applies: applies,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    struct DiagOp {
        vals: Array1<f64>,
    }

    impl HessianAction for DiagOp {
        fn dim(&self) -> usize {
            self.vals.len()
        }

        fn apply(&mut self, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
            Ok(&self.vals * &x)
        }
    }

    #[test]
    fn lanczos_recovers_saddle_mode_and_gap() {
        let vals: Array1<f64> = std::iter::once(-1.7)
            .chain((1..30).map(|i| 0.5 + 0.1 * i as f64))
            .collect();
        let mut op = DiagOp { vals };
        let spec = lanczos_lowest_modes(&mut op, 2, 60, 1.0e-8, 7).unwrap();
        assert!((spec.eigenvalues[0] + 1.7).abs() < 1.0e-6);
        assert!((spec.eigenvalues[1] - 0.6).abs() < 1.0e-6);
        assert!(spec.residuals.iter().all(|&r| r < 1.0e-6));
    }

    #[test]
    fn fd_action_respects_masses_and_schemes() {
        let k = array![2.0, 5.0];
        let masses = [4.0, 9.0];
        let mut central = fd_hessian_from_forces(
            array![0.3, -0.2],
            |x| Ok(-(k.clone() * x)),
            1.0e-5,
            FiniteDifference::Central,
        )
        .unwrap()
        .with_masses_amu(&masses)
        .unwrap();
        let y0 = central.apply(array![1.0, 0.0].view()).unwrap();
        assert!((y0[0] - 2.0 / 4.0).abs() < 1.0e-7 && y0[1].abs() < 1.0e-9);
        assert_eq!(central.eval_count(), 2);

        let mut forward = FdHessianAction::new(
            array![0.3, -0.2],
            |x| Ok(&k * &x),
            1.0e-5,
            FiniteDifference::Forward,
        )
        .unwrap();
        let _ = forward.apply(array![1.0, 0.0].view()).unwrap();
        let _ = forward.apply(array![0.0, 1.0].view()).unwrap();
        // 1 central reference + 2 forward probes = 3 evals.
        assert_eq!(forward.eval_count(), 3);
    }

    #[test]
    fn deflated_slq_matches_exact_logdet_prime() {
        let base = array![0.7, 1.1, 1.9, 2.4, 3.3, 4.8];
        let mut sad = base.clone();
        sad[2] = -1.5;
        let mut op_min = DiagOp { vals: base.clone() };
        let op_sad = DiagOp { vals: sad };
        let e2 = array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut deflated = DeflatedAction::new(op_sad, &[e2]).unwrap();
        let res = slq_logdet_ratio(&mut op_min, &mut deflated, 4, 6, 42).unwrap();
        assert!((res.delta_logdet - base[2].ln()).abs() < 1.0e-6);
    }

    #[test]
    fn slq_rejects_non_positive_ritz_values() {
        let mut op_a = DiagOp {
            vals: array![1.0, 2.0, 3.0],
        };
        let mut op_b = DiagOp {
            vals: array![1.0, -0.5, 3.0],
        };
        assert!(slq_logdet_ratio(&mut op_a, &mut op_b, 2, 3, 1).is_err());
    }
}
