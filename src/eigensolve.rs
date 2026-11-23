//! Sella `eigensolvers.py`: exact dense diag, Rayleigh-Ritz, expand.
//!
//! `eig=true` on a Sella saddle walks a small Ritz subspace via
//! [`rayleigh_ritz_iter`] and [`expand`] (`jd0` default). The GPU
//! waist is dlpk CUDA ([`EigenDevice::Dlpk`] is the same Host path
//! until that feature is linked); this crate does not grow a second
//! device stack.
//!
//! The matrices are symmetric `n x n` arrays, in practice a Hessian in
//! eV/Angstrom^2 (the eigenvalues then carry that unit; the eigenvectors
//! are dimensionless unit columns). Subspace bases hold one vector per
//! column. Failures are [`SaddleError::Shape`] for incompatible array
//! shapes, [`SaddleError::Solver`] for a failed diagonalisation or an
//! empty subspace, and [`SaddleError::Singular`] for a singular expansion
//! system or factorization.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::vecops::{axpy, dot, nrm2};
use rgmin::{ApplyHessian, EigenParams, EigensolverKind, lowest_mode};

use crate::error::SaddleError;
use crate::linalg::{modified_gram_schmidt, symmetrize_vt_av};

/// Where the dense eigen runs. dlpk CUDA is the only device waist.
///
/// Both arms give the same spectrum; [`EigenDevice::Dlpk`] only changes which
/// rgmin route [`lowest_on`] takes and whether [`eigh_on`] consults the GPU policy
/// (see [`crate::gpu`]), and a build without a CUDA kernel runs on the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EigenDevice {
    #[default]
    /// Dense solver on the host (Jacobi, or faer above 128 dimensions). This is the
    /// default and ordinal 0.
    Host,
    /// rgmin `lowest_mode` (vecops / dlpk). Not a second GPU stack. Ordinal 1.
    Dlpk,
}

impl EigenDevice {
    /// C / Python ordinal: 0 for [`EigenDevice::Host`], 1 for [`EigenDevice::Dlpk`].
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Host => 0,
            Self::Dlpk => 1,
        }
    }

    /// Inverse of [`Self::to_abi`]; any ordinal other than 0 or 1 is `None`.
    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Host),
            1 => Some(Self::Dlpk),
            _ => None,
        }
    }
}

/// Sella `eigensolvers.expand` method: how [`expand`] turns the residual of the
/// sought Ritz pair into a new search direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpandKind {
    /// The residual itself (plain Lanczos). Ordinal 0.
    Lanczos,
    /// Generalised Davidson: solve `(P - shift B) t = r`. Ordinal 1.
    Gd,
    #[default]
    /// Jacobi-Davidson with the correction orthogonal to the sought Ritz vector,
    /// through an augmented solve. Sella's default and ordinal 2.
    Jd0,
    /// `Jd0` by projecting the preconditioned residual instead of augmenting the
    /// system. Ordinal 3.
    Jd0Alt,
    /// `Jd0` with the correction orthogonal to every Ritz vector of the subspace,
    /// through an augmented solve. Ordinal 4.
    Mjd0,
    /// `Mjd0` by projection instead of augmentation. Ordinal 5.
    Mjd0Alt,
}

impl ExpandKind {
    /// C / Python ordinal: 0 `Lanczos`, 1 `Gd`, 2 `Jd0`, 3 `Jd0Alt`, 4 `Mjd0`,
    /// 5 `Mjd0Alt`.
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Lanczos => 0,
            Self::Gd => 1,
            Self::Jd0 => 2,
            Self::Jd0Alt => 3,
            Self::Mjd0 => 4,
            Self::Mjd0Alt => 5,
        }
    }

    /// Inverse of [`Self::to_abi`]; any ordinal outside 0 to 5 is `None`.
    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Lanczos),
            1 => Some(Self::Gd),
            2 => Some(Self::Jd0),
            3 => Some(Self::Jd0Alt),
            4 => Some(Self::Mjd0),
            5 => Some(Self::Mjd0Alt),
            _ => None,
        }
    }
}

/// Dense `H v` for [`lowest_mode()`].
struct DenseApply<'a>(&'a Array2<f64>);

impl ApplyHessian for DenseApply<'_> {
    fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let n = self.0.nrows().min(v.len());
        let mut out = Array1::zeros(v.len());
        let vv = v.slice(s![..n]);
        for i in 0..n {
            out[i] = dot(self.0.row(i), vv);
        }
        out
    }
}

/// Cyclic Jacobi stays the dense solver through this dimension.
///
/// The LJ38 free block is 108, so that search keeps the Jacobi
/// spectrum. A larger block, including a Pt island free block of 525,
/// uses [`faer_eigh`] when the `faer` feature is on.
pub(crate) const DENSE_JACOBI_LIMIT: usize = 128;

/// Sella `exact`: dense symmetric eigen of `a`.
///
/// Eigenvalues come back in nondecreasing order. Dimension
/// `DENSE_JACOBI_LIMIT` (128) and below uses cyclic Jacobi. A larger
/// matrix uses faer's self-adjoint eigensolver. A build without the
/// `faer` feature, such as the Kani build, runs cyclic Jacobi at every
/// dimension.
///
/// `a` is a square matrix; it is symmetrised as `(a + a^T) / 2` first. Returns the
/// eigenvalues (length `n`) and the orthonormal eigenvectors as the columns of an
/// `n x n` array.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square; [`SaddleError::Solver`] when
/// the eigen solver does not converge.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::eigensolve::exact_eigh;
///
/// // Hessian of a saddle: curvature +2 along x and -1 along y, in eV/Angstrom^2.
/// let hessian = array![[2.0, 0.0], [0.0, -1.0]];
/// let (values, vectors) = exact_eigh(hessian.view())?;
/// assert!((values[0] + 1.0).abs() < 0.001);
/// assert!((values[1] - 2.0).abs() < 0.001);
/// // The lowest mode is along y.
/// assert!(vectors[(1, 0)].abs() > 0.99);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn exact_eigh(a: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() {
        return Err(SaddleError::Shape("eigh needs a square matrix".into()));
    }
    let n = a.nrows();
    let mut work = a.to_owned();
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (work[(i, j)] + work[(j, i)]);
            work[(i, j)] = s;
            work[(j, i)] = s;
        }
    }
    if n > DENSE_JACOBI_LIMIT {
        faer_eigh(&work)
    } else {
        jacobi_eigh(&mut work)
    }
}

/// Relative singularity, zero-mode, and backward eigenpair-residual threshold
/// `scale * (n as f64) * f64::EPSILON` for an `n x n` matrix of entry or
/// spectral magnitude `scale` (or `0.0` when `scale` is non-finite or
/// non-positive).
///
/// # Derivation
///
/// For an `n x n` matrix `A` with magnitude `scale = max_{i,j} |A_{ij}|`, each
/// component of `A v - lambda v` or `V^T A V` is an `n`-term inner product with
/// unit-vector weights `|V_{kj}| <= 1`. By standard floating-point inner-product
/// error analysis (Higham, *Accuracy and Stability of Numerical Algorithms*,
/// 2nd ed., Lemma 3.1; Wilkinson, *The Algebraic Eigenvalue Problem*, Ch. 3),
/// the backward perturbation `||Delta A||_2` across `n` dimensions is bounded to
/// first order by `scale * n * f64::EPSILON`. By Weyl's inequality
/// (`|tilde_lambda_k - lambda_k| <= ||Delta A||_2`) and Gershgorin's circle
/// theorem, `scale * n * f64::EPSILON` is the `n`-dimensional backward roundoff
/// threshold for zero modes, LU/Cholesky pivots, and full-matrix eigenpair
/// residuals.
///
/// At `n = 1`, `relative_zero_tolerance(scale, 1) = scale * f64::EPSILON` is a
/// single binary64 ULP at magnitude `scale`. In a Rayleigh-Ritz matrix `R` with
/// diagonal entries `O(scale)`, a Givens rotation annihilating `R_{pq}` shifts
/// the diagonal by `-t * R_{pq}` with `|t| <= 1`; when `|R_{pq}| <= scale * f64::EPSILON`,
/// that diagonal shift is at or below one binary64 ULP of `scale` and cannot
/// refine the floating-point eigenvalues further.
pub fn relative_zero_tolerance(scale: f64, n: usize) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale * (n as f64) * f64::EPSILON
    } else {
        0.0
    }
}

#[cfg(test)]
std::thread_local! {
    static TEST_JACOBI_SWEEP_BUDGET: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn with_jacobi_sweep_budget<T>(budget: usize, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<usize>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_JACOBI_SWEEP_BUDGET.with(|cell| cell.set(self.0));
        }
    }
    let prev = TEST_JACOBI_SWEEP_BUDGET.with(|cell| {
        let old = cell.get();
        cell.set(Some(budget));
        old
    });
    let _reset = Reset(prev);
    f()
}

/// Active-subspace cyclic Jacobi refinement on the `m x m` Rayleigh-Ritz
/// submatrix `r` (row stride `stride`), accumulating rotations in `q_col`
/// (`m x m` column-major).
///
/// # Derivation of tolerances and sweep budget
///
/// - **Stopping tolerance `res_tol = scale * n * f64::EPSILON`**: matches the
///   `n`-dimensional backward inner-product error bound of [`relative_zero_tolerance`].
/// - **Rotation threshold `rot_tol = scale * f64::EPSILON`**: any off-diagonal
///   entry `|R_{pq}| <= scale * f64::EPSILON` is within one binary64 ULP at
///   magnitude `scale`, so its quadratic Givens diagonal update `t * R_{pq}`
///   vanishes in binary64 arithmetic.
/// - **Sweep budget `max_sweeps = m` (`m = |S| >= 2`)**: `faer_eigh` invokes
///   this loop only after `faer::Mat::self_adjoint_eigen` and a Newton-Schulz
///   orthonormalization step, so `R = V_orth^T A V_orth` enters with relative
///   off-diagonal norm `||E||_F / scale << 1`, strictly inside the asymptotic
///   quadratic-convergence basin of Wilkinson's cyclic Jacobi theorem
///   (Wilkinson 1962, *Numer. Math.* 4, 296-300). Each cyclic sweep squares the
///   relative off-diagonal coupling across the `m - 1 < m` independent elimination
///   stages of the `m x m` active submatrix, so `m` sweeps (`m >= 2`) is an
///   upper bound for convergence from pre-diagonalized eigenpairs.
#[cfg(feature = "faer")]
fn cyclic_jacobi_subspace(
    r: &mut [f64],
    q_col: &mut [f64],
    m: usize,
    stride: usize,
    res_tol: f64,
    rot_tol: f64,
    max_sweeps: usize,
) -> Result<(), SaddleError> {
    if m <= 1 {
        return Ok(());
    }
    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        for p in 0..m {
            let row_p = &r[p * stride..p * stride + m];
            if !row_p[p].is_finite() {
                return Err(SaddleError::NonFinite("faer_eigh Jacobi diagonal"));
            }
            for &rpq in &row_p[(p + 1)..m] {
                if !rpq.is_finite() {
                    return Err(SaddleError::NonFinite("faer_eigh Jacobi off-diagonal"));
                }
                let mag = rpq.abs();
                if mag > max_off {
                    max_off = mag;
                }
            }
        }
        if max_off <= res_tol {
            return Ok(());
        }
        let mut changed = false;
        for p in 0..m {
            for q in (p + 1)..m {
                let rpq = r[p * stride + q];
                if rpq.abs() <= rot_tol {
                    r[p * stride + q] = 0.0;
                    r[q * stride + p] = 0.0;
                    continue;
                }
                changed = true;
                let rpp = r[p * stride + p];
                let rqq = r[q * stride + q];
                let tau = (rqq - rpp) / (2.0 * rpq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                r[p * stride + p] = rpp - t * rpq;
                r[q * stride + q] = rqq + t * rpq;
                r[p * stride + q] = 0.0;
                r[q * stride + p] = 0.0;
                if s.abs() * max_off > rot_tol {
                    for k in 0..m {
                        if k != p && k != q {
                            let rkp = r[p * stride + k];
                            let rkq = r[q * stride + k];
                            if rkp == 0.0 && rkq == 0.0 {
                                continue;
                            }
                            let nkp = c * rkp - s * rkq;
                            let nkq = s * rkp + c * rkq;
                            r[p * stride + k] = nkp;
                            r[k * stride + p] = nkp;
                            r[q * stride + k] = nkq;
                            r[k * stride + q] = nkq;
                        }
                    }
                }
                let (left, right) = q_col.split_at_mut(q * m);
                let qp_slice = &mut left[p * m..(p + 1) * m];
                let qq_slice = &mut right[..m];
                for (qp, qq) in qp_slice.iter_mut().zip(qq_slice.iter_mut()) {
                    let x = *qp;
                    let y = *qq;
                    *qp = c * x - s * y;
                    *qq = s * x + c * y;
                }
            }
        }
        if !changed {
            return Ok(());
        }
    }
    let mut final_off = 0.0_f64;
    for p in 0..m {
        let row_p = &r[p * stride..p * stride + m];
        if !row_p[p].is_finite() {
            return Err(SaddleError::NonFinite("faer_eigh Jacobi diagonal"));
        }
        for &rpq in &row_p[(p + 1)..m] {
            if !rpq.is_finite() {
                return Err(SaddleError::NonFinite("faer_eigh Jacobi off-diagonal"));
            }
            let mag = rpq.abs();
            if mag > final_off {
                final_off = mag;
            }
        }
    }
    if final_off <= res_tol {
        Ok(())
    } else {
        Err(SaddleError::Solver(
            "Jacobi refinement did not converge within sweep budget".into(),
        ))
    }
}

#[cfg(feature = "faer")]
fn max_eigenpair_residual(
    av: &faer::Mat<f64>,
    vectors: faer::MatRef<'_, f64>,
    values: faer::diag::DiagRef<'_, f64>,
    n: usize,
) -> Result<f64, SaddleError> {
    let mut raw_res = 0.0_f64;
    for col in 0..n {
        let lam = values[col];
        for row in 0..n {
            let err = (av[(row, col)] - vectors[(row, col)] * lam).abs();
            if !err.is_finite() {
                return Err(SaddleError::NonFinite("faer_eigh residual"));
            }
            if err > raw_res {
                raw_res = err;
            }
        }
    }
    Ok(raw_res)
}

/// Self-adjoint spectrum from faer. `a` is symmetrised as `(a + a^T) / 2` first.
/// Eigenvalues are nondecreasing. Columns are the eigenvectors.
///
/// The `faer` feature is on by default. A build without it still
/// compiles, and this function falls back to cyclic Jacobi.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square; [`SaddleError::NonFinite`]
/// when any entry is non-finite; [`SaddleError::Solver`] when the eigensolver
/// does not converge.
pub fn faer_eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    if a.nrows() != a.ncols() {
        return Err(SaddleError::Shape("eigh needs a square matrix".into()));
    }
    if !a.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("faer_eigh"));
    }
    let n = a.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    #[cfg(not(feature = "faer"))]
    {
        let mut work = a.clone();
        for i in 0..n {
            for j in 0..i {
                let s = 0.5 * (work[(i, j)] + work[(j, i)]);
                work[(i, j)] = s;
                work[(j, i)] = s;
            }
        }
        jacobi_eigh(&mut work)
    }
    #[cfg(feature = "faer")]
    {
        let mut scale = 0.0_f64;
        let mat = faer::Mat::from_fn(n, n, |i, j| {
            let s = 0.5 * (a[(i, j)] + a[(j, i)]);
            let abs = s.abs();
            if abs > scale {
                scale = abs;
            }
            s
        });
        if scale == 0.0 {
            return Ok((Array1::zeros(n), Array2::eye(n)));
        }
        let evd = mat
            .self_adjoint_eigen(faer::Side::Lower)
            .map_err(|_| SaddleError::Solver("self-adjoint eigen did not converge".into()))?;
        let values_ref = evd.S();
        let vectors_ref = evd.U();
        for column in 0..n {
            if !values_ref[column].is_finite() {
                return Err(SaddleError::NonFinite("faer_eigh eigenvalue"));
            }
            for row in 0..n {
                if !vectors_ref[(row, column)].is_finite() {
                    return Err(SaddleError::NonFinite("faer_eigh eigenvector"));
                }
            }
        }
        let res_tol = relative_zero_tolerance(scale, n);
        let av_raw = &mat * vectors_ref;
        let raw_res = max_eigenpair_residual(&av_raw, vectors_ref, values_ref, n)?;
        if raw_res <= res_tol {
            let mut values = Array1::zeros(n);
            let mut vectors = Array2::zeros((n, n));
            for col in 0..n {
                values[col] = values_ref[col];
                for row in 0..n {
                    vectors[(row, col)] = vectors_ref[(row, col)];
                }
                if nrm2(vectors.column(col)) <= f64::EPSILON {
                    return Err(SaddleError::Solver(
                        "self-adjoint eigen produced a zero eigenvector".into(),
                    ));
                }
            }
            return Ok((values, vectors));
        }

        let mut gram = vectors_ref.transpose() * vectors_ref;
        for i in 0..n {
            gram[(i, i)] = 1.5 - 0.5 * gram[(i, i)];
            for j in 0..i {
                let s = -0.25 * (gram[(i, j)] + gram[(j, i)]);
                gram[(i, j)] = s;
                gram[(j, i)] = s;
            }
        }
        let v_orth = vectors_ref * &gram;
        let av = &mat * &v_orth;
        let r_mat = v_orth.transpose() * &av;
        let rot_tol = relative_zero_tolerance(scale, 1);
        let mut active = Vec::with_capacity(n);
        for i in 0..n {
            if !r_mat[(i, i)].is_finite() {
                return Err(SaddleError::NonFinite("faer_eigh Rayleigh quotient"));
            }
            let mut row_off = 0.0_f64;
            for j in 0..n {
                if i != j {
                    let rij = 0.5 * (r_mat[(i, j)] + r_mat[(j, i)]).abs();
                    if !rij.is_finite() {
                        return Err(SaddleError::NonFinite("faer_eigh Rayleigh quotient"));
                    }
                    if rij > row_off {
                        row_off = rij;
                    }
                }
            }
            if row_off > rot_tol {
                active.push(i);
            }
        }

        let mut all_vals: Vec<f64> = (0..n).map(|i| r_mat[(i, i)]).collect();
        let mut v_full = v_orth;
        if !active.is_empty() {
            let m = active.len();
            let stride = m | 1;
            let mut q_col = vec![0.0_f64; m * m];
            let mut r = vec![0.0_f64; m * stride];
            for (i, &ri) in active.iter().enumerate() {
                q_col[i * m + i] = 1.0;
                r[i * stride + i] = all_vals[ri];
                for (j, &rj) in active.iter().enumerate().skip(i + 1) {
                    let rij = 0.5 * (r_mat[(ri, rj)] + r_mat[(rj, ri)]);
                    if rij.abs() > rot_tol {
                        r[i * stride + j] = rij;
                        r[j * stride + i] = rij;
                    }
                }
            }
            let max_sweeps = m;
            #[cfg(test)]
            let max_sweeps = TEST_JACOBI_SWEEP_BUDGET.with(|cell| cell.get().unwrap_or(max_sweeps));
            cyclic_jacobi_subspace(&mut r, &mut q_col, m, stride, res_tol, rot_tol, max_sweeps)?;
            let v_act = faer::Mat::from_fn(n, m, |row, j| v_full[(row, active[j])]);
            let q_mat = faer::Mat::from_fn(m, m, |row, col| q_col[col * m + row]);
            let v_act_rot = &v_act * &q_mat;
            for (j, &col) in active.iter().enumerate() {
                all_vals[col] = r[j * stride + j];
                for row in 0..n {
                    v_full[(row, col)] = v_act_rot[(row, j)];
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| all_vals[i].total_cmp(&all_vals[j]));
        let mut values = Array1::zeros(n);
        let mut vectors = Array2::zeros((n, n));
        for (dst, &src) in order.iter().enumerate() {
            let val = all_vals[src];
            if !val.is_finite() {
                return Err(SaddleError::NonFinite("faer_eigh eigenvalue"));
            }
            values[dst] = val;
            for row in 0..n {
                let vec_entry = v_full[(row, src)];
                if !vec_entry.is_finite() {
                    return Err(SaddleError::NonFinite("faer_eigh eigenvector"));
                }
                vectors[(row, dst)] = vec_entry;
            }
            if nrm2(vectors.column(dst)) <= f64::EPSILON {
                return Err(SaddleError::Solver(
                    "self-adjoint eigen produced a zero eigenvector".into(),
                ));
            }
        }
        Ok((values, vectors))
    }
}

/// Self-adjoint eigenvalues only (ascending), without accumulating eigenvectors.
///
/// Symmetrises `a` as `(a + a^T) / 2` before computing eigenvalues.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square; [`SaddleError::NonFinite`]
/// when any entry is non-finite; [`SaddleError::Solver`] when the solver fails.
pub fn exact_eigvalsh(a: ArrayView2<f64>) -> Result<Array1<f64>, SaddleError> {
    if a.nrows() != a.ncols() {
        return Err(SaddleError::Shape(
            "exact_eigvalsh needs a square matrix".into(),
        ));
    }
    if !a.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("exact_eigvalsh"));
    }
    let n = a.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    #[cfg(feature = "faer")]
    {
        let mat = faer::Mat::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
        let raw = mat
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .map_err(|_| SaddleError::Solver("self-adjoint eigenvalues did not converge".into()))?;
        for &v in &raw {
            if !v.is_finite() {
                return Err(SaddleError::NonFinite("exact_eigvalsh output"));
            }
        }
        Ok(Array1::from_vec(raw))
    }
    #[cfg(not(feature = "faer"))]
    {
        let mut work = a.to_owned();
        for i in 0..n {
            for j in 0..i {
                let s = 0.5 * (work[(i, j)] + work[(j, i)]);
                work[(i, j)] = s;
                work[(j, i)] = s;
            }
        }
        Ok(jacobi_eigh(&mut work)?.0)
    }
}

/// Ritz eigenvalues (length `k`, ascending), basis vectors (`n x k`, orthonormal
/// columns) and the operator applied to those vectors (`n x k`).
pub type RitzDecomposition = (Array1<f64>, Array2<f64>, Array2<f64>);

/// Rayleigh-Ritz on the columns of `v` against a dense `a`.
///
/// Returns `(lams, v_rot, a @ v_rot)`. `gamma <= 0` falls through to
/// [`exact_eigh`] on `a` (Sella `exact`).
///
/// `a` is a symmetric `n x n` matrix and `v` an `n x k` basis (one vector per
/// column; it is orthonormalised first). Returns the Ritz values, the rotated
/// basis and `a` times it.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square or `v` has a different row
/// count; [`SaddleError::Solver`] when `v` spans nothing, or the small
/// eigenproblem fails.
pub fn rayleigh_ritz(
    a: ArrayView2<f64>,
    v: ArrayView2<f64>,
    gamma: f64,
) -> Result<RitzDecomposition, SaddleError> {
    if a.nrows() != a.ncols() || v.nrows() != a.nrows() {
        return Err(SaddleError::Shape(
            "Rayleigh-Ritz needs square A and matching V rows".into(),
        ));
    }
    if gamma <= 0.0 {
        let (lams, vecs) = exact_eigh(a)?;
        let av = matmul(a, vecs.view());
        return Ok((lams, vecs, av));
    }
    let q = modified_gram_schmidt(v, None, 1e-14);
    if q.ncols() == 0 {
        return Err(SaddleError::Solver("empty Ritz subspace".into()));
    }
    let av = matmul(a, q.view());
    let a_tilde = symmetrize_vt_av(q.view(), av.view());
    let (lams, vecs) = exact_eigh(a_tilde.view())?;
    let v_rot = matmul(q.view(), vecs.view());
    let av_rot = matmul(av.view(), vecs.view());
    Ok((lams, v_rot, av_rot))
}

/// Dense eigen on `device`. The host arm is [`exact_eigh`].
/// [`EigenDevice::Dlpk`] is the lowest-mode arm through rgmin.
///
/// `a` is a symmetric `n x n` matrix. Returns the ascending eigenvalues and the
/// eigenvector columns.
///
/// # Errors
///
/// The errors of [`exact_eigh`].
pub fn eigh_on(
    device: EigenDevice,
    a: ArrayView2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    match device {
        EigenDevice::Host => exact_eigh(a),
        EigenDevice::Dlpk => crate::gpu::gpu_eigh_env(a),
    }
}

/// Lowest eigenpair through rgmin (`Lanczos` / `RayleighRitz`).
///
/// This is the dlpk waist Sella `_gpu.py` maps onto. Full spectrum
/// stays [`exact_eigh`].
///
/// `a` is a symmetric `n x n` matrix and `seed` the length-`n` starting vector.
/// [`EigenDevice::Host`] runs Lanczos and [`EigenDevice::Dlpk`] Rayleigh-Ritz,
/// both with a Krylov dimension of `n` limited to the range 2 to 8. Returns the
/// lowest eigenvalue and its unit eigenvector.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square or `seed` has the wrong length;
/// [`SaddleError::Solver`] when rgmin fails.
pub fn lowest_on(
    device: EigenDevice,
    a: ArrayView2<f64>,
    seed: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    if a.nrows() != a.ncols() || seed.len() != a.nrows() {
        return Err(SaddleError::Shape(
            "lowest_on needs square A and a matching seed".into(),
        ));
    }
    let x = Array1::zeros(seed.len());
    let kind = match device {
        EigenDevice::Host => EigensolverKind::Lanczos,
        EigenDevice::Dlpk => EigensolverKind::RayleighRitz,
    };
    let params = EigenParams::default()
        .with_kind(kind)
        .with_krylov(seed.len().clamp(2, 8));
    let mode = lowest_mode(&DenseApply(&a.to_owned()), x.view(), seed, &params)
        .map_err(|e| SaddleError::Solver(e.to_string()))?;
    Ok((mode.value, mode.vector))
}

/// Sella `expand`: one Davidson / Jacobi-Davidson / Lanczos direction.
///
/// `v` is the current Ritz basis `(d, n)`, `y` is `A @ v` (or the
/// symmetrized action), `p` is the preconditioner, `b` the metric
/// (`None` is `I`). `lams` / `vecs` are the current Ritz pairs.
#[expect(
    clippy::too_many_arguments,
    reason = "The public expansion API names each operator and basis separately."
)]
/// Shapes: `v` and `y` are `d x n` (`n` basis vectors of length `d`), `p` is
/// `d x d`, `b`, if given, is `d x d`, `lams` has length `n` and `vecs` is `n x n`
/// (the eigenvectors of the projected problem). `shift` is the eigenvalue shift
/// applied to `p`, `method` the correction formula and `seeking` the index of the
/// Ritz pair to improve (`seeking < n`). Returns the length-`d` direction.
///
/// # Errors
///
/// [`SaddleError::Shape`] when the shapes above disagree or `seeking >= n`;
/// [`SaddleError::Singular`] when a correction system is singular;
/// [`SaddleError::NonFinite`] when an input or a correction is non-finite.
pub fn expand(
    v: ArrayView2<f64>,
    y: ArrayView2<f64>,
    p: ArrayView2<f64>,
    b: Option<ArrayView2<f64>>,
    lams: ArrayView1<f64>,
    vecs: ArrayView2<f64>,
    shift: f64,
    method: ExpandKind,
    seeking: usize,
) -> Result<Array1<f64>, SaddleError> {
    let d = v.nrows();
    let n = v.ncols();
    if y.nrows() != d || y.ncols() != n || p.nrows() != d || p.ncols() != d {
        return Err(SaddleError::Shape(
            "expand needs matching V, Y, P shapes".into(),
        ));
    }
    if seeking >= n || lams.len() != n || vecs.nrows() != n || vecs.ncols() != n {
        return Err(SaddleError::Shape(
            "expand Ritz pairs must match the subspace".into(),
        ));
    }
    let bmat = b.map(|m| m.to_owned());
    let bv = match &bmat {
        Some(bm) => matmul(bm.view(), v),
        None => v.to_owned(),
    };
    // R = Y V_rot - B V V_rot * lam  with V already rotated in Sella;
    // here vecs rotates the current subspace.
    let vrot = matmul(v, vecs);
    let yrot = matmul(y, vecs);
    let bvrot = matmul(bv.view(), vecs);
    let mut r = yrot;
    for j in 0..n {
        let mut col = r.column(j).to_owned();
        axpy(-lams[j], bvrot.column(j), &mut col);
        r.column_mut(j).assign(&col);
    }
    let mut pshift = p.to_owned();
    match &bmat {
        Some(bm) => {
            for i in 0..d {
                for j in 0..d {
                    pshift[(i, j)] -= shift * bm[(i, j)];
                }
            }
        }
        None => {
            for i in 0..d {
                pshift[(i, i)] -= shift;
            }
        }
    }
    let rseek = r.column(seeking).to_owned();
    match method {
        ExpandKind::Lanczos => Ok(rseek),
        ExpandKind::Gd => solve_dense(pshift.view(), rseek.view()),
        ExpandKind::Jd0 => {
            let vi = vrot.column(seeking).to_owned();
            let mut aaug = Array2::<f64>::zeros((d + 1, d + 1));
            for i in 0..d {
                for j in 0..d {
                    aaug[(i, j)] = pshift[(i, j)];
                }
                aaug[(i, d)] = vi[i];
                aaug[(d, i)] = vi[i];
            }
            let mut raug = Array1::zeros(d + 1);
            for i in 0..d {
                raug[i] = -rseek[i];
            }
            let z = solve_dense(aaug.view(), raug.view())?;
            Ok(z.slice(s![..d]).to_owned())
        }
        ExpandKind::Jd0Alt => {
            let vi = vrot.column(seeking).to_owned();
            let pprojr = solve_dense(pshift.view(), rseek.view())?;
            let pprojv = solve_dense(pshift.view(), vi.view())?;
            let denom = dot(vi.view(), pprojv.view());
            if denom.abs() < 1e-12 {
                return Ok(pprojr);
            }
            let alpha = dot(vi.view(), pprojr.view()) / denom;
            let mut t = pprojv;
            t.mapv_inplace(|x| x * alpha);
            axpy(-1.0, pprojr.view(), &mut t);
            Ok(t)
        }
        ExpandKind::Mjd0 => {
            let mut aaug = Array2::<f64>::zeros((d + n, d + n));
            for i in 0..d {
                for j in 0..d {
                    aaug[(i, j)] = pshift[(i, j)];
                }
                for j in 0..n {
                    aaug[(i, d + j)] = vrot[(i, j)];
                    aaug[(d + j, i)] = vrot[(i, j)];
                }
            }
            let mut raug = Array1::zeros(d + n);
            for i in 0..d {
                raug[i] = -rseek[i];
            }
            let z = solve_dense(aaug.view(), raug.view())?;
            Ok(z.slice(s![..d]).to_owned())
        }
        ExpandKind::Mjd0Alt => {
            let pprojr = solve_dense(pshift.view(), rseek.view())?;
            let mut rhs = Array1::zeros(n);
            let mut gram = Array2::<f64>::zeros((n, n));
            for j in 0..n {
                let col = vrot.column(j).to_owned();
                let pj = solve_dense(pshift.view(), col.view())?;
                for i in 0..n {
                    gram[(i, j)] = dot(vrot.column(i), pj.view());
                }
                rhs[j] = dot(vrot.column(j), pprojr.view());
            }
            let alpha = solve_dense(gram.view(), rhs.view())?;
            let mut t = Array1::zeros(d);
            for i in 0..d {
                t[i] = dot(vrot.row(i), alpha.view()) - rseek[i];
            }
            solve_dense(pshift.view(), t.view())
        }
    }
}

/// Sella `rayleigh_ritz` loop: grow `v` with [`expand`] until the residuals of the
/// negative Ritz pairs (the lowest one when none is negative) are below
/// `gamma |theta|`, or no new direction independent of the subspace remains.
/// The subspace never exceeds `n` columns.
///
/// `a` is a symmetric `n x n` matrix, `v0` an `n x k` starting basis, `gamma` the
/// relative residual tolerance and `method` the expansion formula. A `gamma` of 0
/// or less returns the exact spectrum of `a`. The result is as for
/// [`rayleigh_ritz`].
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square or `v0` has a different row
/// count; [`SaddleError::Solver`] when `v0` spans nothing; the errors of
/// [`expand`] when a correction solve fails.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use rgsaddle::eigensolve::{ExpandKind, rayleigh_ritz_iter};
///
/// // Curvatures -1, 2 and 3 eV/Angstrom^2.
/// let hessian = array![[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
/// let start = array![[1.0], [1.0], [1.0]];
/// let (values, _basis, _applied) =
///     rayleigh_ritz_iter(hessian.view(), start.view(), 0.1, ExpandKind::Jd0)?;
/// assert!((values[0] + 1.0).abs() < 0.1);
/// # Ok::<(), rgsaddle::SaddleError>(())
/// ```
pub fn rayleigh_ritz_iter(
    a: ArrayView2<f64>,
    v0: ArrayView2<f64>,
    gamma: f64,
    method: ExpandKind,
) -> Result<RitzDecomposition, SaddleError> {
    if a.nrows() != a.ncols() || v0.nrows() != a.nrows() {
        return Err(SaddleError::Shape(
            "iterative Rayleigh-Ritz needs square A and matching V rows".into(),
        ));
    }
    if gamma <= 0.0 {
        let (lams, vecs) = exact_eigh(a)?;
        let av = matmul(a, vecs.view());
        return Ok((lams, vecs, av));
    }
    let n = a.nrows();
    let mut v = modified_gram_schmidt(v0, None, 1e-14);
    if v.ncols() == 0 {
        return Err(SaddleError::Solver("empty Ritz subspace".into()));
    }
    let p = Array2::eye(n);
    let mut av = matmul(a, v.view());
    loop {
        let a_tilde = symmetrize_vt_av(v.view(), av.view());
        let (lams, vecs) = exact_eigh(a_tilde.view())?;
        av = matmul(av.view(), vecs.view());
        v = matmul(v.view(), vecs.view());
        let k = v.ncols();
        if k >= n {
            return Ok((lams, v, av));
        }
        // Sella `nneg = max(1, sum(lams < 0))`.
        let nneg = lams.iter().filter(|&&lam| lam < 0.0).count().max(1).min(k);
        let eye = Array2::eye(k);
        let mut rnorm = Array1::zeros(nneg);
        let mut rcols: Vec<Array1<f64>> = Vec::with_capacity(nneg);
        for j in 0..nneg {
            let mut ri = av.column(j).to_owned();
            axpy(-lams[j], v.column(j), &mut ri);
            rnorm[j] = nrm2(ri.view());
            rcols.push(ri);
        }
        let mut seeking = 0;
        let mut found = false;
        for j in 0..nneg {
            if k == 1 || rnorm[j] >= gamma * lams[j].abs() {
                seeking = j;
                found = true;
                break;
            }
        }
        if !found {
            return Ok((lams, v, av));
        }
        let mut t = expand(
            v.view(),
            av.view(),
            p.view(),
            None,
            lams.view(),
            eye.view(),
            lams[seeking],
            method,
            seeking,
        )?;
        let tn = nrm2(t.view());
        if tn < 1e-18 {
            return Ok((lams, v, av));
        }
        t.mapv_inplace(|x| x / tn);
        // Sella: expand already in span(V) falls back to Lanczos residual.
        let mut tproj = Array1::zeros(n);
        for j in 0..k {
            axpy(dot(v.column(j), t.view()), v.column(j), &mut tproj);
        }
        let mut tperp = t.clone();
        axpy(-1.0, tproj.view(), &mut tperp);
        if nrm2(tperp.view()) < 1e-2 {
            let rn = nrm2(rcols[seeking].view());
            if rn >= 1e-18 {
                t = rcols[seeking].clone();
                t.mapv_inplace(|x| x / rn);
            }
        }
        let mut tcol = Array2::zeros((n, 1));
        for i in 0..n {
            tcol[(i, 0)] = t[i];
        }
        let mut tnew = modified_gram_schmidt(tcol.view(), Some(v.view()), 1e-8);
        if tnew.ncols() == 0 {
            // Sella: try residual columns, then stop.
            for ri in &rcols {
                let mut col = Array2::zeros((n, 1));
                col.column_mut(0).assign(ri);
                tnew = modified_gram_schmidt(col.view(), Some(v.view()), 1e-8);
                if tnew.ncols() == 1 {
                    break;
                }
            }
            if tnew.ncols() == 0 {
                return Ok((lams, v, av));
            }
        }
        let mut vnext = Array2::zeros((n, k + tnew.ncols()));
        for j in 0..k {
            for i in 0..n {
                vnext[(i, j)] = v[(i, j)];
            }
        }
        for j in 0..tnew.ncols() {
            for i in 0..n {
                vnext[(i, k + j)] = tnew[(i, j)];
            }
        }
        let at = matmul(a, tnew.view());
        let mut avnext = Array2::zeros((n, k + tnew.ncols()));
        for j in 0..k {
            for i in 0..n {
                avnext[(i, j)] = av[(i, j)];
            }
        }
        for j in 0..tnew.ncols() {
            for i in 0..n {
                avnext[(i, k + j)] = at[(i, j)];
            }
        }
        v = vnext;
        av = avnext;
    }
}

/// Sella `rayleigh_ritz` on a matrix-free operator.
///
/// `apply` is one Hessian-vector product in the free coordinates.
/// `preconditioner` is Sella's projected model `P`. The loop grows an
/// orthonormal subspace until every negative Ritz residual is below
/// `gamma |theta|`, or the subspace fills the free dimension. Roots
/// past the first `strict_negative` may stop at `gamma` when `|theta|`
/// is below one: that relative test is finer than a one-sided
/// finite-difference product can support. `gamma <= 0` is not accepted
/// here; the caller keeps the model spectrum.
///
/// The residual test reads the `symm=2` products `Y`. With
/// `symmetrized` the Ritz pairs come from `V^T Y` as well, which is
/// Sella's `Atilde`; without it they come from the mean of `V^T AV`
/// and its transpose. Finite-difference products are not symmetric,
/// and the mean then leaves part of each residual inside `V`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rayleigh_ritz_apply<F>(
    n: usize,
    mut apply: F,
    v0: ArrayView2<f64>,
    preconditioner: ArrayView2<f64>,
    gamma: f64,
    method: ExpandKind,
    strict_negative: usize,
    symmetrized: bool,
) -> Result<RitzDecomposition, SaddleError>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    if n == 0 || gamma <= 0.0 || preconditioner.nrows() != n || preconditioner.ncols() != n {
        return Err(SaddleError::Shape(
            "matrix-free Rayleigh-Ritz needs a positive free dimension, gamma, and P".into(),
        ));
    }
    let mut v = if v0.nrows() == n && v0.ncols() > 0 {
        modified_gram_schmidt(v0, None, 1e-12)
    } else {
        Array2::zeros((n, 0))
    };
    if v.ncols() == 0 {
        let (lams, vecs) = exact_eigh(preconditioner)?;
        let nneg = lams.iter().filter(|lam| **lam < 0.0).count().max(1).min(n);
        v = modified_gram_schmidt(vecs.slice(s![.., ..nneg]), None, 1e-12);
    }
    if v.ncols() == 0 {
        return Err(SaddleError::Solver("empty Ritz subspace".into()));
    }
    let mut av = apply_columns(&mut apply, v.view())?;
    let max_columns = n;
    loop {
        let a_tilde = if symmetrized {
            let y = crate::qn::symmetrize_y(v.view(), av.view(), Some(2));
            matmul(v.t(), y.view())
        } else {
            symmetrize_vt_av(v.view(), av.view())
        };
        let (lams, vecs) = exact_eigh(a_tilde.view())?;
        av = matmul(av.view(), vecs.view());
        v = matmul(v.view(), vecs.view());
        let k = v.ncols();
        if k >= max_columns {
            return Ok((lams, v, av));
        }
        let nneg = lams.iter().filter(|lam| **lam < 0.0).count().max(1).min(k);
        let y = if k > 1 {
            crate::qn::symmetrize_y(v.view(), av.view(), Some(2))
        } else {
            av.clone()
        };
        let mut rnorm = Array1::zeros(nneg);
        let mut rcols: Vec<Array1<f64>> = Vec::with_capacity(nneg);
        for j in 0..nneg {
            let mut ri = y.column(j).to_owned();
            axpy(-lams[j], v.column(j), &mut ri);
            rnorm[j] = nrm2(ri.view());
            rcols.push(ri);
        }
        let mut seeking = 0;
        let mut found = false;
        for j in 0..nneg {
            // The uphill roots keep Sella's relative residual. A softer
            // negative root, below the identity curvature, is accepted
            // once its residual is under `gamma`: the relative test then
            // sits under the one-sided finite-difference floor and the
            // subspace would otherwise grow through the free dimension.
            let scale = if j < strict_negative {
                lams[j].abs()
            } else {
                lams[j].abs().max(1.0)
            };
            if rnorm[j] > gamma * scale {
                seeking = j;
                found = true;
                break;
            }
        }
        if !found {
            return Ok((lams, v, av));
        }
        let eye = Array2::eye(k);
        let expanded = expand(
            v.view(),
            y.view(),
            preconditioner,
            None,
            lams.view(),
            eye.view(),
            lams[seeking],
            method,
            seeking,
        );
        let mut t = match expanded {
            Ok(direction) => direction,
            Err(_) => rcols[seeking].clone(),
        };
        let tn = nrm2(t.view());
        if tn < 1e-18 {
            return Ok((lams, v, av));
        }
        t.mapv_inplace(|value| value / tn);
        let mut tproj = Array1::zeros(n);
        for j in 0..k {
            axpy(dot(v.column(j), t.view()), v.column(j), &mut tproj);
        }
        let mut tperp = t.clone();
        axpy(-1.0, tproj.view(), &mut tperp);
        if nrm2(tperp.view()) < 1e-2 {
            let rn = nrm2(rcols[seeking].view());
            if rn >= 1e-18 {
                t = rcols[seeking].clone();
                t.mapv_inplace(|value| value / rn);
            }
        }
        let mut tcol = Array2::zeros((n, 1));
        tcol.column_mut(0).assign(&t);
        let mut tnew = modified_gram_schmidt(tcol.view(), Some(v.view()), 1e-8);
        if tnew.ncols() == 0 {
            for ri in &rcols {
                let mut col = Array2::zeros((n, 1));
                col.column_mut(0).assign(ri);
                tnew = modified_gram_schmidt(col.view(), Some(v.view()), 1e-8);
                if tnew.ncols() == 1 {
                    break;
                }
            }
            if tnew.ncols() == 0 {
                return Ok((lams, v, av));
            }
        }
        let at = apply_columns(&mut apply, tnew.view())?;
        let mut vnext = Array2::zeros((n, k + tnew.ncols()));
        let mut avnext = Array2::zeros((n, k + tnew.ncols()));
        for j in 0..k {
            vnext.column_mut(j).assign(&v.column(j));
            avnext.column_mut(j).assign(&av.column(j));
        }
        for j in 0..tnew.ncols() {
            vnext.column_mut(k + j).assign(&tnew.column(j));
            avnext.column_mut(k + j).assign(&at.column(j));
        }
        v = vnext;
        av = avnext;
    }
}

fn apply_columns<F>(apply: &mut F, v: ArrayView2<f64>) -> Result<Array2<f64>, SaddleError>
where
    F: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let n = v.nrows();
    let mut av = Array2::zeros((n, v.ncols()));
    for j in 0..v.ncols() {
        let column = apply(v.column(j))?;
        if column.len() != n || !column.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::Shape(
                "Hessian action must be a finite free vector".into(),
            ));
        }
        av.column_mut(j).assign(&column);
    }
    Ok(av)
}

#[cfg(feature = "faer")]
use faer::prelude::Solve;

/// Solve a dense square linear system `A x = b` with partial pivoting.
///
/// Uses [`relative_zero_tolerance`] across all matrix sizes and refuses
/// non-finite inputs, singular pivots, and overflowing solutions.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square or `b` has a mismatched length;
/// [`SaddleError::NonFinite`] when `a`, `b`, or the solution contains a non-finite
/// entry; [`SaddleError::Singular`] when a pivot magnitude is at or below
/// [`relative_zero_tolerance`].
pub fn solve_dense(a: ArrayView2<f64>, b: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(SaddleError::Shape(
            "dense solve needs a square system".into(),
        ));
    }
    let mut max_abs = 0.0_f64;
    for &v in a.iter() {
        if !v.is_finite() {
            return Err(SaddleError::NonFinite("solve_dense matrix"));
        }
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !b.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("solve_dense rhs"));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    if max_abs == 0.0 {
        return Err(SaddleError::Singular(0));
    }
    let tol = relative_zero_tolerance(max_abs, n);
    #[cfg(feature = "faer")]
    if n >= 32 {
        let mat = faer::Mat::from_fn(n, n, |i, j| a[(i, j)]);
        let lu = mat.partial_piv_lu();
        let u = lu.U();
        for k in 0..n {
            let d = u[(k, k)].abs();
            if !d.is_finite() || d <= tol {
                return Err(SaddleError::Singular(k));
            }
        }
        let rhs = faer::Mat::from_fn(n, 1, |i, _| b[i]);
        let sol = lu.solve(&rhs);
        let mut out = Array1::zeros(n);
        for i in 0..n {
            let v = sol[(i, 0)];
            if !v.is_finite() {
                return Err(SaddleError::NonFinite("solve_dense solution"));
            }
            out[i] = v;
        }
        return Ok(out);
    }
    let mut m = a.to_owned();
    let mut x = b.to_owned();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[(k, k)].abs();
        for i in (k + 1)..n {
            let mag = m[(i, k)].abs();
            if mag > best {
                best = mag;
                piv = i;
            }
        }
        if !best.is_finite() || best <= tol {
            return Err(SaddleError::Singular(k));
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
            let tmp = x[k];
            x[k] = x[piv];
            x[piv] = tmp;
        }
        let akk = m[(k, k)];
        for i in (k + 1)..n {
            let f = m[(i, k)] / akk;
            m[(i, k)] = 0.0;
            for j in (k + 1)..n {
                m[(i, j)] -= f * m[(k, j)];
            }
            x[i] -= f * x[k];
        }
    }
    for i in (0..n).rev() {
        let mut acc = x[i];
        for j in (i + 1)..n {
            acc -= m[(i, j)] * x[j];
        }
        let xi = acc / m[(i, i)];
        if !xi.is_finite() {
            return Err(SaddleError::NonFinite("solve_dense solution"));
        }
        x[i] = xi;
    }
    Ok(x)
}

/// Solve a dense square system `A X = B` for multiple right-hand side columns.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square or `b.nrows() != a.nrows()`;
/// [`SaddleError::NonFinite`] when inputs or solution entries are non-finite;
/// [`SaddleError::Singular`] when a pivot magnitude is at or below
/// [`relative_zero_tolerance`].
pub fn solve_dense_matrix(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
) -> Result<Array2<f64>, SaddleError> {
    let n = a.nrows();
    let nrhs = b.ncols();
    if a.ncols() != n || b.nrows() != n {
        return Err(SaddleError::Shape(
            "solve_dense_matrix needs square A and matching B rows".into(),
        ));
    }
    let mut max_abs = 0.0_f64;
    for &v in a.iter() {
        if !v.is_finite() {
            return Err(SaddleError::NonFinite("solve_dense_matrix matrix"));
        }
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if !b.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("solve_dense_matrix rhs"));
    }
    if n == 0 || nrhs == 0 {
        return Ok(Array2::zeros((n, nrhs)));
    }
    if max_abs == 0.0 {
        return Err(SaddleError::Singular(0));
    }
    #[cfg(feature = "faer")]
    if n >= 32 {
        let tol = relative_zero_tolerance(max_abs, n);
        let mat = faer::Mat::from_fn(n, n, |i, j| a[(i, j)]);
        let lu = mat.partial_piv_lu();
        let u = lu.U();
        for k in 0..n {
            let d = u[(k, k)].abs();
            if !d.is_finite() || d <= tol {
                return Err(SaddleError::Singular(k));
            }
        }
        let rhs = faer::Mat::from_fn(n, nrhs, |i, j| b[(i, j)]);
        let sol = lu.solve(&rhs);
        let mut out = Array2::zeros((n, nrhs));
        for i in 0..n {
            for j in 0..nrhs {
                let v = sol[(i, j)];
                if !v.is_finite() {
                    return Err(SaddleError::NonFinite("solve_dense_matrix solution"));
                }
                out[(i, j)] = v;
            }
        }
        return Ok(out);
    }
    let mut out = Array2::zeros((n, nrhs));
    for j in 0..nrhs {
        let col = solve_dense(a, b.column(j))?;
        out.column_mut(j).assign(&col);
    }
    Ok(out)
}

/// Log-absolute-determinant `ln|det(A)|` of a dense square matrix via partial-pivoting LU.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square; [`SaddleError::NonFinite`]
/// when any entry or the resulting log-determinant is non-finite;
/// [`SaddleError::Singular`] when a pivot magnitude is at or below
/// [`relative_zero_tolerance`].
pub fn dense_logabsdet(a: ArrayView2<f64>) -> Result<f64, SaddleError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SaddleError::Shape(
            "dense_logabsdet needs a square matrix".into(),
        ));
    }
    let mut max_abs = 0.0_f64;
    for &v in a.iter() {
        if !v.is_finite() {
            return Err(SaddleError::NonFinite("dense_logabsdet"));
        }
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if n == 0 {
        return Ok(0.0);
    }
    if max_abs == 0.0 {
        return Err(SaddleError::Singular(0));
    }
    let tol = relative_zero_tolerance(max_abs, n);
    #[cfg(feature = "faer")]
    if n >= 32 {
        let mat = faer::Mat::from_fn(n, n, |i, j| a[(i, j)]);
        let lu = mat.partial_piv_lu();
        let u = lu.U();
        let mut log_abs = 0.0;
        for k in 0..n {
            let d = u[(k, k)].abs();
            if !d.is_finite() || d <= tol {
                return Err(SaddleError::Singular(k));
            }
            log_abs += d.ln();
        }
        if !log_abs.is_finite() {
            return Err(SaddleError::NonFinite("dense_logabsdet result"));
        }
        return Ok(log_abs);
    }
    let mut m = a.to_owned();
    let mut log_abs = 0.0;
    for col in 0..n {
        let mut piv = col;
        let mut best = m[(col, col)].abs();
        for r in (col + 1)..n {
            let v = m[(r, col)].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if !best.is_finite() || best <= tol {
            return Err(SaddleError::Singular(col));
        }
        if piv != col {
            for j in 0..n {
                let tmp = m[(col, j)];
                m[(col, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
        }
        let diag = m[(col, col)];
        log_abs += diag.abs().ln();
        for r in (col + 1)..n {
            let f = m[(r, col)] / diag;
            for j in (col + 1)..n {
                m[(r, j)] -= f * m[(col, j)];
            }
        }
    }
    if !log_abs.is_finite() {
        return Err(SaddleError::NonFinite("dense_logabsdet result"));
    }
    Ok(log_abs)
}

/// Log-determinant `ln det(A)` of a symmetric positive-definite matrix via Cholesky (`LL^T`).
///
/// A pivot `L_ii^2` at or below [`relative_zero_tolerance`] of `max |a_ij|` is
/// refused, so a uniform rescaling of `a` never changes whether it is accepted.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `a` is not square; [`SaddleError::NonFinite`]
/// when any entry or the resulting log-determinant is non-finite;
/// [`SaddleError::Invalid`] when `a` is not symmetric within `16 * relative_zero_tolerance`;
/// [`SaddleError::Solver`] or [`SaddleError::Singular`] when `a` is not strictly
/// positive definite.
pub fn spd_logdet(a: ArrayView2<f64>) -> Result<f64, SaddleError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SaddleError::Shape(
            "spd_logdet needs a square matrix".into(),
        ));
    }
    let mut max_abs = 0.0_f64;
    for &v in a.iter() {
        if !v.is_finite() {
            return Err(SaddleError::NonFinite("spd_logdet"));
        }
        let av = v.abs();
        if av > max_abs {
            max_abs = av;
        }
    }
    if n == 0 {
        return Ok(0.0);
    }
    if max_abs == 0.0 {
        return Err(SaddleError::Singular(0));
    }
    let tol = relative_zero_tolerance(max_abs, n);
    let sym_tol = 16.0 * tol;
    for i in 0..n {
        for j in 0..i {
            if (a[(i, j)] - a[(j, i)]).abs() > sym_tol {
                return Err(SaddleError::Invalid(
                    "spd_logdet matrix is not symmetric".into(),
                ));
            }
        }
    }
    #[cfg(feature = "faer")]
    if n >= 32 {
        let mat = faer::Mat::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
        let llt = mat
            .llt(faer::Side::Lower)
            .map_err(|_| SaddleError::Solver("matrix is not symmetric positive definite".into()))?;
        let l = llt.L();
        let diag_tol = tol.sqrt();
        let mut log_det = 0.0;
        for i in 0..n {
            let d = l[(i, i)];
            if !d.is_finite() || d <= diag_tol {
                return Err(SaddleError::Solver(
                    "matrix is not symmetric positive definite".into(),
                ));
            }
            log_det += 2.0 * d.ln();
        }
        if !log_det.is_finite() {
            return Err(SaddleError::NonFinite("spd_logdet result"));
        }
        return Ok(log_det);
    }
    let mut l = Array2::<f64>::zeros((n, n));
    let mut log_det = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.5 * (a[(i, j)] + a[(j, i)]);
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if !sum.is_finite() || sum <= tol {
                    return Err(SaddleError::Solver(
                        "matrix is not symmetric positive definite".into(),
                    ));
                }
                let diag = sum.sqrt();
                l[(i, i)] = diag;
                log_det += 2.0 * diag.ln();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }
    if !log_det.is_finite() {
        return Err(SaddleError::NonFinite("spd_logdet result"));
    }
    Ok(log_det)
}

fn matmul(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let mut c = Array2::zeros((a.nrows(), b.ncols()));
    for j in 0..b.ncols() {
        let bj = b.column(j);
        for i in 0..a.nrows() {
            c[(i, j)] = dot(a.row(i), bj);
        }
    }
    c
}

/// Classical Jacobi eigensolver for a small dense symmetric `n x n` matrix.
///
/// # Derivation of tolerance and rotation budget
///
/// - **Stopping tolerance `tol = relative_zero_tolerance(scale, n) = scale * n * f64::EPSILON`**:
///   when `max_{i < j} |a_{ij}| <= tol`, Gershgorin's circle theorem bounds the
///   off-diagonal eigenvalue perturbation by `(n - 1) * tol < scale * n^2 * f64::EPSILON`
///   and the quadratic Rayleigh-Ritz shift `sum_{j != i} |a_{ij}|^2 / |lambda_i - lambda_j|`
///   is below binary64 roundoff.
/// - **Rotation budget `16 + 8 * n * n`**: each iteration selects the largest
///   off-diagonal pivot `|a_{pq}| = max_{i < j} |a_{ij}|` among the
///   `N = n * (n - 1) / 2 < n^2 / 2` upper-triangular pairs, so
///   `|a_{pq}|^2 >= off(A)^2 / N` and `off(A_{k+1})^2 = (1 - 1/N) off(A_k)^2 <= exp(-1/N) off(A_k)^2`
///   (Jacobi 1846; Golub & Van Loan, *Matrix Computations*, 4th ed., Sec. 8.5.2).
///   Even under the worst-case linear contraction rate `exp(-1 / (2N))`, `12` sweeps
///   (`12 * N < 6 * n^2` rotations) contract `off(A)` by `exp(-6) approx 2.48e-3`,
///   entering Wilkinson's quadratic-convergence basin (`off(A) / scale < 1/4`,
///   Henrici 1958; Wilkinson 1962), from which at most `4` quadratic sweeps
///   (`4 * N < 2 * n^2` rotations) reach binary64 precision `2^-52 approx exp(-36.04)`.
///   Summing `12 + 4 = 16` sweeps of `< n^2 / 2` pairs gives `8 * n^2` rotations,
///   plus `16` rotations for small `n in {2, 3}`.
fn jacobi_eigh(a: &mut Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SaddleError::Shape(
            "jacobi_eigh needs a square matrix".into(),
        ));
    }
    let max_sweeps = 16 + 8 * n * n;
    #[cfg(test)]
    let max_sweeps = TEST_JACOBI_SWEEP_BUDGET.with(|cell| cell.get().unwrap_or(max_sweeps));
    let mut scale = 0.0_f64;
    for &x in a.iter() {
        if !x.is_finite() {
            return Err(SaddleError::NonFinite("jacobi_eigh"));
        }
        let ax = x.abs();
        if ax > scale {
            scale = ax;
        }
    }
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let mut v = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        v[(i, i)] = 1.0;
    }
    if n == 1 {
        let mut evals = Array1::zeros(1);
        evals[0] = a[(0, 0)];
        return Ok((evals, v));
    }
    let tol = relative_zero_tolerance(scale, n);
    let mut converged = scale == 0.0;
    for _ in 0..max_sweeps {
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mag = a[(i, j)].abs();
                if mag > max {
                    max = mag;
                    p = i;
                    q = j;
                }
            }
        }
        if max <= tol {
            converged = true;
            break;
        }
        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k != p && k != q {
                let aik = a[(k.min(p), k.max(p))];
                let aiq = a[(k.min(q), k.max(q))];
                let npk = c * aik - s * aiq;
                let nqk = s * aik + c * aiq;
                a[(k.min(p), k.max(p))] = npk;
                a[(k.max(p), k.min(p))] = npk;
                a[(k.min(q), k.max(q))] = nqk;
                a[(k.max(q), k.min(q))] = nqk;
            }
            let vip = v[(k, p)];
            let viq = v[(k, q)];
            v[(k, p)] = c * vip - s * viq;
            v[(k, q)] = s * vip + c * viq;
        }
        a[(p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[(q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;
    }
    if !converged {
        return Err(SaddleError::Solver(
            "Jacobi eigensolver did not converge within sweep budget".into(),
        ));
    }
    let mut evals = Array1::zeros(n);
    for i in 0..n {
        if !a[(i, i)].is_finite() {
            return Err(SaddleError::NonFinite("jacobi_eigh diagonal"));
        }
        evals[i] = a[(i, i)];
    }
    // Sort ascending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let mut ev_sorted = Array1::zeros(n);
    let mut vecs = Array2::zeros((n, n));
    for (dst, &src) in order.iter().enumerate() {
        ev_sorted[dst] = evals[src];
        for r in 0..n {
            vecs[(r, dst)] = v[(r, src)];
        }
    }
    for j in 0..n {
        let col = vecs.column(j).to_owned();
        if nrm2(col.view()) <= f64::EPSILON {
            return Err(SaddleError::Solver(
                "Jacobi produced a zero eigenvector".into(),
            ));
        }
    }
    Ok((ev_sorted, vecs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rgmin::Manifold;
    use rgmin::ManifoldKind;

    #[test]
    fn small_exact_eigh_matches_jacobi_and_faer() {
        let n = 40;
        let mut original = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            original[(i, i)] = 0.1 + 0.5 * i as f64;
            if i + 1 < n {
                original[(i, i + 1)] = 0.25;
                original[(i + 1, i)] = 0.25;
            }
        }
        let (lams, _) = exact_eigh(original.view()).unwrap();
        let (wide, _) = faer_eigh(&original).unwrap();
        let (jacobi, _) = jacobi_eigh(&mut original).unwrap();
        for j in 0..n {
            assert!(
                (lams[j] - jacobi[j]).abs() < 1e-10,
                "jacobi {j} {} vs {}",
                lams[j],
                jacobi[j]
            );
            assert!(
                (wide[j] - jacobi[j]).abs() < 1e-8,
                "faer {j} {} vs {}",
                wide[j],
                jacobi[j]
            );
        }
    }

    #[test]
    fn large_exact_eigh_matches_the_laplacian_spectrum() {
        let n = 160;
        assert!(n > DENSE_JACOBI_LIMIT);
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i + 1 < n {
                a[(i, i + 1)] = -1.0;
                a[(i + 1, i)] = -1.0;
            }
        }
        let (lams, vecs) = exact_eigh(a.view()).unwrap();
        for j in 0..n {
            let expect = 2.0 - 2.0 * (std::f64::consts::PI * (j + 1) as f64 / (n + 1) as f64).cos();
            assert!(
                (lams[j] - expect).abs() < 1e-8,
                "j={j} {} vs {expect}",
                lams[j]
            );
        }
        let mut residual = 0.0;
        for i in 0..n {
            let mut av = 0.0;
            for k in 0..n {
                av += a[(i, k)] * vecs[(k, 0)];
            }
            let diff = av - lams[0] * vecs[(i, 0)];
            residual += diff * diff;
        }
        assert!(residual.sqrt() < 1e-8, "residual {}", residual.sqrt());
    }

    #[test]
    fn exact_eigh_recovers_a_known_spectrum() {
        let mut a = Array2::<f64>::zeros((2, 2));
        a[(0, 0)] = 2.0;
        a[(1, 1)] = 8.0;
        let (lams, vecs) = exact_eigh(a.view()).unwrap();
        assert!((lams[0] - 2.0).abs() < 1e-10, "{}", lams[0]);
        assert!((lams[1] - 8.0).abs() < 1e-10, "{}", lams[1]);
        // A v = lam v
        let av0 = a[(0, 0)] * vecs[(0, 0)] + a[(0, 1)] * vecs[(1, 0)];
        assert!((av0 - lams[0] * vecs[(0, 0)]).abs() < 1e-10);
    }

    #[test]
    fn rayleigh_ritz_on_the_full_space_matches_exact() {
        let mut a = Array2::<f64>::zeros((3, 3));
        a[(0, 0)] = 4.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 9.0;
        a[(0, 1)] = 0.5;
        a[(1, 0)] = 0.5;
        let v = Array2::eye(3);
        let (lams, _, _) = rayleigh_ritz(a.view(), v.view(), 0.1).unwrap();
        let (exact, _) = exact_eigh(a.view()).unwrap();
        for i in 0..3 {
            assert!(
                (lams[i] - exact[i]).abs() < 1e-10,
                "{} vs {}",
                lams[i],
                exact[i]
            );
        }
    }

    #[test]
    fn dlpk_device_shares_the_host_waist() {
        let a = Array2::eye(2);
        let (h, _) = eigh_on(EigenDevice::Host, a.view()).unwrap();
        let (d, _) = eigh_on(EigenDevice::Dlpk, a.view()).unwrap();
        assert!((h[0] - d[0]).abs() < 1e-14);
        assert_eq!(EigenDevice::try_from_abi(0), Some(EigenDevice::Host));
        assert_eq!(EigenDevice::try_from_abi(1), Some(EigenDevice::Dlpk));
        assert_eq!(EigenDevice::try_from_abi(9), None);
    }

    #[test]
    fn ritz_vector_proj_retr_transp_stays_on_the_sphere() {
        let mut a = Array2::<f64>::zeros((3, 3));
        a[(0, 0)] = 4.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 9.0;
        a[(0, 1)] = 0.5;
        a[(1, 0)] = 0.5;
        let (lams, vecs) = exact_eigh(a.view()).unwrap();
        assert_eq!(lams.len(), 3);
        let x = vecs.column(0).to_owned();
        assert!(
            (nrm2(x.view()) - 1.0).abs() < 1e-12,
            "eigenvector left the sphere: ||x||={}",
            nrm2(x.view())
        );
        let man = ManifoldKind::Sphere;
        let v_amb = array![0.2, -0.1, 0.3];
        let s = man.project(&x, &v_amb);
        assert!(
            dot(x.view(), s.view()).abs() < 1e-12,
            "projected step must be tangent: x·s={}",
            dot(x.view(), s.view())
        );
        let y = man.retract(&x, &s);
        assert!(
            (nrm2(y.view()) - 1.0).abs() < 1e-12,
            "retracted point left the sphere: ||y||={}",
            nrm2(y.view())
        );
        let t = man.transport(&x, &y, &s);
        assert!(
            dot(y.view(), t.view()).abs() < 1e-12,
            "transported step leaves T_y: y·t={}",
            dot(y.view(), t.view())
        );
        let v0 = Array2::eye(3);
        let (_, vr, _) = rayleigh_ritz(a.view(), v0.view(), 0.1).unwrap();
        let xr = vr.column(0).to_owned();
        assert!((nrm2(xr.view()) - 1.0).abs() < 1e-12);
        let yr = man.retract(&xr, &man.project(&xr, &v_amb));
        assert!((nrm2(yr.view()) - 1.0).abs() < 1e-12);
    }

    type ExpandFixture = (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array1<f64>,
        Array2<f64>,
    );

    fn expand_fixture() -> ExpandFixture {
        let mut a = Array2::<f64>::zeros((3, 3));
        a[(0, 0)] = -2.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 4.0;
        a[(0, 1)] = 0.4;
        a[(1, 0)] = 0.4;
        let mut v = Array2::zeros((3, 1));
        v[(0, 0)] = 1.0;
        v[(1, 0)] = 0.3;
        v[(2, 0)] = 0.1;
        let vn = nrm2(v.column(0));
        for i in 0..3 {
            v[(i, 0)] /= vn;
        }
        let y = matmul(a.view(), v.view());
        let lams = Array1::from(vec![dot(v.column(0), y.column(0))]);
        let vecs = Array2::eye(1);
        (a, v, y, lams, vecs)
    }

    #[test]
    fn expand_jd0_is_orthogonal_to_the_ritz_vector() {
        let (_a, v, y, lams, vecs) = expand_fixture();
        let p = Array2::eye(3);
        let t = expand(
            v.view(),
            y.view(),
            p.view(),
            None,
            lams.view(),
            vecs.view(),
            lams[0],
            ExpandKind::Jd0,
            0,
        )
        .unwrap();
        assert!(t.iter().all(|x| x.is_finite()));
        let ov = dot(t.view(), v.column(0));
        assert!(
            ov.abs() < 1e-8,
            "jd0 must be orthogonal to the Ritz vector: {ov}"
        );
        let t_l = expand(
            v.view(),
            y.view(),
            p.view(),
            None,
            lams.view(),
            vecs.view(),
            lams[0],
            ExpandKind::Lanczos,
            0,
        )
        .unwrap();
        assert_eq!(t_l.len(), 3);
        assert!(nrm2(t_l.view()) > 1e-12);
    }

    #[test]
    fn expand_every_kind_returns_a_finite_direction() {
        let (_a, v, y, lams, vecs) = expand_fixture();
        let p = Array2::eye(3);
        for kind in [
            ExpandKind::Lanczos,
            ExpandKind::Gd,
            ExpandKind::Jd0,
            ExpandKind::Jd0Alt,
            ExpandKind::Mjd0,
            ExpandKind::Mjd0Alt,
        ] {
            let t = expand(
                v.view(),
                y.view(),
                p.view(),
                None,
                lams.view(),
                vecs.view(),
                lams[0],
                kind,
                0,
            )
            .unwrap();
            assert_eq!(t.len(), 3, "{kind:?}");
            assert!(t.iter().all(|x| x.is_finite()), "{kind:?}");
        }
    }

    #[test]
    fn rayleigh_ritz_iter_recovers_the_lowest() {
        let mut a = Array2::<f64>::zeros((4, 4));
        a[(0, 0)] = -5.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 3.0;
        a[(3, 3)] = 8.0;
        a[(0, 1)] = 0.2;
        a[(1, 0)] = 0.2;
        let mut v0 = Array2::zeros((4, 1));
        v0[(0, 0)] = 1.0;
        let (lams, vecs, _) =
            rayleigh_ritz_iter(a.view(), v0.view(), 0.1, ExpandKind::Jd0).unwrap();
        let (exact, _) = exact_eigh(a.view()).unwrap();
        assert!(
            (lams[0] - exact[0]).abs() < 1e-6,
            "{} vs {}",
            lams[0],
            exact[0]
        );
        assert_eq!(vecs.nrows(), 4);
    }

    #[test]
    fn rayleigh_ritz_iter_returns_once_the_basis_spans_the_space() {
        // A one-dimensional basis already spans R^1. The shift of 1 makes
        // the `Gd` system `(P - shift) t = r` singular, so a loop that keeps
        // expanding fails here instead of returning the exact pair.
        let a = array![[1.0]];
        let v0 = array![[1.0]];
        let (lams, vecs, av) =
            rayleigh_ritz_iter(a.view(), v0.view(), 0.1, ExpandKind::Gd).unwrap();
        assert_eq!(vecs.ncols(), 1);
        assert!((lams[0] - 1.0).abs() < 1e-12, "{}", lams[0]);
        assert!((av[(0, 0)] - vecs[(0, 0)]).abs() < 1e-12);
    }

    #[test]
    fn rayleigh_ritz_iter_never_passes_n_columns() {
        for n in 1..=6 {
            let mut a = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                a[(i, i)] = i as f64 - 1.5;
                if i + 1 < n {
                    a[(i, i + 1)] = 0.4;
                    a[(i + 1, i)] = 0.4;
                }
            }
            let mut v0 = Array2::zeros((n, 1));
            v0[(0, 0)] = 1.0;
            for kind in [
                ExpandKind::Lanczos,
                ExpandKind::Gd,
                ExpandKind::Jd0,
                ExpandKind::Jd0Alt,
                ExpandKind::Mjd0,
                ExpandKind::Mjd0Alt,
            ] {
                let (lams, vecs, av) =
                    rayleigh_ritz_iter(a.view(), v0.view(), 1e-12, kind).unwrap();
                let k = vecs.ncols();
                assert!(k <= n, "n={n} {kind:?}: {k} columns");
                assert_eq!(lams.len(), k);
                assert_eq!(av.ncols(), k);
                let gram = matmul(vecs.t(), vecs.view());
                for i in 0..k {
                    for j in 0..k {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!((gram[(i, j)] - want).abs() < 1e-8, "n={n} {kind:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn lowest_on_finds_the_soft_mode() {
        let mut a = Array2::<f64>::zeros((2, 2));
        a[(0, 0)] = -3.0;
        a[(1, 1)] = 8.0;
        let seed = Array1::from(vec![1.0, 0.1]);
        let (lam, v) = lowest_on(EigenDevice::Host, a.view(), seed.view()).unwrap();
        assert!(lam < 0.0, "lam={lam}");
        assert!(v[0].abs() > v[1].abs());
        let (lam_d, _) = lowest_on(EigenDevice::Dlpk, a.view(), seed.view()).unwrap();
        assert!((lam - lam_d).abs() < 0.5, "{lam} vs {lam_d}");
        assert!(
            (nrm2(v.view()) - 1.0).abs() < 1e-8,
            "lowest mode left the sphere"
        );
        let man = ManifoldKind::Sphere;
        let s = man.project(&v, &array![0.1, -0.2]);
        assert!(dot(v.view(), s.view()).abs() < 1e-12);
        let y = man.retract(&v, &s);
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
        let w = man.transport(&v, &y, &s);
        assert!(dot(y.view(), w.view()).abs() < 1e-12);
    }

    #[test]
    fn faer_n_ge_32_solvers_and_logdet_match_closed_form_analytic_oracles() {
        // 1. Kac-Murdock-Szego SPD Toeplitz matrix A_{ij} = rho^{|i-j|} for n = 40 >= 32:
        //    Closed-form det(A) = (1 - rho^2)^{n - 1}, so ln det(A) = (n - 1) ln(1 - rho^2).
        let n = 40;
        let rho = 0.5_f64;
        let mut kms = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                kms[(i, j)] = rho.powi((i as i32 - j as i32).abs());
            }
        }
        let exact_logdet = (n as f64 - 1.0) * (1.0 - rho * rho).ln();
        let lu_logdet = dense_logabsdet(kms.view()).unwrap();
        let chol_logdet = spd_logdet(kms.view()).unwrap();
        assert!(
            (lu_logdet - exact_logdet).abs() < 1e-12,
            "lu_logdet={lu_logdet} exact={exact_logdet}"
        );
        assert!(
            (chol_logdet - exact_logdet).abs() < 1e-12,
            "chol_logdet={chol_logdet} exact={exact_logdet}"
        );

        // Closed-form inverse of KMS matrix is tridiagonal:
        // (A^{-1})_{0,0} = (A^{-1})_{n-1,n-1} = 1 / (1 - rho^2),
        // (A^{-1})_{i,i} = (1 + rho^2) / (1 - rho^2) for 0 < i < n - 1,
        // (A^{-1})_{i,i+-1} = -rho / (1 - rho^2).
        // Therefore, for b = [1, 1, ..., 1]^T, the exact solution x = A^{-1} b is known in closed form:
        // x_0 = x_{n-1} = 1 / (1 + rho), and x_i = (1 - rho) / (1 + rho) for 0 < i < n - 1.
        let b1 = Array1::from_elem(n, 1.0);
        let x1 = solve_dense(kms.view(), b1.view()).unwrap();
        for i in 0..n {
            let expected = if i == 0 || i + 1 == n {
                1.0 / (1.0 + rho)
            } else {
                (1.0 - rho) / (1.0 + rho)
            };
            assert!(
                (x1[i] - expected).abs() < 1e-13,
                "i={i} got={} expected={expected}",
                x1[i]
            );
        }

        // Multi-column solve A X = B for n = 40 >= 32 with two planted columns.
        let mut x_planted = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x_planted[(i, 0)] = if i == 0 || i + 1 == n {
                1.0 / (1.0 + rho)
            } else {
                (1.0 - rho) / (1.0 + rho)
            };
            x_planted[(i, 1)] = ((i + 1) as f64).sin();
        }
        let b_mat = matmul(kms.view(), x_planted.view());
        let x_solved = solve_dense_matrix(kms.view(), b_mat.view()).unwrap();
        for i in 0..n {
            for c in 0..2 {
                assert!(
                    (x_solved[(i, c)] - x_planted[(i, c)]).abs() < 1e-12,
                    "({i},{c}) got={} expected={}",
                    x_solved[(i, c)],
                    x_planted[(i, c)]
                );
            }
        }

        // 2. Symmetric tridiagonal Toeplitz matrix T_{ii} = 3, T_{i,i+-1} = -1 for n = 48 >= 32:
        //    Closed-form eigenvalues lambda_k = 3 - 2 cos(k pi / (n + 1)), k = 1..=n.
        let m = 48;
        let mut tridiag = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            tridiag[(i, i)] = 3.0;
            if i + 1 < m {
                tridiag[(i, i + 1)] = -1.0;
                tridiag[(i + 1, i)] = -1.0;
            }
        }
        let (evals, evecs) = faer_eigh(&tridiag).unwrap();
        let evals_only = exact_eigvalsh(tridiag.view()).unwrap();
        let mut exact_tridiag_logdet = 0.0;
        for k in 0..m {
            let theta = std::f64::consts::PI * (k + 1) as f64 / (m + 1) as f64;
            let exact_lam = 3.0 - 2.0 * theta.cos();
            exact_tridiag_logdet += exact_lam.ln();
            assert!(
                (evals[k] - exact_lam).abs() < 1e-12,
                "k={k} faer_eigh={} exact={exact_lam}",
                evals[k]
            );
            assert!(
                (evals_only[k] - exact_lam).abs() < 1e-12,
                "k={k} exact_eigvalsh={} exact={exact_lam}",
                evals_only[k]
            );
            // Verify eigenvector residual ||T v_k - lambda_k v_k||_2.
            let mut res_sq = 0.0;
            for i in 0..m {
                let mut tv = 0.0;
                for j in 0..m {
                    tv += tridiag[(i, j)] * evecs[(j, k)];
                }
                let d = tv - exact_lam * evecs[(i, k)];
                res_sq += d * d;
            }
            assert!(res_sq.sqrt() < 1e-12, "k={k} residual={}", res_sq.sqrt());
        }
        let lu_tridiag = dense_logabsdet(tridiag.view()).unwrap();
        let spd_tridiag = spd_logdet(tridiag.view()).unwrap();
        assert!((lu_tridiag - exact_tridiag_logdet).abs() < 1e-11);
        assert!((spd_tridiag - exact_tridiag_logdet).abs() < 1e-11);
    }

    #[test]
    fn relative_singularity_tolerance_is_scale_invariant_across_small_and_large_n() {
        for &n in &[4usize, 40usize] {
            // A well-conditioned matrix uniformly scaled by 1e-100 must succeed
            // (no 1e-18 absolute pivot floor).
            let scale = 1.0e-100;
            let mut scaled = Array2::<f64>::zeros((n, n));
            let mut rhs = Array1::<f64>::zeros(n);
            for i in 0..n {
                scaled[(i, i)] = scale * 2.0;
                rhs[i] = scale * 6.0;
            }
            let sol = solve_dense(scaled.view(), rhs.view()).unwrap();
            for i in 0..n {
                assert!((sol[i] - 3.0).abs() < 1e-12);
            }

            // Rank-deficient matrix with row 2 = row 0 + row 1 must be rejected as Singular.
            let mut singular = Array2::<f64>::eye(n);
            for j in 0..n {
                singular[(0, j)] = (j + 1) as f64;
                singular[(1, j)] = (2 * j + 3) as f64;
                singular[(2, j)] = singular[(0, j)] + singular[(1, j)];
            }
            assert!(matches!(
                solve_dense(singular.view(), Array1::ones(n).view()),
                Err(SaddleError::Singular(_))
            ));
            assert!(matches!(
                solve_dense_matrix(singular.view(), Array2::eye(n).view()),
                Err(SaddleError::Singular(_))
            ));
            assert!(matches!(
                dense_logabsdet(singular.view()),
                Err(SaddleError::Singular(_))
            ));

            // Overflowing solution must return NonFinite.
            let mut tiny = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                tiny[(i, i)] = 1.0e-300;
            }
            let huge = Array1::from_elem(n, 1.0e300);
            assert!(matches!(
                solve_dense(tiny.view(), huge.view()),
                Err(SaddleError::NonFinite(_))
            ));

            // Non-finite inputs must return NonFinite.
            let mut nan_mat = Array2::<f64>::eye(n);
            nan_mat[(n / 2, n / 2)] = f64::NAN;
            assert!(matches!(
                faer_eigh(&nan_mat),
                Err(SaddleError::NonFinite(_))
            ));
            assert!(matches!(
                exact_eigvalsh(nan_mat.view()),
                Err(SaddleError::NonFinite(_))
            ));
            assert!(matches!(
                solve_dense(nan_mat.view(), Array1::ones(n).view()),
                Err(SaddleError::NonFinite(_))
            ));
            assert!(matches!(
                solve_dense_matrix(nan_mat.view(), Array2::eye(n).view()),
                Err(SaddleError::NonFinite(_))
            ));
            assert!(matches!(
                dense_logabsdet(nan_mat.view()),
                Err(SaddleError::NonFinite(_))
            ));
            assert!(matches!(
                spd_logdet(nan_mat.view()),
                Err(SaddleError::NonFinite(_))
            ));

            // spd_logdet refuses indefinite and asymmetric matrices.
            let mut indef = Array2::<f64>::eye(n);
            indef[(n - 1, n - 1)] = -1.0;
            assert!(matches!(
                spd_logdet(indef.view()),
                Err(SaddleError::Solver(_))
            ));
            let mut asym = Array2::<f64>::eye(n);
            asym[(0, 1)] = 0.5;
            assert!(matches!(
                spd_logdet(asym.view()),
                Err(SaddleError::Invalid(_))
            ));
        }
    }

    #[test]
    fn spd_logdet_pivot_tolerance_is_invariant_under_uniform_scaling() {
        // n = 4 takes the manual Cholesky; n = 40 takes faer when that feature is on.
        for &n in &[4usize, 40usize] {
            let mut tridiag = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                tridiag[(i, i)] = 2.0;
                if i + 1 < n {
                    tridiag[(i, i + 1)] = -1.0;
                    tridiag[(i + 1, i)] = -1.0;
                }
            }
            let base_tridiag = ((n + 1) as f64).ln();
            let mut deficient = Array2::<f64>::eye(n);
            deficient[(0, 1)] = 1.0;
            deficient[(1, 0)] = 1.0;
            for &s in &[1.0e-30_f64, 1.0, 1.0e30, 1.0e32] {
                let shift = n as f64 * s.ln();
                let tol = 1e-12 * shift.abs().max(1.0);
                let identity = Array2::<f64>::eye(n) * s;
                let got = spd_logdet(identity.view()).unwrap();
                assert!((got - shift).abs() < tol, "n={n} s={s:e} identity {got}");
                let scaled = &tridiag * s;
                let got = spd_logdet(scaled.view()).unwrap();
                assert!(
                    (got - (shift + base_tridiag)).abs() < tol,
                    "n={n} s={s:e} tridiag {got}"
                );
                let singular = &deficient * s;
                assert!(
                    matches!(spd_logdet(singular.view()), Err(SaddleError::Solver(_))),
                    "n={n} s={s:e} rank-deficient accepted"
                );
            }
        }

        // Scale-invariant relative pivot cutoff across both the host (n = 2 < 32)
        // and faer (n = 40 >= 32) paths for spd_logdet (C1, C2) and LU (LU1-LU3):
        // - 1e32 * I_2, 1e32 * I_40, 1e-30 * I_2, and 1e-30 * I_40 are accepted;
        // - diag(1, ..., 1, 1e-17) and 1e-30 * diag(1, ..., 1, 1e-17) (and 1e30 * diag(1, ..., 1, 1e-17))
        //   have a nonzero sub-tolerance pivot 1e-17 * scale <= n * eps * scale and are refused.
        for &n in &[2usize, 40usize] {
            for &scale in &[1.0e32_f64, 1.0e-30_f64, 1.0_f64] {
                let scaled_eye = Array2::<f64>::from_diag(&Array1::from_elem(n, scale));
                let expected_logdet = (n as f64) * scale.ln();
                let chol = spd_logdet(scaled_eye.view()).unwrap();
                let lu = dense_logabsdet(scaled_eye.view()).unwrap();
                assert!(
                    (chol - expected_logdet).abs() <= 1.0e-12 * (1.0 + expected_logdet.abs()),
                    "n={n} scale={scale:e} spd_logdet={chol} expected={expected_logdet}"
                );
                assert!(
                    (lu - expected_logdet).abs() <= 1.0e-12 * (1.0 + expected_logdet.abs()),
                    "n={n} scale={scale:e} dense_logabsdet={lu} expected={expected_logdet}"
                );
                let rhs = Array1::from_elem(n, scale);
                let x = solve_dense(scaled_eye.view(), rhs.view()).unwrap();
                let x_mat = solve_dense_matrix(scaled_eye.view(), scaled_eye.view()).unwrap();
                for i in 0..n {
                    assert!((x[i] - 1.0).abs() <= 1.0e-12);
                    assert!((x_mat[(i, i)] - 1.0).abs() <= 1.0e-12);
                }

                // Above-tolerance smallest pivot 1e-13 * scale > n * eps * scale is accepted.
                let mut above_tol = scaled_eye.clone();
                above_tol[(n - 1, n - 1)] = scale * 1.0e-13;
                let expected_above = ((n - 1) as f64) * scale.ln() + (scale * 1.0e-13).ln();
                let chol_above = spd_logdet(above_tol.view()).unwrap();
                let lu_above = dense_logabsdet(above_tol.view()).unwrap();
                assert!(
                    (chol_above - expected_above).abs() <= 1.0e-11 * (1.0 + expected_above.abs()),
                    "n={n} scale={scale:e} chol_above={chol_above} expected={expected_above}"
                );
                assert!(
                    (lu_above - expected_above).abs() <= 1.0e-11 * (1.0 + expected_above.abs()),
                    "n={n} scale={scale:e} lu_above={lu_above} expected={expected_above}"
                );
                assert!(solve_dense(above_tol.view(), rhs.view()).is_ok());
                assert!(solve_dense_matrix(above_tol.view(), scaled_eye.view()).is_ok());
            }

            for &scale in &[1.0_f64, 1.0e-30_f64, 1.0e30_f64] {
                let mut sub_tol = Array2::<f64>::from_diag(&Array1::from_elem(n, scale));
                sub_tol[(n - 1, n - 1)] = scale * 1.0e-17;
                assert!(
                    matches!(spd_logdet(sub_tol.view()), Err(SaddleError::Solver(_))),
                    "spd_logdet must refuse n={n} scale={scale:e} diag(1,..,1e-17)"
                );
                assert!(
                    matches!(
                        dense_logabsdet(sub_tol.view()),
                        Err(SaddleError::Singular(_))
                    ),
                    "dense_logabsdet must refuse n={n} scale={scale:e} diag(1,..,1e-17)"
                );
                assert!(
                    matches!(
                        solve_dense(sub_tol.view(), Array1::from_elem(n, scale).view()),
                        Err(SaddleError::Singular(_))
                    ),
                    "solve_dense must refuse n={n} scale={scale:e} diag(1,..,1e-17)"
                );
                assert!(
                    matches!(
                        solve_dense_matrix(sub_tol.view(), Array2::eye(n).view()),
                        Err(SaddleError::Singular(_))
                    ),
                    "solve_dense_matrix must refuse n={n} scale={scale:e} diag(1,..,1e-17)"
                );
            }
        }

        // relative_zero_tolerance guard: non-finite, non-positive, or n == 0 return 0.0.
        assert_eq!(relative_zero_tolerance(0.0, 8), 0.0);
        assert_eq!(relative_zero_tolerance(-0.0, 8), 0.0);
        assert_eq!(relative_zero_tolerance(-3.5, 8), 0.0);
        assert_eq!(relative_zero_tolerance(f64::NAN, 8), 0.0);
        assert_eq!(relative_zero_tolerance(f64::INFINITY, 8), 0.0);
        assert_eq!(relative_zero_tolerance(f64::NEG_INFINITY, 8), 0.0);
        assert_eq!(relative_zero_tolerance(2.5, 0), 0.0);
        assert_eq!(relative_zero_tolerance(2.5, 8), 2.5 * 8.0 * f64::EPSILON);
    }

    fn coupled_stiff_rings_192() -> (Array2<f64>, f64) {
        let l = 64usize;
        let rings = 3usize;
        let n = rings * l;
        let mut a = Array2::<f64>::zeros((n, n));
        let c = 1.0e8_f64;
        let mut max_abs = 0.0_f64;
        for ring in 0..rings {
            let base = ring * l;
            let split = (ring as f64) * 1.0e-5;
            for i in 0..l {
                let v = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / (l as f64)).cos();
                let diag = 2.0 * c + 37.0 * v + split;
                a[(base + i, base + i)] = diag;
                a[(base + i, base + (i + 1) % l)] = -c;
                a[(base + (i + 1) % l, base + i)] = -c;
                if diag.abs() > max_abs {
                    max_abs = diag.abs();
                }
            }
        }
        for i in 0..l {
            for r1 in 0..rings {
                for r2 in (r1 + 1)..rings {
                    let coup = 1.0e-5 * ((i + r1 + r2) as f64 + 1.0);
                    a[(r1 * l + i, r2 * l + i)] = coup;
                    a[(r2 * l + i, r1 * l + i)] = coup;
                }
            }
        }
        (a, max_abs)
    }

    #[test]
    fn jacobi_eigh_refuses_to_return_ok_when_sweep_budget_is_exhausted() {
        let a_small = array![[4.0, 1.0, 0.5], [1.0, 2.0, -0.7], [0.5, -0.7, 3.0]];
        assert!(matches!(
            with_jacobi_sweep_budget(1, || exact_eigh(a_small.view())),
            Err(SaddleError::Solver(_))
        ));
        #[cfg(feature = "faer")]
        {
            let (coupled, _) = coupled_stiff_rings_192();
            assert!(matches!(
                with_jacobi_sweep_budget(1, || faer_eigh(&coupled)),
                Err(SaddleError::Solver(_))
            ));
            assert!(matches!(
                with_jacobi_sweep_budget(1, || exact_eigh(coupled.view())),
                Err(SaddleError::Solver(_))
            ));
        }
    }

    #[test]
    fn faer_eigh_rayleigh_ritz_refinement_reaches_machine_precision() {
        let (coupled, coupled_max_abs) = coupled_stiff_rings_192();
        let coupled_n = coupled.nrows();
        let (c_evals, c_evecs) = exact_eigh(coupled.view()).unwrap();
        let (f_evals, f_evecs) = faer_eigh(&coupled).unwrap();
        let c_only = exact_eigvalsh(coupled.view()).unwrap();
        let c_bound = relative_zero_tolerance(coupled_max_abs, coupled_n);
        let c_orth_bound = (coupled_n as f64) * f64::EPSILON;
        for j in 0..coupled_n {
            assert_eq!(c_evals[j].to_bits(), f_evals[j].to_bits());
            let dlam = (c_evals[j] - c_only[j]).abs();
            assert!(dlam <= c_bound, "coupled j={j} dlam={dlam:e} > {c_bound:e}");
        }
        let c_av = matmul(coupled.view(), f_evecs.view());
        let c_vtv = matmul(f_evecs.t(), f_evecs.view());
        for i in 0..coupled_n {
            for j in 0..coupled_n {
                assert_eq!(c_evecs[(i, j)].to_bits(), f_evecs[(i, j)].to_bits());
                let res = (c_av[(i, j)] - f_evals[j] * f_evecs[(i, j)]).abs();
                assert!(
                    res <= c_bound,
                    "coupled ({i},{j}) res={res:e} > {c_bound:e}"
                );
                let expect = if i == j { 1.0 } else { 0.0 };
                let orth = (c_vtv[(i, j)] - expect).abs();
                assert!(
                    orth <= c_orth_bound,
                    "coupled ({i},{j}) orth={orth:e} > {c_orth_bound:e}"
                );
            }
        }

        for &n in &[96usize, 192usize, 256usize] {
            let mut a = Array2::<f64>::zeros((n, n));
            let c = (n * n) as f64;
            let mut max_abs = 0.0_f64;
            for i in 0..n {
                let v = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / (n as f64)).cos();
                let diag = 2.0 * c + 37.0 * v;
                a[(i, i)] = diag;
                a[(i, (i + 1) % n)] = -c;
                a[((i + 1) % n, i)] = -c;
                if diag.abs() > max_abs {
                    max_abs = diag.abs();
                }
            }
            let (evals, evecs) = faer_eigh(&a).unwrap();
            let evals_only = exact_eigvalsh(a.view()).unwrap();
            let bound = relative_zero_tolerance(max_abs, n);
            let orth_bound = (n as f64) * f64::EPSILON;
            for j in 0..n {
                let dlam = (evals[j] - evals_only[j]).abs();
                assert!(dlam <= bound, "n={n} j={j} dlam={dlam:e} > {bound:e}");
            }
            let av = matmul(a.view(), evecs.view());
            let vt_v = matmul(evecs.t(), evecs.view());
            for i in 0..n {
                for j in 0..n {
                    let res = (av[(i, j)] - evals[j] * evecs[(i, j)]).abs();
                    assert!(res <= bound, "n={n} ({i},{j}) res={res:e} > {bound:e}");
                    let expect = if i == j { 1.0 } else { 0.0 };
                    let orth = (vt_v[(i, j)] - expect).abs();
                    assert!(
                        orth <= orth_bound,
                        "n={n} ({i},{j}) orth={orth:e} > {orth_bound:e}"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "faer")]
    fn faer_eigh_residual_gate_fails_closed_on_nan() {
        let mut av = faer::Mat::<f64>::zeros(2, 2);
        av[(0, 0)] = 1.0;
        av[(1, 0)] = f64::NAN;
        let vectors = faer::Mat::<f64>::identity(2, 2);
        let values = faer::Col::<f64>::from_fn(2, |_| 1.0);
        let res = max_eigenpair_residual(&av, vectors.as_ref(), values.as_ref().as_diagonal(), 2);
        assert!(
            matches!(res, Err(SaddleError::NonFinite("faer_eigh residual"))),
            "expected NonFinite on NaN residual, got {res:?}"
        );
    }

    #[test]
    fn jacobi_eigh_diagonalizes_sub_1e14_off_diagonals_down_to_relative_tolerance() {
        let s = 1.0e-20_f64;
        let a_scaled = array![[2.0 * s, s], [s, 2.0 * s]];
        let (evals_s, evecs_s) = exact_eigh(a_scaled.view()).unwrap();
        let tol_s = relative_zero_tolerance(2.0 * s, 2);
        assert!(
            (evals_s[0] - s).abs() <= tol_s && (evals_s[1] - 3.0 * s).abs() <= tol_s,
            "sub-1e-14 scaled matrix was not diagonalized: evals={evals_s:?}"
        );
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (evecs_s[(0, 0)].abs() - inv_sqrt2).abs() <= 1.0e-12
                && (evecs_s[(1, 0)].abs() - inv_sqrt2).abs() <= 1.0e-12,
            "sub-1e-14 scaled eigenvectors did not rotate: {evecs_s:?}"
        );

        let delta = 5.0e-15_f64;
        let a_unit = array![[1.0, delta], [delta, 1.0]];
        let (evals_u, evecs_u) = exact_eigh(a_unit.view()).unwrap();
        let tol_u = relative_zero_tolerance(1.0, 2);
        assert!(delta > tol_u && delta < 1.0e-14);
        assert!(
            (evals_u[0] - (1.0 - delta)).abs() <= tol_u
                && (evals_u[1] - (1.0 + delta)).abs() <= tol_u,
            "off-diagonal in (tol, 1e-14) was not diagonalized: evals={evals_u:?}"
        );
        let av_u = matmul(a_unit.view(), evecs_u.view());
        for i in 0..2 {
            for j in 0..2 {
                let res = (av_u[(i, j)] - evals_u[j] * evecs_u[(i, j)]).abs();
                assert!(res <= tol_u, "({i},{j}) res={res:e} > tol_u={tol_u:e}");
            }
        }
    }
}
