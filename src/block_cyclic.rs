//! Cyclic block-tridiagonal inertia.
//!
//! Diagonal blocks are the caller-supplied `n × n` matrices. Every neighbor
//! coupling, including the corner that closes the ring, is `-spring` times
//! the identity. The open chain is eliminated by block Thomas. The closing
//! bead is the Schur complement of that chain, and inertia adds across the
//! complements (Haynsworth).
//!
//! Each accepted complement is factored once. Pivot selection reads
//! eigenvalues of the original blocks and does not form those eigenvalues
//! again while eliminating one rotation. The inverse is `V diag(λ⁻¹) Vᵀ`,
//! formed by one GEMM, and later block products use the same GEMM.
//! [`CyclicFactor::solve`] applies one correction when the cyclic residual
//! exceeds [`SOLVE_REFINE_TOLERANCE`] times the right-hand scale.
//!
//! A block must be symmetric within the relative zero tolerance of its
//! largest absolute entry. A larger skew is [`SaddleError::Invalid`],
//! because the factor would otherwise use the symmetric part while the
//! residual used the original entries. A smaller skew is replaced by that
//! symmetric part before the factor and the residual are formed.
//!
//! The best-conditioned block is eliminated first. A zero complement of
//! that rotation is not, by itself, rank deficiency: later rotations are
//! tried. When every rotation meets a zero complement, the assembled
//! spectrum is the rank test. [`SaddleError::Singular`] is returned only
//! when an assembled eigenvalue lies inside the relative zero tolerance,
//! and the index is the preferred rotation's complement.

use ndarray::Array2;

use crate::eigensolve::{exact_eigvalsh, faer_eigh, relative_zero_tolerance};
use crate::error::SaddleError;

/// One iterative correction is applied above this absolute residual scale.
pub const SOLVE_REFINE_TOLERANCE: f64 = 1e-11;

/// Log-absolute-determinant and negative inertia of one cyclic factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockInertia {
    /// Sum of `ln|λ|` over eigenvalues outside the relative zero tolerance.
    pub log_abs_det: f64,
    /// Count of eigenvalues below the negative side of that tolerance.
    pub n_negative: usize,
}

/// Outcome of [`CyclicFactor::solve_report`].
#[derive(Clone, Debug, PartialEq)]
pub struct SolveReport {
    /// Solution in the caller's bead order.
    pub solution: Vec<f64>,
    /// Euclidean norm of `Mx − b` after the optional correction.
    pub residual: f64,
    /// `1` when a correction was applied, `0` otherwise.
    pub corrections: usize,
}

/// Factor of a cyclic block-tridiagonal matrix with scalar-spring corners.
#[derive(Clone, Debug)]
pub struct CyclicFactor {
    inv_f: Vec<Vec<f64>>,
    inv_s: Vec<f64>,
    start: usize,
    n: usize,
    p: usize,
    spring: f64,
    blocks: Vec<Vec<f64>>,
    /// Row-major inverse of the assembled matrix when no rotation factors.
    /// `None` when `inv_f` and `inv_s` are the block Thomas factors.
    dense_inv: Option<Vec<f64>>,
}

/// Log-absolute-determinant and inertia of one cyclic block-tridiagonal matrix.
///
/// A pivot inside the relative zero tolerance, or a non-finite pivot, is an
/// error. Substituting a finite value for that pivot would change every later
/// Schur complement.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when there are fewer than three blocks, `n` is
/// zero, or a block is asymmetric beyond the relative zero tolerance of its
/// largest absolute entry, [`SaddleError::Shape`] when a block is not
/// `n × n`, [`SaddleError::NonFinite`] when an entry or a pivot is
/// non-finite, and [`SaddleError::Singular`] when the assembled matrix has
/// an eigenvalue inside that tolerance. A zero complement of one rotation
/// is not singular when another rotation, or the assembled spectrum, has
/// full rank. The singular index is the preferred rotation's complement.
pub fn logdet_cyclic_blocks(
    diag: &[Vec<f64>],
    n: usize,
    spring: f64,
) -> Result<BlockInertia, SaddleError> {
    logdet_mode(diag, n, spring)
}

/// Log-absolute-determinant and inertia that refuses a zero-tolerance pivot.
///
/// This is the same computation as [`logdet_cyclic_blocks`].
///
/// # Errors
///
/// Same as [`logdet_cyclic_blocks`].
pub fn logdet_cyclic_strict(
    diag: &[Vec<f64>],
    n: usize,
    spring: f64,
) -> Result<BlockInertia, SaddleError> {
    logdet_mode(diag, n, spring)
}

/// Factor `diag` after adding `tau` to every diagonal entry.
///
/// The factor is the open-chain inverses plus the closing Schur inverse.
/// [`CyclicFactor::solve`] refines one step at [`SOLVE_REFINE_TOLERANCE`].
///
/// # Errors
///
/// [`SaddleError::Invalid`] below three blocks, for `n == 0`, or for an
/// asymmetric block, [`SaddleError::Singular`] when the assembled spectrum
/// has an eigenvalue inside the relative zero tolerance, and the shape and
/// finiteness errors of [`logdet_cyclic_blocks`].
pub fn factor_cyclic(
    diag: &[Vec<f64>],
    n: usize,
    spring: f64,
    tau: f64,
) -> Result<CyclicFactor, SaddleError> {
    if !spring.is_finite() || !tau.is_finite() {
        return Err(SaddleError::NonFinite("cyclic factor"));
    }
    let p = diag.len();
    if p < 3 || n == 0 {
        return Err(SaddleError::Invalid(
            "cyclic factor needs at least 3 beads and a positive block size".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(p);
    for block in diag {
        check_block(block, n, "cyclic factor")?;
        let mut shifted = block.clone();
        symmetrize_block(&mut shifted, n);
        for i in 0..n {
            shifted[i * n + i] += tau;
        }
        if shifted.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("cyclic factor shift"));
        }
        blocks.push(shifted);
    }
    let eliminated = eliminate(&blocks, n, spring)?;
    Ok(CyclicFactor {
        inv_f: eliminated.inv_f,
        inv_s: eliminated.inv_s,
        start: eliminated.start,
        n,
        p,
        spring,
        blocks,
        dense_inv: eliminated.dense_inv,
    })
}

impl CyclicFactor {
    /// Solve `Mx = rhs` and correct once when the residual exceeds
    /// [`SOLVE_REFINE_TOLERANCE`] times `‖rhs‖`. A zero right-hand side
    /// refines when the residual is positive.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when `rhs` does not have `beads * n` entries,
    /// and [`SaddleError::NonFinite`] when the solution is non-finite.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SaddleError> {
        Ok(self.solve_report(rhs)?.solution)
    }

    /// [`CyclicFactor::solve`] plus the residual norm and the correction count.
    ///
    /// # Errors
    ///
    /// Same as [`CyclicFactor::solve`].
    pub fn solve_report(&self, rhs: &[f64]) -> Result<SolveReport, SaddleError> {
        let mut solution = self.solve_once(rhs)?;
        let mut residual_vec = cyclic_residual(&self.blocks, self.n, self.spring, &solution, rhs);
        let rhs_norm = euclidean(rhs);
        let mut residual = euclidean(&residual_vec);
        let mut corrections = 0usize;
        if refine_triggered(residual, rhs_norm) {
            let step = self.solve_once(&residual_vec)?;
            for (slot, delta) in solution.iter_mut().zip(&step) {
                *slot -= delta;
            }
            corrections = 1;
            residual_vec = cyclic_residual(&self.blocks, self.n, self.spring, &solution, rhs);
            residual = euclidean(&residual_vec);
        }
        if solution.iter().any(|v| !v.is_finite()) || !residual.is_finite() {
            return Err(SaddleError::NonFinite("cyclic solve"));
        }
        Ok(SolveReport {
            solution,
            residual,
            corrections,
        })
    }

    fn solve_once(&self, rhs: &[f64]) -> Result<Vec<f64>, SaddleError> {
        let n = self.n;
        let p = self.p;
        if rhs.len() != p * n {
            return Err(SaddleError::Shape(format!(
                "cyclic solve expected {} entries, got {}",
                p * n,
                rhs.len()
            )));
        }
        if let Some(inv) = &self.dense_inv {
            return Ok(matvec(inv, rhs, p * n));
        }
        let mut rot_rhs = vec![0.0; p * n];
        for k in 0..p {
            let src = (self.start + k) % p;
            for a in 0..n {
                rot_rhs[k * n + a] = rhs[src * n + a];
            }
        }
        let open_rhs = &rot_rhs[..(p - 1) * n];
        let y_rhs = solve_open_rhs(&self.inv_f, n, self.spring, open_rhs);
        let mut s_rhs = vec![0.0; n];
        for a in 0..n {
            s_rhs[a] = rot_rhs[(p - 1) * n + a] + self.spring * (y_rhs[a] + y_rhs[(p - 2) * n + a]);
        }
        let x_last = matvec(&self.inv_s, &s_rhs, n);
        let mut open_corrected = open_rhs.to_vec();
        for a in 0..n {
            open_corrected[a] += self.spring * x_last[a];
            open_corrected[(p - 2) * n + a] += self.spring * x_last[a];
        }
        let x_open = solve_open_rhs(&self.inv_f, n, self.spring, &open_corrected);
        let mut rot_x = vec![0.0; p * n];
        rot_x[..(p - 1) * n].copy_from_slice(&x_open);
        for a in 0..n {
            rot_x[(p - 1) * n + a] = x_last[a];
        }
        Ok(unrotate(&rot_x, n, p, self.start))
    }
}

fn logdet_mode(diag: &[Vec<f64>], n: usize, spring: f64) -> Result<BlockInertia, SaddleError> {
    if !spring.is_finite() {
        return Err(SaddleError::NonFinite("cyclic logdet spring"));
    }
    let p = diag.len();
    if p < 3 || n == 0 {
        return Err(SaddleError::Invalid(
            "cyclic block logdet needs at least 3 beads and a positive block size".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(p);
    for block in diag {
        check_block(block, n, "cyclic logdet")?;
        let mut owned = block.clone();
        symmetrize_block(&mut owned, n);
        blocks.push(owned);
    }
    let eliminated = eliminate(&blocks, n, spring)?;
    if !eliminated.log_abs.is_finite() {
        return Err(SaddleError::NonFinite("cyclic logdet"));
    }
    Ok(BlockInertia {
        log_abs_det: eliminated.log_abs,
        n_negative: eliminated.n_neg,
    })
}

fn check_block(block: &[f64], n: usize, what: &'static str) -> Result<(), SaddleError> {
    let Some(cells) = n.checked_mul(n) else {
        return Err(SaddleError::Shape(format!(
            "{what} dimension {n} overflows"
        )));
    };
    if block.len() != cells {
        return Err(SaddleError::Shape(format!(
            "{what} expected {cells} entries for a {n} by {n} block, got {}",
            block.len()
        )));
    }
    if block.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite(what));
    }
    let mut scale = 0.0_f64;
    for value in block {
        scale = scale.max(value.abs());
    }
    let tolerance = relative_zero_tolerance(scale, n);
    for row in 0..n {
        for col in (row + 1)..n {
            let gap = (block[row * n + col] - block[col * n + row]).abs();
            if gap > tolerance {
                return Err(SaddleError::Invalid("cyclic block is asymmetric".into()));
            }
        }
    }
    Ok(())
}

fn symmetrize_block(block: &mut [f64], n: usize) {
    for row in 0..n {
        for col in (row + 1)..n {
            let upper = block[row * n + col];
            let lower = block[col * n + row];
            if upper != lower {
                let mean = 0.5 * (upper + lower);
                block[row * n + col] = mean;
                block[col * n + row] = mean;
            }
        }
    }
}

fn renumber(err: SaddleError, index: usize) -> SaddleError {
    match err {
        SaddleError::Singular(_) => SaddleError::Singular(index),
        other => other,
    }
}

fn best_pivot_index(diag: &[Vec<f64>], n: usize) -> Result<usize, SaddleError> {
    let mut best = 0usize;
    let mut best_q = pivot_quality(&diag[0], n)?;
    for (index, block) in diag.iter().enumerate().skip(1) {
        let quality = pivot_quality(block, n)?;
        if quality > best_q {
            best_q = quality;
            best = index;
        }
    }
    Ok(best)
}

fn pivot_quality(block: &[f64], n: usize) -> Result<f64, SaddleError> {
    let arr = Array2::from_shape_vec((n, n), block.to_vec())
        .map_err(|_| SaddleError::Shape(format!("pivot block is not {n} by {n}")))?;
    let evals = exact_eigvalsh(arr.view())?;
    let mut max_abs = 0.0_f64;
    let mut min_abs = f64::INFINITY;
    for lam in evals.iter().copied() {
        let magnitude = lam.abs();
        if magnitude > max_abs {
            max_abs = magnitude;
        }
        if magnitude < min_abs {
            min_abs = magnitude;
        }
    }
    if max_abs == 0.0 || !min_abs.is_finite() {
        Ok(0.0)
    } else {
        Ok(min_abs / max_abs)
    }
}

struct Eliminated {
    start: usize,
    inv_f: Vec<Vec<f64>>,
    inv_s: Vec<f64>,
    log_abs: f64,
    n_neg: usize,
    dense_inv: Option<Vec<f64>>,
}

fn eliminate(diag: &[Vec<f64>], n: usize, spring: f64) -> Result<Eliminated, SaddleError> {
    let preferred = best_pivot_index(diag, n)?;
    let p = diag.len();
    let mut preferred_singular: Option<SaddleError> = None;
    for offset in 0..p {
        let start = (preferred + offset) % p;
        match factor_rotation(diag, n, spring, start) {
            Ok(eliminated) => return Ok(eliminated),
            Err(SaddleError::Singular(index)) => {
                if offset == 0 {
                    preferred_singular = Some(SaddleError::Singular(index));
                }
            }
            Err(err) => return Err(err),
        }
    }
    match assembled_factor(diag, n, spring, preferred)? {
        Some(eliminated) => Ok(eliminated),
        None => Err(preferred_singular.unwrap_or(SaddleError::Singular(0))),
    }
}

fn factor_rotation(
    diag: &[Vec<f64>],
    n: usize,
    spring: f64,
    start: usize,
) -> Result<Eliminated, SaddleError> {
    let rotated = rotate(diag, start);
    let p = rotated.len();
    let (inv_f, mut log_abs, mut n_neg) = factor_open_logged(&rotated[..p - 1], n, spring)?;
    let schur = closing_schur(&rotated, &inv_f, n, spring)?;
    let (inv_s, log_s, neg_s) =
        inverse_and_inertia(&schur, n).map_err(|err| renumber(err, p - 1))?;
    log_abs += log_s;
    n_neg += neg_s;
    Ok(Eliminated {
        start,
        inv_f,
        inv_s,
        log_abs,
        n_neg,
        dense_inv: None,
    })
}

fn assembled_factor(
    diag: &[Vec<f64>],
    n: usize,
    spring: f64,
    start: usize,
) -> Result<Option<Eliminated>, SaddleError> {
    let p = diag.len();
    let ntot = p
        .checked_mul(n)
        .ok_or_else(|| SaddleError::Shape("cyclic dimension overflows".into()))?;
    let mut matrix = Array2::zeros((ntot, ntot));
    for bead in 0..p {
        for row in 0..n {
            for col in 0..n {
                matrix[(bead * n + row, bead * n + col)] = diag[bead][row * n + col];
            }
        }
        let next = (bead + 1) % p;
        for row in 0..n {
            matrix[(bead * n + row, next * n + row)] -= spring;
            matrix[(next * n + row, bead * n + row)] -= spring;
        }
    }
    let (evals, vectors) = faer_eigh(&matrix)?;
    let mut scale = 0.0_f64;
    for lam in evals.iter().copied() {
        scale = scale.max(lam.abs());
    }
    let floor = relative_zero_tolerance(scale, ntot);
    let mut log_abs = 0.0;
    let mut n_neg = 0usize;
    let mut inv_lam = vec![0.0; ntot];
    for (col, lam) in evals.iter().copied().enumerate() {
        if lam.abs() <= floor {
            return Ok(None);
        }
        log_abs += lam.abs().ln();
        if lam < -floor {
            n_neg += 1;
        }
        inv_lam[col] = 1.0 / lam;
    }
    if !log_abs.is_finite() {
        return Err(SaddleError::NonFinite("cyclic logdet"));
    }
    let inv = gemm_symmetric_inverse(&vectors, &inv_lam)?;
    Ok(Some(Eliminated {
        start,
        inv_f: Vec::new(),
        inv_s: Vec::new(),
        log_abs,
        n_neg,
        dense_inv: Some(inv),
    }))
}

fn rotate(diag: &[Vec<f64>], start: usize) -> Vec<Vec<f64>> {
    let p = diag.len();
    let mut out = Vec::with_capacity(p);
    for k in 0..p {
        out.push(diag[(start + k) % p].clone());
    }
    out
}

fn factor_open_logged(
    diag: &[Vec<f64>],
    n: usize,
    spring: f64,
) -> Result<(Vec<Vec<f64>>, f64, usize), SaddleError> {
    let mut inv_f = Vec::with_capacity(diag.len());
    let mut log_abs = 0.0;
    let mut n_neg = 0usize;
    let (inv0, log0, neg0) = inverse_and_inertia(&diag[0], n).map_err(|err| renumber(err, 0))?;
    log_abs += log0;
    n_neg += neg0;
    inv_f.push(inv0);
    for index in 1..diag.len() {
        let coupled = scale_by_spring_square(&inv_f[index - 1], spring);
        let fi = sub(&diag[index], &coupled);
        let (inv, log_i, neg_i) =
            inverse_and_inertia(&fi, n).map_err(|err| renumber(err, index))?;
        log_abs += log_i;
        n_neg += neg_i;
        inv_f.push(inv);
    }
    Ok((inv_f, log_abs, n_neg))
}

fn closing_schur(
    diag: &[Vec<f64>],
    inv_f: &[Vec<f64>],
    n: usize,
    spring: f64,
) -> Result<Vec<f64>, SaddleError> {
    let p_open = inv_f.len();
    let y0 = solve_open_unit(inv_f, n, spring, 0);
    let y_last = solve_open_unit(inv_f, n, spring, p_open - 1);
    let mut mix = y0[0].clone();
    add_block(&mut mix, &y0[p_open - 1]);
    add_block(&mut mix, &y_last[0]);
    add_block(&mut mix, &y_last[p_open - 1]);
    let schur = sub(&diag[p_open], &scale_by_spring_square(&mix, spring));
    if schur.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("cyclic Schur complement"));
    }
    Ok(schur)
}

fn inverse_and_inertia(block: &[f64], n: usize) -> Result<(Vec<f64>, f64, usize), SaddleError> {
    let arr = Array2::from_shape_vec((n, n), block.to_vec())
        .map_err(|_| SaddleError::Shape(format!("inertia block is not {n} by {n}")))?;
    let (evals, vectors) = faer_eigh(&arr)?;
    if evals.iter().any(|lam| !lam.is_finite()) {
        return Err(SaddleError::NonFinite("cyclic pivot"));
    }
    let mut scale = 0.0_f64;
    for lam in evals.iter().copied() {
        if lam.abs() > scale {
            scale = lam.abs();
        }
    }
    let floor = relative_zero_tolerance(scale, n);
    let mut log_abs = 0.0;
    let mut n_neg = 0usize;
    let mut inv_lam = vec![0.0; n];
    for (col, lam) in evals.iter().copied().enumerate() {
        if lam.abs() <= floor {
            return Err(SaddleError::Singular(0));
        }
        log_abs += lam.abs().ln();
        if lam < -floor {
            n_neg += 1;
        }
        inv_lam[col] = 1.0 / lam;
    }
    let inv = gemm_symmetric_inverse(&vectors, &inv_lam)?;
    Ok((inv, log_abs, n_neg))
}

fn gemm_symmetric_inverse(vectors: &Array2<f64>, inv_lam: &[f64]) -> Result<Vec<f64>, SaddleError> {
    let n = inv_lam.len();
    let scaled = faer::Mat::from_fn(n, n, |row, col| vectors[(row, col)] * inv_lam[col]);
    let transposed = faer::Mat::from_fn(n, n, |row, col| vectors[(col, row)]);
    let mut product = faer::Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(
        &mut product,
        faer::Accum::Replace,
        &scaled,
        &transposed,
        1.0,
        faer::Par::Seq,
    );
    let mut out = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let value = product[(row, col)];
            if !value.is_finite() {
                return Err(SaddleError::NonFinite("cyclic inverse"));
            }
            out[row * n + col] = value;
        }
    }
    Ok(out)
}

fn solve_open_unit(inv_f: &[Vec<f64>], n: usize, spring: f64, slot: usize) -> Vec<Vec<f64>> {
    let p = inv_f.len();
    let mut rhs = vec![vec![0.0; n * n]; p];
    for i in 0..n {
        rhs[slot][i * n + i] = 1.0;
    }
    let mut forward = Vec::with_capacity(p);
    forward.push(gemm(&inv_f[0], &rhs[0], n));
    for index in 1..p {
        let mut right = rhs[index].clone();
        add_block(&mut right, &scale(&forward[index - 1], spring));
        forward.push(gemm(&inv_f[index], &right, n));
    }
    let mut x = vec![vec![0.0; n * n]; p];
    x[p - 1] = forward[p - 1].clone();
    for index in (0..p - 1).rev() {
        let extra = gemm(&inv_f[index], &scale(&x[index + 1], spring), n);
        let mut acc = forward[index].clone();
        add_block(&mut acc, &extra);
        x[index] = acc;
    }
    x
}

fn solve_open_rhs(inv_f: &[Vec<f64>], n: usize, spring: f64, rhs: &[f64]) -> Vec<f64> {
    let p = inv_f.len();
    let mut forward = vec![0.0; p * n];
    let first = matvec(&inv_f[0], &rhs[..n], n);
    forward[..n].copy_from_slice(&first);
    for index in 1..p {
        let mut right = vec![0.0; n];
        for a in 0..n {
            right[a] = rhs[index * n + a] + spring * forward[(index - 1) * n + a];
        }
        let step = matvec(&inv_f[index], &right, n);
        forward[index * n..(index + 1) * n].copy_from_slice(&step);
    }
    let mut x = vec![0.0; p * n];
    x[(p - 1) * n..].copy_from_slice(&forward[(p - 1) * n..]);
    for index in (0..p - 1).rev() {
        let mut nxt = vec![0.0; n];
        for a in 0..n {
            nxt[a] = spring * x[(index + 1) * n + a];
        }
        let extra = matvec(&inv_f[index], &nxt, n);
        for a in 0..n {
            x[index * n + a] = forward[index * n + a] + extra[a];
        }
    }
    x
}

fn gemm(left: &[f64], right: &[f64], n: usize) -> Vec<f64> {
    let a = faer::Mat::from_fn(n, n, |row, col| left[row * n + col]);
    let b = faer::Mat::from_fn(n, n, |row, col| right[row * n + col]);
    let mut c = faer::Mat::<f64>::zeros(n, n);
    faer::linalg::matmul::matmul(&mut c, faer::Accum::Replace, &a, &b, 1.0, faer::Par::Seq);
    let mut out = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            out[row * n + col] = c[(row, col)];
        }
    }
    out
}

fn matvec(matrix: &[f64], vector: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for row in 0..n {
        let mut sum = 0.0;
        for col in 0..n {
            sum += matrix[row * n + col] * vector[col];
        }
        y[row] = sum;
    }
    y
}

fn scale(matrix: &[f64], factor: f64) -> Vec<f64> {
    matrix.iter().map(|value| value * factor).collect()
}

// `spring * spring` overflows for a large finite spring. Scaling twice keeps
// a well-conditioned complement finite.
fn scale_by_spring_square(matrix: &[f64], spring: f64) -> Vec<f64> {
    scale(&scale(matrix, spring), spring)
}

fn sub(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| a - b).collect()
}

fn add_block(left: &mut [f64], right: &[f64]) {
    for (slot, value) in left.iter_mut().zip(right) {
        *slot += value;
    }
}

fn unrotate(rotated: &[f64], n: usize, p: usize, start: usize) -> Vec<f64> {
    let mut x = vec![0.0; p * n];
    for k in 0..p {
        let dest = (start + k) % p;
        x[dest * n..(dest + 1) * n].copy_from_slice(&rotated[k * n..(k + 1) * n]);
    }
    x
}

fn cyclic_residual(
    blocks: &[Vec<f64>],
    n: usize,
    spring: f64,
    solution: &[f64],
    rhs: &[f64],
) -> Vec<f64> {
    let applied = cyclic_matvec(blocks, n, spring, solution);
    applied.into_iter().zip(rhs).map(|(y, b)| y - b).collect()
}

fn cyclic_matvec(blocks: &[Vec<f64>], n: usize, spring: f64, vector: &[f64]) -> Vec<f64> {
    let p = blocks.len();
    let mut y = vec![0.0; p * n];
    for bead in 0..p {
        let previous = (bead + p - 1) % p;
        let next = (bead + 1) % p;
        for row in 0..n {
            let mut sum = 0.0;
            for col in 0..n {
                sum += blocks[bead][row * n + col] * vector[bead * n + col];
            }
            sum -= spring * vector[previous * n + row];
            sum -= spring * vector[next * n + row];
            y[bead * n + row] = sum;
        }
    }
    y
}

// Summing squares overflows when a finite entry is near 1e200. Scale the
// accumulator the way LAPACK `dnrm2` does, then multiply by the square root.
fn euclidean(values: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for value in values {
        if *value == 0.0 {
            continue;
        }
        let abs_v = value.abs();
        if scale < abs_v {
            let ratio = scale / abs_v;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = abs_v;
        } else {
            let ratio = abs_v / scale;
            ssq += ratio * ratio;
        }
    }
    scale * ssq.sqrt()
}

fn refine_triggered(residual: f64, rhs_norm: f64) -> bool {
    if rhs_norm == 0.0 {
        residual > 0.0
    } else {
        residual > SOLVE_REFINE_TOLERANCE * rhs_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_cyclic(diag: &[Vec<f64>], n: usize, spring: f64) -> Array2<f64> {
        let p = diag.len();
        let ntot = p * n;
        let mut matrix = Array2::zeros((ntot, ntot));
        for bead in 0..p {
            for row in 0..n {
                for col in 0..n {
                    matrix[(bead * n + row, bead * n + col)] = diag[bead][row * n + col];
                }
            }
            let next = (bead + 1) % p;
            for row in 0..n {
                matrix[(bead * n + row, next * n + row)] -= spring;
                matrix[(next * n + row, bead * n + row)] -= spring;
            }
        }
        matrix
    }

    fn dense_inertia(matrix: &Array2<f64>) -> (f64, usize) {
        let evals = exact_eigvalsh(matrix.view()).expect("dense spectrum");
        let mut scale = 0.0_f64;
        for lam in evals.iter().copied() {
            if lam.abs() > scale {
                scale = lam.abs();
            }
        }
        let floor = relative_zero_tolerance(scale, evals.len());
        let mut log_abs = 0.0;
        let mut n_neg = 0usize;
        for lam in evals.iter().copied() {
            if lam.abs() > floor {
                log_abs += lam.abs().ln();
                if lam < -floor {
                    n_neg += 1;
                }
            }
        }
        (log_abs, n_neg)
    }

    fn sample_blocks(n: usize, p: usize) -> (Vec<Vec<f64>>, f64) {
        let spring = 0.35;
        let mut diag = Vec::with_capacity(p);
        for bead in 0..p {
            let mut block = vec![0.0; n * n];
            for i in 0..n {
                block[i * n + i] = 2.5 + 0.2 * ((bead + 2 * i) % 5) as f64;
                if i + 1 < n {
                    block[i * n + i + 1] = -0.15;
                    block[(i + 1) * n + i] = -0.15;
                }
            }
            if n > 2 {
                block[2] = 0.05;
                block[2 * n] = 0.05;
            }
            diag.push(block);
        }
        (diag, spring)
    }

    #[test]
    fn strict_logdet_matches_one_dense_spectrum() {
        let (mut diag, spring) = sample_blocks(3, 4);
        diag[0][0] = -4.0;
        let got = logdet_cyclic_strict(&diag, 3, spring).expect("strict");
        let (log_abs, n_neg) = dense_inertia(&dense_cyclic(&diag, 3, spring));
        assert!(
            (got.log_abs_det - log_abs).abs() < 1e-8,
            "{} vs {log_abs}",
            got.log_abs_det
        );
        assert_eq!(got.n_negative, n_neg);
        assert!(n_neg >= 1, "the planted diagonal is indefinite");
    }

    #[test]
    fn block_logdet_matches_the_same_spectrum_on_a_shifted_indefinite_block() {
        let (mut diag, spring) = sample_blocks(3, 5);
        diag[2][0] = -1.5;
        let got = logdet_cyclic_blocks(&diag, 3, spring).expect("blocks");
        let (log_abs, n_neg) = dense_inertia(&dense_cyclic(&diag, 3, spring));
        assert!((got.log_abs_det - log_abs).abs() < 1e-8);
        assert_eq!(got.n_negative, n_neg);
    }

    #[test]
    fn strict_factor_refuses_the_exact_spring_zero() {
        let n = 2usize;
        let p = 6usize;
        let spring = 1.0;
        let mut diag = Vec::new();
        for _ in 0..p {
            let mut block = vec![0.0; n * n];
            block[0] = 2.0 * spring;
            block[3] = 4.0 + 2.0 * spring;
            diag.push(block);
        }
        let err = logdet_cyclic_strict(&diag, n, spring).expect_err("zero mode");
        assert!(matches!(err, SaddleError::Singular(_)), "{err}");
        let err = logdet_cyclic_blocks(&diag, n, spring).expect_err("zero mode");
        assert!(matches!(err, SaddleError::Singular(_)), "{err}");
        let err = factor_cyclic(&diag, n, spring, 0.0).expect_err("zero mode");
        assert!(matches!(err, SaddleError::Singular(_)), "{err}");
        let zeros = vec![vec![0.0; n * n]; p];
        let err = factor_cyclic(&zeros, n, 0.0, 0.0).expect_err("empty complement");
        assert!(matches!(err, SaddleError::Singular(_)), "{err}");
    }

    #[test]
    fn three_scalar_zero_blocks_with_spring_one_are_ln2() {
        // Diagonals are zero, so the first complement pivot is zero, but the
        // assembled ring has eigenvalues -2, 1, 1.
        let diag = vec![vec![0.0]; 3];
        let spring = 1.0;
        let inertia = logdet_cyclic_blocks(&diag, 1, spring).expect("ln 2");
        let want = 2.0_f64.ln();
        assert!(
            (inertia.log_abs_det - want).abs() < 1e-8,
            "log|det| {} is not ln 2",
            inertia.log_abs_det
        );
        assert_eq!(inertia.n_negative, 1);
    }

    #[test]
    fn inconsistent_singular_cycle_is_not_a_successful_solve() {
        let n = 2usize;
        let p = 6usize;
        let spring = 1.0;
        let mut diag = Vec::new();
        for _ in 0..p {
            let mut block = vec![0.0; n * n];
            block[0] = 2.0 * spring;
            block[3] = 4.0 + 2.0 * spring;
            diag.push(block);
        }
        let rhs = vec![1.0; p * n];
        match factor_cyclic(&diag, n, spring, 0.0) {
            Ok(factor) => {
                let err = factor.solve(&rhs).expect_err("inconsistent singular solve");
                assert!(
                    matches!(err, SaddleError::Singular(_) | SaddleError::Solver(_)),
                    "{err}"
                );
            }
            Err(err) => assert!(matches!(err, SaddleError::Singular(_)), "{err}"),
        }
    }

    #[test]
    fn shape_and_finiteness_are_typed() {
        let err = logdet_cyclic_strict(&[vec![1.0]], 1, 1.0).expect_err("short");
        assert!(matches!(err, SaddleError::Invalid(_)));
        let mut diag = vec![vec![1.0, 0.0, 0.0, 1.0]; 3];
        diag[1].pop();
        let err = logdet_cyclic_blocks(&diag, 2, 0.2).expect_err("ragged");
        assert!(
            matches!(err, SaddleError::Shape(ref msg) if msg.contains("cyclic logdet expected")),
            "{err}"
        );
        diag[1] = vec![1.0, f64::NAN, 0.0, 1.0];
        let err = logdet_cyclic_strict(&diag, 2, 0.2).expect_err("nan");
        assert!(
            matches!(err, SaddleError::NonFinite("cyclic logdet")),
            "{err}"
        );
        let err = factor_cyclic(&sample_blocks(2, 3).0, 0, 0.2, 0.0).expect_err("empty");
        assert!(matches!(err, SaddleError::Invalid(_)));
        let (diag, _) = sample_blocks(2, 3);
        let err = logdet_cyclic_strict(&diag, 2, f64::NAN).expect_err("nan spring");
        assert!(matches!(
            err,
            SaddleError::NonFinite("cyclic logdet spring")
        ));
        let err = factor_cyclic(&diag, 2, 0.2, f64::INFINITY).expect_err("infinite shift");
        assert!(matches!(err, SaddleError::NonFinite("cyclic factor")));
        let err = factor_cyclic(&sample_blocks(2, 2).0, 2, 0.2, 0.0).expect_err("two beads");
        assert!(matches!(err, SaddleError::Invalid(_)));
        let mut vectors = Array2::zeros((1, 1));
        vectors[(0, 0)] = f64::NAN;
        let err = gemm_symmetric_inverse(&vectors, &[1.0]).expect_err("nan vector");
        assert!(matches!(err, SaddleError::NonFinite(_)));
    }

    #[test]
    fn solve_reproduces_a_known_cyclic_action() {
        let (diag, spring) = sample_blocks(3, 4);
        let factor = factor_cyclic(&diag, 3, spring, 0.0).expect("factor");
        let x_true: Vec<f64> = (0..12).map(|i| ((i + 1) as f64) * 0.17).collect();
        let rhs = cyclic_matvec(&diag, 3, spring, &x_true);
        let report = factor.solve_report(&rhs).expect("solve");
        for (got, want) in report.solution.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
        assert!(report.residual < 1e-8);
        let err = factor.solve(&[0.0]).expect_err("short rhs");
        assert!(matches!(err, SaddleError::Shape(_)));
    }

    #[test]
    fn refinement_threshold_is_one_e_minus_11_and_fires_once() {
        assert!(!refine_triggered(1e-11, 1.0));
        assert!(refine_triggered(1e-11 * 1.0000001, 1.0));
        assert!(!refine_triggered(1e-12, 1.0));
        assert!(refine_triggered(1e-10, 1.0));
        assert!(!refine_triggered(0.0, 1.0));
        // A zero right-hand side refines only when a residual remains.
        assert!(!refine_triggered(0.0, 0.0));
        assert!(refine_triggered(1e-30, 0.0));
        // Relative to ‖b‖, with no unit floor: 1e-12 exceeds 1e-11 * 1e-8.
        assert!(refine_triggered(1e-12, 1e-8));
        assert!(!refine_triggered(1e-11 * 1e-8, 1e-8));
        // A right-hand scale above 1 distinguishes multiplication from division.
        assert!(!refine_triggered(2e-11, 2.0));
        assert!(refine_triggered(2e-11 * 1.0000001, 2.0));
        assert!(refine_triggered(1e-10, 2.0));
        assert_eq!(SOLVE_REFINE_TOLERANCE, 1e-11);

        // A zero spring leaves each Hilbert block independent, so the explicit
        // inverse is inaccurate (residual above 1e-11) but still a contraction.
        // One correction must therefore land strictly closer to the right-hand side.
        let n = 8usize;
        let p = 3usize;
        let spring = 0.0;
        let mut diag = Vec::with_capacity(p);
        for _bead in 0..p {
            let mut block = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    block[i * n + j] = 1.0 / (i + j + 1) as f64;
                }
            }
            diag.push(block);
        }
        let factor = factor_cyclic(&diag, n, spring, 0.0).expect("hilbert factor");
        let rhs: Vec<f64> = (0..p * n)
            .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
            .collect();
        let once = factor.solve_once(&rhs).expect("unrefined");
        let rough = euclidean(&cyclic_residual(&diag, n, spring, &once, &rhs));
        let rhs_norm = euclidean(&rhs);
        assert!(
            refine_triggered(rough, rhs_norm),
            "hilbert residual {rough} on scale {rhs_norm} must cross 1e-11"
        );
        let report = factor.solve_report(&rhs).expect("refined");
        assert_eq!(report.corrections, 1);
        assert!(
            report.residual < rough,
            "refined {} vs rough {rough}",
            report.residual
        );
        let easy = factor_cyclic(&sample_blocks(3, 4).0, 3, 0.35, 0.0).expect("easy");
        let x: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let b = cyclic_matvec(&sample_blocks(3, 4).0, 3, 0.35, &x);
        let easy_report = easy.solve_report(&b).expect("easy solve");
        assert_eq!(easy_report.corrections, 0);
        assert!(easy_report.residual < SOLVE_REFINE_TOLERANCE);

        let (diag, spring) = sample_blocks(3, 4);
        let tau = 0.37;
        let shifted = factor_cyclic(&diag, 3, spring, tau).expect("shifted factor");
        let mut blocks = diag.clone();
        for block in &mut blocks {
            for i in 0..3 {
                block[i * 3 + i] += tau;
            }
        }
        let x_shift: Vec<f64> = (0..12).map(|i| (i as f64) * 0.05 - 0.2).collect();
        let rhs = cyclic_matvec(&blocks, 3, spring, &x_shift);
        let solved = shifted.solve(&rhs).expect("shifted solve");
        for (got, want) in solved.iter().zip(&x_shift) {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
    }

    #[test]
    fn one_negative_diagonal_is_counted_once() {
        let n = 2usize;
        let p = 4usize;
        let spring = 0.2;
        let mut diag = Vec::new();
        for bead in 0..p {
            let mut block = vec![0.0; n * n];
            block[0] = if bead == 1 { -2.0 } else { 3.0 };
            block[3] = 4.0;
            diag.push(block);
        }
        let got = logdet_cyclic_strict(&diag, n, spring).expect("signed");
        let (_, n_neg) = dense_inertia(&dense_cyclic(&diag, n, spring));
        assert_eq!(got.n_negative, n_neg);
        assert_eq!(n_neg, 1);
    }

    #[test]
    fn pivot_choice_odd_ring_and_closing_index_are_pinned() {
        let eye = vec![2.0, 0.0, 0.0, 2.0];
        let ill = vec![1.0, 0.0, 0.0, 50.0];
        let zero = vec![0.0; 4];
        assert!((pivot_quality(&eye, 2).expect("eye") - 1.0).abs() < 1e-12);
        assert!((pivot_quality(&ill, 2).expect("ill") - 0.02).abs() < 1e-12);
        assert_eq!(pivot_quality(&zero, 2).expect("zero"), 0.0);
        assert_eq!(
            best_pivot_index(&[ill.clone(), eye.clone(), ill.clone()], 2).expect("later"),
            1
        );
        assert_eq!(
            best_pivot_index(&[eye.clone(), eye.clone(), ill.clone()], 2).expect("tie"),
            0
        );

        // The best-conditioned block is negative, so the open chain's first
        // complement contributes both negative eigenvalues.
        let mut leading = vec![eye.clone(), ill.clone(), eye.clone()];
        leading[0] = vec![-3.0, 0.0, 0.0, -3.0];
        assert_eq!(best_pivot_index(&leading, 2).expect("leading"), 0);
        let leading_inertia = logdet_cyclic_strict(&leading, 2, 0.0).expect("leading inertia");
        assert_eq!(leading_inertia.n_negative, 2);
        let want_log = 2.0 * 3.0_f64.ln() + 50.0_f64.ln() + 2.0 * 2.0_f64.ln();
        assert!(
            (leading_inertia.log_abs_det - want_log).abs() < 1e-8,
            "{} vs {want_log}",
            leading_inertia.log_abs_det
        );

        // The worst-conditioned block is the closing one and carries the sign.
        let mut closing = vec![eye.clone(), eye.clone(), ill.clone()];
        closing[2] = vec![-5.0, 0.0, 0.0, -0.05];
        assert_eq!(best_pivot_index(&closing, 2).expect("closing pivot"), 0);
        let closing_inertia = logdet_cyclic_strict(&closing, 2, 0.0).expect("closing inertia");
        assert_eq!(closing_inertia.n_negative, 2);

        let mut singular = vec![eye.clone(); 4];
        singular[3] = zero;
        let err = factor_cyclic(&singular, 2, 0.0, 0.0).expect_err("factor index");
        assert!(matches!(err, SaddleError::Singular(3)), "{err}");
        let err = logdet_cyclic_strict(&singular, 2, 0.0).expect_err("logdet index");
        assert!(matches!(err, SaddleError::Singular(3)), "{err}");

        let mut near = vec![eye; 3];
        near[1] = vec![1.0, 0.0, 0.0, 1e-16];
        let err = logdet_cyclic_strict(&near, 2, 0.0).expect_err("near zero");
        assert!(matches!(err, SaddleError::Singular(_)), "{err}");

        let n = 2usize;
        let p = 5usize;
        let (diag, spring) = sample_blocks(n, p);
        let factor = factor_cyclic(&diag, n, spring, 0.0).expect("odd factor");
        let x_true: Vec<f64> = (0..p * n).map(|i| (i as f64) * 0.13 - 0.4).collect();
        let rhs = cyclic_matvec(&diag, n, spring, &x_true);
        let report = factor.solve_report(&rhs).expect("odd solve");
        assert_eq!(report.corrections, 0, "residual {}", report.residual);
        for (got, want) in report.solution.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
    }

    #[test]
    fn large_finite_springs_match_the_inertia_and_log_pivot() {
        // Three scalar beads with diagonal `4|k|`. A positive spring has
        // complements `4k`, `15k/4`, and `10k/3`, so the pivot sum is
        // `ln(50) + 3 ln(k)` and every pivot is positive. The product
        // `spring * spring` is non-finite on this range.
        let positive = |spring: f64| vec![vec![4.0 * spring]; 3];
        for spring in [1.0e160_f64, 1.0e180, 1.0e200] {
            assert!(!(spring * spring).is_finite(), "{spring}");
            let want = (50.0_f64).ln() + 3.0 * spring.ln();
            let got = logdet_cyclic_blocks(&positive(spring), 1, spring).expect("positive");
            assert!(
                (got.log_abs_det - want).abs() < 1e-6,
                "{spring}: {} vs {want}",
                got.log_abs_det
            );
            assert_eq!(got.n_negative, 0, "{spring}");
        }

        let unit = positive(1.0);
        let unit_got = logdet_cyclic_blocks(&unit, 1, 1.0).expect("unit");
        let dense = dense_inertia(&dense_cyclic(&unit, 1, 1.0));
        assert!((unit_got.log_abs_det - (50.0_f64).ln()).abs() < 1e-8);
        assert!((dense.0 - unit_got.log_abs_det).abs() < 1e-8);
        assert_eq!((unit_got.n_negative, dense.1), (0, 0));

        // The closing pivot is `-14k/3`. The open chain stays positive, so
        // the inertia is one negative eigenvalue and `ln|det| = ln(70) + 3 ln(k)`.
        let spring = 1.0e200_f64;
        let indefinite = vec![vec![4.0 * spring], vec![4.0 * spring], vec![-4.0 * spring]];
        let want = (70.0_f64).ln() + 3.0 * spring.ln();
        let got = logdet_cyclic_blocks(&indefinite, 1, spring).expect("indefinite");
        assert!(
            (got.log_abs_det - want).abs() < 1e-6,
            "{} vs {want}",
            got.log_abs_det
        );
        assert_eq!(got.n_negative, 1);
        let indefinite_unit = vec![vec![4.0], vec![4.0], vec![-4.0]];
        let unit_indef = logdet_cyclic_blocks(&indefinite_unit, 1, 1.0).expect("unit indef");
        let dense_indef = dense_inertia(&dense_cyclic(&indefinite_unit, 1, 1.0));
        assert!((unit_indef.log_abs_det - (70.0_f64).ln()).abs() < 1e-8);
        assert_eq!((unit_indef.n_negative, dense_indef.1), (1, 1));

        // A negative spring flips the off-diagonal sign. The complements are
        // `4k`, `15k/4`, and `18k/5`, so `ln|det| = ln(54) + 3 ln(k)`.
        let neg = -1.0e200_f64;
        assert!(!(neg * neg).is_finite());
        let neg_diag = vec![vec![4.0 * neg.abs()]; 3];
        let want_neg = (54.0_f64).ln() + 3.0 * neg.abs().ln();
        let got_neg = logdet_cyclic_blocks(&neg_diag, 1, neg).expect("negative spring");
        assert!(
            (got_neg.log_abs_det - want_neg).abs() < 1e-6,
            "{} vs {want_neg}",
            got_neg.log_abs_det
        );
        assert_eq!(got_neg.n_negative, 0);

        // Two decoupled copies in `2 × 2` blocks double the scalar pivot sum.
        let k = 1.0e160_f64;
        let blocks = vec![vec![4.0 * k, 0.0, 0.0, 4.0 * k]; 3];
        let want_block = 2.0 * (50.0_f64).ln() + 6.0 * k.ln();
        let got_block = logdet_cyclic_blocks(&blocks, 2, k).expect("block");
        assert!(
            (got_block.log_abs_det - want_block).abs() < 1e-6,
            "{} vs {want_block}",
            got_block.log_abs_det
        );
        assert_eq!(got_block.n_negative, 0);
    }

    #[test]
    fn nonsingular_cycle_with_a_zero_intermediate_pivot_is_factored() {
        // Diagonal 1 and spring 1 make the second complement
        // `1 - spring^2 / 1` exactly zero on every rotation. The assembled
        // eigenvalues are -1, 2, 2, so log|det| is ln 4 with one negative.
        let diag = vec![vec![1.0]; 3];
        let spring = 1.0;
        let got = logdet_cyclic_blocks(&diag, 1, spring).expect("nonsingular");
        let want = 4.0_f64.ln();
        assert!(
            (got.log_abs_det - want).abs() < 1e-8,
            "{} vs {want}",
            got.log_abs_det
        );
        assert_eq!(got.n_negative, 1);
        let (log_abs, n_neg) = dense_inertia(&dense_cyclic(&diag, 1, spring));
        assert!((got.log_abs_det - log_abs).abs() < 1e-8);
        assert_eq!(got.n_negative, n_neg);

        let factor = factor_cyclic(&diag, 1, spring, 0.0).expect("factor");
        let x_true = vec![0.5, -0.25, 0.125];
        let rhs = cyclic_matvec(&diag, 1, spring, &x_true);
        let report = factor.solve_report(&rhs).expect("solve");
        for (got_x, want_x) in report.solution.iter().zip(&x_true) {
            assert!((got_x - want_x).abs() < 1e-8, "{got_x} vs {want_x}");
        }
        assert!(report.residual.is_finite());
        assert!(report.residual < 1e-8, "{}", report.residual);
    }

    #[test]
    fn large_finite_entries_keep_a_finite_solve_report() {
        let easy = vec![vec![4.0], vec![5.0], vec![6.0]];
        let easy_factor = factor_cyclic(&easy, 1, 0.25, 0.0).expect("easy factor");
        let easy_rhs = vec![1.0e200, -1.0e200, 0.5e200];
        assert!(
            !easy_rhs
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .is_finite()
        );
        let easy_report = easy_factor
            .solve_report(&easy_rhs)
            .expect("easy finite report");
        assert!(easy_report.residual.is_finite(), "{}", easy_report.residual);
        assert!(easy_report.solution.iter().all(|value| value.is_finite()));

        // Hilbert blocks leave a residual above 1e-11 on a unit right-hand
        // side, so scaling that side by 1e200 makes a naive sum of squares
        // overflow.
        let n = 8usize;
        let p = 3usize;
        let mut diag = Vec::with_capacity(p);
        for _bead in 0..p {
            let mut block = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    block[i * n + j] = 1.0 / (i + j + 1) as f64;
                }
            }
            diag.push(block);
        }
        let factor = factor_cyclic(&diag, n, 0.0, 0.0).expect("hilbert factor");
        let rhs: Vec<f64> = (0..p * n)
            .map(|i| if i % 2 == 0 { 1.0e200 } else { -1.0e200 })
            .collect();
        let report = factor.solve_report(&rhs).expect("finite report");
        assert!(report.residual.is_finite(), "{}", report.residual);
        assert!(report.solution.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn asymmetric_blocks_are_rejected() {
        let mut diag = vec![vec![3.0, 0.1, 0.1, 3.0]; 3];
        diag[0][1] = 0.4;
        let err = logdet_cyclic_blocks(&diag, 2, 0.2).expect_err("asymmetric logdet");
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
        let err = factor_cyclic(&diag, 2, 0.2, 0.0).expect_err("asymmetric factor");
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");
        let err = logdet_cyclic_strict(&diag, 2, 0.2).expect_err("asymmetric strict");
        assert!(matches!(err, SaddleError::Invalid(_)), "{err}");

        let mut close = vec![vec![3.0, 0.1, 0.1, 3.0]; 3];
        close[0][1] = 0.1 + 1.0e-16;
        logdet_cyclic_blocks(&close, 2, 0.2).expect("within tolerance");
        let factor = factor_cyclic(&close, 2, 0.2, 0.0).expect("within tolerance factor");
        let x_true = vec![0.2, -0.1, 0.05, 0.3, -0.2, 0.1];
        let rhs = cyclic_matvec(&close, 2, 0.2, &x_true);
        let report = factor.solve_report(&rhs).expect("within tolerance solve");
        assert!(report.residual.is_finite());
    }
}
