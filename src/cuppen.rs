//! Cuppen's divide-and-conquer for a symmetric tridiagonal matrix.
//!
//! The merge tree is fixed from the dimension: leaves at
//! [`RECURSION_THRESHOLD`], split at `n / 2`, siblings by `rayon::join`
//! under feature `par`. Deflation is serial. Each Gu-Eisenstat secular
//! root is one task. The update weights are recomputed by the Löwner
//! formula, and `Q U` goes through [`crate::simd_gemm::gemm_nn`] on a
//! fixed column partition.
//!
//! Leaves use Francis's implicit QL step (J. G. F. Francis, *Comput. J.*
//! 4 (1962) 332-345, <https://doi.org/10.1093/comjnl/4.4.332>). Products
//! accumulate with `f64::mul_add`. J. J. M. Cuppen, *Numer. Math.* 36 (1980)
//! 177-195 (<https://doi.org/10.1007/BF01396757>). Ming Gu and Stanley
//! C. Eisenstat, *SIAM J. Matrix Anal. Appl.* 16 (1995) 172-191
//! (<https://doi.org/10.1137/S0895479892241287>).

use crate::error::SaddleError;
use crate::simd_gemm::gemm_nn;

/// Leaf size for the Francis QL step. Fixed. Not a hardware query.
const RECURSION_THRESHOLD: usize = 128;

struct Spectrum {
    evals: Vec<f64>,
    evecs: Vec<f64>,
}

/// Eigenvalues ascend. Eigenvectors are column-major.
pub(crate) fn tridiagonal_evd(
    diag: &[f64],
    off: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    let spectrum = cuppen(diag, off)?;
    Ok((spectrum.evals, spectrum.evecs))
}

/// `sqrt(x^2 + y^2)` from a scaled square root. IEEE `sqrt` is correctly
/// rounded, and this form does not call `hypot`, so x86 and aarch64 agree.
pub(crate) fn hypot(x: f64, y: f64) -> f64 {
    let mut ax = x.abs();
    let mut ay = y.abs();
    if ax < ay {
        std::mem::swap(&mut ax, &mut ay);
    }
    if ax == 0.0 {
        return 0.0;
    }
    let ratio = ay / ax;
    ax * (1.0 + ratio * ratio).sqrt()
}

fn cuppen(diag: &[f64], off: &[f64]) -> Result<Spectrum, SaddleError> {
    let n = diag.len();
    if n <= RECURSION_THRESHOLD {
        return francis(diag, off);
    }
    let n1 = n / 2;
    let beta = off[n1 - 1];
    let rho = beta.abs();
    let mut left_diag = diag[..n1].to_vec();
    let mut right_diag = diag[n1..].to_vec();
    if rho != 0.0 {
        left_diag[n1 - 1] -= rho;
        right_diag[0] -= rho;
    }
    let left_off = &off[..n1 - 1];
    let right_off = &off[n1..];
    let (left, right) = descend(&left_diag, left_off, &right_diag, right_off);
    let mut left = left?;
    let mut right = right?;
    if rho == 0.0 {
        return Ok(merge_independent(left, right));
    }
    let sign = if beta >= 0.0 { 1.0 } else { -1.0 };
    merge_rank_one(&mut left, &mut right, rho, sign)
}

fn descend(
    left_diag: &[f64],
    left_off: &[f64],
    right_diag: &[f64],
    right_off: &[f64],
) -> (Result<Spectrum, SaddleError>, Result<Spectrum, SaddleError>) {
    #[cfg(feature = "par")]
    {
        rayon::join(
            || cuppen(left_diag, left_off),
            || cuppen(right_diag, right_off),
        )
    }
    #[cfg(not(feature = "par"))]
    {
        (cuppen(left_diag, left_off), cuppen(right_diag, right_off))
    }
}

fn merge_independent(left: Spectrum, right: Spectrum) -> Spectrum {
    let n1 = left.evals.len();
    let n2 = right.evals.len();
    let n = n1 + n2;
    let mut evals = Vec::with_capacity(n);
    let mut evecs = vec![0.0; n * n];
    let mut i = 0;
    let mut j = 0;
    let mut col = 0;
    while i < n1 || j < n2 {
        let take_left = if j == n2 {
            true
        } else if i == n1 {
            false
        } else {
            left.evals[i] <= right.evals[j]
        };
        if take_left {
            evals.push(left.evals[i]);
            for row in 0..n1 {
                evecs[col * n + row] = left.evecs[i * n1 + row];
            }
            i += 1;
        } else {
            evals.push(right.evals[j]);
            for row in 0..n2 {
                evecs[col * n + n1 + row] = right.evecs[j * n2 + row];
            }
            j += 1;
        }
        col += 1;
    }
    Spectrum { evals, evecs }
}

fn merge_rank_one(
    left: &mut Spectrum,
    right: &mut Spectrum,
    rho: f64,
    sign: f64,
) -> Result<Spectrum, SaddleError> {
    let n1 = left.evals.len();
    let n2 = right.evals.len();
    let n = n1 + n2;
    let mut z_cat = vec![0.0; n];
    for (col, slot) in z_cat.iter_mut().take(n1).enumerate() {
        *slot = left.evecs[(n1 - 1) + col * n1];
    }
    for col in 0..n2 {
        z_cat[n1 + col] = sign * right.evecs[col * n2];
    }
    let mut eval_cat = Vec::with_capacity(n);
    eval_cat.extend_from_slice(&left.evals);
    eval_cat.extend_from_slice(&right.evals);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eval_cat[i].total_cmp(&eval_cat[j]).then(i.cmp(&j)));
    let mut delta = vec![0.0; n];
    let mut z = vec![0.0; n];
    for (new_col, &old) in order.iter().enumerate() {
        delta[new_col] = eval_cat[old];
        z[new_col] = z_cat[old];
    }
    let mut q = vec![0.0; n * n];
    for (new_col, &old) in order.iter().enumerate() {
        if old < n1 {
            for row in 0..n1 {
                q[new_col * n + row] = left.evecs[old * n1 + row];
            }
        } else {
            let src = old - n1;
            for row in 0..n2 {
                q[new_col * n + n1 + row] = right.evecs[src * n2 + row];
            }
        }
    }
    // The children are no longer needed. Drop their storage before the
    // secular product allocates another n by n matrix.
    left.evecs.clear();
    right.evecs.clear();
    left.evals.clear();
    right.evals.clear();

    let deflated = deflate(&mut delta, &mut z, &mut q, rho);
    let active: Vec<usize> = (0..n).filter(|&i| !deflated[i]).collect();
    let m = active.len();
    let mut lambda_of = delta.clone();
    if m == 1 {
        let i = active[0];
        lambda_of[i] = delta[i] + rho * z[i] * z[i];
    } else if m > 1 {
        let d_act: Vec<f64> = active.iter().map(|&i| delta[i]).collect();
        let z_act: Vec<f64> = active.iter().map(|&i| z[i]).collect();
        let z2: Vec<f64> = z_act.iter().map(|value| value * value).collect();
        let mut roots = vec![
            ShiftedRoot {
                shift: 0.0,
                mu: 0.0
            };
            m
        ];
        let tasks = crate::simd_gemm::column_tasks(m).max(1);
        let d_ptr = crate::simd_gemm::SyncPtr::new(d_act.as_ptr());
        let z_ptr = crate::simd_gemm::SyncPtr::new(z2.as_ptr());
        let root_ptr = crate::simd_gemm::SyncMut::new(roots.as_mut_ptr() as *mut f64);
        // SAFETY: each task writes a disjoint index range of `roots`.
        // `ShiftedRoot` is two `f64`s, so the range is `2 * count` floats.
        // The pole vectors are immutable.
        crate::simd_gemm::for_tiles(tasks, move |tile| {
            let (start, count) = crate::simd_gemm::split_range(m, tasks, tile);
            for j in start..start + count {
                let root = secular_root(d_ptr.get(), z_ptr.get(), m, rho, j);
                unsafe {
                    let slot = root_ptr.add(j * 2);
                    *slot = root.shift;
                    *slot.add(1) = root.mu;
                }
            }
        });
        let z_tilde = lowener_signed(&d_act, &roots, rho, &z_act);
        let mut columns = vec![0.0; n * m];
        for (j, root) in roots.iter().enumerate() {
            let mut norm = 0.0;
            for (local, &index) in active.iter().enumerate() {
                let denom = (d_act[local] - root.shift) - root.mu;
                let value = if denom == 0.0 {
                    0.0
                } else {
                    z_tilde[local] / denom
                };
                columns[j * n + index] = value;
                norm = value.mul_add(value, norm);
            }
            norm = norm.sqrt();
            if norm == 0.0 || !norm.is_finite() {
                return Err(SaddleError::Solver("secular eigenvector vanished".into()));
            }
            for &index in &active {
                columns[j * n + index] /= norm;
            }
            lambda_of[active[j]] = root.shift + root.mu;
        }
        // Deflated directions stay as the current basis vectors of Q.
        // Secular directions replace those columns after Q · U.
        let mut mixed = vec![0.0; n * n];
        // Build U: deflated columns are unit vectors, secular columns
        // are `columns`. Then sort by eigenvalue.
        let mut e_mat = vec![0.0; n * n];
        let mut lambdas = vec![0.0; n];
        let mut cursor = 0;
        for i in 0..n {
            if deflated[i] {
                e_mat[cursor * n + i] = 1.0;
                lambdas[cursor] = delta[i];
                cursor += 1;
            }
        }
        for (j, &index) in active.iter().enumerate() {
            for row in 0..n {
                e_mat[cursor * n + row] = columns[j * n + row];
            }
            lambdas[cursor] = lambda_of[index];
            cursor += 1;
        }
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&a, &b| lambdas[a].total_cmp(&lambdas[b]).then(a.cmp(&b)));
        let mut e_sorted = vec![0.0; n * n];
        let mut evals = vec![0.0; n];
        for (new_col, &old) in perm.iter().enumerate() {
            evals[new_col] = lambdas[old];
            for row in 0..n {
                e_sorted[new_col * n + row] = e_mat[old * n + row];
            }
        }
        // SAFETY: Q and U are n by n column-major. The product writes
        // `mixed`, which no other reference aliases.
        unsafe {
            gemm_nn(
                n,
                n,
                n,
                1.0,
                q.as_ptr(),
                n,
                e_sorted.as_ptr(),
                n,
                0.0,
                mixed.as_mut_ptr(),
                n,
            );
        }
        return Ok(Spectrum {
            evals,
            evecs: mixed,
        });
    }
    // Every component deflated, or the single-root case still needs
    // the basis permutation below when m == 1. m == 0 falls through
    // with lambda_of == delta and U a permutation of the identity.
    if m == 1 {
        let index = active[0];
        // One pole. The normalized secular vector is that basis column,
        // and the root is exact: δ + ρ z².
        lambda_of[index] = delta[index] + rho * z[index] * z[index];
    }
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| lambda_of[a].total_cmp(&lambda_of[b]).then(a.cmp(&b)));
    let mut evals = vec![0.0; n];
    let mut evecs = vec![0.0; n * n];
    for (new_col, &old) in perm.iter().enumerate() {
        evals[new_col] = lambda_of[old];
        for row in 0..n {
            evecs[new_col * n + row] = q[old * n + row];
        }
    }
    Ok(Spectrum { evals, evecs })
}

fn deflate(delta: &mut [f64], z: &mut [f64], q: &mut [f64], rho: f64) -> Vec<bool> {
    let n = delta.len();
    let mut scale = rho.abs();
    for value in delta.iter() {
        scale = scale.max(value.abs());
    }
    let tol = 8.0 * f64::EPSILON * scale;
    let mut deflated = vec![false; n];
    let mut guard = 0;
    let mut changed = true;
    while changed && guard < n + 2 {
        guard += 1;
        changed = false;
        let mut i = 0;
        while i < n {
            if deflated[i] {
                i += 1;
                continue;
            }
            if z[i].abs() <= tol {
                z[i] = 0.0;
                deflated[i] = true;
                changed = true;
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < n && deflated[j] {
                j += 1;
            }
            if j < n && (delta[j] - delta[i]).abs() <= tol {
                let radius = hypot(z[i], z[j]);
                if radius == 0.0 {
                    z[j] = 0.0;
                    deflated[j] = true;
                } else {
                    let c = z[i] / radius;
                    let s = z[j] / radius;
                    z[i] = radius;
                    z[j] = 0.0;
                    givens_columns(q, n, i, j, c, s);
                    deflated[j] = true;
                }
                changed = true;
                continue;
            }
            i += 1;
        }
    }
    deflated
}

fn givens_columns(q: &mut [f64], n: usize, i: usize, j: usize, c: f64, s: f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: this CPU has FMA. The body is the same `mul_add` chain,
        // so a machine without FMA matches it through the libcall.
        unsafe { givens_columns_fma(q, n, i, j, c, s) }
        return;
    }
    givens_columns_body(q, n, i, j, c, s);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn givens_columns_fma(q: &mut [f64], n: usize, i: usize, j: usize, c: f64, s: f64) {
    givens_columns_body(q, n, i, j, c, s);
}

#[inline(always)]
fn givens_columns_body(q: &mut [f64], n: usize, i: usize, j: usize, c: f64, s: f64) {
    for row in 0..n {
        let qi = q[i * n + row];
        let qj = q[j * n + row];
        q[i * n + row] = s.mul_add(qj, c * qi);
        q[j * n + row] = (-s).mul_add(qi, c * qj);
    }
}

/// Eigenvalue as `shift + mu`. `shift` is a pole, so `mu` is the
/// distance to that pole and `δ - λ` does not cancel.
#[derive(Clone, Copy)]
#[repr(C)]
struct ShiftedRoot {
    shift: f64,
    mu: f64,
}

fn secular_root(delta: *const f64, z2: *const f64, n: usize, rho: f64, i: usize) -> ShiftedRoot {
    // SAFETY: `delta` and `z2` each have `n` entries. `i < n` because
    // the caller splits `0..n`.
    let delta = unsafe { std::slice::from_raw_parts(delta, n) };
    let z2 = unsafe { std::slice::from_raw_parts(z2, n) };
    let scale = delta
        .iter()
        .fold(rho.abs(), |acc, value| acc.max(value.abs()))
        .max(1.0);
    let (shift, mut lo, mut hi) = if i + 1 < n {
        let left = delta[i];
        let right = delta[i + 1];
        let mid_mu = 0.5 * (right - left);
        let f_mid = secular_shifted(delta, z2, rho, left, mid_mu);
        if f_mid > 0.0 || !f_mid.is_finite() {
            // Root lies in (δ_i, midpoint]. Measure it from the left pole.
            (left, 0.0, mid_mu)
        } else if f_mid == 0.0 {
            return ShiftedRoot {
                shift: left,
                mu: mid_mu,
            };
        } else {
            // Root lies in [midpoint, δ_{i+1}). Measure it from the right pole.
            (right, -mid_mu, 0.0)
        }
    } else {
        let mut weight = 0.0;
        for value in z2 {
            weight += *value;
        }
        let base = delta[n - 1];
        let mut upper = rho * weight;
        if upper.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            upper = rho.abs() + scale;
        }
        let mut guard = 0;
        while secular_shifted(delta, z2, rho, base, upper) <= 0.0 && guard < 80 {
            upper = upper * 2.0 + scale * f64::EPSILON;
            guard += 1;
        }
        (base, 0.0, upper)
    };
    if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
        return ShiftedRoot { shift, mu: 0.0 };
    }
    for _ in 0..120 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if hi - lo <= 8.0 * f64::EPSILON * mid.abs() {
            break;
        }
        let value = secular_shifted(delta, z2, rho, shift, mid);
        if !value.is_finite() || value < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut mu = 0.5 * (lo + hi);
    for _ in 0..12 {
        let (value, deriv) = secular_shifted_deriv(delta, z2, rho, shift, mu);
        if !value.is_finite() || !deriv.is_finite() || deriv == 0.0 {
            break;
        }
        let mut next = mu - value / deriv;
        let inside = next.partial_cmp(&lo) == Some(std::cmp::Ordering::Greater)
            && next.partial_cmp(&hi) == Some(std::cmp::Ordering::Less);
        if !inside {
            next = 0.5 * (lo + hi);
        }
        if value < 0.0 {
            lo = mu.min(next);
        } else {
            hi = mu.max(next);
        }
        if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
            break;
        }
        mu = next;
    }
    ShiftedRoot { shift, mu }
}

/// `1/ρ + Σ z² / ((δ - shift) - μ)`, same sign as `1 + ρ Σ z²/(δ-λ)`
/// when `ρ > 0`. Subtracting `shift` before `μ` keeps a near-pole root.
fn secular_shifted(delta: &[f64], z2: &[f64], rho: f64, shift: f64, mu: f64) -> f64 {
    // The vector path adds the quotients in index order, the same chain
    // as the scalar loop. A zero pole returns the same infinity.
    crate::simd_gemm::sum_div_ordered(1.0 / rho, delta, z2, shift, mu).unwrap_or(f64::NEG_INFINITY)
}

fn secular_shifted_deriv(delta: &[f64], z2: &[f64], rho: f64, shift: f64, mu: f64) -> (f64, f64) {
    crate::simd_gemm::sum_div_deriv_ordered(1.0 / rho, delta, z2, shift, mu)
        .unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
}

fn lowener_signed(delta: &[f64], lambda: &[ShiftedRoot], rho: f64, z: &[f64]) -> Vec<f64> {
    let n = delta.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut mant = 1.0;
        let mut exp = 0i32;
        for j in 0..n {
            let numerator = (lambda[j].shift - delta[i]) + lambda[j].mu;
            let denominator = if j == i { rho } else { delta[j] - delta[i] };
            scale_mul(&mut mant, &mut exp, numerator);
            scale_div(&mut mant, &mut exp, denominator);
        }
        let mut value = ldexp(mant, exp);
        if value < 0.0 {
            value = 0.0;
        }
        let magnitude = value.sqrt();
        out[i] = if z[i] >= 0.0 { magnitude } else { -magnitude };
    }
    out
}

fn scale_mul(mant: &mut f64, exp: &mut i32, factor: f64) {
    if *mant == 0.0 || factor == 0.0 {
        *mant = 0.0;
        *exp = 0;
        return;
    }
    *mant *= factor;
    renormalize(mant, exp);
}

fn scale_div(mant: &mut f64, exp: &mut i32, denominator: f64) {
    if *mant == 0.0 {
        return;
    }
    *mant /= denominator;
    renormalize(mant, exp);
}

fn renormalize(mant: &mut f64, exp: &mut i32) {
    if *mant == 0.0 || !mant.is_finite() {
        return;
    }
    let (next, shift) = frexp(*mant);
    *mant = next;
    *exp += shift;
}

fn frexp(value: f64) -> (f64, i32) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0);
    }
    let bits = value.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        // Subnormal. Scale into the normal range, then undo.
        let scaled = value * (1u64 << 54) as f64;
        let (mant, exp) = frexp(scaled);
        return (mant, exp - 54);
    }
    let mant_bits = (bits & 0x800f_ffff_ffff_ffff) | (1022u64 << 52);
    (f64::from_bits(mant_bits), exp_field - 1022)
}

fn ldexp(mant: f64, exp: i32) -> f64 {
    if mant == 0.0 || !mant.is_finite() || exp == 0 {
        return mant;
    }
    let bits = mant.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        return scale_pow2(mant, exp);
    }
    let new_exp = exp_field + exp;
    if new_exp >= 0x7ff {
        return if mant.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if new_exp <= 0 {
        return scale_pow2(mant, exp);
    }
    let new_bits = (bits & 0x800f_ffff_ffff_ffff) | ((new_exp as u64) << 52);
    f64::from_bits(new_bits)
}

fn scale_pow2(mut value: f64, mut exp: i32) -> f64 {
    while exp > 0 {
        value *= 2.0;
        exp -= 1;
    }
    while exp < 0 {
        value *= 0.5;
        exp += 1;
    }
    value
}

fn francis(diag: &[f64], off: &[f64]) -> Result<Spectrum, SaddleError> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: this CPU has FMA. Rotations stay `mul_add` in the same order.
        return unsafe { francis_fma(diag, off) };
    }
    francis_body(diag, off)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn francis_fma(diag: &[f64], off: &[f64]) -> Result<Spectrum, SaddleError> {
    francis_body(diag, off)
}

#[inline(always)]
fn francis_body(diag: &[f64], off: &[f64]) -> Result<Spectrum, SaddleError> {
    let n = diag.len();
    if n == 0 {
        return Ok(Spectrum {
            evals: Vec::new(),
            evecs: Vec::new(),
        });
    }
    if n == 1 {
        return Ok(Spectrum {
            evals: vec![diag[0]],
            evecs: vec![1.0],
        });
    }
    let mut d = diag.to_vec();
    let mut f = vec![0.0; n];
    f[..n - 1].copy_from_slice(&off[..n - 1]);
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if f[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 30 * n {
                break;
            }
            let shift_g = (d[l + 1] - d[l]) / (2.0 * f[l]);
            let shift_r = hypot(shift_g, 1.0);
            let mut g = d[m] - d[l] + f[l] / (shift_g + sign_abs(shift_r, shift_g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut i = m - 1;
            loop {
                let f_part = s * f[i];
                let b = c * f[i];
                let radius = hypot(f_part, g);
                f[i + 1] = radius;
                if radius == 0.0 {
                    d[i + 1] -= p;
                    f[m] = 0.0;
                    break;
                }
                s = f_part / radius;
                c = g / radius;
                g = d[i + 1] - p;
                let mixed = (2.0 * c).mul_add(b, s.mul_add(d[i] - g, 0.0));
                p = s * mixed;
                d[i + 1] = g + p;
                g = c.mul_add(mixed, -b);
                rotate_columns(&mut z, n, i, c, s);
                if i == l {
                    d[l] -= p;
                    f[l] = g;
                    f[m] = 0.0;
                    break;
                }
                i -= 1;
            }
        }
    }
    let mut scale = 0.0f64;
    for value in &d {
        scale = scale.max(value.abs());
    }
    let mut max_off = 0.0f64;
    for value in f.iter().take(n - 1) {
        max_off = max_off.max(value.abs());
    }
    // A converged QL step has killed the subdiagonal to a few ulps.
    // Accepting `1e-6 · scale` left residuals above the dense tests.
    let tol = 32.0 * f64::EPSILON * scale.max(1.0) * (n as f64);
    if max_off > tol {
        return Err(SaddleError::Solver("Francis QL did not converge".into()));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]).then(i.cmp(&j)));
    let mut evals = vec![0.0; n];
    let mut evecs = vec![0.0; n * n];
    for (new_col, &old) in order.iter().enumerate() {
        evals[new_col] = d[old];
        for row in 0..n {
            evecs[new_col * n + row] = z[old * n + row];
        }
    }
    Ok(Spectrum { evals, evecs })
}

fn sign_abs(magnitude: f64, sign_of: f64) -> f64 {
    if sign_of >= 0.0 {
        magnitude.abs()
    } else {
        -magnitude.abs()
    }
}

fn rotate_columns(z: &mut [f64], n: usize, i: usize, c: f64, s: f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: this CPU has FMA. The body is the same `mul_add` chain.
        unsafe { rotate_columns_fma(z, n, i, c, s) }
        return;
    }
    rotate_columns_body(z, n, i, c, s);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn rotate_columns_fma(z: &mut [f64], n: usize, i: usize, c: f64, s: f64) {
    rotate_columns_body(z, n, i, c, s);
}

#[inline(always)]
fn rotate_columns_body(z: &mut [f64], n: usize, i: usize, c: f64, s: f64) {
    let j = i + 1;
    for row in 0..n {
        let zi = z[i * n + row];
        let zj = z[j * n + row];
        z[j * n + row] = s.mul_add(zi, c * zj);
        z[i * n + row] = (-s).mul_add(zj, c * zi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual(diag: &[f64], off: &[f64], evals: &[f64], evecs: &[f64]) -> f64 {
        let n = diag.len();
        let mut worst = 0.0f64;
        for col in 0..n {
            let mut norm = 0.0;
            for row in 0..n {
                let mut acc = diag[row] * evecs[col * n + row];
                if row > 0 {
                    acc += off[row - 1] * evecs[col * n + row - 1];
                }
                if row + 1 < n {
                    acc += off[row] * evecs[col * n + row + 1];
                }
                let diff = acc - evals[col] * evecs[col * n + row];
                norm += diff * diff;
            }
            worst = worst.max(norm.sqrt());
        }
        worst
    }

    #[test]
    fn tridiagonal_matches_a_diagonal_and_a_small_arrow() {
        let diag = [1.0, -2.0, 0.5, 4.0];
        let off = [0.0, 0.0, 0.0];
        let (evals, _) = tridiagonal_evd(&diag, &off).unwrap();
        let mut expect = diag;
        expect.sort_by(f64::total_cmp);
        for (got, want) in evals.iter().zip(&expect) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        let diag = [2.0, 3.0, -1.0, 0.25, 5.0, -4.0, 1.5, 0.0];
        let off = [0.3, -0.2, 0.4, 0.1, -0.5, 0.2, 0.05];
        let (evals, evecs) = tridiagonal_evd(&diag, &off).unwrap();
        let gap = residual(&diag, &off, &evals, &evecs);
        assert!(gap < 1e-8, "residual {gap}");
        for i in 1..evals.len() {
            assert!(evals[i] >= evals[i - 1]);
        }
        let n = 140;
        let mut wide_diag = vec![0.0; n];
        let mut wide_off = vec![0.0; n - 1];
        let mut state = 0x140u64;
        for value in &mut wide_diag {
            state = state.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(1);
            *value = ((state >> 11) as u32 as f64 / u32::MAX as f64) * 2.0 - 1.0;
        }
        for value in &mut wide_off {
            state = state.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(1);
            *value = ((state >> 11) as u32 as f64 / u32::MAX as f64) * 0.5;
        }
        let (wide_evals, wide_vecs) = tridiagonal_evd(&wide_diag, &wide_off).unwrap();
        let wide = residual(&wide_diag, &wide_off, &wide_evals, &wide_vecs);
        assert!(wide < 1e-10, "n=140 residual {wide}");
    }
}
