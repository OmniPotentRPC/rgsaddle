//! Two-stage dense symmetric eigensolver.
//!
//! Stage 1 reduces the symmetric matrix to bandwidth [`NB`] by blocked
//! Householder panels and a tiled trailing update (Azzam Haidar, Hatem
//! Ltaief, and Jack Dongarra, SC '11,
//! <https://doi.org/10.1145/2063384.2063394>). Stage 2 chases the bulge
//! down to tridiagonal form on a compact band of width [`CHASE_KD`],
//! which is wide enough for the fill and keeps each window contiguous.
//! The tridiagonal eigenproblem is Cuppen's
//! divide-and-conquer through faer when every vector is required
//! (J. J. M. Cuppen, *Numer. Math.* 36 (1980) 177-195,
//! <https://doi.org/10.1007/BF01396757>). faer's solver follows Gu
//! and Eisenstat, *SIAM J. Matrix Anal. Appl.* 16 (1995) 172-191
//! (<https://doi.org/10.1137/S0895479892241287>). The QR step below
//! faer's recursion threshold is Francis, *Comput. J.* 4 (1962) 332-345
//! (<https://doi.org/10.1093/comjnl/4.4.332>). Bisection plus inverse
//! iteration is used when only the lowest vectors are required.
//!
//! Every trailing-update and eigenvector product goes through
//! [`tile_gemm`]. Tile ownership and the faer task count depend on the
//! dimension and on [`NB`], not on the rayon pool.

use ndarray::{Array1, Array2};

use crate::error::SaddleError;

/// Tile size and target half-bandwidth. Fixed. Not a hardware query.
const NB: usize = 64;

/// Half-bandwidth stored during the bulge chase. The chase fill stays
/// inside `2 * NB - 1` subdiagonals; one extra diagonal is margin.
const CHASE_KD: usize = NB * 2;

/// Faer divide-and-conquer task count. A power of two, capped, and
/// independent of the pool size.
const EIGH_TASKS: usize = 32;

/// Seconds in the band reduction, the bulge chase, the tridiagonal
/// eigensolve, and the back-transformation.
#[derive(Clone, Copy, Debug)]
pub struct StageSeconds {
    pub band: f64,
    pub bulge: f64,
    pub tridiagonal: f64,
    pub back_transform: f64,
}

struct Reflector {
    start: usize,
    v: Vec<f64>,
    tau: f64,
}

/// Full spectrum of a symmetric matrix. Eigenvalues ascend. Columns are
/// orthonormal. The caller applies the sign rule.
pub fn eigh_symmetric(
    a: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>, StageSeconds), SaddleError> {
    let n = a.nrows();
    if n == 0 {
        return Ok((
            Array1::zeros(0),
            Array2::zeros((0, 0)),
            StageSeconds::zero(),
        ));
    }
    let mut seconds = StageSeconds::zero();
    let TridiagonalFactor {
        mut matrix,
        refs,
        band_s,
        bulge_s,
        diag,
        off,
    } = tridiagonalize(a)?;
    seconds.band = band_s;
    seconds.bulge = bulge_s;
    let _ = &mut matrix;
    let t0 = std::time::Instant::now();
    let (evals, mut vectors) = tridiagonal_full(&diag, &off)?;
    seconds.tridiagonal = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    apply_reflectors(&mut vectors, n, n, &refs);
    seconds.back_transform = t1.elapsed().as_secs_f64();
    reject_zero_columns(&vectors, n, n)?;
    Ok((pack_evals(&evals), pack_columns(&vectors, n, n), seconds))
}

/// Lowest `k` eigenpairs. `k == n` uses the full spectrum so the pairs
/// match [`eigh_symmetric`]. `k < n` bisects the tridiagonal and runs
/// inverse iteration, then back-transforms only those `k` columns.
pub fn partial_symmetric(
    a: &Array2<f64>,
    k: usize,
) -> Result<(Array1<f64>, Array2<f64>, StageSeconds), SaddleError> {
    let n = a.nrows();
    if k == n {
        return eigh_symmetric(a);
    }
    if n == 0 || k == 0 {
        return Ok((
            Array1::zeros(0),
            Array2::zeros((n, 0)),
            StageSeconds::zero(),
        ));
    }
    let mut seconds = StageSeconds::zero();
    let TridiagonalFactor {
        matrix: _matrix,
        refs,
        band_s,
        bulge_s,
        diag,
        off,
    } = tridiagonalize(a)?;
    seconds.band = band_s;
    seconds.bulge = bulge_s;
    let t0 = std::time::Instant::now();
    let (evals, mut vectors) = tridiagonal_partial(&diag, &off, k)?;
    seconds.tridiagonal = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    apply_reflectors(&mut vectors, n, k, &refs);
    seconds.back_transform = t1.elapsed().as_secs_f64();
    reject_zero_columns(&vectors, n, k)?;
    Ok((pack_evals(&evals), pack_columns(&vectors, n, k), seconds))
}

/// faer's sequential self-adjoint solver, for the single-thread cost of
/// the fixed partition against `Par::Seq`. Not the shipped entry point.
pub fn faer_sequential(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let n = a.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let mut mat = faer::Mat::<f64>::zeros(n, n);
    for col in 0..n {
        for row in 0..n {
            mat[(row, col)] = a[(row, col)];
        }
    }
    let mut vectors = faer::Mat::<f64>::zeros(n, n);
    let mut values = faer::diag::Diag::<f64>::zeros(n);
    let par = faer::Par::Seq;
    let params = faer::Spec::<faer::linalg::evd::SelfAdjointEvdParams, f64>::default();
    let scratch = faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        params,
    );
    let mut buffer = dyn_stack::MemBuffer::new(scratch);
    faer::linalg::evd::self_adjoint_evd(
        mat.as_ref(),
        values.as_mut(),
        Some(vectors.as_mut()),
        par,
        dyn_stack::MemStack::new(&mut buffer),
        params,
    )
    .map_err(|_| SaddleError::Solver("self-adjoint eigensolver did not converge".into()))?;
    let mut evals = Array1::zeros(n);
    let mut vecs = Array2::zeros((n, n));
    for col in 0..n {
        evals[col] = values[col];
        for row in 0..n {
            vecs[(row, col)] = vectors[(row, col)];
        }
    }
    Ok((evals, vecs))
}

impl StageSeconds {
    const fn zero() -> Self {
        Self {
            band: 0.0,
            bulge: 0.0,
            tridiagonal: 0.0,
            back_transform: 0.0,
        }
    }
}

struct TridiagonalFactor {
    matrix: Vec<f64>,
    refs: Vec<Reflector>,
    band_s: f64,
    bulge_s: f64,
    diag: Vec<f64>,
    off: Vec<f64>,
}

fn tridiagonalize(a: &Array2<f64>) -> Result<TridiagonalFactor, SaddleError> {
    let n = a.nrows();
    let mut matrix = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            matrix[col * n + row] = a[(row, col)];
        }
    }
    let t0 = std::time::Instant::now();
    let mut refs = dense_to_band(&mut matrix, n);
    let band_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    band_to_tridiagonal(&mut matrix, n, &mut refs);
    let bulge_s = t1.elapsed().as_secs_f64();
    let mut diag = vec![0.0; n];
    let mut off = vec![0.0; n.saturating_sub(1)];
    for i in 0..n {
        diag[i] = matrix[i * n + i];
        if i + 1 < n {
            off[i] = matrix[i * n + i + 1];
        }
    }
    Ok(TridiagonalFactor {
        matrix,
        refs,
        band_s,
        bulge_s,
        diag,
        off,
    })
}

fn dense_to_band(matrix: &mut [f64], n: usize) -> Vec<Reflector> {
    let mut refs = Vec::new();
    if n <= NB {
        return refs;
    }
    let mut i = 0;
    while i + NB < n {
        let pn = n - (i + NB);
        let pk = pn.min(NB);
        if pn == 0 || pk == 0 {
            break;
        }
        let mut panel = vec![0.0; pn * NB];
        for c in 0..NB {
            let src = (i + c) * n + (i + NB);
            panel[c * pn..(c + 1) * pn].copy_from_slice(&matrix[src..src + pn]);
        }
        let (v, tau, r_panel) = qr_panel(&panel, pn, NB);
        let mut trailing = vec![0.0; pn * pn];
        for c in 0..pn {
            let src = (i + NB + c) * n + (i + NB);
            trailing[c * pn..(c + 1) * pn].copy_from_slice(&matrix[src..src + pn]);
        }
        let mut t_factor = vec![0.0; pk * pk];
        dlarft(&v, pn, &tau, &mut t_factor);
        let mut s2 = vec![0.0; pn * pk];
        tile_gemm(
            pn,
            pk,
            pk,
            1.0,
            v.as_ptr(),
            1,
            pn as isize,
            t_factor.as_ptr(),
            1,
            pk as isize,
            0.0,
            s2.as_mut_ptr(),
            1,
            pn as isize,
        );
        let mut w = vec![0.0; pn * pk];
        gemm_row_tiles(&trailing, pn, &s2, pk, &mut w);
        let mut s1 = vec![0.0; pk * pk];
        tile_gemm(
            pk,
            pk,
            pn,
            1.0,
            s2.as_ptr(),
            pn as isize,
            1,
            w.as_ptr(),
            1,
            pn as isize,
            0.0,
            s1.as_mut_ptr(),
            1,
            pk as isize,
        );
        tile_gemm(
            pn,
            pk,
            pk,
            -0.5,
            v.as_ptr(),
            1,
            pn as isize,
            s1.as_ptr(),
            1,
            pk as isize,
            1.0,
            w.as_mut_ptr(),
            1,
            pn as isize,
        );
        syr2k_tiles(&mut trailing, pn, &v, &w, pk);
        for c in 0..pn {
            let dst = (i + NB + c) * n + (i + NB);
            matrix[dst..dst + pn].copy_from_slice(&trailing[c * pn..(c + 1) * pn]);
        }
        for c in 0..NB {
            for r in 0..pn {
                let val = if r <= c { r_panel[c * pn + r] } else { 0.0 };
                let row = i + NB + r;
                let col = i + c;
                matrix[col * n + row] = val;
                matrix[row * n + col] = val;
            }
        }
        for j in 0..pk {
            let start = i + NB + j;
            let mut reflector = vec![0.0; pn - j];
            for (row, slot) in reflector.iter_mut().enumerate() {
                *slot = v[(j + row) + j * pn];
            }
            refs.push(Reflector {
                start,
                v: reflector,
                tau: tau[j],
            });
        }
        i += NB;
    }
    refs
}

/// Symmetric band in column-major order. Column `col` holds rows
/// `col - kd ..= col + kd` contiguously, with the diagonal at offset `kd`.
struct SymBand {
    n: usize,
    kd: usize,
    ld: usize,
    data: Vec<f64>,
}

impl SymBand {
    fn from_dense(matrix: &[f64], n: usize) -> Self {
        let kd = CHASE_KD;
        let ld = kd * 2 + 1;
        let mut data = vec![0.0; n.saturating_mul(ld)];
        let copy_kd = NB;
        for col in 0..n {
            let row0 = col.saturating_sub(copy_kd);
            let row1 = n.min(col + copy_kd + 1);
            let base = col * ld + (row0 + kd - col);
            data[base..base + (row1 - row0)]
                .copy_from_slice(&matrix[col * n + row0..col * n + row1]);
        }
        Self { n, kd, ld, data }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        if row.abs_diff(col) > self.kd {
            return 0.0;
        }
        self.data[col * self.ld + (row + self.kd - col)]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        if row.abs_diff(col) > self.kd {
            debug_assert!(value == 0.0);
            return;
        }
        self.data[col * self.ld + (row + self.kd - col)] = value;
    }

    fn scatter(&self, matrix: &mut [f64]) {
        let n = self.n;
        for col in 0..n {
            let row0 = col.saturating_sub(self.kd);
            let row1 = n.min(col + self.kd + 1);
            let base = col * self.ld + (row0 + self.kd - col);
            matrix[col * n + row0..col * n + row1]
                .copy_from_slice(&self.data[base..base + (row1 - row0)]);
        }
    }
}

fn band_to_tridiagonal(matrix: &mut [f64], n: usize, refs: &mut Vec<Reflector>) {
    if n < 3 {
        return;
    }
    let mut band = SymBand::from_dense(matrix, n);
    let mut scratch = vec![0.0; n.saturating_mul(2)];
    let mut column = vec![0.0; n];
    let kd = NB;
    for k in 0..n - 2 {
        let end = n.min(k + 1 + kd);
        zero_column_tail(&mut band, k, k + 1, end, refs, &mut scratch, &mut column);
        let mut col = k + 1;
        let mut guard = 0usize;
        let guard_limit = n.saturating_mul(kd.max(1));
        while col + 2 < n && guard < guard_limit {
            guard += 1;
            let bulge_row = col + kd + 1;
            let scan_hi = n.min(col + band.kd + 1);
            if bulge_row >= scan_hi {
                break;
            }
            let mut last = col + 1;
            let mut found = false;
            for r in (bulge_row..scan_hi).rev() {
                if band.get(r, col).abs() > 1e-18 {
                    last = r + 1;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
            let before = refs.len();
            zero_column_tail(
                &mut band,
                col,
                col + kd,
                last,
                refs,
                &mut scratch,
                &mut column,
            );
            if refs.len() == before {
                break;
            }
            col += kd;
        }
    }
    matrix.fill(0.0);
    band.scatter(matrix);
}

fn zero_column_tail(
    band: &mut SymBand,
    col: usize,
    row0: usize,
    row1: usize,
    refs: &mut Vec<Reflector>,
    scratch: &mut [f64],
    column: &mut [f64],
) {
    if row1 <= row0 + 1 || row0 >= band.n {
        return;
    }
    let row1 = row1.min(band.n);
    let len = row1 - row0;
    let x = &mut column[..len];
    for (i, slot) in x.iter_mut().enumerate() {
        *slot = band.get(row0 + i, col);
    }
    let (v, tau, beta) = householder(x);
    if tau == 0.0 {
        return;
    }
    apply_sym_householder(band, row0, &v, tau, scratch);
    band.set(row0, col, beta);
    band.set(col, row0, beta);
    for r in row0 + 1..row1 {
        band.set(r, col, 0.0);
        band.set(col, r, 0.0);
    }
    refs.push(Reflector {
        start: row0,
        v,
        tau,
    });
}

fn apply_sym_householder(
    band: &mut SymBand,
    start: usize,
    v: &[f64],
    tau: f64,
    scratch: &mut [f64],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_sym_householder_avx2(band, start, v, tau, scratch);
            }
            return;
        }
    }
    apply_sym_householder_scalar(band, start, v, tau, scratch);
}

fn apply_sym_householder_scalar(
    band: &mut SymBand,
    start: usize,
    v: &[f64],
    tau: f64,
    scratch: &mut [f64],
) {
    let Some((lo, m, v_at)) = householder_window(band, start, v, tau, scratch) else {
        return;
    };
    let n = band.n;
    let (y, vwin) = scratch.split_at_mut(n);
    let y = &mut y[..m];
    let vwin = &vwin[..m];
    let mut vmv = 0.0;
    for (j, &vj) in v.iter().enumerate() {
        vmv += vj * y[v_at + j];
    }
    let tau2 = tau * tau * vmv;
    for j in 0..m {
        let vj = vwin[j];
        let yj = y[j];
        if vj == 0.0 && yj == 0.0 {
            continue;
        }
        let scale_v = tau2 * vj - tau * yj;
        let scale_y = -tau * vj;
        let col = lo + j;
        let (base, off, count) = band_span(band, col, lo, m);
        for i in 0..count {
            band.data[base + i] += vwin[off + i] * scale_v + y[off + i] * scale_y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_sym_householder_avx2(
    band: &mut SymBand,
    start: usize,
    v: &[f64],
    tau: f64,
    scratch: &mut [f64],
) {
    unsafe {
        let Some((lo, m, v_at)) = householder_window(band, start, v, tau, scratch) else {
            return;
        };
        let n = band.n;
        let (y, vwin) = scratch.split_at_mut(n);
        let y = &mut y[..m];
        let vwin = &vwin[..m];
        let mut vmv = 0.0;
        for (j, &vj) in v.iter().enumerate() {
            vmv += vj * y[v_at + j];
        }
        let tau2 = tau * tau * vmv;
        for j in 0..m {
            let vj = vwin[j];
            let yj = y[j];
            if vj == 0.0 && yj == 0.0 {
                continue;
            }
            let scale_v = tau2 * vj - tau * yj;
            let scale_y = -tau * vj;
            let col = lo + j;
            let (base, off, count) = band_span(band, col, lo, m);
            accumulate_avx2(
                band.data.as_mut_ptr().add(base),
                vwin.as_ptr().add(off),
                y.as_ptr().add(off),
                count,
                scale_v,
                scale_y,
            );
        }
    }
}

/// Fills `scratch` with `y` then `v` on the reflector window.
/// Returns `(lo, m, v_at)` when the reflector is live.
fn householder_window(
    band: &SymBand,
    start: usize,
    v: &[f64],
    tau: f64,
    scratch: &mut [f64],
) -> Option<(usize, usize, usize)> {
    let len = v.len();
    if tau == 0.0 || len == 0 {
        return None;
    }
    // The chase fills out to about two tiles. A window sized from the
    // reflector alone misses that band when the reflector is short.
    // One tile past the reflector covers the fill. Half a tile does not.
    let n = band.n;
    let lo = start.saturating_sub(NB);
    let hi = (start + len + NB).min(n);
    let m = hi - lo;
    let (y, vwin) = scratch.split_at_mut(n);
    let y = &mut y[..m];
    let vwin = &mut vwin[..m];
    y.fill(0.0);
    vwin.fill(0.0);
    let v_at = start - lo;
    vwin[v_at..v_at + len].copy_from_slice(v);
    for (j, &vj) in v.iter().enumerate() {
        if vj == 0.0 {
            continue;
        }
        let col = start + j;
        let (base, off, count) = band_span(band, col, lo, m);
        let column = &band.data[base..base + count];
        for (slot, &ai) in y[off..off + count].iter_mut().zip(column) {
            *slot += ai * vj;
        }
    }
    Some((lo, m, v_at))
}

#[inline]
fn band_span(band: &SymBand, col: usize, lo: usize, m: usize) -> (usize, usize, usize) {
    let hi = lo + m;
    let row0 = lo.max(col.saturating_sub(band.kd));
    let row1 = hi.min(col + band.kd + 1);
    if row1 <= row0 {
        return (0, 0, 0);
    }
    let base = col * band.ld + (row0 + band.kd - col);
    (base, row0 - lo, row1 - row0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(
    matrix: *mut f64,
    vwin: *const f64,
    y: *const f64,
    count: usize,
    scale_v: f64,
    scale_y: f64,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    unsafe {
        let sv = _mm256_set1_pd(scale_v);
        let sy = _mm256_set1_pd(scale_y);
        let mut i = 0;
        while i + 4 <= count {
            let vi = _mm256_loadu_pd(vwin.add(i));
            let yi = _mm256_loadu_pd(y.add(i));
            let acc = _mm256_loadu_pd(matrix.add(i));
            let prod = _mm256_add_pd(_mm256_mul_pd(vi, sv), _mm256_mul_pd(yi, sy));
            _mm256_storeu_pd(matrix.add(i), _mm256_add_pd(acc, prod));
            i += 4;
        }
        while i < count {
            *matrix.add(i) += *vwin.add(i) * scale_v + *y.add(i) * scale_y;
            i += 1;
        }
    }
}

fn householder(x: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = x.len();
    let mut v = vec![0.0; n];
    if n == 0 {
        return (v, 0.0, 0.0);
    }
    let mut sigma = 0.0;
    for value in x.iter().skip(1) {
        sigma += value * value;
    }
    let beta = x[0];
    if sigma == 0.0 {
        v[0] = 1.0;
        return (v, 0.0, beta);
    }
    let norm = beta.hypot(sigma.sqrt());
    let beta_out = if beta >= 0.0 { -norm } else { norm };
    v[0] = 1.0;
    let denom = x[0] - beta_out;
    for i in 1..n {
        v[i] = x[i] / denom;
    }
    let tau = (beta_out - x[0]) / beta_out;
    (v, tau, beta_out)
}

fn qr_panel(panel: &[f64], m: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pk = m.min(k);
    let mut r = panel.to_vec();
    let mut v = vec![0.0; m * pk];
    let mut tau = vec![0.0; pk];
    for j in 0..pk {
        let mut column = vec![0.0; m - j];
        for (i, slot) in column.iter_mut().enumerate() {
            *slot = r[j * m + j + i];
        }
        let (reflector, tau_j, beta) = householder(&column);
        tau[j] = tau_j;
        if tau_j != 0.0 && j + 1 < k {
            for col in j + 1..k {
                let mut dot = 0.0;
                for (i, &vi) in reflector.iter().enumerate() {
                    dot += vi * r[col * m + j + i];
                }
                let scale = tau_j * dot;
                for (i, &vi) in reflector.iter().enumerate() {
                    r[col * m + j + i] -= scale * vi;
                }
            }
        }
        r[j * m + j] = beta;
        for i in 1..reflector.len() {
            r[j * m + j + i] = reflector[i];
        }
        for (i, &vi) in reflector.iter().enumerate() {
            v[(j + i) + j * m] = vi;
        }
    }
    (v, tau, r)
}

fn dlarft(v: &[f64], m: usize, tau: &[f64], t: &mut [f64]) {
    let pk = tau.len();
    t.fill(0.0);
    for j in 0..pk {
        if tau[j] == 0.0 {
            continue;
        }
        if j > 0 {
            let mut y = vec![0.0; j];
            let rows = m - j;
            for c in 0..j {
                let mut dot = 0.0;
                for r in 0..rows {
                    dot += v[(j + r) + c * m] * v[(j + r) + j * m];
                }
                y[c] = -tau[j] * dot;
            }
            let mut product = vec![0.0; j];
            for r in 0..j {
                let mut acc = 0.0;
                for c in 0..j {
                    acc += t[r + c * pk] * y[c];
                }
                product[r] = acc;
            }
            for r in 0..j {
                t[r + j * pk] = product[r];
            }
        }
        t[j + j * pk] = tau[j];
    }
}

fn gemm_row_tiles(c: &[f64], pn: usize, s2: &[f64], pk: usize, w: &mut [f64]) {
    let n_tiles = pn.div_ceil(NB);
    let c_ptr = SyncPtr(c.as_ptr());
    let s2_ptr = SyncPtr(s2.as_ptr());
    let w_ptr = SyncMut(w.as_mut_ptr());
    for_tiles(n_tiles, move |tile| {
        let row0 = tile * NB;
        let rows = NB.min(pn - row0);
        tile_gemm(
            rows,
            pk,
            pn,
            1.0,
            c_ptr.add(row0),
            1,
            pn as isize,
            s2_ptr.get(),
            1,
            pn as isize,
            0.0,
            w_ptr.add(row0),
            1,
            pn as isize,
        );
    });
}

fn syr2k_tiles(c: &mut [f64], pn: usize, v: &[f64], w: &[f64], pk: usize) {
    let n_blocks = pn.div_ceil(NB);
    let n_tiles = n_blocks * (n_blocks + 1) / 2;
    let c_ptr = SyncMut(c.as_mut_ptr());
    let v_ptr = SyncPtr(v.as_ptr());
    let w_ptr = SyncPtr(w.as_ptr());
    for_tiles(n_tiles, move |tile| {
        let (bi, bj) = lower_tile(n_blocks, tile);
        let row0 = bi * NB;
        let col0 = bj * NB;
        let rows = NB.min(pn - row0);
        let cols = NB.min(pn - col0);
        // C_ij -= V_i W_j^T + W_i V_j^T.
        tile_gemm(
            rows,
            cols,
            pk,
            -1.0,
            v_ptr.add(row0),
            1,
            pn as isize,
            w_ptr.add(col0),
            pn as isize,
            1,
            1.0,
            c_ptr.add(col0 * pn + row0),
            1,
            pn as isize,
        );
        tile_gemm(
            rows,
            cols,
            pk,
            -1.0,
            w_ptr.add(row0),
            1,
            pn as isize,
            v_ptr.add(col0),
            pn as isize,
            1,
            1.0,
            c_ptr.add(col0 * pn + row0),
            1,
            pn as isize,
        );
    });
    // The tiles wrote the lower triangle, including the lower half of
    // each diagonal block. Copy that onto the upper triangle.
    for col in 0..pn {
        for row in (col + 1)..pn {
            c[row * pn + col] = c[col * pn + row];
        }
    }
}

fn lower_tile(n_blocks: usize, mut tile: usize) -> (usize, usize) {
    let mut bj = 0;
    while tile >= n_blocks - bj {
        tile -= n_blocks - bj;
        bj += 1;
    }
    (bj + tile, bj)
}

fn apply_reflectors(q: &mut [f64], n: usize, ncols: usize, refs: &[Reflector]) {
    if ncols == 0 || refs.is_empty() {
        return;
    }
    let tasks = column_tasks(ncols);
    let q_ptr = SyncMut(q.as_mut_ptr());
    for_tiles(tasks, move |tile| {
        let (col0, count) = split_range(ncols, tasks, tile);
        for col in col0..col0 + count {
            for reflector in refs.iter().rev() {
                if reflector.tau == 0.0 {
                    continue;
                }
                let mut dot = 0.0;
                for (i, &vi) in reflector.v.iter().enumerate() {
                    dot += vi * unsafe { *q_ptr.add(col * n + reflector.start + i) };
                }
                let scale = reflector.tau * dot;
                for (i, &vi) in reflector.v.iter().enumerate() {
                    unsafe {
                        *q_ptr.add(col * n + reflector.start + i) -= scale * vi;
                    }
                }
            }
        }
    });
}

fn column_tasks(ncols: usize) -> usize {
    let capped = EIGH_TASKS.min(ncols).max(1);
    1usize << (usize::BITS - 1 - capped.leading_zeros())
}

fn split_range(n: usize, tasks: usize, tile: usize) -> (usize, usize) {
    let base = n / tasks;
    let extra = n % tasks;
    if tile < extra {
        (tile * (base + 1), base + 1)
    } else {
        (extra * (base + 1) + (tile - extra) * base, base)
    }
}

fn tridiag_par(n: usize) -> faer::Par {
    #[cfg(feature = "par")]
    {
        if n < 128 {
            faer::Par::Seq
        } else {
            let q = n / 128;
            let tasks = if q <= 1 {
                1
            } else {
                let floor = 1usize << (usize::BITS - 1 - q.leading_zeros());
                floor.clamp(2, EIGH_TASKS)
            };
            if tasks <= 1 {
                faer::Par::Seq
            } else {
                faer::Par::rayon(tasks)
            }
        }
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = n;
        faer::Par::Seq
    }
}

fn tridiagonal_full(diag: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    let n = diag.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if n == 1 {
        return Ok((vec![diag[0]], vec![1.0]));
    }
    let mut evals = diag.to_vec();
    let mut vectors = vec![0.0; n * n];
    let par = tridiag_par(n);
    let params = faer::Spec::<faer::linalg::evd::SelfAdjointEvdParams, f64>::default();
    let scratch = faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        params,
    );
    let mut buffer = dyn_stack::MemBuffer::new(scratch);
    let diag_ref = faer::diag::DiagRef::from_slice(diag);
    let off_ref = faer::diag::DiagRef::from_slice(off);
    let values = faer::diag::DiagMut::from_slice_mut(&mut evals);
    let u = faer::MatMut::from_column_major_slice_mut(&mut vectors, n, n);
    faer::linalg::evd::tridiagonal_self_adjoint_evd(
        diag_ref,
        off_ref,
        values,
        Some(u),
        par,
        dyn_stack::MemStack::new(&mut buffer),
        params,
    )
    .map_err(|_| SaddleError::Solver("tridiagonal eigensolver did not converge".into()))?;
    Ok((evals, vectors))
}

fn tridiagonal_partial(
    diag: &[f64],
    off: &[f64],
    k: usize,
) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    let n = diag.len();
    let (lo, hi) = gershgorin(diag, off);
    let mut evals = Vec::with_capacity(k);
    for which in 0..k {
        evals.push(bisect(diag, off, which, lo, hi));
    }
    let mut vectors = vec![0.0; n * k];
    let scale = hi.abs().max(lo.abs()).max(1.0);
    for (j, &lambda) in evals.iter().enumerate() {
        let mut y = vec![0.0; n];
        y[j % n] = 1.0;
        for _ in 0..6 {
            let shifted: Vec<f64> = diag.iter().map(|value| value - lambda).collect();
            let mut z = tridiag_solve(off, &shifted, off, &y);
            for prev in 0..j {
                let mut dot = 0.0;
                for row in 0..n {
                    dot += vectors[prev * n + row] * z[row];
                }
                for row in 0..n {
                    z[row] -= dot * vectors[prev * n + row];
                }
            }
            let mut norm = 0.0;
            for value in &z {
                norm += value * value;
            }
            norm = norm.sqrt();
            if norm < 1e-14 {
                y.fill(0.0);
                y[(j.saturating_mul(5).saturating_add(3)) % n] = 1.0;
                continue;
            }
            for (slot, value) in y.iter_mut().zip(z) {
                *slot = value / norm;
            }
        }
        let mut residual = 0.0;
        for row in 0..n {
            let mut acc = diag[row] * y[row];
            if row > 0 {
                acc += off[row - 1] * y[row - 1];
            }
            if row + 1 < n {
                acc += off[row] * y[row + 1];
            }
            let diff = acc - lambda * y[row];
            residual += diff * diff;
        }
        if residual.sqrt() > 1e-6 * scale * (n as f64).sqrt() {
            return Err(SaddleError::Solver(
                "inverse iteration did not converge".into(),
            ));
        }
        for row in 0..n {
            vectors[j * n + row] = y[row];
        }
    }
    Ok((evals, vectors))
}

fn gershgorin(diag: &[f64], off: &[f64]) -> (f64, f64) {
    let n = diag.len();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let mut radius = 0.0;
        if i > 0 {
            radius += off[i - 1].abs();
        }
        if i + 1 < n {
            radius += off[i].abs();
        }
        lo = lo.min(diag[i] - radius);
        hi = hi.max(diag[i] + radius);
    }
    let span = (hi - lo).abs().max(1.0);
    (lo - span * 1e-12, hi + span * 1e-12)
}

fn bisect(diag: &[f64], off: &[f64], which: usize, mut lo: f64, mut hi: f64) -> f64 {
    let scale = hi.abs().max(lo.abs()).max(1.0);
    let tol = 8.0 * f64::EPSILON * scale * (diag.len().max(1) as f64);
    for _ in 0..200 {
        if hi - lo <= tol {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if sturm(diag, off, mid) <= which {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn sturm(diag: &[f64], off: &[f64], sigma: f64) -> usize {
    let mut count = 0usize;
    let mut q = diag[0] - sigma;
    if q < 0.0 {
        count += 1;
    }
    q = soften(q);
    for i in 1..diag.len() {
        q = (diag[i] - sigma) - (off[i - 1] * off[i - 1]) / q;
        if q < 0.0 {
            count += 1;
        }
        q = soften(q);
    }
    count
}

fn soften(q: f64) -> f64 {
    if q.abs() < 1e-300 {
        if q < 0.0 { -1e-300 } else { 1e-300 }
    } else {
        q
    }
}

/// Solve `(diag, sub, super) x = rhs` with partial pivoting.
/// `sub` and `super` each have length `diag.len() - 1`.
fn tridiag_solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return Vec::new();
    }
    let mut a = sub.to_vec();
    let mut b = diag.to_vec();
    let mut c = sup.to_vec();
    let mut r = rhs.to_vec();
    let mut c2 = vec![0.0; n];
    for i in 0..n - 1 {
        if a[i].abs() > b[i].abs() {
            std::mem::swap(&mut b[i], &mut a[i]);
            std::mem::swap(&mut c[i], &mut b[i + 1]);
            if i + 1 < n - 1 {
                c2[i] = c[i + 1];
                c[i + 1] = 0.0;
            }
            r.swap(i, i + 1);
        }
        if b[i].abs() < 1e-30 {
            b[i] = if b[i] < 0.0 { -1e-30 } else { 1e-30 };
        }
        let factor = a[i] / b[i];
        b[i + 1] -= factor * c[i];
        if i + 1 < n - 1 {
            c[i + 1] -= factor * c2[i];
        }
        r[i + 1] -= factor * r[i];
    }
    if b[n - 1].abs() < 1e-30 {
        b[n - 1] = if b[n - 1] < 0.0 { -1e-30 } else { 1e-30 };
    }
    r[n - 1] /= b[n - 1];
    if n > 1 {
        r[n - 2] = (r[n - 2] - c[n - 2] * r[n - 1]) / b[n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        r[i] = (r[i] - c[i] * r[i + 1] - c2[i] * r[i + 2]) / b[i];
    }
    r
}

/// `C = alpha * A * B + beta * C` for one tile.
///
/// This is the only GEMM in the reduction and in the eigenvector
/// product. An Ozaki-II backend replaces the body. The call is
/// sequential inside the tile. The caller owns the tile, so the
/// product does not depend on which thread runs it.
#[allow(clippy::too_many_arguments)]
fn tile_gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_row_stride: isize,
    a_col_stride: isize,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_row_stride: isize,
    c_col_stride: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == 0.0 {
        if beta == 0.0 {
            for col in 0..n {
                for row in 0..m {
                    unsafe {
                        *c.offset(c_row_stride * row as isize + c_col_stride * col as isize) = 0.0;
                    }
                }
            }
        } else if beta != 1.0 {
            for col in 0..n {
                for row in 0..m {
                    unsafe {
                        let slot =
                            c.offset(c_row_stride * row as isize + c_col_stride * col as isize);
                        *slot *= beta;
                    }
                }
            }
        }
        return;
    }
    if beta != 0.0 && beta != 1.0 {
        for col in 0..n {
            for row in 0..m {
                unsafe {
                    let slot = c.offset(c_row_stride * row as isize + c_col_stride * col as isize);
                    *slot *= beta;
                }
            }
        }
    }
    let accum = if beta == 0.0 {
        faer::Accum::Replace
    } else {
        faer::Accum::Add
    };
    unsafe {
        let dst = faer::MatMut::<f64>::from_raw_parts_mut(c, m, n, c_row_stride, c_col_stride);
        let lhs = faer::MatRef::<f64>::from_raw_parts(a, m, k, a_row_stride, a_col_stride);
        let rhs = faer::MatRef::<f64>::from_raw_parts(b, k, n, b_row_stride, b_col_stride);
        faer::linalg::matmul::matmul(dst, accum, lhs, rhs, alpha, faer::Par::Seq);
    }
}

fn for_tiles(n_tasks: usize, f: impl Fn(usize) + Send + Sync) {
    if n_tasks == 0 {
        return;
    }
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        (0..n_tasks).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "par"))]
    {
        for task in 0..n_tasks {
            f(task);
        }
    }
}

/// Read pointer shared by tiles. A method keeps the wrapper as the
/// closure capture. A public field would be captured as a raw pointer,
/// which is not `Sync`.
#[derive(Clone, Copy)]
struct SyncPtr(*const f64);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    fn get(self) -> *const f64 {
        self.0
    }

    fn add(self, count: usize) -> *const f64 {
        unsafe { self.0.add(count) }
    }
}

/// Pointer shared by tiles that write disjoint entries.
#[derive(Clone, Copy)]
struct SyncMut(*mut f64);
unsafe impl Send for SyncMut {}
unsafe impl Sync for SyncMut {}

impl SyncMut {
    fn add(self, count: usize) -> *mut f64 {
        unsafe { self.0.add(count) }
    }
}

fn pack_evals(evals: &[f64]) -> Array1<f64> {
    Array1::from_vec(evals.to_vec())
}

fn pack_columns(columns: &[f64], n: usize, k: usize) -> Array2<f64> {
    let mut out = Array2::zeros((n, k));
    for col in 0..k {
        for row in 0..n {
            out[(row, col)] = columns[col * n + row];
        }
    }
    out
}

fn reject_zero_columns(columns: &[f64], n: usize, k: usize) -> Result<(), SaddleError> {
    for col in 0..k {
        let mut norm = 0.0;
        for row in 0..n {
            let value = columns[col * n + row];
            norm += value * value;
        }
        if norm.sqrt() < 1e-18 {
            return Err(SaddleError::Solver(
                "eigensolver produced a zero eigenvector".into(),
            ));
        }
    }
    Ok(())
}

/// Largest-magnitude entry of each column is positive. Ties keep the
/// lowest index. Negation flips the sign bit, so the rule is bit-stable
/// once the column is.
pub fn positive_max_component(vecs: &mut Array2<f64>) {
    let k = vecs.ncols();
    let n = vecs.nrows();
    for col in 0..k {
        let mut best = 0usize;
        let mut best_abs = 0.0f64;
        for row in 0..n {
            let magnitude = vecs[(row, col)].abs();
            if magnitude > best_abs {
                best_abs = magnitude;
                best = row;
            }
        }
        if vecs[(best, col)] < 0.0 {
            for row in 0..n {
                vecs[(row, col)] = -vecs[(row, col)];
            }
        }
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::*;

    fn eval_gap(a: &[f64], b: &[f64], n: usize) -> f64 {
        let mut left = Array2::zeros((n, n));
        let mut right = Array2::zeros((n, n));
        for col in 0..n {
            for row in 0..n {
                left[(row, col)] = a[col * n + row];
                right[(row, col)] = b[col * n + row];
            }
        }
        let (ea, _) = faer_sequential(&left).unwrap();
        let (eb, _) = faer_sequential(&right).unwrap();
        let mut gap = 0.0f64;
        for i in 0..n {
            gap = gap.max((ea[i] - eb[i]).abs());
        }
        gap
    }

    #[test]
    fn reduction_preserves_the_spectrum() {
        for n in [70usize, 140, 256] {
            let mut state = 0x70_u64.wrapping_add(n as u64);
            let mut a = vec![0.0; n * n];
            for col in 0..n {
                for row in col..n {
                    state = state
                        .wrapping_mul(0x5851_F42D_4C95_7F2D)
                        .wrapping_add(0x1405_7B7E_F767_814F);
                    let value = ((state >> 11) as u32 as f64 / u32::MAX as f64).mul_add(2.0, -1.0);
                    a[col * n + row] = value;
                    a[row * n + col] = value;
                }
            }
            let mut matrix = a.clone();
            let mut refs = dense_to_band(&mut matrix, n);
            assert!(eval_gap(&a, &matrix, n) < 1e-8, "n={n} band reduction");
            band_to_tridiagonal(&mut matrix, n, &mut refs);
            assert!(
                eval_gap(&a, &matrix, n) < 1e-8,
                "n={n} tridiagonal reduction"
            );
        }
    }
}
