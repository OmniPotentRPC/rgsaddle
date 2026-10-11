//! Two-stage dense symmetric eigensolver.
//!
//! Stage 1 reduces the symmetric matrix to bandwidth [`NB`] by blocked
//! Householder panels and a tiled trailing update (Azzam Haidar, Hatem
//! Ltaief, and Jack Dongarra, SC '11,
//! <https://doi.org/10.1145/2063384.2063394>). Stage 2 chases the bulge
//! down to tridiagonal form on a compact band of width [`CHASE_KD`],
//! which is wide enough for the fill and keeps each window contiguous.
//! The chase is a task graph. Task `(sweep, step)` becomes ready when
//! its column window is disjoint from every earlier sweep, the
//! dependency in that paper and in ELPA's two-stage solver (Marek et
//! al., *J. Phys.: Condens. Matter* 26 (2014) 213201,
//! <https://doi.org/10.1088/0953-8984/26/21/213201>). Each wave runs
//! that ready set, and a task may take several further steps while the
//! predicted windows stay disjoint. The task id is the sweep index and
//! the step, and reflectors are flushed in sweep order, so the pool
//! size does not change the bits.
//! The first stage is the symmetric band reduction of Bischof, Lang,
//! and Sun, *ACM Trans. Math. Softw.* 26 (2000) 581-601
//! (<https://doi.org/10.1145/365723.365735>). The tridiagonal
//! eigenproblem is Cuppen's divide-and-conquer
//! (J. J. M. Cuppen, *Numer. Math.* 36 (1980) 177-195,
//! <https://doi.org/10.1007/BF01396757>) as carried out by Gu and
//! Eisenstat, *SIAM J. Matrix Anal. Appl.* 16 (1995) 172-191
//! (<https://doi.org/10.1137/S0895479892241287>). The leaf below
//! [`crate::cuppen`] is the implicit-shift QL step of EISPACK `tql2`
//! (Smith et al., *Matrix Eigensystem Routines — EISPACK Guide*,
//! Springer LNCS 6, 1976, <https://doi.org/10.1007/3-540-07546-1>;
//! Bowdler, Martin, Reinsch, and Wilkinson, *Numer. Math.* 11 (1968)
//! 293-306, <https://doi.org/10.1007/BF02165223>). `partial_eigh`
//! bisects with a tolerance relative to ‖T‖, in the style of LAPACK
//! `dstebz`, and inverse-iterates inside each irreducible block, in
//! the style of LAPACK `dstein` (Dhillon, *BIT* 38 (1998) 685-704,
//! <https://doi.org/10.1007/BF02510408>).
//!
//! Every trailing update and the blocked back-transform go through
//! [`crate::simd_gemm::tile_gemm`]. That kernel accumulates
//! `acc = fma(a, b, acc)` in ascending `k`. Tile ownership depends on
//! the dimension and on [`NB`], not on the rayon pool. The WY product
//! width is [`WY_NB`]. [`NB`], [`WY_NB`], [`CHASE_KD`], the Cuppen
//! leaf size and the `n / 2` split, and the partial-spectrum constants
//! [`RESIDUAL_C`], [`INVERSE_ITERS`], and [`CLUSTER_REL`] are part of
//! the bit definition. [`CHASE_BATCH`] and
//! [`crate::simd_gemm::column_tasks`] do not change the bits: the
//! chase still flushes reflectors in sweep order, and the column
//! partition is a function of the dimension alone. None of these
//! constants depend on the instruction set, the core count, or the
//! thread count.

use ndarray::{Array1, Array2};

use crate::error::SaddleError;
use crate::simd_gemm::{self, SyncMut, SyncPtr, for_tiles, tile_gemm};

/// Tile size and target half-bandwidth. Fixed. Part of the bit
/// definition: a different width builds different Householder vectors.
const NB: usize = 64;

/// Columns accumulated into one WY product before the back-transform.
/// Fixed. Part of the bit definition, because the block changes the
/// order of the Householder product. A wider diamond also grows the
/// union of the row supports, so the product does more arithmetic and
/// `dlarft_general` builds a larger `T`. On the bench host, 128 columns
/// was slower than 64 (n=3000, one thread, one rep: 13.4 s against
/// 12.0 s). Not a hardware query and not a function of the pool.
const WY_NB: usize = 64;

/// Half-bandwidth stored during the bulge chase. The chase fill stays
/// inside `2 * NB - 1` subdiagonals; one extra diagonal is margin.
const CHASE_KD: usize = NB * 2;

/// Steps one chase task may run before the scheduler joins the wave.
/// The count is fixed. It does not read the pool, and it does not
/// change the bits: reflectors are still flushed in sweep order.
const CHASE_BATCH: usize = 8;

/// Seconds in the band reduction, the bulge chase, the tridiagonal
/// eigensolve, and the back-transformation.
#[doc(hidden)]
#[non_exhaustive]
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

/// Compact WY block `I - V T Vᵀ`. `v` is `rows × cols` column-major
/// and `t` is `cols × cols` column-major. The block acts on rows
/// `[row, row + rows)`.
struct WyPanel {
    row: usize,
    rows: usize,
    cols: usize,
    v: Vec<f64>,
    t: Vec<f64>,
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
    simd_gemm::refresh_simd()?;
    let mut seconds = StageSeconds::zero();
    let TridiagonalFactor {
        panels,
        chase,
        band_s,
        bulge_s,
        diag,
        off,
    } = tridiagonalize(a)?;
    seconds.band = band_s;
    seconds.bulge = bulge_s;
    let t0 = std::time::Instant::now();
    let (evals, mut vectors) = tridiagonal_full(&diag, &off)?;
    seconds.tridiagonal = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    apply_back_transform(&mut vectors, n, n, &panels, &chase);
    seconds.back_transform = t1.elapsed().as_secs_f64();
    reject_zero_columns(&vectors, n, n, tridiagonal_norm(&diag, &off))?;
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
    simd_gemm::refresh_simd()?;
    let mut seconds = StageSeconds::zero();
    let TridiagonalFactor {
        panels,
        chase,
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
    apply_back_transform(&mut vectors, n, k, &panels, &chase);
    seconds.back_transform = t1.elapsed().as_secs_f64();
    reject_zero_columns(&vectors, n, k, tridiagonal_norm(&diag, &off))?;
    Ok((pack_evals(&evals), pack_columns(&vectors, n, k), seconds))
}

/// faer's self-adjoint solver on `threads` workers.
///
/// `threads <= 1` is `Par::Seq`. A larger count is `Par::rayon`. The
/// bits depend on the CPU and the thread count. Not the shipped entry.
#[doc(hidden)]
#[cfg(feature = "faer-unpinned")]
pub fn faer_with_threads(
    a: &Array2<f64>,
    threads: usize,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
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
    let par = if threads <= 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(threads)
    };
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

/// faer's sequential self-adjoint solver, for the single-thread cost of
/// the fixed partition against `Par::Seq`. Not the shipped entry point.
/// The bits depend on the CPU.
#[doc(hidden)]
#[cfg(feature = "faer-unpinned")]
pub fn faer_sequential(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    faer_with_threads(a, 1)
}

impl StageSeconds {
    /// Median stage times for the bench harness.
    #[doc(hidden)]
    pub fn from_parts(band: f64, bulge: f64, tridiagonal: f64, back_transform: f64) -> Self {
        Self {
            band,
            bulge,
            tridiagonal,
            back_transform,
        }
    }

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
    panels: Vec<WyPanel>,
    chase: Vec<WyPanel>,
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
    let panels = dense_to_band(&mut matrix, n);
    let band_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let mut chase_refs = Vec::new();
    band_to_tridiagonal(&mut matrix, n, &mut chase_refs);
    let chase = group_chase(&chase_refs);
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
        panels,
        chase,
        band_s,
        bulge_s,
        diag,
        off,
    })
}

fn dense_to_band(matrix: &mut [f64], n: usize) -> Vec<WyPanel> {
    let mut panels = Vec::new();
    if n <= NB {
        return panels;
    }
    let mut panel_buf = Vec::new();
    let mut trailing = Vec::new();
    let mut s2 = Vec::new();
    let mut w = Vec::new();
    let mut s1 = Vec::new();
    let mut i = 0;
    while i + NB < n {
        let pn = n - (i + NB);
        let pk = pn.min(NB);
        if pn == 0 || pk == 0 {
            break;
        }
        grow_buf(&mut panel_buf, pn * NB);
        let panel = &mut panel_buf[..pn * NB];
        for c in 0..NB {
            let src = (i + c) * n + (i + NB);
            panel[c * pn..(c + 1) * pn].copy_from_slice(&matrix[src..src + pn]);
        }
        let (v, tau, r_panel) = qr_panel(panel, pn, NB);
        grow_buf(&mut trailing, pn * pn);
        let trailing = &mut trailing[..pn * pn];
        for c in 0..pn {
            let src = (i + NB + c) * n + (i + NB);
            trailing[c * pn..(c + 1) * pn].copy_from_slice(&matrix[src..src + pn]);
        }
        let mut t_factor = vec![0.0; pk * pk];
        dlarft(&v, pn, &tau, &mut t_factor);
        grow_buf(&mut s2, pn * pk);
        let s2 = &mut s2[..pn * pk];
        // SAFETY: `v` is `pn` by `pk`, `t_factor` is `pk` by `pk`, and
        // `s2` is `pn` by `pk`, all column-major.
        unsafe {
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
        }
        grow_buf(&mut w, pn * pk);
        let w = &mut w[..pn * pk];
        gemm_row_tiles(trailing, pn, s2, pk, w);
        grow_buf(&mut s1, pk * pk);
        let s1 = &mut s1[..pk * pk];
        // SAFETY: `s2` and `w` are `pn` by `pk`, `s1` is `pk` by `pk`,
        // and `v` is `pn` by `pk`.
        unsafe {
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
        }
        syr2k_tiles(trailing, pn, &v, w, pk);
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
        panels.push(WyPanel {
            row: i + NB,
            rows: pn,
            cols: pk,
            v,
            t: t_factor,
        });
        i += NB;
    }
    panels
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

/// Compact band addressed by a raw pointer so one chase wave can update
/// disjoint columns from several tasks.
#[derive(Clone, Copy)]
struct BandPtr {
    data: *mut f64,
    n: usize,
    kd: usize,
    ld: usize,
    /// Max-norm of the band at the start of the chase. Bulge tests
    /// scale with it.
    scale: f64,
}

// SAFETY: a chase wave shares this pointer only across tasks whose
// column windows are disjoint. Each task reads and writes its own
// columns.
unsafe impl Send for BandPtr {}
// SAFETY: same disjoint-column contract as `Send`.
unsafe impl Sync for BandPtr {}

impl BandPtr {
    fn index(self, row: usize, col: usize) -> Option<usize> {
        if row >= self.n || col >= self.n || row.abs_diff(col) > self.kd {
            return None;
        }
        let index = col * self.ld + (row + self.kd - col);
        if index >= self.n.saturating_mul(self.ld) {
            return None;
        }
        Some(index)
    }

    fn get(self, row: usize, col: usize) -> f64 {
        match self.index(row, col) {
            Some(index) => {
                // SAFETY: `index` was checked against `n * ld`, the
                // length of the band allocation.
                unsafe { *self.data.add(index) }
            }
            None => 0.0,
        }
    }

    fn set(self, row: usize, col: usize, value: f64) {
        let Some(index) = self.index(row, col) else {
            assert!(
                value == 0.0,
                "nonzero fill outside the symmetric band at ({row}, {col})"
            );
            return;
        };
        // SAFETY: `index` was checked against `n * ld`.
        unsafe {
            *self.data.add(index) = value;
        }
    }
}

fn band_to_tridiagonal(matrix: &mut [f64], n: usize, refs: &mut Vec<Reflector>) {
    if n < 3 {
        return;
    }
    let mut band = SymBand::from_dense(matrix, n);
    let scale = band
        .data
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let view = BandPtr {
        data: band.data.as_mut_ptr(),
        n: band.n,
        kd: band.kd,
        ld: band.ld,
        scale,
    };
    let nsweeps = n - 2;
    let mut sweeps: Vec<ChaseSweep> = (0..nsweeps)
        .map(|k| ChaseSweep {
            k,
            stage: ChaseStage::Annihilate,
            steps: 0,
            pending: Vec::new(),
        })
        .collect();
    // Task (sweep, step) is ready when its column window is disjoint
    // from every earlier sweep. That is the Haidar / ELPA wavefront.
    // Reflectors are flushed in sweep order, so the worker that runs a
    // ready task does not change the bits.
    let mut active: Vec<usize> = Vec::new();
    let mut next_k = 0usize;
    let mut flushed = 0usize;
    let guard_limit = n.saturating_mul(NB.max(1));
    let scratch_len = n.saturating_mul(2);
    let mut scratch_pool = Vec::new();
    let mut column_pool = Vec::new();
    let mut jobs: Vec<ChaseJob> = Vec::new();
    let mut outs: Vec<ChaseBatch> = Vec::new();
    loop {
        active.retain(|&index| {
            let sweep = &mut sweeps[index];
            let exhausted =
                matches!(sweep.stage, ChaseStage::Chase(_)) && sweep.steps >= guard_limit;
            if exhausted || stage_window(n, sweep.k, sweep.stage).is_none() {
                sweep.stage = ChaseStage::Done;
                false
            } else {
                true
            }
        });
        flush_chase(&mut sweeps, &mut flushed, refs);
        while next_k < nsweeps {
            let win = stage_window(n, next_k, ChaseStage::Annihilate).expect("annihilator window");
            let blocked = active.iter().any(|&index| {
                let sweep = &sweeps[index];
                let earlier = stage_window(n, sweep.k, sweep.stage).expect("active window");
                ranges_overlap(earlier, win)
            });
            if blocked {
                break;
            }
            active.push(next_k);
            next_k += 1;
        }
        if active.is_empty() {
            break;
        }
        let mut wave = Vec::new();
        for &index in &active {
            let sweep = &sweeps[index];
            let win = stage_window(n, sweep.k, sweep.stage).expect("wave window");
            let blocked = active.iter().any(|&earlier| {
                let previous = &sweeps[earlier];
                if previous.k >= sweep.k {
                    return false;
                }
                let left = stage_window(n, previous.k, previous.stage).expect("earlier window");
                ranges_overlap(left, win)
            });
            if !blocked {
                wave.push(index);
            }
        }
        debug_assert!(!wave.is_empty());
        let batches = batch_lengths(n, &sweeps, &wave);
        jobs.clear();
        for (slot, &index) in wave.iter().enumerate() {
            jobs.push(ChaseJob {
                k: sweeps[index].k,
                stage: sweeps[index].stage,
                steps: batches[slot],
            });
        }
        let slots = jobs.len();
        if scratch_pool.len() < slots * scratch_len {
            scratch_pool.resize(slots * scratch_len, 0.0);
        }
        if column_pool.len() < slots * n {
            column_pool.resize(slots * n, 0.0);
        }
        if outs.len() < slots {
            outs.resize_with(slots, ChaseBatch::new);
        }
        run_chase_wave(
            view,
            &jobs,
            &mut scratch_pool[..slots * scratch_len],
            &mut column_pool[..slots * n],
            &mut outs[..slots],
        );
        for (slot, &index) in wave.iter().enumerate() {
            let batch = &mut outs[slot];
            sweeps[index].pending.append(&mut batch.reflectors);
            sweeps[index].steps += batch.chase_steps;
            sweeps[index].stage = batch.next;
        }
        active.retain(|&index| !matches!(sweeps[index].stage, ChaseStage::Done));
        flush_chase(&mut sweeps, &mut flushed, refs);
    }
    flush_chase(&mut sweeps, &mut flushed, refs);
    debug_assert_eq!(flushed, nsweeps);
    matrix.fill(0.0);
    band.scatter(matrix);
}

#[derive(Clone, Copy)]
enum ChaseStage {
    Annihilate,
    Chase(usize),
    Done,
}

struct ChaseSweep {
    k: usize,
    stage: ChaseStage,
    steps: usize,
    pending: Vec<Reflector>,
}

struct ChaseOut {
    reflector: Option<Reflector>,
    next: ChaseStage,
}

struct ChaseBatch {
    reflectors: Vec<Reflector>,
    chase_steps: usize,
    next: ChaseStage,
}

impl ChaseBatch {
    fn new() -> Self {
        Self {
            reflectors: Vec::new(),
            chase_steps: 0,
            next: ChaseStage::Done,
        }
    }
}

fn stage_window(n: usize, k: usize, stage: ChaseStage) -> Option<(usize, usize)> {
    match stage {
        ChaseStage::Done => None,
        ChaseStage::Annihilate => {
            let row0 = k + 1;
            if row0 >= n {
                return None;
            }
            let len_max = NB.min(n - row0);
            Some(touch_window(n, k, row0, len_max))
        }
        ChaseStage::Chase(col) => {
            if col + 2 >= n {
                return None;
            }
            let row0 = col + NB;
            if row0 >= n {
                return None;
            }
            let scan_hi = n.min(col + CHASE_KD + 1);
            if row0 + 1 >= scan_hi {
                return None;
            }
            let len_max = scan_hi - row0;
            Some(touch_window(n, col, row0, len_max))
        }
    }
}

fn touch_window(n: usize, col: usize, row0: usize, len_max: usize) -> (usize, usize) {
    let lo = row0.saturating_sub(NB).min(col);
    let hi = (row0 + len_max + NB).min(n);
    (lo, hi)
}

fn ranges_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

/// Next stage if the bulge is still present. An early exit in the
/// kernel reserves columns it then does not write.
fn predicted_next(n: usize, k: usize, stage: ChaseStage) -> ChaseStage {
    match stage {
        ChaseStage::Done => ChaseStage::Done,
        ChaseStage::Annihilate => {
            let col = k + 1;
            if col + 2 < n {
                ChaseStage::Chase(col)
            } else {
                ChaseStage::Done
            }
        }
        ChaseStage::Chase(col) => {
            let next_col = col + NB;
            if next_col + 2 < n {
                ChaseStage::Chase(next_col)
            } else {
                ChaseStage::Done
            }
        }
    }
}

/// Steps one sweep can take before its window meets another sweep.
/// How many steps each sweep in this wave can take without entering
/// another sweep's reserved columns.
fn batch_lengths(n: usize, sweeps: &[ChaseSweep], wave: &[usize]) -> Vec<u8> {
    let mut reserves: Vec<(usize, usize)> = wave
        .iter()
        .map(|&index| stage_window(n, sweeps[index].k, sweeps[index].stage).expect("wave window"))
        .collect();
    let mut batches = vec![1u8; wave.len()];
    for _ in 1..CHASE_BATCH {
        let mut grew = false;
        for slot in 0..wave.len() {
            if batches[slot] as usize >= CHASE_BATCH {
                continue;
            }
            let index = wave[slot];
            let mut stage = sweeps[index].stage;
            for _ in 0..batches[slot] {
                stage = predicted_next(n, sweeps[index].k, stage);
            }
            let Some(win) = stage_window(n, sweeps[index].k, stage) else {
                continue;
            };
            let hits = reserves
                .iter()
                .enumerate()
                .any(|(other, reserved)| other != slot && ranges_overlap(win, *reserved));
            if hits {
                continue;
            }
            reserves[slot].0 = reserves[slot].0.min(win.0);
            reserves[slot].1 = reserves[slot].1.max(win.1);
            batches[slot] = batches[slot].saturating_add(1);
            grew = true;
        }
        if !grew {
            break;
        }
    }
    batches
}

#[derive(Clone, Copy)]
struct ChaseJob {
    k: usize,
    stage: ChaseStage,
    steps: u8,
}

#[derive(Clone, Copy)]
struct ChaseSlots(*mut ChaseBatch);
// SAFETY: each task writes its own slot. The slots partition the
// outcome buffer, and the buffer outlives the wave.
unsafe impl Send for ChaseSlots {}
// SAFETY: same per-slot contract as `Send`.
unsafe impl Sync for ChaseSlots {}

impl ChaseSlots {
    /// # Safety
    ///
    /// `index` must be a slot of the outcome buffer this pointer was
    /// derived from, and that slot must not be written by another task.
    unsafe fn batch(self, index: usize) -> *mut ChaseBatch {
        unsafe { self.0.add(index) }
    }
}

#[derive(Clone, Copy)]
struct ChaseScratch {
    scratch: *mut f64,
    column: *mut f64,
}
// SAFETY: task `t` uses scratch and column strides that do not overlap
// any other task's stride.
unsafe impl Send for ChaseScratch {}
// SAFETY: same stride contract as `Send`.
unsafe impl Sync for ChaseScratch {}

impl ChaseScratch {
    /// # Safety
    ///
    /// `index * len` elements from `scratch` must lie inside the pool
    /// this pointer was derived from, and no other task may write them.
    unsafe fn scratch(self, index: usize, len: usize) -> *mut f64 {
        unsafe { self.scratch.add(index * len) }
    }

    /// # Safety
    ///
    /// `index * n` elements from `column` must lie inside the pool this
    /// pointer was derived from, and no other task may write them.
    unsafe fn column(self, index: usize, n: usize) -> *mut f64 {
        unsafe { self.column.add(index * n) }
    }
}

#[derive(Clone, Copy)]
struct JobPtr(*const ChaseJob);
// SAFETY: tasks only read the job list.
unsafe impl Send for JobPtr {}
// SAFETY: same read-only contract as `Send`.
unsafe impl Sync for JobPtr {}

impl JobPtr {
    /// # Safety
    ///
    /// `index` must be inside the job list this pointer was derived from.
    unsafe fn get(self, index: usize) -> ChaseJob {
        unsafe { *self.0.add(index) }
    }
}

fn run_batch(
    band: BandPtr,
    k: usize,
    mut stage: ChaseStage,
    steps: u8,
    scratch: &mut [f64],
    column: &mut [f64],
    into: &mut ChaseBatch,
) {
    into.reflectors.clear();
    into.chase_steps = 0;
    for _ in 0..steps {
        if matches!(stage, ChaseStage::Done) {
            break;
        }
        let out = chase_job(band, k, stage, scratch, column);
        if let Some(reflector) = out.reflector {
            into.reflectors.push(reflector);
        }
        if matches!(stage, ChaseStage::Chase(_)) {
            into.chase_steps += 1;
        }
        stage = out.next;
    }
    into.next = stage;
}

fn flush_chase(sweeps: &mut [ChaseSweep], flushed: &mut usize, refs: &mut Vec<Reflector>) {
    while *flushed < sweeps.len() && matches!(sweeps[*flushed].stage, ChaseStage::Done) {
        refs.append(&mut sweeps[*flushed].pending);
        *flushed += 1;
    }
}

fn run_chase_wave(
    band: BandPtr,
    jobs: &[ChaseJob],
    scratch_pool: &mut [f64],
    column_pool: &mut [f64],
    outs: &mut [ChaseBatch],
) {
    let n = band.n;
    let scratch_len = n.saturating_mul(2);
    // A wave of one to three sweeps is a few microseconds of band
    // updates. Joining the pool there costs more than the kernel.
    if jobs.len() < 4 {
        for (task, job) in jobs.iter().enumerate() {
            let scratch = &mut scratch_pool[task * scratch_len..(task + 1) * scratch_len];
            let column = &mut column_pool[task * n..(task + 1) * n];
            run_batch(
                band,
                job.k,
                job.stage,
                job.steps,
                scratch,
                column,
                &mut outs[task],
            );
        }
        return;
    }
    let slots = ChaseSlots(outs.as_mut_ptr());
    let pool = ChaseScratch {
        scratch: scratch_pool.as_mut_ptr(),
        column: column_pool.as_mut_ptr(),
    };
    let job_ptr = JobPtr(jobs.as_ptr());
    // SAFETY: task `t` writes outcome slot `t` and its own scratch and
    // column strides. The scheduler put disjoint band-column windows
    // in this wave, including every batched step, so the band writes
    // do not alias.
    for_tiles(jobs.len(), move |task| {
        let job = unsafe { job_ptr.get(task) };
        let scratch =
            unsafe { std::slice::from_raw_parts_mut(pool.scratch(task, scratch_len), scratch_len) };
        let column = unsafe { std::slice::from_raw_parts_mut(pool.column(task, n), n) };
        let batch = unsafe { &mut *slots.batch(task) };
        run_batch(band, job.k, job.stage, job.steps, scratch, column, batch);
    });
}

fn chase_job(
    band: BandPtr,
    k: usize,
    stage: ChaseStage,
    scratch: &mut [f64],
    column: &mut [f64],
) -> ChaseOut {
    match stage {
        ChaseStage::Annihilate => {
            let end = band.n.min(k + 1 + NB);
            let reflector = zero_column_tail(band, k, k + 1, end, scratch, column);
            let col = k + 1;
            let next = if col + 2 < band.n {
                ChaseStage::Chase(col)
            } else {
                ChaseStage::Done
            };
            ChaseOut { reflector, next }
        }
        ChaseStage::Chase(col) => chase_one(band, col, scratch, column),
        ChaseStage::Done => ChaseOut {
            reflector: None,
            next: ChaseStage::Done,
        },
    }
}

fn chase_one(band: BandPtr, col: usize, scratch: &mut [f64], column: &mut [f64]) -> ChaseOut {
    let n = band.n;
    let bulge_row = col + NB + 1;
    let scan_hi = n.min(col + band.kd + 1);
    if bulge_row >= scan_hi {
        return ChaseOut {
            reflector: None,
            next: ChaseStage::Done,
        };
    }
    let mut last = col + 1;
    let mut found = false;
    for row in (bulge_row..scan_hi).rev() {
        // A bulge below a few ulps of the band norm is rounding dust.
        // The test scales with that norm, so a Hessian in atomic units
        // is not compared with an absolute cutoff.
        let bulge_tol = 8.0 * f64::EPSILON * band.scale;
        if band.get(row, col).abs() > bulge_tol {
            last = row + 1;
            found = true;
            break;
        }
    }
    if !found {
        return ChaseOut {
            reflector: None,
            next: ChaseStage::Done,
        };
    }
    let reflector = zero_column_tail(band, col, col + NB, last, scratch, column);
    if reflector.is_none() {
        return ChaseOut {
            reflector: None,
            next: ChaseStage::Done,
        };
    }
    let next_col = col + NB;
    let next = if next_col + 2 < n {
        ChaseStage::Chase(next_col)
    } else {
        ChaseStage::Done
    };
    ChaseOut { reflector, next }
}

fn zero_column_tail(
    band: BandPtr,
    col: usize,
    row0: usize,
    row1: usize,
    scratch: &mut [f64],
    column: &mut [f64],
) -> Option<Reflector> {
    if row1 <= row0 + 1 || row0 >= band.n {
        return None;
    }
    let row1 = row1.min(band.n);
    let len = row1 - row0;
    let x = &mut column[..len];
    for (i, slot) in x.iter_mut().enumerate() {
        *slot = band.get(row0 + i, col);
    }
    let (v, tau, beta) = householder(x);
    if tau == 0.0 {
        return None;
    }
    apply_sym_householder(band, row0, &v, tau, scratch);
    band.set(row0, col, beta);
    band.set(col, row0, beta);
    for r in row0 + 1..row1 {
        band.set(r, col, 0.0);
        band.set(col, r, 0.0);
    }
    Some(Reflector {
        start: row0,
        v,
        tau,
    })
}

fn apply_sym_householder(band: BandPtr, start: usize, v: &[f64], tau: f64, scratch: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_gemm::bulge_uses_avx512() {
            // SAFETY: `bulge_uses_avx512` is true only when the CPU has
            // AVX-512. The kernel writes the same window as the scalar
            // update, one column at a time.
            unsafe {
                apply_sym_householder_avx512(band, start, v, tau, scratch);
            }
            return;
        }
        if simd_gemm::bulge_uses_avx2() {
            // SAFETY: `bulge_uses_avx2` is true only when the CPU has
            // AVX2. The kernel writes the same window as the scalar
            // update, one column at a time.
            unsafe {
                apply_sym_householder_avx2(band, start, v, tau, scratch);
            }
            return;
        }
    }
    apply_sym_householder_scalar(band, start, v, tau, scratch);
}

fn apply_sym_householder_scalar(
    band: BandPtr,
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
        vmv = vj.mul_add(y[v_at + j], vmv);
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
        // SAFETY: `band_span` returns a range inside column `col`, and
        // this kernel only runs on columns inside its chase window.
        let dest = unsafe { std::slice::from_raw_parts_mut(band.data.add(base), count) };
        for i in 0..count {
            let acc = scale_y.mul_add(y[off + i], dest[i]);
            dest[i] = scale_v.mul_add(vwin[off + i], acc);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_sym_householder_avx2(
    band: BandPtr,
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
            vmv = vj.mul_add(y[v_at + j], vmv);
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
                band.data.add(base),
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
    band: BandPtr,
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
        // SAFETY: `band_span` returns a range inside column `col`.
        let column = unsafe { std::slice::from_raw_parts(band.data.add(base), count) };
        simd_gemm::axpy_add(&mut y[off..off + count], column, vj);
    }
    Some((lo, m, v_at))
}

#[inline]
fn band_span(band: BandPtr, col: usize, lo: usize, m: usize) -> (usize, usize, usize) {
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
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_sym_householder_avx512(
    band: BandPtr,
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
            vmv = vj.mul_add(y[v_at + j], vmv);
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
            accumulate_avx512(
                band.data.add(base),
                vwin.as_ptr().add(off),
                y.as_ptr().add(off),
                count,
                scale_v,
                scale_y,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn accumulate_avx512(
    matrix: *mut f64,
    vwin: *const f64,
    y: *const f64,
    count: usize,
    scale_v: f64,
    scale_y: f64,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_storeu_pd};
    unsafe {
        let sv = _mm512_set1_pd(scale_v);
        let sy = _mm512_set1_pd(scale_y);
        let mut i = 0;
        while i + 8 <= count {
            let vi = _mm512_loadu_pd(vwin.add(i));
            let yi = _mm512_loadu_pd(y.add(i));
            let mut acc = _mm512_loadu_pd(matrix.add(i));
            acc = _mm512_fmadd_pd(sy, yi, acc);
            acc = _mm512_fmadd_pd(sv, vi, acc);
            _mm512_storeu_pd(matrix.add(i), acc);
            i += 8;
        }
        while i < count {
            let acc = scale_y.mul_add(*y.add(i), *matrix.add(i));
            *matrix.add(i) = scale_v.mul_add(*vwin.add(i), acc);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_avx2(
    matrix: *mut f64,
    vwin: *const f64,
    y: *const f64,
    count: usize,
    scale_v: f64,
    scale_y: f64,
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};
    unsafe {
        let sv = _mm256_set1_pd(scale_v);
        let sy = _mm256_set1_pd(scale_y);
        let mut i = 0;
        while i + 4 <= count {
            let vi = _mm256_loadu_pd(vwin.add(i));
            let yi = _mm256_loadu_pd(y.add(i));
            let mut acc = _mm256_loadu_pd(matrix.add(i));
            acc = _mm256_fmadd_pd(sy, yi, acc);
            acc = _mm256_fmadd_pd(sv, vi, acc);
            _mm256_storeu_pd(matrix.add(i), acc);
            i += 4;
        }
        while i < count {
            let acc = scale_y.mul_add(*y.add(i), *matrix.add(i));
            *matrix.add(i) = scale_v.mul_add(*vwin.add(i), acc);
            i += 1;
        }
    }
}

fn householder(x: &[f64]) -> (Vec<f64>, f64, f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: this CPU has FMA. The body is the same `mul_add` chain.
        return unsafe { householder_fma(x) };
    }
    householder_body(x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn householder_fma(x: &[f64]) -> (Vec<f64>, f64, f64) {
    householder_body(x)
}

#[inline(always)]
fn householder_body(x: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = x.len();
    let mut v = vec![0.0; n];
    if n == 0 {
        return (v, 0.0, 0.0);
    }
    let mut sigma = 0.0;
    for value in x.iter().skip(1) {
        sigma = value.mul_add(*value, sigma);
    }
    let beta = x[0];
    if sigma == 0.0 {
        v[0] = 1.0;
        return (v, 0.0, beta);
    }
    let norm = crate::cuppen::hypot(beta, sigma.sqrt());
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
            // Each remaining column is an independent dot and axpy, in
            // the same index order as the scalar reflector.
            // SAFETY: columns `j + 1..k` of `r` exist, and the reflector
            // length is `m - j`, so each column's touched rows stay in `r`.
            unsafe {
                simd_gemm::rank1_columns(
                    r.as_mut_ptr().add((j + 1) * m + j),
                    m,
                    k - (j + 1),
                    reflector.as_ptr(),
                    reflector.len(),
                    tau_j,
                );
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
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: this CPU has FMA. The body is the same `mul_add` chain.
        unsafe { dlarft_fma(v, m, tau, t) }
        return;
    }
    dlarft_body(v, m, tau, t);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn dlarft_fma(v: &[f64], m: usize, tau: &[f64], t: &mut [f64]) {
    dlarft_body(v, m, tau, t);
}

#[inline(always)]
fn dlarft_body(v: &[f64], m: usize, tau: &[f64], t: &mut [f64]) {
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
                    dot = v[(j + r) + c * m].mul_add(v[(j + r) + j * m], dot);
                }
                y[c] = -tau[j] * dot;
            }
            let mut product = vec![0.0; j];
            for r in 0..j {
                let mut acc = 0.0;
                for c in 0..j {
                    acc = t[r + c * pk].mul_add(y[c], acc);
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
    let c_ptr = SyncPtr::new(c.as_ptr());
    let s2_ptr = SyncPtr::new(s2.as_ptr());
    let w_ptr = SyncMut::new(w.as_mut_ptr());
    // SAFETY: tile `tile` writes rows `[row0, row0 + rows)` of `w`.
    // Those row ranges partition `0..pn`, so the writes are disjoint.
    // `c` and `s2` are read-only.
    for_tiles(n_tiles, move |tile| {
        let row0 = tile * NB;
        let rows = NB.min(pn - row0);
        // SAFETY: this tile writes rows `[row0, row0 + rows)` of `w`.
        // `c` and `s2` are read-only and cover the product.
        unsafe {
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
        }
    });
}

fn syr2k_tiles(c: &mut [f64], pn: usize, v: &[f64], w: &[f64], pk: usize) {
    let n_blocks = pn.div_ceil(NB);
    let n_tiles = n_blocks * (n_blocks + 1) / 2;
    let c_ptr = SyncMut::new(c.as_mut_ptr());
    let v_ptr = SyncPtr::new(v.as_ptr());
    let w_ptr = SyncPtr::new(w.as_ptr());
    // SAFETY: each tile owns one lower-triangular block `(bi, bj)`.
    // Those blocks partition the lower triangle, so their writes in `c`
    // are disjoint. `v` and `w` are read-only. The upper triangle is
    // copied after every tile has finished.
    for_tiles(n_tiles, move |tile| {
        let (bi, bj) = lower_tile(n_blocks, tile);
        let row0 = bi * NB;
        let col0 = bj * NB;
        let rows = NB.min(pn - row0);
        let cols = NB.min(pn - col0);
        // C_ij -= V_i W_j^T + W_i V_j^T.
        // SAFETY: this tile owns the lower-triangular block `(bi, bj)`.
        // `v` and `w` are read-only.
        unsafe {
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
        }
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

/// Group chase reflectors the way ELPA2 and Haidar, Ltaief and Dongarra
/// apply `Q2`. Reflectors from consecutive sweeps at one band position
/// form a diamond of width at most [`WY_NB`]. That width is part of the
/// bit definition. Stored panels are applied in reverse, so the vector
/// sees the latest sweep at the latest position first.
/// Merge consecutive stage-1 panels until the column count reaches
/// [`WY_NB`]. One panel is left as the QR built it. Two or more are
/// rebuilt with [`dlarft_general`], which is the wider product.
fn group_chase(refs: &[Reflector]) -> Vec<WyPanel> {
    let mut sweeps: Vec<Vec<&Reflector>> = Vec::new();
    let mut current: Vec<&Reflector> = Vec::new();
    for reflector in refs {
        if reflector.tau == 0.0 || reflector.v.is_empty() {
            continue;
        }
        if let Some(previous) = current.last() {
            // One sweep walks down the band in steps of `NB`. The next
            // sweep starts a row lower than this sweep's first reflector,
            // which is not a step of `NB`.
            if reflector.start != previous.start + NB {
                sweeps.push(std::mem::take(&mut current));
            }
        }
        current.push(reflector);
    }
    if !current.is_empty() {
        sweeps.push(current);
    }
    let max_steps = sweeps.iter().map(Vec::len).max().unwrap_or(0);
    // Build panels in the order they are applied to the eigenvectors.
    // `apply_back_transform` walks the stored list backwards, so the
    // list is reversed once at the end. Within a band position the
    // latest sweep is applied first; band positions run from the top
    // of the matrix downward. That product matches the reverse of the
    // order in which the chase created the reflectors.
    let mut apply_order = Vec::new();
    for step in 0..max_steps {
        let mut sweep = sweeps.len();
        while sweep > 0 {
            if step >= sweeps[sweep - 1].len() {
                sweep -= 1;
                continue;
            }
            let mut group = Vec::new();
            while sweep > 0 && group.len() < WY_NB && step < sweeps[sweep - 1].len() {
                sweep -= 1;
                group.push(sweeps[sweep][step]);
            }
            // Columns run from the earliest sweep to the latest. The
            // forward WY product applies the latest column first.
            group.reverse();
            apply_order.push(panel_from_refs(&group));
        }
    }
    apply_order.reverse();
    apply_order
}

fn panel_from_refs(refs: &[&Reflector]) -> WyPanel {
    let lo = refs
        .iter()
        .map(|reflector| reflector.start)
        .min()
        .unwrap_or(0);
    let hi = refs
        .iter()
        .map(|reflector| reflector.start + reflector.v.len())
        .max()
        .unwrap_or(lo);
    let rows = (hi - lo).max(1);
    let cols = refs.len();
    let mut v = vec![0.0; rows * cols];
    let mut tau = vec![0.0; cols];
    for (col, reflector) in refs.iter().enumerate() {
        tau[col] = reflector.tau;
        let local = reflector.start - lo;
        for (row, &value) in reflector.v.iter().enumerate() {
            v[(local + row) + col * rows] = value;
        }
    }
    let mut t = vec![0.0; cols * cols];
    dlarft_general(&v, rows, &tau, &mut t);
    WyPanel {
        row: lo,
        rows,
        cols,
        v,
        t,
    }
}

/// `H_1 ... H_k = I - V T Vᵀ` for full columns of `V`, including the
/// explicit zeros outside each reflector's support.
fn dlarft_general(v: &[f64], m: usize, tau: &[f64], t: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: this CPU has FMA. The body is the same `mul_add` chain.
        unsafe { dlarft_general_fma(v, m, tau, t) }
        return;
    }
    dlarft_general_body(v, m, tau, t);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn dlarft_general_fma(v: &[f64], m: usize, tau: &[f64], t: &mut [f64]) {
    dlarft_general_body(v, m, tau, t);
}

#[inline(always)]
fn dlarft_general_body(v: &[f64], m: usize, tau: &[f64], t: &mut [f64]) {
    let pk = tau.len();
    t.fill(0.0);
    for j in 0..pk {
        if tau[j] == 0.0 {
            continue;
        }
        if j > 0 {
            let mut y = vec![0.0; j];
            for c in 0..j {
                let mut dot = 0.0;
                for r in 0..m {
                    dot = v[r + c * m].mul_add(v[r + j * m], dot);
                }
                y[c] = -tau[j] * dot;
            }
            for r in 0..j {
                let mut acc = 0.0;
                for c in 0..j {
                    acc = t[r + c * pk].mul_add(y[c], acc);
                }
                t[r + j * pk] = acc;
            }
        }
        t[j + j * pk] = tau[j];
    }
}

fn apply_back_transform(
    q: &mut [f64],
    n: usize,
    ncols: usize,
    panels: &[WyPanel],
    chase: &[WyPanel],
) {
    for panel in chase.iter().rev() {
        apply_wy(q, n, ncols, panel);
    }
    for panel in panels.iter().rev() {
        apply_wy(q, n, ncols, panel);
    }
}

struct WyScratch {
    vt: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

thread_local! {
    static WY_SCRATCH: std::cell::RefCell<WyScratch> = const {
        std::cell::RefCell::new(WyScratch {
            vt: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
        })
    };
}

fn grow_buf(buf: &mut Vec<f64>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
}

fn apply_wy(q: &mut [f64], n: usize, ncols: usize, panel: &WyPanel) {
    let m = panel.rows;
    let k = panel.cols;
    if k == 0 || m == 0 || ncols == 0 {
        return;
    }
    if k == 1 {
        simd_gemm::apply_rank1(q, n, ncols, panel.row, &panel.v[..m], panel.t[0]);
        return;
    }
    WY_SCRATCH.with(|cell| {
        let scratch = &mut *cell.borrow_mut();
        grow_buf(&mut scratch.vt, k * m);
        grow_buf(&mut scratch.y, k * ncols);
        grow_buf(&mut scratch.z, k * ncols);
        for col in 0..k {
            for row in 0..m {
                scratch.vt[col + row * k] = panel.v[row + col * m];
            }
        }
        let vt = scratch.vt.as_ptr();
        let y = scratch.y.as_mut_ptr();
        let z = scratch.z.as_mut_ptr();
        // SAFETY: `vt` is k by m, the row block of `q` is m by `ncols`
        // with column stride `n`, and `y` is k by `ncols`. The three
        // products run one after another on this thread. Inside each
        // product the column tiles of the destination are disjoint.
        // Workers only read these buffers, which stay allocated for the
        // duration of the products.
        unsafe {
            simd_gemm::gemm_nn(
                k,
                ncols,
                m,
                1.0,
                vt,
                k,
                q.as_ptr().add(panel.row),
                n,
                0.0,
                y,
                k,
            );
            simd_gemm::gemm_nn(k, ncols, k, 1.0, panel.t.as_ptr(), k, y, k, 0.0, z, k);
            simd_gemm::gemm_nn(
                m,
                ncols,
                k,
                -1.0,
                panel.v.as_ptr(),
                m,
                z,
                k,
                1.0,
                q.as_mut_ptr().add(panel.row),
                n,
            );
        }
    });
}

fn tridiagonal_full(diag: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    crate::cuppen::tridiagonal_evd(diag, off)
}

/// Residual factor in `c · n · ε · ‖T‖`. The cluster test asks for
/// `10 n ε`, so `c` is 10.
const RESIDUAL_C: f64 = 10.0;
const INVERSE_ITERS: usize = 40;
/// LAPACK `dstein` treats eigenvalues closer than this multiple of
/// ‖T‖ as one cluster.
const CLUSTER_REL: f64 = 1e-3;

fn tridiagonal_norm(diag: &[f64], off: &[f64]) -> f64 {
    let mut norm = 0.0f64;
    for i in 0..diag.len() {
        let mut row = diag[i].abs();
        if i > 0 {
            row += off[i - 1].abs();
        }
        if i + 1 < diag.len() {
            row += off[i].abs();
        }
        norm = norm.max(row);
    }
    norm
}

/// An off-diagonal at most `ε ‖T‖` splits the tridiagonal. The test is
/// relative to the matrix norm, so a zero coupling and a soft mode at
/// atomic-unit scale both separate.
fn negligible_off(value: f64, nrm: f64) -> bool {
    value.abs() <= f64::EPSILON * nrm
}

/// Irreducible blocks as half-open index ranges.
fn tridiagonal_blocks(off: &[f64], nrm: f64) -> Vec<(usize, usize)> {
    let n = off.len() + 1;
    let mut blocks = Vec::new();
    let mut start = 0usize;
    for (i, value) in off.iter().enumerate() {
        if negligible_off(*value, nrm) {
            blocks.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < n {
        blocks.push((start, n));
    }
    if blocks.is_empty() {
        blocks.push((0, n));
    }
    blocks
}

struct BlockEigen {
    value: f64,
    block: usize,
    local: usize,
}

/// Fixed start positions inside a block. The sequence does not depend
/// on the pool or on a random draw, so a restart stays bit-stable.
fn push_start(out: &mut Vec<usize>, idx: usize, m: usize) {
    if idx < m && !out.contains(&idx) {
        out.push(idx);
    }
}

fn start_positions(m: usize) -> Vec<usize> {
    if m == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    push_start(&mut out, 0, m);
    push_start(&mut out, m - 1, m);
    push_start(&mut out, m / 2, m);
    let mut step = m / 2;
    while step >= 1 && out.len() < 16 {
        let mut pos = step / 2;
        while pos < m && out.len() < 16 {
            push_start(&mut out, pos, m);
            pos += step.max(1);
        }
        if step == 1 {
            break;
        }
        step /= 2;
    }
    out
}

fn tridiagonal_partial(
    diag: &[f64],
    off: &[f64],
    k: usize,
) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    let n = diag.len();
    let nrm = tridiagonal_norm(diag, off);
    if nrm == 0.0 {
        let mut vectors = vec![0.0; n * k];
        for col in 0..k {
            vectors[col * n + col] = 1.0;
        }
        return Ok((vec![0.0; k], vectors));
    }
    let blocks = tridiagonal_blocks(off, nrm);
    // The k smallest eigenvalues of the union sit among the k smallest
    // of each block. A 1×1 block is its diagonal entry.
    let mut candidates = Vec::new();
    for (bi, &(start, end)) in blocks.iter().enumerate() {
        let m = end - start;
        let take = m.min(k);
        if m == 1 {
            candidates.push(BlockEigen {
                value: diag[start],
                block: bi,
                local: 0,
            });
            continue;
        }
        let block_diag = &diag[start..end];
        let block_off = &off[start..end - 1];
        let (lo, hi) = gershgorin(block_diag, block_off);
        for which in 0..take {
            candidates.push(BlockEigen {
                value: bisect(block_diag, block_off, which, lo, hi, nrm),
                block: bi,
                local: which,
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.value
            .total_cmp(&right.value)
            .then(left.block.cmp(&right.block))
            .then(left.local.cmp(&right.local))
    });
    candidates.truncate(k);
    let evals: Vec<f64> = candidates.iter().map(|item| item.value).collect();
    let mut cluster = vec![0usize; k];
    let mut id = 0usize;
    for j in 1..k {
        if (evals[j] - evals[j - 1]).abs() > CLUSTER_REL * nrm {
            id += 1;
        }
        cluster[j] = id;
    }
    let tol = RESIDUAL_C * (n as f64) * f64::EPSILON * nrm;
    let mut vectors = vec![0.0; n * k];
    for (j, cand) in candidates.iter().enumerate() {
        let (start, end) = blocks[cand.block];
        let m = end - start;
        if m == 1 {
            vectors[j * n + start] = 1.0;
            continue;
        }
        let cluster_prev: Vec<usize> = (0..j).filter(|&prev| cluster[prev] == cluster[j]).collect();
        let block_diag = &diag[start..end];
        let block_off = &off[start..end - 1];
        let lambda = cand.value;
        let positions = start_positions(m);
        let mut converged = false;
        let mut accepted = vec![0.0; m];
        for attempt in 0..=positions.len() {
            let mut y = vec![0.0; m];
            if attempt < positions.len() {
                y[positions[attempt]] = 1.0;
            } else {
                y.fill(1.0);
            }
            for _iter in 0..INVERSE_ITERS {
                let shifted: Vec<f64> = block_diag.iter().map(|value| value - lambda).collect();
                let mut z = tridiag_solve(block_off, &shifted, block_off, &y, nrm);
                for _ in 0..2 {
                    for &prev in &cluster_prev {
                        let mut dot = 0.0f64;
                        for row in 0..m {
                            let left: f64 = vectors[prev * n + start + row];
                            dot = left.mul_add(z[row], dot);
                        }
                        for row in 0..m {
                            let column: f64 = vectors[prev * n + start + row];
                            z[row] = (-dot).mul_add(column, z[row]);
                        }
                    }
                }
                let mut norm = 0.0;
                for value in &z {
                    norm = value.mul_add(*value, norm);
                }
                norm = norm.sqrt();
                if norm == 0.0 || !norm.is_finite() {
                    break;
                }
                for (slot, value) in y.iter_mut().zip(&z) {
                    *slot = value / norm;
                }
                let mut full = vec![0.0; n];
                full[start..end].copy_from_slice(&y);
                if tridiagonal_residual(diag, off, lambda, &full) <= tol {
                    accepted.clone_from(&y);
                    converged = true;
                    break;
                }
            }
            if converged {
                break;
            }
        }
        if !converged {
            return Err(SaddleError::Solver(
                "inverse iteration did not converge".into(),
            ));
        }
        vectors[j * n + start..j * n + end].copy_from_slice(&accepted);
    }
    Ok((evals, vectors))
}

fn tridiagonal_residual(diag: &[f64], off: &[f64], lambda: f64, y: &[f64]) -> f64 {
    let mut residual = 0.0;
    for row in 0..diag.len() {
        let mut acc = diag[row] * y[row];
        if row > 0 {
            acc = off[row - 1].mul_add(y[row - 1], acc);
        }
        if row + 1 < diag.len() {
            acc = off[row].mul_add(y[row + 1], acc);
        }
        let diff = acc - lambda * y[row];
        residual += diff * diff;
    }
    residual.sqrt()
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
    let span = (hi - lo).abs();
    let pad = f64::EPSILON * span.max(tridiagonal_norm(diag, off)) * (n as f64).max(1.0);
    (lo - pad, hi + pad)
}

fn bisect(diag: &[f64], off: &[f64], which: usize, mut lo: f64, mut hi: f64, nrm: f64) -> f64 {
    // LAPACK `dstebz` scales the bisection width by the matrix norm.
    // An absolute floor of 1 would leave a soft mode, whose residual
    // target is `c n ε ‖T‖`, with an eigenvalue coarser than that target.
    let scale = nrm.max(hi.abs()).max(lo.abs());
    let tol = 8.0 * f64::EPSILON * scale * (diag.len().max(1) as f64);
    for _ in 0..200 {
        if hi - lo <= tol {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if sturm(diag, off, mid, nrm) <= which {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn sturm(diag: &[f64], off: &[f64], sigma: f64, nrm: f64) -> usize {
    let mut count = 0usize;
    let mut q = diag[0] - sigma;
    if q < 0.0 {
        count += 1;
    }
    q = soften(q, nrm);
    for i in 1..diag.len() {
        q = (diag[i] - sigma) - (off[i - 1] * off[i - 1]) / q;
        if q < 0.0 {
            count += 1;
        }
        q = soften(q, nrm);
    }
    count
}

fn soften(q: f64, nrm: f64) -> f64 {
    let floor = 1e-300 * nrm;
    if floor == 0.0 || q.abs() >= floor {
        q
    } else if q < 0.0 {
        -floor
    } else {
        floor
    }
}

/// Solve `(diag, sub, super) x = rhs` with partial pivoting.
/// `sub` and `super` each have length `diag.len() - 1`.
fn tridiag_solve(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64], nrm: f64) -> Vec<f64> {
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
        b[i] = pivot_floor(b[i], nrm);
        let factor = a[i] / b[i];
        b[i + 1] -= factor * c[i];
        if i + 1 < n - 1 {
            c[i + 1] -= factor * c2[i];
        }
        r[i + 1] -= factor * r[i];
    }
    b[n - 1] = pivot_floor(b[n - 1], nrm);
    r[n - 1] /= b[n - 1];
    if n > 1 {
        r[n - 2] = (r[n - 2] - c[n - 2] * r[n - 1]) / b[n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        r[i] = (r[i] - c[i] * r[i + 1] - c2[i] * r[i + 2]) / b[i];
    }
    r
}

fn pivot_floor(value: f64, nrm: f64) -> f64 {
    let floor = 1e-30 * nrm;
    if floor != 0.0 && value.abs() < floor {
        if value < 0.0 { -floor } else { floor }
    } else {
        value
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

fn reject_zero_columns(columns: &[f64], n: usize, k: usize, nrm: f64) -> Result<(), SaddleError> {
    let tol = 8.0 * f64::EPSILON * (n.max(1) as f64) * nrm;
    for col in 0..k {
        let mut norm = 0.0;
        for row in 0..n {
            let value = columns[col * n + row];
            norm += value * value;
        }
        if norm.sqrt() <= tol {
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
        let (ea, _, _) = eigh_symmetric(&left).unwrap();
        let (eb, _, _) = eigh_symmetric(&right).unwrap();
        let mut gap = 0.0f64;
        for i in 0..n {
            gap = gap.max((ea[i] - eb[i]).abs());
        }
        gap
    }

    #[test]
    fn reduction_preserves_the_spectrum() {
        let mut sizes = vec![70usize, 140];
        if !cfg!(debug_assertions) {
            sizes.push(256);
        }
        for n in sizes {
            let mut state = 0x70_u64.wrapping_add(n as u64);
            let mut a = vec![0.0; n * n];
            for col in 0..n {
                for row in col..n {
                    state = state
                        .wrapping_mul(0x5851_F42D_4C95_7F2D)
                        .wrapping_add(0x1405_7B7E_F767_814F);
                    let unit = (state >> 11) as u32 as f64 / u32::MAX as f64;
                    let value = unit * 2.0 - 1.0;
                    a[col * n + row] = value;
                    a[row * n + col] = value;
                }
            }
            let mut matrix = a.clone();
            let _panels = dense_to_band(&mut matrix, n);
            assert!(eval_gap(&a, &matrix, n) < 1e-8, "n={n} band reduction");
            let mut chase = Vec::new();
            band_to_tridiagonal(&mut matrix, n, &mut chase);
            assert!(
                eval_gap(&a, &matrix, n) < 1e-8,
                "n={n} tridiagonal reduction"
            );
        }
    }

    #[test]
    fn chase_blocks_match_the_reflector_product() {
        for n in [96usize, 220] {
            let mut state = 0x96_u64.wrapping_add(n as u64);
            let mut a = vec![0.0; n * n];
            for col in 0..n {
                for row in col..n {
                    state = state
                        .wrapping_mul(0x5851_F42D_4C95_7F2D)
                        .wrapping_add(0x1405_7B7E_F767_814F);
                    let unit = (state >> 11) as u32 as f64 / u32::MAX as f64;
                    let value = unit * 2.0 - 1.0;
                    a[col * n + row] = value;
                    a[row * n + col] = value;
                }
            }
            let mut matrix = a;
            let _panels = dense_to_band(&mut matrix, n);
            let mut refs = Vec::new();
            band_to_tridiagonal(&mut matrix, n, &mut refs);
            let chase = group_chase(&refs);
            let mut single = vec![0.0; n];
            let mut blocked = vec![0.0; n];
            for i in 0..n {
                single[i] = (i as f64) * 0.01;
                blocked[i] = single[i];
            }
            for reflector in refs.iter().rev() {
                let mut dot = 0.0;
                for (offset, &value) in reflector.v.iter().enumerate() {
                    dot = value.mul_add(single[reflector.start + offset], dot);
                }
                let scale = reflector.tau * dot;
                for (offset, &value) in reflector.v.iter().enumerate() {
                    let slot = &mut single[reflector.start + offset];
                    *slot = (-scale).mul_add(value, *slot);
                }
            }
            apply_back_transform(&mut blocked, n, 1, &[], &chase);
            let mut gap = 0.0f64;
            for i in 0..n {
                gap = gap.max((single[i] - blocked[i]).abs());
            }
            assert!(gap < 1e-8, "n={n} blocked chase product gap {gap}");
        }
    }
}
