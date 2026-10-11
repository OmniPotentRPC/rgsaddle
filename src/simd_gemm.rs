#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::ptr_offset_with_cast)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(unused_unsafe)]

//! FMA-free tiled matrix product.
//!
//! Each output element accumulates `k` in ascending order as
//! `acc = acc + a * b`, then `alpha * acc + beta * c`. The multiply
//! and the add stay separate, so the bits do not depend on a fused
//! multiply-add. A kernel keeps several rows and several columns in
//! registers. Each of those accumulators repeats the scalar `k` chain.
//! Tile sizes are constants. They do not read the rayon pool.
//!
//! `RGSADDLE_SIMD=scalar|avx2|avx512|neon` selects the kernel for the
//! next [`refresh_simd`] call. An unset variable uses the widest
//! supported kernel. AVX-512, AVX2, NEON, and scalar agree on the bits.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::error::SaddleError;

thread_local! {
    static PACK: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

fn with_pack(len: usize, body: impl FnOnce(&mut [f64])) {
    PACK.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < len {
            buf.resize(len, 0.0);
        }
        body(&mut buf[..len]);
    });
}

const KIND_UNSET: u8 = 0xFF;
const KIND_SCALAR: u8 = 0;
const KIND_AVX2: u8 = 1;
const KIND_AVX512: u8 = 2;
const KIND_NEON: u8 = 3;

static SIMD_KIND: AtomicU8 = AtomicU8::new(KIND_UNSET);

/// Read `RGSADDLE_SIMD` and publish the kernel choice.
///
/// Called at the start of `eigh` and `partial_eigh`, before any tile
/// runs. A release store pairs with the acquire load in the workers.
pub(crate) fn refresh_simd() -> Result<(), SaddleError> {
    let kind = match std::env::var("RGSADDLE_SIMD") {
        Ok(value) => parse_override(value.trim())?,
        Err(_) => detect_kind(),
    };
    SIMD_KIND.store(kind, Ordering::Release);
    Ok(())
}

fn parse_override(value: &str) -> Result<u8, SaddleError> {
    let kind = match value {
        "scalar" => KIND_SCALAR,
        "avx2" => KIND_AVX2,
        "avx512" => KIND_AVX512,
        "neon" => KIND_NEON,
        _ => KIND_SCALAR,
    };
    if !kind_supported(kind) {
        return Err(SaddleError::Solver(
            "RGSADDLE_SIMD asks for a kernel this CPU does not have".into(),
        ));
    }
    Ok(kind)
}

fn host_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn host_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn kind_supported(kind: u8) -> bool {
    match kind {
        KIND_SCALAR => true,
        KIND_AVX2 => host_avx2(),
        KIND_AVX512 => host_avx512(),
        KIND_NEON => cfg!(target_arch = "aarch64"),
        _ => false,
    }
}

fn detect_kind() -> u8 {
    if host_avx512() {
        KIND_AVX512
    } else if host_avx2() {
        KIND_AVX2
    } else if cfg!(target_arch = "aarch64") {
        KIND_NEON
    } else {
        KIND_SCALAR
    }
}

fn current_kind() -> u8 {
    let kind = SIMD_KIND.load(Ordering::Acquire);
    if kind == KIND_UNSET {
        detect_kind()
    } else {
        kind
    }
}

/// Bulge updates use the AVX2 kernel only when that kernel is the
/// selected path or a wider x86 path. The kernel is a separate
/// multiply and add, matching the scalar loop element for element.
#[cfg(target_arch = "x86_64")]
pub(crate) fn bulge_uses_avx2() -> bool {
    matches!(current_kind(), KIND_AVX2 | KIND_AVX512) && host_avx2()
}

/// Eight-wide bulge update. Selected only on the AVX-512 path.
#[cfg(target_arch = "x86_64")]
pub(crate) fn bulge_uses_avx512() -> bool {
    current_kind() == KIND_AVX512 && host_avx512()
}

/// Fixed column-task count. A power of two, at most 32, from `n` only.
pub(crate) fn column_tasks(ncols: usize) -> usize {
    if ncols == 0 {
        return 0;
    }
    let capped = 32usize.min(ncols).max(1);
    1usize << (usize::BITS - 1 - capped.leading_zeros())
}

pub(crate) fn split_range(n: usize, tasks: usize, tile: usize) -> (usize, usize) {
    let base = n / tasks;
    let extra = n % tasks;
    if tile < extra {
        (tile * (base + 1), base + 1)
    } else {
        (extra * (base + 1) + (tile - extra) * base, base)
    }
}

pub(crate) fn for_tiles(n_tasks: usize, f: impl Fn(usize) + Send + Sync) {
    if n_tasks == 0 {
        return;
    }
    #[cfg(feature = "par")]
    {
        // One thread gains nothing from the pool, and a single task has
        // no sibling to join. The column ranges stay the fixed partition
        // of `n`, so either path writes the same bits.
        if n_tasks == 1 || rayon::current_num_threads() == 1 {
            for task in 0..n_tasks {
                f(task);
            }
            return;
        }
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
pub(crate) struct SyncPtr(*const f64);
// SAFETY: tiles only read through this pointer, and each tile stays
// inside the allocation the caller passed. Sharing the wrapper does
// not create a mutable alias.
unsafe impl Send for SyncPtr {}
// SAFETY: the same read-only contract as `Send`.
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    pub(crate) fn new(ptr: *const f64) -> Self {
        Self(ptr)
    }

    pub(crate) fn get(self) -> *const f64 {
        self.0
    }

    pub(crate) fn add(self, count: usize) -> *const f64 {
        // SAFETY: the caller keeps `count` inside the allocation the
        // pointer was derived from.
        unsafe { self.0.add(count) }
    }
}

/// Pointer shared by tiles that write disjoint entries.
#[derive(Clone, Copy)]
pub(crate) struct SyncMut(*mut f64);
// SAFETY: each tile writes a disjoint index range of the allocation.
// The wrapper does not dereference on its own.
unsafe impl Send for SyncMut {}
// SAFETY: the same disjoint-write contract as `Send`.
unsafe impl Sync for SyncMut {}

impl SyncMut {
    pub(crate) fn new(ptr: *mut f64) -> Self {
        Self(ptr)
    }

    pub(crate) fn add(self, count: usize) -> *mut f64 {
        // SAFETY: the caller keeps `count` inside the allocation the
        // pointer was derived from.
        unsafe { self.0.add(count) }
    }
}

/// `C = alpha * A * B + beta * C` for column-major matrices with row
/// stride 1. Column tiles are a fixed partition of `n`.
///
/// # Safety
///
/// `a` holds an `m` by `k` matrix with column stride `lda`. `b` holds
/// a `k` by `n` matrix with column stride `ldb`. `c` holds an `m` by
/// `n` matrix with column stride `ldc`. The leading dimensions are at
/// least the row counts.
pub(crate) unsafe fn gemm_nn(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k > 0 && alpha != 0.0 && blocked_parallel(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) {
        return;
    }
    let tasks = column_tasks(n).max(1);
    let a_ptr = SyncPtr::new(a);
    let b_ptr = SyncPtr::new(b);
    let c_ptr = SyncMut::new(c);
    // SAFETY: task `tile` writes columns `[col0, col0 + count)` of C
    // and those ranges partition `0..n`, so the writes are disjoint.
    // A and B are immutable for the duration of the product.
    for_tiles(tasks, move |tile| {
        let (col0, count) = split_range(n, tasks, tile);
        if count == 0 {
            return;
        }
        tile_gemm(
            m,
            count,
            k,
            alpha,
            a_ptr.get(),
            1,
            lda as isize,
            b_ptr.add(col0 * ldb),
            1,
            ldb as isize,
            beta,
            c_ptr.add(col0 * ldc),
            1,
            ldc as isize,
        );
    });
}

/// `C = alpha * A * B + beta * C` on one tile.
///
/// Row and column strides are in elements. The kernel is sequential.
/// Callers that split columns own disjoint writes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tile_gemm(
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
        scale_or_zero(m, n, beta, c, c_row_stride, c_col_stride);
        return;
    }
    let vector = a_row_stride == 1 && c_row_stride == 1;
    if vector {
        match current_kind() {
            KIND_AVX512 if m >= 8 && cfg!(target_arch = "x86_64") => {
                // SAFETY: `refresh_simd` rejects avx512 when the CPU
                // lacks it, and the auto path checks `avx512f` first.
                // Rows are contiguous, so each 8-wide load stays inside
                // the `m` rows of its column.
                unsafe {
                    gemm_avx512(
                        m,
                        n,
                        k,
                        alpha,
                        a,
                        a_col_stride,
                        b,
                        b_row_stride,
                        b_col_stride,
                        beta,
                        c,
                        c_col_stride,
                    )
                };
                return;
            }
            KIND_AVX2 if m >= 4 && cfg!(target_arch = "x86_64") => {
                // SAFETY: same contract as the AVX-512 kernel, with
                // 4-wide AVX2 loads.
                unsafe {
                    gemm_avx2(
                        m,
                        n,
                        k,
                        alpha,
                        a,
                        a_col_stride,
                        b,
                        b_row_stride,
                        b_col_stride,
                        beta,
                        c,
                        c_col_stride,
                    )
                };
                return;
            }
            KIND_NEON if m >= 2 && cfg!(target_arch = "aarch64") => {
                // SAFETY: NEON is baseline on aarch64. Loads are 2-wide
                // and stay inside the column.
                unsafe {
                    gemm_neon(
                        m,
                        n,
                        k,
                        alpha,
                        a,
                        a_col_stride,
                        b,
                        b_row_stride,
                        b_col_stride,
                        beta,
                        c,
                        c_col_stride,
                    )
                };
                return;
            }
            _ => {}
        }
    }
    gemm_scalar(
        m,
        n,
        k,
        alpha,
        a,
        a_row_stride,
        a_col_stride,
        b,
        b_row_stride,
        b_col_stride,
        beta,
        c,
        c_row_stride,
        c_col_stride,
    );
}

fn scale_or_zero(
    m: usize,
    n: usize,
    beta: f64,
    c: *mut f64,
    c_row_stride: isize,
    c_col_stride: isize,
) {
    if beta == 0.0 {
        for col in 0..n {
            for row in 0..m {
                // SAFETY: `(row, col)` is inside the `m` by `n` tile.
                unsafe {
                    *c.offset(c_row_stride * row as isize + c_col_stride * col as isize) = 0.0;
                }
            }
        }
    } else if beta != 1.0 {
        for col in 0..n {
            for row in 0..m {
                // SAFETY: `(row, col)` is inside the `m` by `n` tile.
                unsafe {
                    let slot = c.offset(c_row_stride * row as isize + c_col_stride * col as isize);
                    *slot *= beta;
                }
            }
        }
    }
}

fn gemm_scalar(
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
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0;
            for p in 0..k {
                // SAFETY: `p` runs over the shared dimension and
                // `(row, col)` stays inside A, B, and C.
                let term = unsafe {
                    let av = *a.offset(a_row_stride * row as isize + a_col_stride * p as isize);
                    let bv = *b.offset(b_row_stride * p as isize + b_col_stride * col as isize);
                    av * bv
                };
                acc += term;
            }
            // SAFETY: `(row, col)` is inside C.
            unsafe {
                let slot = c.offset(c_row_stride * row as isize + c_col_stride * col as isize);
                *slot = if beta == 0.0 {
                    alpha * acc
                } else {
                    alpha * acc + beta * *slot
                };
            }
        }
    }
}

/// Pack `rows` contiguous rows of a column-major matrix so the kernel
/// reads them sequentially. The copy preserves every bit.
fn pack_rows(
    a: *const f64,
    a_col_stride: isize,
    row0: usize,
    rows: usize,
    k: usize,
    packed: &mut [f64],
) {
    for p in 0..k {
        for r in 0..rows {
            // SAFETY: `row0 + r < m` was checked by the caller, and `p < k`.
            packed[p * rows + r] =
                unsafe { *a.offset((row0 + r) as isize + a_col_stride * p as isize) };
        }
    }
}

include!("tile_kernel.rs");

#[cfg(not(target_arch = "x86_64"))]
unsafe fn gemm_avx512(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_col_stride: isize,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    gemm_scalar(
        m,
        n,
        k,
        alpha,
        a,
        1,
        a_col_stride,
        b,
        b_row_stride,
        b_col_stride,
        beta,
        c,
        1,
        c_col_stride,
    );
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn gemm_avx2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_col_stride: isize,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    gemm_scalar(
        m,
        n,
        k,
        alpha,
        a,
        1,
        a_col_stride,
        b,
        b_row_stride,
        b_col_stride,
        beta,
        c,
        1,
        c_col_stride,
    );
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn gemm_neon(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    a_col_stride: isize,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    gemm_scalar(
        m,
        n,
        k,
        alpha,
        a,
        1,
        a_col_stride,
        b,
        b_row_stride,
        b_col_stride,
        beta,
        c,
        1,
        c_col_stride,
    );
}

/// Rank-1 `Q[:, cols] -= tau * v * (v · Q[:, cols])` on rows `[row, row + v.len())`.
///
/// Each column accumulates the dot in ascending index order, then
/// scales. Vector lanes are columns, so each lane matches the scalar
/// column. Column groups are a fixed partition of `ncols`.
pub(crate) fn apply_rank1(q: &mut [f64], n: usize, ncols: usize, row: usize, v: &[f64], tau: f64) {
    let len = v.len();
    if len == 0 || ncols == 0 || tau == 0.0 {
        return;
    }
    let tasks = column_tasks(ncols).max(1);
    let q_ptr = SyncMut::new(q.as_mut_ptr());
    let v_ptr = SyncPtr::new(v.as_ptr());
    // SAFETY: each task owns a disjoint column range of Q. `v` is
    // immutable. Rows outside `[row, row + len)` are not written.
    for_tiles(tasks, move |tile| {
        let (col0, count) = split_range(ncols, tasks, tile);
        let base = q_ptr.add(col0 * n + row);
        rank1_columns(base, n, count, v_ptr.get(), len, tau);
    });
}

pub(crate) fn rank1_columns(
    q: *mut f64,
    ld: usize,
    ncols: usize,
    v: *const f64,
    len: usize,
    tau: f64,
) {
    match current_kind() {
        KIND_AVX512 if cfg!(target_arch = "x86_64") => {
            // SAFETY: avx512 was selected only when the CPU has it.
            unsafe { rank1_avx512(q, ld, ncols, v, len, tau) };
        }
        KIND_AVX2 if cfg!(target_arch = "x86_64") => {
            // SAFETY: avx2 was selected only when the CPU has it.
            unsafe { rank1_avx2(q, ld, ncols, v, len, tau) };
        }
        KIND_NEON if cfg!(target_arch = "aarch64") => {
            // SAFETY: neon is baseline on aarch64.
            unsafe { rank1_neon(q, ld, ncols, v, len, tau) };
        }
        _ => rank1_scalar(q, ld, ncols, v, len, tau),
    }
}

fn rank1_scalar(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    for col in 0..ncols {
        let mut dot = 0.0;
        for i in 0..len {
            // SAFETY: `i < len` and `col < ncols` stay inside the
            // column tile the caller passed.
            let term = unsafe { *v.add(i) * *q.add(col * ld + i) };
            dot += term;
        }
        let scale = tau * dot;
        for i in 0..len {
            // SAFETY: same tile as the dot.
            unsafe {
                let slot = q.add(col * ld + i);
                *slot -= scale * *v.add(i);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rank1_avx512(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    use std::arch::x86_64::{
        _mm512_add_pd, _mm512_i64gather_pd, _mm512_mul_pd, _mm512_set_epi64, _mm512_set1_pd,
        _mm512_setzero_pd, _mm512_storeu_pd,
    };
    unsafe {
        // Lane `t` reads column `t` at this row. The index is in elements
        // and the scale is the size of `f64`.
        let idx = _mm512_set_epi64(
            (7 * ld) as i64,
            (6 * ld) as i64,
            (5 * ld) as i64,
            (4 * ld) as i64,
            (3 * ld) as i64,
            (2 * ld) as i64,
            ld as i64,
            0,
        );
        let mut col = 0;
        while col + 16 <= ncols {
            let mut dot0 = _mm512_setzero_pd();
            let mut dot1 = _mm512_setzero_pd();
            let base0 = q.add(col * ld);
            let base1 = q.add((col + 8) * ld);
            for i in 0..len {
                let vv = _mm512_set1_pd(*v.add(i));
                let y0 = _mm512_i64gather_pd(idx, base0.add(i), 8);
                let y1 = _mm512_i64gather_pd(idx, base1.add(i), 8);
                dot0 = _mm512_add_pd(dot0, _mm512_mul_pd(vv, y0));
                dot1 = _mm512_add_pd(dot1, _mm512_mul_pd(vv, y1));
            }
            let mut dots = [0.0; 16];
            _mm512_storeu_pd(dots.as_mut_ptr(), dot0);
            _mm512_storeu_pd(dots.as_mut_ptr().add(8), dot1);
            for lane in 0..16 {
                let scale = tau * dots[lane];
                axpy_sub_avx512(q.add((col + lane) * ld), v, len, scale);
            }
            col += 16;
        }
        while col + 8 <= ncols {
            let mut dot = _mm512_setzero_pd();
            let base = q.add(col * ld);
            for i in 0..len {
                let vv = _mm512_set1_pd(*v.add(i));
                let yv = _mm512_i64gather_pd(idx, base.add(i), 8);
                dot = _mm512_add_pd(dot, _mm512_mul_pd(vv, yv));
            }
            let mut dots = [0.0; 8];
            _mm512_storeu_pd(dots.as_mut_ptr(), dot);
            for lane in 0..8 {
                let scale = tau * dots[lane];
                axpy_sub_avx512(q.add((col + lane) * ld), v, len, scale);
            }
            col += 8;
        }
        if col < ncols {
            rank1_scalar(q.add(col * ld), ld, ncols - col, v, len, tau);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rank1_avx2(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_i64gather_pd, _mm256_mul_pd, _mm256_set_epi64x, _mm256_set1_pd,
        _mm256_setzero_pd, _mm256_storeu_pd,
    };
    unsafe {
        let idx = _mm256_set_epi64x((3 * ld) as i64, (2 * ld) as i64, ld as i64, 0);
        let mut col = 0;
        while col + 4 <= ncols {
            let mut dot = _mm256_setzero_pd();
            let base = q.add(col * ld);
            for i in 0..len {
                let vv = _mm256_set1_pd(*v.add(i));
                let yv = _mm256_i64gather_pd(base.add(i), idx, 8);
                dot = _mm256_add_pd(dot, _mm256_mul_pd(vv, yv));
            }
            let mut dots = [0.0; 4];
            _mm256_storeu_pd(dots.as_mut_ptr(), dot);
            for lane in 0..4 {
                let scale = tau * dots[lane];
                axpy_sub_avx2(q.add((col + lane) * ld), v, len, scale);
            }
            col += 4;
        }
        if col < ncols {
            rank1_scalar(q.add(col * ld), ld, ncols - col, v, len, tau);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rank1_neon(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};
    unsafe {
        let mut col = 0;
        while col + 2 <= ncols {
            let mut dot = vdupq_n_f64(0.0);
            for i in 0..len {
                let vv = vdupq_n_f64(*v.add(i));
                let mut y = [0.0; 2];
                y[0] = *q.add(col * ld + i);
                y[1] = *q.add((col + 1) * ld + i);
                let yv = vld1q_f64(y.as_ptr());
                dot = vaddq_f64(dot, vmulq_f64(vv, yv));
            }
            let mut dots = [0.0; 2];
            vst1q_f64(dots.as_mut_ptr(), dot);
            for lane in 0..2 {
                let scale = tau * dots[lane];
                axpy_sub_neon(q.add((col + lane) * ld), v, len, scale);
            }
            col += 2;
        }
        if col < ncols {
            rank1_scalar(q.add(col * ld), ld, ncols - col, v, len, tau);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn rank1_avx512(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    rank1_scalar(q, ld, ncols, v, len, tau);
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn rank1_avx2(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    rank1_scalar(q, ld, ncols, v, len, tau);
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn rank1_neon(q: *mut f64, ld: usize, ncols: usize, v: *const f64, len: usize, tau: f64) {
    rank1_scalar(q, ld, ncols, v, len, tau);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(0x5851_F42D_4C95_7F2D)
                .wrapping_add(0x1405_7B7E_F767_814F);
            let bits = (state >> 11) as u32;
            f64::from(bits) / f64::from(u32::MAX) * 2.0 - 1.0
        };
        let a: Vec<f64> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| next()).collect();
        let c: Vec<f64> = (0..m * n).map(|_| next()).collect();
        (a, b, c)
    }

    #[test]
    fn vector_kernels_match_the_scalar_product() {
        refresh_simd().unwrap();
        let shapes = [
            (13usize, 7usize, 11usize),
            (16, 8, 5),
            (17, 9, 3),
            (40, 24, 13),
            (64, 64, 64),
        ];
        for (case, (m, n, k)) in shapes.into_iter().enumerate() {
            let (a, b, c0) = product(m, n, k, 0x6E77_0001 + case as u64);
            for &(alpha, beta) in &[(0.5, -1.25), (1.0, 0.0), (-1.0, 1.0)] {
                let mut scalar = c0.clone();
                tile_gemm_kind(KIND_SCALAR, m, n, k, alpha, &a, &b, beta, &mut scalar);
                for kind in supported_kinds() {
                    let mut out = c0.clone();
                    tile_gemm_kind(kind, m, n, k, alpha, &a, &b, beta, &mut out);
                    for (got, want) in out.iter().zip(&scalar) {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "kind {kind} m={m} n={n} k={k} alpha={alpha} beta={beta}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ordered_div_matches_the_scalar_sum() {
        refresh_simd().unwrap();
        let delta = [1.0, 2.5, -0.0, 4.0, 8.0, 9.0, 10.0, 11.0, f64::from_bits(1)];
        let numer = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9];
        let got = sum_div_ordered(0.25, &delta, &numer, 1.5, 0.125).unwrap();
        let mut want = 0.25;
        for i in 0..delta.len() {
            want += numer[i] / ((delta[i] - 1.5) - 0.125);
        }
        assert_eq!(got.to_bits(), want.to_bits());
        let (got_sum, got_deriv) = sum_div_deriv_ordered(0.25, &delta, &numer, 1.5, 0.125).unwrap();
        let mut deriv = 0.0;
        let mut sum = 0.25;
        for i in 0..delta.len() {
            let diff = (delta[i] - 1.5) - 0.125;
            let term = numer[i] / diff;
            sum += term;
            deriv += term / diff;
        }
        assert_eq!(got_sum.to_bits(), sum.to_bits());
        assert_eq!(got_deriv.to_bits(), deriv.to_bits());
    }

    fn supported_kinds() -> Vec<u8> {
        let mut kinds = vec![KIND_SCALAR];
        if host_avx2() {
            kinds.push(KIND_AVX2);
        }
        if host_avx512() {
            kinds.push(KIND_AVX512);
        }
        if cfg!(target_arch = "aarch64") {
            kinds.push(KIND_NEON);
        }
        kinds
    }

    fn tile_gemm_kind(
        kind: u8,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        c: &mut [f64],
    ) {
        SIMD_KIND.store(kind, Ordering::Release);
        tile_gemm(
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            1,
            k as isize,
            beta,
            c.as_mut_ptr(),
            1,
            m as isize,
        );
    }
}
