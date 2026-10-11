#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::ptr_offset_with_cast)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(unused_unsafe)]

//! Register-tiled kernels.
//!
//! Each accumulator walks `k` from `0` upward as `acc = fma(a, b, acc)`.
//! Columns in one kernel are independent, so a wider register tile matches
//! the scalar `f64::mul_add` chain. Each B column is its own pointer,
//! advanced by `b_row_stride`. The column updates are the `fma_col` macros
//! so every unrolled kernel uses the same association.

#[cfg(target_arch = "aarch64")]
use super::KIND_NEON;
#[cfg(target_arch = "x86_64")]
use super::{KIND_AVX2, KIND_AVX512};
use super::{
    SyncMut, SyncPtr, column_tasks, current_kind, for_tiles, gemm_scalar, pack_rows, split_range,
    tile_gemm, with_pack,
};

/// One column of a two-row panel: broadcast B, advance that column pointer,
/// then `acc = fma(g, b, acc)` on each row group, in that order.
#[cfg(target_arch = "x86_64")]
macro_rules! fma_col_pair {
    ($broadcast:path, $fma:path, $g0:ident, $g1:ident, $stride:expr, $($bp:ident, $lo:ident, $hi:ident),+ $(,)?) => {
        $(
            {
                let bv = $broadcast(*$bp);
                $bp = $bp.offset($stride);
                $lo = $fma($g0, bv, $lo);
                $hi = $fma($g1, bv, $hi);
            }
        )+
    };
}

/// One column of a single-row panel. Same association as [`fma_col_pair`].
#[cfg(target_arch = "x86_64")]
macro_rules! fma_col {
    ($broadcast:path, $fma:path, $g0:ident, $stride:expr, $($bp:ident, $acc:ident),+ $(,)?) => {
        $(
            {
                let bv = $broadcast(*$bp);
                $bp = $bp.offset($stride);
                $acc = $fma($g0, bv, $acc);
            }
        )+
    };
}

/// NEON `vfmaq_f64(acc, a, b)` is `acc + a * b`, the same product as
/// `fma(a, b, acc)` with the accumulator written first.
#[cfg(target_arch = "aarch64")]
macro_rules! fma_col_pair_neon {
    ($g0:ident, $g1:ident, $stride:expr, $($bp:ident, $lo:ident, $hi:ident),+ $(,)?) => {
        $(
            {
                let bv = vdupq_n_f64(*$bp);
                $bp = $bp.offset($stride);
                $lo = vfmaq_f64($lo, $g0, bv);
                $hi = vfmaq_f64($hi, $g1, bv);
            }
        )+
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! fma_col_neon {
    ($g0:ident, $stride:expr, $($bp:ident, $acc:ident),+ $(,)?) => {
        $(
            {
                let bv = vdupq_n_f64(*$bp);
                $bp = $bp.offset($stride);
                $acc = vfmaq_f64($acc, $g0, bv);
            }
        )+
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn store512(
    slot: *mut f64,
    acc: std::arch::x86_64::__m512d,
    alpha: std::arch::x86_64::__m512d,
    beta_v: std::arch::x86_64::__m512d,
    beta: f64,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_storeu_pd};
    unsafe {
        let scaled = _mm512_mul_pd(alpha, acc);
        if beta == 0.0 {
            _mm512_storeu_pd(slot, scaled);
        } else {
            let old = _mm512_loadu_pd(slot);
            _mm512_storeu_pd(slot, _mm512_fmadd_pd(beta_v, old, scaled));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn store256(
    slot: *mut f64,
    acc: std::arch::x86_64::__m256d,
    alpha: std::arch::x86_64::__m256d,
    beta_v: std::arch::x86_64::__m256d,
    beta: f64,
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd};
    unsafe {
        let scaled = _mm256_mul_pd(alpha, acc);
        if beta == 0.0 {
            _mm256_storeu_pd(slot, scaled);
        } else {
            let old = _mm256_loadu_pd(slot);
            _mm256_storeu_pd(slot, _mm256_fmadd_pd(beta_v, old, scaled));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn store_neon(
    slot: *mut f64,
    acc: std::arch::aarch64::float64x2_t,
    alpha: std::arch::aarch64::float64x2_t,
    beta_v: std::arch::aarch64::float64x2_t,
    beta: f64,
) {
    use std::arch::aarch64::{vfmaq_f64, vld1q_f64, vmulq_f64, vst1q_f64};
    unsafe {
        let scaled = vmulq_f64(alpha, acc);
        if beta == 0.0 {
            vst1q_f64(slot, scaled);
        } else {
            let old = vld1q_f64(slot);
            vst1q_f64(slot, vfmaq_f64(scaled, beta_v, old));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn kernel512_16x8(
    k: usize,
    alpha_v: std::arch::x86_64::__m512d,
    beta_v: std::arch::x86_64::__m512d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm512_setzero_pd();
        let mut a_0_1 = _mm512_setzero_pd();
        let mut a_1_0 = _mm512_setzero_pd();
        let mut a_1_1 = _mm512_setzero_pd();
        let mut a_2_0 = _mm512_setzero_pd();
        let mut a_2_1 = _mm512_setzero_pd();
        let mut a_3_0 = _mm512_setzero_pd();
        let mut a_3_1 = _mm512_setzero_pd();
        let mut a_4_0 = _mm512_setzero_pd();
        let mut a_4_1 = _mm512_setzero_pd();
        let mut a_5_0 = _mm512_setzero_pd();
        let mut a_5_1 = _mm512_setzero_pd();
        let mut a_6_0 = _mm512_setzero_pd();
        let mut a_6_1 = _mm512_setzero_pd();
        let mut a_7_0 = _mm512_setzero_pd();
        let mut a_7_1 = _mm512_setzero_pd();
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut bp4 = b.offset(b_col_stride * 4);
        let mut bp5 = b.offset(b_col_stride * 5);
        let mut bp6 = b.offset(b_col_stride * 6);
        let mut bp7 = b.offset(b_col_stride * 7);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = _mm512_loadu_pd(ap);
            let g1 = _mm512_loadu_pd(ap.add(8));
            ap = ap.add(16);
            fma_col_pair!(
                _mm512_set1_pd,
                _mm512_fmadd_pd,
                g0,
                g1,
                b_row_stride,
                bp0,
                a_0_0,
                a_0_1,
                bp1,
                a_1_0,
                a_1_1,
                bp2,
                a_2_0,
                a_2_1,
                bp3,
                a_3_0,
                a_3_1,
                bp4,
                a_4_0,
                a_4_1,
                bp5,
                a_5_0,
                a_5_1,
                bp6,
                a_6_0,
                a_6_1,
                bp7,
                a_7_0,
                a_7_1,
            );
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store512(slot_0, a_0_0, alpha_v, beta_v, beta);
        store512(slot_0.add(8), a_0_1, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store512(slot_1, a_1_0, alpha_v, beta_v, beta);
        store512(slot_1.add(8), a_1_1, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store512(slot_2, a_2_0, alpha_v, beta_v, beta);
        store512(slot_2.add(8), a_2_1, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store512(slot_3, a_3_0, alpha_v, beta_v, beta);
        store512(slot_3.add(8), a_3_1, alpha_v, beta_v, beta);
        let slot_4 = c.offset(c_col_stride * 4);
        store512(slot_4, a_4_0, alpha_v, beta_v, beta);
        store512(slot_4.add(8), a_4_1, alpha_v, beta_v, beta);
        let slot_5 = c.offset(c_col_stride * 5);
        store512(slot_5, a_5_0, alpha_v, beta_v, beta);
        store512(slot_5.add(8), a_5_1, alpha_v, beta_v, beta);
        let slot_6 = c.offset(c_col_stride * 6);
        store512(slot_6, a_6_0, alpha_v, beta_v, beta);
        store512(slot_6.add(8), a_6_1, alpha_v, beta_v, beta);
        let slot_7 = c.offset(c_col_stride * 7);
        store512(slot_7, a_7_0, alpha_v, beta_v, beta);
        store512(slot_7.add(8), a_7_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn kernel512_16x4(
    k: usize,
    alpha_v: std::arch::x86_64::__m512d,
    beta_v: std::arch::x86_64::__m512d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm512_setzero_pd();
        let mut a_0_1 = _mm512_setzero_pd();
        let mut a_1_0 = _mm512_setzero_pd();
        let mut a_1_1 = _mm512_setzero_pd();
        let mut a_2_0 = _mm512_setzero_pd();
        let mut a_2_1 = _mm512_setzero_pd();
        let mut a_3_0 = _mm512_setzero_pd();
        let mut a_3_1 = _mm512_setzero_pd();
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = _mm512_loadu_pd(ap);
            let g1 = _mm512_loadu_pd(ap.add(8));
            ap = ap.add(16);
            fma_col_pair!(
                _mm512_set1_pd,
                _mm512_fmadd_pd,
                g0,
                g1,
                b_row_stride,
                bp0,
                a_0_0,
                a_0_1,
                bp1,
                a_1_0,
                a_1_1,
                bp2,
                a_2_0,
                a_2_1,
                bp3,
                a_3_0,
                a_3_1,
            );
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store512(slot_0, a_0_0, alpha_v, beta_v, beta);
        store512(slot_0.add(8), a_0_1, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store512(slot_1, a_1_0, alpha_v, beta_v, beta);
        store512(slot_1.add(8), a_1_1, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store512(slot_2, a_2_0, alpha_v, beta_v, beta);
        store512(slot_2.add(8), a_2_1, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store512(slot_3, a_3_0, alpha_v, beta_v, beta);
        store512(slot_3.add(8), a_3_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn kernel512_16x1(
    k: usize,
    alpha_v: std::arch::x86_64::__m512d,
    beta_v: std::arch::x86_64::__m512d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm512_setzero_pd();
        let mut a_0_1 = _mm512_setzero_pd();
        for p in 0..k {
            let ap = packed.add(p * 16);
            let g0 = _mm512_loadu_pd(ap.add(0));
            let g1 = _mm512_loadu_pd(ap.add(8));
            let bv_0 = _mm512_set1_pd(*b.offset(b_row_stride * p as isize + b_col_stride * 0));
            a_0_0 = _mm512_fmadd_pd(g0, bv_0, a_0_0);
            a_0_1 = _mm512_fmadd_pd(g1, bv_0, a_0_1);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store512(slot_0, a_0_0, alpha_v, beta_v, beta);
        store512(slot_0.add(8), a_0_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn kernel512_8x8(
    k: usize,
    alpha_v: std::arch::x86_64::__m512d,
    beta_v: std::arch::x86_64::__m512d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm512_setzero_pd();
        let mut a_1_0 = _mm512_setzero_pd();
        let mut a_2_0 = _mm512_setzero_pd();
        let mut a_3_0 = _mm512_setzero_pd();
        let mut a_4_0 = _mm512_setzero_pd();
        let mut a_5_0 = _mm512_setzero_pd();
        let mut a_6_0 = _mm512_setzero_pd();
        let mut a_7_0 = _mm512_setzero_pd();
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut bp4 = b.offset(b_col_stride * 4);
        let mut bp5 = b.offset(b_col_stride * 5);
        let mut bp6 = b.offset(b_col_stride * 6);
        let mut bp7 = b.offset(b_col_stride * 7);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = _mm512_loadu_pd(ap);
            ap = ap.add(8);
            fma_col!(
                _mm512_set1_pd,
                _mm512_fmadd_pd,
                g0,
                b_row_stride,
                bp0,
                a_0_0,
                bp1,
                a_1_0,
                bp2,
                a_2_0,
                bp3,
                a_3_0,
                bp4,
                a_4_0,
                bp5,
                a_5_0,
                bp6,
                a_6_0,
                bp7,
                a_7_0,
            );
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store512(slot_0, a_0_0, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store512(slot_1, a_1_0, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store512(slot_2, a_2_0, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store512(slot_3, a_3_0, alpha_v, beta_v, beta);
        let slot_4 = c.offset(c_col_stride * 4);
        store512(slot_4, a_4_0, alpha_v, beta_v, beta);
        let slot_5 = c.offset(c_col_stride * 5);
        store512(slot_5, a_5_0, alpha_v, beta_v, beta);
        let slot_6 = c.offset(c_col_stride * 6);
        store512(slot_6, a_6_0, alpha_v, beta_v, beta);
        let slot_7 = c.offset(c_col_stride * 7);
        store512(slot_7, a_7_0, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn kernel512_8x1(
    k: usize,
    alpha_v: std::arch::x86_64::__m512d,
    beta_v: std::arch::x86_64::__m512d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm512_setzero_pd();
        for p in 0..k {
            let ap = packed.add(p * 8);
            let g0 = _mm512_loadu_pd(ap.add(0));
            let bv_0 = _mm512_set1_pd(*b.offset(b_row_stride * p as isize + b_col_stride * 0));
            a_0_0 = _mm512_fmadd_pd(g0, bv_0, a_0_0);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store512(slot_0, a_0_0, alpha_v, beta_v, beta);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply512(
    rows: usize,
    n: usize,
    k: usize,
    alpha: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::_mm512_set1_pd;
    unsafe {
        let alpha_v = _mm512_set1_pd(alpha);
        let beta_v = _mm512_set1_pd(beta);
        let mut col = 0;
        if rows == 16 {
            while col + 8 <= n {
                kernel512_16x8(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 8;
            }
            while col + 4 <= n {
                kernel512_16x4(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 4;
            }
            while col < n {
                kernel512_16x1(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 1;
            }
        } else {
            while col + 8 <= n {
                kernel512_8x8(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 8;
            }
            while col < n {
                kernel512_8x1(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn call_apply512(
    rows: usize,
    n: usize,
    k: usize,
    alpha: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    // SAFETY: callers are the AVX-512 product paths, which run only
    // after `refresh_simd` has checked the CPU.
    unsafe {
        apply512(
            rows,
            n,
            k,
            alpha,
            packed,
            b,
            b_row_stride,
            b_col_stride,
            beta,
            c,
            c_col_stride,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn gemm_avx512(
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
    unsafe {
        let mut row0 = 0;
        while row0 + 16 <= m {
            with_pack(16 * k, |packed| {
                pack_rows(a, a_col_stride, row0, 16, k, packed);
                // SAFETY: this function is the AVX-512 entry point. The
                // pack lives until `apply512` returns.
                unsafe {
                    call_apply512(
                        16,
                        n,
                        k,
                        alpha,
                        packed.as_ptr(),
                        b,
                        b_row_stride,
                        b_col_stride,
                        beta,
                        c.offset(row0 as isize),
                        c_col_stride,
                    );
                }
            });
            row0 += 16;
        }
        while row0 + 8 <= m {
            with_pack(8 * k, |packed| {
                pack_rows(a, a_col_stride, row0, 8, k, packed);
                // SAFETY: same AVX-512 entry contract as the 16-row panel.
                unsafe {
                    call_apply512(
                        8,
                        n,
                        k,
                        alpha,
                        packed.as_ptr(),
                        b,
                        b_row_stride,
                        b_col_stride,
                        beta,
                        c.offset(row0 as isize),
                        c_col_stride,
                    );
                }
            });
            row0 += 8;
        }
        if row0 < m {
            gemm_scalar(
                m - row0,
                n,
                k,
                alpha,
                a.offset(row0 as isize),
                1,
                a_col_stride,
                b,
                b_row_stride,
                b_col_stride,
                beta,
                c.offset(row0 as isize),
                1,
                c_col_stride,
            );
        }
    }
}

struct PackedPanel {
    row0: usize,
    rows: usize,
    off: usize,
}

/// Pack every row panel before the column tasks run.
///
/// One shared buffer, then one parallel region. Joining once per panel
/// spent more time in the pool than in the `k = 64` back-transform.
fn pack_row_panels(
    packed: &mut Vec<f64>,
    a: *const f64,
    lda: isize,
    m: usize,
    k: usize,
    widths: &[usize],
) -> (Vec<PackedPanel>, usize) {
    let mut panels = Vec::new();
    let mut off = 0usize;
    let mut row0 = 0usize;
    for &width in widths {
        while row0 + width <= m {
            let need = off + width.saturating_mul(k);
            if packed.len() < need {
                packed.resize(need, 0.0);
            }
            pack_rows(a, lda, row0, width, k, &mut packed[off..need]);
            panels.push(PackedPanel {
                row0,
                rows: width,
                off,
            });
            off = need;
            row0 += width;
        }
    }
    (panels, row0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemm_parallel_avx512(
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
    unsafe {
        let tasks = column_tasks(n).max(1);
        let mut packed = Vec::new();
        let (panels, tail) = pack_row_panels(&mut packed, a, lda as isize, m, k, &[16, 8]);
        let packed_ptr = SyncPtr::new(packed.as_ptr());
        let b_ptr = SyncPtr::new(b);
        let c_ptr = SyncMut::new(c);
        let a_ptr = SyncPtr::new(a);
        // Column tasks only read the packed panels, and each task writes
        // its own columns of C.
        for_tiles(tasks, move |tile| {
            let (col0, count) = split_range(n, tasks, tile);
            if count == 0 {
                return;
            }
            for panel in &panels {
                call_apply512(
                    panel.rows,
                    count,
                    k,
                    alpha,
                    packed_ptr.add(panel.off),
                    b_ptr.add(col0 * ldb),
                    1,
                    ldb as isize,
                    beta,
                    c_ptr.add(panel.row0 + col0 * ldc),
                    ldc as isize,
                );
            }
            if tail < m {
                tile_gemm(
                    m - tail,
                    count,
                    k,
                    alpha,
                    a_ptr.add(tail),
                    1,
                    lda as isize,
                    b_ptr.add(col0 * ldb),
                    1,
                    ldb as isize,
                    beta,
                    c_ptr.add(tail + col0 * ldc),
                    1,
                    ldc as isize,
                );
            }
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel256_8x4(
    k: usize,
    alpha_v: std::arch::x86_64::__m256d,
    beta_v: std::arch::x86_64::__m256d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm256_setzero_pd();
        let mut a_0_1 = _mm256_setzero_pd();
        let mut a_1_0 = _mm256_setzero_pd();
        let mut a_1_1 = _mm256_setzero_pd();
        let mut a_2_0 = _mm256_setzero_pd();
        let mut a_2_1 = _mm256_setzero_pd();
        let mut a_3_0 = _mm256_setzero_pd();
        let mut a_3_1 = _mm256_setzero_pd();
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = _mm256_loadu_pd(ap);
            let g1 = _mm256_loadu_pd(ap.add(4));
            ap = ap.add(8);
            let bv_0 = _mm256_set1_pd(*bp0);
            bp0 = bp0.offset(b_row_stride);
            a_0_0 = _mm256_fmadd_pd(g0, bv_0, a_0_0);
            a_0_1 = _mm256_fmadd_pd(g1, bv_0, a_0_1);
            let bv_1 = _mm256_set1_pd(*bp1);
            bp1 = bp1.offset(b_row_stride);
            a_1_0 = _mm256_fmadd_pd(g0, bv_1, a_1_0);
            a_1_1 = _mm256_fmadd_pd(g1, bv_1, a_1_1);
            let bv_2 = _mm256_set1_pd(*bp2);
            bp2 = bp2.offset(b_row_stride);
            a_2_0 = _mm256_fmadd_pd(g0, bv_2, a_2_0);
            a_2_1 = _mm256_fmadd_pd(g1, bv_2, a_2_1);
            let bv_3 = _mm256_set1_pd(*bp3);
            bp3 = bp3.offset(b_row_stride);
            a_3_0 = _mm256_fmadd_pd(g0, bv_3, a_3_0);
            a_3_1 = _mm256_fmadd_pd(g1, bv_3, a_3_1);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store256(slot_0, a_0_0, alpha_v, beta_v, beta);
        store256(slot_0.add(4), a_0_1, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store256(slot_1, a_1_0, alpha_v, beta_v, beta);
        store256(slot_1.add(4), a_1_1, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store256(slot_2, a_2_0, alpha_v, beta_v, beta);
        store256(slot_2.add(4), a_2_1, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store256(slot_3, a_3_0, alpha_v, beta_v, beta);
        store256(slot_3.add(4), a_3_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel256_8x1(
    k: usize,
    alpha_v: std::arch::x86_64::__m256d,
    beta_v: std::arch::x86_64::__m256d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm256_setzero_pd();
        let mut a_0_1 = _mm256_setzero_pd();
        for p in 0..k {
            let ap = packed.add(p * 8);
            let g0 = _mm256_loadu_pd(ap.add(0));
            let g1 = _mm256_loadu_pd(ap.add(4));
            let bv_0 = _mm256_set1_pd(*b.offset(b_row_stride * p as isize + b_col_stride * 0));
            a_0_0 = _mm256_fmadd_pd(g0, bv_0, a_0_0);
            a_0_1 = _mm256_fmadd_pd(g1, bv_0, a_0_1);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store256(slot_0, a_0_0, alpha_v, beta_v, beta);
        store256(slot_0.add(4), a_0_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel256_4x4(
    k: usize,
    alpha_v: std::arch::x86_64::__m256d,
    beta_v: std::arch::x86_64::__m256d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm256_setzero_pd();
        let mut a_1_0 = _mm256_setzero_pd();
        let mut a_2_0 = _mm256_setzero_pd();
        let mut a_3_0 = _mm256_setzero_pd();
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = _mm256_loadu_pd(ap);
            ap = ap.add(4);
            let bv_0 = _mm256_set1_pd(*bp0);
            bp0 = bp0.offset(b_row_stride);
            a_0_0 = _mm256_fmadd_pd(g0, bv_0, a_0_0);
            let bv_1 = _mm256_set1_pd(*bp1);
            bp1 = bp1.offset(b_row_stride);
            a_1_0 = _mm256_fmadd_pd(g0, bv_1, a_1_0);
            let bv_2 = _mm256_set1_pd(*bp2);
            bp2 = bp2.offset(b_row_stride);
            a_2_0 = _mm256_fmadd_pd(g0, bv_2, a_2_0);
            let bv_3 = _mm256_set1_pd(*bp3);
            bp3 = bp3.offset(b_row_stride);
            a_3_0 = _mm256_fmadd_pd(g0, bv_3, a_3_0);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store256(slot_0, a_0_0, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store256(slot_1, a_1_0, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store256(slot_2, a_2_0, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store256(slot_3, a_3_0, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel256_4x1(
    k: usize,
    alpha_v: std::arch::x86_64::__m256d,
    beta_v: std::arch::x86_64::__m256d,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd};
    unsafe {
        let mut a_0_0 = _mm256_setzero_pd();
        for p in 0..k {
            let ap = packed.add(p * 4);
            let g0 = _mm256_loadu_pd(ap.add(0));
            let bv_0 = _mm256_set1_pd(*b.offset(b_row_stride * p as isize + b_col_stride * 0));
            a_0_0 = _mm256_fmadd_pd(g0, bv_0, a_0_0);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store256(slot_0, a_0_0, alpha_v, beta_v, beta);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply256(
    rows: usize,
    n: usize,
    k: usize,
    alpha: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::x86_64::_mm256_set1_pd;
    unsafe {
        let alpha_v = _mm256_set1_pd(alpha);
        let beta_v = _mm256_set1_pd(beta);
        let mut col = 0;
        if rows == 8 {
            while col + 4 <= n {
                kernel256_8x4(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 4;
            }
            while col < n {
                kernel256_8x1(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 1;
            }
        } else {
            while col + 4 <= n {
                kernel256_4x4(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 4;
            }
            while col < n {
                kernel256_4x1(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn call_apply256(
    rows: usize,
    n: usize,
    k: usize,
    alpha: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    // SAFETY: callers are the AVX2 product paths, which run only after
    // `refresh_simd` has checked the CPU.
    unsafe {
        apply256(
            rows,
            n,
            k,
            alpha,
            packed,
            b,
            b_row_stride,
            b_col_stride,
            beta,
            c,
            c_col_stride,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn gemm_avx2(
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
    unsafe {
        let mut row0 = 0;
        while row0 + 8 <= m {
            with_pack(8 * k, |packed| {
                pack_rows(a, a_col_stride, row0, 8, k, packed);
                call_apply256(
                    8,
                    n,
                    k,
                    alpha,
                    packed.as_ptr(),
                    b,
                    b_row_stride,
                    b_col_stride,
                    beta,
                    c.offset(row0 as isize),
                    c_col_stride,
                );
            });
            row0 += 8;
        }
        while row0 + 4 <= m {
            with_pack(4 * k, |packed| {
                pack_rows(a, a_col_stride, row0, 4, k, packed);
                call_apply256(
                    4,
                    n,
                    k,
                    alpha,
                    packed.as_ptr(),
                    b,
                    b_row_stride,
                    b_col_stride,
                    beta,
                    c.offset(row0 as isize),
                    c_col_stride,
                );
            });
            row0 += 4;
        }
        if row0 < m {
            gemm_scalar(
                m - row0,
                n,
                k,
                alpha,
                a.offset(row0 as isize),
                1,
                a_col_stride,
                b,
                b_row_stride,
                b_col_stride,
                beta,
                c.offset(row0 as isize),
                1,
                c_col_stride,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_parallel_avx2(
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
    unsafe {
        let tasks = column_tasks(n).max(1);
        let mut packed = Vec::new();
        let (panels, tail) = pack_row_panels(&mut packed, a, lda as isize, m, k, &[8, 4]);
        let packed_ptr = SyncPtr::new(packed.as_ptr());
        let b_ptr = SyncPtr::new(b);
        let c_ptr = SyncMut::new(c);
        let a_ptr = SyncPtr::new(a);
        for_tiles(tasks, move |tile| {
            let (col0, count) = split_range(n, tasks, tile);
            if count == 0 {
                return;
            }
            for panel in &panels {
                call_apply256(
                    panel.rows,
                    count,
                    k,
                    alpha,
                    packed_ptr.add(panel.off),
                    b_ptr.add(col0 * ldb),
                    1,
                    ldb as isize,
                    beta,
                    c_ptr.add(panel.row0 + col0 * ldc),
                    ldc as isize,
                );
            }
            if tail < m {
                tile_gemm(
                    m - tail,
                    count,
                    k,
                    alpha,
                    a_ptr.add(tail),
                    1,
                    lda as isize,
                    b_ptr.add(col0 * ldb),
                    1,
                    ldb as isize,
                    beta,
                    c_ptr.add(tail + col0 * ldc),
                    1,
                    ldc as isize,
                );
            }
        });
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kernel_neon_4x4(
    k: usize,
    alpha_v: std::arch::aarch64::float64x2_t,
    beta_v: std::arch::aarch64::float64x2_t,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64};
    unsafe {
        let mut a_0_0 = vdupq_n_f64(0.0);
        let mut a_0_1 = vdupq_n_f64(0.0);
        let mut a_1_0 = vdupq_n_f64(0.0);
        let mut a_1_1 = vdupq_n_f64(0.0);
        let mut a_2_0 = vdupq_n_f64(0.0);
        let mut a_2_1 = vdupq_n_f64(0.0);
        let mut a_3_0 = vdupq_n_f64(0.0);
        let mut a_3_1 = vdupq_n_f64(0.0);
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = vld1q_f64(ap);
            let g1 = vld1q_f64(ap.add(2));
            ap = ap.add(4);
            fma_col_pair_neon!(
                g0,
                g1,
                b_row_stride,
                bp0,
                a_0_0,
                a_0_1,
                bp1,
                a_1_0,
                a_1_1,
                bp2,
                a_2_0,
                a_2_1,
                bp3,
                a_3_0,
                a_3_1,
            );
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store_neon(slot_0, a_0_0, alpha_v, beta_v, beta);
        store_neon(slot_0.add(2), a_0_1, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store_neon(slot_1, a_1_0, alpha_v, beta_v, beta);
        store_neon(slot_1.add(2), a_1_1, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store_neon(slot_2, a_2_0, alpha_v, beta_v, beta);
        store_neon(slot_2.add(2), a_2_1, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store_neon(slot_3, a_3_0, alpha_v, beta_v, beta);
        store_neon(slot_3.add(2), a_3_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kernel_neon_4x1(
    k: usize,
    alpha_v: std::arch::aarch64::float64x2_t,
    beta_v: std::arch::aarch64::float64x2_t,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64};
    unsafe {
        let mut a_0_0 = vdupq_n_f64(0.0);
        let mut a_0_1 = vdupq_n_f64(0.0);
        for p in 0..k {
            let ap = packed.add(p * 4);
            let g0 = vld1q_f64(ap.add(0));
            let g1 = vld1q_f64(ap.add(2));
            let bv_0 = vdupq_n_f64(*b.offset(b_row_stride * p as isize + b_col_stride * 0));
            a_0_0 = vfmaq_f64(a_0_0, g0, bv_0);
            a_0_1 = vfmaq_f64(a_0_1, g1, bv_0);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store_neon(slot_0, a_0_0, alpha_v, beta_v, beta);
        store_neon(slot_0.add(2), a_0_1, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kernel_neon_2x4(
    k: usize,
    alpha_v: std::arch::aarch64::float64x2_t,
    beta_v: std::arch::aarch64::float64x2_t,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64};
    unsafe {
        let mut a_0_0 = vdupq_n_f64(0.0);
        let mut a_1_0 = vdupq_n_f64(0.0);
        let mut a_2_0 = vdupq_n_f64(0.0);
        let mut a_3_0 = vdupq_n_f64(0.0);
        let mut bp0 = b;
        let mut bp1 = b.offset(b_col_stride);
        let mut bp2 = b.offset(b_col_stride * 2);
        let mut bp3 = b.offset(b_col_stride * 3);
        let mut ap = packed;
        for _p in 0..k {
            let g0 = vld1q_f64(ap);
            ap = ap.add(2);
            fma_col_neon!(
                g0,
                b_row_stride,
                bp0,
                a_0_0,
                bp1,
                a_1_0,
                bp2,
                a_2_0,
                bp3,
                a_3_0
            );
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store_neon(slot_0, a_0_0, alpha_v, beta_v, beta);
        let slot_1 = c.offset(c_col_stride * 1);
        store_neon(slot_1, a_1_0, alpha_v, beta_v, beta);
        let slot_2 = c.offset(c_col_stride * 2);
        store_neon(slot_2, a_2_0, alpha_v, beta_v, beta);
        let slot_3 = c.offset(c_col_stride * 3);
        store_neon(slot_3, a_3_0, alpha_v, beta_v, beta);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kernel_neon_2x1(
    k: usize,
    alpha_v: std::arch::aarch64::float64x2_t,
    beta_v: std::arch::aarch64::float64x2_t,
    beta: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64};
    unsafe {
        let mut a_0_0 = vdupq_n_f64(0.0);
        for p in 0..k {
            let ap = packed.add(p * 2);
            let g0 = vld1q_f64(ap.add(0));
            let bv_0 = vdupq_n_f64(*b.offset(b_row_stride * p as isize + b_col_stride * 0));
            a_0_0 = vfmaq_f64(a_0_0, g0, bv_0);
        }
        let slot_0 = c.offset(c_col_stride * 0);
        store_neon(slot_0, a_0_0, alpha_v, beta_v, beta);
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_neon(
    rows: usize,
    n: usize,
    k: usize,
    alpha: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    use std::arch::aarch64::vdupq_n_f64;
    unsafe {
        let alpha_v = vdupq_n_f64(alpha);
        let beta_v = vdupq_n_f64(beta);
        let mut col = 0;
        if rows == 4 {
            while col + 4 <= n {
                kernel_neon_4x4(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 4;
            }
            while col < n {
                kernel_neon_4x1(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 1;
            }
        } else {
            while col + 4 <= n {
                kernel_neon_2x4(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 4;
            }
            while col < n {
                kernel_neon_2x1(
                    k,
                    alpha_v,
                    beta_v,
                    beta,
                    packed,
                    b.offset(b_col_stride * col as isize),
                    b_row_stride,
                    b_col_stride,
                    c.offset(c_col_stride * col as isize),
                    c_col_stride,
                );
                col += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn call_apply_neon(
    rows: usize,
    n: usize,
    k: usize,
    alpha: f64,
    packed: *const f64,
    b: *const f64,
    b_row_stride: isize,
    b_col_stride: isize,
    beta: f64,
    c: *mut f64,
    c_col_stride: isize,
) {
    // SAFETY: NEON is baseline on aarch64, and callers are the NEON paths.
    unsafe {
        apply_neon(
            rows,
            n,
            k,
            alpha,
            packed,
            b,
            b_row_stride,
            b_col_stride,
            beta,
            c,
            c_col_stride,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn gemm_neon(
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
    unsafe {
        let mut row0 = 0;
        while row0 + 4 <= m {
            with_pack(4 * k, |packed| {
                pack_rows(a, a_col_stride, row0, 4, k, packed);
                call_apply_neon(
                    4,
                    n,
                    k,
                    alpha,
                    packed.as_ptr(),
                    b,
                    b_row_stride,
                    b_col_stride,
                    beta,
                    c.offset(row0 as isize),
                    c_col_stride,
                );
            });
            row0 += 4;
        }
        while row0 + 2 <= m {
            with_pack(2 * k, |packed| {
                pack_rows(a, a_col_stride, row0, 2, k, packed);
                call_apply_neon(
                    2,
                    n,
                    k,
                    alpha,
                    packed.as_ptr(),
                    b,
                    b_row_stride,
                    b_col_stride,
                    beta,
                    c.offset(row0 as isize),
                    c_col_stride,
                );
            });
            row0 += 2;
        }
        if row0 < m {
            gemm_scalar(
                m - row0,
                n,
                k,
                alpha,
                a.offset(row0 as isize),
                1,
                a_col_stride,
                b,
                b_row_stride,
                b_col_stride,
                beta,
                c.offset(row0 as isize),
                1,
                c_col_stride,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_parallel_neon(
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
    unsafe {
        let tasks = column_tasks(n).max(1);
        let mut packed = Vec::new();
        let (panels, tail) = pack_row_panels(&mut packed, a, lda as isize, m, k, &[4, 2]);
        let packed_ptr = SyncPtr::new(packed.as_ptr());
        let b_ptr = SyncPtr::new(b);
        let c_ptr = SyncMut::new(c);
        let a_ptr = SyncPtr::new(a);
        for_tiles(tasks, move |tile| {
            let (col0, count) = split_range(n, tasks, tile);
            if count == 0 {
                return;
            }
            for panel in &panels {
                call_apply_neon(
                    panel.rows,
                    count,
                    k,
                    alpha,
                    packed_ptr.add(panel.off),
                    b_ptr.add(col0 * ldb),
                    1,
                    ldb as isize,
                    beta,
                    c_ptr.add(panel.row0 + col0 * ldc),
                    ldc as isize,
                );
            }
            if tail < m {
                tile_gemm(
                    m - tail,
                    count,
                    k,
                    alpha,
                    a_ptr.add(tail),
                    1,
                    lda as isize,
                    b_ptr.add(col0 * ldb),
                    1,
                    ldb as isize,
                    beta,
                    c_ptr.add(tail + col0 * ldc),
                    1,
                    ldc as isize,
                );
            }
        });
    }
}

/// Pack once per row panel and split columns on the fixed task grid.
///
/// Returns false when this build has no vector kernel for `m`.
pub(super) fn blocked_parallel(
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
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if current_kind() == KIND_AVX512 && m >= 8 {
            // SAFETY: `refresh_simd` only publishes avx512 when the CPU
            // has it. Column tasks write disjoint columns.
            unsafe {
                gemm_parallel_avx512(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            return true;
        }
        if current_kind() == KIND_AVX2 && m >= 4 {
            // SAFETY: same contract for the AVX2 kernel.
            unsafe {
                gemm_parallel_avx2(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if current_kind() == KIND_NEON && m >= 2 {
            // SAFETY: NEON is baseline on aarch64.
            unsafe {
                gemm_parallel_neon(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            return true;
        }
    }
    let _ = (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    false
}

/// `y[i] += alpha * x[i]`. Each index is independent, so the vector
/// width does not change the bits.
pub(crate) fn axpy_add(y: &mut [f64], x: &[f64], alpha: f64) {
    let n = y.len().min(x.len());
    if n == 0 || alpha == 0.0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if current_kind() == KIND_AVX512 {
            // SAFETY: avx512 was selected only when the CPU has it.
            unsafe { axpy_add_avx512(y.as_mut_ptr(), x.as_ptr(), n, alpha) };
            return;
        }
        if current_kind() == KIND_AVX2 {
            // SAFETY: avx2 was selected only when the CPU has it.
            unsafe { axpy_add_avx2(y.as_mut_ptr(), x.as_ptr(), n, alpha) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if current_kind() == KIND_NEON {
            // SAFETY: NEON is baseline on aarch64.
            unsafe { axpy_add_neon(y.as_mut_ptr(), x.as_ptr(), n, alpha) };
            return;
        }
    }
    for i in 0..n {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn axpy_add_avx512(y: *mut f64, x: *const f64, n: usize, alpha: f64) {
    use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_storeu_pd};
    unsafe {
        let scale = _mm512_set1_pd(alpha);
        let mut i = 0;
        while i + 8 <= n {
            let yv = _mm512_loadu_pd(y.add(i));
            let xv = _mm512_loadu_pd(x.add(i));
            _mm512_storeu_pd(y.add(i), _mm512_fmadd_pd(scale, xv, yv));
            i += 8;
        }
        while i < n {
            *y.add(i) = alpha.mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn axpy_sub_avx512(y: *mut f64, x: *const f64, n: usize, alpha: f64) {
    use std::arch::x86_64::{_mm512_fnmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_storeu_pd};
    unsafe {
        let scale = _mm512_set1_pd(alpha);
        let mut i = 0;
        while i + 8 <= n {
            let yv = _mm512_loadu_pd(y.add(i));
            let xv = _mm512_loadu_pd(x.add(i));
            _mm512_storeu_pd(y.add(i), _mm512_fnmadd_pd(scale, xv, yv));
            i += 8;
        }
        while i < n {
            *y.add(i) = (-alpha).mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_add_avx2(y: *mut f64, x: *const f64, n: usize, alpha: f64) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};
    unsafe {
        let scale = _mm256_set1_pd(alpha);
        let mut i = 0;
        while i + 4 <= n {
            let yv = _mm256_loadu_pd(y.add(i));
            let xv = _mm256_loadu_pd(x.add(i));
            _mm256_storeu_pd(y.add(i), _mm256_fmadd_pd(scale, xv, yv));
            i += 4;
        }
        while i < n {
            *y.add(i) = alpha.mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn axpy_sub_avx2(y: *mut f64, x: *const f64, n: usize, alpha: f64) {
    use std::arch::x86_64::{_mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};
    unsafe {
        let scale = _mm256_set1_pd(alpha);
        let mut i = 0;
        while i + 4 <= n {
            let yv = _mm256_loadu_pd(y.add(i));
            let xv = _mm256_loadu_pd(x.add(i));
            _mm256_storeu_pd(y.add(i), _mm256_fnmadd_pd(scale, xv, yv));
            i += 4;
        }
        while i < n {
            *y.add(i) = (-alpha).mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_add_neon(y: *mut f64, x: *const f64, n: usize, alpha: f64) {
    use std::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};
    unsafe {
        let scale = vdupq_n_f64(alpha);
        let mut i = 0;
        while i + 2 <= n {
            let yv = vld1q_f64(y.add(i));
            let xv = vld1q_f64(x.add(i));
            vst1q_f64(y.add(i), vfmaq_f64(yv, scale, xv));
            i += 2;
        }
        while i < n {
            *y.add(i) = alpha.mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn axpy_sub_neon(y: *mut f64, x: *const f64, n: usize, alpha: f64) {
    use std::arch::aarch64::{vdupq_n_f64, vfmsq_f64, vld1q_f64, vst1q_f64};
    unsafe {
        let scale = vdupq_n_f64(alpha);
        let mut i = 0;
        while i + 2 <= n {
            let yv = vld1q_f64(y.add(i));
            let xv = vld1q_f64(x.add(i));
            vst1q_f64(y.add(i), vfmsq_f64(yv, scale, xv));
            i += 2;
        }
        while i < n {
            *y.add(i) = (-alpha).mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

/// `base + Σ numer[i] / ((delta[i] - shift) - mu)` in index order.
///
/// `None` when a denominator is zero, matching the scalar early return.
pub(crate) fn sum_div_ordered(
    base: f64,
    delta: &[f64],
    numer: &[f64],
    shift: f64,
    mu: f64,
) -> Option<f64> {
    debug_assert_eq!(delta.len(), numer.len());
    let n = delta.len();
    let mut sum = base;
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if current_kind() == KIND_AVX512 && n >= 8 {
            // SAFETY: avx512 was selected only when the CPU has it.
            // A zero denominator aborts the vector chunk and the scalar
            // tail below sees that same index.
            i = unsafe { sum_div_avx512(&mut sum, delta, numer, shift, mu) };
        } else if current_kind() == KIND_AVX2 && n >= 4 {
            i = unsafe { sum_div_avx2(&mut sum, delta, numer, shift, mu) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if current_kind() == KIND_NEON && n >= 2 {
            i = unsafe { sum_div_neon(&mut sum, delta, numer, shift, mu) };
        }
    }
    while i < n {
        let diff = (delta[i] - shift) - mu;
        if diff == 0.0 {
            return None;
        }
        sum += numer[i] / diff;
        i += 1;
    }
    Some(sum)
}

/// Same sum, plus `Σ (numer[i] / diff) / diff` for the derivative.
pub(crate) fn sum_div_deriv_ordered(
    base: f64,
    delta: &[f64],
    numer: &[f64],
    shift: f64,
    mu: f64,
) -> Option<(f64, f64)> {
    debug_assert_eq!(delta.len(), numer.len());
    let n = delta.len();
    let mut sum = base;
    let mut deriv = 0.0;
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if current_kind() == KIND_AVX512 && n >= 8 {
            i = unsafe { sum_div_deriv_avx512(&mut sum, &mut deriv, delta, numer, shift, mu) };
        } else if current_kind() == KIND_AVX2 && n >= 4 {
            i = unsafe { sum_div_deriv_avx2(&mut sum, &mut deriv, delta, numer, shift, mu) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if current_kind() == KIND_NEON && n >= 2 {
            i = unsafe { sum_div_deriv_neon(&mut sum, &mut deriv, delta, numer, shift, mu) };
        }
    }
    while i < n {
        let diff = (delta[i] - shift) - mu;
        if diff == 0.0 {
            return None;
        }
        let term = numer[i] / diff;
        sum += term;
        deriv += term / diff;
        i += 1;
    }
    Some((sum, deriv))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_div_avx512(
    sum: &mut f64,
    delta: &[f64],
    numer: &[f64],
    shift: f64,
    mu: f64,
) -> usize {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _mm512_cmp_pd_mask, _mm512_div_pd, _mm512_loadu_pd, _mm512_set1_pd,
        _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd,
    };
    unsafe {
        let shift_v = _mm512_set1_pd(shift);
        let mu_v = _mm512_set1_pd(mu);
        let zero = _mm512_setzero_pd();
        let n = delta.len();
        let mut i = 0;
        while i + 8 <= n {
            let diff = _mm512_sub_pd(
                _mm512_sub_pd(_mm512_loadu_pd(delta.as_ptr().add(i)), shift_v),
                mu_v,
            );
            if _mm512_cmp_pd_mask(diff, zero, _CMP_EQ_OQ) != 0 {
                break;
            }
            let term = _mm512_div_pd(_mm512_loadu_pd(numer.as_ptr().add(i)), diff);
            let mut tmp = [0.0; 8];
            _mm512_storeu_pd(tmp.as_mut_ptr(), term);
            for value in tmp {
                *sum += value;
            }
            i += 8;
        }
        i
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_div_deriv_avx512(
    sum: &mut f64,
    deriv: &mut f64,
    delta: &[f64],
    numer: &[f64],
    shift: f64,
    mu: f64,
) -> usize {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _mm512_cmp_pd_mask, _mm512_div_pd, _mm512_loadu_pd, _mm512_set1_pd,
        _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd,
    };
    unsafe {
        let shift_v = _mm512_set1_pd(shift);
        let mu_v = _mm512_set1_pd(mu);
        let zero = _mm512_setzero_pd();
        let n = delta.len();
        let mut i = 0;
        while i + 8 <= n {
            let diff = _mm512_sub_pd(
                _mm512_sub_pd(_mm512_loadu_pd(delta.as_ptr().add(i)), shift_v),
                mu_v,
            );
            if _mm512_cmp_pd_mask(diff, zero, _CMP_EQ_OQ) != 0 {
                break;
            }
            let numer_v = _mm512_loadu_pd(numer.as_ptr().add(i));
            let term = _mm512_div_pd(numer_v, diff);
            let slope = _mm512_div_pd(term, diff);
            let mut terms = [0.0; 8];
            let mut slopes = [0.0; 8];
            _mm512_storeu_pd(terms.as_mut_ptr(), term);
            _mm512_storeu_pd(slopes.as_mut_ptr(), slope);
            for lane in 0..8 {
                *sum += terms[lane];
                *deriv += slopes[lane];
            }
            i += 8;
        }
        i
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_div_avx2(sum: &mut f64, delta: &[f64], numer: &[f64], shift: f64, mu: f64) -> usize {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _mm256_cmp_pd, _mm256_div_pd, _mm256_loadu_pd, _mm256_movemask_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    unsafe {
        let shift_v = _mm256_set1_pd(shift);
        let mu_v = _mm256_set1_pd(mu);
        let zero = _mm256_setzero_pd();
        let n = delta.len();
        let mut i = 0;
        while i + 4 <= n {
            let diff = _mm256_sub_pd(
                _mm256_sub_pd(_mm256_loadu_pd(delta.as_ptr().add(i)), shift_v),
                mu_v,
            );
            let eq = _mm256_cmp_pd(diff, zero, _CMP_EQ_OQ);
            if _mm256_movemask_pd(eq) != 0 {
                break;
            }
            let term = _mm256_div_pd(_mm256_loadu_pd(numer.as_ptr().add(i)), diff);
            let mut tmp = [0.0; 4];
            _mm256_storeu_pd(tmp.as_mut_ptr(), term);
            for value in tmp {
                *sum += value;
            }
            i += 4;
        }
        i
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_div_deriv_avx2(
    sum: &mut f64,
    deriv: &mut f64,
    delta: &[f64],
    numer: &[f64],
    shift: f64,
    mu: f64,
) -> usize {
    use std::arch::x86_64::{
        _CMP_EQ_OQ, _mm256_cmp_pd, _mm256_div_pd, _mm256_loadu_pd, _mm256_movemask_pd,
        _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    unsafe {
        let shift_v = _mm256_set1_pd(shift);
        let mu_v = _mm256_set1_pd(mu);
        let zero = _mm256_setzero_pd();
        let n = delta.len();
        let mut i = 0;
        while i + 4 <= n {
            let diff = _mm256_sub_pd(
                _mm256_sub_pd(_mm256_loadu_pd(delta.as_ptr().add(i)), shift_v),
                mu_v,
            );
            let eq = _mm256_cmp_pd(diff, zero, _CMP_EQ_OQ);
            if _mm256_movemask_pd(eq) != 0 {
                break;
            }
            let numer_v = _mm256_loadu_pd(numer.as_ptr().add(i));
            let term = _mm256_div_pd(numer_v, diff);
            let slope = _mm256_div_pd(term, diff);
            let mut terms = [0.0; 4];
            let mut slopes = [0.0; 4];
            _mm256_storeu_pd(terms.as_mut_ptr(), term);
            _mm256_storeu_pd(slopes.as_mut_ptr(), slope);
            for lane in 0..4 {
                *sum += terms[lane];
                *deriv += slopes[lane];
            }
            i += 4;
        }
        i
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_div_neon(sum: &mut f64, delta: &[f64], numer: &[f64], shift: f64, mu: f64) -> usize {
    use std::arch::aarch64::{vdivq_f64, vdupq_n_f64, vld1q_f64, vst1q_f64, vsubq_f64};
    unsafe {
        let shift_v = vdupq_n_f64(shift);
        let mu_v = vdupq_n_f64(mu);
        let n = delta.len();
        let mut i = 0;
        while i + 2 <= n {
            let diff = vsubq_f64(vsubq_f64(vld1q_f64(delta.as_ptr().add(i)), shift_v), mu_v);
            let mut diffs = [0.0; 2];
            vst1q_f64(diffs.as_mut_ptr(), diff);
            if diffs[0] == 0.0 || diffs[1] == 0.0 {
                break;
            }
            let term = vdivq_f64(vld1q_f64(numer.as_ptr().add(i)), diff);
            let mut tmp = [0.0; 2];
            vst1q_f64(tmp.as_mut_ptr(), term);
            *sum += tmp[0];
            *sum += tmp[1];
            i += 2;
        }
        i
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_div_deriv_neon(
    sum: &mut f64,
    deriv: &mut f64,
    delta: &[f64],
    numer: &[f64],
    shift: f64,
    mu: f64,
) -> usize {
    use std::arch::aarch64::{vdivq_f64, vdupq_n_f64, vld1q_f64, vst1q_f64, vsubq_f64};
    unsafe {
        let shift_v = vdupq_n_f64(shift);
        let mu_v = vdupq_n_f64(mu);
        let n = delta.len();
        let mut i = 0;
        while i + 2 <= n {
            let diff = vsubq_f64(vsubq_f64(vld1q_f64(delta.as_ptr().add(i)), shift_v), mu_v);
            let mut diffs = [0.0; 2];
            vst1q_f64(diffs.as_mut_ptr(), diff);
            if diffs[0] == 0.0 || diffs[1] == 0.0 {
                break;
            }
            let term = vdivq_f64(vld1q_f64(numer.as_ptr().add(i)), diff);
            let slope = vdivq_f64(term, diff);
            let mut terms = [0.0; 2];
            let mut slopes = [0.0; 2];
            vst1q_f64(terms.as_mut_ptr(), term);
            vst1q_f64(slopes.as_mut_ptr(), slope);
            *sum += terms[0];
            *deriv += slopes[0];
            *sum += terms[1];
            *deriv += slopes[1];
            i += 2;
        }
        i
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) unsafe fn gemm_avx512(
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
pub(super) unsafe fn gemm_avx2(
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
pub(super) unsafe fn gemm_neon(
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
