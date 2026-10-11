//! Orthogonality and residual checks for `eigh` and `partial_eigh`.
//!
//! Bounds are `c · n · ε · ‖A‖`, with `c = 1000`. Sizes above the band
//! and Cuppen thresholds run in release builds.

use ndarray::{Array1, Array2, ArrayView2};

fn next_open(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(0x5851_F42D_4C95_7F2D)
        .wrapping_add(0x1405_7B7E_F767_814F);
    let bits = (*state >> 11) as u32;
    (f64::from(bits) / f64::from(u32::MAX)).mul_add(2.0, -1.0)
}

fn matrix_norm(a: ArrayView2<f64>) -> f64 {
    let n = a.nrows();
    let mut norm = 0.0f64;
    for i in 0..n {
        let mut row = 0.0f64;
        for j in 0..n {
            row += a[(i, j)].abs();
        }
        norm = norm.max(row);
    }
    norm
}

fn bound(n: usize, norm: f64) -> f64 {
    1000.0 * (n as f64) * f64::EPSILON * norm
}

fn check_pairs(a: ArrayView2<f64>, evals: &Array1<f64>, vecs: &Array2<f64>) {
    let n = a.nrows();
    let k = evals.len();
    let norm = matrix_norm(a);
    let limit = bound(n, norm);
    let mut ortho = 0.0f64;
    for i in 0..k {
        for j in 0..k {
            let mut product = 0.0f64;
            for row in 0..n {
                product += vecs[(row, i)] * vecs[(row, j)];
            }
            let expect = if i == j { 1.0 } else { 0.0 };
            ortho = ortho.max((product - expect).abs());
        }
    }
    let av = a.dot(vecs);
    let mut residual = 0.0f64;
    for col in 0..k {
        let mut err = 0.0f64;
        for row in 0..n {
            let diff = av[(row, col)] - evals[col] * vecs[(row, col)];
            err += diff * diff;
        }
        residual = residual.max(err.sqrt());
    }
    assert!(
        ortho <= limit.max(1000.0 * (n as f64) * f64::EPSILON),
        "orthogonality {ortho} limit {limit} n={n}"
    );
    assert!(
        residual <= limit,
        "residual {residual} limit {limit} norm {norm} n={n}"
    );
    for col in 0..k {
        let mut best = 0usize;
        let mut best_abs = 0.0;
        for row in 0..n {
            let magnitude = vecs[(row, col)].abs();
            if magnitude > best_abs {
                best_abs = magnitude;
                best = row;
            }
        }
        assert!(vecs[(best, col)] >= 0.0, "column {col} sign at row {best}");
    }
}

fn check_both(a: &Array2<f64>, k: usize) {
    let (full, fvecs) = rgsaddle::eigh(a.view()).expect("eigh");
    check_pairs(a.view(), &full, &fvecs);
    let (partial, pvecs) = rgsaddle::partial_eigh(a.view(), k).expect("partial");
    check_pairs(a.view(), &partial, &pvecs);
    let limit = bound(a.nrows(), matrix_norm(a.view()));
    for i in 0..k {
        assert!(
            (partial[i] - full[i]).abs() <= limit,
            "partial[{i}] {} vs {} limit {limit}",
            partial[i],
            full[i]
        );
    }
}

fn random_symmetric(n: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed;
    let mut a = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let value = next_open(&mut state) * scale;
            a[(i, j)] = value;
            a[(j, i)] = value;
        }
    }
    a
}

fn random_tridiagonal(n: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed;
    let mut a = Array2::zeros((n, n));
    for i in 0..n {
        a[(i, i)] = next_open(&mut state) * scale;
        if i + 1 < n {
            let off = next_open(&mut state) * scale;
            a[(i, i + 1)] = off;
            a[(i + 1, i)] = off;
        }
    }
    a
}

/// Wilkinson `W_{21}^+`: diagonal 10, 9, ..., 0, ..., 9, 10 and unit
/// off-diagonals.
fn wilkinson_21() -> Array2<f64> {
    let n = 21;
    let mut a = Array2::zeros((n, n));
    for i in 0..n {
        a[(i, i)] = (10 - i as i32).abs() as f64;
        if i + 1 < n {
            a[(i, i + 1)] = 1.0;
            a[(i + 1, i)] = 1.0;
        }
    }
    a
}

fn glued_wilkinson(copies: usize, bridge: f64) -> Array2<f64> {
    let block = 21;
    let n = block * copies;
    let mut a = Array2::zeros((n, n));
    let one = wilkinson_21();
    for copy in 0..copies {
        let base = copy * block;
        for i in 0..block {
            for j in 0..block {
                a[(base + i, base + j)] = one[(i, j)];
            }
        }
        if copy + 1 < copies {
            let joint = base + block - 1;
            a[(joint, joint + 1)] = bridge;
            a[(joint + 1, joint)] = bridge;
        }
    }
    a
}

fn block_diagonal() -> Array2<f64> {
    let mut a = Array2::zeros((6, 6));
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 2.0;
    a[(2, 2)] = -3.0;
    a[(3, 3)] = 0.5;
    a[(3, 4)] = 0.25;
    a[(4, 3)] = 0.25;
    a[(4, 4)] = 0.5;
    a[(5, 5)] = 8.0;
    a
}

#[test]
fn partial_eigh_splits_diag_3_1_2() {
    let mut a = Array2::zeros((3, 3));
    a[(0, 0)] = 3.0;
    a[(1, 1)] = 1.0;
    a[(2, 2)] = 2.0;
    let (evals, vecs) = rgsaddle::partial_eigh(a.view(), 1).expect("diag(3, 1, 2)");
    assert!((evals[0] - 1.0).abs() <= bound(3, 3.0));
    check_pairs(a.view(), &evals, &vecs);
    assert!((vecs[(1, 0)].abs() - 1.0).abs() <= 1e-12);
}

#[test]
fn partial_eigh_on_scaled_random_tridiagonals() {
    for scale in [1.0, 0.3, 0.1, 0.01] {
        let a = random_tridiagonal(48, 0x71D1_A6, scale);
        check_both(&a, 1);
        check_both(&a, 10);
    }
}

#[test]
fn eigh_on_block_diagonal_diagonal_and_zero() {
    let blocks = block_diagonal();
    check_both(&blocks, 2);
    let mut diagonal = Array2::zeros((5, 5));
    for i in 0..5 {
        diagonal[(i, i)] = (i as f64) - 2.0;
    }
    check_both(&diagonal, 3);
    let zero = Array2::zeros((4, 4));
    check_both(&zero, 2);
}

#[test]
fn eigh_on_scaled_random_symmetric() {
    for scale in [1e-3, 1e3, 1e-8, 1e8] {
        let a = random_symmetric(12, 0x5CA1_E, scale);
        check_both(&a, 1);
        check_both(&a, 4);
    }
}

#[test]
fn wilkinson_and_glued_wilkinson() {
    let w = wilkinson_21();
    check_both(&w, 4);
    if cfg!(debug_assertions) {
        return;
    }
    let glued = glued_wilkinson(8, 0.0);
    check_both(&glued, 3);
    let bridged = glued_wilkinson(8, 1e-8);
    check_both(&bridged, 3);
}

#[test]
fn orthogonality_above_the_band_and_cuppen_thresholds() {
    if cfg!(debug_assertions) {
        return;
    }
    for n in [200usize, 600] {
        let a = random_symmetric(n, 0xBEEF_0000 + n as u64, 1.0);
        check_both(&a, 4);
        let mut clustered = Array2::zeros((n, n));
        for i in 0..n {
            clustered[(i, i)] = if i < n / 2 { -1.5 } else { 0.25 * (i as f64) };
        }
        check_both(&clustered, 6);
    }
}
