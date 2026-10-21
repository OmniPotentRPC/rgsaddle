//! Componentwise comparison with 256-bit MPFR.
//!
//! The only assertion is the dot-product bound in Higham, Accuracy and
//! Stability of Numerical Algorithms, 2nd ed., SIAM, 2002,
//! doi:10.1137/1.9780898718027, Section 3.1:
//! the absolute error is at most `gamma_n * sum |x_i y_i|`, with
//! `gamma_n = n u / (1 - n u)` and `u = 2^-53`, evaluated in MPFR.
//! The max ULP of each kernel is printed with the input that produced it.

use crate::projection::climbing_image_force;
use crate::tangent::{TangentKind, compute_tangent};
use ndarray::Array1;
use proptest::prelude::*;
use proptest::test_runner::TestRunner;
use rug::Float;
use rug::ops::Pow;

const PREC: u32 = 256;
const SEED: u64 = 0x2026_1011;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn finite_sample(state: &mut u64) -> f64 {
    let bits = splitmix(state);
    let unit = (bits >> 11) as f64 / ((1u64 << 53) as f64);
    -4.0 + 8.0 * unit
}

fn mpfr(x: f64) -> Float {
    Float::with_val(PREC, x)
}

fn gamma(n: usize) -> Float {
    let u = Float::with_val(PREC, 2).pow(-53);
    let nu = Float::with_val(PREC, n as u32) * &u;
    Float::with_val(PREC, &nu / (Float::with_val(PREC, 1) - &nu))
}

fn mpfr_dot(x: &[f64], y: &[f64]) -> Float {
    let mut acc = Float::with_val(PREC, 0);
    for (a, b) in x.iter().zip(y) {
        acc += mpfr(*a) * mpfr(*b);
    }
    acc
}

fn f64_dot(x: &[f64], y: &[f64]) -> f64 {
    let left = Array1::from(x.to_vec());
    let right = Array1::from(y.to_vec());
    left.dot(&right)
}

fn sum_abs_products(x: &[f64], y: &[f64]) -> Float {
    let mut acc = Float::with_val(PREC, 0);
    for (a, b) in x.iter().zip(y) {
        acc += mpfr(*a).abs() * mpfr(*b).abs();
    }
    acc
}

/// Section 3.1. Skip pairs whose products overflow: the bound assumes
/// a finite dot product.
fn assert_dot_bound(x: &[f64], y: &[f64]) {
    assert_eq!(x.len(), y.len());
    let got = f64_dot(x, y);
    let reference = mpfr_dot(x, y);
    if !got.is_finite() || !reference.is_finite() {
        return;
    }
    let n = x.len();
    if n == 0 || (n as f64) * f64::EPSILON >= 1.0 {
        return;
    }
    let err = (mpfr(got) - &reference).abs();
    let bound = gamma(n) * sum_abs_products(x, y);
    assert!(
        err <= bound,
        "dot error {err} exceeds gamma_{n} bound {bound}"
    );
}

fn ulp_distance(got: f64, reference: &Float) -> u64 {
    if !got.is_finite() || !reference.is_finite() {
        return u64::MAX;
    }
    let rounded = reference.to_f64();
    if got.is_nan() && rounded.is_nan() {
        return 0;
    }
    let mut a = got.to_bits() as i64;
    let mut b = rounded.to_bits() as i64;
    if a < 0 {
        a = i64::MIN.wrapping_sub(a);
    }
    if b < 0 {
        b = i64::MIN.wrapping_sub(b);
    }
    a.abs_diff(b)
}

fn climbing_mpfr(force: &[f64], tangent: &[f64], dneb: &[f64]) -> Vec<Float> {
    let par = mpfr_dot(force, tangent);
    force
        .iter()
        .zip(tangent)
        .zip(dneb)
        .map(|((f, t), d)| mpfr(*f) - mpfr(*t) * (Float::with_val(PREC, 2) * &par) + mpfr(*d))
        .collect()
}

fn forward_curvature(h: &[f64], x: &[f64], v: &[f64], dr: f64) -> (f64, Float) {
    let mut diff = Vec::with_capacity(x.len());
    let mut diff_ref = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let g0 = h[i] * x[i];
        let shifted = x[i] + dr * v[i];
        let g1 = h[i] * shifted;
        diff.push((g1 - g0) / dr);
        let g0_r = mpfr(h[i]) * mpfr(x[i]);
        let g1_r = mpfr(h[i]) * (mpfr(x[i]) + mpfr(dr) * mpfr(v[i]));
        diff_ref.push((g1_r - g0_r) / mpfr(dr));
    }
    let got_dot = f64_dot(&diff, v);
    let mut ref_dot = Float::with_val(PREC, 0);
    for (d, dir) in diff_ref.iter().zip(v) {
        ref_dot += d * mpfr(*dir);
    }
    // The Higham check is the f64 difference dotted with v, against MPFR
    // of those same f64 components.
    assert_dot_bound(&diff, v);
    (got_dot, ref_dot)
}

struct UlpRecord {
    ulps: u64,
    kind: &'static str,
    input: String,
}

fn consider(best: &mut UlpRecord, ulps: u64, kind: &'static str, input: String) {
    if ulps > best.ulps && ulps != u64::MAX {
        *best = UlpRecord { ulps, kind, input };
    }
}

#[test]
fn report_max_ulp_against_mpfr() {
    let mut state = SEED;
    let mut best = UlpRecord {
        ulps: 0,
        kind: "none",
        input: String::new(),
    };
    for case in 0..64 {
        let dim = 1 + (splitmix(&mut state) as usize % 6);
        let force: Vec<f64> = (0..dim).map(|_| finite_sample(&mut state)).collect();
        let mut tangent: Vec<f64> = (0..dim).map(|_| finite_sample(&mut state)).collect();
        let norm = tangent.iter().map(|t| t * t).sum::<f64>().sqrt();
        if norm > 0.0 {
            for t in &mut tangent {
                *t /= norm;
            }
        }
        let dneb = vec![0.0; dim];
        assert_dot_bound(&force, &tangent);
        let got = climbing_image_force(
            force.as_slice().into(),
            tangent.as_slice().into(),
            dneb.as_slice().into(),
        );
        let reference = climbing_mpfr(&force, &tangent, &dneb);
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            consider(
                &mut best,
                ulp_distance(*g, r),
                "climbing_image_force",
                format!("case {case} component {i} force={force:?} tangent={tangent:?}"),
            );
        }

        let next: Vec<f64> = (0..dim).map(|_| finite_sample(&mut state)).collect();
        let prev: Vec<f64> = (0..dim).map(|_| finite_sample(&mut state)).collect();
        let energy = finite_sample(&mut state);
        let energy_prev = finite_sample(&mut state);
        let energy_next = finite_sample(&mut state);
        if let Ok(tang) = compute_tangent(
            TangentKind::Improved,
            next.as_slice().into(),
            prev.as_slice().into(),
            energy,
            energy_prev,
            energy_next,
        ) {
            // The normalisation dot is `raw · raw` before the square root.
            // Compare that f64 square against MPFR. The public function
            // returns the unit vector, so rebuild the raw tangent the same way.
            let raw = improved_raw(&next, &prev, energy, energy_prev, energy_next);
            assert_dot_bound(&raw, &raw);
            let raw_norm = mpfr_dot(&raw, &raw).sqrt();
            if raw_norm > Float::with_val(PREC, 1e-10) {
                for (i, component) in tang.iter().enumerate() {
                    let reference = mpfr(raw[i]) / &raw_norm;
                    consider(
                        &mut best,
                        ulp_distance(*component, &reference),
                        "compute_tangent",
                        format!(
                            "case {case} component {i} next={next:?} prev={prev:?} E=({energy},{energy_prev},{energy_next})"
                        ),
                    );
                }
            }
        }

        let h: Vec<f64> = (0..dim)
            .map(|_| finite_sample(&mut state).abs() + 0.25)
            .collect();
        let x: Vec<f64> = (0..dim).map(|_| finite_sample(&mut state)).collect();
        let mut v: Vec<f64> = (0..dim).map(|_| finite_sample(&mut state)).collect();
        let vn = v.iter().map(|t| t * t).sum::<f64>().sqrt();
        if vn > 0.0 {
            for dir in &mut v {
                *dir /= vn;
            }
        }
        let dr = 1e-3;
        let (got_c, ref_c) = forward_curvature(&h, &x, &v, dr);
        consider(
            &mut best,
            ulp_distance(got_c, &ref_c),
            "forward_curvature",
            format!("case {case} h={h:?} x={x:?} v={v:?} dr={dr}"),
        );
    }
    println!("max ULP {} on {} at {}", best.ulps, best.kind, best.input);
}

fn improved_raw(
    next: &[f64],
    prev: &[f64],
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
) -> Vec<f64> {
    if energy_next > energy && energy > energy_prev {
        return next.to_vec();
    }
    if energy > energy_next && energy_prev > energy {
        return prev.to_vec();
    }
    let d_prev = energy_prev - energy;
    let d_next = energy_next - energy;
    let min_diff = d_prev.abs().min(d_next.abs());
    let max_diff = d_prev.abs().max(d_next.abs());
    if d_prev > d_next {
        next.iter()
            .zip(prev)
            .map(|(n, p)| n * min_diff + p * max_diff)
            .collect()
    } else {
        next.iter()
            .zip(prev)
            .map(|(n, p)| n * max_diff + p * min_diff)
            .collect()
    }
}

#[test]
fn proptest_dot_obeys_higham() {
    // SplitMix64 above is the counter used for the printed sample.
    // This runner is pinned so a failure replays.
    // The seed is process-global. The tests in this module do not run in
    // parallel with another proptest that reads it.
    unsafe { std::env::set_var("PROPTEST_RNG_SEED", "20261011") };
    let mut config = ProptestConfig::with_cases(64);
    config.max_shrink_iters = 16;
    let mut runner = TestRunner::new(config);
    let strategy = prop::collection::vec((-8.0f64..8.0, -8.0f64..8.0), 1..8);
    let result = runner.run(&strategy, |pairs| {
        let x: Vec<f64> = pairs.iter().map(|(a, _)| *a).collect();
        let y: Vec<f64> = pairs.iter().map(|(_, b)| *b).collect();
        prop_assume!(x.iter().chain(y.iter()).all(|v| v.is_finite()));
        assert_dot_bound(&x, &y);
        let zeros = vec![0.0; x.len()];
        let got = climbing_image_force(
            x.as_slice().into(),
            y.as_slice().into(),
            zeros.as_slice().into(),
        );
        prop_assume!(got.iter().all(|v| v.is_finite()));
        Ok(())
    });
    if let Err(err) = result {
        panic!("Higham dot bound failed: {err}");
    }
}
