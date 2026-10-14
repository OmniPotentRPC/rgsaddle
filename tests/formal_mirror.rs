//! Bit comparison of the Rust stencil against `formal/golden/stencil.json`.
//!
//! `translated_gradient` is `pub(crate)`, so its rows are checked from
//! `src/kani_proofs.rs`. `SpringKind::Weighted` stores precomputed
//! constants and does not evaluate the energy-weighted law, so those
//! rows are checked against the formula itself. The formula returns
//! `k_min` when `E_max == E_ref` and does not divide.

use ndarray::Array1;
use rgsaddle::projection::climbing_image_force;
use rgsaddle::tangent::{TangentKind, compute_tangent};
use std::path::PathBuf;

fn rat(token: &str) -> f64 {
    let (num, den) = token
        .split_once('/')
        .unwrap_or_else(|| panic!("num/den in {token}"));
    let num: i64 = num.parse().unwrap_or_else(|_| panic!("{token}"));
    let den: i64 = den.parse().unwrap_or_else(|_| panic!("{token}"));
    num as f64 / den as f64
}

fn slash_numbers(line: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('"') {
        rest = &rest[start + 1..];
        let end = rest.find('"').expect("closing quote");
        let token = &rest[..end];
        if token.contains('/') {
            out.push(rat(token));
        }
        rest = &rest[end + 1..];
    }
    out
}

fn section_lines<'a>(text: &'a str, key: &str) -> Vec<&'a str> {
    let marker = format!("\"{key}\"");
    let start = text
        .find(&marker)
        .unwrap_or_else(|| panic!("missing {key}"));
    let mut lines = Vec::new();
    for line in text[start..].lines().skip(1) {
        if line.trim_start().starts_with(']') {
            break;
        }
        if line.contains('/') {
            lines.push(line);
        }
    }
    lines
}

fn bits_eq(got: f64, expected: f64, context: &str) {
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "{context}: got {got:?} ({:016x}) expected {expected:?} ({:016x})",
        got.to_bits(),
        expected.to_bits()
    );
}

#[test]
fn climbing_image_matches_reflect_rows() {
    let text = golden();
    let mut saw = 0;
    for line in section_lines(&text, "reflect") {
        let nums = slash_numbers(line);
        assert!(nums.len().is_multiple_of(4) && nums.len() >= 8, "{line}");
        let dim = nums.len() / 4;
        let v = &nums[0..dim];
        let f = &nums[dim..2 * dim];
        let dneb = &nums[2 * dim..3 * dim];
        let out = &nums[3 * dim..];
        let got = climbing_image_force(f.into(), v.into(), dneb.into());
        for i in 0..dim {
            bits_eq(got[i], out[i], line);
        }
        saw += 1;
    }
    assert!(saw >= 3);
}

#[test]
fn improved_tangent_matches_golden_bits() {
    let text = golden();
    let mut saw = 0;
    for line in section_lines(&text, "tangent") {
        let nums = slash_numbers(line);
        // next, prev, three energies, norm, out. Rows are two-dimensional.
        assert_eq!(nums.len(), 10, "{line}");
        let next = Array1::from(vec![nums[0], nums[1]]);
        let prev = Array1::from(vec![nums[2], nums[3]]);
        let got = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            nums[4],
            nums[5],
            nums[6],
        )
        .expect(line);
        bits_eq(got[0], nums[8], line);
        bits_eq(got[1], nums[9], line);
        saw += 1;
    }
    assert!(saw >= 3);
}

/// Energy-weighted spring constant from the specification.
///
/// `E_i = max(E[i], E[i+1])` is the caller's input. When
/// `E_max == E_ref` the quotient is undefined and this returns `k_min`.
/// `SpringKind::Weighted` does not implement the quotient. It consumes
/// constants the caller already computed, so the guard has to live in
/// that caller.
fn energy_weighted_k(e_i: f64, e_max: f64, e_ref: f64, k_min: f64, k_max: f64) -> f64 {
    if e_max == e_ref || e_i <= e_ref {
        return k_min;
    }
    let alpha = ((e_max - e_i) / (e_max - e_ref)).clamp(0.0, 1.0);
    (1.0 - alpha) * k_max + alpha * k_min
}

#[test]
fn energy_weighted_rows_match_the_guarded_formula() {
    let text = golden();
    let mut saw = 0;
    for line in section_lines(&text, "energy_weighted") {
        let nums = slash_numbers(line);
        assert_eq!(nums.len(), 6, "{line}");
        let got = energy_weighted_k(nums[0], nums[1], nums[2], nums[3], nums[4]);
        bits_eq(got, nums[5], line);
        saw += 1;
    }
    assert!(saw >= 5);
}

fn golden() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("formal/golden/stencil.json");
    std::fs::read_to_string(path).expect("formal/golden/stencil.json")
}
