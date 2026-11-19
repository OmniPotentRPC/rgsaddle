//! Three-component arithmetic shared by the solid-state and constraint code.
//!
//! Each product is written out in index order, the same expression the
//! copies used, so a sum of three terms keeps its bits.

pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Left-to-right product sum. Not [`rgmin::vecops::dot`]: that fold is an
/// eight-partial tree, and these callers were the iterator sum.
pub(crate) fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm_slice(v: &[f64]) -> f64 {
    dot_slice(v, v).sqrt()
}
