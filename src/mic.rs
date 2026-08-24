//! Minimum-image convention via [linkcell](https://github.com/d-SEAMS/linkcell).
//!
//! Band and IRC differences go through this module. vesin builds cutoff
//! neighbour lists when a surface needs them; it is not a second MIC.

use ndarray::Array1;

/// Re-export: the periodic cell is [linkcell::Cell], not a private 3x3.
pub use linkcell::Cell;

/// Wrap every per-atom 3-vector in `diff` to its minimum image.
pub fn wrap_difference(cell: &Cell, diff: &mut Array1<f64>) {
    let origin = cell.origin();
    for a in 0..diff.len() / 3 {
        let q = [
            origin[0] + diff[3 * a],
            origin[1] + diff[3 * a + 1],
            origin[2] + diff[3 * a + 2],
        ];
        let fp = cell.fractional(origin);
        let fq = cell.fractional(q);
        let ds = [
            (fq[0] - fp[0]) - (fq[0] - fp[0]).round(),
            (fq[1] - fp[1]) - (fq[1] - fp[1]).round(),
            (fq[2] - fp[2]) - (fq[2] - fp[2]).round(),
        ];
        let cart = cell.cartesian([fp[0] + ds[0], fp[1] + ds[1], fp[2] + ds[2]]);
        diff[3 * a] = cart[0] - origin[0];
        diff[3 * a + 1] = cart[1] - origin[1];
        diff[3 * a + 2] = cart[2] - origin[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_wraps_across_the_box() {
        let cell = Cell::ortho(10.0, 10.0, 10.0).unwrap();
        let mut d = Array1::from(vec![9.0, 0.0, 0.0]);
        wrap_difference(&cell, &mut d);
        assert!((d[0] + 1.0).abs() < 1e-12, "{}", d[0]);
    }
}
