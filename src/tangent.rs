//! Band tangents, ported from eOn NEBTangent.cpp with identical
//! branch structure.

use ndarray::{Array1, ArrayView1};

/// Which tangent estimate the band uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TangentKind {
    /// Mills, Jonsson, Schenter, Surf. Sci. 324:305 (1995): the
    /// forward difference alone.
    Simple,
    /// Henkelman & Jonsson, JCP 113:9978 (2000): uphill selection at
    /// monotonic points, energy-weighted interpolation at extrema.
    #[default]
    Improved,
}

/// Normalize with the eOn fallback: a near-zero tangent falls back to
/// the forward difference.
fn normalize_tangent(mut tang: Array1<f64>, pos_diff_next: ArrayView1<f64>) -> Array1<f64> {
    let norm = tang.dot(&tang).sqrt();
    if norm > 1e-10 {
        tang /= norm;
        return tang;
    }
    let mut fallback = pos_diff_next.to_owned();
    let norm = fallback.dot(&fallback).sqrt();
    if norm > 1e-10 {
        fallback /= norm;
    }
    fallback
}

/// Unit tangent at one interior image. `pos_diff_next` is
/// `R[i+1] - R[i]`, `pos_diff_prev` is `R[i] - R[i-1]` (the host
/// minimum-images both when periodic).
pub fn compute_tangent(
    kind: TangentKind,
    pos_diff_next: ArrayView1<f64>,
    pos_diff_prev: ArrayView1<f64>,
    energy: f64,
    energy_prev: f64,
    energy_next: f64,
) -> Array1<f64> {
    let tang = match kind {
        TangentKind::Simple => pos_diff_next.to_owned(),
        TangentKind::Improved => {
            if energy_next > energy && energy > energy_prev {
                pos_diff_next.to_owned()
            } else if energy > energy_next && energy_prev > energy {
                pos_diff_prev.to_owned()
            } else {
                // Extremum: energy-weighted combination.
                let energy_diff_prev = energy_prev - energy;
                let energy_diff_next = energy_next - energy;
                let min_diff = energy_diff_prev.abs().min(energy_diff_next.abs());
                let max_diff = energy_diff_prev.abs().max(energy_diff_next.abs());
                if energy_diff_prev > energy_diff_next {
                    &pos_diff_next * min_diff + &pos_diff_prev * max_diff
                } else {
                    &pos_diff_next * max_diff + &pos_diff_prev * min_diff
                }
            }
        }
    };
    normalize_tangent(tang, pos_diff_next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn improved_picks_uphill_forward() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            0.0,
            2.0,
        );
        assert_abs_diff_eq!(t[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn improved_picks_downhill_backward() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            2.0,
            0.0,
        );
        assert_abs_diff_eq!(t[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn improved_weights_extremum_toward_steeper_side() {
        let next = array![1.0, 0.0];
        let prev = array![0.0, 1.0];
        // Maximum: prev side steeper (|dE_prev| = 2 > |dE_next| = 1).
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            3.0,
            1.0,
            2.0,
        );
        // energy_diff_prev (-2) < energy_diff_next (-1): next*max + prev*min.
        let expected = (array![1.0, 0.0] * 2.0 + array![0.0, 1.0] * 1.0) / 5.0f64.sqrt();
        assert_abs_diff_eq!(t[0], expected[0], epsilon = 1e-12);
        assert_abs_diff_eq!(t[1], expected[1], epsilon = 1e-12);
    }

    #[test]
    fn zero_tangent_falls_back_to_forward_difference() {
        let next = array![0.5, 0.0];
        let prev = array![-0.5, 0.0];
        // Extremum with equal, zero energy differences: combination is
        // zero-weighted; fallback normalizes the forward difference.
        let t = compute_tangent(
            TangentKind::Improved,
            next.view(),
            prev.view(),
            1.0,
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(t[0], 1.0, epsilon = 1e-12);
    }
}
