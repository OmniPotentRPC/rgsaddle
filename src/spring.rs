//! Spring forces, ported from eOn NEBSpringForce.cpp.

use ndarray::{Array1, ArrayView1};

/// Per-image spring result: the tangent-parallel component consumed
/// by the NEB and DNEB projections, and the full spring vector
/// consumed by the plain elastic band and the DNEB correction.
pub struct SpringForces {
    pub parallel: Array1<f64>,
    pub full: Array1<f64>,
}

/// Which spring model the band uses.
#[derive(Clone, Debug)]
pub enum SpringKind {
    /// One constant for every segment.
    Uniform { k: f64 },
    /// Henkelman-style energy-weighted constants, one per segment
    /// (`ks[i]` couples images `i` and `i+1`; length `n_images - 1`).
    Weighted { ks: Vec<f64> },
    /// Mandelli & Parrinello (2021) Onsager–Machlup springs; `l_vecs`
    /// holds one force-scaled displacement per image (zero at the
    /// endpoints).
    OnsagerMachlup { k: f64, l_vecs: Vec<Array1<f64>> },
}

impl SpringKind {
    /// Spring forces at interior image `i` (image indexing over the
    /// whole band, endpoints included). `pos_diff_next` is
    /// `R[i+1] - R[i]` and `pos_diff_prev` is `R[i] - R[i-1]`, both
    /// minimum-imaged by a periodic host.
    pub fn compute(
        &self,
        i: usize,
        tangent: ArrayView1<f64>,
        dist_next: f64,
        dist_prev: f64,
        pos_diff_next: ArrayView1<f64>,
        pos_diff_prev: ArrayView1<f64>,
    ) -> SpringForces {
        match self {
            SpringKind::Uniform { k } => SpringForces {
                parallel: tangent.to_owned() * (k * (dist_next - dist_prev)),
                full: (&pos_diff_next - &pos_diff_prev) * *k,
            },
            SpringKind::Weighted { ks } => {
                let k_next = ks[i];
                let k_prev = ks[i - 1];
                SpringForces {
                    parallel: tangent.to_owned() * (k_next * dist_next - k_prev * dist_prev),
                    full: &pos_diff_next * k_next - &pos_diff_prev * k_prev,
                }
            }
            SpringKind::OnsagerMachlup { k, l_vecs } => {
                // Mandelli Eq. 13 then Eq. 15 (project onto tangent).
                // R[i+1] + R[i-1] - 2 R[i] is written as the difference
                // of the two minimum-imaged segments, so a periodic
                // band whose atoms cross the cell sees the same spring.
                let diff = &pos_diff_next - &pos_diff_prev + &l_vecs[i + 1] - &l_vecs[i];
                let f_om = diff * *k;
                let par = f_om.dot(&tangent);
                SpringForces {
                    parallel: tangent.to_owned() * par,
                    full: Array1::zeros(tangent.len()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn uniform_parallel_balances_equal_spacing() {
        let t = array![1.0, 0.0];
        let pdn = array![0.5, 0.0];
        let pdp = array![0.5, 0.0];
        let s = SpringKind::Uniform { k: 2.0 };
        let f = s.compute(1, t.view(), 0.5, 0.5, pdn.view(), pdp.view());
        assert_abs_diff_eq!(f.parallel[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(f.full[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn uniform_parallel_pulls_toward_longer_segment() {
        let t = array![1.0, 0.0];
        let pdn = array![0.7, 0.0];
        let pdp = array![0.3, 0.0];
        let s = SpringKind::Uniform { k: 2.0 };
        let f = s.compute(1, t.view(), 0.7, 0.3, pdn.view(), pdp.view());
        assert_abs_diff_eq!(f.parallel[0], 2.0 * 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(f.full[0], 2.0 * 0.4, epsilon = 1e-12);
    }

    #[test]
    fn onsager_machlup_reads_only_the_segment_differences() {
        // An image whose neighbours sit across a periodic boundary: the
        // host hands minimum-imaged segments, so the spring is the
        // second difference of the segments, not of raw coordinates.
        let t = array![1.0, 0.0];
        let pdn = array![0.6, 0.0];
        let pdp = array![0.4, 0.0];
        let l = vec![array![0.0, 0.0]; 3];
        let s = SpringKind::OnsagerMachlup { k: 3.0, l_vecs: l };
        let f = s.compute(1, t.view(), 0.6, 0.4, pdn.view(), pdp.view());
        assert_abs_diff_eq!(f.parallel[0], 3.0 * 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(f.parallel[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn weighted_full_spring_uses_the_segment_constants() {
        let t = array![1.0, 0.0];
        let pdn = array![0.5, 0.1];
        let pdp = array![0.5, -0.1];
        let s = SpringKind::Weighted {
            ks: vec![1.0, 2.0, 4.0],
        };
        let f = s.compute(1, t.view(), 0.5, 0.5, pdn.view(), pdp.view());
        // k_next = ks[1] = 2, k_prev = ks[0] = 1.
        assert_abs_diff_eq!(f.full[0], 2.0 * 0.5 - 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(f.full[1], 2.0 * 0.1 + 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(f.parallel[0], 2.0 * 0.5 - 0.5, epsilon = 1e-12);
    }
}
