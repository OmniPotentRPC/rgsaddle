//! Spring forces, ported from eOn NEBSpringForce.cpp.

use std::borrow::Cow;

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
    /// Energy-weighted springs of Ásgeirsson et al., J. Chem. Theory
    /// Comput. 17, 4929 (2021), doi:10.1021/acs.jctc.1c00462, set again
    /// from the band energies at every evaluation as eOn's
    /// `buildSpringStrategy` does. [`energy_weighted_constants`] is the
    /// rule. Before the band resolves them, [`SpringKind::compute`]
    /// reads `k_min` on every segment.
    EnergyWeighted { k_min: f64, k_max: f64 },
}

/// Segment constants of the energy-weighted rule. `energies` holds
/// every image, endpoints included, and `ks[i]` couples images `i` and
/// `i + 1`. With `E_ref` the higher endpoint, `E_max` the highest
/// interior image, and `E_i` the higher image of segment `i`, a segment
/// above `E_ref` gets `k_max - (k_max - k_min) (E_max - E_i) /
/// (E_max - E_ref)` and every other segment gets `k_min`. A band whose
/// highest interior image is not above `E_ref` keeps `k_min` throughout.
pub fn energy_weighted_constants(energies: ArrayView1<f64>, k_min: f64, k_max: f64) -> Vec<f64> {
    let n = energies.len();
    let mut ks = vec![k_min; n.saturating_sub(1)];
    if n < 3 {
        return ks;
    }
    let reference = energies[0].max(energies[n - 1]);
    let highest = energies
        .slice(ndarray::s![1..n - 1])
        .fold(f64::NEG_INFINITY, |m, &e| m.max(e));
    let span = highest - reference;
    if span.is_nan() || span < 1e-10 {
        return ks;
    }
    for (i, k) in ks.iter_mut().enumerate() {
        let e = energies[i].max(energies[i + 1]);
        if e > reference {
            let alpha = ((highest - e) / span).clamp(0.0, 1.0);
            *k = (1.0 - alpha) * k_max + alpha * k_min;
        }
    }
    ks
}

impl SpringKind {
    /// The springs at these image energies: per-segment constants for
    /// [`SpringKind::EnergyWeighted`], and `self` for every other kind.
    pub fn at_energies(&self, energies: ArrayView1<f64>) -> Cow<'_, SpringKind> {
        match self {
            SpringKind::EnergyWeighted { k_min, k_max } => Cow::Owned(SpringKind::Weighted {
                ks: energy_weighted_constants(energies, *k_min, *k_max),
            }),
            _ => Cow::Borrowed(self),
        }
    }

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
        let coupling = match self {
            SpringKind::Uniform { k } => crate::stencil::Coupling::Uniform { k: *k },
            SpringKind::Weighted { ks } => crate::stencil::Coupling::Weighted {
                k_next: ks[i],
                k_prev: ks[i - 1],
            },
            SpringKind::OnsagerMachlup { k, l_vecs } => crate::stencil::Coupling::Onsager {
                k: *k,
                l_next: l_vecs[i + 1].view(),
                l_here: l_vecs[i].view(),
            },
            SpringKind::EnergyWeighted { k_min, .. } => {
                crate::stencil::Coupling::Uniform { k: *k_min }
            }
        };
        let forces = crate::stencil::coupling_forces(
            &coupling,
            tangent,
            dist_next,
            dist_prev,
            pos_diff_next,
            pos_diff_prev,
        );
        SpringForces {
            parallel: forces.parallel,
            full: forces.full,
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

    #[test]
    fn energy_weighted_constants_follow_the_segment_rule() {
        // E_ref = 1, E_max = 5. Segment 0 tops at 2, so alpha = 3/4.
        let ks = energy_weighted_constants(array![0.0, 2.0, 5.0, 1.0].view(), 0.97, 9.7);
        assert_eq!(ks.len(), 3);
        assert_abs_diff_eq!(ks[0], 0.25 * 9.7 + 0.75 * 0.97, epsilon = 1e-12);
        assert_abs_diff_eq!(ks[1], 9.7, epsilon = 1e-12);
        assert_abs_diff_eq!(ks[2], 9.7, epsilon = 1e-12);
        // A segment no higher than the higher endpoint stays soft.
        let ks = energy_weighted_constants(array![3.0, 1.0, 4.0, 0.5, 3.0].view(), 1.0, 10.0);
        assert_abs_diff_eq!(ks[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ks[1], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ks[2], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ks[3], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn energy_weighted_constants_stay_soft_without_an_interior_peak() {
        let monotonic = energy_weighted_constants(array![0.0, 1.0, 2.0, 3.0].view(), 0.5, 5.0);
        assert_eq!(monotonic, vec![0.5; 3]);
        let flat = energy_weighted_constants(array![1.0, 1.0, 1.0].view(), 0.5, 5.0);
        assert_eq!(flat, vec![0.5; 2]);
        let nan = energy_weighted_constants(array![0.0, f64::NAN, 0.0].view(), 0.5, 5.0);
        assert_eq!(nan, vec![0.5; 2]);
    }

    #[test]
    fn energy_weighted_springs_resolve_at_each_energy_set() {
        let s = SpringKind::EnergyWeighted {
            k_min: 1.0,
            k_max: 9.0,
        };
        let low = s.at_energies(array![0.0, 4.0, 2.0, 0.0].view());
        let high = s.at_energies(array![0.0, 2.0, 4.0, 0.0].view());
        let (SpringKind::Weighted { ks: a }, SpringKind::Weighted { ks: b }) = (&*low, &*high)
        else {
            panic!("energy-weighted springs resolve to segment constants");
        };
        assert_eq!(a, &vec![9.0, 9.0, 5.0]);
        assert_eq!(b, &vec![5.0, 9.0, 9.0]);
        let uniform = SpringKind::Uniform { k: 2.0 };
        assert!(matches!(
            uniform.at_energies(array![0.0, 1.0, 0.0].view()),
            Cow::Borrowed(SpringKind::Uniform { .. })
        ));
    }
}
