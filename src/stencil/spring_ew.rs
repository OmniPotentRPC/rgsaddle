//! Energy-weighted spring constants.
//!
//! Ásgeirsson, Birgisson, Bjornsson, Becker, Neese, Riplinger, and
//! Jónsson, J. Chem. Theory Comput. 17, 4929 (2021),
//! doi:10.1021/acs.jctc.1c00462. The same law is eOn
//! `buildSpringStrategy` and ORCA `SpringConst` / `SpringConst2`.
//! Goswami, MethodsX 16, 103899 (2026), doi:10.1016/j.mex.2026.103899,
//! records the practice.
//!
//! For segment `i` joining images `i` and `i + 1`, `E_i` is the
//! higher of the two image energies, `E_max` is the highest energy on
//! the band, and `E_ref` is the higher endpoint. When `E_i` is above
//! `E_ref`,
//! `alpha_i = clamp((E_max - E_i) / (E_max - E_ref), 0, 1)` and
//! `k_i = (1 - alpha_i) k_max + alpha_i k_min`. Otherwise `k_i` is
//! `k_min`. When `E_max - E_ref` is below `1e-10` in the host energy
//! unit, including the exact case `E_max = E_ref`, every constant is
//! `k_min`. That is the continuous extension: the denominator is zero
//! and no segment sits strictly above `E_ref`.
//!
//! Defaults are `k_min = 0.972` and `k_max = 9.72` eV/Å^2, eOn's
//! values, which are ORCA's 0.01 and 0.1 Eh/Bohr^2. The climbing
//! image carries no spring force, so its two segment constants are
//! unused while it climbs.

/// Host-unit gap below which `E_max` and `E_ref` are treated as equal.
pub const ENERGY_WEIGHTED_FLAT: f64 = 1e-10;

/// eOn default, ORCA 0.01 Eh/Bohr^2.
pub const K_MIN_EV_PER_A2: f64 = 0.972;

/// eOn default, ORCA 0.1 Eh/Bohr^2.
pub const K_MAX_EV_PER_A2: f64 = 9.72;

/// Write one constant per segment into `out`.
///
/// `out` has length `energies.len() - 1`. The max is a serial loop in
/// image order. Ties do not change the value.
pub fn fill_constants(energies: &[f64], k_min: f64, k_max: f64, out: &mut [f64]) {
    let n = energies.len();
    debug_assert_eq!(out.len() + 1, n);
    if n < 2 {
        return;
    }
    let mut e_max = energies[0];
    for energy in energies.iter().skip(1) {
        if *energy > e_max {
            e_max = *energy;
        }
    }
    let e_ref = if energies[0] >= energies[n - 1] {
        energies[0]
    } else {
        energies[n - 1]
    };
    if e_max - e_ref < ENERGY_WEIGHTED_FLAT {
        out.fill(k_min);
        return;
    }
    let span = e_max - e_ref;
    for i in 0..n - 1 {
        let e_i = if energies[i] >= energies[i + 1] {
            energies[i]
        } else {
            energies[i + 1]
        };
        if e_i > e_ref {
            let alpha = ((e_max - e_i) / span).clamp(0.0, 1.0);
            out[i] = (1.0 - alpha) * k_max + alpha * k_min;
        } else {
            out[i] = k_min;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_barrier_peaks_at_k_max() {
        let energies = [0.0, 0.5, 1.0, 0.5, 0.0];
        let mut ks = [0.0; 4];
        fill_constants(&energies, 1.0, 5.0, &mut ks);
        assert!((ks[0] - 3.0).abs() < 1e-12);
        assert!((ks[1] - 5.0).abs() < 1e-12);
        assert!((ks[2] - 5.0).abs() < 1e-12);
        assert!((ks[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn asymmetric_barrier_uses_the_higher_endpoint() {
        // E_ref is 0.8, the product, not the reactant at 0.
        // The first segment sits at or below E_ref, so it is k_min.
        // The last segment is between E_ref and E_max.
        let energies = [0.0, 0.5, 2.0, 1.2, 0.8];
        let mut ks = [0.0; 4];
        fill_constants(&energies, 1.0, 5.0, &mut ks);
        assert!((ks[0] - 1.0).abs() < 1e-12);
        assert!((ks[1] - 5.0).abs() < 1e-12);
        assert!((ks[2] - 5.0).abs() < 1e-12);
        let alpha = (2.0 - 1.2) / (2.0 - 0.8);
        let expect = (1.0 - alpha) * 5.0 + alpha * 1.0;
        assert!((ks[3] - expect).abs() < 1e-12);
        assert!((alpha - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn flat_band_is_all_k_min() {
        let energies = [1.5, 1.5, 1.5, 1.5];
        let mut ks = [0.0; 3];
        fill_constants(&energies, 0.972, 9.72, &mut ks);
        assert!(ks.iter().all(|k| (*k - 0.972).abs() < 1e-15));
    }

    #[test]
    fn equal_e_max_and_e_ref_is_k_min() {
        // The crest is an endpoint, so the quotient is undefined.
        // The continuous extension is k_min on every segment.
        let energies = [2.0, 1.0, 0.5, 0.0];
        let mut ks = [9.0; 3];
        fill_constants(&energies, 0.25, 4.0, &mut ks);
        assert!(ks.iter().all(|k| (*k - 0.25).abs() < 1e-15));
    }

    #[test]
    fn a_segment_below_e_ref_stays_at_k_min() {
        let energies = [0.0, -0.2, 1.0, 0.3];
        let mut ks = [0.0; 3];
        fill_constants(&energies, 2.0, 8.0, &mut ks);
        assert!((ks[0] - 2.0).abs() < 1e-12);
        assert!(ks[1] > ks[0]);
    }
}
