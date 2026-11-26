//! Closed force-gate enum. Same three norms as eOn / gpr_optim
//! `ConvergenceForceNorm` (`l2Norm`, `linfNorm`, `maxForceOnAtom`).
//!
//! The host picks one. Sessions do not invent a fourth scalar.
//! The Python package exposes these discriminants as `ForceGate`.
//!
//! The gate compares against a force tolerance in the energy-over-length
//! unit of the host (eV/Angstrom for eV and Angstrom). The default,
//! [`ForceGate::MaxForceOnAtom`], is the criterion of ASE's `fmax`
//! (`max_i ||F_i||_2 < fmax`), so an ASE tolerance carries over unchanged.

use ndarray::ArrayView1;
use rgmin::vecops::{nrm2, nrminf};

/// How a session reduces a 3N force (or gradient) to one scalar.
///
/// The vector is flat and atom-major, `[x0, y0, z0, x1, y1, z1, ...]`, in
/// the energy-over-length unit of the host (eV/Angstrom). The sign does not
/// matter, so a force and a gradient give the same scalar.
///
/// Discriminant matches `gprd_params.capnp` `ConvergenceForceNorm`
/// and `rgsaddle_force_gate_t` on the C wire. Python names match
/// eOn / gpr (`L2_NORM`, `LINF_NORM`, `MAX_FORCE_ON_ATOM`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum ForceGate {
    /// Euclidean `||F||_2` over the full 3N vector, in eV/Angstrom.
    L2Norm = 0,
    /// Max absolute component `max_k |F_k|` over the 3N vector, in eV/Angstrom.
    LinfNorm = 1,
    /// Max per-atom `||F_i||_2` over the `[F_ix, F_iy, F_iz]` triples, in
    /// eV/Angstrom. This is ASE's `fmax` test, Sella's `PES.converged` and the
    /// Baker production gate (`max_force_on_atom`). It is the default.
    #[default]
    MaxForceOnAtom = 2,
}

impl ForceGate {
    /// Closed C / Cap'n Proto ordinal: `0` is [`ForceGate::L2Norm`], `1` is
    /// [`ForceGate::LinfNorm`] and `2` is [`ForceGate::MaxForceOnAtom`].
    /// Any other value, negative ones included, returns `None`.
    pub fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::L2Norm),
            1 => Some(Self::LinfNorm),
            2 => Some(Self::MaxForceOnAtom),
            _ => None,
        }
    }

    /// C / Cap'n Proto ordinal, the inverse of [`ForceGate::try_from_abi`].
    pub const fn to_abi(self) -> i32 {
        self as i32
    }

    /// Static C name, for bindings and `rgsaddle_force_gate_name`:
    /// `"L2"`, `"LINF"` or `"MAX_ATOM"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::L2Norm => "L2",
            Self::LinfNorm => "LINF",
            Self::MaxForceOnAtom => "MAX_ATOM",
        }
    }

    /// Reduce `g` (a gradient; same magnitude as `-F`) to the gate scalar.
    ///
    /// `g` has length `3 N`, atom-major, in eV/Angstrom. A finite result is
    /// nonnegative and in the same unit. [`ForceGate::MaxForceOnAtom`] skips
    /// an atom whose norm is NaN when another atom has a positive finite
    /// norm. When a NaN atom is present and every finite atom norm is zero,
    /// the result is NaN, so an all-NaN gradient does not report zero. When
    /// `g` has fewer than three entries, the gate uses the max absolute
    /// component, and a non-finite component yields infinity. Trailing
    /// entries of a length that is not a multiple of three are ignored by
    /// the per-atom gate.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use rgsaddle::ForceGate;
    ///
    /// // Two atoms: forces (3, 4, 0) and (0, 0, 1) in eV/Angstrom.
    /// let f = array![3.0, 4.0, 0.0, 0.0, 0.0, 1.0];
    /// assert_eq!(ForceGate::MaxForceOnAtom.value(f.view()), 5.0);
    /// assert_eq!(ForceGate::LinfNorm.value(f.view()), 4.0);
    /// approx::assert_abs_diff_eq!(ForceGate::L2Norm.value(f.view()), 26f64.sqrt(), epsilon = f64::EPSILON * 8.0);
    /// ```
    pub fn value(self, g: ArrayView1<f64>) -> f64 {
        match self {
            Self::L2Norm => nrm2(g),
            Self::LinfNorm => nrminf(g),
            Self::MaxForceOnAtom => max_force_on_atom(g),
        }
    }
}

fn max_force_on_atom(g: ArrayView1<f64>) -> f64 {
    if g.len() < 3 {
        return nrminf(g);
    }
    let mut max = 0.0;
    let mut saw_nan = false;
    let atoms = g.len() / 3;
    for atom in 0..atoms {
        let base = 3 * atom;
        let norm =
            (g[base] * g[base] + g[base + 1] * g[base + 1] + g[base + 2] * g[base + 2]).sqrt();
        if norm.is_nan() {
            saw_nan = true;
            continue;
        }
        if norm > max {
            max = norm;
        }
    }
    if saw_nan && max <= 0.0 { f64::NAN } else { max }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn three_gates_disagree_on_a_tilted_atom() {
        let g = array![0.0008, 0.0008, 0.0008, 0.0, 0.0, 0.0];
        let l2 = ForceGate::L2Norm.value(g.view());
        let linf = ForceGate::LinfNorm.value(g.view());
        let atom = ForceGate::MaxForceOnAtom.value(g.view());
        assert!((linf - 0.0008).abs() < 1e-14);
        assert!((atom - (3.0_f64 * 0.0008 * 0.0008).sqrt()).abs() < 1e-14);
        assert!(atom > 1e-3);
        assert!(linf < 1e-3);
        assert!((l2 - atom).abs() < 1e-14);
        assert_eq!(ForceGate::try_from_abi(2), Some(ForceGate::MaxForceOnAtom));
        assert_eq!(ForceGate::LinfNorm.to_abi(), 1);
        assert_eq!(ForceGate::try_from_abi(99), None);
    }

    #[test]
    fn max_force_on_atom_does_not_treat_nan_as_zero() {
        let all_nan = array![f64::NAN, f64::NAN, f64::NAN];
        assert!(ForceGate::MaxForceOnAtom.value(all_nan.view()).is_nan());
        let quiet_nan = array![0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0];
        assert!(ForceGate::MaxForceOnAtom.value(quiet_nan.view()).is_nan());
        let mixed = array![3.0, 4.0, 0.0, f64::NAN, 0.0, 0.0];
        assert_eq!(ForceGate::MaxForceOnAtom.value(mixed.view()), 5.0);
    }
}
