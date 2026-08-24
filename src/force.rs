//! Closed force-gate enum. Same three norms as eOn / gpr_optim
//! `ConvergenceForceNorm` (`l2Norm`, `linfNorm`, `maxForceOnAtom`).
//!
//! The host picks one. Sessions do not invent a fourth scalar.
//! Maturin / PyO3 emit a real Python enum (`ForceGate.L2_NORM`, …).

use ndarray::ArrayView1;
use rgmin::vecops::{nrm2, nrminf};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// How a session reduces a 3N force (or gradient) to one scalar.
///
/// Discriminant matches `gprd_params.capnp` `ConvergenceForceNorm`
/// and `rgsaddle_force_gate_t` on the C wire. Python names match
/// eOn / gpr (`L2_NORM`, `LINF_NORM`, `MAX_FORCE_ON_ATOM`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(
        eq,
        eq_int,
        frozen,
        rename_all = "SCREAMING_SNAKE_CASE",
        name = "ForceGate",
        module = "rgsaddle"
    )
)]
pub enum ForceGate {
    /// Euclidean `||F||_2` over the full 3N vector.
    L2Norm = 0,
    /// Max absolute component.
    LinfNorm = 1,
    /// Max per-atom `||F_i||_2`. Sella `PES.converged` and the Baker
    /// production gate (`max_force_on_atom`).
    #[default]
    MaxForceOnAtom = 2,
}

impl ForceGate {
    /// Closed C / Cap'n Proto ordinal. Unknown values are `None`.
    pub fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::L2Norm),
            1 => Some(Self::LinfNorm),
            2 => Some(Self::MaxForceOnAtom),
            _ => None,
        }
    }

    /// C / Cap'n Proto ordinal.
    pub const fn to_abi(self) -> i32 {
        self as i32
    }

    /// Static C name, for bindings and `rgsaddle_force_gate_name`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::L2Norm => "L2",
            Self::LinfNorm => "LINF",
            Self::MaxForceOnAtom => "MAX_ATOM",
        }
    }

    /// Reduce `g` (a gradient; same magnitude as `-F`) to the gate scalar.
    pub fn value(self, g: ArrayView1<f64>) -> f64 {
        match self {
            Self::L2Norm => nrm2(g),
            Self::LinfNorm => nrminf(g),
            Self::MaxForceOnAtom => max_force_on_atom(g),
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl ForceGate {
    /// Construct from the C / Cap'n Proto ordinal. Unknown values raise.
    #[new]
    fn py_new(value: i32) -> pyo3::PyResult<Self> {
        Self::try_from_abi(value).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown force gate {value}"))
        })
    }

    fn __int__(&self) -> i32 {
        self.to_abi()
    }

    fn __index__(&self) -> i32 {
        self.to_abi()
    }
}

fn max_force_on_atom(g: ArrayView1<f64>) -> f64 {
    if g.len() < 3 {
        return nrminf(g);
    }
    let mut m = 0.0;
    let n = g.len() / 3;
    for i in 0..n {
        let fx = g[3 * i];
        let fy = g[3 * i + 1];
        let fz = g[3 * i + 2];
        let nrm = (fx * fx + fy * fy + fz * fz).sqrt();
        if nrm > m {
            m = nrm;
        }
    }
    m
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
}
