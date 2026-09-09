//! Basin constrained dimer forces from Xiao et al., JCP 141, 164111 (2014).
use ndarray::{Array1, ArrayView1, ArrayView2};
use rgmin::{EigenParams, EigensolverKind};
use crate::SaddleError;

#[derive(Clone, Debug)]
pub struct KappaDimerConfig {
    /// Switching length in the position units; beta * kappa is dimensionless.
    pub beta: f64,
    pub eigen: EigenParams,
}

impl Default for KappaDimerConfig {
    fn default() -> Self {
        Self { beta: 5.0, eigen: EigenParams {
            kind: EigensolverKind::Dimer, tol: 1e-6, max_iter: 100,
            ..EigenParams::default()
        }}
    }
}

#[derive(Clone, Debug)]
pub struct KappaDimerForce {
    pub force: Array1<f64>,
    pub tangent_mode: Array1<f64>,
    pub tangent_dimension: usize,
    pub tangent_curvature: f64,
    pub kappa: f64,
    pub gamma_parallel: f64,
    pub gamma_perpendicular: f64,
    pub residual: f64,
    pub actions: usize,
}

/// Equation (6) in the complement of the force and the excluded modes,
/// followed by the equation (7) force. Rows of `excluded` are Cartesian
/// symmetry or constraint directions. The Hessian callback receives full
/// Cartesian vectors; it need not assemble a matrix.
pub fn kappa_dimer_force<F>(
    gradient: ArrayView1<f64>, mode: ArrayView1<f64>, excluded: ArrayView2<f64>,
    apply_hessian: F, config: &KappaDimerConfig,
) -> Result<KappaDimerForce, SaddleError>
where F: Fn(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    let _ = (gradient, mode, excluded, apply_hessian, config);
    Err(SaddleError::Solver("kappa force is not implemented".into()))
}
