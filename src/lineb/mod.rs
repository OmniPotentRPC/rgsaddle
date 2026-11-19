//! Line-integral nudged elastic band (LI-NEB) optimal tunneling paths.
//!
//! Ásgeirsson, Arnaldsson and Jónsson, J. Chem. Phys. 148, 102334 (2018),
//! doi:10.1063/1.5007180, minimize the abbreviated action
//! `W(E) = ∫ √(2(V−E)) ds` at fixed energy. The images are evenly spaced
//! by a parallel spring, the endpoints are held on the equipotential
//! `V = E`, and the action gradient is projected perpendicular to the
//! path. Coordinates are mass-weighted. A molecule is aligned with the
//! Kabsch quaternion and its rigid motions are removed from the force.
//!
//! [`continue_from_mep`] steps the energy down from a minimum-energy
//! path. [`LinebPath`] reports `W`, the half period `−∂W/∂E`, and the
//! half period from constrained dynamics on the cubic spline; the
//! temperature is `ℏ / (k_B τ)` with `τ` the full period.
//! [`hqtst_rate`] rebuilds the ring polymer at `t_j = j τ / P` and
//! evaluates the harmonic quantum TST factor from Hessian-vector
//! products. The paper's finite-difference Hessian interpolation is
//! built only with the `lineb-oracle` feature, for tests and benches.
//!
//! The minimizer is an [`rgmin`](https://github.com/OmniPotentRPC/rgmin)
//! stepper. The potential is a [`PointSurface`], the same surface an
//! rgpot model is wrapped in.
//!
//! # Examples
//!
//! The symmetric Eckart barrier `V = sech²(x)` at `E = 1/4`. The
//! turning points are `±acosh(2)` and the one-way action is
//! `π (√2 − √(1/2))`.
//!
//! ```
//! use ndarray::{Array1, Array2, ArrayView1};
//! use rgsaddle::{
//!     LinebConfig, LinebSession, LinebStatus, Masses, PointSurface, SaddleError,
//! };
//!
//! struct Eckart;
//!
//! impl PointSurface for Eckart {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
//!         let s = 1.0 / x[0].cosh();
//!         let energy = s * s;
//!         Ok((energy, Array1::from_elem(1, -2.0 * s * s * x[0].tanh())))
//!     }
//! }
//!
//! let turning = (1.0 / 0.25_f64).sqrt().acosh();
//! let images = Array2::from_shape_fn((9, 1), |(i, _)| {
//!     let end = 0.5 * turning;
//!     -end + 2.0 * end * (i as f64) / 8.0
//! });
//! let spacing = turning * 2.0 / 8.0;
//! let draft = LinebConfig::new(spacing, spacing);
//! let panels = (2 * draft.quadrature_pairs) as f64;
//! // An endpoint `δ` short of its turning point shortens the half period
//! // by `√(2δ/|∇V|)`. Its force is `endpoint_scale * δ`, so this keeps
//! // both ends inside the quadrature slack below.
//! let tol = draft.endpoint_scale * spacing / (8.0 * panels.powi(4));
//! let config = draft.with_opt_tol(tol).with_perp_tol(tol);
//! let mut session = LinebSession::new(
//!     config.clone(),
//!     Masses::coordinates(Array1::ones(1).view())?,
//!     images,
//!     0.25,
//! )?;
//! let report = session.run(&Eckart, 400)?;
//! assert_eq!(report.status, LinebStatus::Converged);
//! let path = session.path(&Eckart)?;
//! let exact_action = std::f64::consts::PI * (2.0_f64.sqrt() - 0.5_f64.sqrt());
//! let exact_half = std::f64::consts::PI * 2.0_f64.sqrt();
//! let slope = 0.5 * 0.75_f64.sqrt();
//! let slack = (spacing / panels.powi(4) / slope).sqrt();
//! assert!((path.action - exact_action).abs() <= slack);
//! assert!((path.half_period - exact_half).abs() <= slack);
//! assert!((path.half_period - path.half_period_dynamics).abs() <= slack);
//! # Ok::<(), SaddleError>(())
//! ```

mod action;
mod align;
mod continue_energy;
mod mass;
mod optimize;
mod path;
mod rate;

#[cfg(test)]
mod check;

pub use continue_energy::{Continuation, continue_from_mep};
pub use mass::Masses;
pub use optimize::{LinebCalls, LinebConfig, LinebPath, LinebReport, LinebSession, LinebStatus};
#[cfg(any(test, feature = "lineb-oracle"))]
pub use rate::hqtst_rate_interpolated_hessian;
pub use rate::{HqtstRate, hqtst_rate};

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// The surface, counting its evaluations and the Hessian-vector
/// products it computes itself in one tally.
pub(crate) struct Counted<'a, S> {
    pub(crate) surface: &'a S,
    pub(crate) calls: &'a AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.surface.eval(x)
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let product = self.surface.hessian_vector(x, v)?;
        if product.is_some() {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        Ok(product)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.surface.excluded_modes(x)
    }
}
