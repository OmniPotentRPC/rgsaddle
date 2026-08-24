//! Sella order-1 session: P-RFO + TrustRegion over [`CartesianPes`].
//!
//! `sella.optimize.optimize.Sella` with `order=1`, `method=prfo`.
//! Distinct from [`crate::minmode::MinModeSession`] (dimer / Lanczos
//! force inversion). Hosts own the loop.

use ndarray::Array1;
use rgmin::prfo_restricted;

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::pes::CartesianPes;

pub struct SellaSaddleConfig {
    pub delta: f64,
    pub force_tol: f64,
    pub force_gate: crate::ForceGate,
    pub order: usize,
}

impl Default for SellaSaddleConfig {
    fn default() -> Self {
        Self {
            delta: 0.2,
            force_tol: 1e-3,
            force_gate: crate::ForceGate::MaxForceOnAtom,
            order: 1,
        }
    }
}

pub struct SellaSaddleReport {
    pub energy: f64,
    pub max_force: f64,
    pub at_saddle: bool,
}

pub struct SellaSaddleSession {
    pes: CartesianPes,
    config: SellaSaddleConfig,
}

impl SellaSaddleSession {
    pub fn new(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        Ok(Self {
            pes: CartesianPes::new(x, masses)?,
            config,
        })
    }

    pub fn position(&self) -> ndarray::ArrayView1<'_, f64> {
        self.pes.position()
    }

    pub fn reset(&mut self) {
        self.pes.reset();
    }

    pub fn step<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let (energy, g) = surface.eval(self.pes.position())?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let max_force = self.config.force_gate.value(g.view());
        if max_force <= self.config.force_tol {
            return Ok(SellaSaddleReport {
                energy,
                max_force,
                at_saddle: true,
            });
        }
        let (evals, evecs) = self.pes.hessian().eigh();
        let s = prfo_restricted(
            &evals,
            &evecs,
            &g,
            self.config.order.max(1),
            self.config.delta,
        );
        self.pes.kick(surface, s.view())?;
        let (energy, g) = surface.eval(self.pes.position())?;
        let max_force = self.config.force_gate.value(g.view());
        Ok(SellaSaddleReport {
            energy,
            max_force,
            at_saddle: max_force <= self.config.force_tol,
        })
    }

    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut n = 1;
        while !report.at_saddle && n < max_steps {
            report = self.step(surface)?;
            n += 1;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minmode::PointSurface;
    use ndarray::{Array1, ArrayView1};

    struct Well;
    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    #[test]
    fn prfo_step_is_finite() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaSaddleSession::new(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
    }
}
