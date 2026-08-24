//! Sella order-0 session: QN + TrustRegion over [`CartesianPes`].
//!
//! `sella.optimize.optimize.Sella` with `order=0`, `method=qn`.
//! One step is eval, `qn_restricted`, `PES.kick`. The host owns
//! the loop. `run` is a convenience.

use ndarray::Array1;
use rgmin::vecops::nrminf;
use rgmin::qn_restricted;

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::pes::CartesianPes;

pub struct SellaMinConfig {
    pub delta: f64,
    pub force_tol: f64,
}

impl Default for SellaMinConfig {
    fn default() -> Self {
        Self {
            delta: 0.2,
            force_tol: 1e-3,
        }
    }
}

pub struct SellaMinReport {
    pub energy: f64,
    pub max_force: f64,
    pub at_minimum: bool,
}

pub struct SellaMinSession {
    pes: CartesianPes,
    config: SellaMinConfig,
}

impl SellaMinSession {
    pub fn new(
        config: SellaMinConfig,
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

    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<SellaMinReport, SaddleError> {
        let (energy, g) = surface.eval(self.pes.position())?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let max_force = nrminf(g.view());
        if max_force <= self.config.force_tol {
            return Ok(SellaMinReport {
                energy,
                max_force,
                at_minimum: true,
            });
        }
        let (evals, evecs) = self.pes.hessian().eigh();
        let s = qn_restricted(&evals, &evecs, &g, 0, self.config.delta);
        self.pes.kick(surface, s.view())?;
        let (energy, g) = surface.eval(self.pes.position())?;
        let max_force = nrminf(g.view());
        Ok(SellaMinReport {
            energy,
            max_force,
            at_minimum: max_force <= self.config.force_tol,
        })
    }

    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<SellaMinReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut n = 1;
        while !report.at_minimum && n < max_steps {
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
    fn qn_trust_reaches_the_well() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
            },
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        let report = sess.run(&Well, 40).unwrap();
        assert!(report.at_minimum, "force {}", report.max_force);
        assert!((sess.position()[0].abs() - 1.0).abs() < 0.25);
    }
}
