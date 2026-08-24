//! Sella `samd.py`: Bussi-Donadio-Parrinello thermostat on a surface.
//!
//! One host-owned step: velocity-Verlet then the BDP kinetic rescale.
//! The host supplies the Gaussian draw `r` so the session stays
//! deterministic under test. Temperature schedules are the Sella
//! linear and exponential ramps.

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::{axpy, dot};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Linear ramp `T0 + i (Tf - T0) / (n - 1)`.
pub fn t_linear(i: usize, t0: f64, tf: f64, n: usize) -> f64 {
    if n <= 1 {
        return tf;
    }
    t0 + (i as f64) * (tf - t0) / ((n - 1) as f64)
}

/// Exponential ramp `T0 (Tf / T0)^{i / n}`.
pub fn t_exp(i: usize, t0: f64, tf: f64, n: usize) -> f64 {
    if t0 <= 0.0 || tf <= 0.0 {
        return t0;
    }
    t0 * (tf / t0).powf((i as f64) / (n.max(1) as f64))
}

pub struct SamdConfig {
    pub dt: f64,
    pub tau: f64,
    pub t0: f64,
    pub tf: f64,
    pub ngen: usize,
    pub exponential: bool,
}

impl Default for SamdConfig {
    fn default() -> Self {
        Self {
            dt: 0.1,
            tau: 1.0,
            t0: 1.0,
            tf: 0.1,
            ngen: 8,
            exponential: false,
        }
    }
}

pub struct SamdReport {
    pub energy: f64,
    pub kinetic: f64,
    pub temperature: f64,
}

pub struct SamdSession {
    x: Array1<f64>,
    v: Array1<f64>,
    g: Array1<f64>,
    energy: f64,
    config: SamdConfig,
    i: usize,
}

impl SamdSession {
    pub fn new(
        config: SamdConfig,
        x: Array1<f64>,
        v0: Array1<f64>,
        surface: &impl PointSurface,
    ) -> Result<Self, SaddleError> {
        if x.len() != v0.len() {
            return Err(SaddleError::Shape(
                "SAMD velocity must match the 3N frame".into(),
            ));
        }
        let (energy, g) = surface.eval(x.view())?;
        if !g.iter().all(|a| a.is_finite()) {
            return Err(SaddleError::NonFinite("samd gradient"));
        }
        Ok(Self {
            x,
            v: v0,
            g,
            energy,
            config,
            i: 0,
        })
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn velocity(&self) -> ArrayView1<'_, f64> {
        self.v.view()
    }

    /// One BDP step. `r` is length `3N` (Sella `np.random.normal`).
    pub fn step(
        &mut self,
        surface: &impl PointSurface,
        r: ArrayView1<f64>,
    ) -> Result<SamdReport, SaddleError> {
        if r.len() != self.x.len() {
            return Err(SaddleError::Shape(
                "SAMD Gaussian draw must match the 3N frame".into(),
            ));
        }
        let d = self.x.len() as f64;
        let dt = self.config.dt;
        let old_g = self.g.clone();
        // x += dt v - 0.5 dt^2 g
        axpy(dt, self.v.view(), &mut self.x);
        axpy(-0.5 * dt * dt, old_g.view(), &mut self.x);
        let (energy, g) = surface.eval(self.x.view())?;
        if !g.iter().all(|a| a.is_finite()) {
            return Err(SaddleError::NonFinite("samd gradient"));
        }
        // v -= 0.5 dt (g + old_g)
        axpy(-0.5 * dt, g.view(), &mut self.v);
        axpy(-0.5 * dt, old_g.view(), &mut self.v);
        self.g = g;
        self.energy = energy;

        let t = if self.config.exponential {
            t_exp(self.i, self.config.t0, self.config.tf, self.config.ngen)
        } else {
            t_linear(self.i, self.config.t0, self.config.tf, self.config.ngen)
        };
        let k_target = d * t / 2.0;
        let k = dot(self.v.view(), self.v.view()) / 2.0;
        if k > 1e-12 {
            let edttau = (-dt / self.config.tau).exp();
            let edttau2 = (-dt / (2.0 * self.config.tau)).exp();
            let r2 = dot(r, r);
            let alpha2 = edttau
                + k * (1.0 - edttau) * r2 / (d * k)
                + 2.0 * edttau2 * (k_target * (1.0 - edttau) / (d * k)).sqrt() * r[0];
            if alpha2 > 0.0 && alpha2.is_finite() {
                let s = alpha2.sqrt();
                self.v.mapv_inplace(|vi| vi * s);
            }
        }
        self.i += 1;
        let kinetic = dot(self.v.view(), self.v.view()) / 2.0;
        Ok(SamdReport {
            energy,
            kinetic,
            temperature: t,
        })
    }

    pub fn run(
        &mut self,
        surface: &impl PointSurface,
        draws: &[Array1<f64>],
    ) -> Result<SamdReport, SaddleError> {
        if draws.is_empty() {
            return Ok(SamdReport {
                energy: self.energy,
                kinetic: dot(self.v.view(), self.v.view()) / 2.0,
                temperature: self.config.t0,
            });
        }
        let mut report = self.step(surface, draws[0].view())?;
        for r in draws.iter().skip(1) {
            report = self.step(surface, r.view())?;
        }
        Ok(report)
    }
}

/// Project a SAMD increment onto a manifold (Sella waist: stay on set).
pub fn project_velocity<M: rgmin::Manifold>(
    m: &M,
    x: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    m.project(x, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SaddleError;
    use ndarray::{Array1, ArrayView1};
    use rgmin::vecops::nrm2;
    use rgmin::{Manifold, ManifoldKind};

    struct Well;
    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let mut g = Array1::zeros(x.len());
            g[0] = 2.0 * x[0];
            Ok((x[0] * x[0], g))
        }
    }

    #[test]
    fn t_linear_hits_the_endpoints() {
        assert!((t_linear(0, 2.0, 0.5, 5) - 2.0).abs() < 1e-14);
        assert!((t_linear(4, 2.0, 0.5, 5) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn bdp_step_is_finite_and_rescales() {
        let x = Array1::from(vec![0.4, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v0 = Array1::from(vec![0.3, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut sess = SamdSession::new(SamdConfig::default(), x, v0, &Well).unwrap();
        let r = Array1::from(vec![0.2, 0.1, 0.0, 0.0, 0.0, 0.0]);
        let report = sess.step(&Well, r.view()).unwrap();
        assert!(report.energy.is_finite());
        assert!(report.kinetic.is_finite());
        assert!(sess.position().iter().all(|a| a.is_finite()));
        assert!(nrm2(sess.velocity()) > 0.0);
    }

    #[test]
    fn projected_velocity_stays_on_the_rigid_quotient() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let v = Array1::from_elem(9, 0.1);
        let vp = project_velocity(&ManifoldKind::RigidQuotient, &x, &v);
        let vh = ManifoldKind::RigidQuotient.project(&x, &vp);
        assert!(nrm2((&vp - &vh).view()) < 1e-12);
    }
}
