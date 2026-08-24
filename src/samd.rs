//! Sella `samd.py`: Bussi-Donadio-Parrinello thermostat on a surface.
//!
//! One host-owned step: velocity-Verlet then the BDP kinetic rescale.
//! The host supplies the Gaussian draw `r` so the session stays
//! deterministic under test. Temperature schedules are the Sella
//! linear and exponential ramps.
//!
//! [`SamdSession::step`] is the Euclidean Sella loop. A host that
//! needs the point on a set calls [`SamdSession::step_on`]: project
//! the Verlet increment, retract, transport the velocity, then the
//! BDP rescale. Ambient reductions go through [`rgmin::vecops`].

use ndarray::{Array1, ArrayView1};
use rgmin::Manifold;
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
        self.apply_bdp(r, energy)
    }

    /// Riemannian BDP step: tangent Verlet, retract, transport, rescale.
    pub fn step_on<M: Manifold>(
        &mut self,
        man: &M,
        surface: &impl PointSurface,
        r: ArrayView1<f64>,
    ) -> Result<SamdReport, SaddleError> {
        if r.len() != self.x.len() {
            return Err(SaddleError::Shape(
                "SAMD Gaussian draw must match the 3N frame".into(),
            ));
        }
        if man.required_dim(self.x.len()).is_err() {
            return Err(SaddleError::Shape(
                "SAMD packing is not a legal manifold dim".into(),
            ));
        }
        let dt = self.config.dt;
        self.v = man.project(&self.x, &self.v);
        let old_g = man.egrad2rgrad(&self.x, &self.g);
        let y = retract_samd(man, &self.x, &self.v, &self.g, dt);
        let (energy, g_amb) = surface.eval(y.view())?;
        if !g_amb.iter().all(|a| a.is_finite()) {
            return Err(SaddleError::NonFinite("samd gradient"));
        }
        let g_r = man.egrad2rgrad(&y, &g_amb);
        self.v = transport_velocity(man, &self.x, &y, &self.v);
        let old_g_y = transport_velocity(man, &self.x, &y, &old_g);
        axpy(-0.5 * dt, g_r.view(), &mut self.v);
        axpy(-0.5 * dt, old_g_y.view(), &mut self.v);
        self.v = man.project(&y, &self.v);
        self.x = y;
        self.g = g_amb;
        self.energy = energy;
        let report = self.apply_bdp(r, energy)?;
        self.v = man.project(&self.x, &self.v);
        Ok(report)
    }

    fn apply_bdp(&mut self, r: ArrayView1<f64>, energy: f64) -> Result<SamdReport, SaddleError> {
        let d = self.x.len() as f64;
        let t = if self.config.exponential {
            t_exp(self.i, self.config.t0, self.config.tf, self.config.ngen)
        } else {
            t_linear(self.i, self.config.t0, self.config.tf, self.config.ngen)
        };
        let k_target = d * t / 2.0;
        let k = dot(self.v.view(), self.v.view()) / 2.0;
        if let Some(s) = bdp_scale(k, k_target, d, self.config.dt, self.config.tau, r) {
            self.v.mapv_inplace(|vi| vi * s);
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

/// Sella `samd.py` kinetic rescale. `None` leaves the velocity as-is.
fn bdp_scale(k: f64, k_target: f64, d: f64, dt: f64, tau: f64, r: ArrayView1<f64>) -> Option<f64> {
    if k <= 1e-12 {
        return None;
    }
    let edttau = (-dt / tau).exp();
    let edttau2 = (-dt / (2.0 * tau)).exp();
    let r2 = dot(r, r);
    let alpha2 = edttau
        + k * (1.0 - edttau) * r2 / (d * k)
        + 2.0 * edttau2 * (k_target * (1.0 - edttau) / (d * k)).sqrt() * r[0];
    if alpha2 > 0.0 && alpha2.is_finite() {
        Some(alpha2.sqrt())
    } else {
        None
    }
}

/// Project a SAMD increment onto a manifold (Sella waist: stay on set).
pub fn project_velocity<M: Manifold>(m: &M, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    m.project(x, v)
}

/// Verlet increment, project, retract. The arrival point stays on the set.
pub fn retract_samd<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    v: &Array1<f64>,
    egrad: &Array1<f64>,
    dt: f64,
) -> Array1<f64> {
    let g = man.egrad2rgrad(x, egrad);
    let mut s = Array1::zeros(v.len());
    axpy(dt, v.view(), &mut s);
    axpy(-0.5 * dt * dt, g.view(), &mut s);
    let s = man.project(x, &s);
    man.retract(x, &s)
}

/// Vector transport of a SAMD velocity from `x_from` to `x_to`.
pub fn transport_velocity<M: Manifold>(
    man: &M,
    x_from: &Array1<f64>,
    x_to: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    man.transport(x_from, x_to, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SaddleError;
    use crate::constraints::Constraints;
    use crate::internal::pack_cart;
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

    fn water() -> Array1<f64> {
        pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
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

    #[test]
    fn retract_samd_stays_on_the_sphere() {
        let x = Array1::from(vec![0.0, 1.0, 0.0]);
        let v = Array1::from(vec![0.3, 0.0, -0.1]);
        let g = Array1::from(vec![0.4, -0.2, 0.3]);
        let y = retract_samd(&ManifoldKind::Sphere, &x, &v, &g, 0.1);
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
        let vt = transport_velocity(&ManifoldKind::Sphere, &x, &y, &v);
        let vh = ManifoldKind::Sphere.project(&y, &vt);
        assert!(nrm2((&vt - &vh).view()) < 1e-12);
    }

    #[test]
    fn step_on_stays_on_the_sphere() {
        let x = Array1::from(vec![0.0, 1.0, 0.0]);
        let v = Array1::from(vec![0.3, 0.2, -0.1]);
        let mut sess = SamdSession::new(SamdConfig::default(), x, v, &Well).unwrap();
        let r = Array1::from(vec![0.2, 0.1, -0.05]);
        let report = sess
            .step_on(&ManifoldKind::Sphere, &Well, r.view())
            .unwrap();
        assert!(report.energy.is_finite());
        assert!(report.kinetic.is_finite());
        let n = nrm2(sess.position());
        assert!((n - 1.0).abs() < 1e-12, "SAMD step left the sphere: {n}");
        let vt = sess.velocity().to_owned();
        let y = sess.position().to_owned();
        let vh = ManifoldKind::Sphere.project(&y, &vt);
        assert!(nrm2((&vt - &vh).view()) < 1e-12);
    }

    #[test]
    fn step_on_stays_on_the_com_set() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        assert!(cons.residual_norm(x.view()).unwrap() < 1e-14);
        let v0 = Array1::from_elem(9, 0.2);
        let mut sess = SamdSession::new(SamdConfig::default(), x, v0, &Well).unwrap();
        let r = Array1::from_elem(9, 0.1);
        sess.step_on(&cons, &Well, r.view()).unwrap();
        let res = cons.residual_norm(sess.position()).unwrap();
        assert!(res < 1e-10, "SAMD step left the COM set: {res}");
        let vt = sess.velocity().to_owned();
        let y = sess.position().to_owned();
        let vh = cons.project(&y, &vt);
        assert!(nrm2((&vt - &vh).view()) < 1e-12);
    }
}
