//! Sella `samd.py`: Bussi-Donadio-Parrinello thermostat on a surface.
//!
//! One host-owned step: velocity-Verlet then the BDP kinetic rescale.
//! The host supplies the Gaussian draw `r` so the session stays
//! deterministic under test. Temperature schedules are the Sella
//! linear and exponential ramps.
//!
//! Default geometry is Euclidean (Sella's unconstrained `bdp`). A
//! host that needs the point on a set constructs [`SamdSession::on`]
//! or [`SamdSession::with_chart`]: the Verlet increment is
//! `egrad2rgrad` / `project` / `retract`, and the velocity is
//! transported. Ambient reductions go through [`rgmin::vecops`].

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::{axpy, dot};
use rgmin::{Manifold, ManifoldKind};

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::geom::SellaGeom;
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
    geom: SellaGeom,
    i: usize,
}

impl SamdSession {
    /// Euclidean SAMD: Sella `bdp` on the ambient frame.
    pub fn new(
        config: SamdConfig,
        x: Array1<f64>,
        v0: Array1<f64>,
        surface: &impl PointSurface,
    ) -> Result<Self, SaddleError> {
        Self::on(
            config,
            x,
            v0,
            surface,
            SellaGeom::Kind(ManifoldKind::Euclidean),
        )
    }

    /// Retract Verlet on a live equality chart.
    pub fn with_chart(
        config: SamdConfig,
        x: Array1<f64>,
        v0: Array1<f64>,
        surface: &impl PointSurface,
        chart: Constraints,
    ) -> Result<Self, SaddleError> {
        Self::on(config, x, v0, surface, SellaGeom::Chart(chart))
    }

    pub fn on(
        config: SamdConfig,
        x: Array1<f64>,
        v0: Array1<f64>,
        surface: &impl PointSurface,
        geom: SellaGeom,
    ) -> Result<Self, SaddleError> {
        if x.len() != v0.len() {
            return Err(SaddleError::Shape(
                "SAMD velocity must match the 3N frame".into(),
            ));
        }
        if geom.required_dim(x.len()).is_err() {
            return Err(SaddleError::Shape(
                "SAMD point is not a legal packing for the geometry".into(),
            ));
        }
        let (energy, g) = surface.eval(x.view())?;
        if !g.iter().all(|a| a.is_finite()) {
            return Err(SaddleError::NonFinite("samd gradient"));
        }
        let v = geom.project(&x, &v0);
        Ok(Self {
            x,
            v,
            g,
            energy,
            config,
            geom,
            i: 0,
        })
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn velocity(&self) -> ArrayView1<'_, f64> {
        self.v.view()
    }

    pub fn geom(&self) -> &SellaGeom {
        &self.geom
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
        let old_x = self.x.clone();
        let old_g = self.geom.egrad2rgrad(&old_x, &self.g);
        let y = retract_samd(&self.geom, &old_x, &self.v, &self.g, dt);
        let (energy, g) = surface.eval(y.view())?;
        if !g.iter().all(|a| a.is_finite()) {
            return Err(SaddleError::NonFinite("samd gradient"));
        }
        let g_r = self.geom.egrad2rgrad(&y, &g);
        // Split Verlet: half-step at T_x, transport, half-step at T_y.
        axpy(-0.5 * dt, old_g.view(), &mut self.v);
        self.v = self.geom.transport(&old_x, &y, &self.v);
        axpy(-0.5 * dt, g_r.view(), &mut self.v);
        self.v = self.geom.project(&y, &self.v);
        self.x = y;
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
        self.v = self.geom.project(&self.x, &self.v);
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
pub fn project_velocity<M: Manifold>(m: &M, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
    m.project(x, v)
}

/// Verlet increment: rgrad, `dt v - 0.5 dt^2 g`, project, retract.
pub fn retract_samd<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    v: &Array1<f64>,
    egrad: &Array1<f64>,
    dt: f64,
) -> Array1<f64> {
    let g = man.egrad2rgrad(x, egrad);
    let mut s = v.mapv(|vi| dt * vi);
    axpy(-0.5 * dt * dt, g.view(), &mut s);
    let s = man.project(x, &s);
    man.retract(x, &s)
}

/// Transport the SAMD velocity from `x` to the retracted point.
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
    fn t_exp_is_the_sella_ramp() {
        assert!((t_exp(0, 2.0, 0.5, 4) - 2.0).abs() < 1e-14);
        let mid = 2.0 * (0.5_f64 / 2.0).powf(0.5);
        assert!((t_exp(2, 2.0, 0.5, 4) - mid).abs() < 1e-14);
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
    fn samd_retract_stays_on_the_sphere() {
        let x = Array1::from(vec![0.0, 1.0, 0.0]);
        let v = Array1::from(vec![0.3, 0.2, -0.1]);
        let g = Array1::from(vec![0.4, 0.0, 0.0]);
        let y = retract_samd(&ManifoldKind::Sphere, &x, &v, &g, 0.1);
        let n = nrm2(y.view());
        assert!(
            (n - 1.0).abs() < 1e-14,
            "retracted point left the sphere: {n}"
        );
        let mut sess = SamdSession::on(
            SamdConfig::default(),
            x,
            v,
            &Well,
            SellaGeom::Kind(ManifoldKind::Sphere),
        )
        .unwrap();
        let r = Array1::from(vec![0.2, 0.1, -0.05]);
        sess.step(&Well, r.view()).unwrap();
        let n = nrm2(sess.position());
        assert!((n - 1.0).abs() < 1e-12, "SAMD step left the sphere: {n}");
        let vt = sess.velocity().to_owned();
        let vh = ManifoldKind::Sphere.project(&sess.position().to_owned(), &vt);
        assert!(nrm2((&vt - &vh).view()) < 1e-12);
    }

    #[test]
    fn samd_step_stays_on_the_com_set() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        assert!(cons.residual_norm(x.view()).unwrap() < 1e-14);
        let v0 = Array1::from_elem(9, 0.2);
        let mut sess =
            SamdSession::with_chart(SamdConfig::default(), x, v0, &Well, cons.clone()).unwrap();
        let r = Array1::from_elem(9, 0.1);
        sess.step(&Well, r.view()).unwrap();
        let res = cons.residual_norm(sess.position()).unwrap();
        assert!(res < 1e-10, "SAMD step left the COM set: {res}");
        let vt = sess.velocity().to_owned();
        let y = sess.position().to_owned();
        let vh = cons.project(&y, &vt);
        assert!(nrm2((&vt - &vh).view()) < 1e-12);
        let t = transport_velocity(&cons, &water(), &y, &vt);
        let th = cons.project(&y, &t);
        assert!(nrm2((&t - &th).view()) < 1e-12);
    }
}
