//! Sella order-1 session: P-RFO + TrustRegion over a Cartesian or internals PES.
//!
//! `sella.optimize.optimize.Sella` with `order=1`, `method=prfo`.
//! Distinct from [`crate::minmode::MinModeSession`] (dimer / Lanczos
//! force inversion). Default geometry is the rigid quotient; pass a
//! [`Constraints`] chart through [`SellaSaddleSession::with_chart`],
//! or [`SellaSaddleSession::on_internal`] for Sella `InternalPES`.
//! Hosts own the loop.

use ndarray::Array1;
use rgmin::prfo_restricted;
use rgmin::vecops::{axpy, dot, vdot, vnrm2, Vector};
use rgmin::Manifold;

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::geom::{update_trust, SellaGeom, TrustSchedule};
use crate::minmode::PointSurface;
use crate::pes::CartesianPes;
use crate::pes_internal::{InternalPes, SellaPes};

pub struct SellaSaddleConfig {
    pub delta: f64,
    pub sigma_inc: f64,
    pub sigma_dec: f64,
    pub rho_inc: f64,
    pub rho_dec: f64,
    pub delta_min: f64,
    pub force_tol: f64,
    pub force_gate: crate::ForceGate,
    pub order: usize,
    /// Sella `eig=true`: periodic Rayleigh-Ritz on the free Hessian.
    pub eig: bool,
    /// Sella `nsteps_per_diag`.
    pub nsteps_per_diag: usize,
    /// Sella `gamma` residual scale. `<= 0` is exact eigh.
    pub gamma: f64,
    pub eigen_device: crate::EigenDevice,
    /// Sella `rayleigh_ritz(..., method=)`.
    pub expand: crate::ExpandKind,
}

impl SellaSaddleConfig {
    fn schedule(&self) -> TrustSchedule {
        TrustSchedule {
            sigma_inc: self.sigma_inc,
            sigma_dec: self.sigma_dec,
            rho_inc: self.rho_inc,
            rho_dec: self.rho_dec,
            delta_min: self.delta_min,
        }
    }
}

impl Default for SellaSaddleConfig {
    fn default() -> Self {
        let sch = TrustSchedule::saddle();
        Self {
            delta: 0.1,
            sigma_inc: sch.sigma_inc,
            sigma_dec: sch.sigma_dec,
            rho_inc: sch.rho_inc,
            rho_dec: sch.rho_dec,
            delta_min: sch.delta_min,
            force_tol: 1e-3,
            force_gate: crate::ForceGate::MaxForceOnAtom,
            order: 1,
            eig: true,
            nsteps_per_diag: 3,
            gamma: 0.1,
            eigen_device: crate::EigenDevice::Host,
            expand: crate::ExpandKind::Jd0,
        }
    }
}

pub struct SellaSaddleReport {
    pub energy: f64,
    pub max_force: f64,
    pub at_saddle: bool,
    pub rho: f64,
    pub delta: f64,
}

pub struct SellaSaddleSession {
    pes: SellaPes,
    config: SellaSaddleConfig,
    geom: SellaGeom,
    delta: f64,
    rho: f64,
    steps_since_diag: usize,
    ritz_v: Option<ndarray::Array2<f64>>,
}

impl SellaSaddleSession {
    pub fn new(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        let geom = SellaGeom::cartesian(x.len());
        Self::on(config, x, masses, geom)
    }

    /// Retract on a live equality chart instead of the rigid quotient.
    pub fn with_chart(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
    ) -> Result<Self, SaddleError> {
        Self::on(config, x, masses, SellaGeom::Chart(chart))
    }

    pub fn on(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        geom: SellaGeom,
    ) -> Result<Self, SaddleError> {
        let n_free = geom.n_free(x.len());
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::Cartesian(CartesianPes::new(x, masses)?),
            config,
            geom,
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
        })
    }

    /// P-RFO in the internals chart (Sella `InternalPES`).
    pub fn on_internal(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
    ) -> Result<Self, SaddleError> {
        let pes = InternalPes::new(x, masses, chart)?;
        let n_free = pes.n_int().max(1);
        let geom = SellaGeom::Chart(pes.chart().clone());
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::Internal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
        })
    }

    pub fn position(&self) -> ndarray::ArrayView1<'_, f64> {
        self.pes.position()
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn rho(&self) -> f64 {
        self.rho
    }

    pub fn geom(&self) -> &SellaGeom {
        &self.geom
    }

    /// Sella Hessian update (Cartesian or internals, matching the PES).
    pub fn set_update(&mut self, update: crate::HessUpdate) {
        self.pes.set_update(update);
    }

    pub fn internal_pes(&self) -> Option<&InternalPes> {
        self.pes.internal()
    }

    pub fn reset(&mut self) {
        self.pes.reset();
        let n_free = match &self.pes {
            SellaPes::Internal(p) => p.n_int().max(1),
            SellaPes::Cartesian(_) => self.geom.n_free(self.pes.position().len()),
        };
        self.delta = self.config.delta * n_free as f64;
        self.rho = 1.0;
        self.steps_since_diag = 0;
        self.ritz_v = None;
    }

    pub fn step<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        match &self.pes {
            SellaPes::Internal(_) => self.step_internal(surface),
            SellaPes::Cartesian(_) => self.step_cartesian(surface),
        }
    }

    fn diag_free(
        &mut self,
        h_free: &ndarray::Array2<f64>,
    ) -> Result<(ndarray::Array1<f64>, ndarray::Array2<f64>), SaddleError> {
        if self.config.eig && self.steps_since_diag >= self.config.nsteps_per_diag.max(1) {
            self.steps_since_diag = 0;
            let n = h_free.nrows();
            let v0 = self
                .ritz_v
                .clone()
                .filter(|v| v.nrows() == n)
                .unwrap_or_else(|| ndarray::Array2::eye(n));
            let (lams, vecs, _) = crate::rayleigh_ritz_iter(
                h_free.view(),
                v0.view(),
                self.config.gamma,
                self.config.expand,
            )?;
            self.ritz_v = Some(vecs.clone());
            Ok((lams, vecs))
        } else {
            self.steps_since_diag += 1;
            crate::eigh_on(self.config.eigen_device, h_free.view())
        }
    }

    fn step_cartesian<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let x;
        let energy;
        let g_r;
        let u;
        let g_free;
        let h_free;
        {
            let pes = match &self.pes {
                SellaPes::Cartesian(p) => p,
                SellaPes::Internal(_) => unreachable!(),
            };
            x = pes.position().to_owned();
            let (e, g) = surface.eval(x.view())?;
            energy = e;
            if !g.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("sella gradient"));
            }
            g_r = self.geom.egrad2rgrad(&x, &g);
            let max_force = self.config.force_gate.value(g_r.view());
            if max_force <= self.config.force_tol {
                return Ok(SellaSaddleReport {
                    energy,
                    max_force,
                    at_saddle: true,
                    rho: self.rho,
                    delta: self.delta,
                });
            }
            u = self.geom.ufree(&x);
            g_free = crate::geom::u_t_vec(&u, &g_r);
            h_free = crate::geom::u_t_h_u(&u, pes.hessian().hessian());
        }
        let vg = Vector::from_host(g_r.clone());
        let (evals, evecs) = self.diag_free(&h_free)?;
        let pes = match &mut self.pes {
            SellaPes::Cartesian(p) => p,
            SellaPes::Internal(_) => unreachable!(),
        };
        let s_free = prfo_restricted(
            &evals,
            &evecs,
            &g_free,
            self.config.order.max(1),
            self.delta,
        );
        let mut s = crate::geom::u_vec(&u, &s_free);
        s = self.geom.project(&x, &s);
        let sn = vnrm2(&Vector::from_host(s.clone()));
        if sn > self.delta && sn > 0.0 {
            s.mapv_inplace(|v| v * (self.delta / sn));
        }
        let vs = Vector::from_host(s.clone());
        let x1 = self.geom.retract(&x, &s);
        let s1 = self.geom.transport(&x, &x1, &s);
        if !s1.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella transport"));
        }
        let mut d = x1;
        axpy(-1.0, x.view(), &mut d);
        let e0 = energy;
        let hs = pes.hessian().hessian().dot(&s);
        let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
        pes.kick(surface, d.view())?;
        let x_new = pes.position().to_owned();
        let (energy, g1) = surface.eval(x_new.view())?;
        let g1_r = self.geom.egrad2rgrad(&x_new, &g1);
        let max_force = self.config.force_gate.value(g1_r.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(self.delta, self.rho, vnrm2(&vs), &self.config.schedule());
        } else {
            self.rho = 1.0;
        }
        Ok(SellaSaddleReport {
            energy,
            max_force,
            at_saddle: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
        })
    }

    fn step_internal<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let x;
        let energy;
        let g;
        let g_int;
        let h;
        {
            let pes = match &self.pes {
                SellaPes::Internal(p) => p,
                SellaPes::Cartesian(_) => unreachable!(),
            };
            x = pes.position().to_owned();
            let (e, gg) = surface.eval(x.view())?;
            energy = e;
            g = gg;
            if !g.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("sella gradient"));
            }
            let max_force = self.config.force_gate.value(g.view());
            if max_force <= self.config.force_tol {
                return Ok(SellaSaddleReport {
                    energy,
                    max_force,
                    at_saddle: true,
                    rho: self.rho,
                    delta: self.delta,
                });
            }
            g_int = pes.internals_grad(g.view())?;
            h = pes.hessian().hessian().to_owned();
        }
        let vg = Vector::from_host(g_int.clone());
        let (evals, evecs) = self.diag_free(&h)?;
        let pes = match &mut self.pes {
            SellaPes::Internal(p) => p,
            SellaPes::Cartesian(_) => unreachable!(),
        };
        let mut s = prfo_restricted(
            &evals,
            &evecs,
            &g_int,
            self.config.order.max(1),
            self.delta,
        );
        let sn = vnrm2(&Vector::from_host(s.clone()));
        if sn > self.delta && sn > 0.0 {
            s.mapv_inplace(|v| v * (self.delta / sn));
        }
        let vs = Vector::from_host(s.clone());
        let e0 = energy;
        let hs = h.dot(&s);
        let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
        let (energy, g1) = pes.kick(surface, s.view())?;
        let max_force = self.config.force_gate.value(g1.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(self.delta, self.rho, vnrm2(&vs), &self.config.schedule());
        } else {
            self.rho = 1.0;
        }
        Ok(SellaSaddleReport {
            energy,
            max_force,
            at_saddle: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
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
    use crate::constraints::Constraints;
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

    fn com(x: ArrayView1<f64>) -> [f64; 3] {
        let nat = x.len() / 3;
        let mut c = [0.0; 3];
        for i in 0..nat {
            c[0] += x[3 * i];
            c[1] += x[3 * i + 1];
            c[2] += x[3 * i + 2];
        }
        let n = nat as f64;
        [c[0] / n, c[1] / n, c[2] / n]
    }

    #[test]
    fn eig_rayleigh_ritz_step_is_finite() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaSaddleSession::new(
            SellaSaddleConfig {
                eig: true,
                nsteps_per_diag: 1,
                gamma: 0.1,
                ..SellaSaddleConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
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
        assert!(report.delta > 0.0);
    }

    #[test]
    fn trust_radius_starts_at_delta0_times_n_free() {
        let x = Array1::zeros(9);
        let sess = SellaSaddleSession::new(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
        )
        .unwrap();
        assert!(
            (sess.delta() - 0.3).abs() < 1e-14,
            "delta={}",
            sess.delta()
        );
    }

    #[test]
    fn prfo_step_stays_on_the_rigid_quotient() {
        use rgmin::vecops::nrm2;
        use rgmin::ManifoldKind;
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let mut sess = SellaSaddleSession::new(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
        )
        .unwrap();
        let x0 = sess.position().to_owned();
        let c0 = com(x0.view());
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        let x1 = sess.position().to_owned();
        let c1 = com(x1.view());
        assert!(
            (c0[0] - c1[0]).abs() < 1e-10
                && (c0[1] - c1[1]).abs() < 1e-10
                && (c0[2] - c1[2]).abs() < 1e-10,
            "COM drifted {c0:?} -> {c1:?}"
        );
        let d = &x1 - &x0;
        let d_h = ManifoldKind::RigidQuotient.project(&x0, &d);
        let err = nrm2((&d - &d_h).view());
        assert!(err < 1e-10, "step left the horizontal space: {err}");
    }

    #[test]
    fn internals_prfo_step_is_finite() {
        use crate::internal::{CartAxis, Translation};
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(Translation::all(2, CartAxis::X).unwrap(), x.view(), Some(0.0))
            .unwrap();
        let mut sess = SellaSaddleSession::on_internal(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
        )
        .unwrap();
        assert!(sess.internal_pes().is_some());
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn chart_step_keeps_a_fixed_com() {
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let mut chart = Constraints::new(3).unwrap();
        chart.fix_com(x.view()).unwrap();
        let mut sess = SellaSaddleSession::with_chart(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
            chart,
        )
        .unwrap();
        let x0 = sess.position().to_owned();
        let c0 = com(x0.view());
        let _ = sess.step(&Well).unwrap();
        let c1 = com(sess.position());
        assert!(
            (c0[0] - c1[0]).abs() < 1e-8
                && (c0[1] - c1[1]).abs() < 1e-8
                && (c0[2] - c1[2]).abs() < 1e-8,
            "COM drifted {c0:?} -> {c1:?}"
        );
    }
}
