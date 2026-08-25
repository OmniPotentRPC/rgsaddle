//! Sella order-0 session: QN + TrustRegion over a Cartesian or internals PES.
//!
//! `sella.optimize.optimize.Sella` with `order=0`, `method=qn`,
//! `eig=false`. One step is eval, project, the restricted increment
//! (`qn_restricted`, or unrestricted QN then `ras_clip` for RAS), retract,
//! transport, `PES.kick`, then the `delta0` / `sigma` / `rho` trust
//! schedule. Default geometry is [`crate::geom::SellaGeom::cartesian`]
//! (RigidQuotient at N>=3). Pass a [`Constraints`] chart through
//! [`SellaMinSession::with_chart`] to retract on `ker(J)`, or
//! [`SellaMinSession::on_internal`] to QN in the internals chart
//! (Sella `InternalPES`). The host owns the loop. `run` is a convenience.

use ndarray::{s, Array1};
use rgmin::qn_restricted;
use rgmin::vecops::{axpy, dot, vdot, vnrm2, Vector};
use rgmin::Manifold;

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::geom::{update_trust, SellaGeom, TrustSchedule};
use crate::minmode::PointSurface;
use crate::pes::CartesianPes;
use crate::pes_internal::{CellCartesianPes, CellInternalPes, InternalPes, SellaPes};

/// Sella `_default_kwargs['minimum']` plus a force gate.
pub struct SellaMinConfig {
    /// Sella `delta0`. The living trust radius starts here.
    pub delta: f64,
    pub sigma_inc: f64,
    pub sigma_dec: f64,
    pub rho_inc: f64,
    pub rho_dec: f64,
    pub delta_min: f64,
    pub force_tol: f64,
    /// eOn / gpr_optim `ConvergenceForceNorm`.
    pub force_gate: crate::ForceGate,
    /// Restricted step: TrustRegion, RestrictedAtomicStep (Cartesian),
    /// MaxInternalStep (internals). RAS refuses internals.
    pub restricted: crate::RestrictedKind,
}

impl SellaMinConfig {
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

impl Default for SellaMinConfig {
    fn default() -> Self {
        let sch = TrustSchedule::minimum();
        Self {
            delta: 1e-1,
            sigma_inc: sch.sigma_inc,
            sigma_dec: sch.sigma_dec,
            rho_inc: sch.rho_inc,
            rho_dec: sch.rho_dec,
            delta_min: sch.delta_min,
            force_tol: 1e-3,
            force_gate: crate::ForceGate::MaxForceOnAtom,
            restricted: crate::RestrictedKind::TrustRegion,
        }
    }
}

pub struct SellaMinReport {
    pub energy: f64,
    pub max_force: f64,
    pub at_minimum: bool,
    pub rho: f64,
    pub delta: f64,
}

pub struct SellaMinSession {
    pes: SellaPes,
    config: SellaMinConfig,
    geom: SellaGeom,
    delta: f64,
    rho: f64,
}

impl SellaMinSession {
    pub fn new(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        let geom = SellaGeom::cartesian(x.len());
        Self::on(config, x, masses, geom)
    }

    /// Retract on a live equality chart instead of the rigid quotient.
    pub fn with_chart(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
    ) -> Result<Self, SaddleError> {
        Self::on(config, x, masses, SellaGeom::Chart(chart))
    }

    pub fn on(
        config: SellaMinConfig,
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
        })
    }

    /// QN in the internals chart (Sella `InternalPES`).
    ///
    /// `delta0` is scaled by the number of internals. The force gate
    /// still reads the Cartesian gradient.
    pub fn on_internal(
        config: SellaMinConfig,
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
        })
    }

    /// QN in internals with Sella dummy 3-vectors on the Wilson frame.
    pub fn on_internal_dummies(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        dummies: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        let pes = crate::InternalPes::with_dummies(x, masses, chart, dummies)?;
        let n_free = pes.n_int().max(1);
        let geom = SellaGeom::Chart(pes.chart().clone());
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::Internal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
        })
    }

    /// QN in packed `[x_cart; cell_params]` (Sella `CellCartesianPES`).
    ///
    /// Cell DOF are the mask-true lattice entries. The force gate
    /// still reads the Cartesian gradient.
    pub fn on_cell(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        cell: crate::Cell,
        mask: [bool; 9],
    ) -> Result<Self, SaddleError> {
        let mut pes = CellCartesianPes::new(x, masses, cell)?;
        pes.set_mask(mask);
        let n_free = pes.packed_len().max(1);
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::Cell(pes),
            config,
            geom: SellaGeom::Kind(rgmin::ManifoldKind::Euclidean),
            delta,
            rho: 1.0,
        })
    }

    /// QN on the Sella log-deformation cell chart.
    pub fn on_cell_log(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        cell: crate::Cell,
        mask: [bool; 9],
    ) -> Result<Self, SaddleError> {
        let mut pes = CellCartesianPes::new(x, masses, cell)?;
        pes.set_mask(mask);
        pes.set_chart(crate::CellChart::LogDeform);
        let n_free = pes.packed_len().max(1);
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::Cell(pes),
            config,
            geom: SellaGeom::Kind(rgmin::ManifoldKind::Euclidean),
            delta,
            rho: 1.0,
        })
    }

    /// QN in packed `[q_int; cell_params]` (Sella `CellInternalPES`).
    pub fn on_cell_internal(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        cell: crate::Cell,
        mask: [bool; 9],
    ) -> Result<Self, SaddleError> {
        let mut pes = CellInternalPes::new(x, masses, chart, cell)?;
        pes.set_mask(mask);
        let n_free = pes.packed_len().max(1);
        let geom = SellaGeom::Chart(pes.internals().chart().clone());
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::CellInternal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
        })
    }

    /// QN on packed `[q_int; L]` (Sella `CellInternalPES` log chart).
    pub fn on_cell_internal_log(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        cell: crate::Cell,
        mask: [bool; 9],
    ) -> Result<Self, SaddleError> {
        let mut pes = CellInternalPes::new(x, masses, chart, cell)?;
        pes.set_mask(mask);
        pes.set_chart(crate::CellChart::LogDeform);
        let n_free = pes.packed_len().max(1);
        let geom = SellaGeom::Chart(pes.internals().chart().clone());
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: SellaPes::CellInternal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
        })
    }

    pub fn geom(&self) -> &SellaGeom {
        &self.geom
    }

    /// Sella Hessian update (Cartesian or internals, matching the PES).
    pub fn set_update(&mut self, update: crate::HessUpdate) {
        self.pes.set_update(update);
    }

    /// Restricted increment. RAS clips per-atom Cartesian steps.
    /// MaxInternalStep clips internals. TrustRegion is `||s||`.
    pub fn set_restricted(&mut self, kind: crate::RestrictedKind) {
        self.config.restricted = kind;
    }

    /// Switch the cell chart. No-op on a non-cell session.
    pub fn set_cell_chart(&mut self, kind: crate::CellChart) {
        match &mut self.pes {
            SellaPes::Cell(p) => {
                p.set_chart(kind);
                let n = p.packed_len().max(1);
                self.delta = self.config.delta * n as f64;
            }
            SellaPes::CellInternal(p) => {
                p.set_chart(kind);
                let n = p.packed_len().max(1);
                self.delta = self.config.delta * n as f64;
            }
            SellaPes::Cartesian(_) | SellaPes::Internal(_) => {}
        }
    }

    pub fn position(&self) -> ndarray::ArrayView1<'_, f64> {
        self.pes.position()
    }

    /// Internals PES when the session was built with [`Self::on_internal`].
    pub fn internal_pes(&self) -> Option<&InternalPes> {
        self.pes.internal()
    }

    /// Cell PES when the session was built with [`Self::on_cell`].
    pub fn cell_pes(&self) -> Option<&CellCartesianPes> {
        self.pes.cell()
    }

    /// Cell+internals PES when the session was built with [`Self::on_cell_internal`].
    pub fn cell_internal_pes(&self) -> Option<&CellInternalPes> {
        self.pes.cell_internal()
    }

    /// Niggli-reduce a cell session. Cartesian and internals sessions
    /// return `Ok(false)`.
    pub fn maybe_niggli_reduce(&mut self, angle_threshold: f64) -> Result<bool, SaddleError> {
        match &mut self.pes {
            SellaPes::Cell(p) => p.maybe_niggli_reduce(angle_threshold),
            SellaPes::CellInternal(p) => p.maybe_niggli_reduce(angle_threshold),
            SellaPes::Cartesian(_) | SellaPes::Internal(_) => Ok(false),
        }
    }

    /// Living Sella trust radius.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Last accepted `df_actual / df_pred`. `1` when the model is mute.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    pub fn reset(&mut self) {
        self.pes.reset();
        let n_free = match &self.pes {
            SellaPes::Internal(p) => p.n_int().max(1),
            SellaPes::Cell(p) => p.packed_len().max(1),
            SellaPes::CellInternal(p) => p.packed_len().max(1),
            SellaPes::Cartesian(_) => self.geom.n_free(self.pes.position().len()),
        };
        self.delta = self.config.delta * n_free as f64;
        self.rho = 1.0;
    }

    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<SellaMinReport, SaddleError> {
        match &self.pes {
            SellaPes::Internal(_) => self.step_internal(surface),
            SellaPes::Cell(_) => self.step_cell(surface),
            SellaPes::CellInternal(_) => self.step_cell_internal(surface),
            SellaPes::Cartesian(_) => self.step_cartesian(surface),
        }
    }

    fn step_cartesian<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaMinReport, SaddleError> {
        let pes = match &mut self.pes {
            SellaPes::Cartesian(p) => p,
            SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => unreachable!(),
        };
        let x = pes.position().to_owned();
        let (energy, g) = surface.eval(x.view())?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let g_r = self.geom.egrad2rgrad(&x, &g);
        let vg = Vector::from_host(g_r.clone());
        let max_force = self.config.force_gate.value(g_r.view());
        if max_force <= self.config.force_tol {
            return Ok(SellaMinReport {
                energy,
                max_force,
                at_minimum: true,
                rho: self.rho,
                delta: self.delta,
            });
        }
        let u = self.geom.ufree(&x);
        let g_free = crate::geom::u_t_vec(&u, &g_r);
        let h_free = crate::geom::u_t_h_u(&u, pes.hessian().hessian());
        let (evals, evecs) = crate::exact_eigh(h_free.view())?;
        let s_free = self
            .config
            .restricted
            .qn_step(&evals, &evecs, &g_free, 0, self.delta);
        let mut s = crate::geom::u_vec(&u, &s_free);
        s = self.geom.project(&x, &s);
        s = self.config.restricted.clip_cartesian(&s, self.delta)?;
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
        let (energy, g1) = pes.kick(surface, d.view())?;
        let x_new = pes.position().to_owned();
        let g1_r = self.geom.egrad2rgrad(&x_new, &g1);
        let max_force = self.config.force_gate.value(g1_r.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(self.delta, self.rho, vnrm2(&vs), &self.config.schedule());
        } else {
            self.rho = 1.0;
        }
        Ok(SellaMinReport {
            energy,
            max_force,
            at_minimum: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
        })
    }

    fn step_internal<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaMinReport, SaddleError> {
        let pes = match &mut self.pes {
            SellaPes::Internal(p) => p,
            SellaPes::Cartesian(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                unreachable!()
            }
        };
        let x = pes.position().to_owned();
        let (energy, g) = surface.eval(x.view())?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let max_force = self.config.force_gate.value(g.view());
        if max_force <= self.config.force_tol {
            return Ok(SellaMinReport {
                energy,
                max_force,
                at_minimum: true,
                rho: self.rho,
                delta: self.delta,
            });
        }
        self.config.restricted.refuse_internal()?;
        let g_int = pes.internals_grad(g.view())?;
        let vg = Vector::from_host(g_int.clone());
        let h = pes.hessian().hessian();
        let (evals, evecs) = crate::exact_eigh(h.view())?;
        let mut s = qn_restricted(&evals, &evecs, &g_int, 0, self.delta);
        if self.config.restricted == crate::RestrictedKind::MaxInternalStep {
            let w = crate::restricted::weights_for_equalities(pes.chart());
            s = crate::mis_clip(&s, &w, self.delta);
        }
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
        Ok(SellaMinReport {
            energy,
            max_force,
            at_minimum: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
        })
    }

    fn step_cell<S: PointSurface>(&mut self, surface: &S) -> Result<SellaMinReport, SaddleError> {
        let pes = match &mut self.pes {
            SellaPes::Cell(p) => p,
            SellaPes::Cartesian(_) | SellaPes::Internal(_) | SellaPes::CellInternal(_) => {
                unreachable!()
            }
        };
        let x = pes.position().to_owned();
        let c9 = pes.cell9();
        let (energy, g) = surface.eval_in_cell(x.view(), &c9)?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let max_force = self.config.force_gate.value(g.view());
        if max_force <= self.config.force_tol {
            return Ok(SellaMinReport {
                energy,
                max_force,
                at_minimum: true,
                rho: self.rho,
                delta: self.delta,
            });
        }
        let g_p = pes.packed_grad(surface, g.view())?;
        let vg = Vector::from_host(g_p.clone());
        let h = pes.hessian().hessian();
        let (evals, evecs) = crate::exact_eigh(h.view())?;
        let mut s = self
            .config
            .restricted
            .qn_step(&evals, &evecs, &g_p, 0, self.delta);
        s = self.config.restricted.clip_cartesian(&s, self.delta)?;
        let vs = Vector::from_host(s.clone());
        let e0 = energy;
        let hs = h.dot(&s);
        let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
        let (energy, g1p) = pes.kick_packed(surface, s.view())?;
        let n = x.len();
        let g1_cart = g1p.slice(s![..n]).to_owned();
        let max_force = self.config.force_gate.value(g1_cart.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(self.delta, self.rho, vnrm2(&vs), &self.config.schedule());
        } else {
            self.rho = 1.0;
        }
        Ok(SellaMinReport {
            energy,
            max_force,
            at_minimum: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
        })
    }

    fn step_cell_internal<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaMinReport, SaddleError> {
        let pes = match &mut self.pes {
            SellaPes::CellInternal(p) => p,
            _ => unreachable!(),
        };
        let x = pes.position().to_owned();
        let c9 = pes.cell9();
        let (energy, g) = surface.eval_in_cell(x.view(), &c9)?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let max_force = self.config.force_gate.value(g.view());
        if max_force <= self.config.force_tol {
            return Ok(SellaMinReport {
                energy,
                max_force,
                at_minimum: true,
                rho: self.rho,
                delta: self.delta,
            });
        }
        self.config.restricted.refuse_internal()?;
        let g_p = pes.packed_grad(surface, g.view())?;
        let vg = Vector::from_host(g_p.clone());
        let h = pes.hessian().hessian();
        let (evals, evecs) = crate::exact_eigh(h.view())?;
        let mut s = qn_restricted(&evals, &evecs, &g_p, 0, self.delta);
        if self.config.restricted == crate::RestrictedKind::MaxInternalStep {
            // Clip internals block only.
            let nint = pes.internals().n_int();
            let w = crate::restricted::weights_for_equalities(pes.internals().chart());
            let dq = s.slice(s![..nint]).to_owned();
            let clipped = crate::mis_clip(&dq, &w, self.delta);
            for i in 0..nint {
                s[i] = clipped[i];
            }
        }
        let sn = vnrm2(&Vector::from_host(s.clone()));
        if sn > self.delta && sn > 0.0 {
            s.mapv_inplace(|v| v * (self.delta / sn));
        }
        let vs = Vector::from_host(s.clone());
        let e0 = energy;
        let hs = h.dot(&s);
        let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
        let (energy, _) = pes.kick_packed(surface, s.view())?;
        let (_, g1) = surface.eval_in_cell(pes.position(), &pes.cell9())?;
        let max_force = self.config.force_gate.value(g1.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(self.delta, self.rho, vnrm2(&vs), &self.config.schedule());
        } else {
            self.rho = 1.0;
        }
        Ok(SellaMinReport {
            energy,
            max_force,
            at_minimum: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
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
    use crate::geom::update_trust;
    use crate::minmode::PointSurface;
    use ndarray::{Array1, ArrayView1};
    use rgmin::vecops::nrm2;
    use rgmin::ManifoldKind;

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
    fn qn_trust_reaches_the_well() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        let report = sess.run(&Well, 40).unwrap();
        assert!(report.at_minimum, "force {}", report.max_force);
        assert!((sess.position()[0].abs() - 1.0).abs() < 0.25);
        assert!(report.rho.is_finite());
        assert!(report.delta > 0.0);
    }

    #[test]
    fn ras_session_step_differs_from_trust_region() {
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let masses = Array1::from(vec![1.0, 1.0, 1.0]);
        let mut tr = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.01,
                restricted: crate::RestrictedKind::TrustRegion,
                ..SellaMinConfig::default()
            },
            x.clone(),
            masses.clone(),
        )
        .unwrap();
        let mut ras = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.01,
                restricted: crate::RestrictedKind::RestrictedAtomicStep,
                ..SellaMinConfig::default()
            },
            x,
            masses,
        )
        .unwrap();
        let r_tr = tr.step(&Well).unwrap();
        let r_ras = ras.step(&Well).unwrap();
        assert!(r_tr.energy.is_finite());
        assert!(r_ras.energy.is_finite());
        let dx = nrm2((&ras.position().to_owned() - &tr.position().to_owned()).view());
        assert!(
            dx > 1e-8,
            "RAS session step matched TrustRegion: dx={dx} x_tr={:?} x_ras={:?}",
            tr.position(),
            ras.position()
        );
    }

    #[test]
    fn ras_step_stays_on_the_rigid_quotient() {
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let mut sess = SellaMinSession::new(
            SellaMinConfig {
                restricted: crate::RestrictedKind::RestrictedAtomicStep,
                ..SellaMinConfig::default()
            },
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
    fn qn_step_stays_on_the_rigid_quotient() {
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let mut sess = SellaMinSession::new(
            SellaMinConfig::default(),
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
        let v = ManifoldKind::RigidQuotient.project(&x0, &Array1::from_elem(9, 0.1));
        let vt = ManifoldKind::RigidQuotient.transport(&x0, &x1, &v);
        let vt_h = ManifoldKind::RigidQuotient.project(&x1, &vt);
        assert!(nrm2((&vt - &vt_h).view()) < 1e-10);
    }

    #[test]
    fn qn_trust_reaches_the_well_with_unequal_masses() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 16.0]),
        )
        .unwrap();
        let report = sess.run(&Well, 40).unwrap();
        assert!(report.at_minimum, "force {}", report.max_force);
        assert!((sess.position()[0].abs() - 1.0).abs() < 0.25);
    }

    #[test]
    fn trust_radius_starts_at_delta0_times_n_free() {
        let x = Array1::zeros(9);
        let sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.1,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
        )
        .unwrap();
        assert!((sess.delta() - 0.3).abs() < 1e-14, "delta={}", sess.delta());
    }

    #[test]
    fn atom_fmax_is_the_largest_per_atom_norm() {
        let mut g = Array1::zeros(6);
        g[0] = 0.0008;
        g[1] = 0.0008;
        g[2] = 0.0008;
        let n = crate::ForceGate::MaxForceOnAtom.value(g.view());
        assert!(n > 1e-3, "component inf-norm would pass 1e-3; atom |F|={n}");
        assert!((n - (3.0_f64 * 0.0008 * 0.0008).sqrt()).abs() < 1e-14);
        let linf = crate::ForceGate::LinfNorm.value(g.view());
        assert!(linf < 1e-3);
    }

    #[test]
    fn update_trust_shrinks_on_a_bad_rho() {
        let cfg = SellaMinConfig::default();
        let shrunk = update_trust(0.2, 0.0, 0.2, &cfg.schedule());
        assert!(shrunk < 0.2, "bad rho must shrink: {shrunk}");
        assert!((shrunk - 0.2 * cfg.sigma_dec).abs() < 1e-14);
        let grown = update_trust(0.2, 1.0, 0.2, &cfg.schedule());
        assert!(grown > 0.2, "good rho must grow: {grown}");
        assert!((grown - cfg.sigma_inc * 0.2).abs() < 1e-14);
    }

    #[test]
    fn internals_qn_reaches_the_well() {
        use crate::internal::{CartAxis, Translation};
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(
                Translation::all(2, CartAxis::X).unwrap(),
                x.view(),
                Some(0.0),
            )
            .unwrap();
        let mut sess = SellaMinSession::on_internal(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
        )
        .unwrap();
        assert!(sess.internal_pes().is_some());
        assert!((sess.delta() - 0.2).abs() < 1e-14);
        let report = sess.run(&Well, 40).unwrap();
        assert!(report.at_minimum, "force {}", report.max_force);
        assert!((sess.position()[0].abs() - 1.0).abs() < 0.35);
    }

    struct CellQuad;
    impl PointSurface for CellQuad {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let mut g = Array1::zeros(x.len());
            g[0] = 2.0 * x[0];
            Ok((x[0] * x[0], g))
        }
        fn eval_in_cell(
            &self,
            x: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let (e, g) = self.eval(x)?;
            Ok((e + (cell[0] - 2.0) * (cell[0] - 2.0), g))
        }
        fn cell_grad(
            &self,
            _x: ArrayView1<f64>,
            cell: &[f64; 9],
        ) -> Result<Option<[f64; 9]>, SaddleError> {
            let mut g = [0.0; 9];
            g[0] = 2.0 * (cell[0] - 2.0);
            Ok(Some(g))
        }
    }

    #[test]
    fn cartesian_niggli_is_a_noop() {
        let mut sess = SellaMinSession::new(
            SellaMinConfig::default(),
            Array1::zeros(6),
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        assert!(!sess.maybe_niggli_reduce(20.0).unwrap());
    }

    #[test]
    fn cell_qn_moves_the_lattice() {
        let mut x = Array1::zeros(6);
        x[0] = 0.3;
        let cell = crate::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaMinSession::on_cell(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            cell,
            mask,
        )
        .unwrap();
        assert!(sess.cell_pes().is_some());
        assert_eq!(sess.cell_pes().unwrap().n_cell_dof(), 1);
        let a00 = sess.cell_pes().unwrap().cell9()[0];
        assert!((a00 - 3.0).abs() < 1e-12);
        let _ = sess.run(&CellQuad, 30).unwrap();
        let a00 = sess.cell_pes().unwrap().cell9()[0];
        assert!(
            (a00 - 2.0).abs() < 0.35,
            "lattice a00 should walk toward 2, got {a00}"
        );
    }

    #[test]
    fn log_cell_qn_moves_the_lattice() {
        let mut x = Array1::zeros(6);
        x[0] = 0.3;
        let cell = crate::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaMinSession::on_cell_log(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            cell,
            mask,
        )
        .unwrap();
        assert_eq!(
            sess.cell_pes().unwrap().chart(),
            crate::CellChart::LogDeform
        );
        let _ = sess.run(&CellQuad, 40).unwrap();
        let a00 = sess.cell_pes().unwrap().cell9()[0];
        assert!(
            (a00 - 2.0).abs() < 0.5,
            "log-cell a00 should walk toward 2, got {a00}"
        );
    }

    #[test]
    fn cell_internal_qn_is_finite() {
        use crate::internal::{CartAxis, Translation};
        let mut x = Array1::zeros(6);
        x[0] = 0.3;
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(
                Translation::all(2, CartAxis::X).unwrap(),
                x.view(),
                Some(0.0),
            )
            .unwrap();
        let cell = crate::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaMinSession::on_cell_internal(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
            cell,
            mask,
        )
        .unwrap();
        assert!(sess.cell_internal_pes().is_some());
        let report = sess.step(&CellQuad).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cell_internal_log_qn_is_finite() {
        use crate::internal::{CartAxis, Translation};
        let mut x = Array1::zeros(6);
        x[0] = 0.3;
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(
                Translation::all(2, CartAxis::X).unwrap(),
                x.view(),
                Some(0.0),
            )
            .unwrap();
        let cell = crate::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaMinSession::on_cell_internal_log(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
            cell,
            mask,
        )
        .unwrap();
        assert_eq!(
            sess.cell_internal_pes().unwrap().chart(),
            crate::CellChart::LogDeform
        );
        let report = sess.step(&CellQuad).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn internals_mis_clip_stays_finite() {
        use crate::internal::{CartAxis, Translation};
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(
                Translation::all(2, CartAxis::X).unwrap(),
                x.view(),
                Some(0.0),
            )
            .unwrap();
        let mut sess = SellaMinSession::on_internal(
            SellaMinConfig {
                delta: 0.05,
                restricted: crate::RestrictedKind::MaxInternalStep,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
        )
        .unwrap();
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn internals_ras_is_refused() {
        use crate::internal::{CartAxis, Translation};
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut chart = Constraints::new(2).unwrap();
        chart
            .fix_translation(
                Translation::all(2, CartAxis::X).unwrap(),
                x.view(),
                Some(0.0),
            )
            .unwrap();
        let mut sess = SellaMinSession::on_internal(
            SellaMinConfig {
                delta: 0.05,
                restricted: crate::RestrictedKind::RestrictedAtomicStep,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
        )
        .unwrap();
        match sess.step(&Well) {
            Err(SaddleError::Shape(msg)) => {
                assert!(msg.contains("Internal coordinates"));
            }
            Ok(_) => panic!("RAS on internals must refuse"),
            Err(other) => panic!("{other}"),
        }
    }

    #[test]
    fn chart_step_keeps_a_fixed_com() {
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let mut chart = Constraints::new(3).unwrap();
        chart.fix_com(x.view()).unwrap();
        let mut sess = SellaMinSession::with_chart(
            SellaMinConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
            chart,
        )
        .unwrap();
        assert_eq!(sess.geom().n_free(9), 6);
        let x0 = sess.position().to_owned();
        let c0 = com(x0.view());
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        let x1 = sess.position().to_owned();
        let c1 = com(x1.view());
        assert!(
            (c0[0] - c1[0]).abs() < 1e-8
                && (c0[1] - c1[1]).abs() < 1e-8
                && (c0[2] - c1[2]).abs() < 1e-8,
            "COM drifted {c0:?} -> {c1:?}"
        );
    }
}
