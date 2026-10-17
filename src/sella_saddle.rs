//! Sella order-1 session: P-RFO + TrustRegion over a Cartesian or internals PES.
//!
//! `sella.optimize.optimize.Sella` with `order=1`, `method=prfo`.
//! Distinct from [`crate::minmode::MinModeSession`] (dimer / Lanczos
//! force inversion). Default geometry is the rigid quotient; pass a
//! [`Constraints`] chart through [`SellaSaddleSession::with_chart`],
//! or [`SellaSaddleSession::on_internal`] for Sella `InternalPES`,
//! or [`SellaSaddleSession::on_cell_internal`] for `CellInternalPES`.
//! Hosts own the loop.

use crate::prfo::prfo_trust_region;
use ndarray::{Array1, s};
use rgmin::Manifold;
use rgmin::vecops::{Vector, axpy, dot, vdot, vnrm2};

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::geom::{SellaGeom, TrustSchedule, update_trust};
use crate::minmode::PointSurface;
use crate::pes::CartesianPes;
use crate::pes_internal::{CellCartesianPes, CellInternalPes, InternalPes, SellaPes};

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
    /// Sella `gamma` residual scale. `<= 0` skips the finite-difference
    /// diagonalisation and factors the quasi-Newton model exactly.
    pub gamma: f64,
    /// Sella `eta`, the finite-difference step of a true Hessian-vector
    /// product. `<= 0` keeps `1e-4`.
    pub eta: f64,
    /// Sella `threepoint`. False is the one-sided product.
    pub threepoint: bool,
    pub eigen_device: crate::EigenDevice,
    /// Sella `rayleigh_ritz(..., method=)`.
    pub expand: crate::ExpandKind,
    /// Restricted step: TrustRegion, RestrictedAtomicStep (Cartesian),
    /// MaxInternalStep (internals). RAS refuses internals.
    pub restricted: crate::RestrictedKind,
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
            eta: 1e-4,
            threepoint: false,
            eigen_device: crate::EigenDevice::Host,
            expand: crate::ExpandKind::Jd0,
            restricted: crate::RestrictedKind::TrustRegion,
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

struct CurvatureFrame<'a> {
    x: &'a ndarray::Array1<f64>,
    gradient: &'a ndarray::Array1<f64>,
    u: &'a ndarray::Array2<f64>,
    h_free: &'a ndarray::Array2<f64>,
    g_free: &'a ndarray::Array1<f64>,
    first: bool,
}

pub struct SellaSaddleSession {
    pes: SellaPes,
    config: SellaSaddleConfig,
    geom: SellaGeom,
    delta: f64,
    rho: f64,
    steps_since_diag: isize,
    /// The opening finite-difference diagonalisation has run.
    curvature_initialized: bool,
    ritz_v: Option<ndarray::Array2<f64>>,
    /// The trust radius was scaled to the deflated free dimension once.
    free_matched: bool,
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
            curvature_initialized: false,
            ritz_v: None,
            free_matched: false,
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
            curvature_initialized: false,
            ritz_v: None,
            free_matched: false,
        })
    }

    /// P-RFO in packed `[x_cart; cell_params]`.
    pub fn on_cell(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        cell: crate::mic::Cell,
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
            steps_since_diag: 0,
            curvature_initialized: false,
            ritz_v: None,
            free_matched: false,
        })
    }

    /// P-RFO in packed `[q_int; cell_params]` (Sella `CellInternalPES`).
    pub fn on_cell_internal(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        cell: crate::mic::Cell,
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
            steps_since_diag: 0,
            curvature_initialized: false,
            ritz_v: None,
            free_matched: false,
        })
    }

    /// P-RFO on packed `[q_int; L]`.
    pub fn on_cell_internal_log(
        config: SellaSaddleConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
        chart: Constraints,
        cell: crate::mic::Cell,
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
            steps_since_diag: 0,
            curvature_initialized: false,
            ritz_v: None,
            free_matched: false,
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

    /// Seed the Cartesian Hessian with a selected mode and curvature.
    ///
    /// A preceding dimer or Lanczos search can hand its unstable direction to
    /// P-RFO without rebuilding the saddle mechanics in the host. The mode is
    /// projected onto the session's live tangent space before it enters the
    /// quasi-Newton model. Internal and variable-cell sessions use their own
    /// coordinate Hessians and reject a Cartesian seed.
    pub fn seed_mode(
        &mut self,
        mode: ndarray::ArrayView1<f64>,
        curvature: f64,
    ) -> Result<(), SaddleError> {
        if mode.len() != self.pes.position().len() {
            return Err(SaddleError::Shape(
                "seed mode must match the session position".into(),
            ));
        }
        let x = self.pes.position().to_owned();
        let projected = self.geom.project(&x, &mode.to_owned());
        match &mut self.pes {
            SellaPes::Cartesian(pes) => pes.seed_mode(projected.view(), curvature),
            SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => Err(
                SaddleError::Shape("Cartesian mode seed requires a Cartesian session".into()),
            ),
        }
    }

    /// Sella `rayleigh_ritz(..., method=)`.
    pub fn set_expand(&mut self, expand: crate::ExpandKind) {
        self.config.expand = expand;
    }

    /// Restricted increment. RAS clips per-atom Cartesian steps.
    /// MaxInternalStep clips internals. TrustRegion is `||s||`.
    pub fn set_restricted(&mut self, kind: crate::RestrictedKind) {
        self.config.restricted = kind;
    }

    pub fn internal_pes(&self) -> Option<&InternalPes> {
        self.pes.internal()
    }

    pub fn cell_pes(&self) -> Option<&CellCartesianPes> {
        self.pes.cell()
    }

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
        self.steps_since_diag = 0;
        self.curvature_initialized = false;
        self.ritz_v = None;
        self.free_matched = false;
    }

    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<SellaSaddleReport, SaddleError> {
        match &self.pes {
            SellaPes::Internal(_) => self.step_internal(surface),
            SellaPes::Cell(_) => self.step_cell(surface),
            SellaPes::CellInternal(_) => self.step_cell_internal(surface),
            SellaPes::Cartesian(_) => self.step_cartesian(surface),
        }
    }

    fn wants_true_diag(&self) -> bool {
        self.config.eig && self.config.gamma > 0.0 && matches!(self.pes, SellaPes::Cartesian(_))
    }

    fn finite_difference_step(&self) -> f64 {
        if self.config.eta.is_finite() && self.config.eta > 0.0 {
            self.config.eta
        } else {
            1e-4
        }
    }

    fn cartesian_free_hessian(
        &self,
        u: &ndarray::Array2<f64>,
    ) -> Result<ndarray::Array2<f64>, SaddleError> {
        let SellaPes::Cartesian(pes) = &self.pes else {
            return Err(SaddleError::Shape(
                "Cartesian Hessian requires a Cartesian session".into(),
            ));
        };
        Ok(crate::geom::u_t_h_u(u, pes.hessian().hessian()))
    }

    /// Finite-difference Rayleigh-Ritz of the true Hessian, then a
    /// multi-secant update of the quasi-Newton model.
    ///
    /// Sella `PES.diag`: the operator is the projected numerical Hessian,
    /// the preconditioner is the current approximate Hessian, and the
    /// collected secants update that model before the next P-RFO step.
    /// The first seed is the normalized free gradient. A later seed is
    /// the negative eigenspace of the model, which is where the previous
    /// update planted the unstable mode.
    fn refresh_true_curvature<S: PointSurface>(
        &mut self,
        surface: &S,
        frame: &CurvatureFrame<'_>,
    ) -> Result<(), SaddleError> {
        use crate::linalg::NumericalHessian;
        use crate::qn::update_h_ms;

        let x = frame.x;
        let gradient = frame.gradient;
        let u = frame.u;
        let h_free = frame.h_free;
        let g_free = frame.g_free;
        let nfree = u.ncols();
        if nfree == 0 {
            return Ok(());
        }
        let mut numerical = NumericalHessian::new(
            x.clone(),
            gradient.clone(),
            self.finite_difference_step(),
            self.config.threepoint,
            Some(u.clone()),
        )?;
        let seed = if frame.first {
            free_ritz_seed(nfree, g_free, None)
        } else {
            let (lams, vecs) = crate::eigensolve::exact_eigh(h_free.view())?;
            let kept = lams
                .iter()
                .filter(|value| **value < 0.0)
                .count()
                .max(1)
                .min(nfree);
            vecs.slice(ndarray::s![.., ..kept]).to_owned()
        };
        let (_lams, vecs, _actions) = crate::eigensolve::rayleigh_ritz_apply(
            nfree,
            &mut |column| numerical.matvec(surface, column),
            seed.view(),
            h_free.view(),
            self.config.gamma,
            self.config.expand,
            nfree.min(40),
        )?;
        self.ritz_v = Some(vecs);
        let samples = numerical.vs().to_owned();
        let actions = numerical.avs().to_owned();
        if samples.ncols() == 0 || samples.nrows() != x.len() {
            return Ok(());
        }
        let symmetrized = crate::qn::symmetrize_y(samples.view(), actions.view(), Some(2));
        let projected = samples.t().dot(&symmetrized);
        let (_secant_lams, rotation) = crate::eigensolve::exact_eigh(projected.view())?;
        let rotated_steps = samples.dot(&rotation);
        let rotated_actions = actions.dot(&rotation);
        let SellaPes::Cartesian(pes) = &mut self.pes else {
            return Ok(());
        };
        let update = pes.update_kind();
        // Sella `update_H` builds the first model as lam0 * I, with lam0
        // the geometric mean of the absolute secant Ritz values. An
        // identity leaves the unexplored LJ curvature an order of
        // magnitude too soft.
        let previous = pes.hessian().hessian().clone();
        let mut dense = if frame.first {
            let mut log_sum = 0.0;
            let mut count = 0.0;
            for theta in _secant_lams.iter() {
                log_sum += theta.abs().max(1e-12).ln();
                count += 1.0;
            }
            let scale = if count > 0.0 {
                (log_sum / count).exp()
            } else {
                1.0
            };
            let mut scaled = ndarray::Array2::<f64>::eye(x.len());
            scaled *= scale;
            scaled
        } else {
            previous.clone()
        };
        update_h_ms(
            &mut dense,
            rotated_steps.view(),
            rotated_actions.view(),
            update,
            Some(2),
        );
        pes.install_hessian(dense)?;
        // A full-space secant on a lost mode can report curvatures of
        // 10^4. That model takes hundreds of trust steps, so keep the
        // previous Hessian instead.
        let free = crate::geom::u_t_h_u(u, pes.hessian().hessian());
        let absurd = crate::eigensolve::exact_eigh(free.view())
            .map(|(lams, _)| {
                lams.iter()
                    .any(|value| !value.is_finite() || value.abs() > 1.0e4)
            })
            .unwrap_or(true);
        if absurd {
            pes.install_hessian(previous)?;
        }
        Ok(())
    }

    fn schedule_true_diag(&self, evals: &ndarray::Array1<f64>) -> bool {
        if !self.wants_true_diag() || !self.curvature_initialized {
            return false;
        }
        if self.steps_since_diag < self.config.nsteps_per_diag.max(1) as isize {
            return false;
        }
        let order = self.config.order.max(1);
        evals.iter().take(order).any(|value| *value > 0.0)
    }

    fn diag_free(
        &mut self,
        h_free: &ndarray::Array2<f64>,
        g_free: &ndarray::Array1<f64>,
    ) -> Result<(ndarray::Array1<f64>, ndarray::Array2<f64>), SaddleError> {
        // Internal and cell sessions factor the quasi-Newton model.
        // Cartesian sessions finite-difference the true Hessian in
        // [`Self::refresh_true_curvature`] instead. The displacement
        // still uses the complete spectrum: a partial Ritz basis makes
        // P-RFO return a zero step. The factorisation is faer.
        let n = h_free.nrows();
        let due = self.steps_since_diag >= self.config.nsteps_per_diag.max(1) as isize;
        if self.config.eig && self.config.gamma > 0.0 && due {
            self.steps_since_diag = 0;
            let v0 = free_ritz_seed(n, g_free, self.ritz_v.as_ref());
            if let Ok((_lams, vecs, _)) = crate::rayleigh_ritz_iter(
                h_free.view(),
                v0.view(),
                self.config.gamma,
                self.config.expand,
            ) {
                self.ritz_v = Some(vecs);
            }
        } else {
            self.steps_since_diag += 1;
        }
        crate::eigensolve::exact_eigh(h_free.view())
    }

    fn step_cartesian<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let x;
        let energy;
        let gradient;
        let g_r;
        let u;
        let g_free;
        let h_free;
        {
            let pes = match &self.pes {
                SellaPes::Cartesian(p) => p,
                SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                    unreachable!()
                }
            };
            x = pes.position().to_owned();
            let (e, g) = surface.eval(x.view())?;
            energy = e;
            if !g.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("sella gradient"));
            }
            g_r = self.geom.egrad2rgrad(&x, &g);
            gradient = g;
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
            let basis = self.geom.ufree(&x);
            let excluded = surface.excluded_modes(x.view())?;
            if excluded.ncols() != x.len() || !excluded.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Shape("P-RFO excluded directions".into()));
            }
            u = crate::geom::deflate_excluded(&basis, excluded.view());
            if u.ncols() == 0 {
                return Err(SaddleError::Shape("P-RFO free space is empty".into()));
            }
            g_free = crate::geom::u_t_vec(&u, &g_r);
            h_free = crate::geom::u_t_h_u(&u, pes.hessian().hessian());
        }
        let mut h_free = h_free;
        if self.wants_true_diag() && !self.curvature_initialized {
            self.refresh_true_curvature(
                surface,
                &CurvatureFrame {
                    x: &x,
                    gradient: &gradient,
                    u: &u,
                    h_free: &h_free,
                    g_free: &g_free,
                    first: true,
                },
            )?;
            self.curvature_initialized = true;
            // Sella sets the counter to -1 after the opening diagonalisation.
            // The kick below does not diagonalise, so the counter lands on 0.
            self.steps_since_diag = -1;
            h_free = self.cartesian_free_hessian(&u)?;
        }
        if !self.free_matched {
            let nominal = self.geom.n_free(x.len()).max(1);
            let kept = u.ncols().max(1);
            if kept != nominal {
                self.delta *= kept as f64 / nominal as f64;
            }
            self.free_matched = true;
        }
        let vg = Vector::from_host(g_r.clone());
        let (evals, evecs) = crate::eigensolve::exact_eigh(h_free.view())?;
        let order = self.config.order.max(1);
        let s_free = if self.config.restricted.wants_unrestricted_ambient() {
            crate::prfo::PartitionedRationalFunctionOptimization::new(order).get_s(
                &evals,
                &evecs,
                &g_free,
                crate::prfo::PartitionedRationalFunctionOptimization::ALPHA0,
            )
        } else {
            prfo_trust_region(&evals, &evecs, &g_free, order, self.delta)
        };
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
        let (energy, g1, x_new, pred) = {
            let pes = match &mut self.pes {
                SellaPes::Cartesian(p) => p,
                SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                    unreachable!()
                }
            };
            let hs = pes.hessian().hessian().dot(&s);
            let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
            let (energy, g1) = pes.kick_from(surface, d.view(), gradient.view())?;
            let x_new = pes.position().to_owned();
            (energy, g1, x_new, pred)
        };
        let g1_r = self.geom.egrad2rgrad(&x_new, &g1);
        let max_force = self.config.force_gate.value(g1_r.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(
                self.delta,
                self.rho,
                self.config.restricted.step_magnitude(&s),
                &self.config.schedule(),
            );
        } else {
            self.rho = 1.0;
        }
        let at_saddle = max_force <= self.config.force_tol;
        if self.wants_true_diag() && !at_saddle {
            if self.schedule_true_diag(&evals) {
                let basis = self.geom.ufree(&x_new);
                let excluded = surface.excluded_modes(x_new.view())?;
                let u_new = crate::geom::deflate_excluded(&basis, excluded.view());
                if u_new.ncols() > 0 {
                    let g_free_new = crate::geom::u_t_vec(&u_new, &g1_r);
                    let h_free_new = self.cartesian_free_hessian(&u_new)?;
                    self.refresh_true_curvature(
                        surface,
                        &CurvatureFrame {
                            x: &x_new,
                            gradient: &g1,
                            u: &u_new,
                            h_free: &h_free_new,
                            g_free: &g_free_new,
                            first: false,
                        },
                    )?;
                    self.steps_since_diag = 0;
                }
            } else {
                self.steps_since_diag += 1;
            }
        }
        Ok(SellaSaddleReport {
            energy,
            max_force,
            at_saddle,
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
                SellaPes::Cartesian(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                    unreachable!()
                }
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
        self.config.restricted.refuse_internal()?;
        let vg = Vector::from_host(g_int.clone());
        let (evals, evecs) = self.diag_free(&h, &g_int)?;
        let pes = match &mut self.pes {
            SellaPes::Internal(p) => p,
            SellaPes::Cartesian(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                unreachable!()
            }
        };
        let mut s = prfo_trust_region(&evals, &evecs, &g_int, self.config.order.max(1), self.delta);
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
            self.delta = update_trust(
                self.delta,
                self.rho,
                self.config.restricted.step_magnitude(&s),
                &self.config.schedule(),
            );
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

    fn step_cell<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let x;
        let energy;
        let g;
        let g_p;
        let h;
        {
            let pes = match &self.pes {
                SellaPes::Cell(p) => p,
                SellaPes::Cartesian(_) | SellaPes::Internal(_) | SellaPes::CellInternal(_) => {
                    unreachable!()
                }
            };
            x = pes.position().to_owned();
            let c9 = pes.cell9();
            let (e, gg) = surface.eval_in_cell(x.view(), &c9)?;
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
            g_p = pes.packed_grad(surface, g.view())?;
            h = pes.hessian().hessian().to_owned();
        }
        let vg = Vector::from_host(g_p.clone());
        let (evals, evecs) = self.diag_free(&h, &g_p)?;
        let pes = match &mut self.pes {
            SellaPes::Cell(p) => p,
            SellaPes::Cartesian(_) | SellaPes::Internal(_) | SellaPes::CellInternal(_) => {
                unreachable!()
            }
        };
        let order = self.config.order.max(1);
        let mut s = if self.config.restricted.wants_unrestricted_ambient() {
            crate::prfo::PartitionedRationalFunctionOptimization::new(order).get_s(
                &evals,
                &evecs,
                &g_p,
                crate::prfo::PartitionedRationalFunctionOptimization::ALPHA0,
            )
        } else {
            prfo_trust_region(&evals, &evecs, &g_p, order, self.delta)
        };
        s = self.config.restricted.clip_cartesian(&s, self.delta)?;
        let vs = Vector::from_host(s.clone());
        let e0 = energy;
        let hs = h.dot(&s);
        let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
        let (energy, g1p) = pes.kick_packed(surface, s.view())?;
        let n = x.len();
        let mut g1_cart = Array1::zeros(n);
        for i in 0..n {
            g1_cart[i] = g1p[i];
        }
        let max_force = self.config.force_gate.value(g1_cart.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(
                self.delta,
                self.rho,
                self.config.restricted.step_magnitude(&s),
                &self.config.schedule(),
            );
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

    fn step_cell_internal<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<SellaSaddleReport, SaddleError> {
        let x;
        let energy;
        let g;
        let g_p;
        let h;
        let nint;
        {
            let pes = match &self.pes {
                SellaPes::CellInternal(p) => p,
                _ => unreachable!(),
            };
            x = pes.position().to_owned();
            let c9 = pes.cell9();
            let (e, gg) = surface.eval_in_cell(x.view(), &c9)?;
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
            g_p = pes.packed_grad(surface, g.view())?;
            h = pes.hessian().hessian().to_owned();
            nint = pes.internals().n_int();
        }
        self.config.restricted.refuse_internal()?;
        let vg = Vector::from_host(g_p.clone());
        let (evals, evecs) = self.diag_free(&h, &g_p)?;
        let pes = match &mut self.pes {
            SellaPes::CellInternal(p) => p,
            _ => unreachable!(),
        };
        let mut s = prfo_trust_region(&evals, &evecs, &g_p, self.config.order.max(1), self.delta);
        if self.config.restricted == crate::RestrictedKind::MaxInternalStep {
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
            self.delta = update_trust(
                self.delta,
                self.rho,
                self.config.restricted.step_magnitude(&s),
                &self.config.schedule(),
            );
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

/// Columns that seed [`crate::rayleigh_ritz_iter`]. The first call uses
/// the normalized free gradient. A later call uses the stored Ritz
/// vectors. Neither call uses the identity.
pub(crate) fn free_ritz_seed(
    n: usize,
    gradient: &ndarray::Array1<f64>,
    previous: Option<&ndarray::Array2<f64>>,
) -> ndarray::Array2<f64> {
    if let Some(previous) = previous
        && previous.nrows() == n
        && previous.ncols() > 0
        && previous.ncols() <= n
        && previous.iter().all(|v| v.is_finite())
    {
        return previous.to_owned();
    }
    let mut column = ndarray::Array2::zeros((n, 1));
    let mut seed = if gradient.len() == n {
        gradient.to_owned()
    } else {
        ndarray::Array1::zeros(n)
    };
    let norm = rgmin::vecops::nrm2(seed.view());
    if norm > 1e-14 {
        seed.mapv_inplace(|value| value / norm);
    } else if n > 0 {
        seed.fill(0.0);
        seed[0] = 1.0;
    }
    if n > 0 {
        column.column_mut(0).assign(&seed);
    }
    column
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::constraints::Constraints;
    use crate::minmode::PointSurface;
    use ndarray::{Array1, Array2, ArrayView1, array};

    struct Well;
    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    struct CountingWell {
        evaluations: AtomicUsize,
    }

    impl PointSurface for CountingWell {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            Well.eval(x)
        }
    }

    struct RotatedSaddle;

    impl PointSurface for RotatedSaddle {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let inv_sqrt_two = 2.0_f64.sqrt().recip();
            let mode_coordinate = (x[0] + x[1]) * inv_sqrt_two;
            let transverse_coordinate = (x[0] - x[1]) * inv_sqrt_two;
            let mut gradient = x.to_owned();
            gradient[0] = (-mode_coordinate + transverse_coordinate) * inv_sqrt_two;
            gradient[1] = (-mode_coordinate - transverse_coordinate) * inv_sqrt_two;
            let energy = -0.5 * mode_coordinate * mode_coordinate
                + 0.5 * transverse_coordinate * transverse_coordinate
                + 0.5 * x.slice(s![2..]).dot(&x.slice(s![2..]));
            Ok((energy, gradient))
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
    fn prfo_step_evaluates_each_endpoint_once() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let surface = CountingWell {
            evaluations: AtomicUsize::new(0),
        };
        let mut session = SellaSaddleSession::new(
            SellaSaddleConfig {
                eig: false,
                ..SellaSaddleConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();

        session.step(&surface).unwrap();

        assert_eq!(surface.evaluations.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn true_hessian_diag_uses_more_than_the_two_endpoints() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let surface = CountingWell {
            evaluations: AtomicUsize::new(0),
        };
        let mut session = SellaSaddleSession::new(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        session.step(&surface).unwrap();
        assert!(surface.evaluations.load(Ordering::Relaxed) > 2);
    }

    #[test]
    fn seeded_mode_hands_the_selected_unstable_direction_to_prfo() {
        let inv_sqrt_two = 2.0_f64.sqrt().recip();
        let mode = Array1::from(vec![inv_sqrt_two, inv_sqrt_two, 0.0, 0.0, 0.0, 0.0]);
        let x = &mode * 0.2;
        let mut sess = SellaSaddleSession::new(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();

        sess.seed_mode(mode.view(), -1.0).unwrap();
        let before = sess.position().dot(&mode).abs();
        sess.step(&RotatedSaddle).unwrap();
        let after = sess.position().dot(&mode).abs();

        assert!(
            after < before,
            "unstable coordinate did not contract: {before} -> {after}"
        );
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
        assert!((sess.delta() - 0.3).abs() < 1e-14, "delta={}", sess.delta());
    }

    #[test]
    fn prfo_step_stays_on_the_rigid_quotient() {
        use rgmin::ManifoldKind;
        use rgmin::vecops::nrm2;
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
    fn cell_prfo_step_is_finite() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let cell = crate::mic::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaSaddleSession::on_cell(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
            cell,
            mask,
        )
        .unwrap();
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
        assert!(sess.cell_pes().is_some());
    }

    #[test]
    fn cell_internal_prfo_step_is_finite() {
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
        let cell = crate::mic::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaSaddleSession::on_cell_internal(
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
            chart,
            cell,
            mask,
        )
        .unwrap();
        assert!(sess.cell_internal_pes().is_some());
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        assert!(sess.position().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cell_internal_log_prfo_step_is_finite() {
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
        let cell = crate::mic::Cell::ortho(3.0, 3.0, 3.0).unwrap();
        let mut mask = [false; 9];
        mask[0] = true;
        let mut sess = SellaSaddleSession::on_cell_internal_log(
            SellaSaddleConfig::default(),
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
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
    }

    #[test]
    fn internals_prfo_step_is_finite() {
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

    #[test]
    fn ritz_seed_is_the_gradient_and_not_the_identity() {
        let gradient = array![0.0, 3.0, 0.0, -4.0];
        let seed = free_ritz_seed(4, &gradient, None);
        assert_eq!(seed.ncols(), 1);
        assert_eq!(seed.nrows(), 4);
        let norm = (3.0_f64 * 3.0 + 4.0 * 4.0).sqrt();
        assert!((seed[(1, 0)] - 3.0 / norm).abs() < 1e-12);
        assert!((seed[(3, 0)] + 4.0 / norm).abs() < 1e-12);
        assert!(seed[(0, 0)].abs() < 1e-12);
        let mut previous = Array2::<f64>::zeros((4, 2));
        previous[(0, 0)] = 1.0;
        previous[(2, 1)] = 1.0;
        let again = free_ritz_seed(4, &gradient, Some(&previous));
        assert_eq!(again.ncols(), 2);
        assert!((again[(0, 0)] - 1.0).abs() < 1e-15);
        assert!((again[(2, 1)] - 1.0).abs() < 1e-15);
        let eye = Array2::<f64>::eye(4);
        assert_ne!(seed, eye);
        assert_ne!(again, eye);
    }
}
