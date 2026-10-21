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
use ndarray::{Array1, Array2, s};
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
    /// Sella `gamma` residual scale. `<= 0` is exact eigh.
    pub gamma: f64,
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

pub struct SellaSaddleSession {
    pes: SellaPes,
    config: SellaSaddleConfig,
    geom: SellaGeom,
    delta: f64,
    rho: f64,
    steps_since_diag: usize,
    ritz_v: Option<ndarray::Array2<f64>>,
    /// Sella `first_diag`: the Cartesian model has seen one true
    /// Hessian diagonalization.
    hvp_started: bool,
    /// Steps since that diagonalization. Sella waits `nsteps_per_diag`
    /// before it looks at the curvature sign again.
    hvp_since: i32,
    /// Largest atomic displacement allowed when the Euclidean radius
    /// was scaled by `n_free` past one distance unit. Grows and
    /// shrinks with the rho schedule. Unused while [`Self::atom_cap_active`]
    /// is false.
    atom_delta: f64,
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
        let atom_delta = atomic_ceiling(config.delta);
        Ok(Self {
            pes: SellaPes::Cartesian(CartesianPes::new(x, masses)?),
            config,
            geom,
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
            hvp_started: false,
            hvp_since: 0,
            atom_delta,
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
        let atom_delta = atomic_ceiling(config.delta);
        Ok(Self {
            pes: SellaPes::Internal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
            hvp_started: false,
            hvp_since: 0,
            atom_delta,
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
        let atom_delta = atomic_ceiling(config.delta);
        Ok(Self {
            pes: SellaPes::Cell(pes),
            config,
            geom: SellaGeom::Kind(rgmin::ManifoldKind::Euclidean),
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
            hvp_started: false,
            hvp_since: 0,
            atom_delta,
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
        let atom_delta = atomic_ceiling(config.delta);
        Ok(Self {
            pes: SellaPes::CellInternal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
            hvp_started: false,
            hvp_since: 0,
            atom_delta,
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
        let atom_delta = atomic_ceiling(config.delta);
        Ok(Self {
            pes: SellaPes::CellInternal(pes),
            config,
            geom,
            delta,
            rho: 1.0,
            steps_since_diag: 0,
            ritz_v: None,
            hvp_started: false,
            hvp_since: 0,
            atom_delta,
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
        self.ritz_v = None;
        self.hvp_started = false;
        self.hvp_since = 0;
        self.atom_delta = atomic_ceiling(self.config.delta);
    }

    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<SellaSaddleReport, SaddleError> {
        match &self.pes {
            SellaPes::Internal(_) => self.step_internal(surface),
            SellaPes::Cell(_) => self.step_cell(surface),
            SellaPes::CellInternal(_) => self.step_cell_internal(surface),
            SellaPes::Cartesian(_) => self.step_cartesian(surface),
        }
    }

    /// Sella `PES.diag`: iterative diagonalization of a finite-difference
    /// Hessian, then a multi-secant TS-BFGS update of the model.
    ///
    /// Each product reuses the centre gradient (one new force call).
    /// The first call always runs. Later calls run only after
    /// `nsteps_per_diag` steps and only when one of the first `order`
    /// model eigenvalues is still positive. Hermes, Sargsyan, Najm,
    /// and Zádor, J. Chem. Theory Comput. (2022),
    /// doi:10.1021/acs.jctc.2c00395. An order-1 search then keeps
    /// TS-BFGS for the step secant: plain BFGS drops a pair with
    /// `y·s <= 0` and cannot hold the negative mode.
    fn maybe_hvp_diag<S: PointSurface>(
        &mut self,
        surface: &S,
        x: &Array1<f64>,
        gradient: ndarray::ArrayView1<f64>,
        u: &Array2<f64>,
        g_free: &Array1<f64>,
        h_free: &Array2<f64>,
    ) -> Result<(), SaddleError> {
        if !self.config.eig || self.config.gamma <= 0.0 || u.ncols() == 0 {
            return Ok(());
        }
        if !self.wants_hvp(h_free)? {
            return Ok(());
        }
        let nfree = u.ncols();
        let eta = 1e-4;
        let mut numerical = crate::NumericalHessian::new(
            x.clone(),
            gradient.to_owned(),
            eta,
            false,
            Some(u.clone()),
        )?;
        let mut columns: Vec<Array1<f64>> = Vec::new();
        // Sella's first call seeds with the projected gradient. A later
        // call seeds with the softest eigenvector of the model, which
        // is the negative mode when it is still negative and the least
        // positive direction when a step update has just lost it.
        // Seeding that later call with the gradient alone rebuilds the
        // Krylov from scratch.
        if self.hvp_started {
            if let Ok((_, vecs)) = crate::eigensolve::exact_eigh(h_free.view())
                && vecs.ncols() > 0
            {
                columns.push(vecs.column(0).to_owned());
            }
        } else if let Ok((lams, vecs)) = crate::eigensolve::exact_eigh(h_free.view())
            && lams.first().copied().unwrap_or(0.0) < 0.0
        {
            columns.push(vecs.column(0).to_owned());
        }
        if !self.hvp_started && g_free.len() == nfree && rgmin::vecops::nrm2(g_free.view()) >= 1e-12
        {
            columns.push(g_free.clone());
        }
        let v0 = if columns.is_empty() {
            let mut eye = Array2::zeros((nfree, 1));
            eye[(0, 0)] = 1.0;
            eye
        } else {
            let mut seed = Array2::zeros((nfree, columns.len()));
            for (j, column) in columns.iter().enumerate() {
                seed.column_mut(j).assign(column);
            }
            crate::modified_gram_schmidt(seed.view(), None, 1e-12)
        };
        let v0 = if v0.ncols() == 0 {
            let mut eye = Array2::zeros((nfree, 1));
            eye[(0, 0)] = 1.0;
            eye
        } else {
            v0
        };
        let preconditioner = h_free.clone();
        // A repeat diagonalization only has to put the lost negative
        // mode back. The first call keeps `gamma`; a later call can
        // stop at a wider residual and still hand P-RFO a usable mode.
        let gamma = if self.hvp_started {
            self.config.gamma.max(0.5)
        } else {
            self.config.gamma
        };
        let expand = self.config.expand;
        let (lams, vecs, actions) = crate::eigensolve::rayleigh_ritz_apply(
            nfree,
            |direction| numerical.matvec(surface, direction),
            v0.view(),
            preconditioner.view(),
            gamma,
            expand,
            // A repeat diagonalization keeps the relative residual.
            // The identity-curvature floor is only for the first build,
            // where one unresolved soft root would otherwise spend a
            // product for every free coordinate.
            if self.hvp_started {
                nfree
            } else {
                self.config.order.max(1)
            },
        )?;
        if vecs.ncols() == 0 {
            self.hvp_started = true;
            self.hvp_since = 0;
            return Ok(());
        }
        let mut free = if self.hvp_started {
            h_free.clone()
        } else {
            let scale = geometric_curvature(&lams);
            let mut eye = Array2::zeros((nfree, nfree));
            for i in 0..nfree {
                eye[(i, i)] = scale;
            }
            eye
        };
        let kind = if self.config.order >= 1 {
            crate::HessUpdate::TsBfgs
        } else {
            match &self.pes {
                SellaPes::Cartesian(pes) => pes.update_kind(),
                _ => crate::HessUpdate::Bfgs,
            }
        };
        crate::update_h_ms(&mut free, vecs.view(), actions.view(), kind, Some(2));
        if lams.first().copied().unwrap_or(0.0) < 0.0 {
            let lowest = crate::eigensolve::exact_eigh(free.view())
                .ok()
                .and_then(|(values, _)| values.first().copied())
                .unwrap_or(1.0);
            if lowest >= 0.0 {
                plant_rayleigh(&mut free, &vecs.column(0).to_owned(), lams[0]);
            }
        }
        let old = match &self.pes {
            SellaPes::Cartesian(pes) => pes.hessian().hessian().to_owned(),
            SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                return Ok(());
            }
        };
        let installed = embed_free_block(old, u, &free);
        if let SellaPes::Cartesian(pes) = &mut self.pes {
            pes.install_hessian(installed)?;
            if self.config.order >= 1 && pes.update_kind() == crate::HessUpdate::Bfgs {
                pes.set_update(crate::HessUpdate::TsBfgs);
            }
        }
        self.hvp_started = true;
        self.hvp_since = 0;
        Ok(())
    }

    /// Whether this Cartesian step should pay for Hessian products.
    fn wants_hvp(&mut self, h_free: &Array2<f64>) -> Result<bool, SaddleError> {
        if !self.hvp_started {
            return Ok(true);
        }
        let every = self.config.nsteps_per_diag.max(1) as i32;
        if self.hvp_since < every {
            self.hvp_since += 1;
            return Ok(false);
        }
        let order = self.config.order.max(1);
        let negative_missing = match crate::eigensolve::exact_eigh(h_free.view()) {
            Ok((values, _)) => values.iter().take(order).any(|value| *value > 0.0),
            Err(_) => true,
        };
        if negative_missing {
            return Ok(true);
        }
        self.hvp_since += 1;
        Ok(false)
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
            u = self.geom.ufree(&x);
            g_free = crate::geom::u_t_vec(&u, &g_r);
            h_free = crate::geom::u_t_h_u(&u, pes.hessian().hessian());
        }
        self.maybe_hvp_diag(surface, &x, gradient.view(), &u, &g_free, &h_free)?;
        let h_free = {
            let pes = match &self.pes {
                SellaPes::Cartesian(p) => p,
                SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => {
                    unreachable!()
                }
            };
            crate::geom::u_t_h_u(&u, pes.hessian().hessian())
        };
        let vg = Vector::from_host(g_r.clone());
        let (evals, evecs) = crate::eigensolve::exact_eigh(h_free.view())?;
        let cap_active = self.atom_cap_active();
        let atom_delta = self.atom_delta;
        let pes = match &mut self.pes {
            SellaPes::Cartesian(p) => p,
            SellaPes::Internal(_) | SellaPes::Cell(_) | SellaPes::CellInternal(_) => unreachable!(),
        };
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
        if cap_active && crate::restricted::ras_cons(&s) > atom_delta {
            // A uniform scale of the full P-RFO step keeps a bad ratio of
            // uphill to downhill components. Sella's restricted atomic step
            // solves for the alpha whose largest atom move is the ceiling.
            let searched = atomic_prfo_step(
                &CartesianPrfo {
                    evals: &evals,
                    evecs: &evecs,
                    g_free: &g_free,
                    order,
                    u: &u,
                    x: &x,
                    geom: &self.geom,
                },
                atom_delta,
            );
            if crate::restricted::ras_cons(&searched) <= atom_delta * (1.0 + 1e-6) {
                s = searched;
            } else {
                s = rgmin::ras_clip(&s, atom_delta);
            }
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
        let (energy, g1) = pes.kick_from(surface, d.view(), gradient.view())?;
        let x_new = pes.position().to_owned();
        let g1_r = self.geom.egrad2rgrad(&x_new, &g1);
        let max_force = self.config.force_gate.value(g1_r.view());
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            if cap_active {
                self.atom_delta = update_trust(
                    self.atom_delta,
                    self.rho,
                    crate::restricted::ras_cons(&s),
                    &self.config.schedule(),
                );
            } else {
                self.delta = update_trust(
                    self.delta,
                    self.rho,
                    self.config.restricted.step_magnitude(&s),
                    &self.config.schedule(),
                );
            }
        } else {
            self.rho = 1.0;
        }
        Ok(SellaSaddleReport {
            energy,
            max_force,
            at_saddle: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: if cap_active {
                self.atom_delta
            } else {
                self.delta
            },
        })
    }

    /// True when a Cartesian trust region was scaled by `n_free` past
    /// one distance unit while iterative diagonalization is on.
    ///
    /// Sella's Cartesian default is a restricted atomic step at
    /// `delta0`, not `delta0 * n_free`. The scaled radius lets one
    /// atom absorb the whole step and leave the basin. The ceiling
    /// stays off for `eig: false` (those trajectories are recorded
    /// against the Euclidean radius) and when that radius is already
    /// at most one distance unit, including a host `delta` small
    /// enough that `delta * n_free` never reaches it.
    fn atom_cap_active(&self) -> bool {
        self.config.eig
            && self.config.restricted == crate::RestrictedKind::TrustRegion
            && matches!(self.pes, SellaPes::Cartesian(_))
            && self.delta > 1.0
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
        let (evals, evecs) = self.diag_free(&h)?;
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
        let (evals, evecs) = self.diag_free(&h)?;
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
        let (evals, evecs) = self.diag_free(&h)?;
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

/// Free-space P-RFO data for one atomic alpha search.
struct CartesianPrfo<'a> {
    evals: &'a Array1<f64>,
    evecs: &'a Array2<f64>,
    g_free: &'a Array1<f64>,
    order: usize,
    u: &'a Array2<f64>,
    x: &'a Array1<f64>,
    geom: &'a SellaGeom,
}

/// Order-1 P-RFO step whose largest atomic move meets `delta`.
///
/// The alpha loop is Sella `BaseRestrictedStep.get_s`: Newton on the
/// active-atom constraint, with a bisection whenever the Newton alpha
/// leaves the bracket. An interior alpha of 1 is returned unchanged.
fn atomic_prfo_step(basis: &CartesianPrfo<'_>, delta: f64) -> Array1<f64> {
    let newton_safe = basis.order <= 1;
    let tol = if newton_safe { 1e-10 } else { 1e-15 };
    let mut alpha = 1.0;
    let (mut step, mut value, mut slope) = cartesian_prfo_atom(basis, alpha);
    if value < delta {
        return step;
    }
    let mut err = value - delta;
    let mut lower = 0.0_f64;
    let mut upper = 1.0_f64;
    for niter in 0..64 {
        if err.abs() <= tol {
            break;
        }
        let bracket_open = matches!(lower.partial_cmp(&upper), Some(std::cmp::Ordering::Less));
        if !bracket_open || upper - lower <= 4.0 * f64::EPSILON {
            break;
        }
        if err > 0.0 {
            upper = alpha;
        } else {
            lower = alpha;
        }
        let newton = alpha - err / slope;
        let force_bisection = (!newton_safe && niter > 4) || (newton_safe && niter % 12 == 11);
        if !newton.is_finite() || newton <= lower || newton >= upper || force_bisection {
            let midpoint = 0.5 * (lower + upper);
            alpha = if midpoint.is_finite() {
                midpoint
            } else {
                alpha + alpha.abs().max(1.0) * 0.5 * midpoint.signum()
            };
        } else {
            alpha = newton;
        }
        let next = cartesian_prfo_atom(basis, alpha);
        step = next.0;
        value = next.1;
        slope = next.2;
        err = value - delta;
    }
    step
}

fn cartesian_prfo_atom(basis: &CartesianPrfo<'_>, alpha: f64) -> (Array1<f64>, f64, f64) {
    let (s_free, ds_free) = crate::prfo::prfo_step_and_derivative(
        basis.evals,
        basis.evecs,
        basis.g_free,
        basis.order,
        alpha,
    );
    let step = basis
        .geom
        .project(basis.x, &crate::geom::u_vec(basis.u, &s_free));
    let deriv = basis
        .geom
        .project(basis.x, &crate::geom::u_vec(basis.u, &ds_free));
    let (value, slope) = atomic_constraint(&step, &deriv);
    (step, value, slope)
}

/// Largest atomic displacement and its derivative along the active atom.
fn atomic_constraint(step: &Array1<f64>, deriv: &Array1<f64>) -> (f64, f64) {
    let atoms = step.len() / 3;
    if atoms == 0 || deriv.len() != step.len() {
        return (crate::restricted::ras_cons(step), 0.0);
    }
    let mut index = 0usize;
    let mut value = 0.0;
    for atom in 0..atoms {
        let mut length_sq = 0.0;
        for axis in 0..3 {
            let component = step[3 * atom + axis];
            length_sq += component * component;
        }
        let length = length_sq.sqrt();
        if length > value {
            value = length;
            index = atom;
        }
    }
    let mut slope = 0.0;
    if value > 0.0 {
        for axis in 0..3 {
            slope += deriv[3 * index + axis] * step[3 * index + axis];
        }
        slope /= value.max(1e-12);
    }
    (value, slope)
}

/// Per-atom ceiling. A host `delta` below Sella's default is a
/// Euclidean scale factor (`delta * n_free`), not a tighter atomic limit.
fn atomic_ceiling(delta: f64) -> f64 {
    delta.max(SellaSaddleConfig::default().delta)
}

/// Geometric mean of absolute Ritz values. Sella's first Hessian is
/// this scale times the identity before the multi-secant correction.
fn geometric_curvature(lams: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    let mut count = 0.0;
    for lam in lams.iter() {
        acc += lam.abs().max(1e-12).ln();
        count += 1.0;
    }
    if count == 0.0 {
        1.0
    } else {
        (acc / count).exp()
    }
}

/// Set the Rayleigh quotient of `u` to `lam`.
fn plant_rayleigh(h: &mut Array2<f64>, u: &Array1<f64>, lam: f64) {
    let n = h.nrows().min(u.len());
    let mut norm_sq = 0.0;
    for i in 0..n {
        norm_sq += u[i] * u[i];
    }
    if norm_sq <= 1e-24 {
        return;
    }
    let mut rayleigh = 0.0;
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += h[(i, j)] * u[j];
        }
        rayleigh += u[i] * acc;
    }
    let scale = (lam - rayleigh / norm_sq) / norm_sq;
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] += scale * u[i] * u[j];
        }
    }
}

/// Replace the free block of `h` with `h_free`, leaving the complement.
fn embed_free_block(mut h: Array2<f64>, u: &Array2<f64>, h_free: &Array2<f64>) -> Array2<f64> {
    let old_free = crate::geom::u_t_h_u(u, &h);
    let delta = h_free - &old_free;
    let lifted = u.dot(&delta).dot(&u.t());
    h = &h + &lifted;
    let n = h.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let value = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = value;
            h[(j, i)] = value;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::constraints::Constraints;
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
            SellaSaddleConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();

        session.step(&surface).unwrap();

        // Centre gradient, one finite-difference product for the
        // negative mode, and the accepted point. A later step, with
        // that mode already negative, does not diagonalize again.
        let first = surface.evaluations.load(Ordering::Relaxed);
        assert!(first >= 3, "first step calls {first}");
        surface.evaluations.store(0, Ordering::Relaxed);
        session.step(&surface).unwrap();
        assert_eq!(surface.evaluations.load(Ordering::Relaxed), 2);
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

    struct SpreadSaddle;

    impl PointSurface for SpreadSaddle {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let mut energy = -0.5 * x[0] * x[0];
            let mut gradient = x.to_owned();
            gradient[0] = -x[0];
            for i in 1..x.len() {
                energy += 0.5 * x[i] * x[i];
            }
            Ok((energy, gradient))
        }
    }

    #[test]
    fn scaled_trust_radius_still_caps_one_atom() {
        let mut x = Array1::zeros(18);
        for atom in 0..6 {
            x[3 * atom + 1] = atom as f64;
        }
        x[0] = 1.5;
        let mut sess =
            SellaSaddleSession::new(SellaSaddleConfig::default(), x.clone(), Array1::ones(6))
                .unwrap();
        assert!(
            sess.delta() > 1.0,
            "the Euclidean radius should be delta0 * n_free, got {}",
            sess.delta()
        );
        let report = sess.step(&SpreadSaddle).unwrap();
        let step = &sess.position().to_owned() - &x;
        let largest = crate::restricted::ras_cons(&step);
        assert!(
            largest <= 0.1 + 1e-8,
            "one atom moved {largest}, past the Cartesian ceiling"
        );
        assert!(
            report.delta < 0.5,
            "rho updated the Euclidean radius {} instead of the atomic ceiling",
            report.delta
        );
        assert!(
            sess.position()[4] > 0.5,
            "the far atom was pulled to the origin in one step"
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
}
