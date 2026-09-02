//! Intrinsic reaction coordinate session: roll from a first-order
//! saddle to a minimum, one branch at a time.
//!
//! Inner geometry is rgmin `IrcTrust` (Sella IRCTrustRegion /
//! Gonzalez--Schlegel sphere) in the selected coordinate metric.
//! [`IrcSession::new`] selects mass-weighted `ManifoldKind::MwRigid`
//! for atomistic Cartesian coordinates; [`IrcSession::new_euclidean`]
//! selects unit weights for a generic-dimensional surface. The default
//! increment is rgmin `qn_irc_restricted` (Sella QuasiNewtonIRC).
//! [`IrcKind::Morokuma`] is the
//! Ishida--Morokuma--Komornicki predictor-corrector from
//! `gpr_optim` `IRCDriver`. Ambient algebra goes through
//! `rgmin::vecops`. The host owns the loop. `run` is a convenience
//! over `step`.

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::{Vector, axpy, dot, nrm2, nrminf};
use rgmin::{
    BfgsModel, Control, EigensolverKind, IrcTrust, ManifoldKind, Method, Solver, mw_pair,
    qn_irc_restricted, sqrt_masses_3n, to_mw,
};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Which side of the imaginary mode to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrcDirection {
    Forward,
    Reverse,
}

/// How an outer IRC increment is formed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IrcKind {
    /// Gonzalez--Schlegel / Sella sphere in the session metric. Default.
    #[default]
    Gs2,
    /// Ishida--Morokuma--Komornicki predictor-corrector
    /// (`gpr_optim` `IRCDriver` Morokuma). Not GS2.
    Morokuma,
}

impl IrcKind {
    /// C `rgsaddle_irc_kind_t`.
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Gs2 => 0,
            Self::Morokuma => 1,
        }
    }
}

/// Outer IRC controls.
#[derive(Clone, Debug)]
pub struct IrcConfig {
    /// Sphere radius in the session metric (Sella `dx`) and Morokuma `h`.
    pub dx: f64,
    pub force_tol: f64,
    /// eOn / gpr_optim `ConvergenceForceNorm`.
    pub force_gate: crate::ForceGate,
    pub max_move: f64,
    pub method: Method,
    pub max_inner: usize,
    /// Finite-difference Hessian action length for the matrix-free kick.
    pub dr: f64,
    /// Krylov dimension for the lowest-mode kick.
    pub krylov_dim: usize,
    /// Closed lowest-mode backend. Unlinked kinds fail closed.
    pub eigen_kind: EigensolverKind,
    /// GS2 is the default; Morokuma is the ORCA / IRCDriver arm.
    pub kind: IrcKind,
}

impl Default for IrcConfig {
    fn default() -> Self {
        Self {
            dx: 0.1,
            force_tol: 0.05,
            force_gate: crate::ForceGate::MaxForceOnAtom,
            max_move: 0.2,
            method: Method::Steepest,
            max_inner: 10,
            dr: 1e-3,
            krylov_dim: 12,
            eigen_kind: EigensolverKind::Lanczos,
            kind: IrcKind::Gs2,
        }
    }
}

/// One accepted IRC point.
#[derive(Clone, Debug)]
pub struct IrcReport {
    pub energy: f64,
    pub max_force: f64,
    pub arc: f64,
    pub inner_steps: usize,
    pub at_minimum: bool,
}

/// Forward or reverse roll-down over an rgmin `Solver`.
pub struct IrcSession {
    config: IrcConfig,
    x: Array1<f64>,
    x_saddle: Array1<f64>,
    sqrtm: Array1<f64>,
    d1: Array1<f64>,
    mode: Array1<f64>,
    solver: Solver,
    hess: BfgsModel,
    first: bool,
    arc: f64,
    last_step: Option<Array1<f64>>,
    last_outer: Option<Array1<f64>>,
}

impl IrcSession {
    /// `mode` is the Cartesian imaginary-mode direction (not yet scaled).
    pub fn new(
        config: IrcConfig,
        saddle: Array1<f64>,
        masses: Array1<f64>,
        mode: Array1<f64>,
        direction: IrcDirection,
    ) -> Result<Self, SaddleError> {
        let n3 = saddle.len();
        if n3 == 0 || n3 % 3 != 0 {
            return Err(SaddleError::Shape(
                "saddle must be a nonzero 3N Cartesian".into(),
            ));
        }
        if masses.len() * 3 != n3 || mode.len() != n3 {
            return Err(SaddleError::Shape(
                "masses (N) and mode (3N) must match the saddle".into(),
            ));
        }
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: Some(config.max_move),
        };
        let mut solver = Solver::new(config.method.clone(), control, n3);
        solver.set_highs(true);
        // SE(3) quotient needs leftover internal modes: N >= 3.
        if masses.len() >= 3 {
            solver.set_manifold(ManifoldKind::MwRigid);
            solver.set_masses(masses.clone());
        }
        Self::with_coordinate_metric(
            config,
            saddle,
            sqrt_masses_3n(masses.as_slice().unwrap_or(&[])),
            mode,
            direction,
            solver,
        )
    }

    /// Build an IRC over an arbitrary-dimensional Euclidean surface.
    ///
    /// The native coordinates are the path metric, so every coordinate has
    /// unit mass weight. Use [`Self::new`] for atomistic Cartesian positions,
    /// where atomic masses and the rigid-body quotient define the metric.
    pub fn new_euclidean(
        config: IrcConfig,
        saddle: Array1<f64>,
        mode: Array1<f64>,
        direction: IrcDirection,
    ) -> Result<Self, SaddleError> {
        let dimension = saddle.len();
        if dimension == 0 {
            return Err(SaddleError::Shape(
                "saddle must have a nonzero dimension".into(),
            ));
        }
        if mode.len() != dimension {
            return Err(SaddleError::Shape(
                "mode must match the saddle dimension".into(),
            ));
        }
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: Some(config.max_move),
        };
        let mut solver = Solver::new(config.method.clone(), control, dimension);
        solver.set_highs(true);
        Self::with_coordinate_metric(
            config,
            saddle,
            Array1::ones(dimension),
            mode,
            direction,
            solver,
        )
    }

    fn with_coordinate_metric(
        config: IrcConfig,
        saddle: Array1<f64>,
        sqrtm: Array1<f64>,
        mode: Array1<f64>,
        direction: IrcDirection,
        solver: Solver,
    ) -> Result<Self, SaddleError> {
        let dimension = saddle.len();
        if sqrtm.len() != dimension || mode.len() != dimension {
            return Err(SaddleError::Shape(
                "coordinate weights and mode must match the saddle".into(),
            ));
        }
        let mut session = Self {
            config,
            x: saddle.clone(),
            x_saddle: saddle,
            sqrtm,
            d1: Array1::zeros(dimension),
            mode,
            solver,
            hess: BfgsModel::identity(dimension),
            first: true,
            arc: 0.0,
            last_step: None,
            last_outer: None,
        };
        session.seed_ts_mode();
        session.set_direction(direction);
        Ok(session)
    }

    fn seed_ts_mode(&mut self) {
        let n = self.mode.len().min(self.sqrtm.len());
        let mut u = Array1::zeros(self.mode.len());
        for i in 0..n {
            u[i] = self.mode[i] * self.sqrtm[i];
        }
        self.hess.seed_mode(&u, -1.0);
    }

    /// Estimate the imaginary mode with matrix-free Lanczos (FD Hessian
    /// actions). This is the ChASE / PRIMME / Sella lesson: IRC needs
    /// one extremal pair, not a full ELPA / SLATE heev of H.
    pub fn from_surface<S: crate::minmode::PointSurface>(
        config: IrcConfig,
        saddle: Array1<f64>,
        masses: Array1<f64>,
        seed: Array1<f64>,
        direction: IrcDirection,
        surface: &S,
    ) -> Result<Self, SaddleError> {
        let (_, g0) = surface.eval(saddle.view())?;
        let mm = crate::minmode::MinModeConfig {
            dr: config.dr,
            krylov_dim: config.krylov_dim,
            eigen_kind: config.eigen_kind,
            ..crate::minmode::MinModeConfig::default()
        };
        let (mode, _curv, _actions) =
            crate::minmode::lanczos_mode(surface, saddle.view(), g0.view(), seed, &mm)?;
        Self::new(config, saddle, masses, mode, direction)
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn reset(&mut self) {
        self.solver.forget();
        self.hess.forget();
        self.seed_ts_mode();
    }

    /// Restore the saddle and flip the kick sign.
    pub fn set_direction(&mut self, direction: IrcDirection) {
        self.x = self.x_saddle.clone();
        self.d1 = self.kick_vector(direction);
        self.first = true;
        self.arc = 0.0;
        self.last_step = None;
        self.last_outer = None;
        self.solver.forget();
        self.hess.forget();
        self.seed_ts_mode();
    }

    fn kick_vector(&self, direction: IrcDirection) -> Array1<f64> {
        let mut mw = Vector::from_host(self.mode.clone());
        for i in 0..mw.host_view().len() {
            mw.host_mut()[i] *= self.sqrtm[i].max(1e-16);
        }
        let n = nrm2(mw.host_view());
        if n <= 1e-16 {
            return Array1::zeros(self.mode.len());
        }
        let mut v0ts = Array1::zeros(self.mode.len());
        for i in 0..self.mode.len() {
            v0ts[i] = self.config.dx * (mw.host_view()[i] / n) / self.sqrtm[i].max(1e-16);
        }
        // Sella: first nonzero of v0ts is positive, then reverse is -v0ts.
        if let Some(&v) = v0ts.iter().find(|v| v.abs() > 1e-16) {
            if v < 0.0 {
                v0ts.mapv_inplace(|c| -c);
            }
        }
        match direction {
            IrcDirection::Forward => v0ts,
            IrcDirection::Reverse => v0ts.mapv(|c| -c),
        }
    }

    fn trust(&self) -> IrcTrust {
        IrcTrust {
            d1: self.d1.clone(),
            sqrtm: self.sqrtm.clone(),
            dx: self.config.dx,
        }
    }

    /// Sella: project `g` orthogonal to the path in the MW metric.
    fn path_force(&self, g: &Array1<f64>) -> Array1<f64> {
        path_projected_force(g, &self.d1, &self.sqrtm)
    }

    /// Sella `QuasiNewtonIRC` plus `IRCTrustRegion` on the current
    /// MW sphere. The BFGS model is identity until the first accepted
    /// inner pair; GS2 equality is `IrcTrust`.
    fn restricted_increment(&self, g: &Array1<f64>) -> Array1<f64> {
        let (evals, evecs) = self.hess.eigh();
        let allow_interior = self.hess.is_posdef() && self.arc > 8.0 * self.config.dx;
        qn_irc_restricted(&self.trust(), &evals, &evecs, g, allow_interior)
    }

    /// One outer IRC move. [`IrcKind::Gs2`] is Sella / GS2;
    /// [`IrcKind::Morokuma`] is the IRCDriver predictor-corrector.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        if self.config.kind == IrcKind::Morokuma {
            return self.step_morokuma(surface);
        }
        self.step_gs2(surface)
    }

    fn step_gs2<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        let kicked = self.first;
        if self.first {
            axpy(1.0, self.d1.view(), &mut self.x);
            self.first = false;
        }
        let x_start = self.x.clone();

        let (energy0, g0) = surface.eval(self.x.view())?;
        if !g0.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        let max_force0 = self.config.force_gate.value(g0.view());
        let fmax_inner = self.config.force_tol.min(0.01);
        let mut inner_steps = 0;
        let mut energy = energy0;
        let mut g = g0;
        let mut max_force = max_force0;
        let mut last_interior = false;
        for _ in 0..self.config.max_inner {
            inner_steps += 1;
            let s = self.restricted_increment(&g);
            let interior = self.hess.is_posdef()
                && self.arc > 8.0 * self.config.dx
                && self.trust().cons(&s) + 1e-8 < self.config.dx;
            if inner_steps == 1 {
                if let Some(prev) = &self.last_outer {
                    // Reversal after a PD model and a long enough arc
                    // is a well overshoot only when the force is already
                    // under the host tolerance. A stale BFGS pair can
                    // reverse on a molecular surface far from a basin.
                    if self.hess.is_posdef()
                        && self.arc > 8.0 * self.config.dx
                        && max_force <= self.config.force_tol
                        && dot(prev.view(), s.view()) < 0.0
                    {
                        self.d1.fill(0.0);
                        self.last_step = None;
                        self.solver.forget();
                        return Ok(IrcReport {
                            energy,
                            max_force,
                            arc: self.arc,
                            inner_steps,
                            at_minimum: true,
                        });
                    }
                }
            }
            if let Some(prev) = &self.last_step {
                if !kicked
                    && self.hess.is_posdef()
                    && self.arc > 8.0 * self.config.dx
                    && max_force <= self.config.force_tol
                    && dot(prev.view(), s.view()) < 0.0
                {
                    self.d1.fill(0.0);
                    self.last_step = None;
                    self.solver.forget();
                    return Ok(IrcReport {
                        energy,
                        max_force,
                        arc: self.arc,
                        inner_steps,
                        at_minimum: true,
                    });
                }
            }
            self.last_step = Some(s.clone());
            last_interior = interior;
            axpy(1.0, s.view(), &mut self.d1);
            axpy(1.0, s.view(), &mut self.x);
            let ev = surface.eval(self.x.view())?;
            energy = ev.0;
            let y = &ev.1 - &g;
            let (s_mw, y_mw) = mw_pair(&s, &y, &self.sqrtm);
            self.hess.update(&s_mw, &y_mw);
            g = ev.1;
            max_force = self.config.force_gate.value(g.view());
            let g_path = self.path_force(&g);
            let fmax_path = nrminf(g_path.view());
            let bound = self.trust().on_bound(&Array1::zeros(s.len()), 1e-8);
            if (bound && fmax_path <= fmax_inner)
                || (max_force <= self.config.force_tol && interior)
            {
                break;
            }
        }
        let mut outer = self.x.clone();
        axpy(-1.0, x_start.view(), &mut outer);
        self.last_outer = Some(outer);
        self.d1.fill(0.0);
        self.last_step = None;
        self.arc += self.config.dx;
        self.solver.forget();
        Ok(IrcReport {
            energy,
            max_force,
            arc: self.arc,
            inner_steps,
            // Sella: the kick outer step is never a minimum, even when
            // |F| is already under fmax (a TS neighborhood).
            at_minimum: !kicked
                && max_force <= self.config.force_tol
                && last_interior
                && self.arc > 8.0 * self.config.dx,
        })
    }

    /// Ishida--Morokuma--Komornicki PC in mass-weighted Cartesians.
    /// Matches `gpr_optim` `IRCDriver` `IRCMethod::Morokuma`:
    /// `x_pred = x - h g/|g|`, then `x += -h g_avg/|g_avg|`.
    fn step_morokuma<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        let kicked = self.first;
        if self.first {
            axpy(1.0, self.d1.view(), &mut self.x);
            self.first = false;
        }

        let (energy0, g0) = surface.eval(self.x.view())?;
        if !g0.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        let max_force0 = self.config.force_gate.value(g0.view());
        if !kicked && max_force0 <= self.config.force_tol {
            return Ok(IrcReport {
                energy: energy0,
                max_force: max_force0,
                arc: self.arc,
                inner_steps: 0,
                at_minimum: true,
            });
        }
        // Averaged-gradient corrector can keep pointing downhill
        // after the well is passed. A force flip against the last
        // accepted step past the 8-dx window is that overshoot;
        // |g| has often already grown on the far wall. The 8-dx
        // gate keeps the kick and the first downhill steps alive.
        if !kicked && self.arc > 8.0 * self.config.dx {
            if let Some(prev) = &self.last_outer {
                let mut force = g0.clone();
                force.mapv_inplace(|v| -v);
                if dot(prev.view(), force.view()) < 0.0 {
                    // Current point is already on the far wall. Keep
                    // the last downhill geometry.
                    axpy(-1.0, prev.view(), &mut self.x);
                    let (energy, g) = surface.eval(self.x.view())?;
                    let max_force = self.config.force_gate.value(g.view());
                    return Ok(IrcReport {
                        energy,
                        max_force,
                        arc: self.arc,
                        inner_steps: 0,
                        at_minimum: true,
                    });
                }
            }
        }

        let g_mw = to_mw(&g0, &self.sqrtm, true);
        let gn = nrm2(g_mw.view());
        if gn <= 1e-16 {
            return Ok(IrcReport {
                energy: energy0,
                max_force: max_force0,
                arc: self.arc,
                inner_steps: 0,
                at_minimum: !kicked,
            });
        }

        let h = self.config.dx;
        let mut x_pred = self.x.clone();
        for i in 0..x_pred.len().min(self.sqrtm.len()) {
            x_pred[i] += (-h * g_mw[i] / gn) / self.sqrtm[i].max(1e-16);
        }
        let ev_pred = surface.eval(x_pred.view())?;
        if !ev_pred.1.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        let g_pred_mw = to_mw(&ev_pred.1, &self.sqrtm, true);
        let mut g_avg = g_mw;
        axpy(1.0, g_pred_mw.view(), &mut g_avg);
        g_avg.mapv_inplace(|v| 0.5 * v);
        let ga = nrm2(g_avg.view());
        if ga <= 1e-12 {
            return Ok(IrcReport {
                energy: energy0,
                max_force: max_force0,
                arc: self.arc,
                inner_steps: 1,
                at_minimum: !kicked,
            });
        }

        let mut s = Array1::zeros(self.x.len());
        let mut dx_mw_n2 = 0.0;
        for i in 0..s.len().min(self.sqrtm.len()) {
            let dmw = -h * g_avg[i] / ga;
            dx_mw_n2 += dmw * dmw;
            s[i] = dmw / self.sqrtm[i].max(1e-16);
        }
        if let Some(prev) = &self.last_outer {
            if self.arc > 8.0 * h && dot(prev.view(), s.view()) < 0.0 {
                return Ok(IrcReport {
                    energy: energy0,
                    max_force: max_force0,
                    arc: self.arc,
                    inner_steps: 1,
                    at_minimum: true,
                });
            }
        }
        axpy(1.0, s.view(), &mut self.x);
        let ev = surface.eval(self.x.view())?;
        if !ev.1.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        let max_force = self.config.force_gate.value(ev.1.view());
        self.last_outer = Some(s);
        self.d1.fill(0.0);
        self.arc += dx_mw_n2.sqrt();
        Ok(IrcReport {
            energy: ev.0,
            max_force,
            arc: self.arc,
            inner_steps: 2,
            at_minimum: !kicked && max_force <= self.config.force_tol,
        })
    }

    /// Convenience loop over [`IrcSession::step`]; nothing more.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<IrcReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut n = 1;
        while !report.at_minimum && n < max_steps {
            report = self.step(surface)?;
            n += 1;
        }
        Ok(report)
    }
}

/// Mass-weighted path-orthogonal force. Algebra is `rgmin::vecops`.
fn path_projected_force(g: &Array1<f64>, d1: &Array1<f64>, sqrtm: &Array1<f64>) -> Array1<f64> {
    let n = g.len().min(d1.len()).min(sqrtm.len());
    let mut d1m = Array1::zeros(n);
    let mut gm = Array1::zeros(n);
    for i in 0..n {
        d1m[i] = d1[i] * sqrtm[i];
        gm[i] = g[i] / sqrtm[i].max(1e-16);
    }
    let dn = nrm2(d1m.view());
    if dn > 1e-16 {
        let proj = dot(d1m.view(), gm.view()) / dn;
        axpy(-proj / dn, d1m.view(), &mut gm);
    }
    let mut force = Array1::zeros(g.len());
    for i in 0..n {
        force[i] = -gm[i] * sqrtm[i];
    }
    force
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Well;

    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    fn session(dx: f64, masses: Array1<f64>, saddle: Array1<f64>) -> IrcSession {
        let mut mode = Array1::zeros(saddle.len());
        mode[0] = 1.0;
        IrcSession::new(
            IrcConfig {
                dx,
                force_tol: 1e-3,
                ..IrcConfig::default()
            },
            saddle,
            masses,
            mode,
            IrcDirection::Forward,
        )
        .unwrap()
    }

    fn mw_radius(x: &Array1<f64>, center: &Array1<f64>, masses: &Array1<f64>) -> f64 {
        let s = x - center;
        IrcTrust::from_atom_masses(Array1::zeros(x.len()), masses.as_slice().unwrap(), 0.0).cons(&s)
    }

    #[test]
    fn default_kind_is_gs2() {
        assert_eq!(IrcConfig::default().kind, IrcKind::Gs2);
    }

    #[test]
    fn euclidean_session_kicks_an_arbitrary_dimension_by_dx() {
        let dx = 0.15;
        let saddle = Array1::zeros(5);
        let mode: Array1<f64> = Array1::from(vec![1.0, 2.0, -1.0, 0.5, 0.25]);
        let unit_mode = &mode / mode.dot(&mode).sqrt();
        let mut irc = IrcSession::new_euclidean(
            IrcConfig {
                dx,
                force_tol: 1e-12,
                max_inner: 1,
                ..IrcConfig::default()
            },
            saddle.clone(),
            mode,
            IrcDirection::Forward,
        )
        .unwrap();

        let _ = irc.step(&Flat).unwrap();
        let displacement = irc.position().to_owned() - &saddle;

        assert!((displacement.dot(&displacement).sqrt() - dx).abs() < 1e-12);
        assert!((displacement.dot(&unit_mode) - dx).abs() < 1e-12);
    }

    #[test]
    fn euclidean_morokuma_corrector_has_length_dx_in_five_dimensions() {
        let dx = 0.15;
        let saddle = Array1::zeros(5);
        let mut mode = Array1::zeros(5);
        mode[0] = 1.0;
        let mut irc = IrcSession::new_euclidean(
            IrcConfig {
                dx,
                force_tol: 1e-12,
                kind: IrcKind::Morokuma,
                ..IrcConfig::default()
            },
            saddle,
            mode,
            IrcDirection::Forward,
        )
        .unwrap();

        let _ = irc.step(&Well).unwrap();
        let kicked = irc.position().to_owned();
        let _ = irc.step(&Well).unwrap();
        let displacement = irc.position().to_owned() - &kicked;

        assert!((displacement.dot(&displacement).sqrt() - dx).abs() < 1e-8);
    }

    #[test]
    fn morokuma_corrector_has_mw_length_dx() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 1.0]);
        let saddle = Array1::zeros(6);
        let mut irc = IrcSession::new(
            IrcConfig {
                dx,
                force_tol: 1e-12,
                kind: IrcKind::Morokuma,
                ..IrcConfig::default()
            },
            saddle.clone(),
            masses.clone(),
            {
                let mut mode = Array1::zeros(6);
                mode[0] = 1.0;
                mode
            },
            IrcDirection::Forward,
        )
        .unwrap();
        let x0 = irc.position().to_owned();
        let _ = irc.step(&Well).unwrap();
        let kicked = irc.position().to_owned();
        let _ = irc.step(&Well).unwrap();
        let x2 = irc.position().to_owned();
        let r = mw_radius(&x2, &kicked, &masses);
        assert!(
            (r - dx).abs() < 1e-8,
            "Morokuma ||dx||_M={r} h={dx} from {kicked:?} to {x2:?} kick-from {x0:?}"
        );
    }

    #[test]
    fn kick_lands_on_the_mw_sphere() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 4.0]);
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut irc = session(dx, masses.clone(), saddle.clone());
        let _ = irc.step(&Well).unwrap();
        let r = mw_radius(&irc.position().to_owned(), &saddle, &masses);
        assert!((r - dx).abs() < 1e-10, "cons={r} dx={dx}");
    }

    struct Flat;

    impl PointSurface for Flat {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.0, Array1::zeros(x.len())))
        }
    }

    #[test]
    fn unequal_mass_kick_is_not_the_euclidean_sphere() {
        let dx = 0.2;
        let masses = Array1::from(vec![1.0, 16.0]);
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut mode = Array1::zeros(6);
        mode[0] = 1.0;
        mode[3] = 1.0;
        let mut irc = IrcSession::new(
            IrcConfig {
                dx,
                force_tol: 1e-12,
                max_inner: 1,
                ..IrcConfig::default()
            },
            saddle.clone(),
            masses.clone(),
            mode,
            IrcDirection::Forward,
        )
        .unwrap();
        let _ = irc.step(&Flat).unwrap();
        let x = irc.position().to_owned();
        let r = mw_radius(&x, &saddle, &masses);
        assert!((r - dx).abs() < 1e-10, "cons={r} dx={dx}");
        let eucl = (&x - &saddle).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (eucl - dx).abs() > 1e-6,
            "unequal-mass kick must not sit on the Euclidean sphere (eucl={eucl})"
        );
    }

    #[test]
    fn inner_gs2_step_stays_on_the_mw_sphere() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 4.0]);
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut irc = session(dx, masses.clone(), saddle.clone());
        let _ = irc.step(&Well).unwrap();
        let kicked = irc.position().to_owned();
        let _report = irc.step(&Well).unwrap();
        let r = mw_radius(&irc.position().to_owned(), &kicked, &masses);
        assert!(
            (r - dx).abs() < 1e-8,
            "inner step left the MW sphere: cons={r} dx={dx}"
        );
    }
}
