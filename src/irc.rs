//! Intrinsic reaction coordinate session: roll from a first-order
//! saddle to a minimum, one branch at a time.
//!
//! The two sphere kinds keep each inner step on an rgmin `IrcTrust`
//! sphere in the selected coordinate metric: [`IrcKind::SellaSphere`]
//! about the last accepted point (Sella IRCTrustRegion), and
//! [`IrcKind::Gs2`] about the Gonzalez and Schlegel pivot half a step
//! ahead. [`IrcSession::new`] weights atomistic Cartesian coordinates by
//! the atom masses; [`IrcSession::new_euclidean`] selects unit weights
//! for a generic-dimensional surface. Each sphere increment is Sella's
//! QuasiNewtonIRC step, rgmin `qn_irc_get_s` at the level shift that
//! puts it on the sphere.
//! [`IrcKind::Morokuma`] is the
//! Ishida--Morokuma--Komornicki predictor-corrector from
//! `gpr_optim` `IRCDriver`. Ambient algebra goes through
//! `rgmin::vecops`. The host owns the loop. `run` is a convenience
//! over `step`.

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::vecops::{Vector, axpy, dot, nrm2, nrminf};
use rgmin::{
    ApplyHessian, BfgsModel, EigenParams, EigensolverKind, IrcTrust, mw_pair, qn_irc_get_s,
    sqrt_masses_3n, to_mw,
};

use std::cell::RefCell;

use crate::eigensolve::{DENSE_JACOBI_LIMIT, faer_eigh};
use crate::error::SaddleError;
use crate::kappa::Complement;
use crate::minmode::PointSurface;

/// rgmin `BfgsModel::update` keeps a pair only when `y.s` and `|s.B s|`
/// exceed this. [`IrcSession::update_model`] passes each pair at unit
/// length, so the floor is on the curvature along the step.
const BFGS_CURVATURE: f64 = 1e-12;

/// Which side of the imaginary mode to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrcDirection {
    /// The Forward.
    Forward,
    /// The Reverse.
    Reverse,
}

/// How an outer IRC increment is formed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IrcKind {
    /// Gonzalez and Schlegel, J. Chem. Phys. 90, 2154 (1989),
    /// doi:10.1063/1.456010. The next point is the energy minimum on the
    /// sphere of radius `dx / 2` about the pivot `x_k - (dx / 2) g / |g|`,
    /// with the gradient and the step in the session metric. The first
    /// pivot is `dx / 2` from the saddle along the launch mode. The arc of
    /// a circle tangent to the gradient at both ends joins the two points,
    /// so the path error is second order in `dx`.
    Gs2,
    /// Ishida--Morokuma--Komornicki predictor-corrector
    /// (`gpr_optim` `IRCDriver` Morokuma).
    Morokuma,
    /// Sella `IRCTrustRegion` (Hermes, Sargsyan, Najm, and Zador, J. Chem.
    /// Theory Comput. 18, 6974 (2022), doi:10.1021/acs.jctc.2c00395). The
    /// next point is the energy minimum on the sphere of radius `dx` about
    /// the last accepted point, the first about the saddle. The path error
    /// is first order in `dx`. Default.
    #[default]
    SellaSphere,
}

impl IrcKind {
    /// C `rgsaddle_irc_kind_t`.
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Gs2 => 0,
            Self::Morokuma => 1,
            Self::SellaSphere => 2,
        }
    }
}

/// Outer IRC controls.
#[derive(Clone, Debug)]
pub struct IrcConfig {
    /// Sphere radius in the session metric (Sella `dx`) and Morokuma `h`.
    pub dx: f64,
    /// The search stops when the force, measured by `force_gate`, is at
    /// or below this value, in the host's force units.
    pub force_tol: f64,
    /// eOn / gpr_optim `ConvergenceForceNorm`.
    pub force_gate: crate::ForceGate,
    /// How many inner trials one sphere step may take.
    pub max_inner: usize,
    /// Finite-difference Hessian action length for the matrix-free kick.
    pub dr: f64,
    /// Krylov dimension for the lowest-mode kick.
    pub krylov_dim: usize,
    /// Closed lowest-mode backend. Unlinked kinds fail closed.
    pub eigen_kind: EigensolverKind,
    /// [`IrcKind::SellaSphere`] is the default.
    pub kind: IrcKind,
}

impl Default for IrcConfig {
    fn default() -> Self {
        Self {
            dx: 0.1,
            force_tol: 0.05,
            force_gate: crate::ForceGate::MaxForceOnAtom,
            max_inner: 10,
            dr: 1e-3,
            krylov_dim: 12,
            eigen_kind: EigensolverKind::Lanczos,
            kind: IrcKind::SellaSphere,
        }
    }
}

/// One accepted IRC point.
#[derive(Clone, Debug)]
pub struct IrcReport {
    /// Potential energy in the host's energy unit.
    pub energy: f64,
    /// Force measured by [`IrcConfig::force_gate`], in the host's force
    /// units.
    pub max_force: f64,
    /// The arc.
    pub arc: f64,
    /// The inner steps.
    pub inner_steps: usize,
    /// On a sphere step: past the kick, the force at or below `force_tol`,
    /// and positive curvature along the launch mode at this point. The
    /// sphere radius does not enter.
    pub at_minimum: bool,
}

/// Spectrum of the quasi-Newton model between two pair updates.
struct ModelSpectrum {
    evals: Array1<f64>,
    evecs: Array2<f64>,
    /// rgmin `BfgsModel::is_posdef` on these eigenvalues.
    posdef: bool,
}

/// What [`sphere_increment`] makes of a model minimum inside the sphere.
#[derive(Clone, Copy)]
enum Inside {
    /// The minimum itself.
    Keep,
    /// Its radial projection from the centre, as Sella.
    Radial,
    /// The point where the line of the step leaves the sphere.
    Chord,
}

/// Forward or reverse roll-down from a first-order saddle.
pub struct IrcSession {
    config: IrcConfig,
    x: Array1<f64>,
    x_saddle: Array1<f64>,
    sqrtm: Array1<f64>,
    d1: Array1<f64>,
    mode: Array1<f64>,
    hess: BfgsModel,
    /// Pairs that `hess` kept since its last reset.
    hess_pairs: usize,
    /// The requested sphere radius. A rejected trial halves
    /// `config.dx`; each later outer step without one doubles it back
    /// up to this value.
    dx_target: f64,
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
        if n3 == 0 || !n3.is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "saddle must be a nonzero 3N Cartesian".into(),
            ));
        }
        if masses.len() * 3 != n3 || mode.len() != n3 {
            return Err(SaddleError::Shape(
                "masses (N) and mode (3N) must match the saddle".into(),
            ));
        }
        Self::with_coordinate_metric(
            config,
            saddle,
            sqrt_masses_3n(masses.as_slice().unwrap_or(&[])),
            mode,
            direction,
        )
    }

    /// Build an IRC over an arbitrary-dimensional Euclidean surface.
    ///
    /// The native coordinates are the path metric, so every coordinate has
    /// unit mass weight. Use [`Self::new`] for atomistic Cartesian positions,
    /// where atomic masses define the metric.
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
        Self::with_coordinate_metric(config, saddle, Array1::ones(dimension), mode, direction)
    }

    fn with_coordinate_metric(
        config: IrcConfig,
        saddle: Array1<f64>,
        sqrtm: Array1<f64>,
        mode: Array1<f64>,
        direction: IrcDirection,
    ) -> Result<Self, SaddleError> {
        let dimension = saddle.len();
        if sqrtm.len() != dimension || mode.len() != dimension {
            return Err(SaddleError::Shape(
                "coordinate weights and mode must match the saddle".into(),
            ));
        }
        let mut session = Self {
            dx_target: config.dx,
            config,
            x: saddle.clone(),
            x_saddle: saddle,
            sqrtm,
            d1: Array1::zeros(dimension),
            mode,
            hess: BfgsModel::identity(dimension),
            hess_pairs: 0,
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

    /// Launch along the lowest eigenvector of the mass-weighted Hessian
    /// `M^{-1/2} H M^{-1/2}`, the direction in which Fukui's IRC leaves the
    /// saddle, from matrix-free Lanczos on finite-difference Hessian
    /// actions. `seed` and the surface's excluded modes are Cartesian.
    pub fn from_surface<S: crate::minmode::PointSurface>(
        config: IrcConfig,
        saddle: Array1<f64>,
        masses: Array1<f64>,
        seed: Array1<f64>,
        direction: IrcDirection,
        surface: &S,
    ) -> Result<Self, SaddleError> {
        let n3 = saddle.len();
        if n3 == 0 || masses.len() * 3 != n3 || seed.len() != n3 {
            return Err(SaddleError::Shape(
                "masses (N) and seed (3N) must match a nonzero saddle".into(),
            ));
        }
        let (_, g0) = surface.eval(saddle.view())?;
        let params = EigenParams {
            kind: config.eigen_kind,
            krylov: config.krylov_dim,
            ..EigenParams::default()
        };
        let (mode, _curv, _actions) = lowest_via_fd(
            surface,
            saddle.view(),
            g0.view(),
            &sqrt_masses_3n(masses.as_slice().unwrap_or(&[])),
            seed,
            config.dr,
            params,
        )?;
        Self::new(config, saddle, masses, mode, direction)
    }

    /// Cartesian coordinates, length 3N, in the host's length unit.
    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.hess.forget();
        self.hess_pairs = 0;
        self.seed_ts_mode();
    }

    /// Restore the saddle and flip the kick sign.
    pub fn set_direction(&mut self, direction: IrcDirection) {
        self.config.dx = self.dx_target;
        self.x = self.x_saddle.clone();
        self.d1 = self.kick_vector(direction);
        self.first = true;
        self.arc = 0.0;
        self.last_step = None;
        self.last_outer = None;
        self.hess.forget();
        self.hess_pairs = 0;
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
        if let Some(&v) = v0ts.iter().find(|v| v.abs() > 1e-16)
            && v < 0.0
        {
            v0ts.mapv_inplace(|c| -c);
        }
        match direction {
            IrcDirection::Forward => v0ts,
            IrcDirection::Reverse => v0ts.mapv(|c| -c),
        }
    }

    /// Radius of the inner sphere, `dx / 2` about the [`IrcKind::Gs2`] pivot.
    fn radius(&self) -> f64 {
        match self.config.kind {
            IrcKind::Gs2 => 0.5 * self.config.dx,
            _ => self.config.dx,
        }
    }

    fn trust(&self) -> IrcTrust {
        IrcTrust {
            d1: self.d1.clone(),
            sqrtm: self.sqrtm.clone(),
            dx: self.radius(),
        }
    }

    /// `x_k` less the [`IrcKind::Gs2`] pivot, `(dx / 2) g / |g|` in the
    /// session metric, as a coordinate displacement.
    fn pivot_offset(&self, g: &Array1<f64>) -> Array1<f64> {
        let g_mw = to_mw(g, &self.sqrtm, true);
        let norm = nrm2(g_mw.view());
        if norm <= 1e-16 {
            return Array1::zeros(g.len());
        }
        to_mw(&(g_mw * (self.radius() / norm)), &self.sqrtm, true)
    }

    /// Sella: project `g` orthogonal to the path in the MW metric.
    fn path_force(&self, g: &Array1<f64>) -> Array1<f64> {
        path_projected_force(g, &self.d1, &self.sqrtm)
    }

    /// One eigendecomposition of the mass-weighted quasi-Newton model,
    /// shared by every test of an inner trial. Above `DENSE_JACOBI_LIMIT`
    /// it is faer's, as for [`crate::eigensolve::exact_eigh`], since
    /// cyclic Jacobi there costs far more than a surface call. A build
    /// without the `faer` feature keeps rgmin's Jacobi at every size, so a
    /// large session runs slowly there instead of panicking.
    fn model_spectrum(&self) -> Result<ModelSpectrum, SaddleError> {
        let b = self.hess.hessian();
        let (evals, evecs) = if cfg!(feature = "faer") && b.nrows() > DENSE_JACOBI_LIMIT {
            faer_eigh(b)?
        } else {
            self.hess.eigh()
        };
        let posdef = self.hess_pairs > 0 && evals.iter().all(|&e| e > 1e-8);
        Ok(ModelSpectrum {
            evals,
            evecs,
            posdef,
        })
    }

    /// The BFGS update is unchanged when `s` and `y` scale together, while
    /// `y.s` and `s.B s` fall as the square of the trial length. The pair
    /// goes to rgmin at unit length, so its floors bound a curvature.
    fn update_model(&mut self, s_mw: &Array1<f64>, y_mw: &Array1<f64>) {
        let len = nrm2(s_mw.view());
        if len == 0.0 {
            return;
        }
        let s = s_mw / len;
        let y = y_mw / len;
        if dot(y.view(), s.view()) > BFGS_CURVATURE {
            self.hess_pairs += 1;
        }
        self.hess.update(&s, &y);
    }

    /// Sella `QuasiNewtonIRC` plus `IRCTrustRegion` on the current
    /// MW sphere. The BFGS model is identity until the first accepted
    /// inner pair; the sphere is `IrcTrust`.
    /// A [`IrcKind::Gs2`] point starts on its sphere uphill of the pivot,
    /// where the radial projection of a model minimum between the two is
    /// that point itself, so an interior Gs2 step runs on along its line.
    fn restricted_increment(&self, g: &Array1<f64>, spectrum: &ModelSpectrum) -> Array1<f64> {
        let inside = if spectrum.posdef && self.arc > 8.0 * self.config.dx {
            Inside::Keep
        } else if self.config.kind == IrcKind::Gs2 {
            Inside::Chord
        } else {
            Inside::Radial
        };
        sphere_increment(&self.trust(), spectrum, g, inside)
    }

    /// A minimum on a sphere branch: past the kick, the force at or below
    /// `force_tol`, and positive curvature along the saddle mode on the
    /// surface itself.
    /// Beside the saddle that curvature is still the negative eigenvalue,
    /// even where the force is small and the quasi-Newton model is positive
    /// definite, and a rejected trial halves `dx` there. The curvature
    /// costs one surface call, made only once the force is at or below
    /// `force_tol`.
    fn sphere_minimum<S: PointSurface>(
        &self,
        surface: &S,
        kicked: bool,
        max_force: f64,
        g: &Array1<f64>,
    ) -> Result<bool, SaddleError> {
        if kicked || max_force > self.config.force_tol {
            return Ok(false);
        }
        let hv = hessian_action(
            surface,
            self.x.view(),
            g.view(),
            self.mode.view(),
            self.config.dr,
        )?;
        Ok(dot(self.mode.view(), hv.view()) > 0.0)
    }

    /// One outer IRC move. [`IrcKind::SellaSphere`] and [`IrcKind::Gs2`]
    /// minimize on a sphere; [`IrcKind::Morokuma`] is the IRCDriver
    /// predictor-corrector.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        if self.config.kind == IrcKind::Morokuma {
            return self.step_morokuma(surface);
        }
        self.step_sphere(surface)
    }

    fn step_sphere<S: PointSurface>(&mut self, surface: &S) -> Result<IrcReport, SaddleError> {
        let kicked = self.first;
        let x_from = self.x.clone();
        if self.first {
            axpy(1.0, self.d1.view(), &mut self.x);
            if self.config.kind == IrcKind::Gs2 {
                self.d1.mapv_inplace(|c| 0.5 * c);
            }
            self.first = false;
        }
        let x_start = self.x.clone();

        let (energy0, g0) = surface.eval(self.x.view())?;
        if !g0.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("irc gradient"));
        }
        if !kicked && self.config.kind == IrcKind::Gs2 {
            self.d1 = self.pivot_offset(&g0);
        }
        let max_force0 = self.config.force_gate.value(g0.view());
        let fmax_inner = self.config.force_tol.min(0.01);
        let mut inner_steps = 0;
        let mut energy = energy0;
        let mut g = g0;
        let mut max_force = max_force0;
        let mut rejected = false;
        for _ in 0..self.config.max_inner {
            inner_steps += 1;
            let spectrum = self.model_spectrum()?;
            let s = self.restricted_increment(&g, &spectrum);
            let interior = spectrum.posdef
                && self.arc > 8.0 * self.config.dx
                && self.trust().cons(&s) + 1e-8 < self.radius();
            if inner_steps == 1
                && let Some(prev) = &self.last_outer
            {
                // Reversal after a PD model and a long enough arc
                // is a well overshoot only when the force is already
                // under the host tolerance. A stale BFGS pair can
                // reverse on a molecular surface far from a basin.
                if spectrum.posdef
                    && self.arc > 8.0 * self.config.dx
                    && dot(prev.view(), s.view()) < 0.0
                    && self.sphere_minimum(surface, kicked, max_force, &g)?
                {
                    self.d1.fill(0.0);
                    self.last_step = None;
                    return Ok(IrcReport {
                        energy,
                        max_force,
                        arc: self.arc,
                        inner_steps,
                        at_minimum: true,
                    });
                }
            }
            if let Some(prev) = &self.last_step
                && spectrum.posdef
                && self.arc > 8.0 * self.config.dx
                && dot(prev.view(), s.view()) < 0.0
                && self.sphere_minimum(surface, kicked, max_force, &g)?
            {
                self.d1.fill(0.0);
                self.last_step = None;
                return Ok(IrcReport {
                    energy,
                    max_force,
                    arc: self.arc,
                    inner_steps,
                    at_minimum: true,
                });
            }
            self.last_step = Some(s.clone());
            axpy(1.0, s.view(), &mut self.d1);
            axpy(1.0, s.view(), &mut self.x);
            let ev = surface.eval(self.x.view())?;
            // A quasi-Newton displacement cannot raise the IRC energy beyond the tolerance.
            if ev.0 > energy + 1.0e-8 {
                axpy(-1.0, s.view(), &mut self.x);
                axpy(-1.0, s.view(), &mut self.d1);
                self.config.dx = (self.config.dx * 0.5).max(1.0e-4);
                self.last_step = None;
                rejected = true;
                break;
            }
            energy = ev.0;
            let y = &ev.1 - &g;
            let (s_mw, y_mw) = mw_pair(&s, &y, &self.sqrtm);
            self.update_model(&s_mw, &y_mw);
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
        self.arc += match self.config.kind {
            IrcKind::Gs2 => nrm2(to_mw(&(&self.x - &x_from), &self.sqrtm, false).view()),
            _ => self.config.dx,
        };
        if !rejected {
            self.config.dx = (2.0 * self.config.dx).min(self.dx_target);
        }
        // Sella skips the test while the session still sits on the saddle.
        // This also refuses the kick step itself, even where |F| is already
        // at or below fmax beside the saddle. A step whose last trial was
        // rejected still ends on an accepted point with this force.
        let at_minimum = self.sphere_minimum(surface, kicked, max_force, &g)?;
        Ok(IrcReport {
            energy,
            max_force,
            arc: self.arc,
            inner_steps,
            at_minimum,
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
        if !kicked
            && self.arc > 8.0 * self.config.dx
            && let Some(prev) = &self.last_outer
        {
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
        if let Some(prev) = &self.last_outer
            && self.arc > 8.0 * h
            && dot(prev.view(), s.view()) < 0.0
        {
            return Ok(IrcReport {
                energy: energy0,
                max_force: max_force0,
                arc: self.arc,
                inner_steps: 1,
                at_minimum: true,
            });
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

/// rgmin `qn_irc_restricted` with the level shift found by bisection.
///
/// In MW coordinates `s(alpha) + d1 = V w / (|L| + alpha)` with
/// `w = |L| V^T d1 - V^T g`, so its norm falls monotonically in `alpha` and
/// the root lies in `[|w| / dx - max |L|, |w| / dx]`. rgmin's Newton search
/// starts at `alpha = 0`, a pole where a model eigenvalue is zero to
/// round-off (its denominator floor is 1e-16). Each of its 64 iterations at
/// most doubles `alpha + 1e-16`, so its last trial lies below `2^63 1e-16`,
/// about 922: a larger root (a small `dx`) is out of reach, and the radial
/// projection of that step is not the minimum of the model on the sphere.
fn sphere_increment(
    trust: &IrcTrust,
    spectrum: &ModelSpectrum,
    g: &Array1<f64>,
    inside: Inside,
) -> Array1<f64> {
    let g_mw = to_mw(g, &trust.sqrtm, true);
    let d1_mw = to_mw(&trust.d1, &trust.sqrtm, false);
    let step_at = |alpha: f64| {
        let (s_mw, _) = qn_irc_get_s(&spectrum.evals, &spectrum.evecs, &g_mw, &d1_mw, alpha);
        to_mw(&s_mw, &trust.sqrtm, true)
    };
    let s0 = step_at(0.0);
    if trust.cons(&s0) <= trust.dx + 1e-14 {
        return match inside {
            Inside::Keep => s0,
            Inside::Radial => trust.project(&s0),
            Inside::Chord => chord_exit(&d1_mw, &s0, trust),
        };
    }
    let curvature = spectrum.evals.mapv(f64::abs);
    let w = &curvature * &spectrum.evecs.t().dot(&d1_mw) - spectrum.evecs.t().dot(&g_mw);
    let radius = |alpha: f64| {
        w.iter()
            .zip(&curvature)
            .map(|(wi, li)| (wi / (li + alpha)).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut hi = nrm2(w.view()) / trust.dx;
    let mut lo = (hi - curvature.fold(0.0, |a: f64, &b| a.max(b))).max(0.0);
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if radius(mid) > trust.dx {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    step_at(hi)
}

/// `s` scaled to `tau s / |s|` in MW coordinates, the root `tau >= 0` of
/// `|d1 + tau s / |s|| = dx`: past a model minimum inside the sphere to the
/// far side.
fn chord_exit(d1_mw: &Array1<f64>, s: &Array1<f64>, trust: &IrcTrust) -> Array1<f64> {
    let s_mw = to_mw(s, &trust.sqrtm, false);
    let len = nrm2(s_mw.view());
    if len <= 1e-16 {
        return trust.project(s);
    }
    let b = dot(d1_mw.view(), s_mw.view()) / len;
    let c = dot(d1_mw.view(), d1_mw.view()) - trust.dx * trust.dx;
    let tau = -b + (b * b - c).max(0.0).sqrt();
    s.mapv(|v| v * tau / len)
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

/// Finite-difference Hessian action `H v` at `x` with the forward
/// difference of gradients: one surface call per action.
fn hessian_action<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    v: ArrayView1<f64>,
    dr: f64,
) -> Result<Array1<f64>, SaddleError> {
    if let Some(hv) = surface.hessian_vector(x, v)? {
        return Ok(hv);
    }
    let magnitude = v.dot(&v).sqrt();
    if magnitude == 0.0 {
        return Ok(Array1::zeros(v.len()));
    }
    let unit = &v / magnitude;
    let shifted = &x + &(&unit * dr);
    let (_, g1) = surface.eval(shifted.view())?;
    Ok((&g1 - &g0) * (magnitude / dr))
}

/// `M^{-1/2} H M^{-1/2}` on the complement of the excluded directions.
struct FdHvp<'a, S: PointSurface> {
    surface: &'a S,
    g0: Array1<f64>,
    sqrtm: &'a Array1<f64>,
    dr: f64,
    fail: RefCell<Option<SaddleError>>,
    space: Complement,
}

impl<S: PointSurface> ApplyHessian for FdHvp<'_, S> {
    fn apply_hessian(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let cartesian = self.space.lift(v) / self.sqrtm;
        match hessian_action(self.surface, x, self.g0.view(), cartesian.view(), self.dr) {
            Ok(hv) => self.space.reduce((hv / self.sqrtm).view()),
            Err(e) => {
                self.fail.replace(Some(e));
                Array1::zeros(v.len())
            }
        }
    }
}

/// Lowest eigenpair of the mass-weighted Hessian at `x`, with the mode
/// returned as a Cartesian displacement `M^{-1/2} u`. A Cartesian excluded
/// direction `e` is the mass-weighted direction `M^{1/2} e`.
fn lowest_via_fd<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    sqrtm: &Array1<f64>,
    seed: Array1<f64>,
    dr: f64,
    params: EigenParams,
) -> Result<(Array1<f64>, f64, usize), SaddleError> {
    let excluded = surface.excluded_modes(x)?;
    if excluded.ncols() != x.len() || !excluded.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Shape(
            "minimum-mode excluded directions".into(),
        ));
    }
    let mut space = Complement::new(x.len());
    for direction in excluded.rows() {
        space.exclude((&direction * sqrtm).view());
    }
    if space.dimension() == 0 {
        return Err(SaddleError::Shape("minimum-mode space is empty".into()));
    }
    let mut reduced_seed = space.reduce((seed * sqrtm).view());
    if reduced_seed.dot(&reduced_seed).sqrt() <= 64.0 * f64::EPSILON {
        reduced_seed = Array1::from_iter(
            (0..space.dimension()).map(|i| ((i + 1) as f64 * 1.618033988749895).sin()),
        );
    }
    let h = FdHvp {
        surface,
        g0: g0.to_owned(),
        sqrtm,
        dr,
        fail: RefCell::new(None),
        space,
    };
    let out = match rgmin::lowest_mode(&h, x, reduced_seed.view(), &params) {
        Ok(o) => o,
        Err(e) => {
            if let Some(surf) = h.fail.into_inner() {
                return Err(surf);
            }
            return Err(SaddleError::Solver(format!("IRC lowest mode: {e}")));
        }
    };
    if let Some(surf) = h.fail.into_inner() {
        return Err(surf);
    }
    if !out.vector.iter().all(|v| v.is_finite()) || !out.value.is_finite() {
        return Err(SaddleError::NonFinite("lowest-mode eigenpair"));
    }
    let mode = h.space.lift(out.vector.view()) / sqrtm;
    Ok((mode, out.value, out.actions))
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
    fn default_kind_is_the_sella_sphere() {
        assert_eq!(IrcConfig::default().kind, IrcKind::SellaSphere);
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
    fn inner_sella_step_stays_on_the_mw_sphere() {
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

    #[test]
    fn gs2_arc_adds_the_mw_chord_of_each_step() {
        let masses = Array1::from(vec![1.0, 4.0]);
        let sqrtm = sqrt_masses_3n(masses.as_slice().unwrap());
        let saddle = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut mode = Array1::zeros(6);
        mode[0] = 1.0;
        mode[3] = 1.0;
        let mut irc = IrcSession::new(
            IrcConfig {
                dx: 0.15,
                force_tol: 1e-3,
                kind: IrcKind::Gs2,
                ..IrcConfig::default()
            },
            saddle.clone(),
            masses,
            mode,
            IrcDirection::Forward,
        )
        .unwrap();
        let mut last = saddle;
        let mut arc = 0.0;
        for _ in 0..4 {
            let report = irc.step(&Well).unwrap();
            let x = irc.position().to_owned();
            arc += nrm2(to_mw(&(&x - &last), &sqrtm, false).view());
            assert_eq!(report.arc, arc);
            last = x;
        }
    }

    #[test]
    fn interior_gs2_step_runs_on_where_the_radial_one_stays() {
        let dx = 0.2;
        let sqrtm = sqrt_masses_3n(&[1.0, 4.0]);
        let mut d1_mw = Array1::zeros(6);
        d1_mw[3] = dx;
        let trust = IrcTrust {
            d1: to_mw(&d1_mw, &sqrtm, true),
            sqrtm: sqrtm.clone(),
            dx,
        };
        let toward_centre = to_mw(&(&d1_mw * -0.3), &sqrtm, true);
        let radial = to_mw(&trust.project(&toward_centre), &sqrtm, false);
        assert!(nrm2(radial.view()) < 1e-12, "{radial}");
        let chord = to_mw(&chord_exit(&d1_mw, &toward_centre, &trust), &sqrtm, false);
        assert!(nrm2((&chord + &d1_mw * 2.0).view()) < 1e-12, "{chord}");

        let mut oblique_mw = &d1_mw * -0.3;
        oblique_mw[0] = 0.05;
        let s = chord_exit(&d1_mw, &to_mw(&oblique_mw, &sqrtm, true), &trust);
        assert!((trust.cons(&s) - dx).abs() < 1e-12);
        let s_mw = to_mw(&s, &sqrtm, false);
        let cos =
            dot(s_mw.view(), oblique_mw.view()) / (nrm2(s_mw.view()) * nrm2(oblique_mw.view()));
        assert!((cos - 1.0).abs() < 1e-12, "cos {cos}");
    }

    fn sorted(v: &Array1<f64>) -> Vec<f64> {
        let mut out = v.to_vec();
        out.sort_by(f64::total_cmp);
        out
    }

    fn assert_spectrum_matches_rgmin(irc: &IrcSession) {
        let spectrum = irc.model_spectrum().unwrap();
        assert_eq!(spectrum.posdef, irc.hess.is_posdef());
        let ours = sorted(&spectrum.evals);
        let rgmin = sorted(&irc.hess.eigh().0);
        for (a, b) in ours.iter().zip(&rgmin) {
            assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "{a} against {b}");
        }
    }

    #[test]
    fn model_spectrum_above_the_jacobi_limit_matches_rgmin() {
        let n = DENSE_JACOBI_LIMIT + 22;
        let mut mode = Array1::zeros(n);
        mode[0] = 1.0;
        let mut irc = IrcSession::new_euclidean(
            IrcConfig::default(),
            Array1::zeros(n),
            mode.clone(),
            IrcDirection::Forward,
        )
        .unwrap();
        assert_spectrum_matches_rgmin(&irc);
        assert!(!irc.model_spectrum().unwrap().posdef);

        let flat = Array1::from_iter((0..n).map(|i| ((i + 1) as f64 * 0.37).sin()));
        irc.update_model(&flat, &(-&flat));
        assert_eq!(irc.hess_pairs, 0);
        assert_spectrum_matches_rgmin(&irc);

        irc.update_model(&mode, &(&mode * 2.0));
        assert_eq!(irc.hess_pairs, 1);
        assert_spectrum_matches_rgmin(&irc);
        assert!(irc.model_spectrum().unwrap().posdef);

        let s = Array1::from_iter((0..n).map(|i| ((i + 1) as f64 * 1.3).cos()));
        let y = Array1::from_iter((0..n).map(|i| s[i] * (1.0 + 0.01 * i as f64)));
        irc.update_model(&s, &y);
        assert_eq!(irc.hess_pairs, 2);
        assert_spectrum_matches_rgmin(&irc);
    }

    #[test]
    fn a_pair_updates_the_model_alike_at_any_length() {
        let session = || {
            IrcSession::new_euclidean(
                IrcConfig::default(),
                Array1::zeros(2),
                Array1::from(vec![1.0, 0.0]),
                IrcDirection::Forward,
            )
            .unwrap()
        };
        let s = Array1::from(vec![0.6, 0.8]);
        let y = Array1::from(vec![0.6 * 1282.0, 0.8 * 3.0]);
        let mut unit = session();
        unit.update_model(&s, &y);
        assert_eq!(unit.hess_pairs, 1);
        // A power of two scales the pair without rounding.
        let short = 2f64.powi(-27);
        let mut scaled = session();
        scaled.update_model(&(&s * short), &(&y * short));
        assert_eq!(scaled.hess_pairs, 1);
        assert_eq!(scaled.hess.hessian(), unit.hess.hessian());
    }

    #[test]
    fn dx_doubles_back_to_the_requested_radius() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 1.0]);
        let mut irc = session(dx, masses, Array1::zeros(6));
        let _ = irc.step(&Well).unwrap();
        irc.config.dx = dx / 8.0;
        for expected in [dx / 4.0, dx / 2.0, dx, dx] {
            let report = irc.step(&Well).unwrap();
            assert!(!report.at_minimum);
            assert_eq!(irc.config.dx, expected);
        }
    }

    #[test]
    fn each_branch_starts_at_the_requested_radius() {
        let dx = 0.15;
        let masses = Array1::from(vec![1.0, 1.0]);
        let saddle = Array1::zeros(6);
        let mut irc = session(dx, masses.clone(), saddle.clone());
        irc.config.dx = dx / 16.0;
        irc.set_direction(IrcDirection::Reverse);
        assert_eq!(irc.config.dx, dx);
        let _ = irc.step(&Well).unwrap();
        let r = mw_radius(&irc.position().to_owned(), &saddle, &masses);
        assert!((r - dx).abs() < 1e-10, "cons={r} dx={dx}");
    }
}
