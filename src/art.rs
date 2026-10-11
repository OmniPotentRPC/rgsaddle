//! Activation-relaxation technique nouveau.
//!
//! From a local minimum the session applies one random or targeted
//! displacement, on every coordinate or on a local support, and pushes
//! along that direction until the lowest Lanczos curvature falls below
//! [`ArtConfig::eigen_threshold`]. It then climbs that mode and relaxes
//! the force in the perpendicular hyperplane with FIRE.
//!
//! After the threshold the climb is the min-mode reflection of Goswami,
//! Gunde and Jónsson, Front. Chem. (2026),
//! <https://doi.org/10.3389/fchem.2026.1807063>, where dimer translation
//! and the climbing image are one Householder reflection (Eqs. 1–9).
//! The push is the opening of the well in Barkema and Mousseau, Phys.
//! Rev. Lett. 77 (1996) 4358,
//! <https://doi.org/10.1103/PhysRevLett.77.4358>, and Malek and Mousseau,
//! Phys. Rev. E 62 (2000) 7723,
//! <https://doi.org/10.1103/PhysRevE.62.7723>. The Lanczos lowest mode is
//! the one Machado-Charry et al. follow, J. Chem. Phys. 135 (2011)
//! 034102, <https://doi.org/10.1063/1.3609924>. The perpendicular
//! relaxation is the pARTn FIRE hyperplane of Poberžnik et al., Comput.
//! Phys. Commun. 295 (2024) 108961,
//! <https://doi.org/10.1016/j.cpc.2023.108961>, driven by an rgmin FIRE
//! solver on the projected gradient.
//!
//! Lanczos is [`crate::MinModeSession::estimate_mode`] with
//! [`crate::MinModeKind::Lanczos`], including its finite-difference or
//! analytic Hessian action and any excluded directions. `new` does not
//! evaluate the surface. `step` is one macro-step. `run` repeats it.
//! After the curvature crosses the threshold the session climbs with
//! the min-mode reflected force. While the well is still closed, the
//! opening direction seeds the estimate so a higher eigenvector left
//! over from the previous estimate cannot hide the crossing. The first
//! estimate waits until the geometry has moved
//! [`ArtConfig::activation_distance`] from the undisplaced minimum.
//! Convergence is a force at or under the min-mode force tolerance
//! together with a negative lowest curvature, both at the current point.
//!
//! [`run_events`] evaluates many events at once. Event `i` carries
//! [`event_seed`], a counter-based stream key, and nothing is drawn
//! from a generator shared across threads. Outcomes are stored at the
//! event index and a failed event is reported at the lowest such index,
//! so the batch is the same for one thread, many threads, or a split of
//! the index range across ranks.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{
    FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface,
};

/// Where the initial displacement is allowed to be nonzero.
#[derive(Clone, Debug, PartialEq)]
pub enum ArtSupport {
    /// Every coordinate.
    Global,
    /// Coordinate indices that receive the displacement.
    Local(Vec<usize>),
}

impl ArtSupport {
    /// Atom `atom` alone when `radius` is zero, otherwise that atom and
    /// every atom no farther than `radius`. Positions are Cartesian
    /// triples. A zero radius does not pull in a second atom that sits
    /// on the same point.
    pub fn local_atoms(
        positions: ArrayView1<f64>,
        atom: usize,
        radius: f64,
    ) -> Result<Self, SaddleError> {
        if positions.is_empty() || !positions.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "local atom support expects Cartesian triples".into(),
            ));
        }
        if !(radius.is_finite() && radius >= 0.0) {
            return Err(SaddleError::Invalid(
                "local atom radius must be finite and non-negative".into(),
            ));
        }
        let n_atoms = positions.len() / 3;
        if atom >= n_atoms {
            return Err(SaddleError::Invalid(
                "local atom index is outside the system".into(),
            ));
        }
        let mut active = Vec::new();
        let origin = positions.slice(ndarray::s![3 * atom..3 * atom + 3]);
        for other in 0..n_atoms {
            let site = positions.slice(ndarray::s![3 * other..3 * other + 3]);
            let mut distance_sq = 0.0;
            for c in 0..3 {
                let delta = site[c] - origin[c];
                distance_sq += delta * delta;
            }
            let inside = other == atom || (radius > 0.0 && distance_sq <= radius * radius);
            if inside {
                active.extend([3 * other, 3 * other + 1, 3 * other + 2]);
            }
        }
        Ok(Self::Local(active))
    }
}

/// Initial kick off the minimum. The length is `amplitude`.
#[derive(Clone, Debug)]
pub enum ArtDisplacement {
    /// Isotropic direction from `seed`, masked by `support`.
    Random {
        seed: u64,
        support: ArtSupport,
        amplitude: f64,
    },
    /// Host direction, masked by `support`, then scaled to `amplitude`.
    Targeted {
        direction: Array1<f64>,
        support: ArtSupport,
        amplitude: f64,
    },
}

/// Tolerances and budgets for one ART nouveau search.
///
/// Defaults follow pARTn (Poberžnik et al., Comput. Phys. Commun. 295
/// (2024) 108961, <https://doi.org/10.1016/j.cpc.2023.108961>) and the
/// opening in Malek and Mousseau, Phys. Rev. E 62 (2000) 7723,
/// <https://doi.org/10.1103/PhysRevE.62.7723>. Lengths are in the host
/// unit. The estimate and the reflected climb are a [`MinModeConfig`]
/// on [`ArtConfig::mode`], the same fields a dimer or Lanczos search
/// uses. [`ArtConfig::builder`] checks the same bounds as
/// [`ArtSession::new`].
#[derive(Clone, Debug)]
pub struct ArtConfig {
    /// Lowest-mode estimate and the reflected climb.
    ///
    /// `dr`, `difference`, `rotation_tol`, and `krylov_dim` live here
    /// and nowhere else on this config. The default kind is Lanczos,
    /// `krylov_dim` is 16 (pARTn `lanczos_max_size`), and `max_move` is
    /// 0.3 so the push step is not shortened by the cap. `force_tol` is
    /// the stationary-point tolerance.
    pub mode: MinModeConfig,
    /// Leave the basin when the lowest curvature falls strictly below this.
    /// Default `-0.01`, pARTn `eigval_thr`.
    pub eigen_threshold: f64,
    /// Length of one basin push along the initial direction.
    /// Default `0.3`, pARTn's push step.
    pub push_step: f64,
    /// Basin pushes after the initial displacement. The search stops when this is spent while the curvature is still above the threshold.
    pub max_pushes: usize,
    /// Uphill macro-steps once the threshold has been crossed.
    pub max_climbs: usize,
    /// FIRE steps in the hyperplane after a basin push.
    /// Default 3. Malek and Mousseau relax perpendicular to the opening;
    /// pARTn drives that block with FIRE and stops after a fixed count
    /// (`nperp` limited, rather than a full minimization).
    pub perp_steps_push: usize,
    /// FIRE steps in the hyperplane after a reflected climb.
    pub perp_steps_climb: usize,
    /// Lanczos actions required before an opening estimate that stopped
    /// early is trusted. Default 3, pARTn `lanczos_min_size`. A shorter
    /// estimate is repeated from a direction orthogonal to that mode.
    pub krylov_min: usize,
    /// Distance from the undisplaced minimum before the first Lanczos call.
    /// Default `0.3`, one push step. pARTn `ninit` / `delr_thr`: the
    /// opening pushes and relaxes perpendicularly, and does not estimate
    /// the lowest curvature on every push.
    pub activation_distance: f64,
    /// First climb steps that blend the opening direction into the mode.
    /// Default 0, pARTn `nsmooth` left off. A positive value is the blend.
    pub nsmooth: usize,
    /// Length of [`ArtSession::push_over`], the step past a converged
    /// saddle along the unstable mode. Default 0, so `run` stops on the
    /// stationary point. pARTn's push over the saddle is this length.
    pub push_over: f64,
}

impl Default for ArtConfig {
    fn default() -> Self {
        Self {
            mode: MinModeConfig {
                kind: MinModeKind::Lanczos,
                krylov_dim: 16,
                max_move: 0.3,
                ..MinModeConfig::default()
            },
            eigen_threshold: -0.01,
            push_step: 0.3,
            max_pushes: 40,
            max_climbs: 80,
            perp_steps_push: 3,
            perp_steps_climb: 8,
            krylov_min: 3,
            activation_distance: 0.3,
            nsmooth: 0,
            push_over: 0.0,
        }
    }
}

impl ArtConfig {
    /// Start from [`ArtConfig::default`].
    pub fn builder() -> ArtConfigBuilder {
        ArtConfigBuilder {
            config: Self::default(),
        }
    }
}

/// Typed construction of [`ArtConfig`]. Each setter takes the field's
/// real type. [`ArtConfigBuilder::build`] returns [`SaddleError::Invalid`]
/// for a non-finite or non-positive bound.
#[derive(Clone, Debug)]
pub struct ArtConfigBuilder {
    config: ArtConfig,
}

impl ArtConfigBuilder {
    /// Replace the shared min-mode block.
    pub fn mode(mut self, mode: MinModeConfig) -> Self {
        self.config.mode = mode;
        self
    }
    pub fn eigen_threshold(mut self, value: f64) -> Self {
        self.config.eigen_threshold = value;
        self
    }
    pub fn force_tol(mut self, value: f64) -> Self {
        self.config.mode.force_tol = value;
        self
    }
    pub fn push_step(mut self, value: f64) -> Self {
        self.config.push_step = value;
        self
    }
    pub fn max_move(mut self, value: f64) -> Self {
        self.config.mode.max_move = value;
        self
    }
    pub fn max_pushes(mut self, value: usize) -> Self {
        self.config.max_pushes = value;
        self
    }
    pub fn max_climbs(mut self, value: usize) -> Self {
        self.config.max_climbs = value;
        self
    }
    pub fn perp_steps_push(mut self, value: usize) -> Self {
        self.config.perp_steps_push = value;
        self
    }
    pub fn perp_steps_climb(mut self, value: usize) -> Self {
        self.config.perp_steps_climb = value;
        self
    }
    pub fn dr(mut self, value: f64) -> Self {
        self.config.mode.dr = value;
        self
    }
    pub fn difference(mut self, value: FiniteDifference) -> Self {
        self.config.mode.difference = value;
        self
    }
    pub fn rotation_tol(mut self, value: f64) -> Self {
        self.config.mode.rotation_tol = value;
        self
    }
    pub fn krylov_dim(mut self, value: usize) -> Self {
        self.config.mode.krylov_dim = value;
        self
    }
    pub fn krylov_min(mut self, value: usize) -> Self {
        self.config.krylov_min = value;
        self
    }
    pub fn activation_distance(mut self, value: f64) -> Self {
        self.config.activation_distance = value;
        self
    }
    pub fn nsmooth(mut self, value: usize) -> Self {
        self.config.nsmooth = value;
        self
    }
    pub fn push_over(mut self, value: f64) -> Self {
        self.config.push_over = value;
        self
    }
    pub fn build(self) -> Result<ArtConfig, SaddleError> {
        validate_config(&self.config)?;
        Ok(self.config)
    }
}

/// Where a macro-step left the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtStatus {
    /// The lowest curvature is still at or above the threshold.
    Pushing,
    /// The threshold has been crossed and the step climbed the lowest mode.
    Climbing,
    /// Force gate at or under tolerance and the lowest curvature is negative.
    Converged,
    /// The push budget or the climb budget is spent.
    Exhausted,
}

/// One independent search. The displacement carries the event seed.
#[derive(Clone, Debug)]
pub struct ArtEvent {
    pub minimum: Array1<f64>,
    pub displacement: ArtDisplacement,
}

/// Outcome of one event, in the caller's event order.
#[derive(Clone, Debug)]
pub struct ArtOutcome {
    pub status: ArtStatus,
    pub position: Array1<f64>,
    pub energy: f64,
    pub curvature: f64,
    pub max_force: f64,
    pub pushes: usize,
    pub climbs: usize,
    /// `PointSurface::eval` calls for this event.
    pub evaluations: usize,
}

/// One macro-step.
#[derive(Clone, Debug)]
pub struct ArtReport {
    pub status: ArtStatus,
    pub max_force: f64,
    pub curvature: f64,
    pub energy: f64,
    /// Basin pushes taken after the initial displacement.
    pub pushes: usize,
    /// Climb macro-steps taken.
    pub climbs: usize,
    /// `PointSurface::eval` calls during this step.
    pub evaluations: usize,
    /// `PointSurface::eval` calls since [`ArtSession::new`].
    pub total_evaluations: usize,
    /// Centre, push, and perpendicular calls before the curvature crossed.
    /// Stays at the crossing value afterwards. An exhausted opening
    /// reports every call here.
    pub pre_inflection_evaluations: usize,
    /// Lanczos calls in [`ArtReport::pre_inflection_evaluations`].
    pub pre_inflection_lanczos: usize,
    /// Centre evaluations of the opening, the push-phase geometry calls.
    pub pre_inflection_push: usize,
    /// Perpendicular FIRE calls during the opening.
    pub pre_inflection_perp: usize,
    /// Cumulative Lanczos `PointSurface::eval` calls.
    pub lanczos_evaluations: usize,
    /// Cumulative perpendicular FIRE calls.
    pub perp_evaluations: usize,
    /// Cumulative centre evaluations.
    pub centre_evaluations: usize,
    pub iteration: usize,
}

#[derive(Clone)]
struct Centre {
    x: Array1<f64>,
    energy: f64,
    gradient: Array1<f64>,
}

/// ART nouveau from one minimum.
///
/// The position after [`ArtSession::new`] is the displaced minimum.
/// The host calls [`ArtSession::step`] or [`ArtSession::run`].
pub struct ArtSession {
    config: ArtConfig,
    force_gate: ForceGate,
    x: Array1<f64>,
    /// Unit vector of the opening displacement. Basin pushes stay on it.
    push_direction: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    energy: f64,
    modes: MinModeSession,
    /// Undisplaced minimum. Lanczos waits until the geometry has moved
    /// at least [`ArtConfig::activation_distance`] from here.
    origin: Array1<f64>,
    pushes: usize,
    climbs: usize,
    /// Set once the lowest curvature has fallen below the threshold.
    /// Later macro-steps stay on the lowest mode.
    opened: bool,
    /// Latched once the opening has passed `activation_distance`.
    lanczos_armed: bool,
    evaluations: usize,
    pre_inflection_evaluations: usize,
    pre_inflection_lanczos: usize,
    pre_inflection_push: usize,
    pre_inflection_perp: usize,
    pre_inflection_locked: bool,
    lanczos_evaluations: usize,
    perp_evaluations: usize,
    centre_evaluations: usize,
    iteration: usize,
    centre: Option<Centre>,
}

impl ArtSession {
    /// Displace `minimum` and build the Lanczos session. No surface call.
    pub fn new(
        config: ArtConfig,
        minimum: Array1<f64>,
        displacement: ArtDisplacement,
    ) -> Result<Self, SaddleError> {
        validate_config(&config)?;
        if minimum.is_empty() || !minimum.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Invalid(
                "minimum must be non-empty and finite".into(),
            ));
        }
        let (direction, amplitude) = resolve_displacement(&displacement, minimum.len())?;
        let x = &minimum + &(&direction * amplitude);
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("displaced position"));
        }
        let modes = MinModeSession::new(mode_config(&config), x.clone(), direction.clone())?;
        Ok(Self {
            config,
            force_gate: ForceGate::LinfNorm,
            x,
            push_direction: direction.clone(),
            mode: direction,
            curvature: f64::NAN,
            energy: f64::NAN,
            modes,
            origin: minimum,
            pushes: 0,
            climbs: 0,
            opened: false,
            lanczos_armed: false,
            evaluations: 0,
            pre_inflection_evaluations: 0,
            pre_inflection_lanczos: 0,
            pre_inflection_push: 0,
            pre_inflection_perp: 0,
            pre_inflection_locked: false,
            lanczos_evaluations: 0,
            perp_evaluations: 0,
            centre_evaluations: 0,
            iteration: 0,
            centre: None,
        })
    }

    /// One displacement of [`ArtConfig::push_over`] along the unstable
    /// mode, aligned with the opening direction. No surface call. The
    /// length 0 leaves the saddle where `run` stopped.
    pub fn push_over(&mut self) -> Result<(), SaddleError> {
        let length = self.config.push_over;
        if length == 0.0 {
            return Ok(());
        }
        if !(self.curvature.is_finite() && self.curvature < 0.0) {
            return Err(SaddleError::Invalid(
                "push over the saddle needs a negative curvature".into(),
            ));
        }
        let sign = if self.push_direction.dot(&self.mode) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let x = &self.x + &(&self.mode * (sign * length));
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("art push over"));
        }
        self.x = x;
        self.centre = None;
        Ok(())
    }

    /// Select the convergence norm. The default is the maximum absolute component.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        self.force_gate = gate;
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Lowest mode from the last Lanczos estimate. Before the first step
    /// this is the opening direction.
    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    /// Unit opening direction. Basin pushes follow it.
    pub fn push_direction(&self) -> ArrayView1<'_, f64> {
        self.push_direction.view()
    }

    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    pub fn energy(&self) -> f64 {
        self.energy
    }

    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// One macro-step: Lanczos, then a basin push or a climb, then FIRE
    /// in the perpendicular hyperplane.
    ///
    /// A surface error leaves the position where it was. Evaluations
    /// made before that error still count.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<ArtReport, SaddleError> {
        let calls = AtomicUsize::new(0);
        let result = self.drive(surface, &calls);
        self.evaluations += calls.load(Ordering::Relaxed);
        result.map(|mut report| {
            report.evaluations = calls.load(Ordering::Relaxed);
            report.total_evaluations = self.evaluations;
            report
        })
    }

    fn drive<S: PointSurface>(
        &mut self,
        surface: &S,
        calls: &AtomicUsize,
    ) -> Result<ArtReport, SaddleError> {
        let mut x = self.x.clone();
        let centre_before = calls.load(Ordering::Relaxed);
        let (energy, gradient) = self.centre_at(surface, x.view(), calls)?;
        self.centre_evaluations += calls.load(Ordering::Relaxed) - centre_before;
        self.energy = energy;
        self.centre = Some(Centre {
            x: x.clone(),
            energy,
            gradient: gradient.clone(),
        });
        let max_force = self.force_gate.value(gradient.view());
        self.arm_lanczos();
        let _curvature = if self.lanczos_armed || self.opened {
            let estimate = self.lanczos(surface, calls)?;
            self.mode = estimate.mode;
            self.curvature = estimate.curvature;
            if max_force <= self.config.mode.force_tol && estimate.curvature < 0.0 {
                self.note_pre_inflection(true);
                return Ok(self.commit(x, calls, ArtStatus::Converged, max_force));
            }
            if !self.opened && estimate.curvature < self.config.eigen_threshold {
                self.opened = true;
                self.note_pre_inflection(true);
            }
            estimate.curvature
        } else {
            self.curvature
        };
        let climbing = self.opened;
        if climbing && self.climbs >= self.config.max_climbs {
            return Ok(self.commit(x, calls, ArtStatus::Exhausted, max_force));
        }
        if !climbing && self.pushes >= self.config.max_pushes {
            self.note_pre_inflection(true);
            return Ok(self.commit(x, calls, ArtStatus::Exhausted, max_force));
        }
        let mut tau = if climbing {
            self.mode.clone()
        } else {
            self.push_direction.clone()
        };
        if climbing && self.climbs < self.config.nsmooth {
            tau = blend_opening(&tau, &self.push_direction);
        }

        let perp_cap = self.config.mode.max_move;
        let perp_steps = if climbing {
            self.modes.set_force_gate(self.force_gate);
            self.modes.set_position(x.clone(), Some(gradient.clone()))?;
            self.modes.plant_energy(energy);
            self.modes.set_mode(tau.clone())?;
            let translated = self.modes.translate(surface)?;
            calls.fetch_add(translated.evaluations, Ordering::Relaxed);
            self.centre_evaluations += translated.evaluations;
            x = self.modes.position().to_owned();
            self.mode = self.modes.mode().to_owned();
            self.curvature = translated.curvature;
            self.energy = self.modes.energy().unwrap_or(f64::NAN);
            self.climbs += 1;
            if translated.status == MinModeStatus::Converged {
                self.note_pre_inflection(true);
                return Ok(self.commit(x, calls, ArtStatus::Converged, translated.max_force));
            }
            self.config.perp_steps_climb
        } else {
            let step = self.config.push_step.min(self.config.mode.max_move);
            x = &x + &(&tau * step);
            if !x.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("art step"));
            }
            self.pushes += 1;
            self.config.perp_steps_push
        };
        let perp_before = calls.load(Ordering::Relaxed);
        let centre = match relax_perpendicular(
            surface,
            &mut x,
            tau.view(),
            PerpBlock {
                steps: perp_steps,
                max_move: perp_cap,
                force_tol: self.config.mode.force_tol,
                gate: self.force_gate,
            },
            calls,
        ) {
            Ok(centre) => centre,
            Err(err) => {
                if climbing {
                    self.climbs -= 1;
                } else {
                    self.pushes -= 1;
                }
                return Err(err);
            }
        };
        self.perp_evaluations += calls.load(Ordering::Relaxed) - perp_before;
        if climbing && centre.x != self.modes.position() {
            self.modes.reset();
            self.modes
                .set_position(centre.x.clone(), Some(centre.gradient.clone()))?;
            self.modes.plant_energy(centre.energy);
        }
        if !self.opened {
            self.note_pre_inflection(false);
        }
        let status = if climbing {
            ArtStatus::Climbing
        } else {
            ArtStatus::Pushing
        };
        let reported_force = self.force_gate.value(centre.gradient.view());
        self.energy = centre.energy;
        self.centre = Some(centre);
        Ok(self.commit(x, calls, status, reported_force))
    }

    /// Repeat [`ArtSession::step`] until convergence, exhaustion, or `max_steps`.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<ArtReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut taken = 1;
        while !matches!(report.status, ArtStatus::Converged | ArtStatus::Exhausted)
            && taken < max_steps
        {
            report = self.step(surface)?;
            taken += 1;
        }
        Ok(report)
    }

    fn centre_at<S: PointSurface>(
        &self,
        surface: &S,
        x: ArrayView1<f64>,
        calls: &AtomicUsize,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        if let Some(centre) = &self.centre
            && centre.x == x
        {
            return Ok((centre.energy, centre.gradient.clone()));
        }
        let (energy, gradient) = checked_eval(surface, x, calls)?;
        Ok((energy, gradient))
    }

    fn commit(
        &mut self,
        x: Array1<f64>,
        calls: &AtomicUsize,
        status: ArtStatus,
        max_force: f64,
    ) -> ArtReport {
        self.x = x;
        self.iteration += 1;
        let evaluations = calls.load(Ordering::Relaxed);
        ArtReport {
            status,
            max_force,
            curvature: self.curvature,
            energy: self.energy,
            pushes: self.pushes,
            climbs: self.climbs,
            evaluations,
            total_evaluations: self.evaluations + evaluations,
            pre_inflection_evaluations: self.pre_inflection_evaluations,
            pre_inflection_lanczos: self.pre_inflection_lanczos,
            pre_inflection_push: self.pre_inflection_push,
            pre_inflection_perp: self.pre_inflection_perp,
            lanczos_evaluations: self.lanczos_evaluations,
            perp_evaluations: self.perp_evaluations,
            centre_evaluations: self.centre_evaluations,
            iteration: self.iteration,
        }
    }

    fn arm_lanczos(&mut self) {
        if self.lanczos_armed || self.opened {
            self.lanczos_armed = true;
            return;
        }
        if self.config.activation_distance <= 0.0
            || travelled(&self.x, &self.origin) >= self.config.activation_distance
        {
            self.lanczos_armed = true;
        }
    }

    fn note_pre_inflection(&mut self, lock: bool) {
        if self.pre_inflection_locked {
            return;
        }
        self.pre_inflection_evaluations =
            self.centre_evaluations + self.lanczos_evaluations + self.perp_evaluations;
        self.pre_inflection_lanczos = self.lanczos_evaluations;
        self.pre_inflection_push = self.centre_evaluations;
        self.pre_inflection_perp = self.perp_evaluations;
        if lock {
            self.pre_inflection_locked = true;
        }
    }

    fn lanczos<S: PointSurface>(
        &mut self,
        surface: &S,
        calls: &AtomicUsize,
    ) -> Result<crate::minmode::ModeEstimate, SaddleError> {
        self.modes.set_position(
            self.x.clone(),
            Some(self.centre.as_ref().unwrap().gradient.clone()),
        )?;
        let seed = if self.opened {
            self.mode.clone()
        } else {
            self.push_direction.clone()
        };
        self.modes.set_mode(seed)?;
        let mut estimate = self.modes.estimate_mode(surface)?;
        calls.fetch_add(estimate.evaluations, Ordering::Relaxed);
        self.lanczos_evaluations += estimate.evaluations;
        if !self.opened
            && estimate.rotations < self.config.krylov_min
            && self.x.len() > estimate.rotations
        {
            self.modes.set_mode(orthogonal_probe(&estimate.mode))?;
            let second = self.modes.estimate_mode(surface)?;
            calls.fetch_add(second.evaluations, Ordering::Relaxed);
            self.lanczos_evaluations += second.evaluations;
            if second.curvature < estimate.curvature {
                estimate = second;
            }
        }
        Ok(estimate)
    }
}

/// Seed for event `index` under a host `base`.
///
/// The index is mixed in full before the splitmix step, so neighbouring
/// events do not share a generator. The value depends only on the pair.
pub fn event_seed(base: u64, index: u64) -> u64 {
    splitmix64(base ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Run independent searches.
///
/// `threads` is the width of the decomposition. Event `i` runs from its
/// own displacement, on a round-robin bucket when `threads` is greater
/// than one, and the outcome is stored at index `i`. A contiguous split
/// of `events` (one slice per rank) concatenates to the same vector.
/// The [`ForceGate`] is the session default, the maximum absolute
/// component. The returned error, if any, is the one at the lowest
/// event index.
pub fn run_events<S: PointSurface + Sync>(
    config: &ArtConfig,
    events: &[ArtEvent],
    surface: &S,
    max_steps: usize,
    threads: usize,
) -> Result<Vec<ArtOutcome>, SaddleError> {
    let n = events.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let width = threads.max(1).min(n);
    if width == 1 {
        let mut out = Vec::with_capacity(n);
        for event in events {
            out.push(execute_event(config, event, surface, max_steps)?);
        }
        return Ok(out);
    }
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); width];
    for index in 0..n {
        buckets[index % width].push(index);
    }
    let mut slots: Vec<Option<Result<ArtOutcome, SaddleError>>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(width);
        for bucket in &buckets {
            joins.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(bucket.len());
                for &index in bucket {
                    local.push((
                        index,
                        execute_event(config, &events[index], surface, max_steps),
                    ));
                }
                local
            }));
        }
        for join in joins {
            let local = join
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            for (index, result) in local {
                slots[index] = Some(result);
            }
        }
    });
    let mut out = Vec::with_capacity(n);
    for slot in slots {
        let result = slot
            .ok_or_else(|| SaddleError::Invalid("an ART event index was not evaluated".into()))?;
        out.push(result?);
    }
    Ok(out)
}

fn execute_event<S: PointSurface>(
    config: &ArtConfig,
    event: &ArtEvent,
    surface: &S,
    max_steps: usize,
) -> Result<ArtOutcome, SaddleError> {
    let mut session = ArtSession::new(
        config.clone(),
        event.minimum.clone(),
        event.displacement.clone(),
    )?;
    let report = session.run(surface, max_steps)?;
    Ok(ArtOutcome {
        status: report.status,
        position: session.position().to_owned(),
        energy: session.energy(),
        curvature: session.curvature(),
        max_force: report.max_force,
        pushes: report.pushes,
        climbs: report.climbs,
        evaluations: report.total_evaluations,
    })
}

fn travelled(x: &Array1<f64>, origin: &Array1<f64>) -> f64 {
    x.iter()
        .zip(origin.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

fn blend_opening(mode: &Array1<f64>, opening: &Array1<f64>) -> Array1<f64> {
    let sign = if opening.dot(mode) >= 0.0 { 1.0 } else { -1.0 };
    let mixed = &(mode * sign) + opening;
    let norm = mixed.dot(&mixed).sqrt();
    if norm > 1e-14 {
        mixed / norm
    } else {
        mode.clone()
    }
}

fn orthogonal_probe(mode: &Array1<f64>) -> Array1<f64> {
    let axis = mode
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .abs()
                .partial_cmp(&right.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let other = if axis + 1 < mode.len() { axis + 1 } else { 0 };
    let mut probe = Array1::zeros(mode.len());
    probe[other] = 1.0;
    let parallel = probe.dot(mode);
    probe = &probe - &(mode * parallel);
    let norm = probe.dot(&probe).sqrt();
    if norm > 1e-14 { probe / norm } else { probe }
}

fn validate_config(config: &ArtConfig) -> Result<(), SaddleError> {
    let mode = &config.mode;
    if !(config.eigen_threshold.is_finite()
        && mode.force_tol.is_finite()
        && mode.force_tol > 0.0
        && config.push_step.is_finite()
        && config.push_step > 0.0
        && mode.max_move.is_finite()
        && mode.max_move > 0.0
        && mode.dr.is_finite()
        && mode.dr > 0.0
        && mode.rotation_tol.is_finite()
        && mode.rotation_tol > 0.0
        && mode.krylov_dim >= 1
        && config.krylov_min >= 1
        && config.krylov_min <= mode.krylov_dim
        && config.activation_distance.is_finite()
        && config.activation_distance >= 0.0
        && config.push_over.is_finite()
        && config.push_over >= 0.0)
    {
        return Err(SaddleError::Invalid(
            "art tolerances and step lengths must be finite, with positive force, push, move, difference, and rotation bounds"
                .into(),
        ));
    }
    Ok(())
}

fn mode_config(config: &ArtConfig) -> MinModeConfig {
    config.mode.clone()
}

fn resolve_displacement(
    displacement: &ArtDisplacement,
    n: usize,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let (seed, support, amplitude, targeted) = match displacement {
        ArtDisplacement::Random {
            seed,
            support,
            amplitude,
        } => (*seed, support, *amplitude, None),
        ArtDisplacement::Targeted {
            direction,
            support,
            amplitude,
        } => (0, support, *amplitude, Some(direction)),
    };
    if !(amplitude.is_finite() && amplitude > 0.0) {
        return Err(SaddleError::Invalid(
            "initial displacement amplitude must be positive and finite".into(),
        ));
    }
    let active = active_indices(support, n)?;
    let direction = match targeted {
        Some(direction) => {
            if direction.len() != n || !direction.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Invalid(
                    "targeted direction must match the position and be finite".into(),
                ));
            }
            mask_and_normalize(direction, active.as_deref())?
        }
        None => random_unit(n, seed, active.as_deref())?,
    };
    Ok((direction, amplitude))
}

fn active_indices(support: &ArtSupport, n: usize) -> Result<Option<Vec<usize>>, SaddleError> {
    match support {
        ArtSupport::Global => Ok(None),
        ArtSupport::Local(indices) => {
            if indices.is_empty() || indices.iter().any(|i| *i >= n) {
                return Err(SaddleError::Invalid(
                    "local support indices must lie inside the position".into(),
                ));
            }
            Ok(Some(indices.clone()))
        }
    }
}

fn mask_and_normalize(
    direction: &Array1<f64>,
    active: Option<&[usize]>,
) -> Result<Array1<f64>, SaddleError> {
    let mut masked = Array1::zeros(direction.len());
    match active {
        None => masked.assign(direction),
        Some(indices) => {
            for &index in indices {
                masked[index] = direction[index];
            }
        }
    }
    let norm = masked.dot(&masked).sqrt();
    if !(norm.is_finite() && norm > 1e-14) {
        return Err(SaddleError::Invalid(
            "displacement vanishes on its support".into(),
        ));
    }
    Ok(masked / norm)
}

/// Component `index` of a random displacement. The only generator is
/// [`event_seed`]: the draw is that key mapped onto `[-1, 1]`.
fn event_component(seed: u64, index: u64) -> f64 {
    let bits = event_seed(seed, index);
    let unit = ((bits >> 11) as f64) / ((1u64 << 53) as f64);
    let value = unit * 2.0 - 1.0;
    if value.abs() < 1e-12 { 1e-12 } else { value }
}

fn random_unit(n: usize, seed: u64, active: Option<&[usize]>) -> Result<Array1<f64>, SaddleError> {
    let mut direction = Array1::zeros(n);
    match active {
        None => {
            for (index, value) in direction.iter_mut().enumerate() {
                *value = event_component(seed, index as u64);
            }
        }
        Some(indices) => {
            for &index in indices {
                direction[index] = event_component(seed, index as u64);
            }
        }
    }
    mask_and_normalize(&direction, active)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn checked_eval<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    calls: &AtomicUsize,
) -> Result<(f64, Array1<f64>), SaddleError> {
    calls.fetch_add(1, Ordering::Relaxed);
    let (energy, gradient) = surface.eval(x)?;
    if gradient.len() != x.len() {
        return Err(SaddleError::Shape("surface gradient length changed".into()));
    }
    if !energy.is_finite() || !gradient.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("art gradient"));
    }
    Ok((energy, gradient))
}

struct PerpBlock {
    steps: usize,
    max_move: f64,
    force_tol: f64,
    gate: ForceGate,
}

/// FIRE on `g - (g · τ) τ`, with `τ` held fixed for the block.
/// Returns the true energy and gradient at the final point.
fn relax_perpendicular<S: PointSurface>(
    surface: &S,
    x: &mut Array1<f64>,
    tau: ArrayView1<f64>,
    block: PerpBlock,
    calls: &AtomicUsize,
) -> Result<Centre, SaddleError> {
    let (energy, gradient) = checked_eval(surface, x.view(), calls)?;
    let mut centre = Centre {
        x: x.clone(),
        energy,
        gradient,
    };
    if block.steps == 0
        || block
            .gate
            .value(perpendicular(&centre.gradient, tau).view())
            <= block.force_tol
    {
        return Ok(centre);
    }

    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        x.len(),
    );
    solver.set_atom_maxmove(block.max_move);
    let error: Mutex<Option<SaddleError>> = Mutex::new(None);
    let known: Mutex<Centre> = Mutex::new(centre.clone());
    for _ in 0..block.steps {
        let perp_force = perpendicular(&centre.gradient, tau);
        if block.gate.value(perp_force.view()) <= block.force_tol {
            break;
        }
        let error_slot = &error;
        let known_slot = &known;
        let tau_owned = tau.to_owned();
        let oracle = Oracle::unbounded(x.len(), |point: ArrayView1<f64>| {
            let cached = known_slot
                .lock()
                .ok()
                .and_then(|slot| (slot.x == point).then(|| slot.clone()));
            let evaluated = match cached {
                Some(hit) => Ok((hit.energy, hit.gradient)),
                None => checked_eval(surface, point, calls),
            };
            match evaluated {
                Ok((energy, gradient)) => {
                    let perp = perpendicular(&gradient, tau_owned.view());
                    if let Ok(mut slot) = known_slot.lock() {
                        *slot = Centre {
                            x: point.to_owned(),
                            energy,
                            gradient: gradient.clone(),
                        };
                    }
                    (energy, perp)
                }
                Err(err) => {
                    if let Ok(mut slot) = error_slot.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
                    (f64::INFINITY, Array1::zeros(point.len()))
                }
            }
        });
        let stepped = solver.step(&oracle, x);
        if let Some(err) = error.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        let updated = known.lock().ok().map(|slot| slot.clone());
        let Some(updated) = updated else {
            return Err(SaddleError::Solver(
                "perpendicular relaxation lost its evaluation".into(),
            ));
        };
        if updated.x != *x {
            let (energy, gradient) = checked_eval(surface, x.view(), calls)?;
            centre = Centre {
                x: x.clone(),
                energy,
                gradient,
            };
            if let Ok(mut slot) = known.lock() {
                *slot = centre.clone();
            }
        } else {
            centre = updated;
        }
    }
    Ok(centre)
}

fn perpendicular(gradient: &Array1<f64>, tau: ArrayView1<f64>) -> Array1<f64> {
    let parallel = gradient.dot(&tau);
    gradient - &(tau.to_owned() * parallel)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    #[test]
    fn builder_rejects_a_non_positive_push_and_keeps_the_partn_defaults() {
        let config = super::ArtConfig::builder().build().unwrap();
        assert_abs_diff_eq!(config.eigen_threshold, -0.01, epsilon = 0.0);
        assert_abs_diff_eq!(config.push_step, 0.3, epsilon = 0.0);
        assert_abs_diff_eq!(config.activation_distance, 0.3, epsilon = 0.0);
        assert_eq!(config.perp_steps_push, 3);
        assert_eq!(config.mode.kind, super::MinModeKind::Lanczos);
        assert_eq!(config.mode.krylov_dim, 16);
        assert_abs_diff_eq!(config.mode.max_move, 0.3, epsilon = 0.0);
        assert_eq!(config.krylov_min, 3);
        assert!(super::ArtConfig::builder().push_step(0.0).build().is_err());
        assert!(super::ArtConfig::builder().dr(0.0).build().is_err());
    }

    #[test]
    fn event_seed_depends_only_on_the_counter() {
        let base = 0xA47E_5EED;
        assert_eq!(super::event_seed(base, 3), super::event_seed(base, 3));
        let mut seen = std::collections::BTreeSet::new();
        for index in 0..1_024 {
            assert!(seen.insert(super::event_seed(base, index)));
        }
    }

    #[test]
    fn random_displacement_uses_event_seed_components() {
        use ndarray::array;
        let session = super::ArtSession::new(
            super::ArtConfig::default(),
            array![1.0, 0.0, 0.0],
            super::ArtDisplacement::Random {
                seed: 17,
                support: super::ArtSupport::Global,
                amplitude: 0.2,
            },
        )
        .unwrap();
        let raw = array![
            super::event_component(17, 0),
            super::event_component(17, 1),
            super::event_component(17, 2),
        ];
        let norm = raw.dot(&raw).sqrt();
        let direction = session.push_direction();
        for i in 0..3 {
            assert_abs_diff_eq!(direction[i], raw[i] / norm, epsilon = 1e-12);
        }
    }
}
