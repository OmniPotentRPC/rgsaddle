//! Activation-relaxation technique nouveau.
//!
//! From a local minimum the session applies one random or targeted
//! displacement, on every coordinate or on a local support, and pushes
//! along that direction until the lowest Lanczos curvature falls below
//! [`ArtConfig::eigen_threshold`]. It then climbs that mode and relaxes
//! the force in the perpendicular hyperplane with FIRE.
//!
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
//! After the curvature crosses the threshold the session keeps climbing
//! the lowest mode. While the well is still closed, Lanczos is seeded
//! with the opening direction so a higher eigenvector left over from
//! the previous estimate cannot hide the crossing.
//! Convergence is a force at or under [`ArtConfig::force_tol`] together
//! with a negative lowest curvature, both at the current point.
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
use crate::minmode::{FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, PointSurface};

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
#[derive(Clone, Debug)]
pub struct ArtConfig {
    /// Leave the basin when the lowest curvature falls strictly below this.
    pub eigen_threshold: f64,
    /// Stationary-point tolerance on the selected force gate.
    pub force_tol: f64,
    /// Length of one basin push along the initial direction.
    pub push_step: f64,
    /// Largest atom displacement accepted from a push, a climb, or one FIRE step.
    pub max_move: f64,
    /// Basin pushes after the initial displacement. The search stops when this is spent while the curvature is still above the threshold.
    pub max_pushes: usize,
    /// Uphill macro-steps once the threshold has been crossed.
    pub max_climbs: usize,
    /// FIRE steps in the hyperplane after a basin push.
    pub perp_steps_push: usize,
    /// FIRE steps in the hyperplane after a climb.
    pub perp_steps_climb: usize,
    /// Finite-difference displacement passed to the Lanczos session.
    pub dr: f64,
    /// Forward or central Hessian actions.
    pub difference: FiniteDifference,
    /// Lanczos Ritz residual bound.
    pub rotation_tol: f64,
    /// Lanczos Krylov dimension.
    pub krylov_dim: usize,
}

impl Default for ArtConfig {
    fn default() -> Self {
        Self {
            eigen_threshold: 0.0,
            force_tol: 1e-3,
            push_step: 0.1,
            max_move: 0.1,
            max_pushes: 40,
            max_climbs: 80,
            perp_steps_push: 3,
            perp_steps_climb: 8,
            dr: 1e-3,
            difference: FiniteDifference::Forward,
            rotation_tol: 1e-4,
            krylov_dim: 12,
        }
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
    pushes: usize,
    climbs: usize,
    /// Set once the lowest curvature has fallen below the threshold.
    /// Later macro-steps stay on the lowest mode.
    opened: bool,
    evaluations: usize,
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
        let modes = MinModeSession::new(lanczos_config(&config), x.clone(), direction.clone())?;
        Ok(Self {
            config,
            force_gate: ForceGate::LinfNorm,
            x,
            push_direction: direction.clone(),
            mode: direction,
            curvature: f64::NAN,
            energy: f64::NAN,
            modes,
            pushes: 0,
            climbs: 0,
            opened: false,
            evaluations: 0,
            iteration: 0,
            centre: None,
        })
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
        let (energy, gradient) = self.centre_at(surface, x.view(), calls)?;
        self.modes.set_position(x.clone(), Some(gradient.clone()))?;
        // While the well is still closed, seed Lanczos with the opening
        // direction. The previous lowest mode can be an exact higher
        // eigenvector (the soft axes of the double well). One Krylov
        // step then reports that eigenvalue and never sees the reaction
        // coordinate cross the threshold.
        let lanczos_seed = if self.opened {
            self.mode.clone()
        } else {
            self.push_direction.clone()
        };
        self.modes.set_mode(lanczos_seed)?;
        let estimate = self.modes.estimate_mode(surface)?;
        calls.fetch_add(estimate.evaluations, Ordering::Relaxed);
        let max_force = self.force_gate.value(gradient.view());
        self.mode = estimate.mode;
        self.curvature = estimate.curvature;
        self.energy = energy;
        self.centre = Some(Centre {
            x: x.clone(),
            energy,
            gradient: gradient.clone(),
        });

        if max_force <= self.config.force_tol && estimate.curvature < 0.0 {
            return Ok(self.commit(x, calls, ArtStatus::Converged, max_force));
        }

        if estimate.curvature < self.config.eigen_threshold {
            self.opened = true;
        }
        let climbing = self.opened;
        if climbing && self.climbs >= self.config.max_climbs {
            return Ok(self.commit(x, calls, ArtStatus::Exhausted, max_force));
        }
        if !climbing && self.pushes >= self.config.max_pushes {
            return Ok(self.commit(x, calls, ArtStatus::Exhausted, max_force));
        }
        let tau = if climbing {
            self.mode.clone()
        } else {
            self.push_direction.clone()
        };

        if climbing {
            let g_par = gradient.dot(&tau);
            let delta = parallel_displacement(
                g_par,
                estimate.curvature,
                self.config.max_move,
                self.config.push_step,
            );
            x = &x + &(&tau * delta);
            self.climbs += 1;
        } else {
            let step = self.config.push_step.min(self.config.max_move);
            x = &x + &(&tau * step);
            self.pushes += 1;
        }
        if !x.iter().all(|v| v.is_finite()) {
            if climbing {
                self.climbs -= 1;
            } else {
                self.pushes -= 1;
            }
            return Err(SaddleError::NonFinite("art step"));
        }

        // Perpendicular FIRE waits until a negative-curvature Newton step
        // fits in the cap. Earlier, the parallel climb is still long and a
        // hyperplane step leaves the ridge. Once both components are short,
        // one FIRE step uses the Newton length `g_perp / |curvature|`.
        let newton_len = if estimate.curvature < 0.0 {
            (gradient.dot(&tau) / estimate.curvature).abs()
        } else {
            f64::INFINITY
        };
        let perp_gradient = perpendicular(&gradient, tau.view());
        let perp_norm = perp_gradient.dot(&perp_gradient).sqrt();
        let perp_newton = if estimate.curvature < 0.0 {
            perp_norm / estimate.curvature.abs().max(1e-8)
        } else {
            self.config.max_move
        };
        let (perp_steps, perp_cap) = if !climbing {
            (self.config.perp_steps_push, self.config.max_move)
        } else if newton_len > self.config.max_move {
            (0, self.config.max_move)
        } else if perp_newton <= self.config.max_move {
            (1, perp_newton.max(1e-8))
        } else {
            (self.config.perp_steps_climb, self.config.max_move)
        };
        let centre = match relax_perpendicular(
            surface,
            &mut x,
            tau.view(),
            PerpBlock {
                steps: perp_steps,
                max_move: perp_cap,
                force_tol: self.config.force_tol,
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
            iteration: self.iteration,
        }
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

/// Parallel coefficient along a unit mode.
///
/// Negative curvature uses the capped Newton step `-g_par / curvature`.
/// Non-negative curvature walks uphill by at most `push_step`.
fn parallel_displacement(g_par: f64, curvature: f64, cap: f64, push_step: f64) -> f64 {
    if curvature < 0.0 && curvature.is_finite() && g_par.is_finite() {
        let newton = -g_par / curvature;
        if newton.is_finite() {
            return newton.clamp(-cap, cap);
        }
    }
    let length = cap.min(push_step);
    if g_par >= 0.0 { length } else { -length }
}

fn validate_config(config: &ArtConfig) -> Result<(), SaddleError> {
    if !(config.eigen_threshold.is_finite()
        && config.force_tol.is_finite()
        && config.force_tol > 0.0
        && config.push_step.is_finite()
        && config.push_step > 0.0
        && config.max_move.is_finite()
        && config.max_move > 0.0
        && config.dr.is_finite()
        && config.dr > 0.0
        && config.rotation_tol.is_finite()
        && config.rotation_tol > 0.0
        && config.krylov_dim >= 1)
    {
        return Err(SaddleError::Invalid(
            "art tolerances and step lengths must be finite, with positive force, push, move, difference, and rotation bounds"
                .into(),
        ));
    }
    Ok(())
}

fn lanczos_config(config: &ArtConfig) -> MinModeConfig {
    MinModeConfig {
        kind: MinModeKind::Lanczos,
        dr: config.dr,
        difference: config.difference,
        rotation_tol: config.rotation_tol,
        krylov_dim: config.krylov_dim,
        force_tol: config.force_tol,
        max_move: config.max_move,
        method: Method::Fire { kind: FireKind::V2 },
        ..MinModeConfig::default()
    }
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

fn random_unit(n: usize, seed: u64, active: Option<&[usize]>) -> Result<Array1<f64>, SaddleError> {
    let mut state = seed;
    let mut draw = || {
        state = splitmix64(state);
        let uniform = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        uniform.clamp(1e-12, 1.0 - 1e-12)
    };
    let mut normal = || {
        let u1 = draw();
        let u2 = draw();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut direction = Array1::zeros(n);
    match active {
        None => {
            for value in direction.iter_mut() {
                *value = normal();
            }
        }
        Some(indices) => {
            for &index in indices {
                direction[index] = normal();
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
    use super::parallel_displacement;
    use approx::assert_abs_diff_eq;

    #[test]
    fn negative_curvature_takes_the_capped_newton_step() {
        let step = parallel_displacement(1.5, -1.0, 0.1, 0.2);
        assert_abs_diff_eq!(step, 0.1, epsilon = 1e-12);
        let short = parallel_displacement(0.02, -2.0, 0.1, 0.2);
        assert_abs_diff_eq!(short, 0.01, epsilon = 1e-12);
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
    fn non_negative_curvature_walks_uphill() {
        assert_abs_diff_eq!(
            parallel_displacement(0.4, 2.0, 0.3, 0.1),
            0.1,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            parallel_displacement(-0.4, 0.0, 0.3, 0.1),
            -0.1,
            epsilon = 1e-12
        );
    }
}
