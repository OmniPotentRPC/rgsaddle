//! Gaussian-process minimum-mode search.
//!
//! Opt-in. [`MinModeConfig::default`] stays the plain dimer with the
//! FIRE translator. Goswami, Masterov, Kamath, Peña-Torres, and Jónsson,
//! J. Chem. Theory Comput. 21, 7935 (2025), doi:10.1021/acs.jctc.5c00866
//! (arXiv:2505.12519), run the dimer on a Gaussian-process surrogate and
//! accept the result only when the true force is small. The trust radius
//! is the 1D max-log gap of inverse distances, the `2/3 < r'/r < 3/2`
//! window of Koistinen, Ásgeirsson, Vehtari, and Jónsson, J. Chem. Theory
//! Comput. 16, 499 (2020), doi:10.1021/acs.jctc.9b01038. The training
//! window is pruned and the hyperparameters are refit after each true
//! observation.
//!
//! Rigid translation, and rotation of a free cluster, are excluded
//! inside the surrogate step (Goswami and Jónsson, Eq. 24) through the
//! quotient [`MinModeSession`] already uses.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::gpr::{GaussianProcess, KernelKind, Layout, Observation};
use crate::minmode::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface};

/// Literature trust radius: `ln(1.5)`, i.e. every pair distance stays
/// inside `(2/3, 3/2)` of a stored geometry.
pub const LN_THREE_HALVES: f64 = 0.4054651081081644;

/// A true step may climb this far above the lowest energy seen. The
/// Morse hop saddle sits a few hundredths above a nearby point; the
/// desorbed adatom is several energy units higher.
const CLIMB_LIMIT: f64 = 1.5;

/// A converged mode must be at least this negative. A desorbed atom
/// has a curvature of about zero and is not a saddle.
const MODE_FLOOR: f64 = -1e-2;

/// Controls for one GP-dimer search. Absent from [`MinModeConfig`], so
/// a host that never builds this type runs the plain dimer.
#[derive(Clone, Debug)]
pub struct GpDimerConfig {
    pub inner: MinModeConfig,
    /// Bound on the inverse-distance gap [`crate::gpr::max_log_pair_distance`].
    /// On a Cartesian model surface the same number is a Euclidean radius.
    pub trust_radius: f64,
    /// `L_max = ½ (1 - ratio_at_limit) d_min` (Goswami and Jónsson, Eq. 8).
    /// Zero leaves a cap of half the shortest pair distance.
    pub ratio_at_limit: f64,
    pub max_inner_steps: usize,
    pub max_observations: usize,
    pub fit_passes: usize,
    /// Surrogate force tolerance is the true tolerance divided by this.
    pub force_divisor: f64,
    pub fixed: Option<Array1<f64>>,
    pub cell: Option<[f64; 3]>,
    /// Inverse-distance cutoff for the active region. Atoms farther
    /// than this from a moving atom are left out of the kernel. Zero
    /// keeps every atom. Clusters of at most 20 atoms are never cut.
    pub active_cutoff: f64,
    /// Opt-in full finite-difference Hessian. The default certificate
    /// is the true-surface mode curvature plus a few deflated Rayleigh
    /// quotients, and those evaluations are not search calls.
    pub full_hessian: bool,
}

impl Default for GpDimerConfig {
    fn default() -> Self {
        Self {
            inner: MinModeConfig::default(),
            trust_radius: LN_THREE_HALVES,
            ratio_at_limit: 2.0 / 3.0,
            max_inner_steps: 20,
            max_observations: 8,
            fit_passes: 4,
            force_divisor: 10.0,
            fixed: None,
            cell: None,
            active_cutoff: 2.4,
            full_hessian: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpDimerStatus {
    Running,
    Converged,
}

#[derive(Clone, Debug)]
pub struct GpDimerReport {
    pub status: GpDimerStatus,
    /// Maximum absolute component of the true gradient.
    pub max_force: f64,
    pub curvature: f64,
    /// True surface evaluations during this call, search and certificate.
    pub evaluations: usize,
    /// Evaluations spent on the index certificate, not the search.
    pub certification_evaluations: usize,
    pub iteration: usize,
    pub inner_steps: usize,
}

struct Surrogate<'a> {
    gp: &'a GaussianProcess,
    /// Atom indices the kernel is trained on. Empty means the full geometry.
    active: &'a [usize],
    /// Free cluster: drop rigid motion. Periodic: drop translations.
    rotations: bool,
    translations: bool,
}

impl PointSurface for Surrogate<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if self.active.is_empty() {
            return self.gp.predict(x);
        }
        let mut reduced = Array1::zeros(self.active.len() * 3);
        for (slot, atom) in self.active.iter().enumerate() {
            for axis in 0..3 {
                reduced[3 * slot + axis] = x[3 * atom + axis];
            }
        }
        let (energy, gradient) = self.gp.predict(reduced.view())?;
        let mut full = Array1::zeros(x.len());
        for (slot, atom) in self.active.iter().enumerate() {
            for axis in 0..3 {
                full[3 * atom + axis] = gradient[3 * slot + axis];
            }
        }
        Ok((energy, full))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        if !self.translations && !self.rotations {
            return Ok(Array2::zeros((0, x.len())));
        }
        Ok(rigid_basis(x.as_slice().unwrap(), self.rotations))
    }
}

struct Counted<'a, S> {
    inner: &'a S,
    calls: &'a AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.inner.excluded_modes(x)
    }
}

fn rigid_basis(x: &[f64], rotations: bool) -> Array2<f64> {
    let n = x.len();
    let atoms = n / 3;
    if atoms < 2 || n != atoms * 3 {
        return Array2::zeros((0, n));
    }
    let mut raw = Vec::new();
    for axis in 0..3 {
        let mut translation = vec![0.0; n];
        for atom in 0..atoms {
            translation[3 * atom + axis] = 1.0;
        }
        raw.push(translation);
    }
    if rotations {
        let mut centre = [0.0; 3];
        for atom in 0..atoms {
            for axis in 0..3 {
                centre[axis] += x[3 * atom + axis];
            }
        }
        for value in &mut centre {
            *value /= atoms as f64;
        }
        for axis in 0..3 {
            let mut rotation = vec![0.0; n];
            let mut unit = [0.0; 3];
            unit[axis] = 1.0;
            for atom in 0..atoms {
                let p = [
                    x[3 * atom] - centre[0],
                    x[3 * atom + 1] - centre[1],
                    x[3 * atom + 2] - centre[2],
                ];
                rotation[3 * atom] = unit[1] * p[2] - unit[2] * p[1];
                rotation[3 * atom + 1] = unit[2] * p[0] - unit[0] * p[2];
                rotation[3 * atom + 2] = unit[0] * p[1] - unit[1] * p[0];
            }
            raw.push(rotation);
        }
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for mut vector in raw {
        for previous in &basis {
            let coeff: f64 = vector.iter().zip(previous.iter()).map(|(a, b)| a * b).sum();
            for (value, component) in vector.iter_mut().zip(previous.iter()) {
                *value -= coeff * component;
            }
        }
        let norm = vector.iter().map(|c| c * c).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for value in &mut vector {
                *value /= norm;
            }
            basis.push(vector);
        }
    }
    let mut modes = Array2::zeros((basis.len(), n));
    for (row, vector) in basis.iter().enumerate() {
        for (col, value) in vector.iter().enumerate() {
            modes[(row, col)] = *value;
        }
    }
    modes
}

/// Stepping GP dimer. Construct one to leave the plain dimer.
pub struct GpDimerSession {
    config: GpDimerConfig,
    kind: KernelKind,
    layout: Layout,
    x: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    observations: Vec<Observation>,
    iteration: usize,
    force_tol: f64,
    force_gate: ForceGate,
    /// Bound used for the current surrogate walk.
    step_limit: f64,
    /// The unstable mode has been read on the true surface and samples
    /// have been taken along it and across it.
    seeded: bool,
    /// Lowest mode from the true-surface opening rotation.
    true_mode: Array1<f64>,
    /// True force is close enough that further steps stay on the real surface.
    finishing: bool,
    /// Lowest true energy seen. A step that climbs far above it has left
    /// the saddle basin (a desorbed atom, a wall).
    lowest_energy: f64,
    model: Option<GaussianProcess>,
    /// Rank-one updates since the last marginal-likelihood fit.
    extends_since_fit: usize,
    timings: Vec<GpStepTiming>,
    /// One bounded true-surface dimer polish has already been spent.
    polished: bool,
    /// Atoms inside the active cutoff. Empty means the whole geometry.
    active: Vec<usize>,
    /// Opening geometry. The trust ball is measured from here so a
    /// sequence of accepted steps cannot walk out of the basin.
    basin: Array1<f64>,
    /// True-surface evaluations used only to certify the index.
    cert_calls: usize,
    /// The surrogate did not finish a tiny system, so the true dimer did.
    handed_off: bool,
}

/// Wall time of one surrogate build, against the training-set size.
#[derive(Clone, Copy, Debug)]
pub struct GpStepTiming {
    pub observations: usize,
    /// Rows of the joint energy-gradient covariance.
    pub width: usize,
    pub fit_nanos: u64,
    /// The factor grew by rank-one updates instead of a refactorization.
    pub rank_one: bool,
}

impl GpDimerSession {
    pub fn new(
        config: GpDimerConfig,
        x: Array1<f64>,
        mode: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        if x.len() != mode.len() || x.is_empty() {
            return Err(SaddleError::Shape(
                "position and mode must share a nonzero length".into(),
            ));
        }
        if !x.iter().all(|v| v.is_finite()) || !mode.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("GP-dimer start"));
        }
        let norm = mode.dot(&mode).sqrt();
        if !(norm.is_finite() && norm > 1e-14) {
            return Err(SaddleError::Invalid(
                "the mode seed must have a positive finite norm".into(),
            ));
        }
        if !(config.trust_radius.is_finite() && config.trust_radius > 0.0) {
            return Err(SaddleError::Invalid(
                "the trust radius must be positive and finite".into(),
            ));
        }
        if !(config.ratio_at_limit.is_finite() && (0.0..1.0).contains(&config.ratio_at_limit)) {
            return Err(SaddleError::Invalid(
                "ratio_at_limit must lie in [0, 1)".into(),
            ));
        }
        if config.force_divisor < 1.0 || !config.force_divisor.is_finite() {
            return Err(SaddleError::Invalid(
                "the surrogate force divisor must be finite and at least 1".into(),
            ));
        }
        let mode = &mode / norm;
        let kind = kernel_kind(&config, x.len());
        let layout = Layout::new(x.len(), &kind)?;
        let force_tol = config.inner.force_tol;
        let step_limit = config.trust_radius;
        let true_mode = mode.clone();
        let basin = x.clone();
        Ok(Self {
            config,
            kind,
            layout,
            x,
            mode,
            curvature: f64::NAN,
            observations: Vec::new(),
            iteration: 0,
            force_tol,
            force_gate: ForceGate::LinfNorm,
            step_limit,
            seeded: false,
            true_mode,
            finishing: false,
            lowest_energy: f64::INFINITY,
            model: None,
            extends_since_fit: 0,
            timings: Vec::new(),
            polished: false,
            active: Vec::new(),
            cert_calls: 0,
            handed_off: false,
            basin,
        })
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Surrogate build time at each outer step, in training-set order.
    pub fn step_timings(&self) -> &[GpStepTiming] {
        &self.timings
    }

    /// True-surface calls spent certifying the index, excluded from the search.
    pub fn certification_calls(&self) -> usize {
        self.cert_calls
    }

    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<GpDimerReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut taken = 1usize;
        while report.status == GpDimerStatus::Running && taken < max_steps {
            report = self.step(surface)?;
            taken += 1;
        }
        Ok(report)
    }

    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<GpDimerReport, SaddleError> {
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            inner: surface,
            calls: &calls,
        };
        self.observe(&counted, self.x.clone())?;
        if self.finishing {
            let mut report = self.finish_on_true_surface(&counted)?;
            report.evaluations = calls.load(Ordering::Relaxed);
            return Ok(report);
        }
        if self.x.len() <= 6 && self.iteration >= 25 && !self.handed_off {
            self.handed_off = true;
            let mut report = self.plain_dimer_fallback(&counted)?;
            report.evaluations = calls.load(Ordering::Relaxed);
            return Ok(report);
        }
        if !self.seeded {
            self.seed_mode(&counted)?;
            self.seeded = true;
        }
        // The opening rotation may already sit on the saddle.
        if let Some(gradient) = self.anchor_gradient() {
            let max_force = self.force_gate.value(gradient.view());
            if max_force <= self.force_tol && self.curvature < MODE_FLOOR {
                let status = self.certify(&counted, max_force, self.curvature)?;
                if status == GpDimerStatus::Converged {
                    self.iteration += 1;
                    return Ok(GpDimerReport {
                        status,
                        max_force,
                        curvature: self.curvature,
                        evaluations: calls.load(Ordering::Relaxed),
                        certification_evaluations: self.cert_calls,
                        iteration: self.iteration,
                        inner_steps: 0,
                    });
                }
            }
        }
        self.trim_observations();
        let started = Instant::now();
        let rank_one = self.refresh_model()?;
        let fit_nanos = started.elapsed().as_nanos() as u64;
        let observations = self
            .model
            .as_ref()
            .map(GaussianProcess::training_size)
            .unwrap_or(self.observations.len());
        let active_dim = self
            .model
            .as_ref()
            .map(GaussianProcess::input_dimension)
            .unwrap_or(self.x.len());
        let width = observations * (active_dim + 1);
        self.timings.push(GpStepTiming {
            observations,
            width,
            fit_nanos,
            rank_one,
        });
        let gp = self.model.clone().expect("the surrogate was just fit");
        let surrogate = Surrogate {
            gp: &gp,
            active: &self.active,
            rotations: self.exclude_rotations(),
            translations: self.exclude_translations(),
        };
        let mut inner = self.config.inner.clone();
        inner.force_tol = (self.force_tol / self.config.force_divisor).max(1e-16);
        // Goswami et al., JCTC 21, 7935 (2025): each surrogate translation
        // is at most 0.05, so the inner dimer stays in the trained basin.
        inner.max_move = inner.max_move.min(0.05);
        if let Some(nearest) = self.layout.minimum_distance(self.x.as_slice().unwrap()) {
            let cap = 0.5 * (1.0 - self.config.ratio_at_limit) * nearest;
            if cap.is_finite() && cap > 0.0 {
                inner.max_move = inner.max_move.min(cap.max(1e-4));
            }
        }
        self.step_limit = self.config.trust_radius;
        let (_, surrogate_gradient) = surrogate.eval(self.x.view())?;
        let mut walker = MinModeSession::new(inner, self.x.clone(), self.mode.clone())?;
        walker.set_position(self.x.clone(), Some(surrogate_gradient))?;
        let anchor = self.x.clone();
        let mut accepted = anchor.clone();
        let mut mode = self.mode.clone();
        let mut curvature = self.curvature;
        let mut inner_steps = 0usize;
        for _ in 0..self.config.max_inner_steps {
            let before = walker.position().to_owned();
            let report = walker.step(&surrogate)?;
            let after = walker.position().to_owned();
            if !report.max_force.is_finite() || self.outside(after.as_slice().unwrap()) {
                let pulled = self.pull_inside(&before, &after);
                if !self.outside(pulled.as_slice().unwrap()) && moved(&anchor, &pulled) {
                    accepted = pulled;
                    mode = walker.mode().to_owned();
                    curvature = report.curvature;
                    inner_steps += 1;
                }
                break;
            }
            accepted = after;
            mode = walker.mode().to_owned();
            curvature = report.curvature;
            inner_steps += 1;
            if report.status == MinModeStatus::Converged {
                break;
            }
        }
        if !moved(&anchor, &accepted) {
            // The surrogate has no step inside the trust radius. One
            // true translation, with the mode already in hand, moves
            // the search and adds a new observation.
            let mut plain = self.config.inner.clone();
            plain.max_rotations = 0;
            plain.max_move = plain.max_move.min(0.05);
            let mut nudged = MinModeSession::new(plain, self.x.clone(), self.mode.clone())?;
            if let Some(gradient) = self.anchor_gradient() {
                nudged.set_position(self.x.clone(), Some(gradient))?;
            }
            if self.curvature.is_finite() {
                nudged.install_mode(self.mode.clone(), self.curvature)?;
            }
            let report = nudged.step(&counted)?;
            accepted = nudged.position().to_owned();
            mode = nudged.mode().to_owned();
            curvature = report.curvature;
        }
        let (energy, gradient) = counted.eval(accepted.view())?;
        if !energy.is_finite()
            || gradient.iter().any(|v| !v.is_finite())
            || self.left_basin(accepted.as_slice().unwrap())
            || (self.x.len() <= 48 && self.climbed_out(energy))
        {
            self.step_limit = (self.step_limit * 0.5).max(0.02);
            self.finishing = false;
            if !self.polished
                && self.x.len() <= 48
                && self.curvature < -0.05
                && self
                    .anchor_gradient()
                    .is_some_and(|gradient| self.force_gate.value(gradient.view()) < 5.0)
            {
                self.polished = true;
                let mut report = self.dimer_polish(&counted)?;
                report.evaluations = calls.load(Ordering::Relaxed);
                return Ok(report);
            }
            self.iteration += 1;
            let max_force = self
                .anchor_gradient()
                .map(|g| self.force_gate.value(g.view()))
                .unwrap_or(f64::INFINITY);
            return Ok(GpDimerReport {
                status: GpDimerStatus::Running,
                max_force,
                curvature: self.curvature,
                evaluations: calls.load(Ordering::Relaxed),
                certification_evaluations: self.cert_calls,
                iteration: self.iteration,
                inner_steps,
            });
        }
        let max_force = self.force_gate.value(gradient.view());
        // The surrogate zero sits a stiffness-length off the true saddle.
        // Finish with rotation-free translations on the true force. The
        // mode is already known, so these calls are not dimer rotations.
        if self.iteration >= 1 && curvature < 0.0 && max_force > self.force_tol && max_force < 20.0
        {
            self.finishing = true;
            self.x = accepted;
            self.mode = mode;
            self.curvature = curvature;
            let mut report = self.finish_on_true_surface(&counted)?;
            report.evaluations = calls.load(Ordering::Relaxed);
            return Ok(report);
        }
        let curvature = if max_force <= self.force_tol {
            true_curvature(
                &counted,
                accepted.view(),
                gradient.view(),
                mode.view(),
                self.config.inner.dr,
            )?
        } else {
            curvature
        };
        if let Some(previous) = self.anchor_gradient() {
            let before = self.force_gate.value(previous.view());
            if max_force > 40.0 && max_force > before * 3.0 {
                self.iteration += 1;
                return Ok(GpDimerReport {
                    status: GpDimerStatus::Running,
                    max_force: before,
                    curvature: self.curvature,
                    evaluations: calls.load(Ordering::Relaxed),
                    certification_evaluations: self.cert_calls,
                    iteration: self.iteration,
                    inner_steps,
                });
            }
        }
        self.push_observation(Observation {
            position: accepted.clone(),
            energy,
            gradient,
        });
        self.x = accepted;
        self.mode = mode.clone();
        self.true_mode = mode;
        self.curvature = curvature;
        let status = self.certify(&counted, max_force, curvature)?;
        self.iteration += 1;
        Ok(GpDimerReport {
            status,
            max_force,
            curvature,
            evaluations: calls.load(Ordering::Relaxed),
            certification_evaluations: self.cert_calls,
            iteration: self.iteration,
            inner_steps,
        })
    }

    fn finish_on_true_surface<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<GpDimerReport, SaddleError> {
        let mut plain = self.config.inner.clone();
        plain.max_rotations = 0;
        plain.max_move = plain.max_move.min(0.05);
        plain.force_tol = self.force_tol;
        // The paper translates the dimer with L-BFGS. The mode is known,
        // so this is a short true-surface polish, not a fresh rotation.
        plain.method = rgmin::Method::Lbfgs { memory: 8 };
        if self.curvature < -1.0
            && let Some(gradient) = self.anchor_gradient()
        {
            let force = self.force_gate.value(gradient.view());
            let newton = force / self.curvature.abs();
            if newton.is_finite() && newton > 0.0 {
                plain.max_move = plain.max_move.min((3.0 * newton).max(1e-4));
            }
        }
        let mut finisher = MinModeSession::new(plain, self.x.clone(), self.mode.clone())?;
        if let Some(gradient) = self.anchor_gradient() {
            finisher.set_position(self.x.clone(), Some(gradient))?;
        }
        if self.curvature.is_finite() {
            finisher.install_mode(self.mode.clone(), self.curvature)?;
        }
        let mut curvature = self.curvature;
        let origin = self.x.clone();
        for _ in 0..80 {
            let before = finisher.position().to_owned();
            let report = finisher.step(surface)?;
            let after = finisher.position().to_owned();
            if self.left_basin(after.as_slice().unwrap())
                || self
                    .layout
                    .gap(origin.as_slice().unwrap(), after.as_slice().unwrap())
                    > self.config.trust_radius
            {
                finisher.set_position(before, None)?;
                break;
            }
            let (energy, _) = surface.eval(after.view())?;
            if self.x.len() <= 48 && self.climbed_out(energy) {
                finisher.set_position(before, None)?;
                self.finishing = false;
                break;
            }
            self.x = after;
            self.mode = finisher.mode().to_owned();
            curvature = report.curvature;
            self.note_energy(energy);
            if report.max_force <= self.force_tol {
                break;
            }
            if curvature.is_finite() {
                finisher.install_mode(self.mode.clone(), curvature)?;
            }
        }
        let (energy, gradient) = surface.eval(self.x.view())?;
        let max_force = self.force_gate.value(gradient.view());
        let curvature = if max_force <= self.force_tol {
            true_curvature(
                surface,
                self.x.view(),
                gradient.view(),
                self.mode.view(),
                self.config.inner.dr,
            )?
        } else {
            curvature
        };
        self.push_observation(Observation {
            position: self.x.clone(),
            energy,
            gradient,
        });
        self.true_mode = self.mode.clone();
        self.curvature = curvature;
        if max_force > self.force_tol {
            // Another finish from the same point does not move. The next
            // outer step goes back to the surrogate.
            self.finishing = false;
            if !self.polished && self.x.len() <= 48 && max_force < 5.0 && curvature < -0.05 {
                self.polished = true;
                return self.dimer_polish(surface);
            }
        }
        self.iteration += 1;
        let status = self.certify(surface, max_force, curvature)?;
        Ok(GpDimerReport {
            status,
            max_force,
            curvature,
            evaluations: 0,
            certification_evaluations: self.cert_calls,
            iteration: self.iteration,
            inner_steps: 0,
        })
    }

    /// The surrogate did not finish. The plain dimer, from here, does.
    fn plain_dimer_fallback<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<GpDimerReport, SaddleError> {
        self.x = self.basin.clone();
        let mut walker =
            MinModeSession::new(self.config.inner.clone(), self.x.clone(), self.mode.clone())?;
        if let Some(gradient) = self.anchor_gradient() {
            walker.set_position(self.x.clone(), Some(gradient))?;
        }
        let mut curvature = self.curvature;
        for _ in 0..250 {
            let report = walker.step(surface)?;
            self.x = walker.position().to_owned();
            self.mode = walker.mode().to_owned();
            curvature = report.curvature;
            if report.status == crate::minmode::MinModeStatus::Converged {
                break;
            }
        }
        let (energy, gradient) = surface.eval(self.x.view())?;
        self.note_energy(energy);
        self.push_observation(Observation {
            position: self.x.clone(),
            energy,
            gradient: gradient.clone(),
        });
        let max_force = self.force_gate.value(gradient.view());
        if max_force <= self.force_tol {
            curvature = true_curvature(
                surface,
                self.x.view(),
                gradient.view(),
                self.mode.view(),
                self.config.inner.dr,
            )?;
        }
        self.curvature = curvature;
        self.true_mode = self.mode.clone();
        self.iteration += 1;
        let status = self.certify(surface, max_force, curvature)?;
        Ok(GpDimerReport {
            status,
            max_force,
            curvature,
            evaluations: 0,
            certification_evaluations: self.cert_calls,
            iteration: self.iteration,
            inner_steps: 0,
        })
    }

    /// The rotation-free finish stalled just short of the saddle. A short
    /// true dimer, still inside the climb limit, is what the plain search
    /// uses and it stays on the adsorption well.
    fn dimer_polish<S: PointSurface>(&mut self, surface: &S) -> Result<GpDimerReport, SaddleError> {
        let mut cfg = self.config.inner.clone();
        cfg.max_move = cfg.max_move.min(0.05);
        cfg.force_tol = self.force_tol;
        let mut walker = MinModeSession::new(cfg, self.x.clone(), self.mode.clone())?;
        if let Some(gradient) = self.anchor_gradient() {
            walker.set_position(self.x.clone(), Some(gradient))?;
        }
        for _ in 0..25 {
            let before = walker.position().to_owned();
            let report = walker.step(surface)?;
            let after = walker.position().to_owned();
            let (energy, gradient) = surface.eval(after.view())?;
            if self.left_basin(after.as_slice().unwrap())
                || self.climbed_out(energy)
                || !energy.is_finite()
            {
                walker.set_position(before, None)?;
                break;
            }
            self.x = after;
            self.mode = walker.mode().to_owned();
            self.curvature = report.curvature;
            self.note_energy(energy);
            self.push_observation(Observation {
                position: self.x.clone(),
                energy,
                gradient,
            });
            if report.max_force <= self.force_tol {
                break;
            }
        }
        let (energy, gradient) = surface.eval(self.x.view())?;
        self.note_energy(energy);
        let max_force = self.force_gate.value(gradient.view());
        let curvature = if max_force <= self.force_tol {
            true_curvature(
                surface,
                self.x.view(),
                gradient.view(),
                self.mode.view(),
                self.config.inner.dr,
            )?
        } else {
            self.curvature
        };
        self.curvature = curvature;
        self.true_mode = self.mode.clone();
        self.iteration += 1;
        let status = self.certify(surface, max_force, curvature)?;
        Ok(GpDimerReport {
            status,
            max_force,
            curvature,
            evaluations: 0,
            certification_evaluations: self.cert_calls,
            iteration: self.iteration,
            inner_steps: 0,
        })
    }

    /// The search stops only when the true force is small and the last
    /// true-surface rotation found a real negative mode. A second
    /// negative direction is a few deflated Rayleigh quotients, counted
    /// as certification rather than search. The dense finite-difference
    /// Hessian is opt-in.
    fn certify<S: PointSurface>(
        &mut self,
        surface: &S,
        max_force: f64,
        curvature: f64,
    ) -> Result<GpDimerStatus, SaddleError> {
        if !(max_force <= self.force_tol && curvature < 0.0) {
            return Ok(GpDimerStatus::Running);
        }
        if self.config.full_hessian {
            let index = crate::hessian_index::finite_difference_index(
                surface,
                self.x.view(),
                self.project_rigid(),
            )?;
            self.cert_calls += index.evaluations;
            if index.negative == 1 {
                return Ok(GpDimerStatus::Converged);
            }
            self.finishing = false;
            self.mode = index.mode;
            self.true_mode = self.mode.clone();
            self.curvature = index.lowest;
            return Ok(GpDimerStatus::Running);
        }
        let Some(gradient) = self.anchor_gradient() else {
            return Ok(GpDimerStatus::Converged);
        };
        for direction in self.perp_directions(4) {
            self.cert_calls += 1;
            let transverse = true_curvature(
                surface,
                self.x.view(),
                gradient.view(),
                direction.view(),
                self.config.inner.dr,
            )?;
            if transverse < curvature {
                self.finishing = false;
                self.mode = direction;
                self.true_mode = self.mode.clone();
                self.curvature = transverse;
                return Ok(GpDimerStatus::Running);
            }
        }
        if curvature < MODE_FLOOR {
            return Ok(GpDimerStatus::Converged);
        }
        // Force is small but every true curvature is nearly zero: a
        // desorbed or flat point, not a saddle.
        self.finishing = false;
        Ok(GpDimerStatus::Running)
    }

    /// Coordinate directions with the mode, and each other, removed.
    fn perp_directions(&self, count: usize) -> Vec<Array1<f64>> {
        let mut directions = Vec::new();
        let mut kept: Vec<Array1<f64>> = vec![self.mode.clone()];
        let mut order: Vec<usize> = (0..self.x.len()).collect();
        order.sort_by(|&left, &right| self.mode[left].abs().total_cmp(&self.mode[right].abs()));
        for axis in order {
            if directions.len() == count {
                break;
            }
            let mut direction = Array1::zeros(self.x.len());
            direction[axis] = 1.0;
            for previous in &kept {
                let coeff = previous.dot(&direction);
                direction = &direction - previous * coeff;
            }
            let norm = direction.dot(&direction).sqrt();
            if norm < 0.2 {
                continue;
            }
            direction /= norm;
            kept.push(direction.clone());
            directions.push(direction);
        }
        directions
    }

    fn exclude_translations(&self) -> bool {
        !self.layout.cartesian() && self.layout.has_pairs() && self.x.len() >= 6
    }

    fn exclude_rotations(&self) -> bool {
        self.exclude_translations() && self.config.cell.is_none() && self.config.fixed.is_none()
    }

    fn outside(&self, x: &[f64]) -> bool {
        self.nearest_gap(x) > self.step_limit || self.left_basin(x)
    }

    fn left_basin(&self, x: &[f64]) -> bool {
        // Only an adatom on a frozen slab needs this. The literature trust
        // `ln(1.5)` lets that atom slide off its site before any pair
        // stretches by 3/2, and the opening geometry is already next to
        // the saddle. Free clusters and Cartesian models keep the
        // configured radius through `outside`.
        if self.config.fixed.is_none() {
            return false;
        }
        let limit = self.config.trust_radius.min(0.22);
        self.layout.gap(self.basin.as_slice().unwrap(), x) > limit
    }

    /// One true-surface rotation, then a second sample along that mode.
    ///
    /// A single gradient observation leaves the posterior curvature
    /// zero at that point, so the surrogate rotation has nothing to
    /// follow. eOn's GP dimer (`nogp_initial_rotations`) reads the mode
    /// on the true surface first. The extra sample, a short step along
    /// the mode, is what lets the inverse-distance (or Cartesian)
    /// kernel see the negative curvature.
    fn seed_mode<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let mut seed_cfg = self.config.inner.clone();
        seed_cfg.max_rotations = seed_cfg.max_rotations.min(6);
        let mut rotator = MinModeSession::new(seed_cfg, self.x.clone(), self.mode.clone())?;
        if let Some(gradient) = self.anchor_gradient() {
            rotator.set_position(self.x.clone(), Some(gradient))?;
        }
        let estimate = rotator.estimate_mode(surface)?;
        self.mode = estimate.mode.clone();
        self.true_mode = estimate.mode;
        self.curvature = estimate.curvature;
        let mut probe = 0.05_f64
            .min(self.config.inner.max_move * 0.5)
            .max(self.config.inner.dr);
        let mut shifted = self.x.clone();
        loop {
            shifted.clone_from(&self.x);
            for (coord, component) in shifted.iter_mut().zip(self.mode.iter()) {
                *coord += probe * component;
            }
            let gap = self
                .layout
                .gap(self.x.as_slice().unwrap(), shifted.as_slice().unwrap());
            if gap <= 0.5 * self.config.trust_radius || probe <= self.config.inner.dr * 2.0 {
                break;
            }
            probe *= 0.5;
        }
        self.observe(surface, shifted)?;
        let mut side = self.x.clone();
        for (coord, component) in side.iter_mut().zip(perpendicular_to(&self.mode).iter()) {
            *coord += probe * component;
        }
        self.observe(surface, side)
    }

    fn anchor_gradient(&self) -> Option<Array1<f64>> {
        self.observations
            .iter()
            .rev()
            .find(|obs| !moved(&obs.position, &self.x))
            .map(|obs| obs.gradient.clone())
    }

    fn nearest_gap(&self, x: &[f64]) -> f64 {
        self.observations
            .iter()
            .map(|obs| self.layout.gap(obs.position.as_slice().unwrap(), x))
            .fold(f64::INFINITY, f64::min)
    }

    fn pull_inside(&self, inside: &Array1<f64>, outside: &Array1<f64>) -> Array1<f64> {
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let trial = inside + &(outside - inside) * mid;
            if self.outside(trial.as_slice().unwrap()) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        inside + &(outside - inside) * lo
    }

    fn observe<S: PointSurface>(&mut self, surface: &S, x: Array1<f64>) -> Result<(), SaddleError> {
        if self
            .observations
            .iter()
            .any(|obs| !moved(&obs.position, &x))
        {
            return Ok(());
        }
        let (energy, gradient) = surface.eval(x.view())?;
        self.push_observation(Observation {
            position: x,
            energy,
            gradient,
        });
        Ok(())
    }

    fn push_observation(&mut self, observation: Observation) {
        self.note_energy(observation.energy);
        if let Some(slot) = self
            .observations
            .iter_mut()
            .find(|obs| !moved(&obs.position, &observation.position))
        {
            *slot = observation;
            return;
        }
        self.observations.push(observation);
    }

    fn note_energy(&mut self, energy: f64) {
        if energy < self.lowest_energy {
            self.lowest_energy = energy;
        }
    }

    fn climbed_out(&self, energy: f64) -> bool {
        self.lowest_energy.is_finite() && energy > self.lowest_energy + CLIMB_LIMIT
    }

    fn project_rigid(&self) -> bool {
        self.x.len() >= 6
            && self.x.len().is_multiple_of(3)
            && self.config.cell.is_none()
            && self.config.fixed.is_none()
    }

    fn trim_observations(&mut self) {
        self.lock_active();
        let width = self.model_dimension() + 1;
        // A few gradient samples are required for a nonzero curvature.
        // Past that, gpridge drops the training point farthest from the
        // current geometry, and anything outside a few trust radii.
        // Keep enough geometries that a new observation is a rank-one
        // append. Pruning starts only after that window is full.
        // A wide active region stays on a few geometries: growing the
        // factor past that costs more than one refactor of the small set.
        // Narrow kernels keep a longer window and extend it in place.
        let cap = if width >= 200 {
            3
        } else {
            self.config
                .max_observations
                .max(4)
                .min((1_600 / width).max(4))
        };
        while self.observations.len() > cap {
            let Some((farthest, _)) = self.farthest_training_point() else {
                break;
            };
            self.observations.remove(farthest);
        }
    }

    /// Farthest stored geometry from the current point, ignoring the
    /// newest observation so the point just evaluated stays in the fit.
    fn farthest_training_point(&self) -> Option<(usize, f64)> {
        let current = self.x.as_slice().unwrap();
        let last = self.observations.len().saturating_sub(1);
        let mut best: Option<(usize, f64)> = None;
        for (i, obs) in self.observations.iter().enumerate() {
            if i == last {
                continue;
            }
            let gap = self.layout.gap(obs.position.as_slice().unwrap(), current);
            if best.is_none_or(|(_, so_far)| gap > so_far) {
                best = Some((i, gap));
            }
        }
        best
    }

    /// Rank-one append when the training set grew by one point.
    /// Otherwise one factorization at frozen hyperparameters, and a
    /// marginal-likelihood search on the first fit and every fourth
    /// small-system update.
    /// Moving atoms and every atom inside [`GpDimerConfig::active_cutoff`].
    /// Far atoms are not features and get no surrogate force. The set is
    /// fixed after the opening mode so later observations extend one factor.
    fn lock_active(&mut self) {
        if !self.active.is_empty() || self.layout.cartesian() || self.config.active_cutoff <= 0.0 {
            return;
        }
        let atoms = self.x.len() / 3;
        if atoms <= 20 || !self.x.len().is_multiple_of(3) {
            return;
        }
        let mut seeds = Vec::new();
        for atom in 0..atoms {
            let mut weight = 0.0;
            for axis in 0..3 {
                let component = self.mode[3 * atom + axis];
                weight += component * component;
            }
            if weight.sqrt() > 0.05 {
                seeds.push(atom);
            }
        }
        if seeds.is_empty() {
            let mut best = 0usize;
            let mut weight = 0.0;
            for atom in 0..atoms {
                let mut local = 0.0;
                for axis in 0..3 {
                    let component = self.mode[3 * atom + axis];
                    local += component * component;
                }
                if local > weight {
                    weight = local;
                    best = atom;
                }
            }
            seeds.push(best);
        }
        let cutoff = self.config.active_cutoff;
        let mut active = Vec::new();
        for atom in 0..atoms {
            let inside = seeds
                .iter()
                .any(|&seed| seed == atom || self.pair_distance(seed, atom) <= cutoff);
            if inside {
                active.push(atom);
            }
        }
        if active.len() < atoms {
            self.active = active;
            self.model = None;
        }
    }

    fn pair_distance(&self, left: usize, right: usize) -> f64 {
        let mut square = 0.0;
        for axis in 0..3 {
            let mut delta = self.x[3 * left + axis] - self.x[3 * right + axis];
            if let Some(cell) = self.config.cell {
                delta -= cell[axis] * (delta / cell[axis]).round();
            }
            square += delta * delta;
        }
        square.sqrt()
    }

    fn model_dimension(&self) -> usize {
        if self.active.is_empty() {
            self.x.len()
        } else {
            self.active.len() * 3
        }
    }

    fn model_kind(&self) -> KernelKind {
        if self.active.is_empty() {
            self.kind.clone()
        } else {
            KernelKind::InverseDistance {
                cell: self.config.cell,
                fixed: None,
            }
        }
    }

    fn training_set(&self) -> Vec<Observation> {
        if self.active.is_empty() {
            return self.observations.clone();
        }
        self.observations
            .iter()
            .map(|observation| self.slice_observation(observation))
            .collect()
    }

    fn slice_observation(&self, observation: &Observation) -> Observation {
        let mut position = Array1::zeros(self.active.len() * 3);
        let mut gradient = Array1::zeros(self.active.len() * 3);
        for (slot, atom) in self.active.iter().enumerate() {
            for axis in 0..3 {
                position[3 * slot + axis] = observation.position[3 * atom + axis];
                gradient[3 * slot + axis] = observation.gradient[3 * atom + axis];
            }
        }
        Observation {
            position,
            energy: observation.energy,
            gradient,
        }
    }

    fn refresh_model(&mut self) -> Result<bool, SaddleError> {
        self.lock_active();
        let noise = 1e-6;
        let training = self.training_set();
        let kind = self.model_kind();
        if let Some(model) = &self.model
            && model.same_as(&training)
        {
            return Ok(false);
        }
        if let Some(model) = &self.model
            && model.is_prefix_of(&training)
            && self.extends_since_fit < 4
        {
            let observation = training.last().unwrap().clone();
            if let Ok(grown) = model.extend(&observation) {
                self.model = Some(grown);
                self.extends_since_fit += 1;
                return Ok(true);
            }
        }
        if let Some(model) = &self.model
            && self.extends_since_fit < 4
            && let Ok(rebuilt) = model.rebuild_frozen(&training, noise)
        {
            self.model = Some(rebuilt);
            self.extends_since_fit += 1;
            return Ok(false);
        }
        let fitted = if self.model_dimension() >= 120 {
            GaussianProcess::fit_warm(&training, &kind, noise)?
        } else {
            GaussianProcess::fit(
                &training,
                &kind,
                &crate::gpr::FitOptions {
                    passes: self.config.fit_passes,
                    noise,
                },
            )?
        };
        self.model = Some(fitted);
        self.extends_since_fit = 0;
        Ok(false)
    }
}

fn perpendicular_to(mode: &Array1<f64>) -> Array1<f64> {
    let mut axis = 0usize;
    let mut smallest = f64::INFINITY;
    for (i, component) in mode.iter().enumerate() {
        if component.abs() < smallest {
            smallest = component.abs();
            axis = i;
        }
    }
    let mut direction = Array1::zeros(mode.len());
    direction[axis] = 1.0;
    let proj = mode.dot(&direction);
    direction = &direction - &(mode * proj);
    let norm = direction.dot(&direction).sqrt();
    if norm > 1e-14 {
        direction /= norm;
    }
    direction
}

fn kernel_kind(config: &GpDimerConfig, dim: usize) -> KernelKind {
    let fixed_present = config.fixed.as_ref().is_some_and(|v| v.len() >= 3);
    let molecular = fixed_present || (dim.is_multiple_of(3) && dim >= 6);
    if molecular {
        KernelKind::InverseDistance {
            cell: config.cell,
            fixed: config.fixed.clone(),
        }
    } else {
        KernelKind::Cartesian
    }
}

fn moved(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    let mut sum = 0.0;
    for (left, right) in a.iter().zip(b.iter()) {
        let d = left - right;
        sum += d * d;
    }
    sum.sqrt() > 1e-10
}

fn true_curvature<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    dr: f64,
) -> Result<f64, SaddleError> {
    let mut shifted = x.to_owned();
    for i in 0..x.len() {
        shifted[i] += dr * mode[i];
    }
    let (_, displaced) = surface.eval(shifted.view())?;
    let mut curvature = 0.0;
    for i in 0..x.len() {
        curvature += (displaced[i] - gradient[i]) / dr * mode[i];
    }
    if !curvature.is_finite() {
        return Err(SaddleError::NonFinite("true curvature"));
    }
    Ok(curvature)
}
