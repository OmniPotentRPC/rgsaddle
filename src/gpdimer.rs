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

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::gpr::{GaussianProcess, KernelKind, Layout, Observation};
use crate::minmode::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface};

/// Literature trust radius: `ln(1.5)`, i.e. every pair distance stays
/// inside `(2/3, 3/2)` of a stored geometry.
pub const LN_THREE_HALVES: f64 = 0.4054651081081644;

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
    /// True surface evaluations during this call.
    pub evaluations: usize,
    pub iteration: usize,
    pub inner_steps: usize,
}

struct Surrogate<'a> {
    gp: &'a GaussianProcess,
    /// Free cluster: drop rigid motion. Periodic: drop translations.
    rotations: bool,
    translations: bool,
}

impl PointSurface for Surrogate<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.gp.predict(x)
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
        if !self.seeded {
            self.seed_mode(&counted)?;
            self.seeded = true;
        }
        // The opening rotation may already sit on the saddle.
        if let Some(gradient) = self.anchor_gradient() {
            let max_force = self.force_gate.value(gradient.view());
            if max_force <= self.force_tol && self.curvature < 0.0 {
                self.iteration += 1;
                return Ok(GpDimerReport {
                    status: GpDimerStatus::Converged,
                    max_force,
                    curvature: self.curvature,
                    evaluations: calls.load(Ordering::Relaxed),
                    iteration: self.iteration,
                    inner_steps: 0,
                });
            }
        }
        self.trim_observations();
        let passes = if self.x.len() >= 200 {
            self.config.fit_passes.min(2)
        } else {
            self.config.fit_passes
        };
        let gp = GaussianProcess::fit(
            &self.observations,
            &self.kind,
            &crate::gpr::FitOptions {
                passes,
                noise: 1e-6,
            },
        )?;
        let surrogate = Surrogate {
            gp: &gp,
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
        let (_, surrogate_gradient) = gp.predict(self.x.view())?;
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
        if !energy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
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
        self.push_observation(Observation {
            position: accepted.clone(),
            energy,
            gradient,
        });
        self.x = accepted;
        self.mode = mode.clone();
        self.true_mode = mode;
        self.curvature = curvature;
        let status = if max_force <= self.force_tol && curvature < 0.0 {
            GpDimerStatus::Converged
        } else {
            GpDimerStatus::Running
        };
        self.iteration += 1;
        Ok(GpDimerReport {
            status,
            max_force,
            curvature,
            evaluations: calls.load(Ordering::Relaxed),
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
        for _ in 0..80 {
            let report = finisher.step(surface)?;
            self.x = finisher.position().to_owned();
            self.mode = finisher.mode().to_owned();
            curvature = report.curvature;
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
        self.iteration += 1;
        let status = if max_force <= self.force_tol && curvature < 0.0 {
            GpDimerStatus::Converged
        } else {
            GpDimerStatus::Running
        };
        Ok(GpDimerReport {
            status,
            max_force,
            curvature,
            evaluations: 0,
            iteration: self.iteration,
            inner_steps: 0,
        })
    }

    fn exclude_translations(&self) -> bool {
        !self.layout.cartesian() && self.layout.has_pairs() && self.x.len() >= 6
    }

    fn exclude_rotations(&self) -> bool {
        self.exclude_translations() && self.config.cell.is_none() && self.config.fixed.is_none()
    }

    fn outside(&self, x: &[f64]) -> bool {
        self.nearest_gap(x) > self.step_limit
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

    fn trim_observations(&mut self) {
        let width = self.x.len() + 1;
        // A single gradient observation has zero posterior curvature, so
        // even a large system keeps a few recent geometries. The window
        // is the prune: hyperparameters are refit on what remains.
        let by_size = if width >= 400 {
            3
        } else {
            (640 / width).max(3)
        };
        let cap = self.config.max_observations.max(3).min(by_size);
        while self.observations.len() > cap {
            self.observations.remove(0);
        }
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
