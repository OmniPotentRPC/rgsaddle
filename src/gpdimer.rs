//! Gaussian-process minimum-mode search.
//!
//! Opt-in. [`MinModeConfig::default`] stays the plain dimer with the
//! FIRE translator. A [`GpDimerSession`] builds the surrogate in
//! [`crate::gpr`], then rotates and translates it with
//! [`crate::minmode::MinModeSession`] and the configured [`rgmin`]
//! stepper. The trust radius is the 1D max-log gap of inverse
//! distances (Goswami and Jónsson, ChemPhysChem 27 (2026),
//! doi:10.1002/cphc.202500730, Eq. 7; the `2/3 < r'/r < 3/2` window of
//! Koistinen, Ásgeirsson, Vehtari, and Jónsson, J. Chem. Theory Comput.
//! 16, 499 (2020), doi:10.1021/acs.jctc.9b01038). The surrogate walk
//! stops when that gap, against the nearest true observation, exceeds
//! the radius. Convergence is declared only from the true gradient.
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

/// Surrogate rotations farther than this from the true mode are dropped.
const MODE_LOCK: f64 = 0.35;

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
    /// Bound used for the current surrogate walk. Never wider than `radius`.
    step_limit: f64,
    /// Working trust bound. It opens inside [`GpDimerConfig::trust_radius`]
    /// and grows when a true-force step improves, because one gradient
    /// observation does not identify the curvature (Goswami and Jónsson,
    /// the initial rotation on the true surface).
    radius: f64,
    /// The unstable mode has been read on the true surface and a second
    /// sample has been taken along it.
    seeded: bool,
    /// The residual is inside the surrogate bias, so the remaining steps
    /// are plain dimer steps on the true surface.
    polish: bool,
    /// Lowest mode from the true surface. Surrogate rotation may turn it
    /// a little; a turn past [`MODE_LOCK`] is discarded so a bad Hessian
    /// cannot point the dimer at a dissociation.
    true_mode: Array1<f64>,
    /// Consecutive surrogate steps the true force rejected.
    rejects: u32,
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
        let radius = opening_radius(&kind, config.trust_radius);
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
            radius,
            step_limit: radius,
            seeded: false,
            polish: false,
            true_mode,
            rejects: 0,
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
        if self.polish {
            let mut report = self.plain_finish(&counted)?;
            report.evaluations = calls.load(Ordering::Relaxed);
            return Ok(report);
        }
        self.observe(&counted, self.x.clone())?;
        if !self.seeded {
            self.seed_mode(&counted)?;
            self.seeded = true;
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
        // The surrogate Hessian is not identified until several gradient
        // samples exist. Until then the translation uses the true mode.
        if self.observations.len() < 4 {
            inner.max_rotations = 0;
        }
        if let Some(nearest) = self.layout.minimum_distance(self.x.as_slice().unwrap()) {
            let cap = 0.5 * (1.0 - self.config.ratio_at_limit) * nearest;
            if cap.is_finite() && cap > 0.0 {
                inner.max_move = inner.max_move.min(cap.max(1e-4));
            }
        }
        // A Newton length from the true curvature keeps a late step from
        // crossing the saddle and hitting the far side of the trust ball.
        self.step_limit = self.radius;
        // Only the polish, once the walker is already near a stationary
        // point. Earlier, the opening radius limits the extrapolation.
        if self.curvature < -1.0
            && let Some(gradient) = self.anchor_gradient()
        {
            let force = self.force_gate.value(gradient.view());
            let newton = force / self.curvature.abs();
            if force < 1.0 && newton.is_finite() {
                self.step_limit = self.step_limit.min((10.0 * newton).max(5e-3));
            }
        }
        if self.layout.cartesian() {
            inner.max_move = inner.max_move.min((0.5 * self.step_limit).max(1e-4));
        }
        let mut walker = MinModeSession::new(inner, self.x.clone(), self.true_mode.clone())?;
        if let Some(gradient) = self.anchor_gradient() {
            walker.set_position(self.x.clone(), Some(gradient))?;
        }
        if self.curvature.is_finite() {
            walker.install_mode(self.true_mode.clone(), self.curvature)?;
        }
        let inner_limit = if self.curvature.is_finite() {
            1
        } else {
            self.config.max_inner_steps
        };
        let mut accepted = self.x.clone();
        let mut mode = self.mode.clone();
        let mut curvature = self.curvature;
        let mut inner_steps = 0usize;
        for _ in 0..inner_limit {
            let before = walker.position().to_owned();
            let report = walker.step(&surrogate)?;
            let after = walker.position().to_owned();
            if self.outside(after.as_slice().unwrap()) {
                let pulled = self.pull_inside(&before, &after);
                if moved(&before, &pulled) {
                    accepted = pulled;
                    mode = self.locked_mode(walker.mode());
                    curvature = report.curvature;
                    inner_steps += 1;
                }
                break;
            }
            accepted = after;
            mode = self.locked_mode(walker.mode());
            curvature = report.curvature;
            inner_steps += 1;
            if report.status == MinModeStatus::Converged {
                break;
            }
        }
        let stalled = !moved(&self.x, &accepted);
        if stalled {
            let mut plain_cfg = self.config.inner.clone();
            plain_cfg.max_move = plain_cfg.max_move.min(self.radius.max(1e-3));
            let mut plain = MinModeSession::new(plain_cfg, self.x.clone(), self.mode.clone())?;
            if let Some(gradient) = self.anchor_gradient() {
                plain.set_position(self.x.clone(), Some(gradient))?;
            }
            let report = plain.step(&counted)?;
            accepted = plain.position().to_owned();
            mode = plain.mode().to_owned();
            self.true_mode = mode.clone();
            curvature = report.curvature;
        }
        let anchor_force = self
            .anchor_gradient()
            .map(|gradient| self.force_gate.value(gradient.view()))
            .unwrap_or(f64::INFINITY);
        let (energy, gradient) = counted.eval(accepted.view())?;
        let mut max_force = self.force_gate.value(gradient.view());
        let runaway = !max_force.is_finite()
            || !energy.is_finite()
            || max_force > 1.0e6
            || (moved(&self.x, &accepted)
                && max_force > anchor_force * 4.0
                && max_force > anchor_force + 0.5);
        if runaway {
            // The surrogate step left the region the model knows. Stay
            // at the last true point and tighten the trust radius. The
            // rejected geometry is not stored: a wall evaluation would
            // dominate the marginal likelihood.
            self.radius = (self.radius * 0.5).max(0.04);
            self.rejects = self.rejects.saturating_add(1);
            if let Some(gradient) = self.anchor_gradient() {
                max_force = self.force_gate.value(gradient.view());
            }
            curvature = self.curvature;
        } else {
            if max_force <= self.force_tol {
                curvature = true_curvature(
                    &counted,
                    accepted.view(),
                    gradient.view(),
                    mode.view(),
                    self.config.inner.dr,
                )?;
            }
            if max_force < anchor_force {
                self.radius = (self.radius * 1.25).min(self.config.trust_radius);
                self.rejects = 0;
            }
            self.push_observation(Observation {
                position: accepted.clone(),
                energy,
                gradient: gradient.clone(),
            });
            self.x = accepted;
            self.mode = mode;
            self.curvature = curvature;
        }
        let close = self.iteration > 2 && curvature < -1.0 && max_force <= 0.05;
        if !self.polish && close && max_force > self.force_tol {
            // Closer than the surrogate can place a stationary point.
            // The plain dimer finishes on the true force.
            self.polish = true;
            let mut report = self.plain_finish(&counted)?;
            report.evaluations = calls.load(Ordering::Relaxed);
            return Ok(report);
        }
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

    fn plain_finish<S: PointSurface>(&mut self, surface: &S) -> Result<GpDimerReport, SaddleError> {
        let mut plain =
            MinModeSession::new(self.config.inner.clone(), self.x.clone(), self.mode.clone())?;
        if let Some(gradient) = self.anchor_gradient() {
            plain.set_position(self.x.clone(), Some(gradient))?;
        }
        let report = plain.step(surface)?;
        self.x = plain.position().to_owned();
        self.mode = plain.mode().to_owned();
        self.true_mode = self.mode.clone();
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
            report.curvature
        };
        self.push_observation(Observation {
            position: self.x.clone(),
            energy,
            gradient,
        });
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

    /// Keep a surrogate rotation that stays near the true mode.
    fn locked_mode(&self, proposed: ArrayView1<f64>) -> Array1<f64> {
        let overlap = self.true_mode.dot(&proposed).abs().min(1.0);
        if overlap.acos() > MODE_LOCK {
            self.true_mode.clone()
        } else {
            proposed.to_owned()
        }
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
        let mut rotator =
            MinModeSession::new(self.config.inner.clone(), self.x.clone(), self.mode.clone())?;
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
            if gap <= 0.5 * self.radius || probe <= self.config.inner.dr * 2.0 {
                break;
            }
            probe *= 0.5;
        }
        self.observe(surface, shifted)
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
        let by_size = if width >= 512 {
            1
        } else {
            (480 / width).max(1)
        };
        let cap = self.config.max_observations.max(1).min(by_size);
        while self.observations.len() > cap {
            self.observations.remove(0);
        }
    }
}

fn opening_radius(kind: &KernelKind, trust: f64) -> f64 {
    let cap = match kind {
        KernelKind::Cartesian => 0.08,
        KernelKind::InverseDistance { .. } => 0.06,
    };
    trust.min(cap)
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
