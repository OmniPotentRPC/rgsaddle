//! Off-path climbing image: the band hands its climbing image to a
//! minimum-mode search.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. 14 (2026),
//! doi:10.3389/fchem.2026.1807063. The host calls [`OcinebSession::step`].
//! Each call is one band step, one alignment, or one minimum-mode step,
//! and the report names that phase.
//!
//! The switch waits until the climbing-image index has stayed fixed for
//! more than `stability_count` iterations and the band force is under
//! the threshold. The threshold starts at `trigger_factor` times the
//! force of the initial path, and it is at least twice the force
//! tolerance. Alignment rotates the climbing-image tangent onto the
//! lowest Hessian eigenmode. The search starts only when that curvature
//! is negative and the absolute overlap with the tangent is at least
//! `angle_tol` (`1/sqrt(2)` or above; the published value is 0.85).
//!
//! The search returns to the band when the mode leaves that cone, the
//! curvature turns positive, the step budget is spent, or the image
//! meets the force tolerance. Positive curvature puts the climbing
//! image back where the hand-off started and clears the cached mode.
//! A walk that does not lower the force raises the threshold by the
//! linear penalty `(1 + α) / 2`. A walk that does lower it sets the
//! threshold to `F_new (1/2 + 2/5 F_new / F_CI)`, keeps the mode when
//! the image converged inside the cone, and respaces the path at equal
//! arc length with the climbing image held fixed.

use ndarray::{Array1, Array2, ArrayView1};

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface};
use crate::rtr::{reparametrize_equal_arc, reparametrize_equal_arc_in_cell};
use crate::tangent::{self, TangentKind};

/// Published floor on the tangent overlap. Below it the force inversion
/// points away from the saddle of the path.
pub const MIN_ALIGNMENT: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Which solver the last step ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OcinebPhase {
    /// One climbing-image band step.
    Band = 0,
    /// Lowest-mode rotation at the climbing image. No translation.
    Align = 1,
    /// One minimum-mode translation of the climbing image.
    MinMode = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OcinebStatus {
    Running = 0,
    Converged = 1,
}

/// Band, minimum-mode, and the hand-off parameters.
#[derive(Clone, Debug)]
pub struct OcinebConfig {
    pub band: BandConfig,
    pub minmode: MinModeConfig,
    /// Relative trigger, `ci_mmf_after_rel`. Published value 0.31.
    pub trigger_factor: f64,
    /// Absolute trigger, `ci_mmf_after`. Used when `trigger_factor` is 0.
    pub trigger_force: f64,
    /// Minimum `|mode · tangent|`. Published value 0.85.
    pub angle_tol: f64,
    /// Stable climbing-image iterations required before a hand-off.
    /// The counter must exceed this value. Published value 5.
    pub stability_count: i64,
    /// Minimum-mode translations per hand-off. Published value 1000.
    pub max_mmf_steps: usize,
    /// Also restore the climbing image when a walk does not lower the force.
    pub restore_unhelpful: bool,
}

impl Default for OcinebConfig {
    fn default() -> Self {
        let band = BandConfig::default();
        let minmode = MinModeConfig {
            force_tol: band.force_tol,
            max_move: band.max_move,
            ..MinModeConfig::default()
        };
        Self {
            band,
            minmode,
            trigger_factor: 0.31,
            trigger_force: 0.0,
            angle_tol: 0.85,
            stability_count: 5,
            max_mmf_steps: 1000,
            restore_unhelpful: false,
        }
    }
}

/// One host step.
#[derive(Clone, Debug)]
pub struct OcinebReport {
    pub phase: OcinebPhase,
    pub status: OcinebStatus,
    pub max_force: f64,
    pub ci_index: Option<usize>,
    pub iteration: usize,
    pub evaluations: usize,
    pub curvature: f64,
    pub alignment: f64,
    pub rotations: usize,
    /// The minimum-mode search stopped and the band has the next step.
    pub fell_back: bool,
    pub threshold: f64,
}

/// Initial threshold: `max(λ F0, F_abs, 2 tol)`, with `F_abs` only when `λ` is 0.
pub fn initial_threshold(baseline: f64, factor: f64, absolute: f64, force_tol: f64) -> f64 {
    let relative = baseline * factor;
    let abs_term = if factor == 0.0 { absolute } else { 0.0 };
    relative.max(abs_term).max(2.0 * force_tol)
}

/// `F_new (1/2 + 2/5 F_new / F_CI)`, capped at the initial threshold.
pub fn success_threshold(
    conv_force: f64,
    new_force: f64,
    baseline: f64,
    factor: f64,
    force_tol: f64,
) -> f64 {
    let updated = if conv_force > 0.0 {
        new_force * (0.5 + 0.4 * (new_force / conv_force))
    } else {
        0.0
    };
    let cap = initial_threshold(baseline, factor, 0.0, force_tol);
    updated.min(cap)
}

/// Linear penalty `(1/2 + 1/2 α)` on the initial relative threshold,
/// floored at twice the force tolerance.
pub fn backoff_threshold(alignment: f64, baseline: f64, factor: f64, force_tol: f64) -> f64 {
    let alpha = alignment.clamp(0.0, 1.0);
    let penalty = 0.5 + 0.5 * alpha;
    (baseline * factor * penalty).max(2.0 * force_tol)
}

/// The walked image lowered the band force, and the curvature was not positive.
pub fn walk_helped(walked_force: f64, conv_force: f64, positive_curvature: bool) -> bool {
    walked_force < conv_force && !positive_curvature
}

/// `|û · τ̂|`. A zero vector scores 0.
pub fn mode_alignment(mode: ArrayView1<f64>, tangent: ArrayView1<f64>) -> f64 {
    let mn = mode.dot(&mode).sqrt();
    let tn = tangent.dot(&tangent).sqrt();
    if mn <= 1e-14 || tn <= 1e-14 {
        return 0.0;
    }
    (mode.dot(&tangent) / (mn * tn)).abs()
}

/// Hand-off while the climbing image is real, stable, and not yet converged.
#[allow(clippy::too_many_arguments)]
pub fn should_trigger(
    ci_active: bool,
    ci_index: Option<usize>,
    n_interior: usize,
    stability: i64,
    stability_count: i64,
    conv_force: f64,
    force_tol: f64,
    threshold: f64,
) -> bool {
    if !ci_active {
        return false;
    }
    let Some(ci) = ci_index else {
        return false;
    };
    if ci == 0 || ci > n_interior {
        return false;
    }
    if stability <= stability_count {
        return false;
    }
    if conv_force <= force_tol {
        return false;
    }
    conv_force < threshold
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    PositiveCurvature,
    Misaligned,
    Converged,
    Budget,
}

#[derive(Clone)]
struct Mark {
    x: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    force: f64,
}

struct Climb {
    ci: usize,
    tangent: Array1<f64>,
    gradient: Array1<f64>,
    position: Array1<f64>,
}

/// Host-stepped CI-NEB to minimum-mode hand-off.
pub struct OcinebSession {
    band: BandSession,
    tangent_kind: TangentKind,
    cell: Option<crate::band::Cell>,
    force_tol: f64,
    minmode_config: MinModeConfig,
    trigger_factor: f64,
    trigger_force: f64,
    angle_tol: f64,
    stability_count: i64,
    max_mmf_steps: usize,
    restore_unhelpful: bool,
    force_gate: ForceGate,
    highs: bool,
    phase: OcinebPhase,
    baseline: Option<f64>,
    threshold: f64,
    previous_ci: i64,
    stability: i64,
    cached_mode: Option<Array1<f64>>,
    minmode: Option<MinModeSession>,
    walked: usize,
    tangent_at_switch: Option<Array1<f64>>,
    force_at_switch: f64,
    saved_positions: Option<Array2<f64>>,
    best: Option<Mark>,
    mmf_steps: usize,
    host_steps: usize,
    fell_back: bool,
}

impl OcinebSession {
    pub fn new(config: OcinebConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        validate(&config)?;
        crate::band::check_force_driven(&config.minmode.method)?;
        if !(config.minmode.dr.is_finite() && config.minmode.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        let tangent_kind = config.band.tangent;
        let cell = config.band.cell;
        let force_tol = config.band.force_tol;
        let minmode_config = config.minmode.clone();
        let trigger_factor = config.trigger_factor;
        let trigger_force = config.trigger_force;
        let angle_tol = config.angle_tol;
        let stability_count = config.stability_count;
        let max_mmf_steps = config.max_mmf_steps;
        let restore_unhelpful = config.restore_unhelpful;
        let band = BandSession::new(config.band, initial)?;
        Ok(Self {
            band,
            tangent_kind,
            cell,
            force_tol,
            minmode_config,
            trigger_factor,
            trigger_force,
            angle_tol,
            stability_count,
            max_mmf_steps,
            restore_unhelpful,
            force_gate: ForceGate::LinfNorm,
            highs: false,
            phase: OcinebPhase::Band,
            baseline: None,
            threshold: f64::NAN,
            previous_ci: -1,
            stability: 0,
            cached_mode: None,
            minmode: None,
            walked: 0,
            tangent_at_switch: None,
            force_at_switch: f64::NAN,
            saved_positions: None,
            best: None,
            mmf_steps: 0,
            host_steps: 0,
            fell_back: false,
        })
    }

    pub fn phase(&self) -> OcinebPhase {
        self.phase
    }

    pub fn threshold(&self) -> Option<f64> {
        self.baseline.map(|_| self.threshold)
    }

    pub fn stability(&self) -> i64 {
        self.stability
    }

    pub fn positions(&self) -> ndarray::ArrayView2<'_, f64> {
        self.band.positions()
    }

    pub fn climbing_image(&self) -> Option<usize> {
        self.band.climbing_image()
    }

    pub fn set_force_gate(&mut self, gate: ForceGate) {
        if self.force_gate == gate {
            return;
        }
        self.force_gate = gate;
        self.band.set_force_gate(gate);
        self.baseline = None;
        self.threshold = f64::NAN;
        self.previous_ci = -1;
        self.stability = 0;
        self.cached_mode = None;
        self.clear_episode();
        self.phase = OcinebPhase::Band;
    }

    pub fn set_highs(&mut self, enabled: bool) {
        self.highs = enabled;
        self.band.set_highs(enabled);
        if let Some(mm) = &mut self.minmode {
            mm.set_highs(enabled);
        }
    }

    #[cfg(feature = "capi")]
    pub(crate) fn set_rtr(
        &mut self,
        config: Option<crate::rtr::RtrConfig>,
    ) -> Result<(), SaddleError> {
        self.band.set_rtr(config)
    }

    /// Drop the band caches and the hand-off state. The next step
    /// reads a new baseline from the path it is given.
    pub fn reset(&mut self) {
        self.band.reset();
        self.baseline = None;
        self.threshold = f64::NAN;
        self.previous_ci = -1;
        self.stability = 0;
        self.cached_mode = None;
        self.clear_episode();
        self.phase = OcinebPhase::Band;
    }

    /// One band step, one alignment, or one minimum-mode step.
    pub fn step<S: BandSurface + PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<OcinebReport, SaddleError> {
        self.host_steps += 1;
        self.fell_back = false;
        match self.phase {
            OcinebPhase::Band => self.step_band(surface),
            OcinebPhase::Align => self.step_align(surface),
            OcinebPhase::MinMode => self.step_minmode(surface),
        }
    }

    pub fn run<S: BandSurface + PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<OcinebReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut n = 1;
        while report.status == OcinebStatus::Running && n < max_steps {
            report = self.step(surface)?;
            n += 1;
        }
        Ok(report)
    }

    fn step_band<S: BandSurface + PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<OcinebReport, SaddleError> {
        let report = self.band.step(surface)?;
        if self.baseline.is_none()
            && let Some(baseline) = self.band.force_baseline()
        {
            self.baseline = Some(baseline);
            self.threshold = initial_threshold(
                baseline,
                self.trigger_factor,
                self.trigger_force,
                self.force_tol,
            );
        }
        let ci = self.band.climbing_image();
        self.update_stability(ci.map(|i| i as i64).unwrap_or(0));
        let n_interior = self.band.positions().nrows().saturating_sub(2);
        let switch = report.status == BandStatus::Running
            && should_trigger(
                ci.is_some(),
                ci,
                n_interior,
                self.stability,
                self.stability_count,
                report.max_force,
                self.force_tol,
                self.threshold,
            );
        if switch {
            self.force_at_switch = report.max_force;
            self.phase = OcinebPhase::Align;
        }
        let status = if report.status == BandStatus::Converged {
            OcinebStatus::Converged
        } else {
            OcinebStatus::Running
        };
        Ok(self.make_report(
            OcinebPhase::Band,
            status,
            report.max_force,
            report.ci_index,
            report.surface_rows,
            f64::NAN,
            f64::NAN,
            0,
        ))
    }

    fn step_align<S: BandSurface + PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<OcinebReport, SaddleError> {
        let frame = self.climbing_frame()?;
        let seed = match &self.cached_mode {
            Some(mode) if mode.len() == frame.tangent.len() => mode.clone(),
            _ => frame.tangent.clone(),
        };
        let mut mm =
            MinModeSession::new(self.minmode_config.clone(), frame.position.clone(), seed)?;
        mm.set_force_gate(self.force_gate);
        mm.set_highs(self.highs);
        mm.set_position(frame.position.clone(), Some(frame.gradient.clone()))?;
        let est = mm.estimate_mode(surface)?;
        let mut mode = est.mode;
        if mode.dot(&frame.tangent) < 0.0 {
            mode *= -1.0;
            mm.set_mode(mode.clone())?;
        }
        let alignment = mode_alignment(mode.view(), frame.tangent.view());
        let ci_force = self.force_gate.value(frame.gradient.view());
        if est.curvature > 0.0 || alignment < self.angle_tol {
            let positive = est.curvature > 0.0;
            self.abort_before_move(positive, alignment);
            return Ok(self.make_report(
                OcinebPhase::Align,
                OcinebStatus::Running,
                self.force_at_switch,
                Some(frame.ci),
                est.evaluations,
                est.curvature,
                if positive { 0.0 } else { alignment },
                est.rotations,
            ));
        }
        self.walked = frame.ci;
        self.tangent_at_switch = Some(frame.tangent);
        self.saved_positions = Some(self.band.positions().to_owned());
        self.best = Some(Mark {
            x: frame.position,
            mode: mode.clone(),
            curvature: est.curvature,
            force: ci_force,
        });
        self.mmf_steps = 0;
        self.cached_mode = Some(mode);
        self.minmode = Some(mm);
        self.phase = OcinebPhase::MinMode;
        Ok(self.make_report(
            OcinebPhase::Align,
            OcinebStatus::Running,
            ci_force,
            Some(frame.ci),
            est.evaluations,
            est.curvature,
            alignment,
            est.rotations,
        ))
    }

    fn step_minmode<S: BandSurface + PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<OcinebReport, SaddleError> {
        let mm = self
            .minmode
            .as_mut()
            .ok_or_else(|| SaddleError::Invalid("minimum-mode phase has no session".into()))?;
        let report = mm.step(surface)?;
        let mode = mm.mode().to_owned();
        let position = mm.position().to_owned();
        let tangent = self
            .tangent_at_switch
            .clone()
            .ok_or_else(|| SaddleError::Invalid("minimum-mode phase has no tangent".into()))?;
        let alignment = mode_alignment(mode.view(), tangent.view());
        self.consider_best(
            position.clone(),
            mode.clone(),
            report.curvature,
            report.max_force,
        );
        self.write_ci(position.clone())?;
        self.mmf_steps += 1;

        let stop = if report.curvature > 0.0 {
            Some(Stop::PositiveCurvature)
        } else if alignment < self.angle_tol {
            Some(Stop::Misaligned)
        } else if report.status == MinModeStatus::Converged {
            Some(Stop::Converged)
        } else if self.mmf_steps >= self.max_mmf_steps {
            Some(Stop::Budget)
        } else {
            None
        };
        if let Some(stop) = stop {
            self.close_search(
                stop,
                position,
                mode,
                report.curvature,
                report.max_force,
                alignment,
            )?;
            return Ok(self.make_report(
                OcinebPhase::MinMode,
                OcinebStatus::Running,
                report.max_force,
                Some(self.walked),
                report.evaluations,
                report.curvature,
                alignment,
                report.rotations,
            ));
        }
        Ok(self.make_report(
            OcinebPhase::MinMode,
            OcinebStatus::Running,
            report.max_force,
            Some(self.walked),
            report.evaluations,
            report.curvature,
            alignment,
            report.rotations,
        ))
    }

    fn abort_before_move(&mut self, positive_curvature: bool, alignment: f64) {
        let baseline = self.baseline.unwrap_or(0.0);
        let alpha = if positive_curvature { 0.0 } else { alignment };
        self.threshold = backoff_threshold(alpha, baseline, self.trigger_factor, self.force_tol);
        if positive_curvature {
            self.cached_mode = None;
        }
        self.minmode = None;
        self.stability = 0;
        self.phase = OcinebPhase::Band;
        self.fell_back = true;
    }

    fn close_search(
        &mut self,
        stop: Stop,
        position: Array1<f64>,
        mode: Array1<f64>,
        _curvature: f64,
        force: f64,
        alignment: f64,
    ) -> Result<(), SaddleError> {
        let positive = stop == Stop::PositiveCurvature;
        let keep_best = matches!(stop, Stop::Misaligned | Stop::Budget);
        let (commit_x, commit_mode, commit_force, commit_align) = if positive {
            (position, mode, force, 0.0)
        } else if keep_best && let Some(best) = self.best.clone() {
            let align = self
                .tangent_at_switch
                .as_ref()
                .map(|t| mode_alignment(best.mode.view(), t.view()))
                .unwrap_or(0.0);
            (best.x, best.mode, best.force, align)
        } else {
            (position, mode, force, alignment)
        };
        let helped = walk_helped(commit_force, self.force_at_switch, positive);
        let restore = positive || (self.restore_unhelpful && !helped);
        let baseline = self.baseline.unwrap_or(0.0);
        if restore {
            if let Some(saved) = self.saved_positions.clone() {
                self.band.set_positions(saved)?;
            }
            self.cached_mode = None;
            self.threshold =
                backoff_threshold(commit_align, baseline, self.trigger_factor, self.force_tol);
        } else if helped {
            self.write_ci(commit_x)?;
            if stop == Stop::Converged && commit_align + 1e-15 >= self.angle_tol {
                self.cached_mode = Some(commit_mode);
            } else {
                self.cached_mode = None;
            }
            self.threshold = success_threshold(
                self.force_at_switch,
                commit_force,
                baseline,
                self.trigger_factor,
                self.force_tol,
            );
            self.reparametrize()?;
            self.band.forget_optimizer();
        } else {
            self.write_ci(commit_x.clone())?;
            self.cached_mode = None;
            self.threshold =
                backoff_threshold(commit_align, baseline, self.trigger_factor, self.force_tol);
            if self.ci_moved(&commit_x) {
                self.band.forget_optimizer();
            }
        }
        self.stability = 0;
        self.phase = OcinebPhase::Band;
        self.fell_back = matches!(stop, Stop::PositiveCurvature | Stop::Misaligned);
        self.clear_episode();
        Ok(())
    }

    fn reparametrize(&mut self) -> Result<(), SaddleError> {
        let mut positions = self.band.positions().to_owned();
        let anchor = Some(self.walked);
        if let Some(cell) = self.cell {
            reparametrize_equal_arc_in_cell(&mut positions, anchor, &cell);
        } else {
            reparametrize_equal_arc(&mut positions, anchor);
        }
        self.band.set_positions(positions)
    }

    fn ci_moved(&self, x: &Array1<f64>) -> bool {
        let Some(saved) = &self.saved_positions else {
            return false;
        };
        if self.walked >= saved.nrows() {
            return true;
        }
        let d = x - &saved.row(self.walked);
        d.dot(&d).sqrt() > 1e-12
    }

    fn consider_best(&mut self, x: Array1<f64>, mode: Array1<f64>, curvature: f64, force: f64) {
        let replace = match &self.best {
            None => true,
            Some(best) => curvature < best.curvature,
        };
        if replace {
            self.best = Some(Mark {
                x,
                mode,
                curvature,
                force,
            });
        }
    }

    fn write_ci(&mut self, x: Array1<f64>) -> Result<(), SaddleError> {
        let mut positions = self.band.positions().to_owned();
        if self.walked == 0 || self.walked + 1 >= positions.nrows() {
            return Err(SaddleError::Invalid(
                "climbing image left the interior".into(),
            ));
        }
        if x.len() != positions.ncols() {
            return Err(SaddleError::Shape("climbing image length changed".into()));
        }
        positions.row_mut(self.walked).assign(&x);
        self.band.set_positions(positions)
    }

    fn climbing_frame(&self) -> Result<Climb, SaddleError> {
        let ci = self
            .band
            .climbing_image()
            .ok_or_else(|| SaddleError::Invalid("hand-off has no climbing image".into()))?;
        let eval = self
            .band
            .evaluation()
            .ok_or_else(|| SaddleError::Invalid("hand-off has no band evaluation".into()))?;
        let positions = self.band.positions();
        let n = positions.nrows();
        if ci == 0 || ci + 1 >= n {
            return Err(SaddleError::Invalid("climbing image is an endpoint".into()));
        }
        let mut next = &positions.row(ci + 1) - &positions.row(ci);
        let mut prev = &positions.row(ci) - &positions.row(ci - 1);
        if let Some(cell) = &self.cell {
            cell.minimum_image(&mut next);
            cell.minimum_image(&mut prev);
        }
        let tangent = tangent::compute_tangent(
            self.tangent_kind,
            next.view(),
            prev.view(),
            eval.energies[ci],
            eval.energies[ci - 1],
            eval.energies[ci + 1],
        )?;
        Ok(Climb {
            ci,
            tangent,
            gradient: eval.gradients.row(ci).to_owned(),
            position: positions.row(ci).to_owned(),
        })
    }

    fn update_stability(&mut self, ci: i64) {
        if ci == self.previous_ci {
            self.stability += 1;
        } else {
            self.stability = 0;
            self.previous_ci = ci;
            self.cached_mode = None;
        }
    }

    fn clear_episode(&mut self) {
        self.minmode = None;
        self.tangent_at_switch = None;
        self.saved_positions = None;
        self.best = None;
        self.mmf_steps = 0;
    }

    #[allow(clippy::too_many_arguments)]
    fn make_report(
        &self,
        phase: OcinebPhase,
        status: OcinebStatus,
        max_force: f64,
        ci_index: Option<usize>,
        evaluations: usize,
        curvature: f64,
        alignment: f64,
        rotations: usize,
    ) -> OcinebReport {
        OcinebReport {
            phase,
            status,
            max_force,
            ci_index,
            iteration: self.host_steps,
            evaluations,
            curvature,
            alignment,
            rotations,
            fell_back: self.fell_back,
            threshold: self.threshold,
        }
    }
}

fn validate(config: &OcinebConfig) -> Result<(), SaddleError> {
    if config.band.climbing.is_none() {
        return Err(SaddleError::Invalid(
            "the hand-off needs a climbing image".into(),
        ));
    }
    if config.band.solid.is_some() {
        return Err(SaddleError::Invalid(
            "the hand-off does not carry the solid-state block".into(),
        ));
    }
    if !(config.trigger_factor.is_finite() && config.trigger_factor >= 0.0) {
        return Err(SaddleError::Invalid(
            "trigger factor must be finite and non-negative".into(),
        ));
    }
    if !(config.trigger_force.is_finite() && config.trigger_force >= 0.0) {
        return Err(SaddleError::Invalid(
            "trigger force must be finite and non-negative".into(),
        ));
    }
    if !(config.angle_tol.is_finite()
        && config.angle_tol + 1e-4 >= MIN_ALIGNMENT
        && config.angle_tol <= 1.0)
    {
        return Err(SaddleError::Invalid(
            "alignment tolerance must lie between 1/sqrt(2) and 1".into(),
        ));
    }
    if config.stability_count < 0 {
        return Err(SaddleError::Invalid(
            "stability count must be non-negative".into(),
        ));
    }
    if config.max_mmf_steps == 0 {
        return Err(SaddleError::Invalid(
            "minimum-mode step budget must be positive".into(),
        ));
    }
    if !(config.band.force_tol.is_finite() && config.band.force_tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "force tolerance must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn threshold_follows_the_relative_rule() {
        let t = initial_threshold(2.0, 0.31, 0.5, 0.01);
        assert_abs_diff_eq!(t, 0.62, epsilon = 1e-12);
        let absolute = initial_threshold(2.0, 0.0, 0.5, 0.01);
        assert_abs_diff_eq!(absolute, 0.5, epsilon = 1e-12);
        let floor = initial_threshold(0.01, 0.31, 0.0, 0.01);
        assert_abs_diff_eq!(floor, 0.02, epsilon = 1e-12);
    }

    #[test]
    fn success_and_backoff_thresholds() {
        let success = success_threshold(0.8, 0.4, 2.0, 0.31, 0.01);
        assert_abs_diff_eq!(success, 0.28, epsilon = 1e-12);
        let capped = success_threshold(0.2, 0.19, 1.0, 0.31, 0.01);
        assert!(capped <= initial_threshold(1.0, 0.31, 0.0, 0.01) + 1e-12);
        assert_abs_diff_eq!(
            backoff_threshold(0.0, 2.0, 0.31, 0.01),
            0.31,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            backoff_threshold(1.0, 2.0, 0.31, 0.01),
            0.62,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            backoff_threshold(0.0, 0.0, 0.0, 0.01),
            0.02,
            epsilon = 1e-12
        );
    }

    #[test]
    fn trigger_requires_a_stable_interior_image() {
        assert!(!should_trigger(true, Some(2), 7, 5, 5, 0.2, 0.01, 0.5));
        assert!(should_trigger(true, Some(2), 7, 6, 5, 0.2, 0.01, 0.5));
        assert!(!should_trigger(false, Some(2), 7, 6, 5, 0.2, 0.01, 0.5));
        assert!(!should_trigger(true, None, 7, 6, 5, 0.2, 0.01, 0.5));
        assert!(!should_trigger(true, Some(0), 7, 6, 5, 0.2, 0.01, 0.5));
        assert!(!should_trigger(true, Some(8), 7, 6, 5, 0.2, 0.01, 0.5));
        assert!(!should_trigger(true, Some(2), 7, 6, 5, 0.01, 0.01, 0.5));
        assert!(!should_trigger(true, Some(2), 7, 6, 5, 0.5, 0.01, 0.5));
    }

    #[test]
    fn walk_help_ignores_a_positive_curvature() {
        assert!(walk_helped(0.4, 0.8, false));
        assert!(!walk_helped(0.4, 0.8, true));
        assert!(!walk_helped(0.9, 0.8, false));
    }

    #[test]
    fn alignment_is_the_absolute_overlap() {
        let mode = array![0.0, -1.0, 0.0];
        let tangent = array![0.0, 1.0, 0.0];
        assert_abs_diff_eq!(
            mode_alignment(mode.view(), tangent.view()),
            1.0,
            epsilon = 1e-12
        );
        let tangent = array![1.0, 0.0, 0.0];
        assert_abs_diff_eq!(
            mode_alignment(mode.view(), tangent.view()),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn angle_below_the_cone_is_rejected() {
        let mut config = OcinebConfig::default();
        config.angle_tol = 0.5;
        assert!(validate(&config).is_err());
        config.angle_tol = 0.7071;
        assert!(validate(&config).is_ok());
        config.band.climbing = None;
        assert!(validate(&config).is_err());
    }
}
