//! Off-path climbing image: the band hands its climbing image to a
//! minimum-mode climb.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. 14 (2026),
//! doi:10.3389/fchem.2026.1807063. [`OcinebSession`] is a
//! [`BandSession`] whose [`crate::band::MinModeRefine`] carries the
//! published hand-off controls, and the refinement is the band's own:
//! one trigger and re-arm rule (the functions below) and one P-RFO
//! climb. The host calls [`OcinebSession::step`]. Each call is one band
//! step, one alignment, or one climb step, and the report names that
//! phase.
//!
//! The switch waits until the climbing-image index has stayed fixed for
//! more than `stability_count` iterations and the climbing image's
//! largest per-atom force is under the threshold. The threshold starts
//! at `trigger_factor` times the largest per-atom force of the initial
//! path, and it is at least twice the force tolerance. Alignment
//! estimates the lowest mode at the climbing image, seeded with the
//! tangent. The climb starts only when that curvature is negative and
//! the absolute overlap with the tangent is at least `angle_tol`
//! (`1/sqrt(2)` or above; the published value is 0.85).
//!
//! The climb returns to the band when the mode leaves that cone, the
//! curvature turns positive, or the step budget is spent, and the band
//! converges when the climbing image meets the force tolerance. A
//! positive curvature before an aligned point puts the climbing image
//! back where the hand-off started and clears the cached mode. A walk
//! that does not lower the force raises the threshold by the linear
//! penalty `(1 + α) / 2`. A walk that does lower it sets the threshold
//! to `F_new (1/2 + 2/5 F_new / F_CI)` and keeps the mode.

use ndarray::{Array1, Array2, ArrayView1};

use crate::band::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, MinModeRefine, RefinePhase,
};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{MinModeConfig, PointSurface};

/// Published floor on the tangent overlap. Below it the force inversion
/// points away from the saddle of the path.
pub const MIN_ALIGNMENT: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Which solver the last step ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OcinebPhase {
    /// One climbing-image band step.
    Band = 0,
    /// Lowest-mode estimate at the climbing image. No translation.
    Align = 1,
    /// One P-RFO climb step of the climbing image.
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
    /// The band. Its `min_mode` is replaced by these hand-off controls.
    pub band: BandConfig,
    /// The mode estimate: kind, finite difference, rotation and Krylov
    /// controls. The climb is the band's P-RFO step, so `method`,
    /// `force_tol` and `max_move` are not read.
    pub minmode: MinModeConfig,
    /// Relative trigger on the largest per-atom force,
    /// `ci_mmf_after_rel`. Published value 0.31.
    pub trigger_factor: f64,
    /// Absolute trigger, `ci_mmf_after`. Used when `trigger_factor` is 0.
    pub trigger_force: f64,
    /// Minimum `|mode · tangent|`. Published value 0.85.
    pub angle_tol: f64,
    /// Stable climbing-image iterations required before a hand-off.
    /// The counter must exceed this value. Published value 5.
    pub stability_count: i64,
    /// Climb steps per hand-off. Published value 1000.
    pub max_mmf_steps: usize,
    /// Also restore the climbing image when a walk does not lower the force.
    pub restore_unhelpful: bool,
}

impl Default for OcinebConfig {
    fn default() -> Self {
        // The hand-off waits on a relative band force, the eOn
        // ci_after_rel rule. BandConfig::default climbs under an
        // absolute force of 0.5, which is a different arming point.
        let band = BandConfig {
            climbing: Some(CiConfig::after_rel(0.5)),
            ..BandConfig::default()
        };
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
    /// The climb ended on a positive curvature or a mode that left the
    /// tangent cone, and the band has the next step.
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

/// Hand-off while the climbing image is real and stable and its force
/// is under the threshold. A climbing image that already meets the
/// force tolerance is handed off too: the alignment step confirms the
/// negative curvature, and the band converges there.
pub fn should_trigger(
    ci_active: bool,
    ci_index: Option<usize>,
    n_interior: usize,
    stability: i64,
    stability_count: i64,
    conv_force: f64,
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
    conv_force < threshold
}

impl OcinebConfig {
    /// The band's refinement controls for this hand-off.
    pub fn refine(&self) -> MinModeRefine {
        MinModeRefine {
            trigger_factor: self.trigger_factor,
            trigger_force: self.trigger_force,
            alignment_tol: self.angle_tol,
            stable_steps: usize::try_from(self.stability_count).unwrap_or(0) + 1,
            max_steps: self.max_mmf_steps,
            restore_unhelpful: self.restore_unhelpful,
            kind: self.minmode.kind,
            krylov_dim: self.minmode.krylov_dim,
            dr: self.minmode.dr,
            difference: self.minmode.difference,
            rotation_tol: self.minmode.rotation_tol,
            max_rotations: self.minmode.max_rotations,
            ..MinModeRefine::default()
        }
    }
}

/// The band surface, with the climbing image's evaluations sent to the
/// host's point surface.
struct Handoff<'a, S>(&'a S);

impl<S: BandSurface + PointSurface> BandSurface for Handoff<'_, S> {
    fn eval(
        &self,
        positions: ndarray::ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        BandSurface::eval(self.0, positions, energies, gradients)
    }

    fn eval_image(
        &self,
        _image: usize,
        position: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        PointSurface::eval(self.0, position)
    }
}

/// Host-stepped CI-NEB to minimum-mode hand-off.
pub struct OcinebSession {
    band: BandSession,
}

impl OcinebSession {
    pub fn new(config: OcinebConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        validate(&config)?;
        if !(config.minmode.dr.is_finite() && config.minmode.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        let refine = config.refine();
        let band = BandConfig {
            min_mode: Some(refine),
            ..config.band
        };
        Ok(Self {
            band: BandSession::new(band, initial)?,
        })
    }

    /// What the next step runs.
    pub fn phase(&self) -> OcinebPhase {
        match self.band.refine_phase() {
            RefinePhase::Band => OcinebPhase::Band,
            RefinePhase::Align => OcinebPhase::Align,
            RefinePhase::Climb => OcinebPhase::MinMode,
        }
    }

    pub fn threshold(&self) -> Option<f64> {
        self.band.refine_threshold()
    }

    pub fn stability(&self) -> i64 {
        self.band.refine_stability() as i64
    }

    pub fn positions(&self) -> ndarray::ArrayView2<'_, f64> {
        self.band.positions()
    }

    pub fn climbing_image(&self) -> Option<usize> {
        self.band.climbing_image()
    }

    /// The band evaluation at the current positions, if one is cached.
    pub fn evaluation(&self) -> Option<crate::band::BandEvaluation> {
        self.band.evaluation()
    }

    /// A new gate restarts the band and the hand-off state.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        self.band.set_force_gate(gate);
    }

    pub fn set_highs(&mut self, enabled: bool) {
        self.band.set_highs(enabled);
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
    }

    /// One band step, one alignment, or one climb step.
    pub fn step<S: BandSurface + PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<OcinebReport, SaddleError> {
        let report = self.band.step(&Handoff(surface))?;
        let last = self.band.refine_last();
        let phase = match last {
            None => OcinebPhase::Band,
            Some(step) if step.align => OcinebPhase::Align,
            Some(_) => OcinebPhase::MinMode,
        };
        Ok(OcinebReport {
            phase,
            status: if report.status == BandStatus::Converged {
                OcinebStatus::Converged
            } else {
                OcinebStatus::Running
            },
            max_force: report.max_force,
            ci_index: report.ci_index,
            iteration: report.iteration,
            evaluations: report.surface_rows,
            curvature: report.curvature.unwrap_or(f64::NAN),
            alignment: last.map_or(f64::NAN, |s| s.alignment),
            rotations: last.map_or(0, |s| s.rotations),
            fell_back: last.is_some_and(|s| s.fell_back),
            threshold: self.band.refine_threshold().unwrap_or(f64::NAN),
        })
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
        assert!(!should_trigger(true, Some(2), 7, 5, 5, 0.2, 0.5));
        assert!(should_trigger(true, Some(2), 7, 6, 5, 0.2, 0.5));
        assert!(!should_trigger(false, Some(2), 7, 6, 5, 0.2, 0.5));
        assert!(!should_trigger(true, None, 7, 6, 5, 0.2, 0.5));
        assert!(!should_trigger(true, Some(0), 7, 6, 5, 0.2, 0.5));
        assert!(!should_trigger(true, Some(8), 7, 6, 5, 0.2, 0.5));
        assert!(should_trigger(true, Some(2), 7, 6, 5, 0.001, 0.5));
        assert!(!should_trigger(true, Some(2), 7, 6, 5, 0.5, 0.5));
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
        let low = OcinebConfig {
            angle_tol: 0.5,
            ..OcinebConfig::default()
        };
        assert!(validate(&low).is_err());
        let edge = OcinebConfig {
            angle_tol: MIN_ALIGNMENT,
            ..OcinebConfig::default()
        };
        assert!(validate(&edge).is_ok());
        let no_climb = OcinebConfig {
            band: crate::band::BandConfig {
                climbing: None,
                ..edge.band.clone()
            },
            ..edge
        };
        assert!(validate(&no_climb).is_err());
    }

    #[test]
    fn default_climbing_is_relative_and_a_solid_block_is_refused() {
        let config = OcinebConfig::default();
        let climbing = config.band.climbing.expect("climbing image");
        assert_eq!(climbing.trigger_factor, 0.5);
        assert_eq!(climbing.trigger_force, 0.0);
        assert!(validate(&config).is_ok());
        let mut solid = config.clone();
        solid.band.solid = Some(crate::band::SolidState::new(Vec::new()));
        match validate(&solid) {
            Err(crate::error::SaddleError::Invalid(msg)) => {
                assert!(msg.contains("solid-state"), "{msg}");
            }
            other => panic!("expected a refused solid block, got {other:?}"),
        }
        let built = OcinebSession::new(
            solid,
            array![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        );
        match built {
            Err(crate::error::SaddleError::Invalid(msg)) => {
                assert!(msg.contains("solid-state"), "{msg}");
            }
            Ok(_) => panic!("expected a refused solid block"),
            Err(err) => panic!("expected a refused solid block, got {err}"),
        }
    }
}
