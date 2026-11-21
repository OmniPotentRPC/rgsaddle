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
//!
//! The converged climbing image is certified by the band's
//! [`BandConfig::saddle_check`], on by default: the report carries its
//! saddle index, and an escape set in that check climbs again off a
//! saddle of higher index.
//!
//! Units and shapes: positions are one row per image, each row a flat `3N`
//! Cartesian vector in Angstrom (atom-major), endpoints included; forces
//! and thresholds are in eV/Angstrom and curvatures in eV/Angstrom^2.
//! Errors are [`SaddleError::Invalid`] for settings out of range and
//! [`SaddleError::Shape`] for a malformed path, plus whatever the surface
//! returns.

use ndarray::{Array1, Array2, ArrayView1};

use crate::band::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, MinModeRefine, RefinePhase,
};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{MinModeConfig, PointSurface};
use crate::saddle_index::{SaddleIndex, SaddleIndexCheck};

/// Published floor on the tangent overlap, `1/sqrt(2)`: [`OcinebConfig::angle_tol`]
/// may not be lower. Below it the force inversion points away from the saddle of
/// the path.
pub const MIN_ALIGNMENT: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Which solver the last step ran. The C values are the discriminants.
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

/// Whether the hand-off band has converged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OcinebStatus {
    /// The climbing image has not met the force tolerance yet. C value 0.
    Running = 0,
    /// The climbing image meets the force tolerance. C value 1.
    Converged = 1,
}

/// Band, minimum-mode, and the hand-off parameters.
///
/// [`OcinebSession::new`] rejects a `band` without a climbing image or with a
/// solid-state block, a negative or non-finite trigger, an `angle_tol` outside
/// `1/sqrt(2)` to 1 (a `1e-4` slack on the lower end), a negative
/// `stability_count`, a zero `max_mmf_steps` and a non-positive `minmode.dr`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OcinebConfig {
    /// The band. Its `min_mode` is replaced by these hand-off controls.
    pub band: BandConfig,
    /// The mode estimate: kind, finite difference, rotation and Krylov
    /// controls. The climb is the band's P-RFO step, so `method`,
    /// `force_tol` and `max_move` are not read.
    pub minmode: MinModeConfig,
    /// Relative trigger on the largest per-atom force, `ci_mmf_after_rel`, as a
    /// fraction of the largest per-atom force of the initial path. Published value 0.31
    /// (the default); must be finite and non-negative.
    pub trigger_factor: f64,
    /// Absolute trigger, `ci_mmf_after`, in eV/Angstrom. Used when `trigger_factor`
    /// is 0; finite and non-negative. Default 0.
    pub trigger_force: f64,
    /// Minimum `|mode . tangent|`, dimensionless, between `1/sqrt(2)` and 1.
    /// Published value 0.85 (the default).
    pub angle_tol: f64,
    /// Stable climbing-image iterations required before a hand-off. The counter
    /// must exceed this value; non-negative. Published value 5 (the default).
    pub stability_count: i64,
    /// Climb steps per hand-off; at least 1. Published value 1000 (the default).
    pub max_mmf_steps: usize,
    /// Also restore the climbing image when a walk does not lower the force.
    /// Default false.
    pub restore_unhelpful: bool,
}

with_setters!(OcinebConfig {
    with_band => band: val BandConfig,
    with_minmode => minmode: val MinModeConfig,
    with_trigger_factor => trigger_factor: val f64,
    with_trigger_force => trigger_force: val f64,
    with_angle_tol => angle_tol: val f64,
    with_stability_count => stability_count: val i64,
    with_max_mmf_steps => max_mmf_steps: val usize,
    with_restore_unhelpful => restore_unhelpful: val bool,
});

impl Default for OcinebConfig {
    fn default() -> Self {
        // The hand-off waits on a relative band force, the eOn
        // ci_after_rel rule. BandConfig::default climbs under an
        // absolute force of 0.5, which is a different arming point.
        let band = BandConfig::default().with_climbing(CiConfig::after_rel(0.5));
        let minmode = MinModeConfig::default()
            .with_force_tol(band.force_tol)
            .with_max_move(band.max_move);
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

/// One host step: what ran, and where the band stands.
#[derive(Clone, Debug)]
pub struct OcinebReport {
    /// The solver this step ran.
    pub phase: OcinebPhase,
    /// Whether this step is still running or has converged.
    pub status: OcinebStatus,
    /// Largest absolute force component of the band, in eV/Angstrom; the gate is
    /// set by [`OcinebSession::set_force_gate`].
    pub max_force: f64,
    /// Index of the climbing image, or none when no image climbs.
    pub ci_index: Option<usize>,
    /// Number of steps taken by the session so far.
    pub iteration: usize,
    /// Surface evaluations performed by this call.
    pub evaluations: usize,
    /// Curvature along the mode, eV/Angstrom^2. NaN when none was estimated.
    pub curvature: f64,
    /// `|mode . tangent|` of the last alignment, dimensionless; NaN before any.
    pub alignment: f64,
    /// Mode rotations the last alignment or climb step spent; 0 when none ran.
    pub rotations: usize,
    /// The climb ended on a positive curvature or a mode that left the
    /// tangent cone, and the band has the next step.
    pub fell_back: bool,
    /// Current hand-off force threshold, in eV/Angstrom; NaN when none is armed.
    pub threshold: f64,
    /// Saddle index of the converged climbing image; see
    /// [`crate::BandReport::saddle_index`].
    pub saddle_index: Option<SaddleIndex>,
}

/// `baseline` is the largest per-atom force of the initial path, `factor` the relative
/// trigger, `absolute` the absolute trigger and `force_tol` the band tolerance, all
/// forces in eV/Angstrom. The result is in eV/Angstrom.
pub fn initial_threshold(baseline: f64, factor: f64, absolute: f64, force_tol: f64) -> f64 {
    let relative = baseline * factor;
    let abs_term = if factor == 0.0 { absolute } else { 0.0 };
    relative.max(abs_term).max(2.0 * force_tol)
}

/// `F_new (1/2 + 2/5 F_new / F_CI)`, capped at the initial threshold. `conv_force` is
/// `F_CI`, the climbing-image force before the walk, and `new_force` is `F_new`,
/// the force after it; both in eV/Angstrom. A `conv_force` of 0 or less gives 0.
/// The other arguments are those of [`initial_threshold`] (with a zero absolute
/// trigger).
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

/// Linear penalty `(1/2 + 1/2 alpha)` on the initial relative threshold, floored at
/// twice the force tolerance. `alignment` is `alpha`, restricted to 0 to 1;
/// forces are in eV/Angstrom.
pub fn backoff_threshold(alignment: f64, baseline: f64, factor: f64, force_tol: f64) -> f64 {
    let alpha = alignment.clamp(0.0, 1.0);
    let penalty = 0.5 + 0.5 * alpha;
    (baseline * factor * penalty).max(2.0 * force_tol)
}

/// The walked image lowered the band force, and the curvature was not positive.
/// Forces are in eV/Angstrom; the test is `walked_force < conv_force`.
pub fn walk_helped(walked_force: f64, conv_force: f64, positive_curvature: bool) -> bool {
    walked_force < conv_force && !positive_curvature
}

/// `|u . t| / (|u| |t|)`, the absolute cosine between `mode` and `tangent`, in
/// 0 to 1. A vector of norm `1e-14` or less scores 0.
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
///
/// `stability` must exceed `stability_count`, `ci_index` must name an interior image
/// (1 to `n_interior`), and `conv_force` must be below `threshold` (eV/Angstrom).
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
    /// The band's refinement controls for this hand-off, built from the trigger,
    /// alignment, stability and mode settings of the config.
    pub fn refine(&self) -> MinModeRefine {
        MinModeRefine::default()
            .with_trigger_factor(self.trigger_factor)
            .with_trigger_force(self.trigger_force)
            .with_alignment_tol(self.angle_tol)
            .with_climb_alignment(true)
            .with_retry_factor(None)
            .with_stable_steps(usize::try_from(self.stability_count).unwrap_or(0) + 1)
            .with_max_steps(self.max_mmf_steps)
            .with_restore_unhelpful(self.restore_unhelpful)
            .with_kind(self.minmode.kind)
            .with_krylov_dim(self.minmode.krylov_dim)
            .with_dr(self.minmode.dr)
            .with_difference(self.minmode.difference)
            .with_rotation_tol(self.minmode.rotation_tol)
            .with_rotation_angle_tol(self.minmode.rotation_angle_tol)
            .with_max_rotations(self.minmode.max_rotations)
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
///
/// # Examples
///
/// OCI-NEB between minima B and C of the Müller-Brown surface, one atom
/// in the plane z = 0. The surface serves the band through
/// [`BandSurface`] and the climbing image's mode estimate through
/// [`PointSurface`].
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
/// use rgsaddle::{
///     BandSurface, OcinebConfig, OcinebSession, OcinebStatus, PointSurface, SaddleError,
///     linear_path,
/// };
///
/// # const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
/// # const XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
/// # const XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
/// # const YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
/// # const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
/// # const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
/// # fn muller_brown(x: f64, y: f64) -> (f64, f64, f64) {
/// #     let (mut e, mut gx, mut gy) = (0.0, 0.0, 0.0);
/// #     for i in 0..4 {
/// #         let (dx, dy) = (x - X0[i], y - Y0[i]);
/// #         let t = A[i] * (XX[i] * dx * dx + XY[i] * dx * dy + YY[i] * dy * dy).exp();
/// #         e += t;
/// #         gx += t * (2.0 * XX[i] * dx + XY[i] * dy);
/// #         gy += t * (XY[i] * dx + 2.0 * YY[i] * dy);
/// #     }
/// #     (e, gx, gy)
/// # }
/// struct MullerBrown;
///
/// impl PointSurface for MullerBrown {
///     fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (e, gx, gy) = muller_brown(r[0], r[1]);
///         Ok((e, array![gx, gy, 0.0]))
///     }
/// }
///
/// impl BandSurface for MullerBrown {
///     fn eval(
///         &self,
///         positions: ArrayView2<f64>,
///         energies: &mut Array1<f64>,
///         gradients: &mut Array2<f64>,
///     ) -> Result<(), SaddleError> {
///         for (i, r) in positions.outer_iter().enumerate() {
///             let (e, g) = PointSurface::eval(self, r)?;
///             energies[i] = e;
///             gradients.row_mut(i).assign(&g);
///         }
///         Ok(())
///     }
/// }
///
/// let b = array![-0.050011, 0.466694, 0.0];
/// let c = array![0.623499, 0.028038, 0.0];
/// let config = OcinebConfig::default();
/// let band = config.band.clone().with_max_move(0.05).with_force_tol(1e-2);
/// let config = config.with_band(band);
/// let mut session = OcinebSession::new(config, linear_path(b.view(), c.view(), 7, None)?)?;
/// let report = session.run(&MullerBrown, 2000)?;
///
/// assert_eq!(report.status, OcinebStatus::Converged);
/// let saddle = session.positions().row(report.ci_index.unwrap()).to_owned();
/// assert!((saddle[0] - 0.212).abs() < 0.01 && (saddle[1] - 0.293).abs() < 0.01);
/// # Ok::<(), SaddleError>(())
/// ```
pub struct OcinebSession {
    band: BandSession,
}

impl OcinebSession {
    /// Open the band on `initial`, one image per row, endpoints included.
    ///
    /// `initial` is the starting path, one row per image with the endpoints included,
    /// each row a flat `3N` vector in Angstrom.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] for settings listed under [`OcinebConfig`], and the
    /// errors of [`BandSession::new`] for a malformed path.
    pub fn new(config: OcinebConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        validate(&config)?;
        if !(config.minmode.dr.is_finite() && config.minmode.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        let refine = config.refine();
        let band = config.band.with_min_mode(refine);
        Ok(Self {
            band: BandSession::new(band, initial)?,
        })
    }

    /// What the next step runs: a band step, an alignment or a climb step.
    pub fn phase(&self) -> OcinebPhase {
        match self.band.refine_phase() {
            RefinePhase::Band => OcinebPhase::Band,
            RefinePhase::Align => OcinebPhase::Align,
            RefinePhase::Climb => OcinebPhase::MinMode,
        }
    }

    /// Current hand-off force threshold in eV/Angstrom, or `None` before it is armed.
    pub fn threshold(&self) -> Option<f64> {
        self.band.refine_threshold()
    }

    /// Consecutive iterations the climbing-image index has stayed fixed.
    pub fn stability(&self) -> i64 {
        self.band.refine_stability() as i64
    }

    /// Cartesian coordinates in Angstrom: one row per image, endpoints included.
    pub fn positions(&self) -> ndarray::ArrayView2<'_, f64> {
        self.band.positions()
    }

    /// Row index of the climbing image, or `None` when no image climbs.
    pub fn climbing_image(&self) -> Option<usize> {
        self.band.climbing_image()
    }

    /// The band evaluation at the current positions, if one is cached.
    pub fn evaluation(&self) -> Option<crate::band::BandEvaluation> {
        self.band.evaluation()
    }

    /// See [`BandSession::saddle_index`].
    pub fn saddle_index(&self) -> Option<&SaddleIndex> {
        self.band.saddle_index()
    }

    /// See [`BandSession::saddle_escapes`].
    pub fn saddle_escapes(&self) -> usize {
        self.band.saddle_escapes()
    }

    /// See [`BandSession::saddle_evaluations`].
    pub fn saddle_evaluations(&self) -> usize {
        self.band.saddle_evaluations()
    }

    /// See [`BandSession::set_saddle_check`].
    ///
    /// `check` replaces the band's saddle-index check between steps; `None` turns it off.
    ///
    /// # Errors
    ///
    /// The errors of [`BandSession::set_saddle_check`]; the session is unchanged.
    pub fn set_saddle_check(&mut self, check: Option<SaddleIndexCheck>) -> Result<(), SaddleError> {
        self.band.set_saddle_check(check)
    }

    /// A new gate restarts the band and the hand-off state.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        self.band.set_force_gate(gate);
    }

    /// Select the optional HiGHS feasible-set step of the band; requires the
    /// `highs` feature. Disabled by default.
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
    ///
    /// The returned [`OcinebReport`] names the phase that ran.
    ///
    /// # Errors
    ///
    /// The errors of [`BandSession::step`] and of `surface`.
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
            saddle_index: report.saddle_index,
        })
    }

    /// Loop [`Self::step`] until convergence or `max_steps` steps.
    ///
    /// At least one step is taken, even for `max_steps` of 0. Returns the last report.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::step`].
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
        let low = OcinebConfig::default().with_angle_tol(0.5);
        assert!(validate(&low).is_err());
        let edge = OcinebConfig::default().with_angle_tol(MIN_ALIGNMENT);
        assert!(validate(&edge).is_ok());
        let band = edge.band.clone().with_climbing(None);
        let no_climb = edge.with_band(band);
        assert!(validate(&no_climb).is_err());
    }

    #[test]
    fn default_climbing_is_relative_and_a_solid_block_is_refused() {
        let config = OcinebConfig::default();
        let climbing = config.band.climbing.expect("climbing image");
        assert_eq!(climbing.trigger_factor, 0.5);
        assert_eq!(climbing.trigger_force, 0.0);
        assert!(validate(&config).is_ok());
        let band = config
            .band
            .clone()
            .with_solid(crate::band::SolidState::new(Vec::new()));
        let solid = config.clone().with_band(band);
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
