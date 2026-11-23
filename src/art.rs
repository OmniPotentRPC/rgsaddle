//! Activation-relaxation technique nouveau, one persisted phase at a time.
//!
//! Malek and Mousseau, Phys. Rev. E 62, 7723 (2000),
//! <https://doi.org/10.1103/PhysRevE.62.7723>, leave a minimum by pushing
//! along one direction and relaxing the energy in the perpendicular
//! hyperplane until the lowest curvature is negative, then follow that
//! eigenvector while the perpendicular force is minimized. The search
//! here keeps that sequence as a phase on the session:
//!
//! 1. [`ArtPhase::Push`] applies the host seed, masked, with no generator
//!    inside the library;
//! 2. [`ArtPhase::Perpendicular`] relaxes the hyperplane with
//!    [`crate::relax_to_tolerance`];
//! 3. [`ArtPhase::Lanczos`] reads the lowest curvature from
//!    [`crate::MinModeSession::estimate_mode`];
//! 4. [`ArtPhase::Climb`] takes one [`crate::MinModeSession::step`] and
//!    returns to the perpendicular relaxation;
//! 5. [`ArtPhase::Certify`] calls [`fn@crate::saddle_index`];
//! 6. [`ArtPhase::PushOver`] steps both ways along the unstable mode;
//! 7. [`ArtPhase::Quench`] minimizes both images with no constraint;
//! 8. [`ArtPhase::Report`] records whether the quenches are reversible
//!    and [`run_events`] marks a later copy of the same saddle.
//!
//! Every `PointSurface::eval` performed for the session is appended to
//! [`ArtSession::force_calls`]. The perpendicular block and the quenches
//! are [`crate::relax_to_tolerance`]. The curvature and the climb are a
//! Lanczos [`crate::MinModeSession`] with its ART-nouveau shortcut and
//! its own saddle check left off, so those numerics are not copied.
//! [`ArtConfig::saddle`] is the certificate.
//!
//! ```
//! use ndarray::{Array1, ArrayView1, array};
//! use rgsaddle::{ArtConfig, ArtSession, ArtStatus, MinModeConfig, MinModeKind, PointSurface, SaddleError};
//!
//! struct Well;
//! impl PointSurface for Well {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
//!         let (a, b, c) = (x[0], x[1], x[2]);
//!         let energy = 0.25 * (a * a - 1.0).powi(2) + 0.5 * (b * b + c * c);
//!         Ok((energy, array![a * a * a - a, b, c]))
//!     }
//! }
//!
//! let mode = MinModeConfig::default()
//!     .with_kind(MinModeKind::Lanczos)
//!     .with_saddle_check(None)
//!     .with_force_tol(0.02)
//!     .with_max_move(0.1)
//!     .with_krylov_dim(4)
//!     .with_max_rotations(6);
//! let config = ArtConfig::default()
//!     .with_mode(mode)
//!     .with_push_length(0.8)
//!     .with_max_pushes(3)
//!     .with_max_climbs(25)
//!     .with_push_over(0.25);
//! let mut session = ArtSession::new(
//!     config,
//!     array![-1.0, 0.0, 0.0],
//!     array![1.0, 0.1, 0.0],
//!     array![1.0, 1.0, 1.0],
//! )?;
//! let report = session.run(&Well, 80)?;
//! assert_eq!(report.status, ArtStatus::Connected);
//! assert_eq!(report.saddle_index, Some(1));
//! # Ok::<(), SaddleError>(())
//! ```

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayView1};

use rgmin::Method;

use crate::cell_log::M3;
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, PointSurface};
use crate::relax::{RelaxConfig, relax_to_tolerance};
use crate::saddle_index::{SaddleIndexCheck, saddle_index};
use crate::stencil::perp;

/// Largest cluster, in atoms, for which identical-atom permutations are compared.
/// Eight atoms is 40320 labelings.
const PERMUTATION_ATOM_CAP: usize = 8;

/// Cell passed to [`relax_to_tolerance`]. The ART surface is molecular:
/// [`PointSurface::eval_in_cell`] falls through to [`PointSurface::eval`].
const OPEN_CELL: M3 = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];

/// Where an [`ArtSession`] is waiting.
///
/// The value is stored on the session and is the phase `step` runs next.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtPhase {
    /// Displace along the masked seed, or again along that same direction.
    Push,
    /// Minimize the force perpendicular to the active direction.
    Perpendicular,
    /// Lowest Lanczos curvature at the current point.
    Lanczos,
    /// One lowest-mode translation.
    Climb,
    /// Index of the stationary point.
    Certify,
    /// Steps of [`ArtConfig::push_over`] along both signs of the mode.
    PushOver,
    /// Unconstrained minimization of both images.
    Quench,
    /// Reversibility of the pair of minima.
    Report,
    /// No further step. The status is the verdict.
    Finished,
}

/// Verdict of a finished event.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtStatus {
    /// A phase is still pending.
    Running,
    /// Index 1, both quenches downhill, one image is the reactant, and
    /// the other image is a different minimum.
    Connected,
    /// Both quenches landed in one basin. Not a completed exit.
    Reverted,
    /// Both images are downhill and distinct, and neither is the reactant.
    Irreversible,
    /// The certificate is not a converged index-1 saddle.
    NotIndexOne,
    /// A quench energy is not below the saddle.
    NotDownhill,
    /// The lowest curvature became non-negative during the climb.
    LostCurvature,
    /// The push budget, the climb budget, or [`ArtSession::run`]'s cap was spent.
    Exhausted,
}

/// Which leg the perpendicular relaxation and the Lanczos read belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    Opening,
    Climbing,
}

/// Tolerances and budgets of one ART nouveau event.
///
/// [`ArtConfig::default`] is in the host's units: pushes of one host length
/// and the published defaults of the nested blocks. It is not a potential
/// preset. [`ArtConfig::lj_art`] is the Lennard-Jones reduced unit preset,
/// and [`ArtConfig::for_units`] scales that preset. The estimator in
/// [`ArtConfig::mode`] must be Lanczos. `artn` and `saddle_check` on that
/// block are cleared: the phases below own the push, the relaxation, and
/// the certificate.
///
/// Non-exhaustive: start from [`ArtConfig::default`] and chain `with_*`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ArtConfig {
    /// Lanczos estimator and the climb translation.
    pub mode: MinModeConfig,
    /// Opening ends once the lowest curvature is strictly below this.
    pub eigen_threshold: f64,
    /// Length of the first displacement along the masked seed.
    pub push_length: f64,
    /// Length of each later opening push.
    pub push_step: f64,
    /// Opening pushes, including the first, before the event stops.
    pub max_pushes: usize,
    /// Climb translations before the event stops.
    pub max_climbs: usize,
    /// Hyperplane minimization after every push and every climb.
    pub perpendicular: RelaxConfig,
    /// Unconstrained minimization of each pushed-over image.
    pub quench: RelaxConfig,
    /// Length of each of the two steps off the certified saddle.
    pub push_over: f64,
    /// Index certificate. Escape is not read.
    pub saddle: SaddleIndexCheck,
    /// Structures at or under this distance are the same basin.
    pub basin_rmsd: f64,
    /// Saddles at or under this distance are one event.
    pub duplicate_rmsd: f64,
    /// Compare structures as identical atoms, up to permutation.
    pub permute_identical: bool,
}

with_setters!(ArtConfig {
    with_mode => mode: val MinModeConfig,
    with_eigen_threshold => eigen_threshold: val f64,
    with_push_length => push_length: val f64,
    with_push_step => push_step: val f64,
    with_max_pushes => max_pushes: val usize,
    with_max_climbs => max_climbs: val usize,
    with_perpendicular => perpendicular: val RelaxConfig,
    with_quench => quench: val RelaxConfig,
    with_push_over => push_over: val f64,
    with_saddle => saddle: val SaddleIndexCheck,
    with_basin_rmsd => basin_rmsd: val f64,
    with_duplicate_rmsd => duplicate_rmsd: val f64,
    with_permute_identical => permute_identical: val bool,
});

impl Default for ArtConfig {
    /// One host length for each push, with the nested blocks at their own
    /// defaults and Lanczos selected. The perpendicular force tolerance
    /// matches the mode force tolerance. These values do not assume a
    /// potential or a unit system. [`Self::lj_art`] holds the Lennard-Jones
    /// reduced unit numbers.
    fn default() -> Self {
        let mode = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_saddle_check(None);
        let perpendicular = RelaxConfig::default().with_force_tol(mode.force_tol);
        Self {
            mode,
            eigen_threshold: -SaddleIndexCheck::default().negative_tol,
            push_length: 1.0,
            push_step: 1.0,
            max_pushes: 12,
            max_climbs: 40,
            perpendicular,
            quench: RelaxConfig::default(),
            push_over: 1.0,
            saddle: SaddleIndexCheck::default(),
            basin_rmsd: 0.08,
            duplicate_rmsd: 0.08,
            permute_identical: false,
        }
    }
}

impl ArtConfig {
    /// ART parameters in Lennard-Jones reduced units (`epsilon = sigma = 1`).
    ///
    /// These numbers are not an eV/Angstrom preset. Use [`Self::for_units`]
    /// when the potential's energy or length scale is not one. The saddle
    /// certificate uses central Hessian differences: under unit conversion,
    /// both its displacement and curvature tolerance scale exactly, while the
    /// central truncation error remains below that scaled tolerance.
    #[must_use]
    pub fn lj_art() -> Self {
        let mode = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_saddle_check(None)
            .with_krylov_dim(12)
            .with_max_rotations(12)
            .with_force_tol(0.01)
            .with_max_move(0.2);
        let perpendicular = RelaxConfig::default()
            .with_force_tol(0.01)
            .with_max_steps(800)
            .with_max_atom_move(0.1)
            .with_initial_step(0.05);
        let quench = RelaxConfig::default()
            .with_force_tol(0.001)
            .with_max_steps(4_000)
            .with_max_atom_move(0.1)
            .with_initial_step(0.05);
        Self {
            mode,
            eigen_threshold: -0.01,
            push_length: 0.4,
            push_step: 0.3,
            max_pushes: 12,
            max_climbs: 40,
            perpendicular,
            quench,
            push_over: 0.15,
            saddle: SaddleIndexCheck::default().with_difference(FiniteDifference::Central),
            basin_rmsd: 0.08,
            duplicate_rmsd: 0.08,
            permute_identical: false,
        }
    }

    /// Scale the Lennard-Jones ART preset to the host's potential units.
    ///
    /// `energy_scale` is the host energy represented by one reduced energy
    /// unit and `length_scale` is the host length represented by one reduced
    /// length unit. Each field scales by its own unit. Lengths scale by
    /// `length_scale`, forces by `energy_scale / length_scale`, and
    /// curvatures by `energy_scale / length_scale^2`. The FIRE time step
    /// [`MinModeConfig::time_step`] scales as length over the square root of
    /// energy, because the opening move from rest is the force times its
    /// square. [`MinModeConfig::rotation_angle_tol`] is an angle in radians
    /// and [`MinModeConfig::rotation_tol`] is taken as a dimensionless
    /// rotation stop, so both keep their values.
    ///
    /// [`RelaxConfig::initial_step`] follows the relaxation method. FIRE
    /// and quick-min read it as a time step, length over the square root
    /// of energy. Steepest descent, L-BFGS, nonlinear CG, DIIS and
    /// Barzilai-Borwein multiply the force by it in their first step, so it
    /// is length squared over energy. Adam moves along a direction of unit
    /// atom length, so it is a length. Any other relaxation method returns
    /// [`SaddleError::Invalid`]. The preset does not read
    /// [`SaddleIndexCheck::escape`], so that push is left unset.
    pub fn for_units(energy_scale: f64, length_scale: f64) -> Result<Self, SaddleError> {
        let config = Self::lj_art().scaled(energy_scale, length_scale)?;
        config.validate()?;
        Ok(config)
    }

    /// This configuration with every dimensioned field converted by the
    /// unit factors of [`Self::for_units`].
    fn scaled(mut self, energy_scale: f64, length_scale: f64) -> Result<Self, SaddleError> {
        if !(energy_scale.is_finite()
            && energy_scale > 0.0
            && length_scale.is_finite()
            && length_scale > 0.0)
        {
            return Err(SaddleError::Invalid(
                "ART energy and length scales must be finite and positive".into(),
            ));
        }
        let force_scale = energy_scale / length_scale;
        let curvature_scale = force_scale / length_scale;
        let time_scale = length_scale / energy_scale.sqrt();
        self.mode.dr *= length_scale;
        self.mode.force_tol *= force_scale;
        self.mode.max_move *= length_scale;
        self.mode.time_step *= time_scale;
        self.eigen_threshold *= curvature_scale;
        self.push_length *= length_scale;
        self.push_step *= length_scale;
        for relax in [&mut self.perpendicular, &mut self.quench] {
            relax.force_tol *= force_scale;
            relax.max_atom_move *= length_scale;
            relax.initial_step = scaled_initial_step(
                &relax.method,
                relax.initial_step,
                energy_scale,
                length_scale,
            )?;
        }
        self.push_over *= length_scale;
        self.saddle.residual_tol *= curvature_scale;
        self.saddle.negative_tol *= curvature_scale;
        self.saddle.dr *= length_scale;
        self.basin_rmsd *= length_scale;
        self.duplicate_rmsd *= length_scale;
        Ok(self)
    }

    /// Reject a configuration the session cannot run.
    ///
    /// Returns [`SaddleError::Invalid`] when a length is not finite and
    /// positive, a budget is zero, the nested relaxation or saddle check is
    /// refused, or the mode force tolerance is not finite and positive. A
    /// perpendicular force tolerance looser than that mode force tolerance
    /// is the same error and the message names both values. [`ArtSession::new`]
    /// and [`run_events`] call this before the first evaluation.
    pub fn validate(&self) -> Result<(), SaddleError> {
        if self.mode.kind != MinModeKind::Lanczos {
            return Err(SaddleError::Invalid(
                "ART monitors the lowest curvature with Lanczos".into(),
            ));
        }
        if !(self.eigen_threshold.is_finite() && self.eigen_threshold <= 0.0) {
            return Err(SaddleError::Invalid(
                "ART eigenvalue threshold must be finite and non-positive".into(),
            ));
        }
        for (name, value) in [
            ("push length", self.push_length),
            ("push step", self.push_step),
            ("push over the saddle", self.push_over),
            ("basin distance", self.basin_rmsd),
            ("duplicate distance", self.duplicate_rmsd),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(SaddleError::Invalid(format!(
                    "ART {name} must be finite and positive"
                )));
            }
        }
        if self.max_pushes == 0 || self.max_climbs == 0 {
            return Err(SaddleError::Invalid(
                "ART push and climb budgets must be at least one".into(),
            ));
        }
        self.perpendicular.check()?;
        self.quench.check()?;
        if !(self.mode.force_tol.is_finite() && self.mode.force_tol > 0.0) {
            return Err(SaddleError::Invalid(
                "ART mode force tolerance must be finite and positive".into(),
            ));
        }
        if self.perpendicular.force_tol > self.mode.force_tol {
            return Err(SaddleError::Invalid(format!(
                "ART perpendicular_force_tol={:e} energy/length is looser than \
                 mode_force_tol={:e} energy/length",
                self.perpendicular.force_tol, self.mode.force_tol
            )));
        }
        self.saddle.validate()?;
        Ok(())
    }

    fn estimator(&self) -> MinModeConfig {
        let mut mode = self.mode.clone();
        mode.artn = None;
        mode.saddle_check = None;
        mode
    }
}

/// Scale one relaxation opening step into the host's units.
///
/// FIRE's and quick-min's first move from rest is `F dt^2` with unit mass,
/// so `dt` has units of length over the square root of energy. Steepest
/// descent, empty-memory L-BFGS, unpreconditioned nonlinear CG, DIIS and
/// Barzilai-Borwein open on `-step * g`, so the step is length squared over
/// energy. Adam opens along a direction scaled to unit atom length, so its
/// step is a length. BFGS opens every line search on its inverse Hessian,
/// which starts as the identity and then carries the secant units, so no
/// one factor converts it. That method and every other one is refused: its
/// step is not left at the reduced unit value.
fn scaled_initial_step(
    method: &Method,
    initial_step: f64,
    energy_scale: f64,
    length_scale: f64,
) -> Result<f64, SaddleError> {
    let factor = match method {
        Method::Fire { .. } | Method::QuickMin => length_scale / energy_scale.sqrt(),
        Method::Steepest
        | Method::Lbfgs { .. }
        | Method::Nlcg { .. }
        | Method::Diis { .. }
        | Method::Bb => length_scale * length_scale / energy_scale,
        Method::Adam { .. } => length_scale,
        _ => {
            return Err(SaddleError::Invalid(format!(
                "ART cannot scale the initial step of relaxation method {}",
                method.as_str()
            )));
        }
    };
    Ok(initial_step * factor)
}

/// One force evaluation the session requested.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtForceCall {
    /// Phase that requested the evaluation.
    pub phase: ArtPhase,
    /// Geometry passed to the surface.
    pub position: Array1<f64>,
    /// Energy the surface returned.
    pub energy: f64,
    /// [`ForceGate::LinfNorm`] of the gradient the surface returned.
    pub max_force: f64,
}

/// Reversibility of the two quenched images.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtReversibility {
    /// Exactly one quench lies in the reactant basin.
    pub returns_to_reactant: bool,
    /// That other image is a different minimum.
    pub product_is_distinct: bool,
    /// Both images lie in one basin. A completed exit is not this.
    pub reverted: bool,
    /// Both quench energies are strictly below the saddle.
    pub downhill: bool,
    /// Distance from the back image to the reactant.
    pub back_rmsd: f64,
    /// Distance from the forward image to the reactant.
    pub forward_rmsd: f64,
    /// Distance between the two quenched images.
    pub separation_rmsd: f64,
}

/// One macro-step.
#[derive(Clone, Debug)]
pub struct ArtReport {
    /// Verdict so far.
    pub status: ArtStatus,
    /// Phase stored for the next step. [`ArtPhase::Finished`] after the verdict.
    pub phase: ArtPhase,
    /// Largest absolute force component at the current point.
    pub max_force: f64,
    /// Lowest curvature from the last Lanczos estimate.
    pub curvature: f64,
    /// Energy at the current point.
    pub energy: f64,
    /// Opening pushes taken.
    pub pushes: usize,
    /// Climb translations taken.
    pub climbs: usize,
    /// Force evaluations during this step.
    pub evaluations: usize,
    /// Force evaluations since [`ArtSession::new`].
    pub total_evaluations: usize,
    /// Steps taken since [`ArtSession::new`].
    pub iteration: usize,
    /// Certificate, once [`ArtPhase::Certify`] has run.
    pub saddle_index: Option<usize>,
    /// Filled by [`ArtPhase::Report`].
    pub reversibility: Option<ArtReversibility>,
}

/// A finished or partial event.
#[derive(Clone, Debug)]
pub struct ArtOutcome {
    /// Verdict. [`ArtStatus::Running`] if the host stopped early.
    pub status: ArtStatus,
    /// Phase stored at the end.
    pub phase: ArtPhase,
    /// Phases executed, in order, including repeats.
    pub phases: Vec<ArtPhase>,
    /// Certified point, when the search reached [`ArtPhase::Certify`].
    pub saddle: Option<Array1<f64>>,
    /// Energy of [`Self::saddle`].
    pub saddle_energy: Option<f64>,
    /// Index from [`fn@crate::saddle_index`], when certification ran.
    pub saddle_index: Option<usize>,
    /// Whether that certificate met its residual tolerance.
    pub certificate_converged: bool,
    /// Curvatures returned with the certificate.
    pub curvatures: Vec<f64>,
    /// Image reached by the step back toward the reactant.
    pub minimum_back: Option<Array1<f64>>,
    /// Image reached by the step away from the reactant.
    pub minimum_forward: Option<Array1<f64>>,
    /// Energy of [`Self::minimum_back`].
    pub back_energy: Option<f64>,
    /// Energy of [`Self::minimum_forward`].
    pub forward_energy: Option<f64>,
    /// Largest per-atom force on the back image.
    pub back_force: Option<f64>,
    /// Largest per-atom force on the forward image.
    pub forward_force: Option<f64>,
    /// Lowest curvature at the end of the climb.
    pub curvature: f64,
    /// Energy at the end of the event.
    pub energy: f64,
    /// Largest absolute force component at the end of the event.
    pub max_force: f64,
    /// Opening pushes taken.
    pub pushes: usize,
    /// Climb translations taken.
    pub climbs: usize,
    /// Reversibility, once [`ArtPhase::Report`] has run.
    pub reversibility: Option<ArtReversibility>,
    /// Earlier event in a [`run_events`] batch with the same saddle.
    pub duplicate_of: Option<usize>,
    /// Every force evaluation of the event.
    pub force_calls: Vec<ArtForceCall>,
}

/// One later event whose saddle matches an earlier one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtDuplicate {
    /// Index in the batch.
    pub event: usize,
    /// Earlier event with the same saddle.
    pub same_saddle_as: usize,
    /// [`structure_distance`] between the two saddles.
    pub rmsd: f64,
}

/// Independent events from one reactant.
#[derive(Clone, Debug)]
pub struct ArtBatch {
    /// One outcome per seed, in the order given.
    pub outcomes: Vec<ArtOutcome>,
    /// Pairs that landed on one saddle.
    pub duplicates: Vec<ArtDuplicate>,
}

/// Host direction for one event.
///
/// `seed` is the displacement direction. `mask` multiplies it element
/// by element before it is normalized. A zero mask entry drops that
/// coordinate. The library does not draw either array.
#[derive(Clone, Debug)]
pub struct ArtSeed {
    /// Direction supplied by the host.
    pub seed: Array1<f64>,
    /// Per-coordinate weight. Zero keeps that coordinate fixed at the push.
    pub mask: Array1<f64>,
}

/// Mask of ones.
#[must_use]
pub fn full_mask(n: usize) -> Array1<f64> {
    Array1::from_elem(n, 1.0)
}

/// Mask that is 1 on the Cartesian coordinates of `atoms` and 0 elsewhere.
pub fn atom_mask(n_coordinates: usize, atoms: &[usize]) -> Result<Array1<f64>, SaddleError> {
    if n_coordinates == 0 || !n_coordinates.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "an atom mask needs whole Cartesian triples".into(),
        ));
    }
    let n_atoms = n_coordinates / 3;
    let mut mask = Array1::zeros(n_coordinates);
    for &atom in atoms {
        if atom >= n_atoms {
            return Err(SaddleError::Invalid(
                "atom mask index is outside the system".into(),
            ));
        }
        mask[3 * atom] = 1.0;
        mask[3 * atom + 1] = 1.0;
        mask[3 * atom + 2] = 1.0;
    }
    Ok(mask)
}

/// ART nouveau from one minimum.
///
/// [`ArtSession::new`] does not evaluate the surface. The position is
/// the reactant until the first [`ArtPhase::Push`].
pub struct ArtSession {
    config: ArtConfig,
    gate: ForceGate,
    modes: MinModeSession,
    phase: ArtPhase,
    status: ArtStatus,
    leg: Leg,
    origin: Array1<f64>,
    direction: Array1<f64>,
    x: Array1<f64>,
    mode: Array1<f64>,
    gradient: Array1<f64>,
    plane_anchor: Array1<f64>,
    plane_axis: Array1<f64>,
    energy: f64,
    curvature: f64,
    max_force: f64,
    pushes: usize,
    climbs: usize,
    iteration: usize,
    saddle_index: Option<usize>,
    certificate_converged: bool,
    curvatures: Vec<f64>,
    saddle: Option<Array1<f64>>,
    saddle_energy: f64,
    forward_start: Option<Array1<f64>>,
    back_start: Option<Array1<f64>>,
    minimum_forward: Option<Array1<f64>>,
    minimum_back: Option<Array1<f64>>,
    forward_energy: f64,
    back_energy: f64,
    forward_force: f64,
    back_force: f64,
    reversibility: Option<ArtReversibility>,
    phases: Vec<ArtPhase>,
    calls: Arc<Mutex<Vec<ArtForceCall>>>,
}

impl ArtSession {
    /// Store `minimum` and the masked `seed`. No surface call.
    pub fn new(
        config: ArtConfig,
        minimum: Array1<f64>,
        seed: Array1<f64>,
        mask: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        config.validate()?;
        if minimum.is_empty()
            || !minimum.len().is_multiple_of(3)
            || !minimum.iter().all(|value| value.is_finite())
        {
            return Err(SaddleError::Invalid(
                "ART minimum must be finite Cartesian triples".into(),
            ));
        }
        if config.permute_identical && minimum.len() / 3 > PERMUTATION_ATOM_CAP {
            return Err(SaddleError::Invalid(format!(
                "identical-atom ART comparison accepts at most {PERMUTATION_ATOM_CAP} atoms"
            )));
        }
        let direction = masked_direction(&seed, &mask)?;
        if direction.len() != minimum.len() {
            return Err(SaddleError::Shape("ART seed must match the minimum".into()));
        }
        let modes = MinModeSession::new(config.estimator(), minimum.clone(), direction.clone())?;
        let n = minimum.len();
        Ok(Self {
            config,
            gate: ForceGate::LinfNorm,
            modes,
            phase: ArtPhase::Push,
            status: ArtStatus::Running,
            leg: Leg::Opening,
            origin: minimum.clone(),
            direction: direction.clone(),
            x: minimum,
            mode: direction.clone(),
            gradient: Array1::zeros(n),
            plane_anchor: Array1::zeros(n),
            plane_axis: direction,
            energy: f64::NAN,
            curvature: f64::NAN,
            max_force: f64::NAN,
            pushes: 0,
            climbs: 0,
            iteration: 0,
            saddle_index: None,
            certificate_converged: false,
            curvatures: Vec::new(),
            saddle: None,
            saddle_energy: f64::NAN,
            forward_start: None,
            back_start: None,
            minimum_forward: None,
            minimum_back: None,
            forward_energy: f64::NAN,
            back_energy: f64::NAN,
            forward_force: f64::NAN,
            back_force: f64::NAN,
            reversibility: None,
            phases: Vec::new(),
            calls: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Phase the next [`ArtSession::step`] will run.
    #[must_use]
    pub fn phase(&self) -> ArtPhase {
        self.phase
    }

    /// Verdict so far.
    #[must_use]
    pub fn status(&self) -> ArtStatus {
        self.status
    }

    /// Current geometry.
    #[must_use]
    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Opening direction. Basin pushes stay on it.
    #[must_use]
    pub fn push_direction(&self) -> ArrayView1<'_, f64> {
        self.direction.view()
    }

    /// Last Lanczos mode. Before the first estimate this is the opening direction.
    #[must_use]
    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    /// Last Lanczos curvature.
    #[must_use]
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Force evaluations so far, in order.
    #[must_use]
    pub fn force_calls(&self) -> Vec<ArtForceCall> {
        match self.calls.lock() {
            Ok(guard) => guard.clone(),
            Err(poison) => poison.into_inner().clone(),
        }
    }

    /// The event as it stands, including every force call.
    #[must_use]
    pub fn outcome(&self) -> ArtOutcome {
        ArtOutcome {
            status: self.status,
            phase: self.phase,
            phases: self.phases.clone(),
            saddle: self.saddle.clone(),
            saddle_energy: self.saddle.as_ref().map(|_| self.saddle_energy),
            saddle_index: self.saddle_index,
            certificate_converged: self.certificate_converged,
            curvatures: self.curvatures.clone(),
            minimum_back: self.minimum_back.clone(),
            minimum_forward: self.minimum_forward.clone(),
            back_energy: self.minimum_back.as_ref().map(|_| self.back_energy),
            forward_energy: self.minimum_forward.as_ref().map(|_| self.forward_energy),
            back_force: self.minimum_back.as_ref().map(|_| self.back_force),
            forward_force: self.minimum_forward.as_ref().map(|_| self.forward_force),
            curvature: self.curvature,
            energy: self.energy,
            max_force: self.max_force,
            pushes: self.pushes,
            climbs: self.climbs,
            reversibility: self.reversibility,
            duplicate_of: None,
            force_calls: self.force_calls(),
        }
    }

    /// Run the stored phase once.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<ArtReport, SaddleError> {
        if self.phase == ArtPhase::Finished {
            return Ok(self.report(0));
        }
        let before = self.call_len();
        self.phases.push(self.phase);
        self.advance(surface)?;
        let evaluations = self.call_len().saturating_sub(before);
        self.iteration += 1;
        Ok(self.report(evaluations))
    }

    /// Repeat [`ArtSession::step`] until the verdict or `max_steps` steps.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<ArtReport, SaddleError> {
        let mut taken = 0;
        let mut report = self.report(0);
        while self.phase != ArtPhase::Finished && taken < max_steps {
            report = self.step(surface)?;
            taken += 1;
        }
        if self.phase != ArtPhase::Finished {
            self.status = ArtStatus::Exhausted;
            self.phase = ArtPhase::Finished;
            report = self.report(0);
        }
        Ok(report)
    }

    fn advance<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        match self.phase {
            ArtPhase::Push => self.push(surface),
            ArtPhase::Perpendicular => self.perpendicular(surface),
            ArtPhase::Lanczos => self.lanczos(surface),
            ArtPhase::Climb => self.climb(surface),
            ArtPhase::Certify => self.certify(surface),
            ArtPhase::PushOver => self.push_over(),
            ArtPhase::Quench => self.quench(surface),
            ArtPhase::Report => self.report_phase(),
            ArtPhase::Finished => Ok(()),
        }
    }

    fn push<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let length = if self.pushes == 0 {
            self.config.push_length
        } else {
            self.config.push_step
        };
        let next = &self.x + &(&self.direction * length);
        let (energy, gradient) = self.observe(surface, next.view(), ArtPhase::Push)?;
        self.x = next;
        self.energy = energy;
        self.gradient = gradient;
        self.max_force = self.gate.value(self.gradient.view());
        self.pushes += 1;
        self.leg = Leg::Opening;
        self.plane_anchor = self.x.clone();
        self.plane_axis = self.direction.clone();
        self.phase = ArtPhase::Perpendicular;
        Ok(())
    }

    fn perpendicular<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let anchor = self.plane_anchor.clone();
        let axis = self.plane_axis.clone();
        let start = self.x.clone();
        let plane = Hyperplane {
            inner: surface,
            log: Arc::clone(&self.calls),
            gate: self.gate,
            phase: ArtPhase::Perpendicular,
            axis: axis.clone(),
        };
        let relaxed = relax_to_tolerance(&plane, start, OPEN_CELL, &self.config.perpendicular)?;
        let mut position = relaxed.positions;
        snap_to_plane(&mut position, &anchor, &axis);
        let (energy, gradient) = self.observe(surface, position.view(), ArtPhase::Perpendicular)?;
        self.x = position;
        self.energy = energy;
        self.gradient = gradient;
        self.max_force = self.gate.value(self.gradient.view());
        self.phase = ArtPhase::Lanczos;
        Ok(())
    }

    fn lanczos<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let logged = Counting {
            inner: surface,
            log: Arc::clone(&self.calls),
            gate: self.gate,
            phase: ArtPhase::Lanczos,
        };
        self.modes
            .set_position(self.x.clone(), Some(self.gradient.clone()))?;
        let seed = match self.leg {
            Leg::Opening => self.direction.clone(),
            Leg::Climbing => self.mode.clone(),
        };
        self.modes.set_mode(seed)?;
        let estimate = self.modes.estimate_mode(&logged)?;
        self.mode = estimate.mode;
        self.curvature = estimate.curvature;
        match self.leg {
            Leg::Opening => {
                if self.curvature < self.config.eigen_threshold {
                    self.leg = Leg::Climbing;
                    self.phase = ArtPhase::Climb;
                } else if self.pushes >= self.config.max_pushes {
                    self.finish(ArtStatus::Exhausted);
                } else {
                    self.phase = ArtPhase::Push;
                }
            }
            Leg::Climbing => {
                if !(self.curvature.is_finite() && self.curvature < 0.0) {
                    self.finish(ArtStatus::LostCurvature);
                } else if self.max_force <= self.config.mode.force_tol {
                    self.phase = ArtPhase::Certify;
                } else if self.climbs >= self.config.max_climbs {
                    self.finish(ArtStatus::Exhausted);
                } else {
                    self.phase = ArtPhase::Climb;
                }
            }
        }
        Ok(())
    }

    fn climb<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let logged = Counting {
            inner: surface,
            log: Arc::clone(&self.calls),
            gate: self.gate,
            phase: ArtPhase::Climb,
        };
        self.modes
            .set_position(self.x.clone(), Some(self.gradient.clone()))?;
        self.modes.set_mode(self.mode.clone())?;
        let stepped = self.modes.step(&logged)?;
        self.curvature = stepped.curvature;
        self.mode = self.modes.mode().to_owned();
        self.x = self.modes.position().to_owned();
        self.climbs += 1;
        drop(logged);
        let (energy, gradient) = self.observe(surface, self.x.view(), ArtPhase::Climb)?;
        self.energy = energy;
        self.gradient = gradient;
        self.max_force = self.gate.value(self.gradient.view());
        self.leg = Leg::Climbing;
        self.plane_anchor = self.x.clone();
        self.plane_axis = self.mode.clone();
        self.phase = ArtPhase::Perpendicular;
        Ok(())
    }

    fn certify<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let logged = Counting {
            inner: surface,
            log: Arc::clone(&self.calls),
            gate: self.gate,
            phase: ArtPhase::Certify,
        };
        let found = saddle_index(
            &self.config.saddle,
            &logged,
            self.x.view(),
            Some(self.gradient.view()),
            Some(self.mode.view()),
            &[],
        )?;
        self.saddle_index = Some(found.index);
        self.certificate_converged = found.converged;
        self.curvatures = found.curvatures;
        self.saddle = Some(self.x.clone());
        self.saddle_energy = self.energy;
        if found.converged && found.index == 1 {
            self.phase = ArtPhase::PushOver;
        } else {
            self.finish(ArtStatus::NotIndexOne);
        }
        Ok(())
    }

    fn push_over(&mut self) -> Result<(), SaddleError> {
        let mode = self.outward_mode();
        let step = &mode * self.config.push_over;
        let forward = &self.x + &step;
        let back = &self.x - &step;
        if !forward.iter().all(|value| value.is_finite())
            || !back.iter().all(|value| value.is_finite())
        {
            return Err(SaddleError::NonFinite("ART push over the saddle"));
        }
        self.forward_start = Some(forward);
        self.back_start = Some(back);
        self.phase = ArtPhase::Quench;
        Ok(())
    }

    fn quench<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let forward_start = self
            .forward_start
            .clone()
            .ok_or_else(|| SaddleError::Invalid("ART quench has no forward image".into()))?;
        let back_start = self
            .back_start
            .clone()
            .ok_or_else(|| SaddleError::Invalid("ART quench has no back image".into()))?;
        let logged = Counting {
            inner: surface,
            log: Arc::clone(&self.calls),
            gate: self.gate,
            phase: ArtPhase::Quench,
        };
        let forward = relax_to_tolerance(&logged, forward_start, OPEN_CELL, &self.config.quench)?;
        let back = relax_to_tolerance(&logged, back_start, OPEN_CELL, &self.config.quench)?;
        self.minimum_forward = Some(forward.positions);
        self.minimum_back = Some(back.positions);
        self.forward_energy = forward.energy;
        self.back_energy = back.energy;
        self.forward_force = forward.max_force;
        self.back_force = back.max_force;
        self.phase = ArtPhase::Report;
        Ok(())
    }

    fn report_phase(&mut self) -> Result<(), SaddleError> {
        let back = self
            .minimum_back
            .clone()
            .ok_or_else(|| SaddleError::Invalid("ART report has no back minimum".into()))?;
        let forward = self
            .minimum_forward
            .clone()
            .ok_or_else(|| SaddleError::Invalid("ART report has no forward minimum".into()))?;
        let (status, reversibility) = connection_verdict(
            self.origin.view(),
            back.view(),
            forward.view(),
            self.saddle_energy,
            self.back_energy,
            self.forward_energy,
            self.config.basin_rmsd,
            self.config.permute_identical,
        )?;
        self.reversibility = Some(reversibility);
        self.finish(status);
        Ok(())
    }

    fn outward_mode(&self) -> Array1<f64> {
        let mut mode = self.mode.clone();
        let from_reactant = &self.x - &self.origin;
        let scale = from_reactant.dot(&from_reactant).sqrt();
        let reference = if scale.is_finite() && scale > 0.0 {
            &from_reactant / scale
        } else {
            self.direction.clone()
        };
        if mode.dot(&reference) < 0.0 {
            mode.mapv_inplace(|value| -value);
        }
        mode
    }

    fn finish(&mut self, status: ArtStatus) {
        self.status = status;
        self.phase = ArtPhase::Finished;
    }

    fn observe<S: PointSurface>(
        &self,
        surface: &S,
        x: ArrayView1<f64>,
        phase: ArtPhase,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        observed_eval(surface, x, &self.calls, self.gate, phase)
    }

    fn call_len(&self) -> usize {
        match self.calls.lock() {
            Ok(guard) => guard.len(),
            Err(poison) => poison.into_inner().len(),
        }
    }

    fn report(&self, evaluations: usize) -> ArtReport {
        ArtReport {
            status: self.status,
            phase: self.phase,
            max_force: self.max_force,
            curvature: self.curvature,
            energy: self.energy,
            pushes: self.pushes,
            climbs: self.climbs,
            evaluations,
            total_evaluations: self.call_len(),
            iteration: self.iteration,
            saddle_index: self.saddle_index,
            reversibility: self.reversibility,
        }
    }
}

/// Run one event per seed, then mark duplicate saddles.
///
/// Event `i` is seed `i`. A failure returns that event's error. The
/// structural distance is [`structure_distance`].
pub fn run_events<S: PointSurface>(
    config: &ArtConfig,
    minimum: &Array1<f64>,
    seeds: &[ArtSeed],
    surface: &S,
    max_steps: usize,
) -> Result<ArtBatch, SaddleError> {
    config.validate()?;
    let mut outcomes = Vec::with_capacity(seeds.len());
    for (index, seed) in seeds.iter().enumerate() {
        let mut session = ArtSession::new(
            config.clone(),
            minimum.clone(),
            seed.seed.clone(),
            seed.mask.clone(),
        )
        .map_err(|error| {
            SaddleError::Invalid(format!("ART event {index} was not started: {error}"))
        })?;
        session
            .run(surface, max_steps)
            .map_err(|error| SaddleError::Invalid(format!("ART event {index} failed: {error}")))?;
        outcomes.push(session.outcome());
    }
    let duplicates = assign_duplicates(
        &mut outcomes,
        config.duplicate_rmsd,
        config.permute_identical,
    )?;
    Ok(ArtBatch {
        outcomes,
        duplicates,
    })
}

/// Classify the two quenched images.
///
/// [`ArtStatus::Reverted`] is returned whenever the images lie in one
/// basin, and that status is never [`ArtStatus::Connected`].
#[allow(clippy::too_many_arguments)]
pub fn connection_verdict(
    reactant: ArrayView1<f64>,
    back: ArrayView1<f64>,
    forward: ArrayView1<f64>,
    saddle_energy: f64,
    back_energy: f64,
    forward_energy: f64,
    basin_rmsd: f64,
    permute_identical: bool,
) -> Result<(ArtStatus, ArtReversibility), SaddleError> {
    if !(basin_rmsd.is_finite() && basin_rmsd > 0.0) {
        return Err(SaddleError::Invalid(
            "ART basin distance must be finite and positive".into(),
        ));
    }
    let back_rmsd = structure_distance(reactant, back, permute_identical)?;
    let forward_rmsd = structure_distance(reactant, forward, permute_identical)?;
    let separation_rmsd = structure_distance(back, forward, permute_identical)?;
    // The unstable mode at a curved saddle need not point back along the
    // arrival displacement, so either image may be the reactant.
    let back_home = back_rmsd <= basin_rmsd;
    let forward_home = forward_rmsd <= basin_rmsd;
    let reverted = separation_rmsd <= basin_rmsd || (back_home && forward_home);
    let returns_to_reactant = (back_home || forward_home) && !reverted;
    let product_is_distinct = returns_to_reactant && separation_rmsd > basin_rmsd;
    let downhill = back_energy < saddle_energy && forward_energy < saddle_energy;
    let status = if reverted {
        ArtStatus::Reverted
    } else if !downhill {
        ArtStatus::NotDownhill
    } else if !returns_to_reactant {
        ArtStatus::Irreversible
    } else {
        ArtStatus::Connected
    };
    Ok((
        status,
        ArtReversibility {
            returns_to_reactant,
            product_is_distinct,
            reverted,
            downhill,
            back_rmsd,
            forward_rmsd,
            separation_rmsd,
        },
    ))
}

/// Distance used for basins and duplicate saddles.
///
/// One atom is the Euclidean distance: centering it would erase the
/// coordinate. Two or more atoms are centered, then either the Kabsch
/// RMSD or, when `permute_identical` is set, the RMSD of the sorted
/// pairwise distances. The sorted spectrum is unchanged by translation,
/// rotation, and permutation, which is the identical-atom comparison of
/// Malek and Mousseau. More than eight atoms is refused when
/// permutations are requested, because the caller should not discover
/// that limit only after a search.
pub fn structure_distance(
    left: ArrayView1<f64>,
    right: ArrayView1<f64>,
    permute_identical: bool,
) -> Result<f64, SaddleError> {
    if left.len() != right.len() || left.is_empty() || !left.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "ART structure distance needs matching Cartesian triples".into(),
        ));
    }
    if !left.iter().all(|value| value.is_finite()) || !right.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("ART structure distance"));
    }
    let n_atoms = left.len() / 3;
    if n_atoms == 1 {
        return Ok(euclidean(left, right));
    }
    if permute_identical {
        if n_atoms > PERMUTATION_ATOM_CAP {
            return Err(SaddleError::Invalid(format!(
                "identical-atom ART comparison accepts at most {PERMUTATION_ATOM_CAP} atoms"
            )));
        }
        return Ok(spectrum_rmsd(&atoms_of(left), &atoms_of(right)));
    }
    Ok(kabsch_rmsd(
        &center_atoms(&atoms_of(left)),
        &center_atoms(&atoms_of(right)),
    ))
}

fn assign_duplicates(
    outcomes: &mut [ArtOutcome],
    tolerance: f64,
    permute_identical: bool,
) -> Result<Vec<ArtDuplicate>, SaddleError> {
    let mut duplicates = Vec::new();
    for event in 0..outcomes.len() {
        let Some(saddle) = outcomes[event].saddle.clone() else {
            continue;
        };
        for earlier in 0..event {
            if outcomes[earlier].duplicate_of.is_some() {
                continue;
            }
            let Some(previous) = outcomes[earlier].saddle.clone() else {
                continue;
            };
            let rmsd = structure_distance(previous.view(), saddle.view(), permute_identical)?;
            if rmsd <= tolerance {
                outcomes[event].duplicate_of = Some(earlier);
                duplicates.push(ArtDuplicate {
                    event,
                    same_saddle_as: earlier,
                    rmsd,
                });
                break;
            }
        }
    }
    Ok(duplicates)
}

fn masked_direction(seed: &Array1<f64>, mask: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    if seed.len() != mask.len() || seed.is_empty() {
        return Err(SaddleError::Shape(
            "ART seed and mask must share a nonzero length".into(),
        ));
    }
    if !seed.iter().all(|value| value.is_finite()) || !mask.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("ART seed"));
    }
    let mut direction = Array1::zeros(seed.len());
    for index in 0..seed.len() {
        direction[index] = seed[index] * mask[index];
    }
    let norm = direction.dot(&direction).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(SaddleError::Invalid(
            "masked ART seed must be nonzero".into(),
        ));
    }
    direction.mapv_inplace(|value| value / norm);
    Ok(direction)
}

fn snap_to_plane(position: &mut Array1<f64>, anchor: &Array1<f64>, axis: &Array1<f64>) {
    let mut along = 0.0;
    for index in 0..position.len() {
        along += (position[index] - anchor[index]) * axis[index];
    }
    if !along.is_finite() {
        return;
    }
    for index in 0..position.len() {
        position[index] -= along * axis[index];
    }
}

fn observed_eval<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    log: &Arc<Mutex<Vec<ArtForceCall>>>,
    gate: ForceGate,
    phase: ArtPhase,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let (energy, gradient) = surface.eval(x)?;
    let shaped = gradient.len() == x.len();
    let finite = shaped && energy.is_finite() && gradient.iter().all(|value| value.is_finite());
    let max_force = if finite {
        gate.value(gradient.view())
    } else {
        f64::NAN
    };
    push_call(
        log,
        ArtForceCall {
            phase,
            position: x.to_owned(),
            energy,
            max_force,
        },
    );
    if !shaped {
        return Err(SaddleError::Shape(
            "ART gradient must match the position".into(),
        ));
    }
    if !finite {
        return Err(SaddleError::NonFinite("ART surface"));
    }
    Ok((energy, gradient))
}

fn push_call(log: &Arc<Mutex<Vec<ArtForceCall>>>, call: ArtForceCall) {
    match log.lock() {
        Ok(mut guard) => guard.push(call),
        Err(poison) => poison.into_inner().push(call),
    }
}

struct Counting<'a, S> {
    inner: &'a S,
    log: Arc<Mutex<Vec<ArtForceCall>>>,
    gate: ForceGate,
    phase: ArtPhase,
}

impl<S: PointSurface> PointSurface for Counting<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        observed_eval(self.inner, x, &self.log, self.gate, self.phase)
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.inner.hessian_vector(x, v)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.inner.excluded_modes(x)
    }
}

struct Hyperplane<'a, S> {
    inner: &'a S,
    log: Arc<Mutex<Vec<ArtForceCall>>>,
    gate: ForceGate,
    phase: ArtPhase,
    axis: Array1<f64>,
}

impl<S: PointSurface> PointSurface for Hyperplane<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = observed_eval(self.inner, x, &self.log, self.gate, self.phase)?;
        let excluded = self.inner.excluded_modes(x)?;
        Ok((energy, project_hyperplane(gradient, &self.axis, &excluded)))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.inner.excluded_modes(x)
    }
}

fn project_hyperplane(
    gradient: Array1<f64>,
    axis: &Array1<f64>,
    excluded: &Array2<f64>,
) -> Array1<f64> {
    let mut force = perp(gradient.view(), axis.view());
    if excluded.ncols() != force.len() {
        return force;
    }
    for row in excluded.rows() {
        let mut direction = perp(row, axis.view());
        let norm = direction.dot(&direction).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            continue;
        }
        direction.mapv_inplace(|value| value / norm);
        if direction.iter().all(|value| value.is_finite()) {
            force = perp(force.view(), direction.view());
        }
    }
    force
}

fn euclidean(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn atoms_of(x: ArrayView1<f64>) -> Vec<[f64; 3]> {
    let mut atoms = Vec::with_capacity(x.len() / 3);
    let mut index = 0;
    while index + 3 <= x.len() {
        atoms.push([x[index], x[index + 1], x[index + 2]]);
        index += 3;
    }
    atoms
}

fn center_atoms(atoms: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut com = [0.0; 3];
    for atom in atoms {
        for axis in 0..3 {
            com[axis] += atom[axis];
        }
    }
    let scale = atoms.len() as f64;
    for value in &mut com {
        *value /= scale;
    }
    atoms
        .iter()
        .map(|atom| [atom[0] - com[0], atom[1] - com[1], atom[2] - com[2]])
        .collect()
}

fn spectrum_rmsd(left: &[[f64; 3]], right: &[[f64; 3]]) -> f64 {
    let mut a = pair_distances(left);
    let mut b = pair_distances(right);
    sort_finite(&mut a);
    sort_finite(&mut b);
    let mut error = 0.0;
    for (left_distance, right_distance) in a.iter().zip(&b) {
        let delta = left_distance - right_distance;
        error += delta * delta;
    }
    if a.is_empty() {
        0.0
    } else {
        (error / a.len() as f64).sqrt()
    }
}

fn pair_distances(atoms: &[[f64; 3]]) -> Vec<f64> {
    let mut distances = Vec::new();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let mut square = 0.0;
            for axis in 0..3 {
                let delta = atoms[i][axis] - atoms[j][axis];
                square += delta * delta;
            }
            distances.push(square.sqrt());
        }
    }
    distances
}

fn sort_finite(values: &mut [f64]) {
    values.sort_by(|left, right| match left.partial_cmp(right) {
        Some(order) => order,
        None => std::cmp::Ordering::Equal,
    });
}

fn kabsch_rmsd(reference: &[[f64; 3]], moving: &[[f64; 3]]) -> f64 {
    let Some(rotation) = kabsch_rotation(reference, moving) else {
        return plain_rmsd(reference, moving);
    };
    let mut error = 0.0;
    for (point, target) in moving.iter().zip(reference) {
        let rotated = mat_vec(rotation, *point);
        for axis in 0..3 {
            let delta = rotated[axis] - target[axis];
            error += delta * delta;
        }
    }
    if reference.is_empty() {
        0.0
    } else {
        (error / reference.len() as f64).sqrt()
    }
}

fn plain_rmsd(reference: &[[f64; 3]], moving: &[[f64; 3]]) -> f64 {
    let mut error = 0.0;
    for (point, target) in moving.iter().zip(reference) {
        for axis in 0..3 {
            let delta = point[axis] - target[axis];
            error += delta * delta;
        }
    }
    if reference.is_empty() {
        0.0
    } else {
        (error / reference.len() as f64).sqrt()
    }
}

fn kabsch_rotation(reference: &[[f64; 3]], moving: &[[f64; 3]]) -> Option<[[f64; 3]; 3]> {
    let mut covariance = [[0.0; 3]; 3];
    for (point, target) in moving.iter().zip(reference) {
        for (cov_row, &target_coord) in covariance.iter_mut().zip(target) {
            for (entry, &point_coord) in cov_row.iter_mut().zip(point) {
                *entry += target_coord * point_coord;
            }
        }
    }
    let mut gram = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let mut sum = 0.0;
            for cov_row in &covariance {
                sum += cov_row[row] * cov_row[column];
            }
            gram[row][column] = sum;
        }
    }
    let (mut vectors, values) = jacobi_eigen(gram);
    let mut sigma = [0.0; 3];
    let mut largest = 0.0;
    for index in 0..3 {
        let value = if values[index] > 0.0 {
            values[index]
        } else {
            0.0
        };
        sigma[index] = value.sqrt();
        if sigma[index] > largest {
            largest = sigma[index];
        }
    }
    let mut left = [[0.0; 3]; 3];
    let floor = largest * f64::EPSILON * 32.0;
    for (column, &singular) in sigma.iter().enumerate() {
        if singular > floor {
            for (left_row, cov_row) in left.iter_mut().zip(&covariance) {
                let mut sum = 0.0;
                for (&cov_entry, vector_row) in cov_row.iter().zip(&vectors) {
                    sum += cov_entry * vector_row[column];
                }
                left_row[column] = sum / singular;
            }
        }
    }
    complete_left(&mut left, &sigma, floor);
    let mut rotation = mat_mul(left, mat_transpose(vectors));
    if mat_det(rotation) < 0.0 {
        let mut column = 0;
        if sigma[1] < sigma[column] {
            column = 1;
        }
        if sigma[2] < sigma[column] {
            column = 2;
        }
        for vector_row in &mut vectors {
            vector_row[column] = -vector_row[column];
        }
        if sigma[column] > floor {
            for (left_row, cov_row) in left.iter_mut().zip(&covariance) {
                let mut sum = 0.0;
                for (&cov_entry, vector_row) in cov_row.iter().zip(&vectors) {
                    sum += cov_entry * vector_row[column];
                }
                left_row[column] = sum / sigma[column];
            }
        }
        complete_left(&mut left, &sigma, floor);
        rotation = mat_mul(left, mat_transpose(vectors));
    }
    mat_det(rotation).is_finite().then_some(rotation)
}

fn complete_left(left: &mut [[f64; 3]; 3], sigma: &[f64; 3], floor: f64) {
    let mut missing = None;
    let mut present = Vec::new();
    for (column, value) in sigma.iter().enumerate() {
        if *value > floor {
            present.push(column);
        } else {
            missing = Some(column);
        }
    }
    if present.len() == 2
        && let Some(column) = missing
    {
        let a = [
            left[0][present[0]],
            left[1][present[0]],
            left[2][present[0]],
        ];
        let b = [
            left[0][present[1]],
            left[1][present[1]],
            left[2][present[1]],
        ];
        let cross = cross(a, b);
        let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if norm.is_finite() && norm > 0.0 {
            for row in 0..3 {
                left[row][column] = cross[row] / norm;
            }
        }
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn jacobi_eigen(mut matrix: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let mut pivot_row = 0;
        let mut pivot_column = 1;
        let mut biggest = 0.0;
        for (row, column) in pairs {
            let value = matrix[row][column].abs();
            if value > biggest {
                biggest = value;
                pivot_row = row;
                pivot_column = column;
            }
        }
        if biggest == 0.0 {
            break;
        }
        let app = matrix[pivot_row][pivot_row];
        let aqq = matrix[pivot_column][pivot_column];
        let apq = matrix[pivot_row][pivot_column];
        let tau = (aqq - app) / (2.0 * apq);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let mut row_p = [0.0; 3];
        let mut row_q = [0.0; 3];
        for index in 0..3 {
            row_p[index] = matrix[index][pivot_row];
            row_q[index] = matrix[index][pivot_column];
        }
        for index in 0..3 {
            if index == pivot_row || index == pivot_column {
                continue;
            }
            let aik = cosine * row_p[index] - sine * row_q[index];
            let ajk = sine * row_p[index] + cosine * row_q[index];
            matrix[index][pivot_row] = aik;
            matrix[pivot_row][index] = aik;
            matrix[index][pivot_column] = ajk;
            matrix[pivot_column][index] = ajk;
        }
        matrix[pivot_row][pivot_row] =
            cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[pivot_column][pivot_column] =
            sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[pivot_row][pivot_column] = 0.0;
        matrix[pivot_column][pivot_row] = 0.0;
        for vector_row in &mut vectors {
            let vip = vector_row[pivot_row];
            let viq = vector_row[pivot_column];
            vector_row[pivot_row] = cosine * vip - sine * viq;
            vector_row[pivot_column] = sine * vip + cosine * viq;
        }
    }
    (vectors, [matrix[0][0], matrix[1][1], matrix[2][2]])
}

fn mat_transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mat_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            out[row][column] = left[row][0] * right[0][column]
                + left[row][1] * right[1][column]
                + left[row][2] * right[2][column];
        }
    }
    out
}

fn mat_vec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn mat_det(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Well {
        calls: AtomicUsize,
    }

    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let (a, b, c) = (x[0], x[1], x[2]);
            let energy = 0.25 * (a * a - 1.0).powi(2) + 0.5 * (b * b + c * c);
            Ok((energy, array![a * a * a - a, b, c]))
        }
    }

    fn well_config() -> ArtConfig {
        let mode = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_saddle_check(None)
            .with_force_tol(0.02)
            .with_max_move(0.1)
            .with_krylov_dim(4)
            .with_max_rotations(6);
        ArtConfig::default()
            .with_mode(mode)
            .with_push_length(0.8)
            .with_max_pushes(3)
            .with_max_climbs(25)
            .with_push_over(0.25)
    }

    #[test]
    fn default_art_tolerances_are_consistent() {
        let host = ArtConfig::default();
        let lj = ArtConfig::lj_art();
        host.validate().unwrap();
        assert_eq!(host.perpendicular.force_tol, host.mode.force_tol);
        assert_eq!(host.push_length, 1.0);
        assert_eq!(host.saddle.difference, FiniteDifference::Forward);
        assert_ne!(host.push_length, lj.push_length);
        assert_eq!(lj.saddle.difference, FiniteDifference::Central);
        assert!(lj.perpendicular.force_tol <= lj.mode.force_tol);
    }

    #[test]
    fn looser_perpendicular_tolerance_has_a_typed_error() {
        let mode_tol = ArtConfig::default().mode.force_tol;
        let perp_tol = 2.0 * mode_tol;
        let config = ArtConfig::default()
            .with_perpendicular(RelaxConfig::default().with_force_tol(perp_tol));
        match config.validate() {
            Err(SaddleError::Invalid(message)) => {
                assert!(message.contains(&format!(
                    "perpendicular_force_tol={perp_tol:e} energy/length"
                )));
                assert!(message.contains(&format!("mode_force_tol={mode_tol:e} energy/length")));
            }
            other => panic!("unexpected ART validation result: {other:?}"),
        }
    }

    #[test]
    fn a_non_positive_mode_force_tolerance_is_rejected() {
        for force_tol in [0.0, f64::NAN, f64::INFINITY] {
            let config =
                ArtConfig::default().with_mode(ArtConfig::default().mode.with_force_tol(force_tol));
            match config.validate() {
                Err(SaddleError::Invalid(message)) => {
                    assert!(message.contains("mode force tolerance must be finite and positive"));
                }
                other => panic!("unexpected ART validation result: {other:?}"),
            }
        }
    }

    #[test]
    fn art_consumers_reject_a_loose_perpendicular_tolerance() {
        let config = ArtConfig::default().with_perpendicular(
            RelaxConfig::default().with_force_tol(2.0 * ArtConfig::default().mode.force_tol),
        );
        assert!(matches!(
            ArtSession::new(
                config.clone(),
                array![-1.0, 0.0, 0.0],
                array![1.0, 0.0, 0.0],
                full_mask(3),
            ),
            Err(SaddleError::Invalid(_))
        ));
        let minimum = array![-1.0, 0.0, 0.0];
        let surface = Well {
            calls: AtomicUsize::new(0),
        };
        assert!(matches!(
            run_events(&config, &minimum, &[], &surface, 10),
            Err(SaddleError::Invalid(_))
        ));
    }

    #[test]
    fn art_unit_scaling_covers_lengths_forces_and_curvatures() {
        let reduced = ArtConfig::lj_art();
        let scaled = ArtConfig::for_units(2.0, 4.0).unwrap();
        let length_scale = 4.0;
        let force_scale = 0.5;
        let curvature_scale = 0.125;
        assert_eq!(scaled.mode.dr, length_scale * reduced.mode.dr);
        assert_eq!(scaled.mode.max_move, length_scale * reduced.mode.max_move);
        assert_eq!(scaled.push_length, length_scale * reduced.push_length);
        assert_eq!(scaled.push_step, length_scale * reduced.push_step);
        assert_eq!(
            scaled.perpendicular.max_atom_move,
            length_scale * reduced.perpendicular.max_atom_move
        );
        assert_eq!(
            scaled.quench.max_atom_move,
            length_scale * reduced.quench.max_atom_move
        );
        assert_eq!(scaled.push_over, length_scale * reduced.push_over);
        assert_eq!(scaled.saddle.dr, length_scale * reduced.saddle.dr);
        assert_eq!(scaled.basin_rmsd, length_scale * reduced.basin_rmsd);
        assert_eq!(scaled.duplicate_rmsd, length_scale * reduced.duplicate_rmsd);
        assert_eq!(scaled.mode.force_tol, force_scale * reduced.mode.force_tol);
        assert_eq!(
            scaled.perpendicular.force_tol,
            force_scale * reduced.perpendicular.force_tol
        );
        assert_eq!(
            scaled.quench.force_tol,
            force_scale * reduced.quench.force_tol
        );
        assert_eq!(scaled.mode.rotation_tol, reduced.mode.rotation_tol);
        assert_eq!(
            scaled.mode.rotation_angle_tol,
            reduced.mode.rotation_angle_tol
        );
        assert_eq!(
            scaled.eigen_threshold,
            curvature_scale * reduced.eigen_threshold
        );
        assert_eq!(
            scaled.saddle.residual_tol,
            curvature_scale * reduced.saddle.residual_tol
        );
        assert_eq!(
            scaled.saddle.negative_tol,
            curvature_scale * reduced.saddle.negative_tol
        );
        let step_scale = length_scale / 2.0_f64.sqrt();
        assert_eq!(scaled.mode.time_step, step_scale * reduced.mode.time_step);
        assert_eq!(
            scaled.perpendicular.initial_step,
            step_scale * reduced.perpendicular.initial_step
        );
        assert_eq!(
            scaled.quench.initial_step,
            step_scale * reduced.quench.initial_step
        );
        scaled.validate().unwrap();
    }

    #[test]
    fn art_initial_step_scale_follows_the_relaxation_method() {
        let time = 4.0 / 2.0_f64.sqrt();
        let mobility = 4.0 * 4.0 / 2.0;
        for (method, factor) in [
            (Method::fire(rgmin::FireKind::V2), time),
            (Method::QuickMin, time),
            (Method::Steepest, mobility),
            (Method::lbfgs(), mobility),
            (Method::polak_ribiere_plus(), mobility),
            (Method::diis(), mobility),
            (Method::Bb, mobility),
            (Method::adam(), 4.0),
        ] {
            let step = scaled_initial_step(&method, 0.05, 2.0, 4.0).unwrap();
            assert_eq!(step, 0.05 * factor, "{}", method.as_str());
        }
        for (method, name) in [
            (Method::Bfgs, "bfgs"),
            (Method::Dogleg, "dogleg"),
            (Method::Ode12r, "ode12r"),
        ] {
            match scaled_initial_step(&method, 0.05, 2.0, 4.0) {
                Err(SaddleError::Invalid(message)) => {
                    assert!(message.contains(&format!("relaxation method {name}")));
                }
                other => panic!("unexpected ART step scale result: {other:?}"),
            }
        }
    }

    fn unit_fields(config: &ArtConfig) -> [(&'static str, f64); 23] {
        [
            ("mode.dr", config.mode.dr),
            ("mode.force_tol", config.mode.force_tol),
            ("mode.max_move", config.mode.max_move),
            ("mode.time_step", config.mode.time_step),
            ("mode.rotation_tol", config.mode.rotation_tol),
            ("mode.rotation_angle_tol", config.mode.rotation_angle_tol),
            ("eigen_threshold", config.eigen_threshold),
            ("push_length", config.push_length),
            ("push_step", config.push_step),
            ("perpendicular.force_tol", config.perpendicular.force_tol),
            (
                "perpendicular.max_atom_move",
                config.perpendicular.max_atom_move,
            ),
            (
                "perpendicular.initial_step",
                config.perpendicular.initial_step,
            ),
            ("quench.force_tol", config.quench.force_tol),
            ("quench.max_atom_move", config.quench.max_atom_move),
            ("quench.initial_step", config.quench.initial_step),
            ("push_over", config.push_over),
            ("saddle.residual_tol", config.saddle.residual_tol),
            ("saddle.negative_tol", config.saddle.negative_tol),
            ("saddle.dr", config.saddle.dr),
            ("basin_rmsd", config.basin_rmsd),
            ("duplicate_rmsd", config.duplicate_rmsd),
            ("mode.krylov_dim", config.mode.krylov_dim as f64),
            ("max_pushes", config.max_pushes as f64),
        ]
    }

    fn assert_same_units(left: &ArtConfig, right: &ArtConfig, label: &str) {
        for ((name, a), (_, b)) in unit_fields(left).into_iter().zip(unit_fields(right)) {
            assert!(
                (a - b).abs() <= 1e-14 * a.abs().max(b.abs()),
                "{label}: {name} {a:e} != {b:e}"
            );
        }
    }

    #[test]
    fn art_unit_scaling_round_trips() {
        let reduced = ArtConfig::lj_art();
        assert_same_units(
            &ArtConfig::for_units(1.0, 1.0).unwrap(),
            &reduced,
            "identity",
        );
        for (energy, length) in [(2.0, 4.0), (0.4093, 2.338), (1.0e-3, 50.0), (7.5, 0.3)] {
            let there = ArtConfig::for_units(energy, length).unwrap();
            let back = there.clone().scaled(1.0 / energy, 1.0 / length).unwrap();
            assert_same_units(&back, &reduced, "inverse");
            let composed = reduced
                .clone()
                .scaled(energy, 1.0)
                .unwrap()
                .scaled(1.0, length)
                .unwrap();
            assert_same_units(&composed, &there, "composed");
        }
    }

    #[test]
    fn art_unit_scaling_round_trips_each_relaxation_method() {
        for method in [
            Method::QuickMin,
            Method::Steepest,
            Method::lbfgs(),
            Method::Bb,
            Method::adam(),
        ] {
            let relax = RelaxConfig::default()
                .with_method(method.clone())
                .with_force_tol(0.01);
            let config = ArtConfig::lj_art()
                .with_perpendicular(relax.clone())
                .with_quench(relax);
            let back = config
                .clone()
                .scaled(0.4093, 2.338)
                .unwrap()
                .scaled(1.0 / 0.4093, 1.0 / 2.338)
                .unwrap();
            assert_same_units(&back, &config, method.as_str());
        }
    }

    #[test]
    fn art_unit_scales_must_be_positive_and_finite() {
        for (energy, length) in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (1.0, f64::NAN),
            (1.0, f64::INFINITY),
        ] {
            match ArtConfig::for_units(energy, length) {
                Err(SaddleError::Invalid(message)) => assert_eq!(
                    message,
                    "ART energy and length scales must be finite and positive"
                ),
                other => panic!("unexpected ART unit-scale result: {other:?}"),
            }
        }
    }

    #[test]
    fn art_validation_rejects_each_boundary_independently() {
        for config in [
            ArtConfig::default().with_eigen_threshold(0.01),
            ArtConfig::default().with_push_length(0.0),
            ArtConfig::default().with_max_pushes(0),
            ArtConfig::default().with_max_climbs(0),
        ] {
            assert!(matches!(config.validate(), Err(SaddleError::Invalid(_))));
        }
    }

    #[test]
    fn a_zero_masked_seed_is_rejected() {
        let error = ArtSession::new(
            ArtConfig::default(),
            array![0.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![1.0, 0.0, 0.0],
        );
        assert!(error.is_err());
    }

    #[test]
    fn the_first_push_honors_the_mask() {
        let mut session = ArtSession::new(
            ArtConfig::default(),
            array![-1.0, 0.2, -0.3],
            array![1.0, 4.0, 9.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap();
        let surface = Well {
            calls: AtomicUsize::new(0),
        };
        let report = session.step(&surface).unwrap();
        assert_eq!(report.phase, ArtPhase::Perpendicular);
        assert!((session.position()[1] - 0.2).abs() < 0.000_001);
        assert!((session.position()[2] + 0.3).abs() < 0.000_001);
        assert!(session.position()[0] > -1.0);
        assert_eq!(session.force_calls().len(), 1);
        assert_eq!(surface.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn identical_endpoints_are_a_revert_and_not_a_completed_exit() {
        let reactant = array![0.0, 0.0, 0.0];
        let (status, reversibility) = connection_verdict(
            reactant.view(),
            reactant.view(),
            reactant.view(),
            1.0,
            0.0,
            0.0,
            0.08,
            false,
        )
        .unwrap();
        assert_eq!(status, ArtStatus::Reverted);
        assert!(reversibility.reverted);
        assert_ne!(status, ArtStatus::Connected);
    }

    #[test]
    fn either_quench_may_recover_the_reactant() {
        let reactant = array![0.0, 0.0, 0.0];
        let product = array![2.0, 0.0, 0.0];
        let (status, reversibility) = connection_verdict(
            reactant.view(),
            product.view(),
            reactant.view(),
            1.0,
            -1.0,
            -1.0,
            0.2,
            false,
        )
        .unwrap();
        assert_eq!(status, ArtStatus::Connected);
        assert!(reversibility.returns_to_reactant);
        assert!(reversibility.product_is_distinct);
        assert!(reversibility.downhill);
        assert!(!reversibility.reverted);
        assert_ne!(status, ArtStatus::Irreversible);
    }

    #[test]
    fn rotated_and_permuted_copies_match() {
        let reference = array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.2, 0.7, 0.15];
        let mut rotated = Array1::zeros(9);
        for atom in 0..3 {
            let x = reference[3 * atom];
            let y = reference[3 * atom + 1];
            let z = reference[3 * atom + 2];
            rotated[3 * atom] = -y;
            rotated[3 * atom + 1] = x;
            rotated[3 * atom + 2] = z;
        }
        let rigid = structure_distance(reference.view(), rotated.view(), false).unwrap();
        assert!(rigid < 0.000_1, "{rigid}");
        let mut swapped = reference.clone();
        for axis in 0..3 {
            let first = swapped[axis];
            swapped[axis] = swapped[3 + axis];
            swapped[3 + axis] = first;
        }
        let labeled = structure_distance(reference.view(), swapped.view(), false).unwrap();
        assert!(labeled > 0.1, "{labeled}");
        let identical = structure_distance(reference.view(), swapped.view(), true).unwrap();
        assert!(identical < 0.000_1, "{identical}");
    }

    #[test]
    fn the_double_well_connects_and_exposes_every_force_call() {
        let surface = Well {
            calls: AtomicUsize::new(0),
        };
        let mut session = ArtSession::new(
            well_config(),
            array![-1.0, 0.0, 0.0],
            array![1.0, 0.1, 0.0],
            full_mask(3),
        )
        .unwrap();
        let report = session.run(&surface, 80).unwrap();
        assert_eq!(report.status, ArtStatus::Connected, "{report:?}");
        assert_eq!(report.saddle_index, Some(1));
        let outcome = session.outcome();
        assert_eq!(
            outcome.force_calls.len(),
            surface.calls.load(Ordering::Relaxed)
        );
        assert!(
            outcome
                .force_calls
                .iter()
                .any(|call| call.phase == ArtPhase::Push)
        );
        assert!(
            outcome
                .force_calls
                .iter()
                .any(|call| call.phase == ArtPhase::Perpendicular)
        );
        assert!(
            outcome
                .force_calls
                .iter()
                .any(|call| call.phase == ArtPhase::Lanczos)
        );
        assert!(
            outcome
                .force_calls
                .iter()
                .any(|call| call.phase == ArtPhase::Climb)
        );
        assert!(
            outcome
                .force_calls
                .iter()
                .any(|call| call.phase == ArtPhase::Certify)
        );
        assert!(
            outcome
                .force_calls
                .iter()
                .any(|call| call.phase == ArtPhase::Quench)
        );
        let order = [
            ArtPhase::Push,
            ArtPhase::Perpendicular,
            ArtPhase::Lanczos,
            ArtPhase::Climb,
            ArtPhase::Certify,
            ArtPhase::PushOver,
            ArtPhase::Quench,
            ArtPhase::Report,
        ];
        let mut cursor = 0;
        for phase in order {
            let found = outcome.phases[cursor..]
                .iter()
                .position(|item| *item == phase)
                .unwrap_or_else(|| panic!("missing {phase:?} in {:?}", outcome.phases));
            cursor += found;
        }
        let reversibility = outcome.reversibility.unwrap();
        assert!(reversibility.returns_to_reactant);
        assert!(reversibility.product_is_distinct);
        assert!(reversibility.downhill);
        assert!(!reversibility.reverted);
        let forward = outcome.minimum_forward.unwrap();
        assert!(forward[0] > 0.5, "{forward:?}");
    }

    #[test]
    fn a_shared_basin_fails_the_event() {
        let surface = Well {
            calls: AtomicUsize::new(0),
        };
        let config = well_config().with_basin_rmsd(3.0);
        let mut session = ArtSession::new(
            config,
            array![-1.0, 0.0, 0.0],
            array![1.0, 0.1, 0.0],
            full_mask(3),
        )
        .unwrap();
        let report = session.run(&surface, 80).unwrap();
        assert_eq!(report.status, ArtStatus::Reverted, "{report:?}");
        let reversibility = report.reversibility.unwrap();
        assert!(reversibility.reverted);
        assert_ne!(report.status, ArtStatus::Connected);
        let forward = session.outcome().minimum_forward.unwrap();
        let back = session.outcome().minimum_back.unwrap();
        assert!(forward[0] > 0.5 && back[0] < -0.5, "{forward:?} {back:?}");
    }

    #[test]
    fn repeated_runs_match_call_for_call() {
        let once = run_well();
        let twice = run_well();
        assert_eq!(once.len(), twice.len());
        for (left, right) in once.iter().zip(&twice) {
            assert_eq!(left.phase, right.phase);
            assert_eq!(left.energy.to_bits(), right.energy.to_bits());
            assert_eq!(left.max_force.to_bits(), right.max_force.to_bits());
            assert_eq!(left.position, right.position);
        }
    }

    fn run_well() -> Vec<ArtForceCall> {
        let surface = Well {
            calls: AtomicUsize::new(0),
        };
        let mut session = ArtSession::new(
            well_config(),
            array![-1.0, 0.0, 0.0],
            array![1.0, 0.1, 0.0],
            full_mask(3),
        )
        .unwrap();
        session.run(&surface, 80).unwrap();
        session.force_calls()
    }
}
