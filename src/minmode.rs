//! Minimum-mode saddle search: find the lowest curvature direction,
//! invert the force along it, take one step. Stepping, like the band.
//!
//! A session is long-lived: [`MinModeSession::set_position`] moves it
//! to a new point (with the host's gradient there, when the host has
//! one) and keeps the mode and the dimer's Polak-Ribiere rotation
//! history as the next rotation's seed, and
//! [`MinModeSession::estimate_mode`] refreshes the mode without a
//! translation, for a host that climbs with its own optimizer.
//! [`MinModeSession::set_lbfgs_rotate`] replaces that Polak-Ribiere
//! plane with L-BFGS in the tangent space (Kästner and Sherwood, J.
//! Chem. Phys. 128, 014106 (2008), doi:10.1063/1.2815812). It is off
//! by default.

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::kappa::{Complement, KappaDimerConfig, kappa_dimer_force};
use crate::saddle_index::{
    Certificate, Probe, SaddleIndex, SaddleIndexCheck, certify, escape_push,
};

#[path = "rotation_lbfgs.rs"]
mod rotation_lbfgs;

/// The caller's surface for a single geometry.
pub trait PointSurface: Sync {
    /// Energy and Cartesian gradient at `x`, in the host's units.
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError>;

    /// Independent geometries, one per row, with results in the same order.
    /// Providers may execute the geometries together or concurrently.
    fn eval_batch(&self, points: ArrayView2<f64>) -> Result<Vec<(f64, Array1<f64>)>, SaddleError> {
        points
            .rows()
            .into_iter()
            .map(|row| self.eval(row))
            .collect()
    }

    /// Optional analytic Cartesian Hessian action. None selects gradient differences.
    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        _v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(None)
    }

    /// Rigid or constrained Cartesian directions, one direction per row.
    /// Molecular callers supply translations and rotations; fixed substrates
    /// supply only their actual symmetries. These are evaluated at the query.
    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(Array2::zeros((0, x.len())))
    }

    /// Energy and Cartesian gradient in a periodic cell, row-major 3x3.
    ///
    /// The energy must follow every entry of the cell, and so must
    /// [`Self::cauchy_stress`]. Held diagonal strains with free shears
    /// change only the tilts, at a fixed diagonal and volume, so a box
    /// refreshed only when lx, ly, lz or the volume change never sees
    /// that search move, and a frozen or fully free run does not reveal
    /// it. The finite-difference stress passes general cells, not only
    /// the lower-triangular ones a search steps through. Default
    /// ignores the cell and calls [`Self::eval`].
    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        _cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.eval(x)
    }

    /// Optional `dE/dC`, row-major 3x3. `None` means the session
    /// finite-differences [`Self::eval_in_cell`].
    fn cell_grad(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let _ = (x, cell);
        Ok(None)
    }

    /// Cauchy stress, row-major 3×3.
    ///
    /// `σ = (1/V) ∂E/∂ε` for the right strain `h ← h (I + ε)` at fixed
    /// fractional coordinates, in the energy and length units of
    /// [`Self::eval_in_cell`]. It is positive in tension, so a
    /// stretched cell has a positive trace and a pressure tensor is
    /// `-σ`. LAMMPS reports `pxx pyy pzz pxy pxz pyz`, a pressure in
    /// bar under `units metal`, so `σ_ab = -p_ab / nktv2p` with
    /// `nktv2p = 1.6021765e6`, and its off-diagonal order is not the
    /// xx, yy, zz, yz, xz, xy of [`crate::STRAIN_VOIGT`]. The
    /// generalized cell force is `-(V/J) σ`: a flipped sign walks
    /// every free strain uphill. The virial is `V σ`. `None` asks the
    /// solid-state searches to finite-difference
    /// [`Self::eval_in_cell`], and [`crate::check_cauchy_stress`]
    /// compares a supplied tensor with that difference entry by entry.
    /// Held diagonals with free shears read only `σ_yx`, `σ_zx` and
    /// `σ_zy`, so agreement on the diagonal says nothing about that
    /// search. A frozen cell never calls this: that search is
    /// [`crate::MinModeSession`] on [`Self::eval`].
    fn cauchy_stress(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let _ = (x, cell);
        Ok(None)
    }
}

/// Finite difference behind every Hessian action.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FiniteDifference {
    /// `(g(x + dr v) - g(x)) / dr`: one gradient per action (the centre
    /// is known), error `dr/2 T[v,v,v]`.
    #[default]
    Forward,
    /// `(g(x + dr v) - g(x - dr v)) / (2 dr)`: two gradients per action,
    /// error `dr^2/6 Q[v,v,v,v]`.
    Central,
}

/// How the lowest mode is estimated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MinModeKind {
    /// Finite-difference dimer (Henkelman-Jonsson 1999) rotated by the
    /// modified Newton step of Heyden, Bell, and Keil (2005) and
    /// Kastner and Sherwood (2008) along conjugate rotation
    /// directions: one gradient per rotation, the gradient at the
    /// rotated dimer is extrapolated.
    #[default]
    Dimer,
    /// Lanczos on finite-difference Hessian actions: a Krylov
    /// subspace per compute, one gradient per action. The estimate
    /// stops when the lowest Ritz pair's residual divided by the
    /// absolute Ritz value meets `rotation_tol`, together with the
    /// relative change of that value. A Ritz value whose magnitude is
    /// at most 1e-14 of the curvature scale, the largest Hessian
    /// action norm the estimate has formed, divides the residual by
    /// that scale. The torque window stops once two vectors leave
    /// that same ratio under 1, dividing by the same scale when the
    /// Ritz value is that small.
    Lanczos,
}

impl MinModeKind {
    /// The discriminant used by the C and Python interfaces.
    pub fn to_abi(self) -> i32 {
        match self {
            Self::Dimer => 0,
            Self::Lanczos => 1,
        }
    }
}

/// Controls the opt-in ART nouveau push--relax protocol.
///
/// The defaults reproduce the generic parts of the published ARTn
/// workflow rather than any system-specific atom selection: delay the
/// first Lanczos solve for three pushes, take six perpendicular
/// relaxation steps while leaving the basin, use five during activation,
/// cap a mode-following push at 0.2 Angstrom, and use 16 Lanczos actions.
/// Jay et al., J. Chem. Theory Comput. 16, 6726 (2020),
/// doi:10.1021/acs.jctc.0c00541, "Previous ARTn workflow" and
/// "ARTn parameters".
///
/// The initial `mode` passed to [`MinModeSession::new`] is the random
/// basin-exit direction. ARTn keeps it for basin pushes while Lanczos,
/// once enabled, is warm-started from the previous lowest direction.
/// Length and curvature defaults use Angstrom, eV/Angstrom, and
/// eV/Angstrom squared; callers using other units must rescale them.
///
/// Non-exhaustive: start from [`ArtNConfig::default`] and chain the
/// `with_*` setters.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ArtNConfig {
    /// Start Lanczos after this many random-direction pushes.
    pub lanczos_start: usize,
    /// Enter activation once the lowest eigenvalue is below this value.
    pub eigenvalue_threshold: f64,
    /// Length of each random-direction basin push.
    pub basin_push: f64,
    /// Perpendicular steepest-descent steps after each basin push.
    pub basin_relaxations: usize,
    /// Perpendicular steepest-descent steps after each activation push.
    pub activation_relaxations: usize,
    /// Scale multiplying the projected gradient during relaxation.
    pub relaxation_step: f64,
    /// Most Lanczos Hessian actions per geometry.
    pub lanczos_actions: usize,
    /// Lower bound on `|lambda|` in the activation Newton step.
    pub curvature_floor: f64,
}

with_setters!(ArtNConfig {
    with_lanczos_start => lanczos_start: val usize,
    with_eigenvalue_threshold => eigenvalue_threshold: val f64,
    with_basin_push => basin_push: val f64,
    with_basin_relaxations => basin_relaxations: val usize,
    with_activation_relaxations => activation_relaxations: val usize,
    with_relaxation_step => relaxation_step: val f64,
    with_lanczos_actions => lanczos_actions: val usize,
    with_curvature_floor => curvature_floor: val f64,
});

impl Default for ArtNConfig {
    fn default() -> Self {
        Self {
            lanczos_start: 3,
            eigenvalue_threshold: -0.01,
            basin_push: 0.2,
            basin_relaxations: 6,
            activation_relaxations: 5,
            relaxation_step: 0.1,
            lanczos_actions: 16,
            curvature_floor: 0.5,
        }
    }
}

pub(crate) fn validate_artn(kind: MinModeKind, artn: &ArtNConfig) -> Result<(), SaddleError> {
    if kind != MinModeKind::Lanczos {
        return Err(SaddleError::Invalid(
            "ARTn requires the Lanczos minimum-mode estimator".into(),
        ));
    }
    if !(artn.eigenvalue_threshold.is_finite() && artn.eigenvalue_threshold <= 0.0) {
        return Err(SaddleError::Invalid(
            "ARTn eigenvalue threshold must be finite and non-positive".into(),
        ));
    }
    if !(artn.basin_push.is_finite() && artn.basin_push > 0.0) {
        return Err(SaddleError::Invalid(
            "ARTn basin push must be positive and finite".into(),
        ));
    }
    if !(artn.relaxation_step.is_finite() && artn.relaxation_step > 0.0) {
        return Err(SaddleError::Invalid(
            "ARTn relaxation step must be positive and finite".into(),
        ));
    }
    if artn.lanczos_actions == 0 {
        return Err(SaddleError::Invalid(
            "ARTn Lanczos actions must be at least 1".into(),
        ));
    }
    if !(artn.curvature_floor.is_finite() && artn.curvature_floor > 0.0) {
        return Err(SaddleError::Invalid(
            "ARTn curvature floor must be positive and finite".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
/// Rotation, Lanczos, and translation controls for [`MinModeSession`].
///
/// Non-exhaustive: start from [`MinModeConfig::default`] and chain the
/// `with_*` setters.
#[non_exhaustive]
pub struct MinModeConfig {
    /// Which estimator produces the lowest mode.
    pub kind: MinModeKind,
    /// Dimer separation / finite-difference displacement, Angstrom.
    /// The curvature is a forward difference against the centre
    /// gradient, with truncation error `dr/2 * T[v,v,v]` (`T` the third
    /// derivative) and noise error `2 eps_g / dr` for gradient noise
    /// `eps_g`; see `validation/fd_curvature.py`.
    pub dr: f64,
    /// Forward (default) or central difference for the actions.
    pub difference: FiniteDifference,
    /// Rotation stop. The default is 1e-4.
    ///
    /// A dimer reads this as the rotational force
    /// `|H v - (v.H v) v|`, in the host's curvature units. At the
    /// default and any looser value, a dimer with its angle test
    /// disabled also stops after one rotation once that force is under one.
    /// The improved dimer default uses the angle test instead. A tighter
    /// value keeps driving the residual, and the torque window stays out
    /// of that loop.
    /// [`MinModeSession::set_torque_window`] turns the window on or off
    /// whatever this value is.
    ///
    /// Lanczos reads the same number as a dimensionless tolerance. The
    /// estimate stops when the lowest Ritz residual divided by the
    /// absolute Ritz value is at most this, and, with no host relative
    /// tolerance, when the relative change of that Ritz value is too.
    /// A Ritz value whose magnitude is at most 1e-14 of the estimate's
    /// curvature scale, the largest Hessian action norm it has formed,
    /// has no magnitude of its own, so the residual and the change are
    /// divided by that scale instead and the quotient is never formed.
    /// Rescaling the energy or length units leaves every one of these
    /// decisions unchanged. Once two vectors are in hand, the torque
    /// window stops when that same ratio is under 1.
    pub rotation_tol: f64,
    /// Dimer rotation also stops when the optimal rotation angle of an
    /// iteration falls under this many radians. The default is eOn's
    /// `converged_angle`, five degrees
    /// ([`DEFAULT_ROTATION_ANGLE_TOL`]; Kästner and Sherwood, J. Chem.
    /// Phys. 128, 014106 (2008), doi:10.1063/1.2815812). Zero disables
    /// the angle test and restores the inferred classic torque window
    /// at the stock residual; see `rotation_finished`. Under a
    /// `rotation_tol` tighter than the stock 1e-4 the angle test applies
    /// only once [`MinModeConfig::with_rotation_angle_tol`] has set it,
    /// so the default drives the residual down to `rotation_tol`.
    pub rotation_angle_tol: f64,
    /// Whether [`MinModeConfig::with_rotation_angle_tol`] set
    /// `rotation_angle_tol`.
    pub(crate) rotation_angle_tol_set: bool,
    /// Most dimer rotations, or Lanczos actions, in one estimate.
    pub max_rotations: usize,
    /// Krylov dimension for [`MinModeKind::Lanczos`].
    pub krylov_dim: usize,
    /// Translation stops when max|F| falls under this and the
    /// curvature along the lowest mode is negative.
    pub force_tol: f64,
    /// Largest translation, 0.2 by default. The stepper caps each atom's
    /// move at this length. While the curvature along the mode is not
    /// negative, dimer and Lanczos take one step along the mode instead,
    /// scaled so its largest atom moves this length. Under
    /// [`MinModeSession::set_cg_translate`] that step has this global
    /// Euclidean norm, then is shortened so no atom moves farther.
    pub max_move: f64,
    /// FIRE's opening time step for the translation, 1 by default. The
    /// integrator's largest time step is 2.5 times this. With unit mass
    /// the first move from rest is `F dt^2`, so the step has units of
    /// length over the square root of energy. The stalled-FIRE probe
    /// reads the stability bound `|λ| > 4 / dt^2` from it. L-BFGS does
    /// not read it.
    pub time_step: f64,
    /// Which rgmin stepper advances the assembled force.
    pub method: Method,
    /// Seed an empty translation from the mode curvature.
    ///
    /// Off by default, and the configured stepper is unchanged. On, the
    /// first empty L-BFGS step uses `H0 = 1/|λ|` and the first FIRE
    /// displacement uses that same scale. Later steps are the stepper's
    /// own updates, still capped by `max_move`. eOn 3.6.0 does the L-BFGS
    /// half in `LBFGS::getStep` from `MinModeObjectiveFunction::knownCurvature`,
    /// which returns `-λ` once the mode is negative
    /// (`https://github.com/TheochemUI/eOn/blob/v3.6.0/client/MinModeSaddleSearch.cpp`).
    pub seed_curvature: bool,
    /// Certify the converged point: its lowest curvatures off the
    /// directions [`PointSurface::excluded_modes`] names, by block
    /// Lanczos on finite-difference Hessian actions seeded with the
    /// search's mode (see [`mod@crate::saddle_index`]). The index is
    /// [`MinModeReport::saddle_index`] and the actions count in
    /// [`MinModeReport::evaluations`]. On by default; the search itself
    /// steps as without it. With [`SaddleIndexCheck::escape`] set, a
    /// point of index two or more is pushed off its extra negative
    /// modes and the search starts again there along the negative mode
    /// closest to its own. `None` skips the check.
    pub saddle_check: Option<SaddleIndexCheck>,
    /// ART nouveau push--relax controls. `None` keeps ordinary
    /// minimum-mode following byte-for-byte unchanged.
    pub artn: Option<ArtNConfig>,
}

with_setters!(MinModeConfig {
    with_kind => kind: val MinModeKind,
    with_dr => dr: val f64,
    with_difference => difference: val FiniteDifference,
    with_rotation_tol => rotation_tol: val f64,
    with_max_rotations => max_rotations: val usize,
    with_krylov_dim => krylov_dim: val usize,
    with_force_tol => force_tol: val f64,
    with_max_move => max_move: val f64,
    with_time_step => time_step: val f64,
    with_method => method: val Method,
    with_seed_curvature => seed_curvature: val bool,
    with_saddle_check => saddle_check: opt SaddleIndexCheck,
});

impl MinModeConfig {
    /// Set [`MinModeConfig::rotation_angle_tol`]. The value applies under
    /// any `rotation_tol`, including a tighter one than the stock 1e-4.
    #[must_use]
    pub fn with_rotation_angle_tol(mut self, rotation_angle_tol: f64) -> Self {
        self.rotation_angle_tol = rotation_angle_tol;
        self.rotation_angle_tol_set = true;
        self
    }

    /// Refuse a configuration no minimum-mode search can run.
    ///
    /// Returns [`SaddleError::Invalid`] when the finite-difference step or
    /// [`Self::time_step`] is not finite and positive, the force or a
    /// rotation tolerance is negative or not finite, the rotation or Krylov
    /// budget is zero, [`Self::max_move`] is not finite, the saddle check
    /// or ART nouveau block is refused, or [`Self::method`] tests the
    /// oracle value. [`MinModeSession::new`] calls this; nothing is clamped.
    pub(crate) fn validate(&self) -> Result<(), SaddleError> {
        if !(self.dr.is_finite() && self.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        if !(self.force_tol.is_finite() && self.force_tol >= 0.0) {
            return Err(SaddleError::Invalid(
                "force tolerance must be finite and non-negative".into(),
            ));
        }
        if !(self.rotation_tol.is_finite()
            && self.rotation_tol >= 0.0
            && self.rotation_angle_tol.is_finite()
            && self.rotation_angle_tol >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "rotation tolerances must be finite and non-negative".into(),
            ));
        }
        if self.max_rotations == 0 {
            return Err(SaddleError::Invalid(
                "max_rotations must be at least 1".into(),
            ));
        }
        if self.krylov_dim == 0 {
            return Err(SaddleError::Invalid(
                "Lanczos krylov dimension must be at least 1".into(),
            ));
        }
        if !self.max_move.is_finite() {
            return Err(SaddleError::Invalid(
                "max_move must be finite; a non-positive value leaves the step uncapped".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "FIRE time step must be finite and positive".into(),
            ));
        }
        if let Some(check) = &self.saddle_check {
            check.validate()?;
        }
        if let Some(artn) = &self.artn {
            validate_artn(self.kind, artn)?;
        }
        crate::band::check_force_driven(&self.method)
    }

    /// Enable the ART nouveau push--relax protocol.
    #[must_use]
    pub fn with_artn(mut self, artn: ArtNConfig) -> Self {
        self.artn = Some(artn);
        self
    }
}

impl Default for MinModeConfig {
    fn default() -> Self {
        Self {
            kind: MinModeKind::Dimer,
            dr: 1e-3,
            difference: FiniteDifference::Forward,
            rotation_tol: STOCK_ROTATION_TOL,
            rotation_angle_tol: DEFAULT_ROTATION_ANGLE_TOL,
            rotation_angle_tol_set: false,
            max_rotations: 20,
            krylov_dim: 12,
            force_tol: 1e-3,
            max_move: 0.2,
            time_step: FIRE_OPENING_DT,
            method: Method::fire(rgmin::FireKind::V2),
            seed_curvature: false,
            saddle_check: Some(SaddleIndexCheck::default()),
            artn: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Outcome of one [`MinModeSession`] step.
pub enum MinModeStatus {
    /// Either the convergence force exceeds `force_tol` or the lowest curvature is non-negative.
    Running,
    /// The convergence force is at or below `force_tol` and the lowest curvature is negative.
    Converged,
}

#[derive(Clone, Debug)]
/// Diagnostics returned by [`MinModeSession::step`] and [`MinModeSession::run`].
pub struct MinModeReport {
    /// Whether this step is still running or has converged.
    pub status: MinModeStatus,
    /// Largest absolute force component, in the host's force units.
    pub max_force: f64,
    /// Curvature along the mode, energy per length squared.
    pub curvature: f64,
    /// Rotation iterations (dimer) or Hessian actions (Lanczos).
    pub rotations: usize,
    /// Number of steps taken, including steps that leave the atoms in place.
    pub iteration: usize,
    /// Surface evaluations during this call, centre included.
    pub evaluations: usize,
    /// Saddle index of the converged point
    /// ([`MinModeConfig::saddle_check`]). `None` while the search runs
    /// or with the check off. A step that pushes off a saddle of higher
    /// index reports that index and runs on.
    pub saddle_index: Option<SaddleIndex>,
}

/// A refreshed lowest mode at the current point.
#[derive(Clone, Debug)]
pub struct ModeEstimate {
    /// Estimated lowest-curvature direction. A stored mode is normalized.
    pub mode: Array1<f64>,
    /// Curvature along the mode, energy per length squared.
    pub curvature: f64,
    /// Rotation iterations (dimer) or Hessian actions (Lanczos).
    pub rotations: usize,
    /// Surface evaluations, the centre included when it was neither
    /// cached nor supplied by the host.
    pub evaluations: usize,
}

/// Rotation of the mode between two steps past which the translation
/// optimizer's memory is dropped.
///
/// The effective gradient is `R g` with `R = I - 2 τ τ'`. Goswami, Gunde,
/// and Jónsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, show
/// that this reflector drives toward the same saddle while the mode stays
/// inside a 45 degree cone (`|τ · τ'| > 1/√2`). Kästner and Sherwood,
/// J. Chem. Phys. 128, 014106 (2008), doi:10.1063/1.2815812, keep the
/// L-BFGS pairs across that continuous turn: the superlinear step is the
/// history. A few degrees of the same eigenvector (the lowest mode turns
/// by about 6–10 degrees per step while a search is still leaving a
/// minimum) is not a new mode. Dropping the pairs there, and zeroing the
/// FIRE velocity, restarts the step on every iteration.
///
/// Past 45 degrees the inverted force can point away from the saddle the
/// pairs were built for, so the memory goes. Only the cached evaluation
/// is dropped inside the cone (`validation/mode_reset.py`:
/// `|R' - R|_2 = 2 |sin θ|`).
pub const MODE_RESET_ANGLE: f64 = 0.25 * PI;

/// Improved-dimer angle stop, eOn `converged_angle` of five degrees.
/// Kästner and Sherwood, J. Chem. Phys. 128, 014106 (2008),
/// doi:10.1063/1.2815812. Heyden, Bell, and Keil, J. Chem. Phys. 123,
/// 224101 (2005), doi:10.1063/1.2104507.
pub const DEFAULT_ROTATION_ANGLE_TOL: f64 = 5.0 * PI / 180.0;

/// Stock rotational-force residual. At this value and looser, the dimer
/// and Lanczos use eOn's classic torque window. A stricter residual
/// keeps iterating.
const STOCK_ROTATION_TOL: f64 = 1e-4;

/// eOn `dimer_torque_max`. After one rotation, a rotational force under
/// this is finished (classic dimer, `torque_min` .. `torque_max`,
/// `rotations_min` = 1). Lanczos reads it as the bound on the
/// dimensionless residual once it holds two vectors: the Ritz residual
/// divided by the absolute Ritz value, or by the curvature scale, the
/// largest Hessian action norm the estimate has formed, when the Ritz
/// value is at most 1e-14 of that scale.
const TORQUE_MAX: f64 = 1.0;

/// Scale of the step that replaces a capped FIRE move which left the
/// negative-curvature neighborhood.
///
/// The reflected force is not a potential gradient (Goswami, Gunde, and
/// Jónsson, Front. Chem. (2026), doi:10.3389/fchem.2026.1807063, Eq. 9).
/// From rest, FIRE 2.0 moves by `F dt^2`: a zero power inside the
/// opening delay leaves `dt` unchanged (Guénolé, Nöhring, Vaid, Houllé,
/// Xie, Prakash, and Bitzek, Comput. Mater. Sci. 175, 109584 (2020),
/// doi:10.1016/j.commatsci.2020.109584, Algorithm 2; the initial `dt` is
/// [`MinModeConfig::time_step`]). When that move exceeds `max_move` the per-atom
/// cap truncates it, and the truncated move can leave the
/// negative-curvature neighborhood. The replacement, taken from the last
/// point that was still negative, is the empty-memory L-BFGS step
/// `0.01 F`. A capped step whose arrival stays negative is kept, so a
/// basin wider than `max_move` follows the integrator.
const FIRE_EMPTY_SCALE: f64 = 0.01;

/// Default [`MinModeConfig::time_step`].
const FIRE_OPENING_DT: f64 = 1.0;

/// Each replaced capped FIRE move restarts the integrator with `dt`
/// scaled by FIRE's `f_dec`, the response to a missed step that the
/// opening delay of FIRE 2.0 withholds. Without it the restart from rest
/// repeats the same capped move out of the neighborhood.
const FIRE_RETREAT_DT_SCALE: f64 = 0.5;

/// Whether a trial energy rose, and the step length to try next.
///
/// `raised` is the conjugate-gradient translator's decision. `step` is
/// the length of the next trial, in the host's length unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaisedStep {
    /// True when the trial energy is more than `1e-4` above the previous energy.
    pub raised: bool,
    /// Half the trial length when `raised` is true, otherwise that length.
    pub step: f64,
}

/// Half the step when the trial energy rises.
///
/// `previous` is the energy before the step and `trial` the energy at
/// the trial point, both in the host's energy unit. `step` is that
/// trial's length in the host's length unit. The conjugate-gradient
/// translator treats a trial more than `1e-4` above `previous` as a
/// rise and tries half the length. This is that comparison and that
/// new length. A missing previous energy is a non-finite `previous`,
/// which is not a rise.
///
/// Nothing here evaluates a surface.
///
/// ```
/// use rgsaddle::halve_raised_energy;
/// let kept = halve_raised_energy(1.0, 1.0, 0.4);
/// assert!(!kept.raised);
/// assert_eq!(kept.step.to_bits(), 0.4_f64.to_bits());
/// let halved = halve_raised_energy(1.0, 1.0002, 0.4);
/// assert!(halved.raised);
/// assert_eq!(halved.step.to_bits(), (0.4_f64 * 0.5).to_bits());
/// ```
pub fn halve_raised_energy(previous: f64, trial: f64, step: f64) -> RaisedStep {
    let raised = trial > previous + 1e-4;
    RaisedStep {
        raised,
        step: if raised { step * 0.5 } else { step },
    }
}

/// A first-order saddle has a vanishing force and one negative
/// curvature. A vanishing force alone also describes a minimum, which
/// the inverted-force step then leaves; the gate therefore requires
/// `curvature < 0` alongside `max_force <= force_tol`.
fn converged(max_force: f64, curvature: f64, force_tol: f64) -> bool {
    max_force <= force_tol && curvature < 0.0
}

fn normalize(mut v: Array1<f64>) -> Array1<f64> {
    let n = v.dot(&v).sqrt();
    if n > 1e-14 {
        v /= n;
    }
    v
}

/// A host mode must admit the unit normalization used by the curvature
/// and reflected-force contracts. Reject invalid norms before division.
fn checked_mode(mode: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let norm = mode.dot(&mode).sqrt();
    if !(norm.is_finite() && norm > 1e-14) {
        return Err(SaddleError::Invalid(
            "mode must have a finite norm above 1e-14".into(),
        ));
    }
    Ok(mode / norm)
}

/// The surface, counting the calls one compute makes and checking
/// what comes back.
struct Counted<'a, S: PointSurface> {
    surface: &'a S,
    calls: &'a AtomicUsize,
    space: Option<&'a Complement>,
    /// The displaced point of a finite-difference action.
    shifted: Mutex<Array1<f64>>,
}

impl<S: PointSurface> Counted<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (e, g) = self.surface.eval(x)?;
        if g.len() != x.len() {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        if !e.is_finite() || !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("min-mode gradient"));
        }
        Ok((e, g))
    }

    /// Gradient at `x + step * unit`.
    fn gradient_along(
        &self,
        x: ArrayView1<f64>,
        unit: ArrayView1<f64>,
        step: f64,
    ) -> Result<Array1<f64>, SaddleError> {
        let mut shifted = self.shifted.lock().unwrap_or_else(|e| e.into_inner());
        if shifted.len() != x.len() {
            *shifted = Array1::zeros(x.len());
        }
        Zip::from(&mut *shifted)
            .and(x)
            .and(unit)
            .for_each(|s, &x, &u| *s = x + u * step);
        Ok(self.eval(shifted.view())?.1)
    }

    /// Finite-difference Hessian action `H unit`: forward against the
    /// centre gradient (one call) or central (two calls).
    fn action(
        &self,
        x: ArrayView1<f64>,
        g0: ArrayView1<f64>,
        unit: ArrayView1<f64>,
        config: &MinModeConfig,
    ) -> Result<Array1<f64>, SaddleError> {
        let lifted;
        let direction = if let Some(space) = self.space {
            lifted = space.lift(unit);
            lifted.view()
        } else {
            unit
        };
        let action = if let Some(hv) = self.surface.hessian_vector(x, direction)? {
            if hv.len() != x.len() {
                return Err(SaddleError::Shape(
                    "surface Hessian action length changed".into(),
                ));
            }
            if !hv.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("surface Hessian action"));
            }
            hv
        } else {
            let dr = config.dr;
            match config.difference {
                FiniteDifference::Forward => {
                    let mut action = self.gradient_along(x, direction, dr)?;
                    Zip::from(&mut action)
                        .and(g0)
                        .for_each(|a, &g| *a = (*a - g) / dr);
                    action
                }
                FiniteDifference::Central => {
                    let mut action = self.gradient_along(x, direction, dr)?;
                    let minus = self.gradient_along(x, direction, -dr)?;
                    let width = 2.0 * dr;
                    Zip::from(&mut action)
                        .and(&minus)
                        .for_each(|a, &m| *a = (*a - m) / width);
                    action
                }
            }
        };
        Ok(match self.space {
            Some(space) => space.reduce(action.view()),
            None => action,
        })
    }
}

/// Dimer rotation by the modified Newton step.
///
/// With `h(n) = (g(x + dr n) - g0) / dr`, the curvature along the unit
/// vector `n` is `C = h.n` and its gradient on the sphere is
/// `2 (h - C n)`. Rotating in the plane of `n` and a unit `theta`
/// perpendicular to it, `n(phi) = n cos phi + theta sin phi`, a
/// quadratic surface gives `C(phi) = a0/2 + a1 cos 2phi + b1 sin 2phi`
/// with `b1 = h.theta`. One gradient at a trial angle `phi1` fixes
/// `a1`, the minimizing `phi` follows in closed form, and the gradient
/// at the rotated dimer is the linear combination
/// `g(phi) = g0 + sin(phi1 - phi)/sin(phi1) (g1 - g0)
///          + sin(phi)/sin(phi1) (g1' - g0)`, which for the Hessian
/// action is `h(phi) = w_old h + w_trial h1` with the same weights,
/// exact on a quadratic surface (`validation/dimer_rotation.py`), so
/// the next iteration starts without a gradient call. Rotation
/// directions are Polak-Ribiere conjugate on the rotational force,
/// with the previous direction carried along the rotation. With
/// [`MinModeSession::set_lbfgs_rotate`] the plane is the L-BFGS step
/// instead, and that memory restarts at each estimate.
/// Floor of the dimer's trial angle, radians. The Kastner-Sherwood
/// guess `phi1 = atan(|b1| / |C|) / 2` falls under it once the
/// rotational force is under `2e-4 |C|`.
const MIN_TRIAL_ANGLE: f64 = 1e-4;

/// Largest weight `|sin(phi1 - phi)| / sin(phi1)` or
/// `|sin(phi)| / sin(phi1)` with which `rotate_dimer` extrapolates the
/// measured actions from a trial angle on its floor. Past it the rotated
/// dimer is measured (`validation/dimer_extrapolation.py`).
const MAX_EXTRAPOLATION: f64 = 8.0;

/// Polak-Ribiere state for dimer rotations.
///
/// The rotational objective moves smoothly with the dimer centre, so the
/// previous force and direction remain useful after a translation. Keeping
/// them makes the first rotation at the new centre conjugate instead of
/// restarting as steepest descent. The vectors live in Cartesian space;
/// the session maps them through any excluded-mode quotient.
#[derive(Clone, Debug, Default)]
struct DimerRotationHistory {
    force: Option<Array1<f64>>,
    direction: Option<Array1<f64>>,
    centre: Option<Array1<f64>>,
}

impl DimerRotationHistory {
    fn clear(&mut self) {
        self.force = None;
        self.direction = None;
        self.centre = None;
    }

    fn retain_dimension(&mut self, dimension: usize) {
        if self
            .force
            .as_ref()
            .is_some_and(|force| force.len() != dimension)
            || self
                .direction
                .as_ref()
                .is_some_and(|direction| direction.len() != dimension)
        {
            self.clear();
        }
    }
}

/// Session-level stop controls of one dimer rotation.
#[derive(Clone, Copy, Debug)]
struct RotationStop {
    /// See [`MinModeSession::set_torque_window`].
    torque_window: Option<bool>,
    /// See [`MinModeSession::set_rotation_stop_length`]; 0 is off.
    length: f64,
    /// See [`MinModeSession::set_lbfgs_rotate`].
    lbfgs: bool,
}

/// Force error that the rotation still to come would leave in a
/// translation of `length`.
///
/// Kästner and Sherwood, J. Chem. Phys. 128, 014106 (2008),
/// doi:10.1063/1.2815812, estimate the remaining rotation angle from
/// the rotational force `F` and the curvature `C` as
/// `ω = atan(F / |C|) / 2`, the trial angle the [`MinModeKind::Dimer`]
/// rotation uses. A
/// translation of length `L` along a mode that is `ω` off moves by
/// `L |sin ω|` across the true mode, which a curvature of magnitude
/// `|C|` turns into a force error `|C| L |sin ω|`. The magnitude is the
/// larger of `|C|` and the two-mode estimate of the curvature after
/// the rotation, `C - F tan ω`: a dimer halfway between a negative and
/// a positive mode reads `C ≈ 0` while the rotation ahead is 45 degrees
/// (`validation/rotation_force_stop.py`).
pub fn rotation_force_error(force_norm: f64, curvature: f64, length: f64) -> f64 {
    let omega = 0.5 * force_norm.abs().atan2(curvature.abs());
    let rotated = curvature - force_norm.abs() * omega.tan();
    curvature.abs().max(rotated.abs()) * length * omega.sin().abs()
}

/// True when [`rotation_force_error`] is within `force_tol`. A
/// non-positive length or tolerance, or a non-finite input, is false.
fn rotation_force_converged(force_norm: f64, curvature: f64, length: f64, force_tol: f64) -> bool {
    if !(length.is_finite() && length > 0.0 && force_tol.is_finite() && force_tol > 0.0) {
        return false;
    }
    if !(force_norm.is_finite() && curvature.is_finite()) {
        return false;
    }
    rotation_force_error(force_norm, curvature, length) <= force_tol
}

fn rotate_dimer<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    mode: Array1<f64>,
    config: &MinModeConfig,
    stop: RotationStop,
    history: &mut DimerRotationHistory,
) -> Result<(Array1<f64>, f64, usize), SaddleError> {
    let mut n = normalize(mode);
    let mut h = surface.action(x, g0, n.view(), config)?;
    let mut curvature = crate::stencil::curvature_along(h.view(), n.view());
    let mut rotations = 0;
    // Rotational force (descent direction of the curvature on the
    // sphere) and the conjugate direction of the previous iteration,
    // including the last iteration at the previous translated centre.
    // A repeated estimate at one centre is an independent refinement:
    // only a new centre activates the carried conjugate direction.
    if history.centre.as_ref().is_some_and(|centre| centre == x) {
        history.force = None;
        history.direction = None;
    }
    history.retain_dimension(n.len());
    let mut force_prev = history.force.take();
    let mut dir_prev = history.direction.take();
    // One set of buffers for every iteration; the pairs kept as the
    // previous force and direction trade places with them.
    let dim = n.len();
    let mut force = Array1::zeros(dim);
    let mut dir = Array1::zeros(dim);
    let mut theta = Array1::zeros(dim);
    let mut n1 = Array1::zeros(dim);
    let mut n_new = Array1::zeros(dim);
    let mut theta_rotated = Array1::zeros(dim);
    let mut change = Array1::zeros(dim);
    let mut gradient = Array1::zeros(dim);
    let mut lbfgs = rotation_lbfgs::RotationLbfgs::new();
    let angle_stop = angle_stop_on(config);
    loop {
        let along = h.dot(&n);
        Zip::from(&mut force)
            .and(&h)
            .and(&n)
            .for_each(|f, &h, &n| *f = -(h - n * along));
        let force_norm = force.dot(&force).sqrt();
        if rotation_finished(config, stop.torque_window, force_norm, rotations)
            || rotation_force_converged(force_norm, curvature, stop.length, config.force_tol)
        {
            // A finished rotation keeps its force for the next centre.
            // The cap leaves the previous iteration's pair: that
            // rotation did not meet the residual.
            if rotations < config.max_rotations && force_norm.is_finite() {
                keep(&mut force_prev, &mut force);
                if let Some(mut carried) = dir_prev.take() {
                    perp_in_place(&mut carried, &n);
                    if carried.dot(&carried).sqrt() > 1e-14 {
                        dir_prev = Some(carried);
                    }
                }
            }
            break;
        }
        if stop.lbfgs {
            // Curvature gradient. `force` is already its descent direction.
            Zip::from(&mut gradient)
                .and(&force)
                .for_each(|component, &descent| *component = -descent);
            dir.assign(&lbfgs.direction(&n, &gradient));
        } else {
            dir.assign(&force);
            if let (Some(fp), Some(dp)) = (&force_prev, &dir_prev) {
                Zip::from(&mut change)
                    .and(&force)
                    .and(fp)
                    .for_each(|c, &f, &p| *c = f - p);
                let gamma = (change.dot(&force) / fp.dot(fp)).max(0.0);
                if gamma.is_finite() {
                    Zip::from(&mut dir)
                        .and(dp)
                        .for_each(|d, &p| *d += p * gamma);
                }
            }
        }
        perp_in_place(&mut dir, &n);
        if stop.lbfgs && dir.dot(&dir).sqrt() <= 1e-14 * (1.0 + force_norm) {
            dir.assign(&force);
            perp_in_place(&mut dir, &n);
        }
        let dir_norm = dir.dot(&dir).sqrt();
        if dir_norm <= 1e-14 * (1.0 + force_norm) {
            break;
        }
        Zip::from(&mut theta)
            .and(&dir)
            .for_each(|t, &d| *t = d / dir_norm);
        // dC/dphi at phi = 0 is 2 h.theta (negative along a descent
        // direction). The trial angle is the Kastner-Sherwood guess
        // from a harmonic model, kept inside (1e-4, pi/4]. eOn's
        // ImprovedDimer::compute (client/ImprovedDimer.cpp) checks
        // |phi_prime| <= phi_tol before rotating to tau_prime and
        // evaluating the trial force.
        let b1 = h.dot(&theta);
        let phi_trial = 0.5 * (b1.abs() / (curvature.abs() + 1e-300)).atan();
        if angle_stop && phi_trial <= config.rotation_angle_tol {
            if rotations == 0 {
                keep(&mut force_prev, &mut force);
                keep(&mut dir_prev, &mut dir);
            }
            break;
        }
        let phi1 = phi_trial.clamp(MIN_TRIAL_ANGLE, 0.25 * PI);
        let (trial_sin, trial_cos) = (phi1.sin(), phi1.cos());
        Zip::from(&mut n1)
            .and(&n)
            .and(&theta)
            .for_each(|v, &n, &t| *v = n * trial_cos + t * trial_sin);
        normalize_in_place(&mut n1);
        let h1 = surface.action(x, g0, n1.view(), config)?;
        let c1 = crate::stencil::curvature_along(h1.view(), n1.view());
        rotations += 1;
        // Fourier fit through C(0), C'(0) = 2 b1, and C(phi1).
        let a1 = (curvature - c1 + b1 * (2.0 * phi1).sin()) / (1.0 - (2.0 * phi1).cos());
        let a0 = 2.0 * (curvature - a1);
        let c_at = |p: f64| 0.5 * a0 + a1 * (2.0 * p).cos() + b1 * (2.0 * p).sin();
        let mut phi = 0.5 * (b1 / a1).atan();
        if c_at(phi) > c_at(phi + 0.5 * PI) {
            phi += 0.5 * PI;
        }
        if phi > 0.5 * PI {
            phi -= PI;
        }
        let c_min = c_at(phi);
        let slack = 1e-10 * (curvature.abs() + c1.abs() + 1.0);
        if !(c_min.is_finite() && phi.is_finite()) || c_min > c1.min(curvature) + slack {
            // The harmonic model does not hold here (anharmonic or
            // noisy surface): keep the better of the two measured
            // dimers and continue from it.
            if c1 < curvature {
                std::mem::swap(&mut n, &mut n1);
                h = h1;
                curvature = c1;
            }
            keep(&mut force_prev, &mut force);
            theta.mapv_inplace(|t| t * dir_norm);
            keep(&mut dir_prev, &mut theta);
            if angle_stop && phi1 < config.rotation_angle_tol {
                break;
            }
            continue;
        }
        let (s, c) = phi.sin_cos();
        let s1 = phi1.sin();
        let w_old = (phi1 - phi).sin() / s1;
        let w_trial = s / s1;
        Zip::from(&mut theta_rotated)
            .and(&theta)
            .and(&n)
            .for_each(|r, &t, &n| *r = t * c - n * s);
        Zip::from(&mut n_new)
            .and(&n)
            .and(&theta)
            .for_each(|v, &n, &t| *v = n * c + t * s);
        normalize_in_place(&mut n_new);
        if phi1 <= MIN_TRIAL_ANGLE && w_old.abs().max(w_trial.abs()) > MAX_EXTRAPOLATION {
            // A forward-difference action is not the exact H v: its
            // Rayleigh quotient carries dr/2 T[v,v,v], so C'(0) differs
            // from 2 h.theta by dr/2 T[n,n,theta] and the fitted a1 by
            // that over phi1. On the floor of the trial angle that
            // ratio is largest, and a step far past it, with the gradient
            // extrapolated there, inherits the error multiplied by
            // sin(phi1 - phi) / sin(phi1). Measure the
            // rotated dimer instead; when it contradicts the model the
            // model is below its resolution here and the better
            // measured dimer is the estimate.
            let h_new = surface.action(x, g0, n_new.view(), config)?;
            let c_new = crate::stencil::curvature_along(h_new.view(), n_new.view());
            rotations += 1;
            if c_new.is_nan() || c_new > c1.min(curvature) + slack {
                if c1 < curvature {
                    std::mem::swap(&mut n, &mut n1);
                    curvature = c1;
                }
                break;
            }
            h = h_new;
            std::mem::swap(&mut n, &mut n_new);
            curvature = c_new;
        } else {
            Zip::from(&mut h)
                .and(&h1)
                .for_each(|h, &h1| *h = *h * w_old + h1 * w_trial);
            std::mem::swap(&mut n, &mut n_new);
            curvature = crate::stencil::curvature_along(h.view(), n.view());
        }
        keep(&mut force_prev, &mut force);
        theta_rotated.mapv_inplace(|t| t * dir_norm);
        keep(&mut dir_prev, &mut theta_rotated);
        if angle_stop && phi.abs() < config.rotation_angle_tol {
            break;
        }
    }
    history.force = force_prev;
    history.direction = dir_prev;
    history.centre = Some(x.to_owned());
    Ok((n, curvature, rotations))
}

/// Make `buffer` the value of `slot`, handing the slot's old storage
/// back as the buffer. An empty slot takes the buffer and leaves an
/// empty one, which the next write resizes.
fn keep(slot: &mut Option<Array1<f64>>, buffer: &mut Array1<f64>) {
    match slot {
        Some(held) if held.len() == buffer.len() => std::mem::swap(held, buffer),
        _ => {
            let dim = buffer.len();
            *slot = Some(std::mem::replace(buffer, Array1::zeros(dim)));
        }
    }
}

/// Component of `v` perpendicular to the unit vector `n`, in place:
/// `v - (v . n) n`, the arithmetic of [`crate::stencil::perp`].
fn perp_in_place(v: &mut Array1<f64>, n: &Array1<f64>) {
    let along = v.dot(n);
    Zip::from(v).and(n).for_each(|v, &n| *v -= n * along);
}

/// [`normalize`] in place.
fn normalize_in_place(v: &mut Array1<f64>) {
    let norm = v.dot(v).sqrt();
    if norm > 1e-14 {
        *v /= norm;
    }
}

/// Classic dimer torque window plus the residual and the rotation cap.
///
/// eOn's classic dimer stops when the torque is under `torque_min`, or after
/// `rotations_min` (one) once the torque has fallen under `torque_max`
/// (1 eV/Å²). [`STOCK_ROTATION_TOL`] plays the role of a loose
/// `torque_min`: the window is on. A tighter `rotation_tol` is a request
/// to drive the rotational force down, and the window stays out of that
/// loop. `torque_window` overrides that inference when it is `Some`.
/// The angle test in [`rotate_dimer`] is separate.
fn rotation_finished(
    config: &MinModeConfig,
    torque_window: Option<bool>,
    force_norm: f64,
    rotations: usize,
) -> bool {
    if !force_norm.is_finite()
        || force_norm <= config.rotation_tol
        || rotations >= config.max_rotations
    {
        return true;
    }
    torque_window_on(config, torque_window) && rotations >= 1 && force_norm < TORQUE_MAX
}

/// Whether the classic torque window applies: `torque_window` when it
/// is `Some`, otherwise when `rotation_tol` is at least
/// [`STOCK_ROTATION_TOL`] and either this is Lanczos or the improved
/// dimer angle test is disabled.
fn torque_window_on(config: &MinModeConfig, torque_window: Option<bool>) -> bool {
    match torque_window {
        Some(window) => window,
        None => {
            (config.kind == MinModeKind::Lanczos || config.rotation_angle_tol <= 0.0)
                && at_stock_residual(config)
        }
    }
}

/// `rotation_tol` is the stock residual or looser.
fn at_stock_residual(config: &MinModeConfig) -> bool {
    config.rotation_tol + 1e-12 >= STOCK_ROTATION_TOL
}

/// Whether the improved dimer angle test stops a rotation: a positive
/// `rotation_angle_tol` at the stock residual, or one set through
/// [`MinModeConfig::with_rotation_angle_tol`] under a tighter
/// `rotation_tol`. eOn's ImprovedDimer::compute
/// (client/ImprovedDimer.cpp) checks `|phi_prime| <= phi_tol` before the
/// trial rotation and the fitted `|phi_min|` after it.
fn angle_stop_on(config: &MinModeConfig) -> bool {
    config.rotation_angle_tol > 0.0 && (at_stock_residual(config) || config.rotation_angle_tol_set)
}

/// How a climb along the mode is capped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClimbCap {
    /// The largest atom moves by `max_move`, eOn's
    /// `maxAtomMotionAppliedV` on a long step along the mode: its FIRE
    /// step and its L-BFGS step at a negative secant curvature
    /// (`1000 * f`, client/LBFGS.cpp).
    Atom,
    /// The step has Euclidean norm `max_move` along the normalized mode,
    /// then no atom moves by more than `max_move`: eOn's conjugate
    /// gradients single step at a non-positive line curvature
    /// (client/ConjugateGradients.cpp).
    Global,
}

/// Climb along `mode` by one `max_move` under `cap`.
///
/// Henkelman and Jónsson, J. Chem. Phys. 111, 7010 (1999),
/// doi:10.1063/1.480097: while the curvature is not negative the search
/// moves on the mode alone. The sign is the energy gradient's, so the
/// step increases the energy along the mode. A vanishing parallel part
/// still leaves along `mode`: a stationary point with non-negative
/// curvature is a minimum, not a saddle.
pub(crate) fn convex_displacement(
    position: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    max_move: f64,
    cap: ClimbCap,
) -> Array1<f64> {
    let sign = if gradient.dot(&mode) < 0.0 { -1.0 } else { 1.0 };
    let mut trial = position.to_owned();
    for (slot, component) in trial.iter_mut().zip(mode.iter()) {
        *slot += sign * component;
    }
    match cap {
        ClimbCap::Atom => scale_to_atom_cap(position, &mut trial, max_move),
        ClimbCap::Global => {
            scale_to_global_cap(position, &mut trial, max_move);
            shorten_to_atom_cap(position, &mut trial, max_move);
        }
    }
    trial
}

/// [`convex_displacement`] that keeps `+mode` on a valley floor.
///
/// With `s` the length of the capped climb along the unit `mode`, a
/// point within half a step of the minimum of the parabola along it
/// (`g0 . mode + 0.5 * curvature * s >= 0`) already climbs in energy
/// along `+mode`; following the sign of a sub-step relaxation residual
/// would reverse the caller's oriented mode across the well.
pub(crate) fn climb_displacement(
    position: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    curvature: f64,
    max_move: f64,
    cap: ClimbCap,
) -> Array1<f64> {
    let forward = convex_displacement(position, mode, mode, max_move, cap);
    let mut length = 0.0;
    for (to, from) in forward.iter().zip(position.iter()) {
        length += (to - from) * (to - from);
    }
    if gradient.dot(&mode) + 0.5 * curvature * length.sqrt() >= 0.0 {
        forward
    } else {
        convex_displacement(position, gradient, mode, max_move, cap)
    }
}

/// Rescale the displacement so its Euclidean norm equals `cap`.
fn scale_to_global_cap(origin: ArrayView1<f64>, trial: &mut Array1<f64>, cap: f64) {
    if !(cap.is_finite() && cap > 0.0) {
        return;
    }
    let n = trial.len();
    let mut norm2 = 0.0;
    for k in 0..n {
        let d = trial[k] - origin[k];
        norm2 += d * d;
    }
    let norm = norm2.sqrt();
    if norm > 0.0 {
        let scale = cap / norm;
        for k in 0..n {
            trial[k] = origin[k] + scale * (trial[k] - origin[k]);
        }
    }
}

/// Return the largest per-atom (xyz triple) displacement.
fn largest_atom_move(origin: ArrayView1<f64>, trial: ArrayView1<f64>) -> f64 {
    let n = trial.len();
    let mut max_atom = 0.0;
    let mut i = 0;
    while i + 3 <= n {
        let dx = trial[i] - origin[i];
        let dy = trial[i + 1] - origin[i + 1];
        let dz = trial[i + 2] - origin[i + 2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r > max_atom {
            max_atom = r;
        }
        i += 3;
    }
    if i < n {
        let mut r2 = 0.0;
        while i < n {
            let d = trial[i] - origin[i];
            r2 += d * d;
            i += 1;
        }
        max_atom = max_atom.max(r2.sqrt());
    }
    max_atom
}

/// Rescale the displacement so its largest per-atom (xyz triple) length
/// equals `cap`. A trailing partial triple counts as one more atom,
/// matching rgmin `scale_step_atom` and eOn `maxAtomMotionApplied`. The
/// direction is unchanged; a step shorter than the cap is lengthened.
fn scale_to_atom_cap(origin: ArrayView1<f64>, trial: &mut Array1<f64>, cap: f64) {
    if !(cap.is_finite() && cap > 0.0) {
        return;
    }
    let max_atom = largest_atom_move(origin, trial.view());
    if max_atom > 0.0 {
        let scale = cap / max_atom;
        let n = trial.len();
        for k in 0..n {
            trial[k] = origin[k] + scale * (trial[k] - origin[k]);
        }
    }
}

/// `origin - FIRE_EMPTY_SCALE * eff`, shortened so no atom moves by
/// more than `max_move`.
fn empty_scaled_step(origin: ArrayView1<f64>, eff: ArrayView1<f64>, max_move: f64) -> Array1<f64> {
    let mut trial = origin.to_owned();
    for (slot, component) in trial.iter_mut().zip(eff.iter()) {
        *slot -= FIRE_EMPTY_SCALE * *component;
    }
    if max_move.is_finite() && max_move > 0.0 {
        shorten_to_atom_cap(origin, &mut trial, max_move);
    }
    trial
}

/// Shorten `trial` so its largest per-atom move is at most `cap`.
/// A step that already fits is left as it is.
fn shorten_to_atom_cap(origin: ArrayView1<f64>, trial: &mut Array1<f64>, cap: f64) {
    if !(cap.is_finite() && cap > 0.0) {
        return;
    }
    let max_atom = largest_atom_move(origin, trial.view());
    if max_atom > cap {
        let scale = cap / max_atom;
        let n = trial.len();
        for k in 0..n {
            trial[k] = origin[k] + scale * (trial[k] - origin[k]);
        }
    }
}

/// The L-BFGS pair test in rgmin: `s'y` must clear `1e-8 |s| |y|`.
fn pair_curvature_ok(s: &Array1<f64>, y: &Array1<f64>) -> bool {
    let sy = s.dot(y);
    let sn = s.dot(s).sqrt();
    let yn = y.dot(y).sqrt();
    sy.is_finite() && sy > 1e-8 * sn * yn
}

/// `1/|λ|` when `λ` is a finite negative curvature. That is the L-BFGS
/// scale `H0` and the factor on the first FIRE displacement.
fn curvature_scale(curvature: f64) -> Option<f64> {
    if !(curvature.is_finite() && curvature < 0.0) {
        return None;
    }
    let scale = 1.0 / -curvature;
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

/// Stiffness past which FIRE's opening timestep `dt` is unstable, times
/// `dt^2`. Symplectic Euler wants `dt < 2/sqrt(|λ|)`, so `|λ| > 4 / dt^2`
/// is already past that timestep.
const FIRE_CURVATURE_FLOOR: f64 = 4.0;
/// Force evaluations with no real drop before a probe is allowed.
/// A short plateau still belongs to FIRE. The vacancy cycle does not
/// leave, so two thousand flat steps is the signal.
const FIRE_STALL_STEPS: u32 = 2000;
/// A drop smaller than this fraction of the best force is still a stall.
/// Slow progress (a fraction of a percent per step) must not latch.
const FIRE_STALL_DROP: f64 = 1e-4;

/// Fraction of the Lanczos curvature scale at or below which a Ritz
/// value has no magnitude of its own.
const LANCZOS_NEAR_ZERO: f64 = 1e-14;

/// The magnitude that divides a Lanczos residual or Ritz change.
///
/// `scale` is the curvature scale of the estimate: the largest Hessian
/// action norm `|H q_j|` over its Krylov vectors. It is a scale for the
/// relative residual, and it moves with the Hessian under a change of
/// energy or length units. A curvature whose magnitude is at most
/// [`LANCZOS_NEAR_ZERO`] times `scale` is replaced by `scale`. `None`
/// when an input is not finite or `scale` is negative.
fn lanczos_divisor(curvature: f64, scale: f64) -> Option<f64> {
    if !(curvature.is_finite() && scale.is_finite() && scale >= 0.0) {
        return None;
    }
    let magnitude = curvature.abs();
    Some(if magnitude <= LANCZOS_NEAR_ZERO * scale {
        scale
    } else {
        magnitude
    })
}

/// Whether `|residual| / |curvature|` is at most `tol`.
///
/// `curvature` is the Ritz value and `scale` the curvature scale of
/// [`lanczos_divisor`]. Dividing the residual by the absolute Ritz value
/// cancels the energy and length units, so one `tol` means the same stop
/// after a change of units. A Ritz value at most 1e-14 of `scale` has no
/// usable magnitude: the residual is then divided by `scale`, which
/// carries the same units. The quotient is never formed. A non-finite
/// residual, curvature, scale, or tolerance does not agree.
fn lanczos_residual_agreed(residual: f64, curvature: f64, scale: f64, tol: f64) -> bool {
    if !(residual.is_finite() && tol.is_finite() && tol >= 0.0) {
        return false;
    }
    lanczos_divisor(curvature, scale).is_some_and(|divisor| residual <= tol * divisor)
}

/// Torque-window form of [`lanczos_residual_agreed`].
///
/// The ratio is strictly under `bound`. A Ritz value near zero against
/// `scale` uses the same guard: the residual is divided by `scale`, and
/// the quotient is never formed.
fn lanczos_residual_below(residual: f64, curvature: f64, scale: f64, bound: f64) -> bool {
    if !(residual.is_finite() && bound.is_finite() && bound >= 0.0) {
        return false;
    }
    lanczos_divisor(curvature, scale).is_some_and(|divisor| residual < bound * divisor)
}

/// Relative change of a Ritz value. A previous value at most 1e-14 of
/// `scale` only counts as agreement when the new value is inside the
/// tolerance of `scale` too, matching eOn's guarded division (a zero
/// denominator does not report agreement) in units that cancel.
fn ritz_value_agreed(new: f64, old: f64, scale: f64, tol: f64) -> bool {
    if !(new.is_finite() && tol.is_finite() && tol >= 0.0) {
        return false;
    }
    let Some(divisor) = lanczos_divisor(old, scale) else {
        return false;
    };
    if old.abs() <= LANCZOS_NEAR_ZERO * scale {
        return new.abs() <= tol * divisor;
    }
    (new - old).abs() / divisor <= tol
}

/// Stop controls for one Lanczos estimate.
///
/// `previous` is the curvature retained from the last geometry. When
/// `early` is set and that curvature is finite, the first product
/// returns immediately if its Rayleigh quotient agrees with `previous`
/// to `rotation_tol` (eOn `quit_early`). `early` stays false when the
/// previous solve stopped on the Krylov cap. `rel_tol` is the connect
/// climb's fraction of `|λ|`; `None` keeps the stock residual test.
struct LanczosStop {
    rel_tol: Option<f64>,
    previous: f64,
    early: bool,
    /// [`MinModeSession::set_rotation_stop_length`]; 0 is off.
    length: f64,
    /// The classic torque window of [`rotation_finished`].
    window: bool,
}

/// Krylov vectors of one Lanczos estimate with their Hessian actions.
pub type KrylovBlock = Vec<(Array1<f64>, Array1<f64>)>;

/// Lanczos lowest mode.
///
/// Olsen, Kroes, Henkelman, Arnaldsson, and Jónsson, J. Chem. Phys.
/// 121, 9776 (2004), doi:10.1063/1.1809574, as eOn implements it: stop
/// when the relative change in the lowest Ritz value falls under
/// `rotation_tol`, or when the residual `beta |s|` divided by the
/// absolute Ritz value does. A Ritz value at most 1e-14 of the
/// curvature scale, the largest Hessian action norm formed so far,
/// divides by that scale. A seed that is already the mode closes the
/// Krylov on the first product. Under the torque window the estimate
/// also stops once it holds two vectors and that ratio is under
/// [`TORQUE_MAX`], dividing by the same scale when the Ritz value is
/// that small. The dimer window stays an absolute rotational force.
#[allow(clippy::too_many_arguments)]
fn lanczos_mode<S: PointSurface>(
    surface: &Counted<'_, S>,
    x: ArrayView1<f64>,
    g0: ArrayView1<f64>,
    seed: Array1<f64>,
    config: &MinModeConfig,
    stop: LanczosStop,
    mut block: Option<&mut KrylovBlock>,
    work: &mut LanczosWork,
) -> Result<(Array1<f64>, f64, usize, bool), SaddleError> {
    let n = seed.len();
    let m = config.krylov_dim.min(n).max(1);
    let LanczosWork {
        q,
        w,
        alpha,
        beta,
        ritz,
        spare,
        ..
    } = work;
    alpha.clear();
    beta.clear();
    if q.first().is_some_and(|q0| q0.len() != n) {
        q.clear();
    }
    if w.len() != n {
        *w = Array1::zeros(n);
    }
    if q.is_empty() {
        q.push(seed);
    } else {
        q[0].assign(&seed);
    }
    normalize_in_place(&mut q[0]);

    let mut actions = 0;
    let mut ew_old: Option<f64> = None;
    let mut scale = 0.0_f64;
    loop {
        let j = alpha.len();
        let hv = surface.action(x, g0, q[j].view(), config)?;
        actions += 1;
        scale = scale.max(hv.dot(&hv).sqrt());
        let a = crate::stencil::curvature_along(hv.view(), q[j].view());
        alpha.push(a);
        Zip::from(&mut *w)
            .and(&hv)
            .and(&q[j])
            .for_each(|w, &h, &q| *w = h - q * a);
        if let Some(block) = block.as_deref_mut() {
            let basis = match spare.pop() {
                Some((mut held, _)) if held.len() == n => {
                    held.assign(&q[j]);
                    held
                }
                _ => q[j].clone(),
            };
            block.push((basis, hv));
        }
        // eOn quit_early: the curvature along the incoming mode is
        // already within tolerance of the previous estimate, so the
        // direction is kept. One Hessian action, then the translation.
        // A host relative tolerance (the connect climb) uses that
        // tolerance. The stock estimate uses `rotation_tol`, and only
        // after a solve that met its test (`early`).
        if j == 0 && lanczos_quit_early(&stop, a, scale, config.rotation_tol) {
            return Ok((q[0].clone(), a, actions, true));
        }
        if j > 0 {
            let b = beta[j - 1];
            Zip::from(&mut *w)
                .and(&q[j - 1])
                .for_each(|w, &q| *w -= q * b);
        }
        // Full reorthogonalization: the Krylov dimension is small and
        // finite-difference actions are noisy.
        for qi in q[..=j].iter() {
            let overlap = w.dot(qi);
            Zip::from(&mut *w)
                .and(qi)
                .for_each(|w, &q| *w -= q * overlap);
        }
        let b = w.dot(&*w).sqrt();
        let theta = ritz.lowest(alpha, beta);
        let s = &ritz.vector;
        let residual = b * s[j].abs();
        // Olsen, Kroes, Henkelman, Arnaldsson, and Jónsson, J. Chem.
        // Phys. 121, 9776 (2004), doi:10.1063/1.1809574: stop on the
        // relative change of the lowest Ritz value as well as on the
        // residual. A one-digit residual in `|theta|` accepts a high
        // mode on the first product, so it is not a stop here. A set
        // `rel_tol` replaces `rotation_tol` for that relative test.
        let agreed = lanczos_ritz_agreed(stop.rel_tol, config.rotation_tol, theta, ew_old, scale);
        // The Ritz residual `H y - theta y` is the rotational force of
        // the Ritz vector `y`, and `theta` its curvature. Dividing by
        // `|theta|`, or by `scale` when `theta` is near zero, makes the
        // stop dimensionless, so a change of energy or length units
        // does not move it. The curvature-aware length stop still reads
        // the pair in curvature units.
        let residual_ok = lanczos_residual_agreed(residual, theta, scale, config.rotation_tol)
            || rotation_force_converged(residual, theta, stop.length, config.force_tol);
        // The first two vectors span the dimer's rotation plane {v, H v},
        // and their Ritz vector is the dimer rotated to the lowest
        // curvature in it (Zeng, Xiao, and Henkelman, J. Chem. Phys. 140,
        // 044115 (2014), doi:10.1063/1.4862410). The Lanczos window uses
        // the same ratio as the residual stop, strictly under 1. It is
        // not a converged solve: the next estimate still spends its
        // second product.
        let windowed =
            stop.window && j >= 1 && lanczos_residual_below(residual, theta, scale, TORQUE_MAX);
        let capped = alpha.len() == m;
        // An invariant Krylov subspace: the new direction is rounding
        // against the curvature scale.
        let breakdown = b <= 1e-12 * scale;
        if residual_ok || agreed || windowed || capped || breakdown {
            let mut mode = Array1::zeros(n);
            for (qi, &si) in q[..=j].iter().zip(s.iter()) {
                Zip::from(&mut mode).and(qi).for_each(|m, &q| *m += q * si);
            }
            let converged = residual_ok || agreed || breakdown;
            normalize_in_place(&mut mode);
            return Ok((mode, theta, actions, converged));
        }
        ew_old = Some(theta);
        beta.push(b);
        if q.len() == j + 1 {
            q.push(Array1::zeros(n));
        }
        Zip::from(&mut q[j + 1])
            .and(&*w)
            .for_each(|q, &w| *q = w / b);
    }
}

/// Storage one Lanczos estimate hands to the next: the Krylov vectors,
/// the residual, the tridiagonal coefficients, the Ritz eigensolver's
/// tables, and the pairs of the Krylov blocks the session dropped,
/// whose basis buffers later blocks reuse. All of it is written before
/// it is read.
#[derive(Default)]
struct LanczosWork {
    q: Vec<Array1<f64>>,
    w: Array1<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    ritz: RitzWork,
    spare: KrylovBlock,
    /// An empty block whose storage the next estimate fills.
    block: KrylovBlock,
}

/// Tables of [`lowest_ritz`], kept between calls.
#[derive(Default)]
struct RitzWork {
    /// The tridiagonal, rotated in place, `k * k` row-major.
    t: Vec<f64>,
    /// Eigenvector table `v[row * k + col]`.
    v: Vec<f64>,
    /// The lowest eigenvector.
    vector: Vec<f64>,
}

impl RitzWork {
    /// [`lowest_ritz`] into these tables; the eigenvector lands in
    /// `self.vector`.
    fn lowest(&mut self, alpha: &[f64], beta: &[f64]) -> f64 {
        let k = alpha.len();
        self.t.clear();
        self.t.resize(k * k, 0.0);
        for i in 0..k {
            self.t[i * k + i] = alpha[i];
            if i + 1 < k {
                self.t[i * k + i + 1] = beta[i];
                self.t[(i + 1) * k + i] = beta[i];
            }
        }
        jacobi_eigen(&mut self.t, &mut self.v, k);
        let mut lowest = 0;
        for i in 1..k {
            if self.t[i * k + i] < self.t[lowest * k + lowest] {
                lowest = i;
            }
        }
        self.vector.clear();
        self.vector
            .extend((0..k).map(|row| self.v[row * k + lowest]));
        self.t[lowest * k + lowest]
    }
}

/// First-product return. `rel_tol` is the connect climb's relative Ritz
/// tolerance. Without it, the stock estimate returns only when the
/// previous solve converged and the Rayleigh quotient still agrees with
/// `rotation_tol`. Both comparisons use the curvature scale `|H q_0|`
/// of the first product.
fn lanczos_quit_early(stop: &LanczosStop, rayleigh: f64, scale: f64, rotation_tol: f64) -> bool {
    if let Some(tol) = stop.rel_tol {
        return ritz_value_agreed(rayleigh, stop.previous, scale, tol);
    }
    stop.early && ritz_value_agreed(rayleigh, stop.previous, scale, rotation_tol)
}

/// Relative change of successive lowest Ritz values. `rel_tol` replaces
/// the stock tolerance without changing the guarded comparison against
/// the curvature scale `scale`.
fn lanczos_ritz_agreed(
    rel_tol: Option<f64>,
    rotation_tol: f64,
    theta: f64,
    previous: Option<f64>,
    scale: f64,
) -> bool {
    let Some(old) = previous else {
        return false;
    };
    ritz_value_agreed(theta, old, scale, rel_tol.unwrap_or(rotation_tol))
}

/// Lowest eigenpair of the symmetric tridiagonal (alpha, beta), with
/// the eigenvector in the Lanczos basis.
fn lowest_ritz(alpha: &[f64], beta: &[f64]) -> (f64, Vec<f64>) {
    let mut work = RitzWork::default();
    let theta = work.lowest(alpha, beta);
    (theta, work.vector)
}

/// Components of an eigenvector with eigenvalue `lambda` on the Lanczos
/// vectors `q_1 .. q_{k+1}`, from `k` diagonals and `k` off-diagonals.
///
/// `H q_j = beta_{j-1} q_{j-1} + alpha_j q_j + beta_j q_{j+1}`, so
/// `<q_j, u>` for an eigenpair `(lambda, u)` obeys the three-term
/// recurrence of the Lanczos polynomials, `<q_{j+1}, u> = p_j(lambda)
/// <q_1, u>` (Parlett, The Symmetric Eigenvalue Problem, SIAM (1998),
/// doi:10.1137/1.9781611971163, Ch. 7). Only rows `1 .. k` of the
/// tridiagonal enter, so `q_{k+1}` takes its component without the
/// action that would give `alpha_{k+1}`. At the exact eigenvalue this is
/// the projection of `u` on the `k + 1` vectors; at the Ritz value the
/// last component vanishes and it is the Ritz vector. `None` when the
/// recurrence leaves the floating-point range.
fn lanczos_components(alpha: &[f64], beta: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let k = alpha.len();
    let mut p = Vec::with_capacity(k + 1);
    p.push(1.0);
    for j in 1..=k {
        let back = if j >= 2 { beta[j - 2] * p[j - 2] } else { 0.0 };
        let next = ((lambda - alpha[j - 1]) * p[j - 1] - back) / beta[j - 1];
        p.push(next);
        if next.abs() > 1e150 {
            for v in p.iter_mut() {
                *v *= 1e-150;
            }
        }
    }
    p.iter().all(|v| v.is_finite()).then_some(p)
}

/// Estimate of the lowest eigenvalue below the current Ritz value
/// `theta`. Ritz values of the lowest eigenvalue fall monotonically onto
/// it (Cauchy interlacing). The next one is predicted from `T_{k+1}`
/// with `alpha_{k+1} = alpha_k`, the Lanczos diagonal settling on the
/// centre of the spectrum. Aitken's delta-squared (Aitken, Proc. R. Soc.
/// Edinburgh 46, 289 (1927), doi:10.1017/S0370164600022070) then
/// extrapolates `previous`, `theta`, and that prediction when the steps
/// fall and shrink. Otherwise the prediction is the estimate.
fn extrapolated_eigenvalue(alpha: &[f64], beta: &[f64], previous: Option<f64>, theta: f64) -> f64 {
    let mut ahead = alpha.to_vec();
    ahead.push(alpha[alpha.len() - 1]);
    let (next, _) = lowest_ritz(&ahead, beta);
    let Some(previous) = previous else {
        return next;
    };
    let (d1, d2) = (theta - previous, next - theta);
    if d1 < 0.0 && d2 < 0.0 && d2 > d1 {
        next - d2 * d2 / (d2 - d1)
    } else {
        next
    }
}

/// Cyclic Jacobi for a small symmetric matrix `a`, `n * n` row-major,
/// diagonalized in place: the eigenvalues end on its diagonal. `v`
/// receives the eigenvector table `v[row * n + col]`.
// The rotation sweeps touch two columns of one row at a time, so an
// index loop is the readable form here.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen(a: &mut [f64], v: &mut Vec<f64>, n: usize) {
    v.clear();
    v.resize(n * n, 0.0);
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale = a
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let off_tolerance = 1e-12 * scale;
    for _sweep in 0..64 {
        let mut off = 0.0;
        for i in 0..n {
            for value in a[i * n + i + 1..(i + 1) * n].iter() {
                off += value * value;
            }
        }
        if off <= off_tolerance * off_tolerance {
            break;
        }
        for p in 0..n {
            for qi in (p + 1)..n {
                if a[p * n + qi].abs() <= 1e-18 * scale {
                    continue;
                }
                let theta = (a[qi * n + qi] - a[p * n + p]) / (2.0 * a[p * n + qi]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + qi];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + qi] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[qi * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[qi * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + qi];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + qi] = s * vkp + c * vkq;
                }
            }
        }
    }
}

/// The surface's answer at one point.
#[derive(Clone)]
struct PointEval {
    x: Array1<f64>,
    /// `None` when the host supplied the gradient alone.
    energy: Option<f64>,
    gradient: Array1<f64>,
}

/// Effective gradient of a min-mode translation.
///
/// Henkelman and Jonsson, J. Chem. Phys. 111, 7010 (1999): while the
/// curvature along the unit mode is not negative, the step climbs that
/// mode alone. The effective gradient is minus the parallel part and
/// the perpendicular force is dropped. Once the curvature is negative
/// the perpendicular force stays and the parallel part is reflected,
/// `g - 2 (g · τ) τ`.
pub(crate) fn translated_gradient(
    gradient: &Array1<f64>,
    mode: &Array1<f64>,
    curvature: f64,
) -> Array1<f64> {
    if curvature < 0.0 {
        crate::stencil::reflect(gradient.view(), mode.view())
    } else {
        crate::stencil::ascend(gradient.view(), mode.view())
    }
}

/// Lanczos basis for [`MinModeSession::push_estimate`]. One geometry,
/// one centre gradient, orthonormal columns in the free space.
struct LanczosCarry {
    x: Array1<f64>,
    g0: Array1<f64>,
    q: Vec<Array1<f64>>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    /// Lowest Ritz value after the previous action.
    theta: Option<f64>,
    /// Largest Hessian action norm formed by this basis.
    scale: f64,
}

/// A FIRE translation the per-atom cap shortened while the curvature
/// was negative. `origin` is that point; `eff` is the effective
/// gradient the step descended.
struct CappedFireStep {
    origin: Array1<f64>,
    eff: Array1<f64>,
    curvature: f64,
}

/// Mutable phase of an opt-in ART nouveau search.
struct ArtNState {
    /// Fixed random direction used to leave the basin.
    push_direction: Array1<f64>,
    /// Completed random-direction pushes.
    basin_pushes: usize,
    /// Whether the lowest eigenvalue crossed the configured threshold.
    activated: bool,
}

/// Stepping minimum-mode saddle search.
///
/// # Examples
///
/// A dimer search on the double well V = (x² − 1)² + 2y² + 2z². The
/// start lies off the saddle at the origin, with a guess for the mode
/// that is not the true lowest mode along x.
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface, SaddleError};
///
/// struct DoubleWell;
///
/// impl PointSurface for DoubleWell {
///     fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (x, y, z) = (r[0], r[1], r[2]);
///         let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
///         Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
///     }
/// }
///
/// let start = array![0.3, 0.1, 0.0];
/// let guess = array![1.0, 0.5, 0.0];
/// let mut session = MinModeSession::new(MinModeConfig::default(), start, guess)?;
/// let report = session.run(&DoubleWell, 200)?;
///
/// assert_eq!(report.status, MinModeStatus::Converged);
/// assert!(report.curvature < 0.0);
/// assert!(session.position().iter().all(|c| c.abs() < 1e-3));
/// assert!(session.mode()[0].abs() > 0.99);
/// # Ok::<(), SaddleError>(())
/// ```
pub struct MinModeSession {
    config: MinModeConfig,
    force_gate: ForceGate,
    kappa: Option<KappaDimerConfig>,
    x: Array1<f64>,
    mode: Array1<f64>,
    curvature: f64,
    solver: Solver,
    iteration: usize,
    /// Connect's improved Lanczos. When set, the first product returns
    /// if the Rayleigh quotient agrees with the carried curvature to
    /// this fraction, and later products stop on the relative change of
    /// the lowest Ritz value. `None` keeps the stock test: the residual
    /// divided by the absolute Ritz value, and the relative Ritz change,
    /// both at `rotation_tol`.
    lanczos_rel_tol: Option<f64>,
    /// Classic dimer torque window. `None` turns it on when
    /// `rotation_tol` is at least the stock 1e-4.
    torque_window: Option<bool>,
    /// Translation length of the curvature-aware rotation stop; 0 is off.
    rotation_stop_length: f64,
    /// Dimer rotational CG state retained across translations.
    dimer_rotation: DimerRotationHistory,
    /// Gradient at `x`, when known: from the last translation's
    /// accepted point or from the host.
    centre: Option<PointEval>,
    /// `estimate_mode` has already refreshed the mode at `x`. The next
    /// `step` translates that mode instead of estimating it again.
    mode_fresh: bool,
    /// Angle between the mode the solver last used and the mode
    /// `estimate_mode` just stored. Applied once, on the next step.
    pending_turn: f64,
    /// The last Lanczos solve met its residual or relative Ritz test.
    /// A Krylov that only hit its dimension cap does not, so the next
    /// estimate at this point keeps iterating instead of returning
    /// the truncated vector after one product.
    /// [`MinModeSession::reset`] clears it with the Krylov block.
    lanczos_ready: bool,
    /// Krylov basis kept across [`MinModeSession::push_estimate`]
    /// calls at one geometry. [`MinModeSession::estimate_mode`] and
    /// [`MinModeSession::step`] do not read it.
    lanczos_carry: Option<LanczosCarry>,
    /// Last FIRE translation that the per-atom cap shortened, when its
    /// curvature was negative. The next step drops it if the arrival is
    /// still negative. An arrival with non-negative curvature left the
    /// neighborhood; the step is replaced from [`CappedFireStep::origin`].
    capped_fire: Option<CappedFireStep>,
    /// FIRE time step the translation solver restarts with; below
    /// [`MinModeConfig::time_step`] after a capped move was replaced.
    fire_dt: f64,
    /// The translator has already taken a step, so a later FIRE move
    /// is the integrator's own.
    solver_stepped: bool,
    /// FIRE may take the curvature step `|F|/|λ|` when that move fits
    /// under the atom cap. Off, the integrator's own step is used.
    fire_curvature: bool,
    /// Smallest force and energy seen since they last fell. Counts
    /// steps that improved neither, so a stiff FIRE limit cycle can
    /// be left without treating a slow descent as a stall.
    /// [`MinModeSession::reset`] clears the latch with the search.
    fire_stall_force: f64,
    fire_stall_energy: f64,
    fire_stall_count: u32,
    /// The empty L-BFGS memory has already been seeded from `|λ|`.
    /// A later guard that drops the pairs uses the stepper's own scale.
    lbfgs_seeded: bool,
    /// Polak-Ribiere steps in the plane perpendicular to the mode, with
    /// a Newton step along it. Off, the configured stepper translates.
    cg_translate: bool,
    /// L-BFGS planes for dimer rotation. Off, Polak-Ribiere chooses them.
    lbfgs_rotate: bool,
    cg_prev_perp: Option<Array1<f64>>,
    cg_dir: Option<Array1<f64>>,
    /// Barzilai-Borwein length from the previous perpendicular step.
    cg_alpha: Option<f64>,
    /// Rebuild L-BFGS pairs in the plane perpendicular to the mode
    /// after each turn inside [`MODE_RESET_ANGLE`].
    lbfgs_project: bool,
    /// `(s, y)` the solver accepted, before projection onto the mode.
    lbfgs_shadow: VecDeque<(Array1<f64>, Array1<f64>)>,
    /// Effective gradient of the last successful estimate: the reflected
    /// force when the curvature is negative, and the reversed parallel
    /// part when it is not. The index-k step reads this for `k = 1`.
    last_translated: Option<Array1<f64>>,
    /// The first surface error raised inside the translation oracle
    /// during the current step; the rgmin oracle signature has no
    /// error channel of its own.
    surface_error: Mutex<Option<SaddleError>>,
    /// The Krylov vectors and actions of the last Lanczos estimate in
    /// the full space; empty after a dimer or with excluded modes.
    krylov: KrylovBlock,
    /// Buffers of the Lanczos estimate.
    lanczos_work: LanczosWork,
    saddle: SaddleTrack,
    /// ART nouveau phase state; absent for the ordinary min-mode driver.
    artn: Option<ArtNState>,
}

/// The last certificate and the escapes taken.
#[derive(Default)]
struct SaddleTrack {
    /// The position the certificate holds for.
    certified: Option<(Array1<f64>, Certificate)>,
    /// Pushes since the host last moved or reset the session.
    attempts: usize,
    /// Pushes over the session.
    escapes: usize,
    /// Surface evaluations the certificates and pushes spent over the
    /// session.
    evaluations: usize,
}

/// Solver used by the min-mode translation. `istep` is FIRE's opening
/// timestep and is unused by L-BFGS. An empty L-BFGS memory with the
/// NEB guards on still scales by 0.01; the curvature seed bypasses that
/// one step instead of changing the guard.
pub(crate) fn build_solver(
    method: &Method,
    n: usize,
    max_move: f64,
    istep: f64,
    highs: bool,
    lbfgs_guards: bool,
) -> Solver {
    let control = Control::default()
        .with_maxiter(usize::MAX)
        .with_gtol(0.0)
        .with_istep(istep)
        .with_maxmove(None);
    let mut solver = Solver::new(method.clone(), control, n);
    // Per-atom cap over consecutive xyz triples (eOn maxAtomMotionApplied).
    // A non-positive max_move leaves the step uncapped.
    solver.set_atom_maxmove(max_move);
    // The reflected force is the Householder translation of Goswami, Gunde,
    // and Jónsson, Front. Chem. 14, 1807063 (2026),
    // doi:10.3389/fchem.2026.1807063, Equation 3. It is not the gradient of
    // the energy the oracle reports, so L-BFGS takes its two-loop direction
    // with one call (rgmin Accept::Step) and the band's cap and angle guards.
    if matches!(method, Method::Lbfgs { .. }) {
        solver.set_accept(Accept::Step);
        solver.set_lbfgs_neb_guards(lbfgs_guards);
    }
    solver.set_highs(highs);
    solver
}

impl MinModeSession {
    /// Build a minimum-mode session at `x` with `mode` normalized as the initial direction.
    pub fn new(
        config: MinModeConfig,
        x: Array1<f64>,
        mode: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        if x.len() != mode.len() || x.is_empty() {
            return Err(SaddleError::Shape(
                "position and mode must share a nonzero length".into(),
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        config.validate()?;
        let mode = checked_mode(mode)?;
        let artn = config.artn.as_ref().map(|_| ArtNState {
            push_direction: mode.clone(),
            basin_pushes: 0,
            activated: false,
        });
        let lbfgs_guards = true;
        let solver = build_solver(
            &config.method,
            x.len(),
            config.max_move,
            config.time_step,
            false,
            lbfgs_guards,
        );
        let fire_dt = config.time_step;
        Ok(Self {
            config,
            force_gate: ForceGate::LinfNorm,
            kappa: None,
            x,
            mode,
            curvature: f64::NAN,
            solver,
            iteration: 0,
            lanczos_rel_tol: None,
            torque_window: None,
            rotation_stop_length: 0.0,
            dimer_rotation: DimerRotationHistory::default(),
            centre: None,
            surface_error: Mutex::new(None),
            krylov: Vec::new(),
            lanczos_work: LanczosWork::default(),
            mode_fresh: false,
            pending_turn: 0.0,
            lanczos_ready: false,
            lanczos_carry: None,
            capped_fire: None,
            fire_dt,
            solver_stepped: false,
            lbfgs_seeded: false,
            fire_curvature: true,
            fire_stall_force: f64::NAN,
            fire_stall_energy: f64::NAN,
            fire_stall_count: 0,
            cg_translate: false,
            lbfgs_rotate: false,
            cg_prev_perp: None,
            cg_dir: None,
            cg_alpha: None,
            lbfgs_project: false,
            lbfgs_shadow: VecDeque::new(),
            last_translated: None,
            artn,
            saddle: SaddleTrack::default(),
        })
    }

    /// Choose dimer rotation planes by L-BFGS in the tangent space.
    ///
    /// Off, the default, the planes stay Polak-Ribiere. On, the
    /// direction is the limited-memory BFGS step of Kästner and
    /// Sherwood, J. Chem. Phys. 128, 014106 (2008),
    /// doi:10.1063/1.2815812. The angle in that plane and the
    /// extrapolated gradient are unchanged, and the memory restarts
    /// at each estimate. Lanczos does not read this switch.
    pub fn set_lbfgs_rotate(&mut self, enabled: bool) {
        self.lbfgs_rotate = enabled;
    }

    /// Translate by Polak-Ribiere conjugate gradients perpendicular to
    /// the mode and a Newton step along it. Off, the default, the
    /// configured stepper takes the step.
    pub fn set_cg_translate(&mut self, enabled: bool) {
        self.cg_translate = enabled;
        if !enabled {
            self.clear_cg();
        }
    }

    /// Project stored L-BFGS pairs onto the plane perpendicular to the
    /// mode after each turn inside the Householder cone. Off, the
    /// default, a turn inside that cone keeps the pairs as the solver
    /// stored them.
    pub fn set_lbfgs_project(&mut self, enabled: bool) {
        self.lbfgs_project = enabled;
        if !enabled {
            self.lbfgs_shadow.clear();
        }
    }

    /// Seed the empty translation from `|λ|`. Off, the default, leaves
    /// the stepper's own opening scale.
    pub fn set_seed_curvature(&mut self, enabled: bool) {
        self.config.seed_curvature = enabled;
    }

    /// Translation stops when the largest force component falls under
    /// `tol` and the curvature is negative.
    pub fn set_force_tol(&mut self, tol: f64) -> Result<(), SaddleError> {
        if !(tol.is_finite() && tol > 0.0) {
            return Err(SaddleError::Invalid(
                "force tolerance must be positive and finite".into(),
            ));
        }
        self.config.force_tol = tol;
        Ok(())
    }

    /// The tolerance [`Self::set_force_tol`] last stored.
    pub fn force_tol(&self) -> f64 {
        self.config.force_tol
    }

    /// Replace a stalled FIRE step on a stiff negative mode with a
    /// short directional probe. On by default. Off, FIRE keeps the
    /// timestep it was built with.
    pub fn set_fire_curvature(&mut self, enabled: bool) {
        self.fire_curvature = enabled;
    }

    fn note_fire_progress(&mut self, force: f64, energy: f64) {
        if !force.is_finite() {
            return;
        }
        // Once neither the force nor the energy has moved for
        // [`FIRE_STALL_STEPS`], keep probing. A probe that lowers the
        // force must not hand the next step back to FIRE's opening
        // timestep, which rebuilds the cycle. A search that is still
        // descending is left alone.
        if self.fire_stall_count >= FIRE_STALL_STEPS {
            return;
        }
        let force_drop = !self.fire_stall_force.is_finite()
            || force < self.fire_stall_force * (1.0 - FIRE_STALL_DROP);
        let energy_drop = energy.is_finite()
            && (!self.fire_stall_energy.is_finite() || energy < self.fire_stall_energy - 1e-8);
        if force_drop || energy_drop {
            if force_drop {
                self.fire_stall_force = force;
            }
            if energy_drop {
                self.fire_stall_energy = energy;
            }
            self.fire_stall_count = 0;
        } else {
            self.fire_stall_count = self.fire_stall_count.saturating_add(1);
        }
    }

    /// Stop a Lanczos estimate when the lowest Ritz value's relative
    /// change falls under `tol`. The improved Lanczos search uses this
    /// so a translation does not wait for a tight residual. A set
    /// tolerance also leaves the inferred torque window out; only
    /// [`MinModeSession::set_torque_window`] with `Some(true)` adds it.
    /// `None` restores the stock test: the residual divided by the
    /// absolute Ritz value, and the relative Ritz change, both at
    /// `rotation_tol`, and the inferred window.
    pub fn set_lanczos_rel_tol(&mut self, tol: Option<f64>) -> Result<(), SaddleError> {
        if let Some(tol) = tol
            && !(tol.is_finite() && tol > 0.0)
        {
            return Err(SaddleError::Invalid(
                "Lanczos relative tolerance must be positive and finite".into(),
            ));
        }
        self.lanczos_rel_tol = tol;
        Ok(())
    }

    /// The Krylov vectors and Hessian actions of the last Lanczos
    /// estimate, empty after a dimer rotation, when the surface
    /// excludes directions, or after [`MinModeSession::reset`].
    pub fn krylov_block(&self) -> &[(Array1<f64>, Array1<f64>)] {
        &self.krylov
    }

    /// Turn the classic dimer torque window on or off apart from
    /// `rotation_tol`. With the window on, a rotation also stops after
    /// one iteration once the rotational force is under one (eOn
    /// `torque_max`), and a Lanczos estimate stops once two vectors
    /// leave the Ritz residual divided by the absolute Ritz value under
    /// one. A Ritz value whose magnitude is at most 1e-14 of the
    /// estimate's curvature scale, the largest Hessian action norm it
    /// has formed, uses that scale as the divisor. `None`, the default,
    /// keeps the window on for a
    /// dimer exactly when its angle test is disabled and `rotation_tol`
    /// is at least the stock 1e-4. Lanczos keeps the window only while
    /// no [`MinModeSession::set_lanczos_rel_tol`] is set.
    /// eOn's improved dimer uses its angle criterion with no torque
    /// window. `Some(false)` also makes a Lanczos estimate run to its
    /// residual.
    pub fn set_torque_window(&mut self, window: Option<bool>) {
        self.torque_window = window;
    }

    /// Enable or disable the ART nouveau protocol on this session.
    ///
    /// Enabling requires a Lanczos session. The current mode becomes
    /// the fixed basin-exit direction, and either change restarts the
    /// search state while retaining the current position and mode.
    pub fn set_artn(&mut self, artn: Option<ArtNConfig>) -> Result<(), SaddleError> {
        if let Some(config) = &artn {
            validate_artn(self.config.kind, config)?;
        }
        self.config.artn = artn;
        self.artn = self.config.artn.as_ref().map(|_| ArtNState {
            push_direction: self.mode.clone(),
            basin_pushes: 0,
            activated: false,
        });
        self.forget_search();
        self.saddle.certified = None;
        self.saddle.attempts = 0;
        Ok(())
    }

    /// Stop a rotation once the force error it could still remove is
    /// within `force_tol`.
    ///
    /// With `length` positive, the dimer stops rotating when
    /// [`rotation_force_error`] of its rotational force and curvature,
    /// `|C| length |sin ω|`, is at most `force_tol`, before the trial
    /// rotation spends a gradient. Lanczos stops on the same bound for
    /// its Ritz residual and Ritz value. The other stops are unchanged
    /// and still apply. `length` is the translation the mode steers:
    /// `max_move` bounds the next step, and the dimer separation `dr`
    /// is the length over which the dimer itself feels the error. 0,
    /// the default, turns the stop off.
    pub fn set_rotation_stop_length(&mut self, length: f64) -> Result<(), SaddleError> {
        if !(length.is_finite() && length >= 0.0) {
            return Err(SaddleError::Invalid(
                "rotation stop length must be finite and non-negative".into(),
            ));
        }
        self.rotation_stop_length = length;
        Ok(())
    }

    /// Translation length of the curvature-aware rotation stop; 0 is off.
    #[must_use]
    pub fn rotation_stop_length(&self) -> f64 {
        self.rotation_stop_length
    }

    fn rotation_stop(&self) -> RotationStop {
        RotationStop {
            torque_window: self.torque_window,
            length: self.rotation_stop_length,
            lbfgs: self.lbfgs_rotate && self.config.kind == MinModeKind::Dimer,
        }
    }

    /// Select the optional HiGHS feasible-set step; requires the `highs` feature.
    /// Disabled by default.
    pub fn set_highs(&mut self, enabled: bool) {
        self.solver.set_highs(enabled);
    }

    /// Select the convergence norm; the default is maximum absolute component.
    pub fn set_force_gate(&mut self, gate: ForceGate) {
        self.force_gate = gate;
    }

    /// L-BFGS NEB guards (empty-memory scale 0.01, cap and angle resets).
    /// On by default for an L-BFGS min-mode session, because the reflected
    /// force is not a potential gradient. `false` restores scale one.
    pub fn set_lbfgs_neb_guards(&mut self, enabled: bool) {
        self.solver.set_lbfgs_neb_guards(enabled);
    }

    /// Select the basin constraint without changing the rotation or translation method.
    pub fn set_kappa(&mut self, config: Option<KappaDimerConfig>) -> Result<(), SaddleError> {
        if let Some(c) = &config
            && (!c.beta.is_finite()
                || c.beta <= 0.0
                || !c.eigen.tol.is_finite()
                || c.eigen.tol <= 0.0)
        {
            return Err(SaddleError::Invalid(
                "kappa beta and tolerance must be positive and finite".into(),
            ));
        }
        self.kappa = config;
        self.reset();
        Ok(())
    }

    /// Cartesian coordinates, length 3N, in the host's length unit.
    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Estimated lowest-curvature direction. A stored mode is normalized.
    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    /// Curvature along [`MinModeSession::mode`] from the last
    /// estimate; NaN before the first.
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Effective gradient of the last step, the vector the translation
    /// followed. `None` before the first successful estimate.
    pub fn translated_force(&self) -> Option<ArrayView1<'_, f64>> {
        self.last_translated.as_ref().map(|force| force.view())
    }

    /// Restart the FIRE translation integrator from rest with time step
    /// `dt`; the other parameters are the integrator's defaults.
    fn set_fire_dt(&mut self, dt: f64) -> Result<(), SaddleError> {
        let config = rgmin::FireConfig::ase()
            .with_dt(dt)
            .with_dt_max(2.5 * self.config.time_step);
        self.solver
            .set_fire_config(config)
            .map_err(|e| SaddleError::Solver(e.to_string()))?;
        self.fire_dt = dt;
        Ok(())
    }

    /// Drop optimizer history, dimer rotation history, carried
    /// curvature, the FIRE stall latch, the Lanczos Krylov block, and
    /// the cached centre gradient at a surface-epoch boundary. The
    /// iteration count returns to zero. The FIRE time step returns to
    /// the configured opening value: `forget` keeps a step the
    /// integrator has grown, and a retreated step is restored the same
    /// way. The mode stays as the next seed. The saddle-index
    /// certificate and the escape count go too.
    pub fn reset(&mut self) {
        self.forget_search();
        self.iteration = 0;
        self.saddle.certified = None;
        self.saddle.attempts = 0;
    }

    fn forget_search(&mut self) {
        self.solver.forget();
        if matches!(self.config.method, Method::Fire { .. }) {
            // `forget` keeps the time step FIRE has grown. A FIRE
            // session accepts this restore, including after a retreat
            // lowered `fire_dt`.
            let _ = self.set_fire_dt(self.config.time_step);
        }
        self.dimer_rotation.clear();
        self.clear_cg();
        self.lbfgs_shadow.clear();
        self.curvature = f64::NAN;
        self.fire_stall_force = f64::NAN;
        self.fire_stall_energy = f64::NAN;
        self.fire_stall_count = 0;
        self.krylov.clear();
        self.lanczos_ready = false;
        self.capped_fire = None;
        self.solver_stepped = false;
        self.lbfgs_seeded = false;
        self.centre = None;
        self.mode_fresh = false;
        self.pending_turn = 0.0;
        self.lanczos_carry = None;
        self.last_translated = None;
        if let Some(artn) = &mut self.artn {
            artn.push_direction = self.mode.clone();
            artn.basin_pushes = 0;
            artn.activated = false;
        }
    }

    /// Move to `x` and keep the session: the mode stays as the next
    /// rotation's seed (a host stepping along a saddle search reuses
    /// it, so a mode that still holds costs one rotation evaluation),
    /// and the optimizer history and the dimer rotational CG history
    /// stay. `gradient` is the host's energy gradient at `x` when it
    /// has one; the next compute then skips the centre evaluation. A
    /// new position starts the escape count of
    /// [`MinModeConfig::saddle_check`] again.
    pub fn set_position(
        &mut self,
        x: Array1<f64>,
        gradient: Option<Array1<f64>>,
    ) -> Result<(), SaddleError> {
        if x.len() != self.x.len() {
            return Err(SaddleError::Shape("position length must stay fixed".into()));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        self.centre = match gradient {
            Some(gradient) => {
                if gradient.len() != x.len() {
                    return Err(SaddleError::Shape(
                        "gradient must match the position".into(),
                    ));
                }
                if !gradient.iter().all(|v| v.is_finite()) {
                    return Err(SaddleError::NonFinite("min-mode gradient"));
                }
                Some(PointEval {
                    x: x.clone(),
                    energy: None,
                    gradient,
                })
            }
            None => None,
        };
        if x != self.x {
            self.saddle.attempts = 0;
        }
        self.place(x);
        Ok(())
    }

    fn place(&mut self, x: Array1<f64>) {
        self.x = x;
        self.capped_fire = None;
        self.mode_fresh = false;
        self.pending_turn = 0.0;
        self.lanczos_carry = None;
        self.last_translated = None;
    }

    /// Replace the mode seed with a host-chosen direction.
    pub fn set_mode(&mut self, mode: Array1<f64>) -> Result<(), SaddleError> {
        if mode.len() != self.x.len() {
            return Err(SaddleError::Shape("mode must match the position".into()));
        }
        let mode = checked_mode(mode)?;
        // The solver's cached force at this point was inverted along the
        // old mode; a turn past the Householder cone also retires the pairs.
        let turned = self.mode.dot(&mode).abs().min(1.0).acos();
        self.retire_translation_memory(turned);
        self.dimer_rotation.clear();
        self.curvature = f64::NAN;
        self.mode = mode;
        self.capped_fire = None;
        self.mode_fresh = false;
        self.pending_turn = 0.0;
        self.lanczos_ready = false;
        self.lanczos_carry = None;
        self.last_translated = None;
        if let Some(artn) = &mut self.artn {
            artn.push_direction = self.mode.clone();
            artn.basin_pushes = 0;
            artn.activated = false;
        }
        Ok(())
    }

    /// Drop the cached inverted force. A turn past [`MODE_RESET_ANGLE`]
    /// also drops the L-BFGS pairs and the FIRE velocity. Inside the
    /// cone the history stays: it is the same mode, turned.
    fn retire_translation_memory(&mut self, turned: f64) {
        if turned > MODE_RESET_ANGLE {
            self.solver.forget();
            self.lbfgs_shadow.clear();
            self.clear_cg();
            return;
        }
        if self.lbfgs_project && matches!(self.config.method, Method::Lbfgs { .. }) {
            self.reload_projected_pairs();
        } else {
            self.solver.forget_evaluation();
        }
    }

    fn clear_cg(&mut self) {
        self.cg_prev_perp = None;
        self.cg_dir = None;
        self.cg_alpha = None;
    }

    /// Drop the solver's pairs and restore them projected off the mode.
    ///
    /// The effective Hessian the translator minimises is the
    /// Householder reflection of the true one (Goswami, Gunde, and
    /// Jónsson, Front. Chem. 14, 1807063 (2026),
    /// doi:10.3389/fchem.2026.1807063, Eq. 3). Kästner and Sherwood,
    /// J. Chem. Phys. 128, 014106 (2008), doi:10.1063/1.2815812, keep
    /// the quasi-Newton history across a continuous turn and build it
    /// in the plane perpendicular to the mode. A pair's component
    /// along the mode belongs to the reflected curvature, not to that
    /// plane, so it is removed and a pair that then has no curvature
    /// is dropped. Past [`MODE_RESET_ANGLE`] the history is cleared
    /// instead.
    fn reload_projected_pairs(&mut self) {
        let mode = self.mode.clone();
        let memory = match self.config.method {
            Method::Lbfgs { memory, .. } => memory.max(1),
            _ => 1,
        };
        let stored = std::mem::take(&mut self.lbfgs_shadow);
        for (mut s, mut y) in stored {
            perp_in_place(&mut s, &mode);
            perp_in_place(&mut y, &mode);
            self.note_lbfgs_pair(&s, &y);
        }
        while self.lbfgs_shadow.len() > memory {
            self.lbfgs_shadow.pop_front();
        }
        self.solver.forget();
        for (s, y) in &self.lbfgs_shadow {
            let _ = self.solver.push_pair(s.view(), y.view());
        }
    }

    fn note_lbfgs_pair(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        if !self.lbfgs_project || !pair_curvature_ok(s, y) {
            return;
        }
        let memory = match self.config.method {
            Method::Lbfgs { memory, .. } => memory.max(1),
            _ => 1,
        };
        self.lbfgs_shadow.push_back((s.clone(), y.clone()));
        while self.lbfgs_shadow.len() > memory {
            self.lbfgs_shadow.pop_front();
        }
    }

    fn centre_gradient<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
    ) -> Result<Array1<f64>, SaddleError> {
        if let Some(c) = &self.centre
            && c.x == self.x
        {
            return Ok(c.gradient.clone());
        }
        let (energy, gradient) = surface.eval(self.x.view())?;
        self.centre = Some(PointEval {
            x: self.x.clone(),
            energy: Some(energy),
            gradient: gradient.clone(),
        });
        Ok(gradient)
    }

    /// The lowest mode at `position`. A Lanczos estimate in the full
    /// space leaves its Krylov vectors and actions in `block`.
    fn mode_at<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        block: &mut KrylovBlock,
    ) -> Result<(Array1<f64>, f64, usize, bool), SaddleError> {
        block.clear();
        let excluded = surface.surface.excluded_modes(position)?;
        if excluded.ncols() != position.len() || !excluded.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "minimum-mode excluded directions".into(),
            ));
        }
        if excluded.nrows() == 0 {
            return self.estimate_free(surface, position, gradient, Some(block));
        }
        let mut space = Complement::new(position.len());
        for direction in excluded.rows() {
            space.exclude(direction);
        }
        if space.dimension() == 0 {
            return Err(SaddleError::Shape("minimum-mode space is empty".into()));
        }
        let mut seed = space.reduce(self.mode.view());
        let replaced_seed = seed.dot(&seed).sqrt() <= 64.0 * f64::EPSILON;
        if replaced_seed {
            seed = Array1::from_iter(
                (0..space.dimension()).map(|i| ((i + 1) as f64 * 1.618033988749895).sin()),
            );
        }
        let reduced = Counted {
            shifted: Default::default(),
            surface: surface.surface,
            calls: surface.calls,
            space: Some(&space),
        };
        // The quotient's Lanczos vectors are not kept: the climb model
        // reads the full-space block only.
        let (mode, curvature, rotations, ready) = self.estimate_reduced(
            &reduced,
            position,
            gradient,
            checked_mode(seed)?,
            replaced_seed,
            &space,
            None,
        )?;
        Ok((space.lift(mode.view()), curvature, rotations, ready))
    }

    /// [`MinModeSession::mode_at`] with the Krylov block it leaves
    /// becoming [`MinModeSession::krylov_block`]. A failed estimate
    /// keeps the previous block. The block it replaces lends its
    /// vectors to a later estimate.
    fn mode_into_krylov<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Result<(Array1<f64>, f64, usize, bool), SaddleError> {
        let mut block = std::mem::take(&mut self.lanczos_work.block);
        let answer = self.mode_at(surface, position, gradient, &mut block);
        let mut retired = match answer {
            Ok(_) => std::mem::replace(&mut self.krylov, block),
            Err(_) => block,
        };
        self.lanczos_work.spare.append(&mut retired);
        self.lanczos_work.block = retired;
        answer
    }

    /// Dimer or Lanczos in the full space. A failed callback does not
    /// commit the cloned rotation history.
    fn estimate_free<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        block: Option<&mut KrylovBlock>,
    ) -> Result<(Array1<f64>, f64, usize, bool), SaddleError> {
        match self.config.kind {
            MinModeKind::Dimer => {
                let mut history = self.dimer_rotation.clone();
                let (mode, curvature, rotations) = rotate_dimer(
                    surface,
                    position,
                    gradient,
                    self.mode.clone(),
                    &self.config,
                    self.rotation_stop(),
                    &mut history,
                )?;
                self.dimer_rotation = history;
                Ok((mode, curvature, rotations, true))
            }
            MinModeKind::Lanczos => {
                self.lanczos_at(surface, position, gradient, self.mode.clone(), false, block)
            }
        }
    }

    /// Estimate in an excluded-mode quotient. `space` lifts a dimer
    /// history that was rotated in the reduced coordinates. A replaced
    /// seed drops the old curvature and the old rotational pair only
    /// after the surface call succeeds. `block` is the Krylov capture
    /// of the full-space path and is `None` here.
    #[allow(clippy::too_many_arguments)]
    fn estimate_reduced<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        seed: Array1<f64>,
        replaced_seed: bool,
        space: &Complement,
        block: Option<&mut KrylovBlock>,
    ) -> Result<(Array1<f64>, f64, usize, bool), SaddleError> {
        match self.config.kind {
            MinModeKind::Dimer => {
                let mut history = if replaced_seed {
                    DimerRotationHistory::default()
                } else {
                    DimerRotationHistory {
                        force: self
                            .dimer_rotation
                            .force
                            .as_ref()
                            .map(|force| space.reduce(force.view())),
                        direction: self
                            .dimer_rotation
                            .direction
                            .as_ref()
                            .map(|direction| space.reduce(direction.view())),
                        centre: self.dimer_rotation.centre.clone(),
                    }
                };
                let (mode, curvature, rotations) = rotate_dimer(
                    surface,
                    position,
                    gradient,
                    seed,
                    &self.config,
                    self.rotation_stop(),
                    &mut history,
                )?;
                self.dimer_rotation = DimerRotationHistory {
                    force: history.force.as_ref().map(|force| space.lift(force.view())),
                    direction: history
                        .direction
                        .as_ref()
                        .map(|direction| space.lift(direction.view())),
                    centre: history.centre,
                };
                Ok((mode, curvature, rotations, true))
            }
            MinModeKind::Lanczos => {
                let answer =
                    self.lanczos_at(surface, position, gradient, seed, replaced_seed, block)?;
                if replaced_seed {
                    self.dimer_rotation.clear();
                }
                Ok(answer)
            }
        }
    }

    /// Lanczos from `seed`. `block` receives the Krylov vectors and
    /// their actions when the caller keeps them for the climb model.
    fn lanczos_at<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        position: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        seed: Array1<f64>,
        replaced_seed: bool,
        block: Option<&mut KrylovBlock>,
    ) -> Result<(Array1<f64>, f64, usize, bool), SaddleError> {
        let artn_config = self.config.artn.as_ref();
        let mut configured;
        let config = if let Some(artn) = artn_config {
            configured = self.config.clone();
            // ARTn uses a fixed, small Lanczos budget at every geometry.
            // Its previous eigenvector is the seed, not a one-product
            // quit-early certificate.
            configured.max_rotations = artn.lanczos_actions;
            configured.krylov_dim = artn.lanczos_actions;
            configured.rotation_tol = 0.0;
            &configured
        } else {
            &self.config
        };
        // A seed invented because the old mode was eliminated does not
        // carry that mode's curvature into quit-early.
        let previous = if replaced_seed || artn_config.is_some() {
            f64::NAN
        } else {
            self.curvature
        };
        let early = artn_config.is_none() && !replaced_seed && self.lanczos_ready;
        // A host relative tolerance is that host's own stop, so only an
        // explicit window applies on top of it.
        let window = match (artn_config, self.lanczos_rel_tol) {
            (Some(_), _) => false,
            (None, Some(_)) => self.torque_window == Some(true),
            (None, None) => torque_window_on(&self.config, self.torque_window),
        };
        let mut work = std::mem::take(&mut self.lanczos_work);
        let answer = lanczos_mode(
            surface,
            position,
            gradient,
            seed,
            config,
            LanczosStop {
                rel_tol: self.lanczos_rel_tol,
                previous,
                early,
                length: self.rotation_stop_length,
                window,
            },
            block,
            &mut work,
        );
        self.lanczos_work = work;
        answer
    }

    fn refresh_mode<S: PointSurface>(
        &mut self,
        surface: &Counted<'_, S>,
        g0: ArrayView1<f64>,
    ) -> Result<(f64, usize), SaddleError> {
        let position = self.x.clone();
        let (mode, curvature, rotations, ready) =
            self.mode_into_krylov(surface, position.view(), g0)?;
        self.mode = mode;
        self.curvature = curvature;
        self.lanczos_ready = ready;
        Ok((curvature, rotations))
    }

    /// Refresh the lowest mode at the current point without moving.
    /// The centre gradient comes from the cache, from
    /// [`MinModeSession::set_position`], or from one evaluation.
    /// A failed surface evaluation leaves retained dimer rotation
    /// history unchanged.
    pub fn estimate_mode<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<ModeEstimate, SaddleError> {
        // A finished solve must not resume a partial basis afterwards.
        self.lanczos_carry = None;
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            shifted: Default::default(),
            surface,
            calls: &calls,
            space: None,
        };
        // A central difference never reads the centre gradient, so a
        // rotation-only estimate does not evaluate it.
        let centre_known = self.centre.as_ref().is_some_and(|c| c.x == self.x);
        let g0 = if self.config.difference == FiniteDifference::Central && !centre_known {
            Array1::zeros(self.x.len())
        } else {
            self.centre_gradient(&counted)?
        };
        let old = self.mode.clone();
        let (curvature, rotations) = self.refresh_mode(&counted, g0.view())?;
        self.pending_turn = old.dot(&self.mode).abs().min(1.0).acos();
        self.mode_fresh = true;
        Ok(ModeEstimate {
            mode: self.mode.clone(),
            curvature,
            rotations,
            evaluations: calls.load(Ordering::Relaxed),
        })
    }

    /// Advance a Lanczos estimate by one Hessian action.
    ///
    /// [`MinModeSession::estimate_mode`] runs to its own stop, the
    /// residual test or the torque window. A host that judges the mode
    /// itself (an overlap with a vector it does not hand to the
    /// session) repeats this call.
    /// The Krylov basis stays until the geometry or the seed changes,
    /// and it grows to the free dimension rather than stopping at
    /// `krylov_dim`. Returning the same vector after the residual test
    /// would end that host's loop while the basis still had a direction
    /// left. Dimer sessions take the full rotation, matching
    /// [`MinModeSession::estimate_mode`].
    ///
    /// After `k` actions the mode lies in the `k + 1` Lanczos vectors:
    /// the last one is the residual direction the action already
    /// produced. Its components are the Lanczos polynomials at an
    /// extrapolated lowest eigenvalue (`lanczos_components`,
    /// `extrapolated_eigenvalue`), so no surface call is added. The
    /// reported curvature is the Ritz value of the `k` vectors.
    pub fn push_estimate<S: PointSurface>(
        &mut self,
        surface: &S,
    ) -> Result<ModeEstimate, SaddleError> {
        if self.config.kind != MinModeKind::Lanczos {
            return self.estimate_mode(surface);
        }
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            shifted: Default::default(),
            surface,
            calls: &calls,
            space: None,
        };
        let centre_known = self.centre.as_ref().is_some_and(|c| c.x == self.x);
        let g0 = if self.config.difference == FiniteDifference::Central && !centre_known {
            Array1::zeros(self.x.len())
        } else {
            self.centre_gradient(&counted)?
        };
        let excluded = surface.excluded_modes(self.x.view())?;
        if excluded.ncols() != self.x.len() || !excluded.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "minimum-mode excluded directions".into(),
            ));
        }
        let mut space = Complement::new(self.x.len());
        if excluded.nrows() > 0 {
            for direction in excluded.rows() {
                space.exclude(direction);
            }
            if space.dimension() == 0 {
                return Err(SaddleError::Shape("minimum-mode space is empty".into()));
            }
        }
        let dim = if excluded.nrows() == 0 {
            self.x.len()
        } else {
            space.dimension()
        };
        let restart = self.lanczos_carry.as_ref().is_none_or(|carry| {
            carry.x != self.x || carry.q.first().is_none_or(|q| q.len() != dim)
        });
        if restart {
            let seed = if excluded.nrows() == 0 {
                self.mode.clone()
            } else {
                let mut seed = space.reduce(self.mode.view());
                if seed.dot(&seed).sqrt() <= 64.0 * f64::EPSILON {
                    seed = Array1::from_iter(
                        (0..dim).map(|i| ((i + 1) as f64 * 1.618033988749895).sin()),
                    );
                }
                checked_mode(seed)?
            };
            self.lanczos_carry = Some(LanczosCarry {
                x: self.x.clone(),
                g0,
                q: vec![normalize(seed)],
                alpha: Vec::new(),
                beta: Vec::new(),
                theta: None,
                scale: 0.0,
            });
        }
        let Some(mut carry) = self.lanczos_carry.take() else {
            return Err(SaddleError::Invalid("lanczos basis was not stored".into()));
        };
        let g0 = carry.g0.clone();
        let j = carry.alpha.len();
        if j >= carry.q.len() || j >= dim {
            let mode = self.mode.clone();
            let curvature = self.curvature;
            self.lanczos_carry = Some(carry);
            return Ok(ModeEstimate {
                mode,
                curvature,
                rotations: 0,
                evaluations: calls.load(Ordering::Relaxed),
            });
        }
        let counted = Counted {
            shifted: Default::default(),
            surface,
            calls: &calls,
            space: if excluded.nrows() == 0 {
                None
            } else {
                Some(&space)
            },
        };
        let hv = match counted.action(self.x.view(), g0.view(), carry.q[j].view(), &self.config) {
            Ok(hv) => hv,
            Err(err) => {
                self.lanczos_carry = Some(carry);
                return Err(err);
            }
        };
        carry.scale = carry.scale.max(hv.dot(&hv).sqrt());
        let a = crate::stencil::curvature_along(hv.view(), carry.q[j].view());
        carry.alpha.push(a);
        let mut w = &hv - &(&carry.q[j] * a);
        if j > 0 {
            w = &w - &(&carry.q[j - 1] * carry.beta[j - 1]);
        }
        for qi in &carry.q {
            let overlap = w.dot(qi);
            w = &w - &(qi * overlap);
        }
        let b = w.dot(&w).sqrt();
        let (theta, s) = lowest_ritz(&carry.alpha, &carry.beta);
        let capped = carry.alpha.len() == dim;
        let coefficients = if !capped && b > 1e-12 * carry.scale {
            carry.beta.push(b);
            carry.q.push(w / b);
            let target = extrapolated_eigenvalue(&carry.alpha, &carry.beta, carry.theta, theta);
            lanczos_components(&carry.alpha, &carry.beta, target).unwrap_or(s)
        } else {
            s
        };
        carry.theta = Some(theta);
        let mut mode = Array1::zeros(dim);
        for (qi, ci) in carry.q.iter().zip(coefficients.iter()) {
            mode = &mode + &(qi * *ci);
        }
        mode = normalize(mode);
        let mut mode = if excluded.nrows() == 0 {
            mode
        } else {
            normalize(space.lift(mode.view()))
        };
        if mode.dot(&self.mode) < 0.0 {
            mode *= -1.0;
        }
        self.mode = mode;
        self.curvature = theta;
        // The basis is partial. A translation must finish the mode
        // on its own, and must not treat this curvature as a finished
        // Ritz value.
        self.mode_fresh = false;
        self.lanczos_ready = false;
        self.lanczos_carry = Some(carry);
        Ok(ModeEstimate {
            mode: self.mode.clone(),
            curvature: theta,
            rotations: 1,
            evaluations: calls.load(Ordering::Relaxed),
        })
    }

    fn seed_applies(&self, curvature: f64) -> bool {
        self.config.seed_curvature && self.kappa.is_none() && curvature_scale(curvature).is_some()
    }

    /// The `F/|λ|` displacement fits in `max_move`. A longer one is the
    /// capped steepest step any positive scale takes, so the seed waits
    /// until the curvature scale is the step itself.
    fn seed_fits(&self, gradient: &Array1<f64>, mode: &Array1<f64>, curvature: f64) -> bool {
        let Some(scale) = curvature_scale(curvature) else {
            return false;
        };
        let cap = self.config.max_move;
        if !(cap.is_finite() && cap > 0.0) {
            return true;
        }
        let effective = translated_gradient(gradient, mode, curvature);
        let mut trial = self.x.clone();
        for (slot, component) in trial.iter_mut().zip(effective.iter()) {
            *slot -= scale * *component;
        }
        largest_atom_move(self.x.view(), trial.view()) <= cap
    }

    /// One displacement `F/|λ|`, shortened to `max_move`.
    ///
    /// L-BFGS stores the secant so the next step is the two-loop update
    /// of Liu and Nocedal, Math. Program. 45, 503 (1989),
    /// doi:10.1007/BF01589116. FIRE does not: its next step is the
    /// ordinary integrator, still at its opening timestep (Bitzek,
    /// Koskinen, Gähler, Moseler, and Gumbsch, Phys. Rev. Lett. 97,
    /// 170201 (2006), doi:10.1103/PhysRevLett.97.170201).
    fn translate_curvature_seed<S: PointSurface>(
        &mut self,
        counted: &Counted<'_, S>,
        gradient: &Array1<f64>,
        mode: &Array1<f64>,
        curvature: f64,
        store_pair: bool,
    ) -> Result<(Array1<f64>, Option<PointEval>), SaddleError> {
        let Some(scale) = curvature_scale(curvature) else {
            return Ok((self.x.clone(), self.centre.clone()));
        };
        let effective = translated_gradient(gradient, mode, curvature);
        let mut trial = self.x.clone();
        for (slot, component) in trial.iter_mut().zip(effective.iter()) {
            *slot -= scale * *component;
        }
        shorten_to_atom_cap(self.x.view(), &mut trial, self.config.max_move);
        if largest_atom_move(self.x.view(), trial.view()) == 0.0 {
            return Ok((self.x.clone(), self.centre.clone()));
        }
        let (energy, arrived) = counted.eval(trial.view())?;
        if store_pair {
            let effective_new = translated_gradient(&arrived, mode, curvature);
            let step = &trial - &self.x;
            let secant = &effective_new - &effective;
            let _ = self.solver.push_pair(step.view(), secant.view());
            self.solver.forget_evaluation();
        }
        Ok((
            trial.clone(),
            Some(PointEval {
                x: trial,
                energy: Some(energy),
                gradient: arrived,
            }),
        ))
    }

    /// A short probe step out of a stiff FIRE limit cycle.
    ///
    /// FIRE 2.0 keeps its opening timestep through its initial delay
    /// (Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash, and Bitzek,
    /// Comput. Mater. Sci. 175, 109584 (2020),
    /// doi:10.1016/j.commatsci.2020.109584).
    /// The reflected force is Equation 3 of Goswami, Gunde, and
    /// Jónsson, Front. Chem. 14, 1807063 (2026),
    /// doi:10.3389/fchem.2026.1807063. On the vacancy the mode force is
    /// already small and the perpendicular force is not, so a cap of
    /// 0.2 steps across that well and the force stalls near 1. The
    /// probe is used only after a long stretch with no drop in force
    /// or energy, only when `|λ|` is past [`FIRE_CURVATURE_FLOOR`]
    /// over the square of [`MinModeConfig::time_step`],
    /// and only for a step no longer than 0.05 that does not raise
    /// the energy. After that stall the probe stays in place while
    /// the curvature remains steep, so the opening timestep cannot
    /// rebuild the cycle. A search that is still descending never
    /// takes it. `Ok(None)` leaves the point to FIRE.
    fn translate_stalled_fire<S: PointSurface>(
        &mut self,
        counted: &Counted<'_, S>,
        gradient: &Array1<f64>,
        mode: &Array1<f64>,
        curvature: f64,
    ) -> Result<Option<(Array1<f64>, PointEval)>, SaddleError> {
        let floor = FIRE_CURVATURE_FLOOR / (self.config.time_step * self.config.time_step);
        if !(curvature.is_finite() && curvature < -floor) {
            if self.fire_stall_count >= FIRE_STALL_STEPS {
                self.fire_stall_count = 0;
                self.fire_stall_force = f64::NAN;
                self.fire_stall_energy = f64::NAN;
            }
            return Ok(None);
        }
        if self.fire_stall_count < FIRE_STALL_STEPS {
            return Ok(None);
        }
        let max_move = self.config.max_move;
        if !(max_move.is_finite() && max_move > 0.0) {
            return Ok(None);
        }
        let eff = translated_gradient(gradient, mode, curvature);
        let enorm = eff.dot(&eff).sqrt();
        if !(enorm.is_finite() && enorm > 1e-12) {
            return Ok(None);
        }
        let descent = &eff / -enorm;
        let delta = self.config.dr.max(1e-4);
        let probe = &self.x + &(&descent * delta);
        let (_, probed) = counted.eval(probe.view())?;
        let eff_probe = translated_gradient(&probed, mode, curvature);
        let slope = (&eff_probe - &eff).dot(&descent) / delta;
        // A negative slope is uphill in the effective force. A small
        // positive slope is a soft direction: the length below is then
        // the cap, not a long Newton step.
        if !(slope.is_finite() && slope > 0.0) {
            return Ok(None);
        }
        let mut length = (enorm / slope).min(0.05).min(max_move);
        if !(length.is_finite() && length > 1e-6) {
            return Ok(None);
        }
        let current = self.centre.as_ref().and_then(|point| point.energy);
        let mut accepted: Option<(Array1<f64>, f64, Array1<f64>)> = None;
        for _ in 0..4 {
            let mut trial = &self.x + &(&descent * length);
            shorten_to_atom_cap(self.x.view(), &mut trial, max_move);
            if largest_atom_move(self.x.view(), trial.view()) <= 0.0 {
                break;
            }
            let (energy, arrived) = counted.eval(trial.view())?;
            let raised = current.is_some_and(|have| energy > have + 1e-3);
            if !raised {
                accepted = Some((trial, energy, arrived));
                break;
            }
            length *= 0.5;
        }
        let Some((trial, energy, arrived)) = accepted else {
            return Ok(None);
        };
        self.solver.forget();
        Ok(Some((
            trial.clone(),
            PointEval {
                x: trial,
                energy: Some(energy),
                gradient: arrived,
            },
        )))
    }

    /// One conjugate-gradient translation.
    ///
    /// The parallel part is the Newton step `-g_parallel / λ` of hybrid
    /// eigenvector following (Wales; Baker, J. Comput. Chem. 7, 385
    /// (1986), doi:10.1002/jcc.540070402). The perpendicular part is
    /// Polak-Ribiere, the same conjugacy Kästner and Sherwood, J. Chem.
    /// Phys. 128, 014106 (2008), doi:10.1063/1.2815812, use for the
    /// dimer rotation, with the Barzilai-Borwein length when the last
    /// perpendicular secant is positive and `1/|λ|` otherwise. The atom
    /// cap is the trust radius. A step that raises the energy is halved,
    /// up to four times.
    fn translate_cg<S: PointSurface>(
        &mut self,
        counted: &Counted<'_, S>,
        gradient: &Array1<f64>,
        mode: &Array1<f64>,
        curvature: f64,
    ) -> Result<(Array1<f64>, Option<PointEval>), SaddleError> {
        let scale = curvature_scale(curvature).unwrap_or(0.0);
        let parallel = gradient.dot(mode);
        let mut g_perp = gradient.clone();
        perp_in_place(&mut g_perp, mode);
        let mut direction = -&g_perp;
        if let (Some(prev), Some(dir)) = (&self.cg_prev_perp, &self.cg_dir) {
            let delta = &g_perp - prev;
            let denom = prev.dot(prev);
            if denom > 1e-24 {
                let beta = (g_perp.dot(&delta) / denom).max(0.0);
                direction = &direction + &(dir * beta);
            }
        }
        perp_in_place(&mut direction, mode);
        let alpha = self
            .cg_alpha
            .filter(|alpha| alpha.is_finite() && *alpha > 0.0)
            .unwrap_or(scale);
        let mut delta = &direction * alpha;
        // `-g_parallel / λ` is `g_parallel / |λ|` while λ is negative.
        if scale > 0.0 {
            delta = &delta + &(mode * (parallel * scale));
        }
        let origin = self.x.clone();
        let current = self.centre.as_ref().and_then(|point| point.energy);
        let mut length = 1.0;
        let mut chosen: Option<(Array1<f64>, f64, Array1<f64>)> = None;
        for _ in 0..5 {
            let mut trial = &origin + &(&delta * length);
            shorten_to_atom_cap(origin.view(), &mut trial, self.config.max_move);
            if largest_atom_move(origin.view(), trial.view()) <= 1e-12 {
                break;
            }
            let (energy, arrived) = counted.eval(trial.view())?;
            let decision = halve_raised_energy(current.unwrap_or(f64::NAN), energy, length);
            chosen = Some((trial, energy, arrived));
            if !decision.raised {
                break;
            }
            length = decision.step;
        }
        let Some((trial, energy, arrived)) = chosen else {
            return Ok((origin, self.centre.clone()));
        };
        let mut step_perp = &trial - &origin;
        perp_in_place(&mut step_perp, mode);
        let mut g_new = arrived.clone();
        perp_in_place(&mut g_new, mode);
        let secant = &g_new - &g_perp;
        let sy = step_perp.dot(&secant);
        let ss = step_perp.dot(&step_perp);
        self.cg_alpha = (sy > 1e-12 && ss > 0.0).then_some(ss / sy);
        self.cg_prev_perp = Some(g_new);
        self.cg_dir = Some(direction);
        Ok((
            trial.clone(),
            Some(PointEval {
                x: trial,
                energy: Some(energy),
                gradient: arrived,
            }),
        ))
    }

    /// One reflected-force step of the configured translator.
    ///
    /// The returned point is the solver's accepted geometry. The
    /// evaluation, when the oracle recorded it, is that point's energy
    /// and the true gradient (the force the solver saw was the
    /// translated one).
    fn translate_with_solver<S: PointSurface>(
        &mut self,
        surface: &S,
        counted: &Counted<'_, S>,
        g0: &Array1<f64>,
        tau: &Array1<f64>,
        translation_curvature: f64,
    ) -> Result<(Array1<f64>, Option<PointEval>), SaddleError> {
        let kappa = self.kappa.as_ref();
        let config = &self.config;
        let error_slot = &self.surface_error;
        if let Ok(mut slot) = error_slot.lock() {
            slot.take();
        }
        let last: Mutex<Option<PointEval>> = Mutex::new(None);
        // The solver re-reads the start under the new mode (its cached
        // force was inverted along the old one); the centre answers it
        // without a surface call when its energy is known.
        let start = self.x.clone();
        let start_energy = self
            .centre
            .as_ref()
            .filter(|c| c.x == start)
            .and_then(|c| c.energy);
        let oracle = Oracle::unbounded(self.x.len(), |xv: ArrayView1<f64>| {
            let answer = match start_energy {
                Some(e) if xv == start => Ok((e, g0.clone())),
                _ => counted.eval(xv),
            };
            let answer = answer.and_then(|(e, g)| {
                let eff = if let Some(kappa) = kappa {
                    let excluded = surface.excluded_modes(xv)?;
                    let out = kappa_dimer_force(
                        g.view(),
                        tau.view(),
                        excluded.view(),
                        |v| {
                            let magnitude = v.dot(&v).sqrt();
                            if magnitude == 0.0 {
                                return Ok(Array1::zeros(v.len()));
                            }
                            let unit = &v / magnitude;
                            Ok(counted.action(xv, g.view(), unit.view(), config)? * magnitude)
                        },
                        kappa,
                    )?;
                    -out.force
                } else {
                    translated_gradient(&g, tau, translation_curvature)
                };
                Ok((e, g, eff))
            });
            match answer {
                Ok((e, g, eff)) => {
                    if let Ok(mut slot) = last.lock() {
                        *slot = Some(PointEval {
                            x: xv.to_owned(),
                            energy: Some(e),
                            gradient: g,
                        });
                    }
                    (e, eff)
                }
                Err(err) => {
                    if let Ok(mut slot) = error_slot.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
                    (f64::INFINITY, Array1::zeros(xv.len()))
                }
            }
        });

        let mut x = self.x.clone();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        // The surface error outranks whatever the solver made of an
        // infinite energy and a zero gradient. `x` is a local copy;
        // the session's position stays where it was.
        if let Some(err) = self.surface_error.lock().ok().and_then(|mut s| s.take()) {
            return Err(err);
        }
        stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        let at_new = last.into_inner().ok().flatten().filter(|p| p.x == x);
        Ok((x, at_new))
    }

    /// Apply one ARTn push and its bounded perpendicular steepest-descent
    /// relaxations, leaving the final gradient cached.
    fn artn_push_relax<S: PointSurface>(
        &mut self,
        counted: &Counted<'_, S>,
        direction: &Array1<f64>,
        push: f64,
        relaxations: usize,
        relaxation_step: f64,
    ) -> Result<(), SaddleError> {
        let origin = self.x.clone();
        let mut trial = &origin + &(direction * push);
        shorten_to_atom_cap(origin.view(), &mut trial, push.abs());
        let (mut energy, mut gradient) = counted.eval(trial.view())?;
        self.x = trial;
        for _ in 0..relaxations {
            let mut perpendicular = gradient.clone();
            perp_in_place(&mut perpendicular, direction);
            if self.force_gate.value(perpendicular.view()) <= self.config.force_tol {
                break;
            }
            let origin = self.x.clone();
            let mut relaxed = &origin - &(perpendicular * relaxation_step);
            shorten_to_atom_cap(origin.view(), &mut relaxed, self.config.max_move);
            if largest_atom_move(origin.view(), relaxed.view()) <= 1e-15 {
                break;
            }
            (energy, gradient) = counted.eval(relaxed.view())?;
            self.x = relaxed;
        }
        self.centre = Some(PointEval {
            x: self.x.clone(),
            energy: Some(energy),
            gradient,
        });
        Ok(())
    }

    /// One outer iteration of the opt-in ART nouveau protocol.
    fn step_artn<S: PointSurface>(&mut self, surface: &S) -> Result<MinModeReport, SaddleError> {
        let artn_config = self
            .config
            .artn
            .clone()
            .ok_or_else(|| SaddleError::Invalid("ARTn controls are not configured".into()))?;
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            shifted: Default::default(),
            surface,
            calls: &calls,
            space: None,
        };
        let gradient = self.centre_gradient(&counted)?;
        let (activated, basin_pushes, basin_direction) = {
            let state = self
                .artn
                .as_ref()
                .ok_or_else(|| SaddleError::Invalid("ARTn phase state is absent".into()))?;
            (
                state.activated,
                state.basin_pushes,
                state.push_direction.clone(),
            )
        };

        let mut rotations = 0;
        if !activated {
            if basin_pushes >= artn_config.lanczos_start {
                let (curvature, used) = self.refresh_mode(&counted, gradient.view())?;
                rotations = used;
                if curvature < artn_config.eigenvalue_threshold
                    && let Some(state) = &mut self.artn
                {
                    state.activated = true;
                }
            }
            if self.artn.as_ref().is_some_and(|state| !state.activated) {
                self.artn_push_relax(
                    &counted,
                    &basin_direction,
                    artn_config.basin_push,
                    artn_config.basin_relaxations,
                    artn_config.relaxation_step,
                )?;
                if let Some(state) = &mut self.artn {
                    state.basin_pushes += 1;
                }
                self.iteration += 1;
                let max_force = self.centre.as_ref().map_or(f64::NAN, |point| {
                    self.force_gate.value(point.gradient.view())
                });
                return Ok(MinModeReport {
                    status: MinModeStatus::Running,
                    max_force,
                    curvature: self.curvature,
                    rotations,
                    iteration: self.iteration,
                    evaluations: calls.load(Ordering::Relaxed),
                    saddle_index: None,
                });
            }
        }

        // The threshold-crossing estimate is already current. Later
        // activation iterations refresh it, seeded by the previous mode.
        let curvature = if activated {
            let (curvature, used) = self.refresh_mode(&counted, gradient.view())?;
            rotations = used;
            curvature
        } else {
            self.curvature
        };
        let max_force = self.force_gate.value(gradient.view());
        if converged(max_force, curvature, self.config.force_tol) {
            return Ok(MinModeReport {
                status: MinModeStatus::Converged,
                max_force,
                curvature,
                rotations,
                iteration: self.iteration,
                evaluations: calls.load(Ordering::Relaxed),
                saddle_index: None,
            });
        }

        // Choose the arbitrary Lanczos sign toward the force, then use
        // Jay et al.'s bounded |F_parallel|/max(|lambda|, floor) push.
        let mut direction = self.mode.clone();
        if gradient.dot(&direction) > 0.0 {
            direction.mapv_inplace(|value| -value);
            self.mode = direction.clone();
        }
        let parallel_force = (-gradient.dot(&direction)).max(0.0);
        let mut push = parallel_force / curvature.abs().max(artn_config.curvature_floor);
        if self.config.max_move > 0.0 {
            push = push.min(self.config.max_move);
        }
        self.artn_push_relax(
            &counted,
            &direction,
            -push,
            artn_config.activation_relaxations,
            artn_config.relaxation_step,
        )?;
        self.last_translated = Some(translated_gradient(&gradient, &direction, curvature));
        self.iteration += 1;
        let arrived_force = self.centre.as_ref().map_or(max_force, |point| {
            self.force_gate.value(point.gradient.view())
        });
        Ok(MinModeReport {
            status: MinModeStatus::Running,
            max_force: arrived_force,
            curvature,
            rotations,
            iteration: self.iteration,
            evaluations: calls.load(Ordering::Relaxed),
            saddle_index: None,
        })
    }

    /// One min-mode step: refresh the lowest mode, invert the force
    /// along it, take one solver step.
    ///
    /// A surface error inside the step returns `Err` with the position
    /// unchanged. The solver's internal state has seen the failed
    /// evaluation; a host that recovers the surface and steps again
    /// may call [`MinModeSession::reset`] first to drop it.
    ///
    /// A converged step certifies the point
    /// ([`MinModeConfig::saddle_check`]) at its first convergence
    /// there; later converged steps report the same certificate.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<MinModeReport, SaddleError> {
        let mut report = self.step_inner(surface)?;
        if report.status == MinModeStatus::Converged {
            self.saddle_check(surface, &mut report)?;
        }
        Ok(report)
    }

    /// The saddle index certified at the current position, if any.
    #[must_use]
    pub fn saddle_index(&self) -> Option<&SaddleIndex> {
        let (x, certificate) = self.saddle.certified.as_ref()?;
        (*x == self.x).then_some(&certificate.index)
    }

    /// The certified lowest mode at the current position when a
    /// converged saddle certificate is present, oriented along
    /// [`MinModeSession::mode`], or [`MinModeSession::mode`] itself. A
    /// certificate whose budget ran out first holds an unresolved Ritz
    /// vector, which is not the mode.
    pub(crate) fn plane_mode(&self) -> Array1<f64> {
        if let Some((x, certificate)) = &self.saddle.certified
            && *x == self.x
            && certificate.index.converged
            && let Some(vector) = certificate.vectors.first()
        {
            if vector.dot(&self.mode) < 0.0 {
                return vector.mapv(|v| -v);
            }
            return vector.clone();
        }
        self.mode.clone()
    }

    /// Pushes off a saddle of higher index over the session.
    #[must_use]
    pub fn saddle_escapes(&self) -> usize {
        self.saddle.escapes
    }

    /// Surface evaluations the saddle-index certificates and the escape
    /// pushes spent over the session. Each also counts in the
    /// [`MinModeReport::evaluations`] of the step that spent it.
    #[must_use]
    pub fn saddle_evaluations(&self) -> usize {
        self.saddle.evaluations
    }

    /// Replace [`MinModeConfig::saddle_check`]; `None` turns the check
    /// off. The last certificate and the escape count are dropped. On
    /// an error the session is unchanged.
    pub fn set_saddle_check(&mut self, check: Option<SaddleIndexCheck>) -> Result<(), SaddleError> {
        if let Some(check) = &check {
            check.validate()?;
        }
        self.config.saddle_check = check;
        self.saddle.certified = None;
        self.saddle.attempts = 0;
        Ok(())
    }

    /// Certify a converged point, and push off a saddle of higher index
    /// when the check carries an escape.
    fn saddle_check<S: PointSurface>(
        &mut self,
        surface: &S,
        report: &mut MinModeReport,
    ) -> Result<(), SaddleError> {
        let Some(check) = self.config.saddle_check else {
            return Ok(());
        };
        let mut spent = 0;
        let cached = self
            .saddle
            .certified
            .as_ref()
            .filter(|(x, _)| *x == self.x)
            .map(|(_, certificate)| certificate.clone());
        let certificate = match cached {
            Some(certificate) => certificate,
            None => {
                let mut centre_calls = 0;
                let gradient = match self.centre.as_ref().filter(|c| c.x == self.x) {
                    Some(centre) => centre.gradient.clone(),
                    None => {
                        centre_calls = 1;
                        surface.eval(self.x.view())?.1
                    }
                };
                let rows = surface.excluded_modes(self.x.view())?;
                if rows.nrows() > 0 && rows.ncols() != self.x.len() {
                    return Err(SaddleError::Shape(
                        "excluded modes must match the position".into(),
                    ));
                }
                let mut certificate = certify(
                    &check,
                    surface,
                    Probe {
                        x: self.x.view(),
                        gradient: gradient.view(),
                        held: Vec::new(),
                        excluded: rows.outer_iter().map(|r| r.to_owned()).collect(),
                        hints: vec![self.mode.clone()],
                    },
                )?;
                certificate.index.evaluations += centre_calls;
                spent += certificate.index.evaluations;
                self.saddle.certified = Some((self.x.clone(), certificate.clone()));
                certificate
            }
        };
        report.saddle_index = Some(certificate.index.clone());
        if let Some(escape) = check.escape
            && certificate.index.index >= 2
            && self.saddle.attempts < escape.max_attempts
        {
            let push = escape_push(
                surface,
                self.x.view(),
                &certificate,
                self.mode.view(),
                escape.push_at(self.saddle.attempts),
            )?;
            self.saddle.attempts += 1;
            if let Some(push) = push {
                spent += push.evaluations;
                self.forget_search();
                self.place(push.start);
                self.set_mode(push.reaction)?;
                self.saddle.escapes += 1;
                report.status = MinModeStatus::Running;
            }
        }
        report.evaluations += spent;
        self.saddle.evaluations += spent;
        Ok(())
    }

    fn step_inner<S: PointSurface>(&mut self, surface: &S) -> Result<MinModeReport, SaddleError> {
        if self.config.artn.is_some() {
            return self.step_artn(surface);
        }
        // The translation finishes the mode itself. A partial basis
        // from push_estimate belongs to the pre-step seed.
        self.lanczos_carry = None;
        let calls = AtomicUsize::new(0);
        let counted = Counted {
            shifted: Default::default(),
            surface,
            calls: &calls,
            space: None,
        };
        let mut g0 = self.centre_gradient(&counted)?;
        // A host that just called `estimate_mode` already paid for this
        // mode. Translating it again would repeat those Hessian actions.
        let (mut curvature, mut rotations) = if self.mode_fresh {
            self.mode_fresh = false;
            let turned = self.pending_turn;
            self.pending_turn = 0.0;
            self.retire_translation_memory(turned);
            (self.curvature, 0)
        } else {
            let previous_mode = self.mode.clone();
            let refreshed = self.refresh_mode(&counted, g0.view())?;
            // The solver cached the inverted force at this point under the
            // previous mode; the step must see the latest one. The mode's
            // sign is arbitrary, so the angle reads |cos|.
            let turned = previous_mode.dot(&self.mode).abs().min(1.0).acos();
            self.retire_translation_memory(turned);
            refreshed
        };

        // A capped FIRE step whose arrival is no longer negative left the
        // neighborhood. Replace it from the last negative point with the
        // empty-memory step, then estimate the mode there. A capped step
        // that is still negative is unchanged, and so is L-BFGS.
        if let Some(left) = self.capped_fire.take()
            && left.curvature < 0.0
            && curvature >= 0.0
            && self.kappa.is_none()
        {
            let retreated =
                empty_scaled_step(left.origin.view(), left.eff.view(), self.config.max_move);
            if largest_atom_move(left.origin.view(), retreated.view()) > 0.0 {
                let (energy, gradient) = counted.eval(retreated.view())?;
                self.solver.forget();
                self.set_fire_dt(self.fire_dt * FIRE_RETREAT_DT_SCALE)?;
                self.x = retreated;
                self.centre = Some(PointEval {
                    x: self.x.clone(),
                    energy: Some(energy),
                    gradient: gradient.clone(),
                });
                let previous_mode = self.mode.clone();
                let (arrived, arrived_rotations) = self.refresh_mode(&counted, gradient.view())?;
                let turned = previous_mode.dot(&self.mode).abs().min(1.0).acos();
                self.retire_translation_memory(turned);
                curvature = arrived;
                rotations += arrived_rotations;
                g0 = gradient;
            }
        }

        self.last_translated = Some(translated_gradient(&g0, &self.mode, curvature));
        let max_force = self.force_gate.value(g0.view());
        if converged(max_force, curvature, self.config.force_tol) {
            return Ok(MinModeReport {
                status: MinModeStatus::Converged,
                max_force,
                curvature,
                rotations,
                iteration: self.iteration,
                evaluations: calls.load(Ordering::Relaxed),
                saddle_index: None,
            });
        }

        // Henkelman and Jonsson 1999: climb the mode alone while its
        // curvature is not negative, and reflect the parallel part
        // once the curvature is negative. The mode estimate for this
        // step is the one just refreshed; Lanczos and the dimer share it,
        // and so do L-BFGS and FIRE. A non-negative curvature takes one
        // global max_move along the mode, followed by eOn's per-atom
        // cap. The basin constraint keeps the solver: its force is not
        // that pure climb.
        let tau = self.mode.clone();
        let translation_curvature = curvature;
        let full_climb = translation_curvature >= 0.0
            && self.kappa.is_none()
            && self.config.max_move.is_finite()
            && self.config.max_move > 0.0;
        let lbfgs = matches!(self.config.method, Method::Lbfgs { .. });
        let fire = matches!(self.config.method, Method::Fire { .. });
        let curvature_seed = self.seed_applies(translation_curvature)
            && self.seed_fits(&g0, &tau, translation_curvature)
            && ((lbfgs && !self.lbfgs_seeded && self.solver.pair_count() == 0)
                || (fire && !self.solver_stepped));
        let (x, at_new) = if full_climb {
            let cap = if self.cg_translate {
                ClimbCap::Global
            } else {
                ClimbCap::Atom
            };
            let climbed = climb_displacement(
                self.x.view(),
                g0.view(),
                tau.view(),
                translation_curvature,
                self.config.max_move,
                cap,
            );
            let (energy, gradient) = counted.eval(climbed.view())?;
            // The cached inverted force belongs to the old centre. The
            // pairs and the FIRE velocity stay for the concave region.
            self.solver.forget_evaluation();
            let point = PointEval {
                x: climbed.clone(),
                energy: Some(energy),
                gradient,
            };
            (climbed, Some(point))
        } else if curvature_seed {
            let seeded =
                self.translate_curvature_seed(&counted, &g0, &tau, translation_curvature, lbfgs)?;
            if lbfgs {
                self.lbfgs_seeded = true;
            }
            self.solver_stepped = true;
            seeded
        } else if self.cg_translate && self.kappa.is_none() && translation_curvature < 0.0 {
            self.solver_stepped = true;
            self.translate_cg(&counted, &g0, &tau, translation_curvature)?
        } else if fire
            && self.fire_curvature
            && self.kappa.is_none()
            && let Some((trial, point)) =
                self.translate_stalled_fire(&counted, &g0, &tau, translation_curvature)?
        {
            self.solver_stepped = true;
            (trial, Some(point))
        } else {
            self.solver_stepped = true;
            self.translate_with_solver(surface, &counted, &g0, &tau, translation_curvature)?
        };
        if self.lbfgs_project
            && lbfgs
            && let Some(point) = &at_new
        {
            let step = &x - &self.x;
            let old_eff = translated_gradient(&g0, &tau, translation_curvature);
            let new_eff = translated_gradient(&point.gradient, &tau, translation_curvature);
            self.note_lbfgs_pair(&step, &(&new_eff - &old_eff));
        }
        let hit_cap = self.config.max_move.is_finite()
            && self.config.max_move > 0.0
            && largest_atom_move(self.x.view(), x.view()) + 1e-8 >= self.config.max_move;
        self.capped_fire = if hit_cap
            && translation_curvature < 0.0
            && !full_climb
            && !curvature_seed
            && self.kappa.is_none()
            && matches!(self.config.method, Method::Fire { .. })
        {
            Some(CappedFireStep {
                origin: self.x.clone(),
                eff: translated_gradient(&g0, &tau, translation_curvature),
                curvature: translation_curvature,
            })
        } else {
            None
        };
        let centre = match at_new {
            Some(p) => p,
            None => {
                let (energy, gradient) = counted.eval(x.view())?;
                PointEval {
                    x: x.clone(),
                    energy: Some(energy),
                    gradient,
                }
            }
        };
        let max_force = self.force_gate.value(centre.gradient.view());
        if max_force <= self.config.force_tol && x != self.x {
            // Force and curvature in a convergence decision describe the
            // same point. The accepted gradient supplies the centre, so
            // confirmation pays only for the required Hessian actions.
            let (mode, accepted_curvature, accepted_rotations, ready) =
                self.mode_into_krylov(&counted, x.view(), centre.gradient.view())?;
            let turned = self.mode.dot(&mode).abs().min(1.0).acos();
            self.retire_translation_memory(turned);
            self.mode = mode;
            self.curvature = accepted_curvature;
            self.lanczos_ready = ready;
            curvature = accepted_curvature;
            rotations += accepted_rotations;
        }
        self.x = x;
        // A step that leaves the atoms in place still counts, so a host loop
        // capped on `iteration` stops at a stationary positive curvature.
        self.iteration += 1;
        self.centre = Some(centre);
        if fire {
            let energy = self.centre.as_ref().and_then(|point| point.energy);
            self.note_fire_progress(max_force, energy.unwrap_or(f64::NAN));
        }
        let status = if converged(max_force, curvature, self.config.force_tol) {
            MinModeStatus::Converged
        } else {
            MinModeStatus::Running
        };
        Ok(MinModeReport {
            status,
            max_force,
            curvature,
            rotations,
            iteration: self.iteration,
            evaluations: calls.load(Ordering::Relaxed),
            saddle_index: None,
        })
    }

    /// Convenience loop over [`MinModeSession::step`].
    ///
    /// Every step counts in [`MinModeReport::iteration`], including one
    /// that leaves the atoms in place, so a stationary point cannot spin.
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<MinModeReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == MinModeStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArtNConfig, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface,
        SaddleIndexCheck, rotation_force_converged, rotation_force_error, translated_gradient,
    };
    use crate::error::SaddleError;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, ArrayView1, array};
    use rgmin::Method;

    #[test]
    fn positive_curvature_keeps_only_the_reversed_mode() {
        let gradient = array![1.0, -4.0, 0.5];
        let mode = array![1.0, 0.0, 0.0];
        let eff = translated_gradient(&gradient, &mode, 0.25);
        assert_abs_diff_eq!(eff[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[2], 0.0, epsilon = 1e-12);
        let flat = translated_gradient(&gradient, &mode, 0.0);
        assert_abs_diff_eq!(flat[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn negative_curvature_reflects_the_parallel_part() {
        let gradient = array![1.0, -4.0, 0.5];
        let mode = array![1.0, 0.0, 0.0];
        let eff = translated_gradient(&gradient, &mode, -1.5);
        assert_abs_diff_eq!(eff[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[1], -4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eff[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn negative_curvature_seeds_the_inverse_scale() {
        assert_abs_diff_eq!(super::curvature_scale(-2.0).unwrap(), 0.5, epsilon = 1e-15);
        assert!(super::curvature_scale(0.0).is_none());
        assert!(super::curvature_scale(1.5).is_none());
        assert!(super::curvature_scale(f64::NAN).is_none());
    }

    #[test]
    fn artn_defaults_are_pinned() {
        let artn = ArtNConfig::default();
        assert_eq!(artn.lanczos_start, 3);
        assert_eq!(artn.basin_relaxations, 6);
        assert_eq!(artn.activation_relaxations, 5);
        assert_eq!(artn.lanczos_actions, 16);
        assert_eq!(artn.basin_push.to_bits(), 0.2_f64.to_bits());
        assert_eq!(artn.curvature_floor.to_bits(), 0.5_f64.to_bits());
        assert_eq!(artn.eigenvalue_threshold.to_bits(), (-0.01_f64).to_bits());
    }

    #[test]
    fn artn_is_lanczos_only() {
        let config = MinModeConfig::default().with_artn(ArtNConfig::default());
        let error = MinModeSession::new(config, array![0.0, 0.0], array![1.0, 0.0])
            .err()
            .expect("a dimer ARTn configuration is invalid");
        assert!(error.to_string().contains("Lanczos"));
    }

    #[test]
    fn artn_delays_lanczos_and_relaxes_perpendicular_to_the_push() {
        let artn = ArtNConfig::default()
            .with_lanczos_start(2)
            .with_basin_relaxations(2);
        let config = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_artn(artn);
        let mut session = MinModeSession::new(config, array![0.0, 0.3], array![1.0, 0.0]).unwrap();
        let first = session.step(&RotatingQuadratic).unwrap();
        assert_eq!(first.rotations, 0);
        assert!(first.curvature.is_nan());
        assert_abs_diff_eq!(session.position()[0], 0.2, epsilon = 1e-12);
        assert!(
            session.position()[1].abs() < 0.3,
            "the perpendicular coordinate was relaxed"
        );
        let second = session.step(&RotatingQuadratic).unwrap();
        assert_eq!(second.rotations, 0);
        let third = session.step(&RotatingQuadratic).unwrap();
        assert!(third.rotations > 0, "Lanczos starts after two pushes");
    }

    #[test]
    fn artn_activation_follows_the_warm_lanczos_mode() {
        let artn = ArtNConfig::default()
            .with_lanczos_start(0)
            .with_activation_relaxations(2);
        let config = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_force_tol(1e-10)
            .with_artn(artn);
        let mut session = MinModeSession::new(config, array![0.3, 0.2], array![1.0, 0.1]).unwrap();
        let first = session.step(&RotatingQuadratic).unwrap();
        assert!(first.rotations > 0);
        assert!(first.curvature < -0.01);
        assert!(session.position()[0].abs() < 0.3);
        assert!(session.position()[1].abs() < 0.2);
        let previous = session.mode().to_owned();
        let second = session.step(&RotatingQuadratic).unwrap();
        assert!(second.rotations > 0);
        assert!(session.mode().dot(&previous).abs() > 0.99);
    }

    #[test]
    fn seeded_lbfgs_step_leaves_a_secant_pair() {
        let config = MinModeConfig::default()
            .with_seed_curvature(true)
            .with_method(Method::lbfgs_with_memory(10))
            .with_force_tol(1e-12)
            .with_max_move(0.2);
        // The Newton step 0.15 fits in max_move 0.2, so the seed is taken
        // and the secant is what the next two-loop step would read.
        let mut session =
            MinModeSession::new(config, array![0.15, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let report = session.step(&QuadraticSaddle).unwrap();
        assert_eq!(report.status, super::MinModeStatus::Converged);
        assert!(session.solver.pair_count() > 0);
        assert!(session.position()[0].abs() < 1e-9);
    }

    /// V = -x^2 + y^2 + z^2. One reflected L-BFGS step stores a pair.
    struct QuadraticSaddle;

    impl PointSurface for QuadraticSaddle {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let e = -x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            Ok((e, array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]))
        }
    }

    fn rotate_in_plane(mode: &Array1<f64>, theta: f64) -> Array1<f64> {
        let mut axis = array![0.0, 1.0, 0.0];
        if mode.dot(&axis).abs() > 0.9 {
            axis = array![0.0, 0.0, 1.0];
        }
        let parallel = mode.dot(&axis);
        let mut ortho = &axis - &(mode * parallel);
        let norm = ortho.dot(&ortho).sqrt();
        ortho /= norm;
        let (s, c) = theta.sin_cos();
        &(mode * c) + &(&ortho * s)
    }

    /// A turn inside the Householder cone keeps the L-BFGS pairs. A
    /// genuine swap past [`super::MODE_RESET_ANGLE`] drops them. Goswami,
    /// Gunde, and Jónsson, Front. Chem. (2026),
    /// doi:10.3389/fchem.2026.1807063; Kästner and Sherwood, J. Chem.
    /// Phys. 128, 014106 (2008), doi:10.1063/1.2815812.
    #[test]
    fn lbfgs_pairs_survive_a_turn_inside_the_cone() {
        let config = MinModeConfig::default()
            .with_method(Method::lbfgs_with_memory(10))
            .with_force_tol(1e-8)
            .with_max_move(0.05)
            .with_rotation_angle_tol(0.0);
        let mut session =
            MinModeSession::new(config, array![0.4, 0.15, -0.1], array![1.0, 0.2, 0.0]).unwrap();
        session.step(&QuadraticSaddle).unwrap();
        let kept = session.solver.pair_count();
        assert!(kept > 0, "a reflected step stores an L-BFGS pair");
        let inside = rotate_in_plane(
            &session.mode().to_owned(),
            10.0 * std::f64::consts::PI / 180.0,
        );
        session.set_mode(inside).unwrap();
        assert_eq!(
            session.solver.pair_count(),
            kept,
            "ten degrees is the same mode"
        );
        let swapped = rotate_in_plane(&session.mode().to_owned(), 0.5 * std::f64::consts::PI);
        session.set_mode(swapped).unwrap();
        assert_eq!(
            session.solver.pair_count(),
            0,
            "ninety degrees clears the pairs"
        );
    }

    #[test]
    fn lanczos_components_at_a_ritz_value_are_the_ritz_vector() {
        let alpha = [2.0, -1.0, 3.0];
        let beta = [0.7, 0.4, 0.5];
        let (theta, s) = super::lowest_ritz(&alpha, &beta[..2]);
        let p = super::lanczos_components(&alpha, &beta, theta).unwrap();
        assert_abs_diff_eq!(p[3], 0.0, epsilon = 1e-10);
        let scale = s[0] / p[0];
        for (pj, sj) in p.iter().zip(s.iter()) {
            assert_abs_diff_eq!(pj * scale, *sj, epsilon = 1e-10);
        }
    }

    /// At the exact eigenvalue the components are the eigenvector's
    /// overlaps with the Lanczos vectors, the last one included.
    #[test]
    fn lanczos_components_at_the_eigenvalue_project_the_eigenvector() {
        let d = [-1.0, 0.5, 1.0, 2.0, 4.0, 8.0];
        let apply = |v: &Array1<f64>| Array1::from_iter(v.iter().zip(d.iter()).map(|(a, b)| a * b));
        let seed = array![0.3, 0.5, 0.4, 0.4, 0.4, 0.4];
        let mut q = vec![super::normalize(seed)];
        let (mut alpha, mut beta) = (Vec::new(), Vec::new());
        for j in 0..3 {
            let hv = apply(&q[j]);
            let a = hv.dot(&q[j]);
            let mut w = &hv - &(&q[j] * a);
            if j > 0 {
                w = &w - &(&q[j - 1] * beta[j - 1]);
            }
            for qi in q.iter() {
                let overlap = w.dot(qi);
                w = &w - &(qi * overlap);
            }
            let b = w.dot(&w).sqrt();
            alpha.push(a);
            beta.push(b);
            q.push(w / b);
        }
        let p = super::lanczos_components(&alpha, &beta, -1.0).unwrap();
        assert_eq!(p.len(), 4);
        for (pj, qj) in p.iter().zip(q.iter()) {
            assert_abs_diff_eq!(pj * q[0][0], qj[0], epsilon = 1e-12);
        }
    }

    struct RotatingQuadratic;

    impl PointSurface for RotatingQuadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((
                -x[0] * x[0] + 2.0 * x[1] * x[1],
                array![-2.0 * x[0], 4.0 * x[1]],
            ))
        }
    }

    struct FailingSurface;

    impl PointSurface for FailingSurface {
        fn eval(&self, _x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Err(SaddleError::Shape("deliberate failure".into()))
        }
    }

    struct AnisotropicQuadratic;

    impl PointSurface for AnisotropicQuadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((
                -x[0] * x[0] + 0.5 * x[1] * x[1] + 2.0 * x[2] * x[2] + 4.0 * x[3] * x[3],
                array![-2.0 * x[0], x[1], 4.0 * x[2], 8.0 * x[3]],
            ))
        }
    }

    #[test]
    fn dimer_and_lanczos_leave_different_trajectories() {
        let start = array![0.4, -0.2, 0.3, 0.15];
        let mode = array![0.2, 0.7, -0.1, 0.4];
        let open = |kind| {
            MinModeSession::new(
                MinModeConfig::default()
                    .with_kind(kind)
                    .with_force_tol(1e-12)
                    .with_max_move(0.2),
                start.clone(),
                mode.clone(),
            )
            .unwrap()
        };
        let mut dimer = open(MinModeKind::Dimer);
        let mut lanczos = open(MinModeKind::Lanczos);
        let surface = AnisotropicQuadratic;
        let mut separated = false;
        for _ in 0..4 {
            dimer.step(&surface).unwrap();
            lanczos.step(&surface).unwrap();
            let delta = &dimer.position() - &lanczos.position();
            if delta.dot(&delta).sqrt() > 1e-8 {
                separated = true;
                break;
            }
        }
        assert!(
            separated,
            "dimer and lanczos shared a trajectory from the same start"
        );
    }

    struct ExcludedAnisotropicQuadratic;

    impl PointSurface for ExcludedAnisotropicQuadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            AnisotropicQuadratic.eval(x)
        }

        fn excluded_modes(&self, _x: ArrayView1<f64>) -> Result<ndarray::Array2<f64>, SaddleError> {
            Ok(array![[1.0, 0.0, 0.0, 0.0]])
        }
    }

    #[test]
    fn improved_dimer_does_not_infer_the_classic_torque_window() {
        let stock = MinModeConfig::default();
        let eon = MinModeConfig::default().with_rotation_tol(0.1);
        let tight = MinModeConfig::default().with_rotation_tol(1e-6);
        let classic = MinModeConfig::default().with_rotation_angle_tol(0.0);
        let lanczos = MinModeConfig::default().with_kind(MinModeKind::Lanczos);
        // A rotational force of 0.5 after one rotation is inside the window.
        assert!(!super::rotation_finished(&stock, None, 0.5, 1));
        assert!(!super::rotation_finished(&eon, None, 0.5, 1));
        assert!(!super::rotation_finished(&tight, None, 0.5, 1));
        assert!(super::rotation_finished(&classic, None, 0.5, 1));
        assert!(super::rotation_finished(&lanczos, None, 0.5, 1));
        assert!(!super::rotation_finished(&eon, Some(false), 0.5, 1));
        assert!(super::rotation_finished(&tight, Some(true), 0.5, 1));
        // The window needs one rotation and a force under 1.
        assert!(!super::rotation_finished(&eon, Some(true), 0.5, 0));
        assert!(!super::rotation_finished(&eon, Some(true), 1.5, 1));
        // The residual and the cap stop the rotation with the window off.
        assert!(super::rotation_finished(&eon, Some(false), 0.05, 1));
        assert!(super::rotation_finished(
            &eon,
            Some(false),
            0.5,
            eon.max_rotations
        ));
    }

    #[test]
    fn positive_climb_caps_the_atom_or_the_global_step() {
        let component = 0.5_f64.sqrt();
        let position = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let gradient = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mode = array![component, 0.0, 0.0, component, 0.0, 0.0];
        let climb = |cap| {
            let trial =
                super::convex_displacement(position.view(), gradient.view(), mode.view(), 0.2, cap);
            let step = &trial - &position;
            (
                step.dot(&step).sqrt(),
                super::largest_atom_move(position.view(), trial.view()),
            )
        };
        let (norm, atom) = climb(super::ClimbCap::Atom);
        assert_abs_diff_eq!(atom, 0.2, epsilon = 1e-15);
        assert_abs_diff_eq!(norm, 0.2 * 2.0_f64.sqrt(), epsilon = 1e-15);
        let (norm, atom) = climb(super::ClimbCap::Global);
        assert_abs_diff_eq!(norm, 0.2, epsilon = 1e-15);
        assert!(atom < 0.2);
    }

    #[test]
    fn minimum_mode_fire_opens_at_the_eon_timestep() {
        let session = MinModeSession::new(
            MinModeConfig::default(),
            array![0.0, 0.0, 0.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(session.fire_dt, 1.0);
        assert_eq!(session.lanczos_rel_tol, None);
    }

    #[test]
    fn minimum_mode_fire_opens_at_the_configured_timestep() {
        let mut session = MinModeSession::new(
            MinModeConfig::default().with_time_step(0.25),
            array![0.0, 0.0, 0.0],
            array![1.0, 0.0, 0.0],
        )
        .unwrap();
        assert_eq!(session.fire_dt, 0.25);
        session.set_fire_dt(0.125).unwrap();
        session.forget_search();
        assert_eq!(session.fire_dt, 0.25);
        for time_step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match MinModeSession::new(
                MinModeConfig::default().with_time_step(time_step),
                array![0.0, 0.0, 0.0],
                array![1.0, 0.0, 0.0],
            )
            .err()
            {
                Some(SaddleError::Invalid(message)) => {
                    assert_eq!(message, "FIRE time step must be finite and positive");
                }
                other => panic!("unexpected time step result: {other:?}"),
            }
        }
    }

    #[test]
    fn minimum_mode_config_validation_rejects_each_field_with_its_own_message() {
        MinModeConfig::default().validate().unwrap();
        let lanczos = || MinModeConfig::default().with_kind(MinModeKind::Lanczos);
        let bad_artn = |artn: ArtNConfig| lanczos().with_artn(artn);
        let cases: Vec<(MinModeConfig, &str)> = vec![
            (
                MinModeConfig::default().with_dr(0.0),
                "finite-difference step must be positive and finite",
            ),
            (
                MinModeConfig::default().with_dr(f64::NAN),
                "finite-difference step must be positive and finite",
            ),
            (
                MinModeConfig::default().with_force_tol(-1.0e-3),
                "force tolerance must be finite and non-negative",
            ),
            (
                MinModeConfig::default().with_force_tol(f64::INFINITY),
                "force tolerance must be finite and non-negative",
            ),
            (
                MinModeConfig::default().with_rotation_tol(-1.0e-4),
                "rotation tolerances must be finite and non-negative",
            ),
            (
                MinModeConfig::default().with_rotation_angle_tol(f64::NAN),
                "rotation tolerances must be finite and non-negative",
            ),
            (
                MinModeConfig::default().with_max_rotations(0),
                "max_rotations must be at least 1",
            ),
            (
                MinModeConfig::default().with_krylov_dim(0),
                "Lanczos krylov dimension must be at least 1",
            ),
            (
                MinModeConfig::default().with_max_move(f64::NAN),
                "max_move must be finite",
            ),
            (
                MinModeConfig::default().with_time_step(0.0),
                "FIRE time step must be finite and positive",
            ),
            (
                MinModeConfig::default().with_time_step(f64::NEG_INFINITY),
                "FIRE time step must be finite and positive",
            ),
            (
                MinModeConfig::default()
                    .with_saddle_check(SaddleIndexCheck::default().with_block(0)),
                "saddle-index check needs",
            ),
            (
                bad_artn(ArtNConfig::default().with_basin_push(0.0)),
                "ARTn basin push must be positive and finite",
            ),
            (
                MinModeConfig::default().with_method(Method::Bfgs),
                "tests the oracle value",
            ),
        ];
        for (config, expected) in cases {
            match config.validate() {
                Err(SaddleError::Invalid(message)) => {
                    assert!(message.contains(expected), "{message} lacks {expected}");
                }
                other => panic!("unexpected validation result for {expected}: {other:?}"),
            }
        }
        for artn in [
            ArtNConfig::default().with_eigenvalue_threshold(0.1),
            ArtNConfig::default().with_relaxation_step(f64::NAN),
            ArtNConfig::default().with_lanczos_actions(0),
            ArtNConfig::default().with_curvature_floor(0.0),
        ] {
            assert!(matches!(
                MinModeConfig::default()
                    .with_kind(MinModeKind::Lanczos)
                    .with_artn(artn)
                    .validate(),
                Err(SaddleError::Invalid(_))
            ));
        }
        assert!(matches!(
            MinModeConfig::default()
                .with_kind(MinModeKind::Dimer)
                .with_artn(ArtNConfig::default())
                .validate(),
            Err(SaddleError::Invalid(message)) if message.contains("requires the Lanczos")
        ));
    }

    #[test]
    fn minimum_mode_config_validation_keeps_an_uncapped_max_move() {
        MinModeConfig::default()
            .with_max_move(0.0)
            .validate()
            .unwrap();
        MinModeConfig::default()
            .with_max_move(-1.0)
            .validate()
            .unwrap();
        MinModeConfig::default()
            .with_force_tol(0.0)
            .validate()
            .unwrap();
    }

    #[test]
    fn improved_dimer_skips_trial_evaluation_inside_angle_tol() {
        let angle = 2.0_f64.to_radians();
        let mut session = MinModeSession::new(
            MinModeConfig::default(),
            array![0.3, 0.2, 0.1, 0.4],
            array![angle.cos(), angle.sin(), 0.0, 0.0],
        )
        .unwrap();
        let estimate = session.estimate_mode(&AnisotropicQuadratic).unwrap();
        assert_eq!(estimate.rotations, 0);
        assert_eq!(estimate.evaluations, 2);
    }

    /// `V = 0.5 x' diag(h) x`.
    struct DiagonalQuadratic(Array1<f64>);

    impl PointSurface for DiagonalQuadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let gradient = &self.0 * &x;
            Ok((0.5 * x.dot(&gradient), gradient))
        }
    }

    impl DiagonalQuadratic {
        fn rotational_force(&self, mode: &Array1<f64>) -> f64 {
            let action = &self.0 * mode;
            let residual = &action - &(mode * action.dot(mode));
            residual.dot(&residual).sqrt()
        }
    }

    fn tight_dimer_estimate(config: MinModeConfig) -> (super::ModeEstimate, f64) {
        let surface = DiagonalQuadratic(array![-1.0, 2.0, 8.0]);
        let mode = super::normalize(array![1.0, 0.04, 0.03]);
        let mut session = MinModeSession::new(config, array![0.1, -0.05, 0.02], mode).unwrap();
        let estimate = session.estimate_mode(&surface).unwrap();
        let residual = surface.rotational_force(&estimate.mode);
        (estimate, residual)
    }

    #[test]
    fn tight_rotation_tol_drives_the_improved_dimer_to_its_residual() {
        // The seed is about three degrees off the negative mode, inside
        // the default angle test at both the trial and the fitted angle.
        let rotation_tol = 1e-8;
        let (estimate, residual) =
            tight_dimer_estimate(MinModeConfig::default().with_rotation_tol(rotation_tol));
        assert!(
            residual < rotation_tol,
            "{estimate:?} residual {residual:e}"
        );
        assert!(estimate.rotations > 1, "{estimate:?}");
    }

    #[test]
    fn an_explicit_default_angle_keeps_the_angle_stop_under_a_tight_residual() {
        let rotation_tol = 1e-8;
        let (estimate, residual) = tight_dimer_estimate(
            MinModeConfig::default()
                .with_rotation_tol(rotation_tol)
                .with_rotation_angle_tol(super::DEFAULT_ROTATION_ANGLE_TOL),
        );
        assert_eq!(estimate.rotations, 1, "{estimate:?}");
        assert!(residual > rotation_tol, "{residual:e}");
        assert!(super::angle_stop_on(
            &MinModeConfig::default()
                .with_rotation_tol(rotation_tol)
                .with_rotation_angle_tol(super::DEFAULT_ROTATION_ANGLE_TOL)
        ));
        assert!(!super::angle_stop_on(
            &MinModeConfig::default().with_rotation_tol(rotation_tol)
        ));
        assert!(super::angle_stop_on(&MinModeConfig::default()));
        assert!(!super::angle_stop_on(
            &MinModeConfig::default().with_rotation_angle_tol(0.0)
        ));
    }

    #[test]
    fn the_stock_residual_keeps_its_rounding_slack() {
        let stock = super::STOCK_ROTATION_TOL;
        let classic = |tol: f64| {
            MinModeConfig::default()
                .with_rotation_angle_tol(0.0)
                .with_rotation_tol(tol)
        };
        assert!(super::torque_window_on(&classic(stock - 5e-13), None));
        assert!(!super::torque_window_on(&classic(stock - 2e-12), None));
        assert!(super::angle_stop_on(
            &MinModeConfig::default().with_rotation_tol(stock - 5e-13)
        ));
        assert!(!super::angle_stop_on(
            &MinModeConfig::default().with_rotation_tol(stock - 2e-12)
        ));
    }

    #[test]
    fn stalled_fire_probes_a_curvature_past_the_euler_bound() {
        // V = -5 x^2 + 40 y^2 + z^2: the curvature along x is -10, past
        // the dt = 1 bound |λ| = 4 and under the old floor of 25.
        let surface = DiagonalQuadratic(array![-10.0, 80.0, 2.0]);
        let mode = array![1.0, 0.0, 0.0];
        let mut session = MinModeSession::new(
            MinModeConfig::default(),
            array![0.002, 0.15, 0.0],
            mode.clone(),
        )
        .unwrap();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let counted = super::Counted {
            surface: &surface,
            calls: &calls,
            space: None,
            shifted: std::sync::Mutex::new(Array1::zeros(3)),
        };
        let gradient = surface.eval(session.x.view()).unwrap().1;
        for (curvature, probes) in [(-10.0, true), (-24.0, true), (-3.9, false)] {
            session.fire_stall_count = super::FIRE_STALL_STEPS;
            let probe = session
                .translate_stalled_fire(&counted, &gradient, &mode, curvature)
                .unwrap();
            assert_eq!(probe.is_some(), probes, "curvature {curvature}");
        }
    }

    #[test]
    fn reset_then_step_matches_a_fresh_session_after_a_fire_stall() {
        // V = -5 x^2 + 40 y^2 + z^2. A stall latch left at the probe
        // threshold takes the curvature step; a new session does not.
        let surface = DiagonalQuadratic(array![-10.0, 80.0, 2.0]);
        let start = array![0.002, 0.15, 0.0];
        let mode = array![1.0, 0.0, 0.0];
        let config = MinModeConfig::default().with_saddle_check(None);
        let mut stalled = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
        stalled.fire_stall_count = super::FIRE_STALL_STEPS;
        stalled.fire_stall_force = 1.0;
        stalled.fire_stall_energy = -1.0;
        stalled.reset();
        let mut fresh = MinModeSession::new(config, start, mode).unwrap();
        assert_eq!(stalled.fire_stall_count, fresh.fire_stall_count);
        assert_eq!(
            stalled.fire_stall_force.is_nan(),
            fresh.fire_stall_force.is_nan()
        );
        assert_eq!(
            stalled.fire_stall_energy.is_nan(),
            fresh.fire_stall_energy.is_nan()
        );
        let stalled_report = stalled.step(&surface).unwrap();
        let fresh_report = fresh.step(&surface).unwrap();
        assert_eq!(stalled_report.status, fresh_report.status);
        assert_eq!(stalled.position(), fresh.position());
        assert_eq!(stalled.mode(), fresh.mode());
        assert_abs_diff_eq!(stalled.curvature(), fresh.curvature(), epsilon = 0.0);
        assert_eq!(stalled.fire_stall_count, fresh.fire_stall_count);
    }

    #[test]
    fn reset_then_step_matches_a_fresh_session_after_a_lanczos_estimate() {
        let surface = DiagonalQuadratic(array![-10.0, 80.0, 2.0]);
        let start = array![0.3, 0.2, 0.1];
        let mode = array![1.0, 0.0, 0.0];
        let config = MinModeConfig::default()
            .with_kind(MinModeKind::Lanczos)
            .with_krylov_dim(3)
            .with_saddle_check(None);
        let mut aged = MinModeSession::new(config.clone(), start, mode).unwrap();
        aged.step(&surface).unwrap();
        assert!(!aged.krylov_block().is_empty());
        aged.reset();
        let mut fresh =
            MinModeSession::new(config, aged.position().to_owned(), aged.mode().to_owned())
                .unwrap();
        assert_eq!(aged.krylov_block(), fresh.krylov_block());
        assert!(aged.krylov_block().is_empty());
        aged.step(&surface).unwrap();
        fresh.step(&surface).unwrap();
        assert_eq!(aged.position(), fresh.position());
        assert_eq!(aged.mode(), fresh.mode());
        assert_eq!(aged.krylov_block(), fresh.krylov_block());
        assert_abs_diff_eq!(aged.curvature(), fresh.curvature(), epsilon = 0.0);
    }

    #[test]
    fn reset_returns_the_iteration_count_to_zero() {
        let surface = DiagonalQuadratic(array![-1.0, 4.0, 8.0]);
        let start = array![0.4, 0.2, 0.1];
        let mode = array![1.0, 0.0, 0.0];
        let config = MinModeConfig::default().with_saddle_check(None);
        let mut aged = MinModeSession::new(config.clone(), start, mode).unwrap();
        let first = aged.step(&surface).unwrap();
        assert!(first.iteration >= 1);
        aged.reset();
        let mut fresh =
            MinModeSession::new(config, aged.position().to_owned(), aged.mode().to_owned())
                .unwrap();
        assert_eq!(aged.iteration, fresh.iteration);
        let aged_step = aged.step(&surface).unwrap();
        let fresh_step = fresh.step(&surface).unwrap();
        assert_eq!(aged_step.iteration, fresh_step.iteration);
        assert_eq!(aged.position(), fresh.position());
        assert_eq!(aged.mode(), fresh.mode());
        assert_abs_diff_eq!(aged.curvature(), fresh.curvature(), epsilon = 0.0);
    }

    /// Eight downhill translations grow the FIRE time step (`n_min` is
    /// 5, and the opening step has no velocity, so it does not count).
    /// `forget` keeps that step, so the next move differs from a new
    /// session until reset restores the configured opening value.
    #[test]
    fn reset_restores_the_fire_time_step_a_new_session_opens_with() {
        let surface = DiagonalQuadratic(array![-0.2, 4.0, 8.0]);
        let start = array![0.4, 0.0, 0.0];
        let mode = array![1.0, 0.0, 0.0];
        let config = MinModeConfig::default()
            .with_saddle_check(None)
            .with_time_step(0.25)
            .with_max_move(0.0)
            .with_force_tol(0.0);
        let mut aged = MinModeSession::new(config.clone(), start, mode).unwrap();
        for _ in 0..8 {
            let report = aged.step(&surface).unwrap();
            assert_eq!(report.status, MinModeStatus::Running);
        }
        aged.reset();
        let mut fresh =
            MinModeSession::new(config, aged.position().to_owned(), aged.mode().to_owned())
                .unwrap();
        assert_eq!(aged.fire_dt, fresh.fire_dt);
        aged.solver.start_fire_trace();
        fresh.solver.start_fire_trace();
        let aged_step = aged.step(&surface).unwrap();
        let fresh_step = fresh.step(&surface).unwrap();
        let aged_trace = aged.solver.take_fire_trace();
        let fresh_trace = fresh.solver.take_fire_trace();
        assert!(
            !aged_trace.is_empty(),
            "the reset session did not step with FIRE"
        );
        assert!(
            !fresh_trace.is_empty(),
            "the new session did not step with FIRE"
        );
        assert_eq!(aged_trace[0].dt.to_bits(), fresh_trace[0].dt.to_bits());
        assert_eq!(aged_step.status, fresh_step.status);
        assert_eq!(aged.position(), fresh.position());
        assert_eq!(aged.mode(), fresh.mode());
        assert_abs_diff_eq!(aged.curvature(), fresh.curvature(), epsilon = 0.0);
    }

    #[test]
    fn eigen_follow_climb_keeps_the_mode_on_a_valley_floor() {
        // The parallel gradient -0.05 is within half a step (0.5 * 1 *
        // 0.2 = 0.1) of the minimum along +x, so +x already climbs.
        let position = array![-0.05, 0.0, 0.0];
        let gradient = array![-0.05, 0.0, 0.0];
        let mode = array![1.0, 0.0, 0.0];
        let kept = super::climb_displacement(
            position.view(),
            gradient.view(),
            mode.view(),
            1.0,
            0.2,
            super::ClimbCap::Atom,
        );
        assert_abs_diff_eq!(kept[0], 0.15, epsilon = 1e-15);
        // Past half a step the gradient's sign decides, as before.
        let far = super::climb_displacement(
            position.view(),
            array![-0.3, 0.0, 0.0].view(),
            mode.view(),
            1.0,
            0.2,
            super::ClimbCap::Atom,
        );
        assert_abs_diff_eq!(far[0], -0.25, epsilon = 1e-15);
    }

    fn dimer_rotations(rotation_tol: f64, window: Option<bool>) -> (usize, Array1<f64>) {
        let config = MinModeConfig::default()
            .with_rotation_tol(rotation_tol)
            .with_rotation_angle_tol(0.0);
        let mut session = MinModeSession::new(
            config,
            array![0.3, 0.2, 0.1, 0.4],
            array![0.5, 0.5, 0.5, 0.5],
        )
        .unwrap();
        if let Some(window) = window {
            session.set_torque_window(Some(window));
        }
        let estimate = session.estimate_mode(&AnisotropicQuadratic).unwrap();
        (estimate.rotations, estimate.mode)
    }

    #[test]
    fn unset_torque_window_keeps_the_inferred_rotation() {
        for tol in [1e-4, 0.1, 1e-6] {
            let inferred = tol + 1e-12 >= super::STOCK_ROTATION_TOL;
            assert_eq!(
                dimer_rotations(tol, None),
                dimer_rotations(tol, Some(inferred))
            );
        }
    }

    fn torque(mode: &Array1<f64>) -> f64 {
        let h = array![-2.0 * mode[0], mode[1], 4.0 * mode[2], 8.0 * mode[3]];
        let force = &h - &(mode * mode.dot(&h));
        force.dot(&force).sqrt()
    }

    #[test]
    fn torque_window_off_rotates_to_the_residual() {
        let (window, early) = dimer_rotations(0.1, None);
        let (residual, mode) = dimer_rotations(0.1, Some(false));
        assert!(residual > window);
        assert!(torque(&early) > 0.1 && torque(&early) < 1.0);
        assert!(torque(&mode) <= 0.1);
    }

    #[test]
    fn torque_window_on_stops_a_tight_residual_early() {
        let (residual, _) = dimer_rotations(1e-6, None);
        let (window, early) = dimer_rotations(1e-6, Some(true));
        assert!(residual > window);
        assert!(torque(&early) < 1.0);
        assert_eq!((window, early), dimer_rotations(0.1, None));
    }

    fn windowed_estimate(
        kind: MinModeKind,
        rotation_tol: f64,
        window: Option<bool>,
        rel_tol: Option<f64>,
    ) -> (super::ModeEstimate, bool) {
        let config = MinModeConfig::default()
            .with_kind(kind)
            .with_rotation_tol(rotation_tol)
            .with_rotation_angle_tol(0.0);
        let mut session = MinModeSession::new(
            config,
            array![0.3, 0.2, 0.1, 0.4],
            array![1.0, 0.3, 0.2, 0.1],
        )
        .unwrap();
        session.set_torque_window(window);
        session.set_lanczos_rel_tol(rel_tol).unwrap();
        let estimate = session.estimate_mode(&AnisotropicQuadratic).unwrap();
        (estimate, session.lanczos_ready)
    }

    /// The seed's rotational force is 1.6, and one rotation in
    /// span{v, H v} leaves 0.71. One stock dimer rotation and the
    /// two-vector Lanczos estimate both take the lowest curvature in that
    /// plane (Zeng, Xiao, and Henkelman, J. Chem. Phys. 140, 044115
    /// (2014), doi:10.1063/1.4862410), and the torque window ends both
    /// there.
    #[test]
    fn stock_lanczos_stops_where_one_dimer_rotation_does() {
        let (dimer, _) = windowed_estimate(MinModeKind::Dimer, 1e-4, None, None);
        let (lanczos, ready) = windowed_estimate(MinModeKind::Lanczos, 1e-4, None, None);
        assert_eq!(dimer.rotations, 1);
        assert_eq!(lanczos.rotations, 2);
        assert_eq!(lanczos.evaluations, dimer.evaluations);
        assert_abs_diff_eq!(lanczos.curvature, dimer.curvature, epsilon = 1e-8);
        assert_abs_diff_eq!(lanczos.mode.dot(&dimer.mode).abs(), 1.0, epsilon = 1e-8);
        assert!(torque(&lanczos.mode) > 0.1 && torque(&lanczos.mode) < 1.0);
        assert!(!ready, "a windowed estimate is not a converged solve");
    }

    #[test]
    fn lanczos_residual_agreement_is_relative_and_guards_a_vanishing_ritz_value() {
        assert!(super::lanczos_residual_agreed(1.0e-6, -2.0, 2.0, 1.0e-4));
        assert!(!super::lanczos_residual_agreed(3.0e-4, -2.0, 2.0, 1.0e-4));
        assert!(super::lanczos_residual_agreed(1.0e-4, 0.0, 1.0, 1.0e-4));
        assert!(!super::lanczos_residual_agreed(2.0e-4, 0.0, 1.0, 1.0e-4));
        assert!(super::lanczos_residual_agreed(1.0e-4, 1.0e-15, 1.0, 1.0e-4));
        assert!(!super::lanczos_residual_agreed(
            1.0e-4, 1.0e-12, 1.0, 1.0e-4
        ));
        assert!(!super::lanczos_residual_agreed(0.0, f64::NAN, 1.0, 1.0e-4));
        assert!(!super::lanczos_residual_agreed(f64::NAN, -1.0, 1.0, 1.0e-4));
        assert!(!super::lanczos_residual_agreed(0.0, -1.0, 1.0, f64::NAN));
        assert!(!super::lanczos_residual_agreed(0.0, -1.0, f64::NAN, 1.0e-4));
        assert!(!super::lanczos_residual_agreed(0.0, -1.0, -1.0, 1.0e-4));
        assert!(super::lanczos_residual_below(0.5, -2.0, 2.0, 1.0));
        assert!(!super::lanczos_residual_below(2.5, -2.0, 2.0, 1.0));
        assert!(!super::lanczos_residual_below(2.0, -2.0, 2.0, 1.0));
        assert!(super::lanczos_residual_below(0.5, 0.0, 1.0, 1.0));
        assert!(!super::lanczos_residual_below(1.0, 0.0, 1.0, 1.0));
        assert!(!super::lanczos_residual_below(0.0, f64::INFINITY, 1.0, 1.0));
        assert!(super::ritz_value_agreed(-1.00001, -1.0, 1.0, 1.0e-4));
        assert!(!super::ritz_value_agreed(-1.001, -1.0, 1.0, 1.0e-4));
        assert!(super::ritz_value_agreed(1.0e-5, 0.0, 1.0, 1.0e-4));
        assert!(!super::ritz_value_agreed(1.0e-3, 0.0, 1.0, 1.0e-4));
        assert!(super::ritz_value_agreed(0.0, 0.0, 0.0, 1.0e-4));
        assert!(!super::ritz_value_agreed(1.0e-300, 0.0, 0.0, 1.0e-4));
        assert!(!super::ritz_value_agreed(0.0, f64::NAN, 1.0, 1.0e-4));
    }

    /// Every near-zero fallback reads its tolerance against the
    /// curvature scale, so multiplying the residual, the Ritz values,
    /// and the scale by one factor never changes a decision.
    #[test]
    fn lanczos_stop_tests_ignore_a_common_curvature_factor() {
        let tol = 1.0e-2;
        let cases = [
            (5.0e-3, 0.0, 0.0, 1.0),
            (2.0e-2, 0.0, 0.0, 1.0),
            (5.0e-3, 1.0e-16, 3.0e-15, 1.0),
            (5.0e-1, -0.5, -0.501, 2.0),
            (2.0e-2, -1.0, -1.05, 1.5),
        ];
        let verdicts = |factor: f64| {
            cases
                .iter()
                .map(|&(residual, theta, old, scale)| {
                    let (residual, theta, old, scale) = (
                        residual * factor,
                        theta * factor,
                        old * factor,
                        scale * factor,
                    );
                    [
                        super::lanczos_residual_agreed(residual, theta, scale, tol),
                        super::lanczos_residual_below(residual, theta, scale, super::TORQUE_MAX),
                        super::ritz_value_agreed(theta, old, scale, tol),
                    ]
                })
                .collect::<Vec<_>>()
        };
        let unit = verdicts(1.0);
        assert!(unit.iter().flatten().any(|&v| v) && unit.iter().flatten().any(|&v| !v));
        for factor in [1.0e-3, 1.0e3] {
            assert_eq!(verdicts(factor), unit, "factor {factor:e}");
        }
    }

    /// `V = 0.5 x' diag(h) x` with the exact Hessian action, so a
    /// rescaled `h` scales every product by the same factor.
    struct ExactDiagonal(Array1<f64>);

    impl PointSurface for ExactDiagonal {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let gradient = &self.0 * &x;
            Ok((0.5 * x.dot(&gradient), gradient))
        }

        fn hessian_vector(
            &self,
            _x: ArrayView1<f64>,
            v: ArrayView1<f64>,
        ) -> Result<Option<Array1<f64>>, SaddleError> {
            Ok(Some(&self.0 * &v))
        }
    }

    /// The seed has a zero Rayleigh quotient on `diag(-1, 1, 2, 4)`, so
    /// the first Ritz value takes the near-zero fallback. An absolute
    /// fallback accepted the seed at the 1e-13 scale and refined it at
    /// 1e13. Against the curvature scale every scale makes the same
    /// decisions and returns the same lowest mode.
    #[test]
    fn lanczos_decisions_survive_rescaling_the_hessian() {
        let estimate = |factor: f64| {
            let surface = ExactDiagonal(array![-1.0, 1.0, 2.0, 4.0] * factor);
            let seed = array![8.0_f64.sqrt(), 2.0_f64.sqrt(), 1.0, 1.0] / 12.0_f64.sqrt();
            let config = MinModeConfig::default()
                .with_kind(MinModeKind::Lanczos)
                .with_rotation_tol(1.0e-2);
            let mut session =
                MinModeSession::new(config, array![0.1, -0.2, 0.05, 0.3], seed).expect("session");
            session.set_torque_window(Some(false));
            let estimate = session.estimate_mode(&surface).expect("estimate");
            (estimate, session.lanczos_ready)
        };
        let (unit, unit_ready) = estimate(1.0);
        assert!(unit.rotations > 1, "{unit:?}");
        assert!(unit_ready, "{unit:?}");
        assert!(unit.mode[0].abs() > 0.99, "{unit:?}");
        for factor in [1.0e-13, 1.0e13] {
            let (scaled, ready) = estimate(factor);
            assert_eq!(scaled.rotations, unit.rotations, "factor {factor:e}");
            assert_eq!(scaled.evaluations, unit.evaluations, "factor {factor:e}");
            assert_eq!(ready, unit_ready, "factor {factor:e}");
            assert_abs_diff_eq!(scaled.curvature / factor, unit.curvature, epsilon = 1e-8);
            assert_abs_diff_eq!(scaled.mode.dot(&unit.mode).abs(), 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn lanczos_reads_the_torque_window() {
        let stock = windowed_estimate(MinModeKind::Lanczos, 1e-4, None, None).0;
        let (off, ready) = windowed_estimate(MinModeKind::Lanczos, 1e-4, Some(false), None);
        assert!(off.rotations > stock.rotations);
        assert!(torque(&off.mode) <= 1e-4);
        assert!(ready);
        let tight = windowed_estimate(MinModeKind::Lanczos, 1e-6, None, None).0;
        assert_eq!(tight.rotations, off.rotations);
        let forced = windowed_estimate(MinModeKind::Lanczos, 1e-6, Some(true), None).0;
        assert_eq!(forced.rotations, stock.rotations);
        assert_eq!(forced.mode, stock.mode);
    }

    #[test]
    fn a_host_relative_tolerance_keeps_its_own_lanczos_stop() {
        let host = windowed_estimate(MinModeKind::Lanczos, 1e-4, None, Some(1e-3)).0;
        let off = windowed_estimate(MinModeKind::Lanczos, 1e-4, Some(false), Some(1e-3)).0;
        assert_eq!(host.rotations, off.rotations);
        assert_eq!(host.mode, off.mode);
        assert!(host.rotations > 2);
        let on = windowed_estimate(MinModeKind::Lanczos, 1e-4, Some(true), Some(1e-3)).0;
        assert_eq!(on.rotations, 2);
    }

    #[test]
    fn dimer_rotation_history_crosses_translations_but_resets_at_boundaries() {
        let config = MinModeConfig::default()
            .with_max_rotations(1)
            .with_rotation_tol(1e-12);
        let mut session = MinModeSession::new(config, array![0.3, 0.2], array![0.8, 0.6]).unwrap();
        session.estimate_mode(&RotatingQuadratic).unwrap();
        assert!(session.dimer_rotation.force.is_some());
        assert!(session.dimer_rotation.direction.is_some());

        session.set_position(array![0.31, 0.19], None).unwrap();
        assert!(session.dimer_rotation.force.is_some());
        assert!(session.dimer_rotation.direction.is_some());

        session.set_mode(array![1.0, 0.0]).unwrap();
        assert!(session.dimer_rotation.force.is_none());
        assert!(session.dimer_rotation.direction.is_none());
        assert!(session.curvature().is_nan());

        session.estimate_mode(&RotatingQuadratic).unwrap();
        session.reset();
        assert!(session.dimer_rotation.force.is_none());
        assert!(session.dimer_rotation.direction.is_none());
        assert!(session.curvature().is_nan());
    }

    #[test]
    fn failed_dimer_action_preserves_rotation_history() {
        let config = MinModeConfig::default()
            .with_rotation_tol(1e-12)
            .with_rotation_angle_tol(std::f64::consts::PI);
        let mut session = MinModeSession::new(config, array![0.3, 0.2], array![0.8, 0.6]).unwrap();
        session.estimate_mode(&RotatingQuadratic).unwrap();
        assert!(session.dimer_rotation.force.is_some());
        assert!(session.dimer_rotation.direction.is_some());
        let force = session.dimer_rotation.force.clone();
        let direction = session.dimer_rotation.direction.clone();
        let centre = session.dimer_rotation.centre.clone();

        session
            .set_position(array![0.31, 0.19], Some(array![-0.62, 0.76]))
            .unwrap();
        assert!(session.estimate_mode(&FailingSurface).is_err());
        assert_eq!(session.dimer_rotation.force, force);
        assert_eq!(session.dimer_rotation.direction, direction);
        assert_eq!(session.dimer_rotation.centre, centre);
    }

    #[test]
    fn an_excluded_mode_drops_unrelated_curvature_and_rotation_history() {
        let config = MinModeConfig::default()
            .with_kind(super::MinModeKind::Lanczos)
            .with_rotation_tol(1e-12)
            .with_krylov_dim(3);
        let position = array![0.3, 0.2, 0.1, 0.4];
        let mut session =
            MinModeSession::new(config, position, array![1.0, 0.0, 0.0, 0.0]).unwrap();
        let fallback =
            Array1::from_iter((0..3).map(|i| ((i + 1) as f64 * 1.618033988749895).sin()));
        session.curvature = fallback
            .iter()
            .zip([1.0, 4.0, 8.0])
            .map(|(component, eigenvalue)| component * component * eigenvalue)
            .sum::<f64>()
            / fallback.dot(&fallback);
        session.dimer_rotation.force = Some(array![1.0, 0.0, 0.0, 0.0]);
        session.dimer_rotation.direction = Some(array![0.0, 1.0, 0.0, 0.0]);
        session.dimer_rotation.centre = Some(session.position().to_owned());

        assert!(
            session
                .step(&ExcludedAnisotropicQuadratic)
                .unwrap()
                .rotations
                > 1
        );
        assert!(session.dimer_rotation.force.is_none());
        assert!(session.dimer_rotation.direction.is_none());
    }

    /// V = -50 x^2 + 40 y^2 + z^2. The force along x is small and the
    /// force along y is not. Curvature along x is -100.
    struct StiffWell;

    impl PointSurface for StiffWell {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (a, b, c) = (x[0], x[1], x[2]);
            Ok((
                -50.0 * a * a + 40.0 * b * b + c * c,
                array![-100.0 * a, 80.0 * b, 2.0 * c],
            ))
        }
    }

    #[test]
    fn fire_on_a_stiff_negative_curvature_steps_inside_the_cap() {
        let config = MinModeConfig::default()
            .with_force_tol(1e-3)
            .with_max_move(0.2)
            .with_rotation_angle_tol(0.0);
        let start = array![0.002, 0.15, 0.0];
        let mode = array![1.0, 0.0, 0.0];
        let mut plain = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
        plain.set_fire_curvature(false);
        let stalled = plain.run(&StiffWell, 25).unwrap();
        assert!(
            plain.position()[1].abs() > 1e-2 || stalled.status != super::MinModeStatus::Converged,
            "stock FIRE finished at {}",
            plain.position()
        );

        let mut stable = MinModeSession::new(config, start, mode).unwrap();
        let report = stable.run(&StiffWell, 260).unwrap();
        assert_eq!(report.status, super::MinModeStatus::Converged, "{report:?}");
        assert!(stable.position()[1].abs() < 1e-2, "{}", stable.position());
    }

    #[test]
    fn cg_translate_on_a_stiff_well_converges() {
        let config = MinModeConfig::default()
            .with_force_tol(1e-3)
            .with_max_move(0.2)
            .with_rotation_angle_tol(0.0);
        let mut session =
            MinModeSession::new(config, array![0.002, 0.15, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        session.set_cg_translate(true);
        let report = session.run(&StiffWell, 30).unwrap();
        assert_eq!(report.status, super::MinModeStatus::Converged, "{report:?}");
        assert!(session.position()[0].abs() < 1e-2, "{}", session.position());
        assert!(session.position()[1].abs() < 1e-2, "{}", session.position());
    }

    #[test]
    fn projected_lbfgs_on_a_stiff_well_converges() {
        let config = MinModeConfig::default()
            .with_force_tol(1e-3)
            .with_max_move(0.2)
            .with_method(Method::lbfgs_with_memory(10))
            .with_rotation_angle_tol(0.0);
        let mut session =
            MinModeSession::new(config, array![0.002, 0.15, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        session.set_lbfgs_project(true);
        let report = session.run(&StiffWell, 40).unwrap();
        assert_eq!(report.status, super::MinModeStatus::Converged, "{report:?}");
        assert!(session.position()[1].abs() < 1e-2, "{}", session.position());
    }

    #[test]
    fn rotation_force_error_is_curvature_length_and_sine() {
        let (force, curvature, length) = (0.3, -2.0, 0.2);
        let omega = 0.5 * f64::atan2(force, 2.0);
        let rotated: f64 = curvature - force * omega.tan();
        let expected = rotated.abs() * length * omega.sin();
        assert_abs_diff_eq!(
            rotation_force_error(force, curvature, length),
            expected,
            epsilon = 1e-15
        );
        assert_eq!(rotation_force_error(0.0, -2.0, length), 0.0);
        assert!(rotation_force_converged(force, curvature, length, 1.0));
        assert!(!rotation_force_converged(force, curvature, length, 1e-3));
        assert!(!rotation_force_converged(force, curvature, 0.0, 1.0));
        assert!(!rotation_force_converged(force, curvature, length, 0.0));
        assert!(!rotation_force_converged(f64::NAN, curvature, length, 1.0));
    }

    #[test]
    fn rotation_force_error_sees_the_mode_between_two_curvatures() {
        // Halfway between curvatures -1 and +1 the dimer reads C = 0 and
        // a rotational force of 1; the rotation ahead is 45 degrees and
        // reaches the curvature -1.
        let error = rotation_force_error(1.0, 0.0, 1.0);
        assert_abs_diff_eq!(error, (0.25 * std::f64::consts::PI).sin(), epsilon = 1e-12);
        assert!(!rotation_force_converged(1.0, 0.0, 1.0, 0.5));
    }

    fn stopped_rotations(kind: MinModeKind, length: f64) -> (usize, usize, Array1<f64>, f64) {
        let config = MinModeConfig::default()
            .with_kind(kind)
            .with_rotation_tol(1e-6)
            .with_rotation_angle_tol(0.0);
        let mut session = MinModeSession::new(
            config,
            array![0.3, 0.2, 0.1, 0.4],
            array![0.5, 0.5, 0.5, 0.5],
        )
        .unwrap();
        session.set_rotation_stop_length(length).unwrap();
        assert_eq!(session.rotation_stop_length(), length);
        let estimate = session.estimate_mode(&AnisotropicQuadratic).unwrap();
        (
            estimate.rotations,
            estimate.evaluations,
            estimate.mode,
            estimate.curvature,
        )
    }

    #[test]
    fn rotation_stop_length_zero_keeps_the_rotation() {
        for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
            let config = MinModeConfig::default()
                .with_kind(kind)
                .with_rotation_tol(1e-6)
                .with_rotation_angle_tol(0.0);
            let mut session = MinModeSession::new(
                config,
                array![0.3, 0.2, 0.1, 0.4],
                array![0.5, 0.5, 0.5, 0.5],
            )
            .unwrap();
            assert_eq!(session.rotation_stop_length(), 0.0);
            let stock = session.estimate_mode(&AnisotropicQuadratic).unwrap();
            let (rotations, evaluations, mode, _) = stopped_rotations(kind, 0.0);
            assert_eq!(rotations, stock.rotations);
            assert_eq!(evaluations, stock.evaluations);
            assert_eq!(mode, stock.mode);
        }
    }

    #[test]
    fn rotation_stop_length_ends_the_rotation_inside_the_force_tolerance() {
        for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
            let (full, full_calls, full_mode, _) = stopped_rotations(kind, 0.0);
            let (cut, cut_calls, cut_mode, curvature) = stopped_rotations(kind, 1e-3);
            assert!(cut < full, "{kind:?}: {cut} vs {full}");
            assert!(
                cut_calls < full_calls,
                "{kind:?}: {cut_calls} vs {full_calls}"
            );
            // The mode that stopped early still has the negative
            // curvature and the force error it promised: the angle to
            // the converged mode, times the curvature and the length, is
            // inside force_tol.
            assert!(curvature < 0.0, "{kind:?}: {curvature}");
            let cosine = cut_mode.dot(&full_mode).abs().min(1.0);
            let sine = (1.0 - cosine * cosine).sqrt();
            assert!(2.0 * 1e-3 * sine <= 1e-3, "{kind:?}: sin {sine}");
        }
    }

    #[test]
    fn rotation_stop_length_rejects_negative_and_non_finite_values() {
        let mut session = MinModeSession::new(
            MinModeConfig::default(),
            array![0.3, 0.2, 0.1, 0.4],
            array![0.5, 0.5, 0.5, 0.5],
        )
        .unwrap();
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(session.set_rotation_stop_length(bad).is_err());
        }
        assert_eq!(session.rotation_stop_length(), 0.0);
    }
}
