//! Signed per-atom direction cover and arm picker for saddle searches
//! from one minimum.
//!
//! Adaptive kinetic Monte Carlo needs, for each minimum it visits, the
//! low saddles out of it and some measure of how much of the search
//! space has been tried (Xu and Henkelman, *J. Chem. Phys.* **2008**,
//! *129*, 114104, <https://doi.org/10.1063/1.2976010>). Pedersen,
//! Hafstein and Jónsson (*SIAM J. Sci. Comput.* **2011**, *33*, 633,
//! <https://doi.org/10.1137/100792743>) show that which saddles a
//! minimum-mode search reaches, and how often, is set by the initial
//! displacement: its centre, its radius and the width of the Gaussian
//! on the neighbouring atoms. Béland, Osetsky, Stoller and Xu
//! (*Comput. Mater. Sci.* **2015**, *100*, 124,
//! <https://doi.org/10.1016/j.commatsci.2014.12.001>) compare two
//! on-the-fly catalogues built on such local starts. This module supplies
//! only those starts and the order to try them in. The climb itself is
//! [`crate::ArtSession`]: push, perpendicular relaxation, Lanczos, the
//! lowest-mode climb, the index-1 certificate and the two quenches.
//!
//! # Cover
//!
//! Every atom the caller marks active is its own centre. Each centre
//! carries the same evenly spread set of signed unit vectors on its own
//! sphere \(S^2\): a direction \(\mathbf{u}\) and its opposite are
//! separate candidates, because they lead to different saddles. The
//! points are spread by the Thomson-like repulsion
//! \(\sum_{i\neq j} 1/\gamma_{ij}\) on the geodesic
//! \(\gamma_{ij}=\arccos(\mathbf{u}_i\cdot\mathbf{u}_j)\) of Plasencia
//! Gutiérrez, Argáez and Jónsson (*J. Chem. Theory Comput.* **2017**,
//! *13*, 125, <https://doi.org/10.1021/acs.jctc.5b01216>), relaxed
//! Gauss-Seidel from a seeded start, and ordered farthest point first. An
//! arm is one (centre, cover point) pair. Its seed is the cover vector on
//! the centre and, optionally, independent Gaussian components on every
//! atom within [`DirectionCoverConfig::kick_radius`] of the centre.
//!
//! The ridge walk the seed starts is the one in the cited literature:
//! the gradient-only uphill tracing of Quapp (*Chem. Phys. Lett.*
//! **1996**, *253*, 286, <https://doi.org/10.1016/0009-2614(96)00255-2>)
//! and the reduced-gradient following of Quapp, Hirsch, Imig and
//! Heidrich (*J. Comput. Chem.* **1998**, *19*, 1087,
//! <https://doi.org/10.1002/(SICI)1096-987X(19980715)19:9%3C1087::AID-JCC9%3E3.0.CO;2-M>),
//! the activation-relaxation technique of Barkema and Mousseau
//! (*Phys. Rev. Lett.* **1996**, *77*, 4358,
//! <https://doi.org/10.1103/PhysRevLett.77.4358>) and Malek and Mousseau
//! (*Phys. Rev. E* **2000**, *62*, 7723,
//! <https://doi.org/10.1103/PhysRevE.62.7723>), and the dimer of
//! Henkelman and Jónsson (*J. Chem. Phys.* **1999**, *111*, 7010,
//! <https://doi.org/10.1063/1.480097>). Xiao, Wu and Henkelman (*J.
//! Chem. Phys.* **2014**, *141*, 164111,
//! <https://doi.org/10.1063/1.4898664>) confine the search to the basin
//! of the start so that the saddle it reaches leads out of that basin.
//! Here the same requirement is checked afterwards: an arm counts as a
//! saddle only when [`crate::ArtStatus::Connected`] says one quench
//! returned to the start and the other did not.
//!
//! # Pick
//!
//! [`ArmPolicy::Gap`] scores an untried arm by its angular gap, the
//! angle to the nearest tried direction on the same centre, times the
//! number of distinct states among the start and the products of the
//! three nearest tried directions. The start always counts, so a hole
//! bounded by arms reaching products A and B scores three times its
//! angle, one reaching only A twice, and one whose edges all came back
//! or failed once. Failed, non-converged and still pending arms add no
//! state. With nothing tried on a centre the score is \(\pi\). A hole
//! bounded by arms that led to different minima is where an unseen
//! saddle can sit, so it is tried before a hole whose edges all led to
//! the same place.
//! [`ArmPolicy::ExpectedImprovement`] is the expected improvement of
//! Jones, Schonlau and Welch (*J. Global Optim.* **1998**, *13*, 455,
//! <https://doi.org/10.1023/A:1008306431147>) on the barrier, from a
//! Gaussian process on each centre's sphere with a Gaussian kernel of
//! the chordal distance, times the gap fraction
//! \(\gamma/\pi\). [`ArmPolicy::Auto`] takes the expected improvement
//! once two different barriers are known and the gap score before; the
//! suggestion names the policy that produced it.
//!
//! # Coverage readout
//!
//! [`CoverageReport`] gives, per centre, the exact covering radius of
//! the tried signed directions on \(S^2\) ([`covering_radius_s2`]), the
//! records of every arm, the distinct saddles against draws and the
//! frequency of frequencies a Good-Turing (Good, *Biometrika* **1953**,
//! *40*, 237, <https://doi.org/10.1093/biomet/40.3-4.237>) or
//! capture-recapture estimate needs, and the next suggested arm with its
//! score.
//!
//! Nothing here builds a Hessian. Arms run in parallel on rayon's pool
//! in batches of [`DirectionCoverConfig::batch`]; a batch is picked
//! before it runs and recorded in pick order, so the records depend on
//! the seed and the inputs and not on the thread count.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;

use crate::art::{ArtConfig, ArtOutcome, ArtSeed, ArtSession, ArtStatus};
use crate::band::Cell;
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Relative energy change that ends the cover relaxation.
const COVER_ETOL: f64 = 1e-9;
/// Smallest step of the cover relaxation.
const COVER_STEP_MIN: f64 = 1e-14;
/// Largest step of the cover relaxation.
const COVER_STEP_MAX: f64 = 0.2;
/// Shrink factor of a rejected cover step.
const COVER_STEP_SHRINK: f64 = 2.5;
/// Tried directions whose products count toward a gap's boundary.
const BOUNDARY_NEIGHBOURS: usize = 3;
/// Diagonal of the barrier kernel matrix added per unit variance.
const KERNEL_NUGGET: f64 = 1e-8;

/// Which score orders the untried arms.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArmPolicy {
    /// Angular gap times the distinct states on its boundary, the start
    /// included.
    Gap,
    /// Expected improvement of the barrier times the gap fraction.
    /// [`DirectionCover::suggest`] returns [`SaddleError::Invalid`]
    /// until two different barriers are known.
    ExpectedImprovement,
    /// [`ArmPolicy::ExpectedImprovement`] once it can score, otherwise
    /// [`ArmPolicy::Gap`].
    Auto,
}

/// Cover, neighbour kick, climb, de-duplication and pick settings.
///
/// Lengths and energies are the caller's. Start from
/// [`DirectionCoverConfig::default`] and chain the `with_*` setters.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DirectionCoverConfig {
    /// Signed cover points on each centre's sphere. 64 by default; at
    /// least 4.
    pub cover_points: usize,
    /// Gauss-Seidel sweeps of the cover relaxation. 400 by default.
    pub cover_sweeps: usize,
    /// Atoms within this distance of a centre get a Gaussian kick. Zero,
    /// the default, kicks no neighbour.
    pub kick_radius: f64,
    /// Standard deviation of each kick component, relative to the unit
    /// cover vector on the centre. 0.1 by default.
    pub kick_width: f64,
    /// Periodic cell for the neighbour distances and the structure
    /// distances. `None` is an open system.
    pub cell: Option<Cell>,
    /// The ART nouveau event every arm runs.
    pub art: ArtConfig,
    /// Phase cap passed to [`ArtSession::run`]. 2000 by default.
    pub art_max_steps: usize,
    /// Saddles closer than this in energy and in
    /// [`DirectionCoverConfig::saddle_rmsd_tol`] are one saddle.
    pub saddle_energy_tol: f64,
    /// Per-atom root-mean-square distance under which two saddles of
    /// close energy are one saddle.
    pub saddle_rmsd_tol: f64,
    /// Product minima closer than this in energy and in
    /// [`DirectionCoverConfig::minimum_rmsd_tol`] are one state.
    pub minimum_energy_tol: f64,
    /// Per-atom root-mean-square distance under which two minima of
    /// close energy are one state.
    pub minimum_rmsd_tol: f64,
    /// Score that orders the untried arms.
    pub policy: ArmPolicy,
    /// Length scale of the expected-improvement kernel, a Gaussian of
    /// the chordal distance between unit directions. The chord is the
    /// angle in radians for small angles. 0.6 by default; any finite
    /// positive value keeps the kernel positive definite.
    pub length_scale: f64,
    /// Margin subtracted from the best barrier in the expected
    /// improvement. Zero by default.
    pub ei_margin: f64,
    /// Arms picked together and run in parallel. 1 by default.
    pub batch: usize,
}

impl Default for DirectionCoverConfig {
    fn default() -> Self {
        Self {
            cover_points: 64,
            cover_sweeps: 400,
            kick_radius: 0.0,
            kick_width: 0.1,
            cell: None,
            art: ArtConfig::default(),
            art_max_steps: 2000,
            saddle_energy_tol: 1e-3,
            saddle_rmsd_tol: 0.05,
            minimum_energy_tol: 1e-3,
            minimum_rmsd_tol: 0.05,
            policy: ArmPolicy::Gap,
            length_scale: 0.6,
            ei_margin: 0.0,
            batch: 1,
        }
    }
}

with_setters!(DirectionCoverConfig {
    with_cover_points => cover_points: val usize,
    with_cover_sweeps => cover_sweeps: val usize,
    with_kick_radius => kick_radius: val f64,
    with_kick_width => kick_width: val f64,
    with_cell => cell: opt Cell,
    with_art => art: val ArtConfig,
    with_art_max_steps => art_max_steps: val usize,
    with_saddle_energy_tol => saddle_energy_tol: val f64,
    with_saddle_rmsd_tol => saddle_rmsd_tol: val f64,
    with_minimum_energy_tol => minimum_energy_tol: val f64,
    with_minimum_rmsd_tol => minimum_rmsd_tol: val f64,
    with_policy => policy: val ArmPolicy,
    with_length_scale => length_scale: val f64,
    with_ei_margin => ei_margin: val f64,
    with_batch => batch: val usize,
});

impl DirectionCoverConfig {
    fn validate(&self) -> Result<(), SaddleError> {
        if self.cover_points < 4 {
            return Err(SaddleError::Invalid(
                "direction cover needs at least four points per centre".into(),
            ));
        }
        if self.art_max_steps == 0 || self.batch == 0 {
            return Err(SaddleError::Invalid(
                "ART phase cap and batch size must be at least one".into(),
            ));
        }
        for (name, value) in [
            ("kick radius", self.kick_radius),
            ("kick width", self.kick_width),
            ("expected-improvement margin", self.ei_margin),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(SaddleError::Invalid(format!(
                    "direction cover {name} must be finite and non-negative"
                )));
            }
        }
        for (name, value) in [
            ("saddle energy tolerance", self.saddle_energy_tol),
            ("saddle distance tolerance", self.saddle_rmsd_tol),
            ("minimum energy tolerance", self.minimum_energy_tol),
            ("minimum distance tolerance", self.minimum_rmsd_tol),
            ("kernel length scale", self.length_scale),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(SaddleError::Invalid(format!(
                    "direction cover {name} must be finite and positive"
                )));
            }
        }
        if let Some(cell) = &self.cell {
            let h = cell.0;
            let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
                - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
                + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
            if !(det.is_finite() && det.abs() > 0.0) {
                return Err(SaddleError::Invalid(
                    "direction cover cell must be finite and non-singular".into(),
                ));
            }
        }
        Ok(())
    }
}

/// When [`DirectionCover::run`] stops launching arms.
///
/// At least one limit must be set. The force-call limit is checked
/// between batches, so the last batch can pass it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoverBudget {
    /// Most arms recorded on this minimum, counting earlier records.
    pub max_arms: Option<usize>,
    /// Most surface evaluations, counting earlier records and the
    /// energy of the minimum.
    pub max_force_calls: Option<usize>,
}

with_setters!(CoverBudget {
    with_max_arms => max_arms: opt usize,
    with_max_force_calls => max_force_calls: opt usize,
});

/// One candidate: a centre atom and a point of its cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArmId {
    /// Atom index of the centre.
    pub atom: usize,
    /// Index into [`DirectionCover::cover`].
    pub point: usize,
}

/// Why an arm that did not reach a connected saddle stopped.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ArmFailure {
    /// The stationary point is not a converged index-1 saddle.
    NotIndexOne,
    /// A quench energy is not below the saddle.
    NotDownhill,
    /// Both quenches are distinct minima and neither is the start.
    Disconnected,
    /// The lowest curvature became non-negative during the climb, so
    /// the event stopped before any quench.
    LostCurvature,
    /// ART could not recover the tracked index-1 mode.
    IndexOneLost {
        /// Recovery attempts made before the arm stopped.
        attempts: usize,
        /// Curvature of the rejected mode.
        curvature: f64,
        /// Absolute overlap with the last accepted mode.
        overlap: f64,
    },
    /// The event returned an error: a surface evaluation, or a phase
    /// solver such as the perpendicular relaxation that ran out of
    /// steps. The message is the error's.
    Error(String),
}

/// Verdict of one arm.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ArmOutcome {
    /// An index-1 saddle connected to the start, not seen before.
    NewSaddle,
    /// A connected saddle already recorded under this saddle id.
    DuplicateSaddle(usize),
    /// Both quenches came back to the start.
    ReturnedToStart,
    /// The event ended without a connected saddle.
    Failed(ArmFailure),
    /// A push, climb, quench or phase budget ran out.
    NonConverged,
}

#[derive(Clone, Debug)]
enum TrialKind {
    Connected {
        saddle: Array1<f64>,
        saddle_energy: f64,
        product: Array1<f64>,
        product_energy: f64,
    },
    Returned,
    Failed(ArmFailure),
    NonConverged,
}

/// What one arm produced, before it is compared with the saddles and
/// minima already recorded.
///
/// [`DirectionCover::run_arm`] makes one. A caller that runs its own
/// search from [`DirectionCover::seed`] builds one with the
/// constructors and passes it to [`DirectionCover::record`].
#[derive(Clone, Debug)]
pub struct ArmTrial {
    arm: ArmId,
    kind: TrialKind,
    force_calls: usize,
}

impl ArmTrial {
    /// A saddle connected to the start, with the minimum on its far side.
    pub fn connected(
        arm: ArmId,
        saddle: Array1<f64>,
        saddle_energy: f64,
        product: Array1<f64>,
        product_energy: f64,
        force_calls: usize,
    ) -> Self {
        Self {
            arm,
            kind: TrialKind::Connected {
                saddle,
                saddle_energy,
                product,
                product_energy,
            },
            force_calls,
        }
    }

    /// The arm came back to the start.
    pub fn returned(arm: ArmId, force_calls: usize) -> Self {
        Self {
            arm,
            kind: TrialKind::Returned,
            force_calls,
        }
    }

    /// The arm ended without a connected saddle.
    pub fn failed(arm: ArmId, failure: ArmFailure, force_calls: usize) -> Self {
        Self {
            arm,
            kind: TrialKind::Failed(failure),
            force_calls,
        }
    }

    /// A budget inside the arm ran out.
    pub fn non_converged(arm: ArmId, force_calls: usize) -> Self {
        Self {
            arm,
            kind: TrialKind::NonConverged,
            force_calls,
        }
    }

    /// The arm this trial belongs to.
    #[must_use]
    pub fn arm(&self) -> ArmId {
        self.arm
    }

    /// Surface evaluations the arm spent.
    #[must_use]
    pub fn force_calls(&self) -> usize {
        self.force_calls
    }
}

/// One recorded arm.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ArmRecord {
    /// Centre and cover point.
    pub arm: ArmId,
    /// Signed unit cover vector on the centre.
    pub direction: [f64; 3],
    /// Verdict.
    pub outcome: ArmOutcome,
    /// Saddle id for a new or duplicate saddle.
    pub saddle: Option<usize>,
    /// Product state id for a connected saddle.
    pub product: Option<usize>,
    /// Saddle energy minus the energy of the start.
    pub barrier: Option<f64>,
    /// Rate prefactor, when the caller attaches one with
    /// [`DirectionCover::set_prefactor`].
    pub prefactor: Option<f64>,
    /// Surface evaluations of this arm.
    pub force_calls: usize,
}

/// A distinct saddle out of the start.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct SaddleEntry {
    /// Saddle id, in order of discovery.
    pub id: usize,
    /// Saddle geometry from the first arm that found it.
    pub position: Array1<f64>,
    /// Saddle energy.
    pub energy: f64,
    /// Saddle energy minus the energy of the start.
    pub barrier: f64,
    /// Product state on the far side.
    pub product: usize,
    /// First arm that reached it.
    pub first_arm: ArmId,
    /// Arms that reached it, the first included.
    pub hits: usize,
}

/// A distinct minimum reached past a saddle.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ProductState {
    /// Product id, in order of discovery.
    pub id: usize,
    /// Geometry from the first arm that reached it.
    pub position: Array1<f64>,
    /// Energy.
    pub energy: f64,
}

/// Next arm to try and the score that chose it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmSuggestion {
    /// Centre and cover point.
    pub arm: ArmId,
    /// Acquisition score: radians times the boundary state count for
    /// [`ArmPolicy::Gap`], energy times the gap fraction for
    /// [`ArmPolicy::ExpectedImprovement`].
    pub score: f64,
    /// Policy that produced the score, never [`ArmPolicy::Auto`].
    pub policy: ArmPolicy,
}

/// Coverage of one centre.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct AtomCoverage {
    /// Atom index.
    pub atom: usize,
    /// Arms recorded on this centre.
    pub tried: usize,
    /// [`covering_radius_s2`] of the tried directions, radians. Pi when
    /// none is tried.
    pub covering_radius: f64,
    /// Distinct states (the start and product ids) the tried arms led to.
    pub states: usize,
}

/// Coverage of the searched space of one minimum.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageReport {
    /// Dimension of the searched space, three per centre.
    pub subspace_dim: usize,
    /// Largest per-centre covering radius, radians.
    pub covering_radius: f64,
    /// One entry per centre, in atom order.
    pub atoms: Vec<AtomCoverage>,
    /// Every recorded arm, in record order.
    pub records: Vec<ArmRecord>,
    /// Distinct saddles, in discovery order.
    pub saddles: Vec<SaddleEntry>,
    /// Distinct product minima, in discovery order.
    pub products: Vec<ProductState>,
    /// Distinct saddles after each record.
    pub discovery: Vec<usize>,
    /// Arms that reached a connected saddle.
    pub saddle_draws: usize,
    /// Entry `k - 1` is the number of saddles reached by exactly `k` arms.
    pub frequency_counts: Vec<usize>,
    /// Surface evaluations, the energy of the minimum included.
    pub force_calls: usize,
    /// What the configured policy would try next. `None` once every arm
    /// is recorded.
    pub next: Option<ArmSuggestion>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Start,
    Product(usize),
    Unlabelled,
}

/// The cover, the records and the picker for one minimum.
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::{
///     ArmOutcome, ArtConfig, CoverBudget, DirectionCover, DirectionCoverConfig, MinModeConfig,
///     MinModeKind, PointSurface, SaddleError,
/// };
///
/// struct DoubleWell;
///
/// impl PointSurface for DoubleWell {
///     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (px, py, pz) = (x[0], x[1], x[2]);
///         let e = (px * px - 1.0).powi(2) + 0.25 * (py * py + pz * pz);
///         Ok((e, array![4.0 * px * (px * px - 1.0), 0.5 * py, 0.5 * pz]))
///     }
/// }
///
/// let mode = MinModeConfig::default()
///     .with_kind(MinModeKind::Lanczos)
///     .with_saddle_check(None)
///     .with_force_tol(0.02)
///     .with_max_move(0.1)
///     .with_krylov_dim(4)
///     .with_max_rotations(6);
/// let art = ArtConfig::default()
///     .with_mode(mode)
///     .with_push_length(0.5)
///     .with_push_step(0.3)
///     .with_max_pushes(12)
///     .with_max_climbs(25)
///     .with_push_over(0.25);
/// let config = DirectionCoverConfig::default().with_art(art);
/// let mut cover = DirectionCover::new(&DoubleWell, array![-1.0, 0.0, 0.0], None, 7, config)?;
/// let report = cover.run(&DoubleWell, CoverBudget::default().with_max_arms(12))?;
/// assert_eq!(report.saddles.len(), 1);
/// assert!(report.saddles[0].position[0].abs() < 0.05);
/// assert!(report.saddles[0].hits > 1);
/// assert!(
///     report
///         .records
///         .iter()
///         .any(|r| r.outcome == ArmOutcome::DuplicateSaddle(0))
/// );
/// assert!(report.next.is_some());
/// # Ok::<(), SaddleError>(())
/// ```
pub struct DirectionCover {
    config: DirectionCoverConfig,
    minimum: Array1<f64>,
    energy: f64,
    seed: u64,
    centres: Vec<usize>,
    slot: Vec<Option<usize>>,
    cover: Vec<[f64; 3]>,
    tried: Vec<Vec<(usize, State)>>,
    done: Vec<Vec<bool>>,
    records: Vec<ArmRecord>,
    saddles: Vec<SaddleEntry>,
    products: Vec<ProductState>,
    discovery: Vec<usize>,
    force_calls: usize,
}

impl DirectionCover {
    /// Cover the active atoms of `minimum` and evaluate its energy once.
    ///
    /// `active` holds one flag per atom; `None` makes every atom a
    /// centre. `seed` sets the cover and the neighbour kicks.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] for a minimum that is not whole finite
    /// triples or a mask of the wrong length, [`SaddleError::Invalid`]
    /// for an empty mask or a bad setting, including one that
    /// [`ArtSession::new`] refuses, and any surface error at the minimum.
    pub fn new<S: PointSurface>(
        surface: &S,
        minimum: Array1<f64>,
        active: Option<&[bool]>,
        seed: u64,
        config: DirectionCoverConfig,
    ) -> Result<Self, SaddleError> {
        config.validate()?;
        if minimum.is_empty() || !minimum.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "direction cover needs a non-empty 3N Cartesian minimum".into(),
            ));
        }
        if !minimum.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("direction cover minimum"));
        }
        let n_atoms = minimum.len() / 3;
        let centres: Vec<usize> = match active {
            Some(mask) => {
                if mask.len() != n_atoms {
                    return Err(SaddleError::Shape(format!(
                        "active mask has {} entries for {n_atoms} atoms",
                        mask.len()
                    )));
                }
                (0..n_atoms).filter(|&atom| mask[atom]).collect()
            }
            None => (0..n_atoms).collect(),
        };
        if centres.is_empty() {
            return Err(SaddleError::Invalid("active mask selects no atom".into()));
        }
        let mut slot = vec![None; n_atoms];
        for (index, &atom) in centres.iter().enumerate() {
            slot[atom] = Some(index);
        }
        let cover = sphere_cover(config.cover_points, config.cover_sweeps, seed);
        let mut this = Self {
            minimum,
            energy: f64::NAN,
            seed,
            tried: vec![Vec::new(); centres.len()],
            done: vec![vec![false; cover.len()]; centres.len()],
            slot,
            centres,
            cover,
            config,
            records: Vec::new(),
            saddles: Vec::new(),
            products: Vec::new(),
            discovery: Vec::new(),
            force_calls: 0,
        };
        let probe = this.seed(ArmId {
            atom: this.centres[0],
            point: 0,
        })?;
        ArtSession::new(
            this.config.art.clone(),
            this.minimum.clone(),
            probe.seed,
            probe.mask,
        )?;
        let (energy, gradient) = surface.eval(this.minimum.view())?;
        this.force_calls = 1;
        if gradient.len() != this.minimum.len() {
            return Err(SaddleError::Shape(
                "surface gradient does not match the minimum".into(),
            ));
        }
        if !energy.is_finite() {
            return Err(SaddleError::NonFinite("direction cover minimum energy"));
        }
        this.energy = energy;
        Ok(this)
    }

    /// Energy of the minimum.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// The minimum.
    #[must_use]
    pub fn minimum(&self) -> ArrayView1<'_, f64> {
        self.minimum.view()
    }

    /// Centre atoms, in atom order.
    #[must_use]
    pub fn centres(&self) -> &[usize] {
        &self.centres
    }

    /// Signed unit vectors every centre carries, farthest point first.
    #[must_use]
    pub fn cover(&self) -> &[[f64; 3]] {
        &self.cover
    }

    /// Records so far, in record order.
    #[must_use]
    pub fn records(&self) -> &[ArmRecord] {
        &self.records
    }

    /// Distinct saddles so far.
    #[must_use]
    pub fn saddles(&self) -> &[SaddleEntry] {
        &self.saddles
    }

    /// Surface evaluations so far, the energy of the minimum included.
    #[must_use]
    pub fn force_calls(&self) -> usize {
        self.force_calls
    }

    fn centre_slot(&self, arm: ArmId) -> Result<usize, SaddleError> {
        let slot =
            self.slot.get(arm.atom).copied().flatten().ok_or_else(|| {
                SaddleError::Invalid(format!("atom {} is not a centre", arm.atom))
            })?;
        if arm.point >= self.cover.len() {
            return Err(SaddleError::Invalid(format!(
                "cover point {} is outside the {}-point cover",
                arm.point,
                self.cover.len()
            )));
        }
        Ok(slot)
    }

    fn delta(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> Array1<f64> {
        let mut d = &a - &b;
        if let Some(cell) = &self.config.cell {
            cell.minimum_image(&mut d);
        }
        d
    }

    fn rmsd(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let d = self.delta(a, b);
        (d.dot(&d) / (d.len() / 3) as f64).sqrt()
    }

    /// ART seed and mask of one arm.
    ///
    /// The seed is the cover vector on the centre plus, on every other
    /// atom within [`DirectionCoverConfig::kick_radius`], three normal
    /// components of standard deviation
    /// [`DirectionCoverConfig::kick_width`]. The kick is a function of
    /// the cover seed and the arm only. The mask is one on the centre and
    /// the kicked atoms.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when the atom is not a centre or the
    /// point is outside the cover.
    pub fn seed(&self, arm: ArmId) -> Result<ArtSeed, SaddleError> {
        self.centre_slot(arm)?;
        let n = self.minimum.len();
        let mut seed = Array1::zeros(n);
        let mut mask = Array1::zeros(n);
        let u = self.cover[arm.point];
        for axis in 0..3 {
            seed[3 * arm.atom + axis] = u[axis];
            mask[3 * arm.atom + axis] = 1.0;
        }
        if self.config.kick_radius > 0.0 && self.config.kick_width > 0.0 {
            let mut rng = SplitMix64::new(mix(
                self.seed,
                (arm.atom as u64).wrapping_mul(0x1000_0000_01B3) ^ arm.point as u64,
            ));
            let centre = self
                .minimum
                .slice(ndarray::s![3 * arm.atom..3 * arm.atom + 3]);
            let reach = self.config.kick_radius * self.config.kick_radius;
            for atom in 0..n / 3 {
                if atom == arm.atom {
                    continue;
                }
                let mut d = &self.minimum.slice(ndarray::s![3 * atom..3 * atom + 3]) - &centre;
                if let Some(cell) = &self.config.cell {
                    cell.minimum_image(&mut d);
                }
                if d.dot(&d) > reach {
                    continue;
                }
                for axis in 0..3 {
                    seed[3 * atom + axis] = self.config.kick_width * rng.normal();
                    mask[3 * atom + axis] = 1.0;
                }
            }
        }
        Ok(ArtSeed { seed, mask })
    }

    /// Run one arm through [`ArtSession`] without recording it.
    ///
    /// # Errors
    ///
    /// The errors of [`DirectionCover::seed`] and [`ArtSession::new`].
    /// A failure inside the event is an [`ArmFailure::Error`] trial, except
    /// typed index-1 loss, which is [`ArmFailure::IndexOneLost`].
    pub fn run_arm<S: PointSurface>(
        &self,
        surface: &S,
        arm: ArmId,
    ) -> Result<ArmTrial, SaddleError> {
        let seed = self.seed(arm)?;
        let mut session = ArtSession::new(
            self.config.art.clone(),
            self.minimum.clone(),
            seed.seed,
            seed.mask,
        )?;
        let counted = Counted {
            inner: surface,
            calls: AtomicUsize::new(0),
        };
        let run = session.run(&counted, self.config.art_max_steps);
        let calls = counted.calls.load(Ordering::Relaxed);
        if let Err(error) = run {
            let failure = match error {
                SaddleError::ArtIndexOneLost {
                    attempts,
                    curvature,
                    overlap,
                } => ArmFailure::IndexOneLost {
                    attempts,
                    curvature,
                    overlap,
                },
                other => ArmFailure::Error(other.to_string()),
            };
            return Ok(ArmTrial::failed(arm, failure, calls));
        }
        Ok(self.trial_from(arm, session.outcome(), calls))
    }

    fn trial_from(&self, arm: ArmId, outcome: ArtOutcome, calls: usize) -> ArmTrial {
        match outcome.status {
            ArtStatus::Connected => {
                let (Some(saddle), Some(saddle_energy), Some(back), Some(forward)) = (
                    outcome.saddle,
                    outcome.saddle_energy,
                    outcome.minimum_back,
                    outcome.minimum_forward,
                ) else {
                    return ArmTrial::failed(
                        arm,
                        ArmFailure::Error("connected event without its minima".into()),
                        calls,
                    );
                };
                let back_energy = outcome.back_energy.unwrap_or(f64::NAN);
                let forward_energy = outcome.forward_energy.unwrap_or(f64::NAN);
                let back_far = self.rmsd(back.view(), self.minimum.view());
                let forward_far = self.rmsd(forward.view(), self.minimum.view());
                let (product, product_energy) = if forward_far >= back_far {
                    (forward, forward_energy)
                } else {
                    (back, back_energy)
                };
                ArmTrial::connected(arm, saddle, saddle_energy, product, product_energy, calls)
            }
            ArtStatus::Reverted => ArmTrial::returned(arm, calls),
            ArtStatus::LostCurvature => ArmTrial::failed(arm, ArmFailure::LostCurvature, calls),
            ArtStatus::Irreversible => ArmTrial::failed(arm, ArmFailure::Disconnected, calls),
            ArtStatus::NotIndexOne => ArmTrial::failed(arm, ArmFailure::NotIndexOne, calls),
            ArtStatus::NotDownhill => ArmTrial::failed(arm, ArmFailure::NotDownhill, calls),
            ArtStatus::Exhausted | ArtStatus::Running => ArmTrial::non_converged(arm, calls),
        }
    }

    /// Compare a trial with the saddles and minima so far and store it.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] for an arm that is not a candidate or is
    /// already recorded, [`SaddleError::Shape`] or
    /// [`SaddleError::NonFinite`] for a connected trial whose geometry
    /// or energies do not fit the minimum.
    pub fn record(&mut self, trial: ArmTrial) -> Result<&ArmRecord, SaddleError> {
        let arm = trial.arm;
        let slot = self.centre_slot(arm)?;
        if self.done[slot][arm.point] {
            return Err(SaddleError::Invalid(format!(
                "arm (atom {}, point {}) is already recorded",
                arm.atom, arm.point
            )));
        }
        let mut record = ArmRecord {
            arm,
            direction: self.cover[arm.point],
            outcome: ArmOutcome::NonConverged,
            saddle: None,
            product: None,
            barrier: None,
            prefactor: None,
            force_calls: trial.force_calls,
        };
        let state = match trial.kind {
            TrialKind::Connected {
                saddle,
                saddle_energy,
                product,
                product_energy,
            } => {
                if saddle.len() != self.minimum.len() || product.len() != self.minimum.len() {
                    return Err(SaddleError::Shape(
                        "connected trial geometry does not match the minimum".into(),
                    ));
                }
                if !(saddle_energy.is_finite()
                    && product_energy.is_finite()
                    && saddle.iter().chain(product.iter()).all(|v| v.is_finite()))
                {
                    return Err(SaddleError::NonFinite("connected trial"));
                }
                let product_id = self.product_id(product, product_energy);
                let found = self.saddles.iter().position(|entry| {
                    (entry.energy - saddle_energy).abs() <= self.config.saddle_energy_tol
                        && self.rmsd(entry.position.view(), saddle.view())
                            <= self.config.saddle_rmsd_tol
                });
                let id = match found {
                    Some(index) => {
                        self.saddles[index].hits += 1;
                        record.outcome = ArmOutcome::DuplicateSaddle(self.saddles[index].id);
                        self.saddles[index].id
                    }
                    None => {
                        let id = self.saddles.len();
                        self.saddles.push(SaddleEntry {
                            id,
                            position: saddle,
                            energy: saddle_energy,
                            barrier: saddle_energy - self.energy,
                            product: product_id,
                            first_arm: arm,
                            hits: 1,
                        });
                        record.outcome = ArmOutcome::NewSaddle;
                        id
                    }
                };
                record.saddle = Some(id);
                record.product = Some(product_id);
                record.barrier = Some(saddle_energy - self.energy);
                State::Product(product_id)
            }
            TrialKind::Returned => {
                record.outcome = ArmOutcome::ReturnedToStart;
                State::Start
            }
            TrialKind::Failed(failure) => {
                record.outcome = ArmOutcome::Failed(failure);
                State::Unlabelled
            }
            TrialKind::NonConverged => State::Unlabelled,
        };
        self.done[slot][arm.point] = true;
        self.tried[slot].push((arm.point, state));
        self.force_calls += trial.force_calls;
        self.discovery.push(self.saddles.len());
        self.records.push(record);
        Ok(self.records.last().expect("just pushed"))
    }

    fn product_id(&mut self, product: Array1<f64>, energy: f64) -> usize {
        let found = self.products.iter().position(|state| {
            (state.energy - energy).abs() <= self.config.minimum_energy_tol
                && self.rmsd(state.position.view(), product.view()) <= self.config.minimum_rmsd_tol
        });
        match found {
            Some(index) => self.products[index].id,
            None => {
                let id = self.products.len();
                self.products.push(ProductState {
                    id,
                    position: product,
                    energy,
                });
                id
            }
        }
    }

    /// Attach a rate prefactor to a recorded arm.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when the arm is not recorded or the value
    /// is not finite and positive.
    pub fn set_prefactor(&mut self, arm: ArmId, prefactor: f64) -> Result<(), SaddleError> {
        if !(prefactor.is_finite() && prefactor > 0.0) {
            return Err(SaddleError::Invalid(
                "prefactor must be finite and positive".into(),
            ));
        }
        let record = self
            .records
            .iter_mut()
            .find(|record| record.arm == arm)
            .ok_or_else(|| SaddleError::Invalid("prefactor for an arm not recorded".into()))?;
        record.prefactor = Some(prefactor);
        Ok(())
    }

    /// Next arm under [`DirectionCoverConfig::policy`]. `None` once
    /// every arm is recorded. Equal scores go to the centre with fewer
    /// tried arms, then to the lower atom and cover point.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] for [`ArmPolicy::ExpectedImprovement`]
    /// before two different barriers are recorded, and
    /// [`SaddleError::Singular`] when the barrier kernel matrix of a
    /// centre cannot be factored.
    pub fn suggest(&self) -> Result<Option<ArmSuggestion>, SaddleError> {
        Ok(self.suggest_batch(1)?.into_iter().next())
    }

    /// Up to `count` distinct arms, picked one after another as if each
    /// earlier pick were already tried with no result yet.
    ///
    /// # Errors
    ///
    /// As [`DirectionCover::suggest`].
    pub fn suggest_batch(&self, count: usize) -> Result<Vec<ArmSuggestion>, SaddleError> {
        let model = match self.config.policy {
            ArmPolicy::Gap => None,
            ArmPolicy::ExpectedImprovement => Some(self.barrier_model()?.ok_or_else(|| {
                SaddleError::Invalid(
                    "expected improvement needs two different recorded barriers".into(),
                )
            })?),
            ArmPolicy::Auto => self.barrier_model()?,
        };
        let mut tried = self.tried.clone();
        let mut done = self.done.clone();
        let mut picks = Vec::with_capacity(count);
        while picks.len() < count {
            let mut best: Option<(ArmSuggestion, usize)> = None;
            for (slot, &atom) in self.centres.iter().enumerate() {
                let load = tried[slot].len();
                for point in 0..self.cover.len() {
                    if done[slot][point] {
                        continue;
                    }
                    let gap = self.gap_score(&tried[slot], point);
                    let (score, policy) = match &model {
                        None => (gap, ArmPolicy::Gap),
                        Some(model) => (
                            model.expected_improvement(slot, self.cover[point])
                                * gap_angle(&self.cover, &tried[slot], point)
                                / PI,
                            ArmPolicy::ExpectedImprovement,
                        ),
                    };
                    let wins = best.as_ref().is_none_or(|(held, held_load)| {
                        score > held.score || (score == held.score && load < *held_load)
                    });
                    if wins {
                        let arm = ArmId { atom, point };
                        best = Some((ArmSuggestion { arm, score, policy }, load));
                    }
                }
            }
            let Some((pick, _)) = best else {
                break;
            };
            let slot = self.slot[pick.arm.atom].expect("centre");
            done[slot][pick.arm.point] = true;
            tried[slot].push((pick.arm.point, State::Unlabelled));
            picks.push(pick);
        }
        Ok(picks)
    }

    /// Angle from `point` to the nearest tried direction times the
    /// distinct states in the start plus the states of the
    /// [`BOUNDARY_NEIGHBOURS`] nearest tried directions. Unlabelled
    /// directions add no state. Pi on a centre with nothing tried.
    fn gap_score(&self, tried: &[(usize, State)], point: usize) -> f64 {
        if tried.is_empty() {
            return PI;
        }
        let u = self.cover[point];
        let mut near: Vec<(f64, State)> = tried
            .iter()
            .map(|&(other, state)| (angle(u, self.cover[other]), state))
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let gap = near[0].0;
        let mut states = vec![State::Start];
        for &(_, state) in near.iter().take(BOUNDARY_NEIGHBOURS) {
            if state != State::Unlabelled && !states.contains(&state) {
                states.push(state);
            }
        }
        gap * states.len() as f64
    }

    fn barrier_model(&self) -> Result<Option<BarrierModel>, SaddleError> {
        let mut points = vec![Vec::new(); self.centres.len()];
        let mut values = Vec::new();
        for record in &self.records {
            if let Some(barrier) = record.barrier {
                let slot = self.slot[record.arm.atom].expect("centre");
                points[slot].push((record.direction, barrier));
                values.push(barrier);
            }
        }
        if values.len() < 2 {
            return Ok(None);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (values.len() - 1) as f64;
        if !(variance > 0.0 && variance.is_finite()) {
            return Ok(None);
        }
        let best = values.iter().copied().fold(f64::INFINITY, f64::min);
        let centres = points
            .into_iter()
            .map(|obs| CentreModel::fit(obs, mean, variance, self.config.length_scale))
            .collect::<Result<_, _>>()?;
        Ok(Some(BarrierModel {
            centres,
            mean,
            variance,
            target: best - self.config.ei_margin,
        }))
    }

    /// Pick, run and record arms until the budget or the cover is spent.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] for a budget with no limit, and the
    /// errors of [`DirectionCover::suggest_batch`],
    /// [`DirectionCover::run_arm`] and [`DirectionCover::record`].
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        budget: CoverBudget,
    ) -> Result<CoverageReport, SaddleError> {
        if budget.max_arms.is_none() && budget.max_force_calls.is_none() {
            return Err(SaddleError::Invalid(
                "direction cover budget needs an arm or force-call limit".into(),
            ));
        }
        loop {
            let arms_left = budget
                .max_arms
                .map_or(usize::MAX, |cap| cap.saturating_sub(self.records.len()));
            let calls_left = budget
                .max_force_calls
                .is_none_or(|cap| self.force_calls < cap);
            if arms_left == 0 || !calls_left {
                break;
            }
            let picks = self.suggest_batch(self.config.batch.min(arms_left))?;
            if picks.is_empty() {
                break;
            }
            let this = &*self;
            let trials: Vec<Result<ArmTrial, SaddleError>> = picks
                .par_iter()
                .map(|pick| this.run_arm(surface, pick.arm))
                .collect();
            for trial in trials {
                self.record(trial?)?;
            }
        }
        self.coverage()
    }

    /// Coverage readout of everything recorded so far.
    ///
    /// # Errors
    ///
    /// As [`DirectionCover::suggest`].
    pub fn coverage(&self) -> Result<CoverageReport, SaddleError> {
        let atoms: Vec<AtomCoverage> = self
            .centres
            .iter()
            .enumerate()
            .map(|(slot, &atom)| {
                let points: Vec<[f64; 3]> = self.tried[slot]
                    .iter()
                    .map(|&(p, _)| self.cover[p])
                    .collect();
                let mut states: Vec<State> = Vec::new();
                for &(_, state) in &self.tried[slot] {
                    if state != State::Unlabelled && !states.contains(&state) {
                        states.push(state);
                    }
                }
                AtomCoverage {
                    atom,
                    tried: points.len(),
                    covering_radius: covering_radius_s2(&points),
                    states: states.len(),
                }
            })
            .collect();
        let covering_radius = atoms
            .iter()
            .map(|entry| entry.covering_radius)
            .fold(0.0, f64::max);
        let most = self
            .saddles
            .iter()
            .map(|entry| entry.hits)
            .max()
            .unwrap_or(0);
        let mut frequency_counts = vec![0; most];
        for entry in &self.saddles {
            frequency_counts[entry.hits - 1] += 1;
        }
        let saddle_draws = self.saddles.iter().map(|entry| entry.hits).sum();
        Ok(CoverageReport {
            subspace_dim: 3 * self.centres.len(),
            covering_radius,
            atoms,
            records: self.records.clone(),
            saddles: self.saddles.clone(),
            products: self.products.clone(),
            discovery: self.discovery.clone(),
            saddle_draws,
            frequency_counts,
            force_calls: self.force_calls,
            next: self.suggest()?,
        })
    }
}

fn gap_angle(cover: &[[f64; 3]], tried: &[(usize, State)], point: usize) -> f64 {
    tried
        .iter()
        .map(|&(other, _)| angle(cover[point], cover[other]))
        .fold(PI, f64::min)
}

struct CentreModel {
    points: Vec<[f64; 3]>,
    weights: Array1<f64>,
    factor: Array2<f64>,
    length_scale: f64,
}

impl CentreModel {
    /// Fit the barriers of one centre.
    ///
    /// The kernel matrix is positive semi-definite, so the nugget
    /// [`KERNEL_NUGGET`] times the variance bounds its smallest
    /// eigenvalue from below. Scaled to a unit diagonal, that bound is
    /// about `KERNEL_NUGGET`, and Demmel's sufficient condition for
    /// Cholesky in floating point, smallest eigenvalue above
    /// \(n\gamma_{n+1} \approx n^2 u\) (Higham, *Accuracy and Stability of
    /// Numerical Algorithms*, 2nd ed., Theorem 10.7), then holds for
    /// fewer than 9000 tried directions on the centre.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Singular`] when a pivot is not positive.
    fn fit(
        obs: Vec<([f64; 3], f64)>,
        mean: f64,
        variance: f64,
        length_scale: f64,
    ) -> Result<Self, SaddleError> {
        let n = obs.len();
        let points: Vec<[f64; 3]> = obs.iter().map(|(p, _)| *p).collect();
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[(i, j)] = variance * kernel(points[i], points[j], length_scale);
            }
            k[(i, i)] += KERNEL_NUGGET * variance;
        }
        let factor = cholesky(&k)?;
        let residual = Array1::from_iter(obs.iter().map(|(_, y)| y - mean));
        let weights = cholesky_solve(&factor, &residual);
        Ok(Self {
            points,
            weights,
            factor,
            length_scale,
        })
    }
}

struct BarrierModel {
    centres: Vec<CentreModel>,
    mean: f64,
    variance: f64,
    target: f64,
}

impl BarrierModel {
    fn expected_improvement(&self, slot: usize, u: [f64; 3]) -> f64 {
        let model = &self.centres[slot];
        let kx = Array1::from_iter(
            model
                .points
                .iter()
                .map(|&p| self.variance * kernel(u, p, model.length_scale)),
        );
        let mu = self.mean + kx.dot(&model.weights);
        let v = forward_substitute(&model.factor, &kx);
        let sigma = (self.variance - v.dot(&v)).max(0.0).sqrt();
        let gain = self.target - mu;
        if sigma <= 1e-12 * self.variance.sqrt() {
            return gain.max(0.0);
        }
        let z = gain / sigma;
        gain * normal_cdf(z) + sigma * normal_pdf(z)
    }
}

/// Gaussian of the chordal distance, \(\exp(-\lVert a-b\rVert^2/2\ell^2)\).
///
/// It is the Gaussian kernel of \(\mathbb{R}^3\) restricted to the unit
/// sphere, so its Gram matrix on distinct directions is positive
/// definite for every \(\ell > 0\). The Gaussian of the geodesic angle
/// is not positive definite on \(S^2\) for long length scales (Feragen,
/// Lauze and Hauberg, CVPR 2015). For small angles the chord is the
/// angle.
fn kernel(a: [f64; 3], b: [f64; 3], length_scale: f64) -> f64 {
    let chord2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    (-0.5 * chord2 / (length_scale * length_scale)).exp()
}

/// Lower Cholesky factor of a symmetric positive definite matrix.
///
/// # Errors
///
/// [`SaddleError::Singular`] with the row of the first pivot that is not
/// finite and positive.
fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>, SaddleError> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(SaddleError::Singular(i));
                }
                l[(i, i)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }
    Ok(l)
}

fn forward_substitute(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[(i, k)] * y[k];
        }
        y[i] = sum / l[(i, i)];
    }
    y
}

fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let y = forward_substitute(l, b);
    let n = y.len();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[(k, i)] * x[k];
        }
        x[i] = sum / l[(i, i)];
    }
    x
}

fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Standard normal distribution function through the complementary
/// error function of Numerical Recipes, `erfcc`, fractional error under
/// 1.2e-7.
fn normal_cdf(z: f64) -> f64 {
    let x = -z / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let poly = -x * x - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    let erfc = if x >= 0.0 { erfc } else { 2.0 - erfc };
    0.5 * erfc
}

/// Surface counting the evaluations of one arm.
struct Counted<'a, S: PointSurface> {
    inner: &'a S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
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

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot3(a, b).clamp(-1.0, 1.0).acos()
}

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot3(v, v).sqrt();
    (n.is_finite() && n > 1e-12).then(|| [v[0] / n, v[1] / n, v[2] / n])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Covering radius of signed unit vectors on \(S^2\): the largest angle
/// from any point of the sphere to the nearest of `points`, in radians.
///
/// Exact. The largest empty cap is centred on a vertex of the spherical
/// Voronoi diagram (a circumcentre of three points), on the far point of
/// a Voronoi edge (the antipode of the midpoint of two points), or on
/// the antipode of one point, and every such candidate is evaluated.
/// Pi for no point, and the directions are not folded: a direction and
/// its opposite are two points. Entries need not be normalized; a zero
/// vector is skipped.
#[must_use]
pub fn covering_radius_s2(points: &[[f64; 3]]) -> f64 {
    let points: Vec<[f64; 3]> = points.iter().filter_map(|&p| unit(p)).collect();
    if points.is_empty() {
        return PI;
    }
    let reach = |x: [f64; 3]| {
        points
            .iter()
            .map(|&p| angle(x, p))
            .fold(f64::INFINITY, f64::min)
    };
    let mut best = 0.0_f64;
    let mut consider = |x: [f64; 3]| {
        best = best.max(reach(x));
    };
    for &a in &points {
        consider([-a[0], -a[1], -a[2]]);
    }
    for (i, &a) in points.iter().enumerate() {
        for &b in &points[i + 1..] {
            match unit([a[0] + b[0], a[1] + b[1], a[2] + b[2]]) {
                Some(m) => consider([-m[0], -m[1], -m[2]]),
                None => {
                    let seed = if a[0].abs() < 0.9 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    if let Some(normal) = unit(cross(a, seed)) {
                        consider(normal);
                    }
                }
            }
        }
    }
    for (i, &a) in points.iter().enumerate() {
        for (j, &b) in points.iter().enumerate().skip(i + 1) {
            for &c in &points[j + 1..] {
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                if let Some(n) = unit(cross(ab, ac)) {
                    consider(n);
                    consider([-n[0], -n[1], -n[2]]);
                }
            }
        }
    }
    best
}

/// `n` signed unit vectors on \(S^2\), Thomson-spread from a seeded
/// start and in farthest-point order.
///
/// The relaxation is the Gauss-Seidel sweep of SoftSaddle
/// `triangulation_hight_v2` with exponent one: each point in turn steps
/// down the tangent gradient of its repulsion and keeps the step only if
/// its energy does not rise, halving by 2.5 otherwise. The order starts
/// at the Euclidean diameter pair and adds the point with the largest
/// summed distance to those already placed, SoftSaddle
/// `sort_initial_disp`.
#[must_use]
pub fn sphere_cover(n: usize, sweeps: usize, seed: u64) -> Vec<[f64; 3]> {
    if n == 0 {
        return Vec::new();
    }
    let mut rng = SplitMix64::new(mix(seed, 0x5EED_C0FE));
    let mut points: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            loop {
                if let Some(u) = unit([rng.normal(), rng.normal(), rng.normal()]) {
                    break u;
                }
            }
        })
        .collect();
    if n > 1 {
        relax_cover(&mut points, sweeps);
    }
    let order = farthest_order(&points);
    order.into_iter().map(|i| points[i]).collect()
}

fn pair_energy(a: [f64; 3], b: [f64; 3]) -> f64 {
    1.0 / (angle(a, b) + f64::EPSILON)
}

fn relax_cover(points: &mut [[f64; 3]], sweeps: usize) {
    let n = points.len();
    let mut step = vec![1.0_f64; n];
    let total = |points: &[[f64; 3]]| {
        let mut sum = 0.0;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                sum += pair_energy(points[i], points[j]);
            }
        }
        sum
    };
    let mut energy = total(points);
    for sweep in 0..sweeps {
        for j in 0..n {
            let mut grad = [0.0; 3];
            for i in 0..n {
                if i == j {
                    continue;
                }
                let d = dot3(points[i], points[j]).clamp(-1.0, 1.0);
                let g = d.acos();
                let tangent = (1.0 - d * d).sqrt();
                let w = 1.0 / ((tangent + f64::EPSILON) * (g * g + f64::EPSILON));
                for (component, &point) in grad.iter_mut().zip(&points[i]) {
                    *component += w * point;
                }
            }
            let radial = dot3(grad, points[j]);
            for (component, &point) in grad.iter_mut().zip(&points[j]) {
                *component -= radial * point;
            }
            let own = |p: [f64; 3], points: &[[f64; 3]]| {
                (0..n)
                    .filter(|&i| i != j)
                    .map(|i| pair_energy(p, points[i]))
                    .sum::<f64>()
            };
            let before = own(points[j], points);
            loop {
                let trial = unit([
                    points[j][0] - step[j] * grad[0],
                    points[j][1] - step[j] * grad[1],
                    points[j][2] - step[j] * grad[2],
                ]);
                if let Some(trial) = trial
                    && own(trial, points) <= before
                {
                    points[j] = trial;
                    step[j] = step[j].min(COVER_STEP_MAX);
                    break;
                }
                if step[j] > COVER_STEP_MIN {
                    step[j] = (step[j] / COVER_STEP_SHRINK).max(COVER_STEP_MIN);
                } else {
                    break;
                }
            }
        }
        let now = total(points);
        if sweep >= 10 && (energy - now).abs() <= COVER_ETOL * now.abs() {
            break;
        }
        energy = now;
        if (sweep + 1) % 20 == 0 {
            step.fill(COVER_STEP_MAX);
        }
    }
}

fn farthest_order(points: &[[f64; 3]]) -> Vec<usize> {
    let n = points.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let dist = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let (mut first, mut second, mut far) = (0, 1, -1.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(points[i], points[j]);
            if d > far {
                far = d;
                first = i;
                second = j;
            }
        }
    }
    let mut order = vec![first, second];
    let mut taken = vec![false; n];
    taken[first] = true;
    taken[second] = true;
    let mut sums: Vec<f64> = (0..n)
        .map(|j| dist(points[first], points[j]) + dist(points[second], points[j]))
        .collect();
    while order.len() < n {
        let mut pick = usize::MAX;
        let mut held = -1.0;
        for j in 0..n {
            if !taken[j] && sums[j] > held {
                held = sums[j];
                pick = j;
            }
        }
        taken[pick] = true;
        order.push(pick);
        for j in 0..n {
            sums[j] += dist(points[pick], points[j]);
        }
    }
    order
}

fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.state, 0)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let r = u * u + v * v;
            if r > 0.0 && r < 1.0 {
                return u * (-2.0 * r.ln() / r).sqrt();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_points_are_unit_distinct_and_signed() {
        let cover = sphere_cover(64, 400, 7);
        assert_eq!(cover.len(), 64);
        let mut closest = PI;
        for (i, &a) in cover.iter().enumerate() {
            assert!((dot3(a, a) - 1.0).abs() < 1e-12);
            for &b in &cover[i + 1..] {
                closest = closest.min(angle(a, b));
            }
        }
        // Tammes for 64 points is about 0.26 rad; a random set is far below.
        assert!(closest > 0.2, "closest pair {closest}");
        assert_eq!(cover, sphere_cover(64, 400, 7));
        assert_ne!(cover, sphere_cover(64, 400, 8));
    }

    #[test]
    fn exact_radius_of_small_sets() {
        assert_eq!(covering_radius_s2(&[]), PI);
        assert!((covering_radius_s2(&[[0.0, 0.0, 1.0]]) - PI).abs() < 1e-12);
        let pair = covering_radius_s2(&[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]);
        assert!((pair - 0.5 * PI).abs() < 1e-12, "{pair}");
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let octahedron = covering_radius_s2(&axes);
        assert!((octahedron - (1.0_f64 / 3.0).sqrt().acos()).abs() < 1e-12);
        let folded = covering_radius_s2(&[axes[0], axes[2], axes[4]]);
        assert!(
            folded > octahedron + 0.5,
            "signed directions are not folded"
        );
    }

    /// Smallest pivot of an unguarded Cholesky factorization of `a`.
    fn smallest_pivot(a: &Array2<f64>) -> f64 {
        let n = a.nrows();
        let mut l = Array2::<f64>::zeros((n, n));
        let mut smallest = f64::INFINITY;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[(i, j)];
                for k in 0..j {
                    sum -= l[(i, k)] * l[(j, k)];
                }
                if i == j {
                    smallest = smallest.min(sum);
                    l[(i, i)] = sum.abs().sqrt();
                } else {
                    l[(i, j)] = sum / l[(j, j)];
                }
            }
        }
        smallest
    }

    fn cover_gram(points: &[[f64; 3]], length_scale: f64) -> Array2<f64> {
        let n = points.len();
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[(i, j)] = kernel(points[i], points[j], length_scale);
            }
            k[(i, i)] += 1e-8;
        }
        k
    }

    #[test]
    fn barrier_kernel_is_positive_definite_on_the_cover() {
        let cover = sphere_cover(64, 400, 7);
        let negative: Vec<(f64, f64)> = [0.1, 0.6, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0]
            .into_iter()
            .map(|length_scale| {
                (
                    length_scale,
                    smallest_pivot(&cover_gram(&cover, length_scale)),
                )
            })
            .filter(|&(_, pivot)| pivot <= 0.0)
            .collect();
        assert!(
            negative.is_empty(),
            "(length scale, smallest Cholesky pivot): {negative:?}"
        );
    }

    #[test]
    fn barrier_posterior_interpolates_at_a_long_length_scale() {
        let cover = sphere_cover(64, 400, 7);
        let (seen, held_out) = cover.split_at(48);
        let obs: Vec<([f64; 3], f64)> = seen
            .iter()
            .map(|&u| (u, 1.0 + 0.3 * u[2] + 0.1 * u[0] * u[1]))
            .collect();
        let values: Vec<f64> = obs.iter().map(|(_, y)| *y).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (values.len() - 1) as f64;
        let centre = CentreModel::fit(obs.clone(), mean, variance, 2.0).unwrap();
        let cross = |u: [f64; 3]| {
            Array1::from_iter(
                centre
                    .points
                    .iter()
                    .map(|&p| variance * kernel(u, p, centre.length_scale)),
            )
        };
        let worst = obs
            .iter()
            .map(|&(u, y)| (mean + cross(u).dot(&centre.weights) - y).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-3 * variance.sqrt(),
            "posterior mean misses an observed barrier by {worst}"
        );
        for &u in held_out {
            let v = forward_substitute(&centre.factor, &cross(u));
            let posterior = variance - v.dot(&v);
            assert!(
                posterior > 0.0 && posterior <= variance,
                "posterior variance {posterior} outside (0, {variance}] at {u:?}"
            );
        }
    }

    #[test]
    fn cholesky_refuses_an_indefinite_matrix() {
        let indefinite = ndarray::array![[1.0, 2.0], [2.0, 1.0]];
        assert!(matches!(
            cholesky(&indefinite),
            Err(SaddleError::Singular(1))
        ));
        let definite = ndarray::array![[4.0, 2.0], [2.0, 3.0]];
        let l = cholesky(&definite).unwrap();
        assert!((l.dot(&l.t()) - &definite).iter().all(|e| e.abs() < 1e-12));
    }

    #[test]
    fn normal_cdf_matches_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.0) - 0.841_344_746).abs() < 1e-6);
        assert!((normal_cdf(-2.0) - 0.022_750_132).abs() < 1e-6);
    }
}
