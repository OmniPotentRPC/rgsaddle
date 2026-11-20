//! Downward search of a solution landscape.
//!
//! An index-k saddle is a maximum on a k-dimensional subspace and a
//! minimum on its complement. The high-index optimization-based shrinking
//! dimer (HiOSD) dynamics of Yin, Zhang, and Zhang, SIAM J. Sci. Comput.
//! 41, A3576 (2019), doi:10.1137/19M1253356, climb that subspace and
//! descend on the complement. This crate already takes one such step in
//! [`crate::IndexKSession`]: the translated force reflects the k lowest
//! modes once their curvatures are negative, which is the dimer
//! translation of Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063
//! (2026), doi:10.3389/fchem.2026.1807063, Equation 3, applied to the
//! whole block.
//!
//! The pathway map of Yin, Wang, Chen, Zhang, and Zhang, Phys. Rev. Lett.
//! 124, 090601 (2020), doi:10.1103/PhysRevLett.124.090601, connects
//! critical points whose indices differ by one. From a saddle of index k,
//! certified here by [`mod@crate::saddle_index`] (the same block Lanczos
//! certificate a converged dimer records; Goswami, Masterov, Kamath,
//! Peña-Torres, and Jónsson, J. Chem. Theory Comput. 21, 7935 (2025),
//! doi:10.1021/acs.jctc.5c00866), the search pushes a finite step along
//! one unstable eigenvector and follows index-(k-1) HiOSD dynamics until
//! that search converges. The point is certified and stored when its
//! index is exactly one less than the parent. A leg that lands on any
//! other index returns [`SaddleError::Solver`], and the search repeats
//! on an accepted child. Index 0 is a minimum: the leg is an
//! [`rgmin`](https://github.com/OmniPotentRPC/rgmin) solve on the true
//! gradient, not a reflected force.
//!
//! # Limits
//!
//! The push is one Euclidean step of length [`LandscapeConfig::push`]
//! along a unit eigenvector. It has to leave the neighbourhood in which
//! the perturbed curvature is still among the k-1 lowest, or the
//! index-(k-1) dynamics return to the parent and the search returns
//! [`SaddleError::Solver`]. Distinct critical points closer than
//! [`LandscapeConfig::duplicate_distance`] are one node. With a
//! [`Cell`], that distance is the Euclidean norm of
//! [`Cell::minimum_image`]. [`LandscapeConfig::max_nodes`] and
//! [`LandscapeConfig::max_evaluations`] are hard stops:
//! [`SaddleError::LandscapeBound`] names the limit and the counts, and
//! the partial pathway is not returned as a finished map.
//!
//! The coordinate length is arbitrary. A cell is only valid when that
//! length is a multiple of three, because the minimum-image helper wraps
//! Cartesian triples.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Control, Oracle, Solver, StopNorm};

use crate::Cell;
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::indexk::{IndexKConfig, IndexKSession};
use crate::minmode::{MinModeStatus, PointSurface};
use crate::relax::RelaxConfig;
use crate::saddle_index::{self, Certificate, Probe, SaddleIndexCheck};

/// Euclidean distance below which two critical points are one node.
///
/// The length unit is the coordinate unit of the surface. Distinct
/// critical points of interest must sit further apart than this, and
/// further apart than the position uncertainty left by the force
/// tolerance.
pub const DEFAULT_DUPLICATE_DISTANCE: f64 = 1.0e-3;

/// Euclidean length of the push along one unit unstable eigenvector.
pub const DEFAULT_LANDSCAPE_PUSH: f64 = 0.1;

/// Most critical points a search accepts, including the parent.
pub const DEFAULT_LANDSCAPE_MAX_NODES: usize = 64;

/// Most force evaluations a search may spend.
pub const DEFAULT_LANDSCAPE_MAX_EVALUATIONS: usize = 100_000;

/// Most index-(k-1) translations, and the step cap passed to the minimizer
/// when it is tighter than [`RelaxConfig::max_steps`].
pub const DEFAULT_LANDSCAPE_MAX_STEPS: usize = 2_000;

/// Residual column norm, after orthonormalization, below which a supplied
/// unstable block is linearly dependent.
const MODE_DEPENDENCE_TOL: f64 = 1.0e-8;

/// Which configured limit [`SaddleError::LandscapeBound`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandscapeBoundKind {
    /// [`LandscapeConfig::max_nodes`] was already full when a new critical
    /// point was found.
    Nodes,
    /// [`LandscapeConfig::max_evaluations`] was exhausted.
    Evaluations,
}

impl std::fmt::Display for LandscapeBoundKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nodes => "node",
            Self::Evaluations => "force-evaluation",
        })
    }
}

/// Controls of [`LandscapeSession`].
///
/// Non-exhaustive: start from [`LandscapeConfig::default`] and chain the
/// `with_*` setters.
///
/// `search.k` is replaced on each leg by the child index. The child
/// index-k search clears `search.search.saddle_check` and
/// `search.search.artn` on its own copy, so a certificate escape does not
/// run inside a leg. The landscape certifies every accepted point with
/// [`LandscapeConfig::certificate`]. `certificate.escape` is not read.
///
/// The parent and every saddle leg stop on [`crate::MinModeConfig::force_tol`].
/// A minimum stops on [`RelaxConfig::force_tol`]. All three measure the
/// force with [`LandscapeConfig::force_gate`], and the index-k leg is given
/// that same gate.
/// [`LandscapeConfig::max_steps`] caps each saddle leg. A minimum leg
/// uses the smaller of that cap and [`RelaxConfig::max_steps`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LandscapeConfig {
    /// Euclidean separation at or below which two positions are one node.
    pub duplicate_distance: f64,
    /// Euclidean length of the displacement along a unit unstable eigenvector.
    pub push: f64,
    /// Most nodes, including the parent. Reaching it is
    /// [`SaddleError::LandscapeBound`], not a shorter map.
    pub max_nodes: usize,
    /// Most calls to [`PointSurface::eval`]. Reaching it is
    /// [`SaddleError::LandscapeBound`].
    pub max_evaluations: usize,
    /// Most translations of one index-(k-1) leg, and a cap on one minimum leg.
    pub max_steps: usize,
    /// Index-k stepper used for every child of index 1 or more.
    pub search: IndexKConfig,
    /// rgmin stepper used when the child index is 0.
    pub relax: RelaxConfig,
    /// Certificate of every accepted critical point.
    pub certificate: SaddleIndexCheck,
    /// Norm used for the parent, every saddle leg, and every minimum.
    /// A minimum compares it with [`RelaxConfig::force_tol`]. The parent
    /// and a saddle leg compare it with [`crate::MinModeConfig::force_tol`].
    pub force_gate: ForceGate,
    /// Periodic cell for [`Cell::minimum_image`] distances. `None` is Euclidean.
    pub cell: Option<Cell>,
}

with_setters!(LandscapeConfig {
    with_duplicate_distance => duplicate_distance: val f64,
    with_push => push: val f64,
    with_max_nodes => max_nodes: val usize,
    with_max_evaluations => max_evaluations: val usize,
    with_max_steps => max_steps: val usize,
    with_search => search: val IndexKConfig,
    with_relax => relax: val RelaxConfig,
    with_certificate => certificate: val SaddleIndexCheck,
    with_force_gate => force_gate: val ForceGate,
    with_cell => cell: opt Cell,
});

impl Default for LandscapeConfig {
    fn default() -> Self {
        let mut search = IndexKConfig::default();
        search.search.saddle_check = None;
        Self {
            duplicate_distance: DEFAULT_DUPLICATE_DISTANCE,
            push: DEFAULT_LANDSCAPE_PUSH,
            max_nodes: DEFAULT_LANDSCAPE_MAX_NODES,
            max_evaluations: DEFAULT_LANDSCAPE_MAX_EVALUATIONS,
            max_steps: DEFAULT_LANDSCAPE_MAX_STEPS,
            search,
            relax: RelaxConfig::default(),
            certificate: SaddleIndexCheck::default(),
            force_gate: ForceGate::LinfNorm,
            cell: None,
        }
    }
}

impl LandscapeConfig {
    fn validate(&self, n: usize) -> Result<(), SaddleError> {
        if !(self.duplicate_distance.is_finite() && self.duplicate_distance > 0.0) {
            return Err(SaddleError::Invalid(
                "duplicate distance must be finite and positive".into(),
            ));
        }
        if !(self.push.is_finite() && self.push > 0.0) {
            return Err(SaddleError::Invalid(
                "landscape push must be finite and positive".into(),
            ));
        }
        if self.max_nodes == 0 || self.max_evaluations == 0 || self.max_steps == 0 {
            return Err(SaddleError::Invalid(
                "landscape node, evaluation, and step limits must be at least 1".into(),
            ));
        }
        if self.cell.is_some() && !n.is_multiple_of(3) {
            return Err(SaddleError::Invalid(
                "a periodic cell applies to coordinates grouped by threes".into(),
            ));
        }
        self.validate_search(n)?;
        self.relax.check()?;
        self.certificate.validate()?;
        Ok(())
    }

    fn validate_search(&self, n: usize) -> Result<(), SaddleError> {
        let search = &self.search.search;
        if self.search.k == 0 {
            return Err(SaddleError::Invalid("index k must be at least 1".into()));
        }
        if self.search.k > n {
            return Err(SaddleError::Invalid("index k exceeds the dimension".into()));
        }
        if !(search.force_tol.is_finite() && search.force_tol > 0.0) {
            return Err(SaddleError::Invalid(
                "landscape force tolerance must be finite and positive".into(),
            ));
        }
        if !(search.dr.is_finite() && search.dr > 0.0) {
            return Err(SaddleError::Invalid(
                "finite-difference step must be positive and finite".into(),
            ));
        }
        if !(search.rotation_tol.is_finite()
            && search.rotation_tol >= 0.0
            && search.rotation_angle_tol.is_finite()
            && search.rotation_angle_tol >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "rotation tolerances must be finite and non-negative".into(),
            ));
        }
        if search.max_rotations == 0 {
            return Err(SaddleError::Invalid(
                "max_rotations must be at least 1".into(),
            ));
        }
        if search.krylov_dim == 0 {
            return Err(SaddleError::Invalid(
                "Lanczos krylov dimension must be at least 1".into(),
            ));
        }
        if !search.max_move.is_finite() {
            return Err(SaddleError::Invalid(
                "max_move must be finite; a non-positive value leaves the step uncapped".into(),
            ));
        }
        if !(self.search.residual_tol.is_finite() && self.search.residual_tol >= 0.0) {
            return Err(SaddleError::Invalid(
                "residual tolerance must be finite and non-negative".into(),
            ));
        }
        if self.search.max_block_iterations == 0 {
            return Err(SaddleError::Invalid(
                "block iteration cap must be positive".into(),
            ));
        }
        if let Some(check) = &search.saddle_check {
            check.validate()?;
        }
        if let Some(artn) = &search.artn {
            crate::minmode::validate_artn(search.kind, artn)?;
        }
        crate::band::check_force_driven(&search.method)?;
        Ok(())
    }
}

/// One certified critical point on a [`PathwayMap`].
#[derive(Clone, Debug)]
pub struct CriticalPoint {
    /// Coordinates, in the surface's length unit.
    pub position: Array1<f64>,
    /// Energy at [`Self::position`].
    pub energy: f64,
    /// Number of curvatures under [`SaddleIndexCheck::negative_tol`].
    pub index: usize,
    /// Lowest certified curvatures, ascending, energy per length squared.
    pub curvatures: Vec<f64>,
    /// Unstable eigenvectors, shape `(n, index)`, lowest curvature first.
    /// Columns are unit length. Empty at a minimum.
    pub modes: Array2<f64>,
    /// Force evaluations spent accepting this node: the parent's
    /// certificate, or the child search plus its certificate.
    pub evaluations: usize,
}

/// One downward connection. `direction` is a column of the parent's
/// [`CriticalPoint::modes`], and `sign` is `+1` or `-1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LandscapeEdge {
    /// Index into [`PathwayMap::nodes`].
    pub parent: usize,
    /// Index into [`PathwayMap::nodes`].
    pub child: usize,
    /// Column of the parent's unstable block.
    pub direction: usize,
    /// Which way along that column the push stepped.
    pub sign: i8,
    /// Force evaluations of this leg, including the child's certificate
    /// when the leg discovered that child.
    pub evaluations: usize,
}

/// Directed pathway map. Node 0 is the parent the search started from.
#[derive(Clone, Debug)]
pub struct PathwayMap {
    /// Critical points, parent first.
    pub nodes: Vec<CriticalPoint>,
    /// Downward connections. A repeated visit adds an edge and does not
    /// search that child again.
    pub edges: Vec<LandscapeEdge>,
    /// Every [`PointSurface::eval`] of the search.
    pub evaluations: usize,
}

/// Downward solution-landscape search.
///
/// `modes`, when set, has shape `(n, k)` and is the unstable block from
/// a converged [`IndexKSession`]: one column per negative curvature.
/// The column count must equal the index [`crate::saddle_index()`] certifies
/// at the parent. Omit `modes` and the certificate's Ritz vectors are
/// the unstable directions.
///
/// # Examples
///
/// The separable well
/// `V = (x^4 - 2 x^2) / 4 + (y^4 - y^2) / 4 + z^2 / 2`
/// has an index-2 saddle at the origin, four index-1 saddles, and four
/// minima. A push of length 0.5 leaves the region in which the perturbed
/// curvature is still the lowest.
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::{
///     IndexKConfig, LandscapeConfig, LandscapeSession, MinModeConfig, PointSurface, RelaxConfig,
///     SaddleError,
/// };
///
/// struct Wells;
///
/// impl PointSurface for Wells {
///     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (a, b, c) = (x[0], x[1], x[2]);
///         let energy = (a.powi(4) - 2.0 * a * a) / 4.0
///             + (b.powi(4) - b * b) / 4.0
///             + c * c / 2.0;
///         Ok((energy, array![a.powi(3) - a, b.powi(3) - 0.5 * b, c]))
///     }
///
///     fn hessian_vector(
///         &self,
///         x: ArrayView1<f64>,
///         v: ArrayView1<f64>,
///     ) -> Result<Option<Array1<f64>>, SaddleError> {
///         Ok(Some(array![
///             (3.0 * x[0] * x[0] - 1.0) * v[0],
///             (3.0 * x[1] * x[1] - 0.5) * v[1],
///             v[2],
///         ]))
///     }
/// }
///
/// let search = MinModeConfig::default()
///     .with_force_tol(1.0e-6)
///     .with_max_move(0.1)
///     .with_saddle_check(None);
/// let config = LandscapeConfig::default()
///     .with_push(0.5)
///     .with_duplicate_distance(1.0e-2)
///     .with_max_steps(400)
///     .with_search(IndexKConfig::new(1, search))
///     .with_relax(
///         RelaxConfig::default()
///             .with_force_tol(1.0e-6)
///             .with_max_atom_move(0.1)
///             .with_initial_step(0.05),
///     );
/// let session = LandscapeSession::new(config, array![0.0, 0.0, 0.0], None)?;
/// let map = session.run(&Wells)?;
/// assert_eq!(map.nodes[0].index, 2);
/// assert_eq!(map.nodes.iter().filter(|node| node.index == 1).count(), 4);
/// assert_eq!(map.nodes.iter().filter(|node| node.index == 0).count(), 4);
/// # Ok::<(), SaddleError>(())
/// ```
pub struct LandscapeSession {
    config: LandscapeConfig,
    parent: Array1<f64>,
    modes: Option<Array2<f64>>,
}

impl LandscapeSession {
    /// `parent` is a critical point. `modes` is the optional unstable block.
    pub fn new(
        config: LandscapeConfig,
        parent: Array1<f64>,
        modes: Option<Array2<f64>>,
    ) -> Result<Self, SaddleError> {
        if parent.is_empty() || !parent.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Invalid(
                "landscape parent must be a nonempty finite position".into(),
            ));
        }
        if let Some(modes) = &modes {
            if modes.nrows() != parent.len() || modes.ncols() == 0 {
                return Err(SaddleError::Shape(
                    "unstable block must have shape (n, k) with k at least 1".into(),
                ));
            }
            independent_modes(modes)?;
        }
        config.validate(parent.len())?;
        Ok(Self {
            config,
            parent,
            modes,
        })
    }

    /// Search downward from the parent until every unstable direction has
    /// been followed, or a limit or a leg fails.
    pub fn run<S: PointSurface>(&self, surface: &S) -> Result<PathwayMap, SaddleError> {
        let budget = Budget {
            inner: surface,
            calls: AtomicUsize::new(0),
            limit: self.config.max_evaluations,
        };
        self.search(&budget).map(|mut map| {
            map.evaluations = budget.calls.load(Ordering::Relaxed);
            map
        })
    }

    fn search<S: PointSurface>(&self, surface: &Budget<'_, S>) -> Result<PathwayMap, SaddleError> {
        let calls = &surface.calls;
        let before = calls.load(Ordering::Relaxed);
        let parent = self
            .accept_parent(surface)
            .map_err(|err| note(err, 0, calls.load(Ordering::Relaxed)))?;
        let mut nodes = vec![parent];
        nodes[0].evaluations = calls.load(Ordering::Relaxed) - before;
        let mut edges = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(0usize);
        while let Some(parent_i) = queue.pop_front() {
            let parent_index = nodes[parent_i].index;
            if parent_index == 0 {
                continue;
            }
            let origin = nodes[parent_i].position.clone();
            let modes = nodes[parent_i].modes.clone();
            for direction in 0..parent_index {
                for sign in [1.0_f64, -1.0] {
                    let spent = calls.load(Ordering::Relaxed);
                    let branch = self
                        .branch(
                            surface,
                            &nodes,
                            Leg {
                                parent: parent_i,
                                origin: &origin,
                                modes: &modes,
                                direction,
                                sign,
                            },
                        )
                        .map_err(|err| note(err, nodes.len(), calls.load(Ordering::Relaxed)))?;
                    let evaluations = calls.load(Ordering::Relaxed) - spent;
                    match branch {
                        Branch::New(mut point) => {
                            if nodes.len() >= self.config.max_nodes {
                                return Err(SaddleError::LandscapeBound {
                                    kind: LandscapeBoundKind::Nodes,
                                    limit: self.config.max_nodes,
                                    nodes: nodes.len(),
                                    evaluations: calls.load(Ordering::Relaxed),
                                });
                            }
                            point.evaluations = evaluations;
                            let child = nodes.len();
                            nodes.push(point);
                            edges.push(LandscapeEdge {
                                parent: parent_i,
                                child,
                                direction,
                                sign: sign_byte(sign),
                                evaluations,
                            });
                            queue.push_back(child);
                        }
                        Branch::Known(child) => {
                            edges.push(LandscapeEdge {
                                parent: parent_i,
                                child,
                                direction,
                                sign: sign_byte(sign),
                                evaluations,
                            });
                        }
                    }
                }
            }
        }
        Ok(PathwayMap {
            nodes,
            edges,
            evaluations: calls.load(Ordering::Relaxed),
        })
    }

    fn accept_parent<S: PointSurface>(
        &self,
        surface: &Budget<'_, S>,
    ) -> Result<CriticalPoint, SaddleError> {
        let (energy, gradient) = surface.eval(self.parent.view())?;
        let force = self.config.force_gate.value(gradient.view());
        if !(energy.is_finite() && force <= self.config.search.search.force_tol) {
            return Err(SaddleError::Invalid(format!(
                "landscape parent force {force} is above {}",
                self.config.search.search.force_tol
            )));
        }
        let mut hints = Vec::new();
        if let Some(modes) = &self.modes {
            for column in modes.columns() {
                hints.push(column.to_owned());
            }
        }
        let certified = self.certify(surface, self.parent.view(), gradient.view(), hints)?;
        if let Some(modes) = &self.modes
            && modes.ncols() != certified.index.index
        {
            return Err(SaddleError::Invalid(format!(
                "unstable block has {} columns and the certificate index is {}",
                modes.ncols(),
                certified.index.index
            )));
        }
        if !certified.index.converged {
            return Err(SaddleError::Solver(
                "parent saddle-index certificate did not converge".into(),
            ));
        }
        point_from(self.parent.clone(), energy, certified)
    }

    fn branch<S: PointSurface>(
        &self,
        surface: &Budget<'_, S>,
        nodes: &[CriticalPoint],
        leg: Leg<'_>,
    ) -> Result<Branch, SaddleError> {
        let axis = leg.modes.column(leg.direction);
        let mut start = leg.origin.clone();
        start.scaled_add(leg.sign * self.config.push, &axis);
        let parent_index = nodes[leg.parent].index;
        let child_index = parent_index - 1;
        let (position, energy, gradient) = if child_index == 0 {
            self.minimize(surface, start)?
        } else {
            let warm = without_column(leg.modes, leg.direction);
            let position = self.pursue(surface, start, child_index, warm)?;
            let (energy, gradient) = surface.eval(position.view())?;
            (position, energy, gradient)
        };
        let force = self.config.force_gate.value(gradient.view());
        let tol = if child_index == 0 {
            self.config.relax.force_tol
        } else {
            self.config.search.search.force_tol
        };
        if !(energy.is_finite() && force <= tol) {
            return Err(SaddleError::Solver(format!(
                "downward leg from node {} direction {} stopped at force {force}",
                leg.parent, leg.direction
            )));
        }
        if let Some(known) = nearest(
            nodes,
            self.config.cell.as_ref(),
            position.view(),
            self.config.duplicate_distance,
        )? {
            if nodes[known].index != child_index {
                return Err(SaddleError::Solver(format!(
                    "downward leg from node {} direction {} returned index {} at node {known}, child index is {child_index}",
                    leg.parent, leg.direction, nodes[known].index
                )));
            }
            return Ok(Branch::Known(known));
        }
        let certified = self.certify(surface, position.view(), gradient.view(), Vec::new())?;
        if !certified.index.converged {
            return Err(SaddleError::Solver(
                "child saddle-index certificate did not converge".into(),
            ));
        }
        if certified.index.index != child_index {
            return Err(SaddleError::Solver(format!(
                "downward leg from node {} direction {} certified index {}, child index is {child_index}",
                leg.parent, leg.direction, certified.index.index
            )));
        }
        Ok(Branch::New(point_from(position, energy, certified)?))
    }

    fn certify<S: PointSurface>(
        &self,
        surface: &Budget<'_, S>,
        x: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
        hints: Vec<Array1<f64>>,
    ) -> Result<Certificate, SaddleError> {
        let excluded_matrix = surface.excluded_modes(x)?;
        if excluded_matrix.ncols() != x.len() && excluded_matrix.nrows() != 0 {
            return Err(SaddleError::Shape(
                "excluded modes must have one column per coordinate".into(),
            ));
        }
        let excluded: Vec<Array1<f64>> = excluded_matrix
            .rows()
            .into_iter()
            .map(|row| row.to_owned())
            .collect();
        let mut check = self.config.certificate;
        check.escape = None;
        saddle_index::certify(
            &check,
            surface,
            Probe {
                x,
                gradient,
                held: Vec::new(),
                excluded,
                hints,
            },
        )
    }

    fn pursue<S: PointSurface>(
        &self,
        surface: &Budget<'_, S>,
        start: Array1<f64>,
        k: usize,
        warm: Array2<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let mut search = self.config.search.search.clone();
        search.saddle_check = None;
        search.artn = None;
        let config = IndexKConfig::new(k, search)
            .with_residual_tol(self.config.search.residual_tol)
            .with_max_block_iterations(self.config.search.max_block_iterations)
            .with_block(self.config.search.block);
        let warm = if warm.ncols() == k { Some(warm) } else { None };
        let mut session = IndexKSession::new(config, start, warm)?;
        session.set_force_gate(self.config.force_gate);
        let mut report = session.step(surface)?;
        while report.status == MinModeStatus::Running && report.iteration < self.config.max_steps {
            report = session.step(surface)?;
        }
        if report.status != MinModeStatus::Converged {
            return Err(SaddleError::Solver(format!(
                "index-{k} search did not converge in {} steps at force {}",
                report.iteration, report.max_force
            )));
        }
        Ok(report.position)
    }

    fn minimize<S: PointSurface>(
        &self,
        surface: &Budget<'_, S>,
        start: Array1<f64>,
    ) -> Result<(Array1<f64>, f64, Array1<f64>), SaddleError> {
        let n = start.len();
        let (energy, gradient) = surface.eval(start.view())?;
        let tol = self.config.relax.force_tol;
        if energy.is_finite() && self.config.force_gate.value(gradient.view()) <= tol {
            return Ok((start, energy, gradient));
        }
        let steps = self.config.max_steps.min(self.config.relax.max_steps);
        let mut solver = Solver::new(
            self.config.relax.method.clone(),
            Control::default()
                .with_maxiter(steps)
                .with_gtol(tol)
                .with_stop_norm(stop_norm(self.config.force_gate))
                .with_istep(self.config.relax.initial_step),
            n,
        );
        solver.set_atom_maxmove(self.config.relax.max_atom_move);
        let slot: Mutex<Option<SaddleError>> = Mutex::new(None);
        let oracle = Oracle::unbounded(n, |x: ArrayView1<f64>| match surface.eval(x) {
            Ok(value) => value,
            Err(err) => {
                let mut guard = match slot.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if guard.is_none() {
                    *guard = Some(err);
                }
                (f64::NAN, Array1::zeros(x.len()))
            }
        });
        let mut position = start;
        let stepped = solver.run(&oracle, &mut position);
        let stored = match slot.lock() {
            Ok(mut guard) => guard.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(err) = stored {
            return Err(err);
        }
        let report = stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        let (energy, gradient) = surface.eval(position.view())?;
        let force = self.config.force_gate.value(gradient.view());
        if !(energy.is_finite() && force <= tol) {
            return Err(SaddleError::Solver(format!(
                "minimizer stopped at force {force} after {} steps ({:?})",
                report.steps, report.status
            )));
        }
        Ok((position, energy, gradient))
    }
}

enum Branch {
    New(CriticalPoint),
    Known(usize),
}

struct Budget<'a, S> {
    inner: &'a S,
    calls: AtomicUsize,
    limit: usize,
}

impl<S: PointSurface> PointSurface for Budget<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let used = self.calls.load(Ordering::Relaxed);
        if used >= self.limit {
            return Err(SaddleError::LandscapeBound {
                kind: LandscapeBoundKind::Evaluations,
                limit: self.limit,
                nodes: 0,
                evaluations: used,
            });
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.inner.hessian_vector(x, vector)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.inner.excluded_modes(x)
    }
}

fn note(err: SaddleError, nodes: usize, evaluations: usize) -> SaddleError {
    match err {
        SaddleError::LandscapeBound { kind, limit, .. } => SaddleError::LandscapeBound {
            kind,
            limit,
            nodes,
            evaluations,
        },
        other => other,
    }
}

fn sign_byte(sign: f64) -> i8 {
    if sign > 0.0 { 1 } else { -1 }
}

fn stop_norm(gate: ForceGate) -> StopNorm {
    match gate {
        ForceGate::L2Norm => StopNorm::Two,
        ForceGate::LinfNorm => StopNorm::Inf,
        ForceGate::MaxForceOnAtom => StopNorm::MaxAtom,
    }
}

fn point_from(
    position: Array1<f64>,
    energy: f64,
    certified: Certificate,
) -> Result<CriticalPoint, SaddleError> {
    let index = certified.index.index;
    if certified.vectors.len() < index {
        return Err(SaddleError::Solver(
            "saddle-index certificate is missing an unstable eigenvector".into(),
        ));
    }
    let n = position.len();
    let mut modes = Array2::zeros((n, index));
    for (column, vector) in certified.vectors.iter().take(index).enumerate() {
        modes
            .column_mut(column)
            .assign(&unit_column(vector.view())?);
    }
    Ok(CriticalPoint {
        position,
        energy,
        index,
        curvatures: certified.index.curvatures,
        modes,
        evaluations: 0,
    })
}

fn independent_modes(modes: &Array2<f64>) -> Result<(), SaddleError> {
    if !modes.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("landscape mode"));
    }
    let orth = crate::linalg::modified_gram_schmidt(modes.view(), None, MODE_DEPENDENCE_TOL);
    if orth.ncols() != modes.ncols() {
        return Err(SaddleError::Invalid(
            "unstable block is linearly dependent".into(),
        ));
    }
    Ok(())
}

fn unit_column(vector: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    if !vector.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("unstable eigenvector"));
    }
    let norm = vector.dot(&vector).sqrt();
    if !(norm.is_finite() && norm > MODE_DEPENDENCE_TOL) {
        return Err(SaddleError::Solver(
            "certified unstable eigenvector has no length".into(),
        ));
    }
    Ok(vector.to_owned() / norm)
}

struct Leg<'a> {
    parent: usize,
    origin: &'a Array1<f64>,
    modes: &'a Array2<f64>,
    direction: usize,
    sign: f64,
}

fn without_column(modes: &Array2<f64>, drop: usize) -> Array2<f64> {
    let mut out = Array2::zeros((modes.nrows(), modes.ncols().saturating_sub(1)));
    let mut column = 0;
    for index in 0..modes.ncols() {
        if index == drop {
            continue;
        }
        out.column_mut(column).assign(&modes.column(index));
        column += 1;
    }
    out
}

fn nearest(
    nodes: &[CriticalPoint],
    cell: Option<&Cell>,
    position: ArrayView1<f64>,
    tolerance: f64,
) -> Result<Option<usize>, SaddleError> {
    let mut best: Option<(usize, f64)> = None;
    for (index, node) in nodes.iter().enumerate() {
        let distance = separation(cell, node.position.view(), position)?;
        if distance <= tolerance && best.is_none_or(|(_, so_far)| distance < so_far) {
            best = Some((index, distance));
        }
    }
    Ok(best.map(|(index, _)| index))
}

/// Euclidean length of the minimum-image displacement from `b` to `a`.
fn separation(
    cell: Option<&Cell>,
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
) -> Result<f64, SaddleError> {
    if a.len() != b.len() {
        return Err(SaddleError::Shape(
            "critical-point positions must share a length".into(),
        ));
    }
    let mut diff = &a.to_owned() - &b;
    if let Some(cell) = cell {
        cell.minimum_image(&mut diff);
    }
    let distance = diff.dot(&diff).sqrt();
    if !distance.is_finite() {
        return Err(SaddleError::NonFinite("critical-point distance"));
    }
    Ok(distance)
}
