//! Growing string method of Peters, Heyden, Bell, and Chakraborty,
//! J. Chem. Phys. 120, 7877 (2004), doi:10.1063/1.1691018, with
//! hand-off upon join to the simplified [`StringSession`] of E, Ren,
//! and Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007),
//! doi:10.1063/1.2720838.
//!
//! The search starts from two separate string fragments anchored at
//! the reactant and product endpoints, each carrying one initial
//! interior frontier node (\(w_{\mathrm{lf}}\) and \(w_{\mathrm{rf}}\)).
//! On each step while growing:
//! 1. All movable fragment nodes are evaluated together in a single
//!    batched surface call with their stable [`ImageId`]s preserved
//!    (endpoints are evaluated once and cached).
//! 2. Path tangents \(\hat{t}_k\) are evaluated from the cubic spline
//!    ([`Reparametrization::NotAKnot`], default) or polyline
//!    ([`Reparametrization::Linear`]) spanning both fragments across
//!    the vacant interior gap, and both the forces and frontier step
//!    displacements are constrained perpendicular to \(\hat{t}_k\).
//! 3. The reactant and product fragments relax separately through
//!    independent [`rgmin::Solver`] instances and secant histories.
//! 4. Each frontier that satisfies the configurable node-add criterion
//!    (\(\lVert f^\perp(w_{\mathrm{frontier}}) \rVert \le
//!    \mathrm{frontier\_tol}\) after `min_steps_per_node`, or upon
//!    reaching `max_steps_per_node`) spawns a new frontier node inward
//!    across the gap with a fresh [`ImageId`] and reparametrizes its
//!    fragment to equal arc length.
//! 5. The fragments join when the string is full
//!    (`left_nodes + right_nodes == n_images`) and either this step
//!    added the node that filled the last slot or both frontiers
//!    satisfy the node-add criterion, so no node is left to grow, or
//!    when the frontier gap falls within `join_tol`. A full string
//!    whose frontiers are not both ready keeps relaxing as fragments.
//!    On a join the fragments merge into a continuous equal-arc-length
//!    chain and hand off to [`StringSession`], preserving every
//!    image's [`ImageId`] and cached endpoint energies.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::{FireKind, Method, Oracle, Solver};

use super::string::{StringConfig, StringSession, check_string};
use super::{
    BandConfig, BandStatus, BandSurface, Cell, CiConfig, MinModeRefine, SAME_ROW_TOL, build_solver,
    check_fixed, check_force_driven, pin_fixed_rows, remove_net_translation, zero_fixed,
};
use crate::autoneb::ImageEvalPool;
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::reparam::{Reparametrization, Spline, reparametrize};
use crate::saddle_index::{SaddleIndex, SaddleIndexCheck};
use crate::stencil::{ImageId, perp};
use crate::tangent::{TangentKind, compute_tangent};

/// Phase of a [`GrowingStringSession`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowingStringPhase {
    /// Reactant and product fragments are growing inward and relaxing separately.
    Growing,
    /// Fragments have joined and handed off to the simplified [`StringSession`].
    Joined,
}

/// Configuration for [`GrowingStringSession`].
///
/// Peters, Heyden, Bell, and Chakraborty, J. Chem. Phys. 120, 7877
/// (2004), doi:10.1063/1.1691018.
///
/// Non-exhaustive: start from [`GrowingStringConfig::default`] and
/// chain the `with_*` setters.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView2, array};
/// use rgsaddle::{
///     BandStatus, BandSurface, DEFAULT_FORCE_TOL, GrowingStringConfig, GrowingStringPhase,
///     GrowingStringSession, SaddleError,
/// };
///
/// struct CurvedDoubleWell;
///
/// impl BandSurface for CurvedDoubleWell {
///     fn eval(
///         &self,
///         positions: ArrayView2<f64>,
///         energies: &mut Array1<f64>,
///         gradients: &mut Array2<f64>,
///     ) -> Result<(), SaddleError> {
///         for (i, r) in positions.outer_iter().enumerate() {
///             let (x, y, z) = (r[0], r[1], r[2]);
///             let valley = y - (1.0 - x * x);
///             energies[i] = (1.0 - x * x).powi(2) + 4.0 * valley * valley + 2.0 * z * z;
///             gradients[(i, 0)] = -4.0 * x * (1.0 - x * x) + 16.0 * x * valley;
///             gradients[(i, 1)] = 8.0 * valley;
///             gradients[(i, 2)] = 4.0 * z;
///         }
///         Ok(())
///     }
/// }
///
/// let config = GrowingStringConfig::default()
///     .with_n_images(9)
///     .with_frontier_tol(0.25)
///     .with_force_tol(DEFAULT_FORCE_TOL)
///     .with_max_move(0.1)
///     .with_min_mode(None);
/// let mut session = GrowingStringSession::new(
///     config,
///     array![-1.0, 0.0, 0.0],
///     array![1.0, 0.0, 0.0],
/// )?;
/// let report = session.run(&CurvedDoubleWell, 600)?;
/// assert_eq!(report.status, BandStatus::Converged);
/// assert_eq!(report.phase, GrowingStringPhase::Joined);
/// assert!(session.string_session().is_some());
/// # Ok::<(), SaddleError>(())
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GrowingStringConfig {
    /// Underlying band and simplified-string configuration used during
    /// growth and handed to [`StringSession`] once the fragments join.
    pub band: BandConfig,
    /// Target total number of images in the full joined string
    /// (endpoints included; at least 4, 11 by default).
    pub n_images: usize,
    /// Frontier perpendicular-force threshold for adding a new node
    /// (Peters et al. 2004, Eq. 7). Default is `0.2` (`1.0 / 5.0`).
    pub frontier_tol: f64,
    /// Minimum relaxation steps at the current fragment size before a
    /// frontier may spawn the next node. Default is `1`.
    pub min_steps_per_node: usize,
    /// Optional cap on relaxation steps at the current fragment size
    /// before forcing a node addition. `None` uses an automatic cap of
    /// `min_steps_per_node + n_images`.
    pub max_steps_per_node: Option<usize>,
    /// Optional Euclidean distance threshold between the left and right
    /// frontiers at or below which the fragments join immediately and
    /// fill to `n_images`. `None` by default.
    pub join_tol: Option<f64>,
    /// Norm used to test `frontier_tol` and report `max_force`.
    pub force_gate: ForceGate,
}

with_setters!(GrowingStringConfig {
    with_band => band: val BandConfig,
    with_n_images => n_images: val usize,
    with_frontier_tol => frontier_tol: val f64,
    with_min_steps_per_node => min_steps_per_node: val usize,
    with_max_steps_per_node => max_steps_per_node: opt usize,
    with_join_tol => join_tol: opt f64,
    with_force_gate => force_gate: val ForceGate,
});

impl GrowingStringConfig {
    /// Equal arc-length reparametrization scheme for fragment and full-string placement.
    #[must_use]
    pub fn with_reparametrization(mut self, reparametrization: Reparametrization) -> Self {
        self.band.string = Some(StringConfig::default().with_reparametrization(reparametrization));
        self
    }

    /// Convergence tolerance on the full string after join.
    #[must_use]
    pub fn with_force_tol(mut self, force_tol: f64) -> Self {
        self.band.force_tol = force_tol;
        self
    }

    /// Maximum per-atom displacement per step.
    #[must_use]
    pub fn with_max_move(mut self, max_move: f64) -> Self {
        self.band.max_move = max_move;
        self
    }

    /// First-order stepper method.
    #[must_use]
    pub fn with_method(mut self, method: Method) -> Self {
        self.band.method = method;
        self
    }

    /// Climbing-image configuration after the fragments join.
    #[must_use]
    pub fn with_climbing(mut self, climbing: Option<CiConfig>) -> Self {
        self.band.climbing = climbing;
        self
    }

    /// OCI-NEB minimum-mode refinement on the joined string's climbing image.
    #[must_use]
    pub fn with_min_mode(mut self, min_mode: Option<MinModeRefine>) -> Self {
        self.band.min_mode = min_mode;
        self
    }

    /// Saddle-index certification on the joined string's climbing image.
    #[must_use]
    pub fn with_saddle_check(mut self, saddle_check: Option<SaddleIndexCheck>) -> Self {
        self.band.saddle_check = saddle_check;
        self
    }

    /// Periodic simulation cell for minimum-image differences.
    #[must_use]
    pub fn with_cell(mut self, cell: Option<Cell>) -> Self {
        self.band.cell = cell;
        self
    }

    /// Per-atom fixed mask shared across all images.
    #[must_use]
    pub fn with_fixed_atoms(mut self, fixed_atoms: Vec<bool>) -> Self {
        self.band.fixed_atoms = fixed_atoms;
        self
    }

    /// Whether to remove mean Cartesian translation on free-atom bands.
    #[must_use]
    pub fn with_remove_translation(mut self, remove_translation: bool) -> Self {
        self.band.remove_translation = remove_translation;
        self
    }

    /// Initial solver time step (`dt`).
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.band.time_step = time_step;
        self
    }

    /// Select the quick-min stepper.
    #[must_use]
    pub fn with_quickmin(mut self, quickmin: bool) -> Self {
        self.band.quickmin = quickmin;
        self
    }
}

impl Default for GrowingStringConfig {
    fn default() -> Self {
        Self {
            band: BandConfig::default().with_string(StringConfig::spline()),
            n_images: 11,
            frontier_tol: 1.0 / 5.0,
            min_steps_per_node: 1,
            max_steps_per_node: None,
            join_tol: None,
            force_gate: ForceGate::LinfNorm,
        }
    }
}

/// Diagnostics reported after one [`GrowingStringSession::step`].
#[derive(Clone, Debug)]
pub struct GrowingStringReport {
    /// Whether the session is still running or has converged after joining.
    pub status: BandStatus,
    /// Current phase (`Growing` while fragments advance, `Joined` after hand-off).
    pub phase: GrowingStringPhase,
    /// Largest convergence force norm across the active interior images.
    pub max_force: f64,
    /// Convergence force norm at the reactant (left) frontier node (`0.0` after join).
    pub left_frontier_force: f64,
    /// Convergence force norm at the product (right) frontier node (`0.0` after join).
    pub right_frontier_force: f64,
    /// Number of nodes on the reactant fragment (endpoints included; `0` after join).
    pub left_nodes: usize,
    /// Number of nodes on the product fragment (endpoints included; `0` after join).
    pub right_nodes: usize,
    /// Total active images (`left_nodes + right_nodes` while growing, `n_images` once joined).
    pub n_images: usize,
    /// Nodes grown on this step (`0`, `1`, or `2`).
    pub nodes_added: usize,
    /// Whether the two fragments joined and handed off to [`StringSession`] on this step.
    pub joined_this_step: bool,
    /// Climbing-image index once joined and armed, if any.
    pub ci_index: Option<usize>,
    /// Total steps taken by this session.
    pub iteration: usize,
    /// Surface image rows evaluated during this step.
    pub surface_rows: usize,
    /// Cumulative surface image rows evaluated across all steps of this session.
    pub total_surface_rows: usize,
    /// Certified saddle index of the climbing image at convergence, if enabled.
    pub saddle_index: Option<SaddleIndex>,
}

/// Independent optimizer state for one growing fragment (`F_left` or `F_right`).
struct FragmentRelaxer {
    solver: Solver,
    pairs: Vec<(Array1<f64>, Array1<f64>)>,
    prev_step: Option<Array1<f64>>,
    prev_force: Option<Array1<f64>>,
    dt: Option<f64>,
    steps_at_size: usize,
}

impl FragmentRelaxer {
    fn new(config: &BandConfig, interior_dof: usize) -> Self {
        let method = if config.quickmin {
            Method::QuickMin
        } else {
            config.method.clone()
        };
        let solver = build_solver(config, &method, interior_dof);
        Self {
            solver,
            pairs: Vec::new(),
            prev_step: None,
            prev_force: None,
            dt: None,
            steps_at_size: 0,
        }
    }

    fn update_from_previous(
        &mut self,
        config: &BandConfig,
        reparam: Reparametrization,
        current_force: &Array1<f64>,
    ) {
        let (Some(s_vec), Some(prev_f)) = (self.prev_step.take(), self.prev_force.take()) else {
            return;
        };
        let interior_dof = current_force.len();
        if s_vec.len() != interior_dof || prev_f.len() != interior_dof {
            self.pairs.clear();
            return;
        }
        let y_vec = &prev_f - current_force;
        let sy = s_vec.dot(&y_vec);
        let yy = y_vec.dot(&y_vec);
        let overshot = s_vec.dot(current_force) < 0.0;

        let mut dt_cur = match self.dt {
            Some(dt) => dt,
            None => config.time_step,
        };
        if sy > 0.0 && yy > 0.0 {
            let alpha_bb = sy / yy;
            let ceiling = if config.quickmin {
                alpha_bb.sqrt()
            } else {
                match config.method {
                    Method::Fire {
                        kind: FireKind::V2, ..
                    } => (4.0 * alpha_bb).sqrt(),
                    Method::Fire { .. } => alpha_bb.sqrt(),
                    _ => alpha_bb,
                }
            };
            if ceiling < dt_cur && ceiling > 0.0 && ceiling.is_finite() {
                dt_cur = ceiling;
            }
        }
        if overshot {
            dt_cur *= 0.5;
        }
        self.dt = Some(dt_cur);

        if !config.quickmin
            && let Method::Lbfgs { memory, .. } = config.method
        {
            let uphill_kink = reparam == Reparametrization::Linear && s_vec.dot(&prev_f) <= 0.0;
            self.solver.forget();
            if sy > 0.0 && !uphill_kink {
                if memory > 0 {
                    if self.pairs.len() == memory {
                        self.pairs.remove(0);
                    }
                    self.pairs.push((s_vec, y_vec));
                }
                for (ps, py) in &self.pairs {
                    let _ = self.solver.push_pair(ps.view(), py.view());
                }
            } else {
                self.pairs.clear();
            }
        } else {
            let method = if config.quickmin {
                Method::QuickMin
            } else {
                config.method.clone()
            };
            let mut stepped_cfg = config.clone();
            stepped_cfg.time_step = dt_cur;
            self.solver = build_solver(&stepped_cfg, &method, interior_dof);
        }
    }

    fn resize(&mut self, config: &BandConfig, interior_dof: usize) {
        let mut stepped_cfg = config.clone();
        if let Some(dt) = self.dt {
            stepped_cfg.time_step = dt;
        }
        let method = if config.quickmin {
            Method::QuickMin
        } else {
            config.method.clone()
        };
        self.solver = build_solver(&stepped_cfg, &method, interior_dof);
        self.pairs.clear();
        self.prev_step = None;
        self.prev_force = None;
        self.steps_at_size = 0;
    }

    fn step_fragment(
        &mut self,
        config: &BandConfig,
        before: &Array1<f64>,
        force_perp: &Array1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let dim = before.len();
        let norm = force_perp.dot(force_perp).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Ok(before.clone());
        }
        if !config.quickmin
            && matches!(config.method, Method::Lbfgs { .. })
            && self.pairs.is_empty()
        {
            let scale = match self.dt {
                Some(dt) => dt,
                None => config.time_step,
            };
            let mut step = force_perp * scale;
            let max_atom = super::min_mode::max_atom(step.view());
            if max_atom > config.max_move && max_atom > 0.0 {
                step *= config.max_move / max_atom;
            }
            let x_evol = before + &step;
            if !x_evol.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("growing string fragment step"));
            }
            return Ok(x_evol);
        }
        let start_grad = -force_perp;
        let oracle = Oracle::unbounded(dim, |xv: ArrayView1<f64>| {
            if xv == before.view() {
                (0.0, start_grad.clone())
            } else {
                let delta = &xv - before;
                (-force_perp.dot(&delta), start_grad.clone())
            }
        });
        let mut x_evol = before.clone();
        self.solver
            .step(&oracle, &mut x_evol)
            .map_err(|e| SaddleError::Solver(e.to_string()))?;
        if !x_evol.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("growing string fragment step"));
        }
        Ok(x_evol)
    }
}

/// Growing string session of Peters, Heyden, Bell, and Chakraborty,
/// J. Chem. Phys. 120, 7877 (2004), doi:10.1063/1.1691018, handing off
/// to [`StringSession`] once the reactant and product fragments join.
pub struct GrowingStringSession {
    config: GrowingStringConfig,
    left_pos: Array2<f64>,
    right_pos: Array2<f64>,
    left_ids: Vec<ImageId>,
    right_ids: Vec<ImageId>,
    next_id: ImageId,
    left_fragment: FragmentRelaxer,
    right_fragment: FragmentRelaxer,
    endpoint_energies: Option<[f64; 2]>,
    last_energies: Vec<f64>,
    last_frontier_tangents: Option<(Array1<f64>, Array1<f64>)>,
    pool: Option<ImageEvalPool>,
    joined: Option<StringSession>,
    iteration: usize,
    total_surface_rows: usize,
}

fn validate_growing_config(
    config: &GrowingStringConfig,
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
) -> Result<(), SaddleError> {
    if config.n_images < 4 {
        return Err(SaddleError::Shape(format!(
            "growing string needs n_images >= 4; got {}",
            config.n_images
        )));
    }
    let dof = reactant.len();
    if dof == 0 || !dof.is_multiple_of(3) || product.len() != dof {
        return Err(SaddleError::Shape(format!(
            "growing string endpoints must share a positive 3N length; got {} and {}",
            reactant.len(),
            product.len()
        )));
    }
    if !reactant.iter().chain(product.iter()).all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("growing string endpoints"));
    }
    if !(config.frontier_tol.is_finite() && config.frontier_tol > 0.0) {
        return Err(SaddleError::Invalid(
            "growing string frontier_tol must be finite and positive".into(),
        ));
    }
    if config.min_steps_per_node == 0 {
        return Err(SaddleError::Invalid(
            "growing string min_steps_per_node must be at least 1".into(),
        ));
    }
    if let Some(max_steps) = config.max_steps_per_node
        && max_steps < config.min_steps_per_node
    {
        return Err(SaddleError::Invalid(
            "growing string max_steps_per_node must be at least min_steps_per_node".into(),
        ));
    }
    if let Some(join_tol) = config.join_tol
        && !(join_tol.is_finite() && join_tol >= 0.0)
    {
        return Err(SaddleError::Invalid(
            "growing string join_tol must be finite and non-negative".into(),
        ));
    }
    if !(config.band.force_tol.is_finite() && config.band.force_tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "band force tolerance must be finite and non-negative".into(),
        ));
    }
    if !(config.band.max_move.is_finite() && config.band.max_move > 0.0) {
        return Err(SaddleError::Invalid(
            "band max_move must be finite and positive".into(),
        ));
    }
    if !(config.band.time_step.is_finite() && config.band.time_step > 0.0) {
        return Err(SaddleError::Invalid(
            "band time step must be positive and finite".into(),
        ));
    }
    check_string(&config.band)?;
    let method = if config.band.quickmin {
        Method::QuickMin
    } else {
        config.band.method.clone()
    };
    check_force_driven(&method)?;
    if let Some(refine) = &config.band.min_mode {
        refine.validate()?;
    }
    if let Some(check) = &config.band.saddle_check {
        check.validate()?;
    }
    let mut pair = Array2::zeros((2, dof));
    pair.row_mut(0).assign(&reactant);
    pair.row_mut(1).assign(&product);
    check_fixed(&config.band, pair.view())?;
    Ok(())
}

impl GrowingStringSession {
    fn auto_max_steps_per_node(&self) -> usize {
        self.config.min_steps_per_node + self.config.n_images
    }

    fn max_steps_per_node(&self) -> usize {
        match self.config.max_steps_per_node {
            Some(cap) => cap,
            None => self.auto_max_steps_per_node(),
        }
    }

    fn knot_tolerance(&self, s: &[f64]) -> f64 {
        if s.is_empty() {
            return SAME_ROW_TOL;
        }
        let span = s[s.len() - 1] - s[0];
        let scale = if span.is_finite() && span > 0.0 {
            span
        } else {
            1.0
        };
        scale * (s.len() as f64) * f64::EPSILON.sqrt()
    }

    fn distinct_knots_for_spline(
        &self,
        s: &[f64],
        points: &Array2<f64>,
    ) -> (Vec<usize>, Vec<f64>, Array2<f64>) {
        let n = s.len();
        let tol = self.knot_tolerance(s);
        let mut kept = vec![0];
        for i in 1..n {
            let last = kept[kept.len() - 1];
            if s[i] - s[last] > tol {
                kept.push(i);
            } else if i + 1 == n && last != 0 {
                let end = kept.len() - 1;
                kept[end] = i;
            }
        }
        let knots = kept.iter().map(|&i| s[i]).collect::<Vec<_>>();
        let mut values = Array2::zeros((kept.len(), points.ncols()));
        for (row, &i) in kept.iter().enumerate() {
            values.row_mut(row).assign(&points.row(i));
        }
        (kept, knots, values)
    }

    fn fit_chain_spline(&self, unwrapped: &Array2<f64>, s: &[f64]) -> Option<(Vec<usize>, Spline)> {
        let (kept, knots, values) = self.distinct_knots_for_spline(s, unwrapped);
        Spline::fit(&knots, &values).map(|spline| (kept, spline))
    }

    fn frontier_arc(s: &[f64], kept: &[usize], arcs: &[f64], at: usize) -> f64 {
        let target = s[at];
        let first = kept[0];
        if target <= s[first] {
            return arcs[0];
        }
        let last = kept[kept.len() - 1];
        if target >= s[last] {
            return arcs[arcs.len() - 1];
        }
        for hi in 1..kept.len() {
            let lo = hi - 1;
            let i0 = kept[lo];
            let i1 = kept[hi];
            let s0 = s[i0];
            let s1 = s[i1];
            if target <= s1 {
                let span = s1 - s0;
                if !(span.is_finite() && span > 0.0) {
                    return arcs[lo];
                }
                let frac = (target - s0) / span;
                return arcs[lo] + frac * (arcs[hi] - arcs[lo]);
            }
        }
        arcs[arcs.len() - 1]
    }

    /// Start a growing string session between `reactant` and `product`
    /// (`3 * n_atoms` coordinates each).
    ///
    /// Image IDs `0` and `1` are assigned to the reactant and product
    /// endpoints, `2` and `3` to the initial left and right frontier
    /// nodes, and `4, 5, ...` to each node grown inward along the path.
    pub fn new(
        mut config: GrowingStringConfig,
        reactant: Array1<f64>,
        product: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        if config.band.string.is_none() {
            config.band.string = Some(StringConfig::spline());
        }
        validate_growing_config(&config, reactant.view(), product.view())?;
        let dof = reactant.len();
        let mut chord = &product - &reactant;
        if let Some(cell) = &config.band.cell {
            cell.minimum_image(&mut chord);
        }
        zero_fixed(&config.band.fixed_atoms, &mut chord);
        let span = chord.dot(&chord).sqrt();
        if !(span.is_finite() && span > SAME_ROW_TOL) {
            return Err(SaddleError::Invalid(
                "growing string endpoints must be distinct".into(),
            ));
        }

        let n_seg = (config.n_images - 1) as f64;
        let step_frac = 1.0 / n_seg;
        let left_frontier = &reactant + &(&chord * step_frac);
        let right_frontier = &reactant + &(&chord * (1.0 - step_frac));

        let mut left_pos = Array2::zeros((2, dof));
        left_pos.row_mut(0).assign(&reactant);
        left_pos.row_mut(1).assign(&left_frontier);
        pin_fixed_rows(&config.band.fixed_atoms, &mut left_pos);

        let mut right_pos = Array2::zeros((2, dof));
        right_pos.row_mut(0).assign(&right_frontier);
        right_pos.row_mut(1).assign(&product);
        if !config.band.fixed_atoms.is_empty() {
            for (atom, _) in config
                .band
                .fixed_atoms
                .iter()
                .enumerate()
                .filter(|(_, f)| **f)
            {
                for c in 3 * atom..3 * atom + 3 {
                    right_pos[(0, c)] = reactant[c];
                }
            }
        }

        let left_fragment = FragmentRelaxer::new(&config.band, dof);
        let right_fragment = FragmentRelaxer::new(&config.band, dof);

        Ok(Self {
            config,
            left_pos,
            right_pos,
            left_ids: vec![0, 2],
            right_ids: vec![3, 1],
            next_id: 4,
            left_fragment,
            right_fragment,
            endpoint_energies: None,
            last_energies: Vec::new(),
            last_frontier_tangents: None,
            pool: None,
            joined: None,
            iteration: 0,
            total_surface_rows: 0,
        })
    }

    /// Construct a [`GrowingStringSession`] from a `2 x 3N` array of
    /// `[reactant, product]` rows.
    pub fn from_endpoints(
        config: GrowingStringConfig,
        endpoints: ArrayView2<f64>,
    ) -> Result<Self, SaddleError> {
        if endpoints.nrows() != 2 {
            return Err(SaddleError::Shape(format!(
                "growing string from_endpoints expects 2 rows; got {}",
                endpoints.nrows()
            )));
        }
        Self::new(
            config,
            endpoints.row(0).to_owned(),
            endpoints.row(1).to_owned(),
        )
    }

    /// Current phase (`Growing` or `Joined`).
    #[must_use]
    pub fn phase(&self) -> GrowingStringPhase {
        if self.joined.is_some() {
            GrowingStringPhase::Joined
        } else {
            GrowingStringPhase::Growing
        }
    }

    /// Whether the reactant and product fragments have joined and handed
    /// off to [`StringSession`].
    #[must_use]
    pub fn is_joined(&self) -> bool {
        self.joined.is_some()
    }

    /// Borrow the simplified [`StringSession`] once the fragments have joined.
    #[must_use]
    pub fn string_session(&self) -> Option<&StringSession> {
        self.joined.as_ref()
    }

    /// Consume into the simplified [`StringSession`] once the fragments have joined.
    #[must_use]
    pub fn into_string_session(self) -> Option<StringSession> {
        self.joined
    }

    /// Current positions along the chain: `[left_fragment; right_fragment]`
    /// while growing, or the full `n_images x 3N` string once joined.
    #[must_use]
    pub fn positions(&self) -> Array2<f64> {
        if let Some(joined) = &self.joined {
            return joined.positions().to_owned();
        }
        let m_l = self.left_pos.nrows();
        let m_r = self.right_pos.nrows();
        let dof = self.left_pos.ncols();
        let mut out = Array2::zeros((m_l + m_r, dof));
        out.slice_mut(s![..m_l, ..]).assign(&self.left_pos);
        out.slice_mut(s![m_l.., ..]).assign(&self.right_pos);
        out
    }

    /// Reactant (left) fragment positions, from fixed reactant row `0` to
    /// the left frontier `left_nodes - 1`. After joining, this is the
    /// reactant-side partition of the current reparametrized string.
    #[must_use]
    pub fn left_fragment(&self) -> ArrayView2<'_, f64> {
        if let Some(joined) = &self.joined {
            return joined
                .positions()
                .slice_move(s![..self.left_pos.nrows(), ..]);
        }
        self.left_pos.view()
    }

    /// Product (right) fragment positions, from the right frontier row `0`
    /// to the fixed product `right_nodes - 1`. After joining, this is the
    /// product-side partition of the current reparametrized string.
    #[must_use]
    pub fn right_fragment(&self) -> ArrayView2<'_, f64> {
        if let Some(joined) = &self.joined {
            return joined
                .positions()
                .slice_move(s![self.left_pos.nrows().., ..]);
        }
        self.right_pos.view()
    }

    /// Stable [`ImageId`]s of the current chain rows in order.
    #[must_use]
    pub fn ids(&self) -> Vec<ImageId> {
        if let Some(joined) = &self.joined {
            return joined.ids().to_vec();
        }
        let mut out = Vec::with_capacity(self.left_ids.len() + self.right_ids.len());
        out.extend_from_slice(&self.left_ids);
        out.extend_from_slice(&self.right_ids);
        out
    }

    /// Stable [`ImageId`]s of the reactant fragment.
    #[must_use]
    pub fn left_ids(&self) -> &[ImageId] {
        &self.left_ids
    }

    /// Stable [`ImageId`]s of the product fragment.
    #[must_use]
    pub fn right_ids(&self) -> &[ImageId] {
        &self.right_ids
    }

    /// Unit frontier tangents `(\hat{t}_{lf}, \hat{t}_{rf})` from the most
    /// recent growing step, if any.
    #[must_use]
    pub fn frontier_tangents(&self) -> Option<(ArrayView1<'_, f64>, ArrayView1<'_, f64>)> {
        self.last_frontier_tangents
            .as_ref()
            .map(|(l, r)| (l.view(), r.view()))
    }

    /// Energies of the current chain images from the most recent evaluation.
    #[must_use]
    pub fn energies(&self) -> Vec<f64> {
        if let Some(joined) = &self.joined
            && let Some(eval) = joined.evaluation()
        {
            return eval.energies.to_vec();
        }
        self.last_energies.clone()
    }

    /// Certified saddle index of the climbing image at convergence, if any.
    #[must_use]
    pub fn saddle_index(&self) -> Option<&SaddleIndex> {
        self.joined.as_ref().and_then(|s| s.saddle_index())
    }

    /// Number of steps taken so far.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Cumulative surface image evaluations performed across the session.
    #[must_use]
    pub fn total_surface_rows(&self) -> usize {
        self.total_surface_rows
    }

    /// Set worker threads for batched image evaluation.
    pub fn set_image_threads(&mut self, threads: usize) -> Result<(), SaddleError> {
        self.pool = match threads {
            1 => None,
            _ => Some(ImageEvalPool::new(threads)?),
        };
        if let Some(joined) = &mut self.joined {
            joined.set_image_threads(threads)?;
        }
        Ok(())
    }

    /// Worker thread count for batched image evaluation.
    #[must_use]
    pub fn image_threads(&self) -> usize {
        match &self.pool {
            Some(pool) => pool.threads(),
            None => 1,
        }
    }

    /// Replace the per-atom fixed mask.
    pub fn set_fixed_atoms(&mut self, fixed: Vec<bool>) -> Result<(), SaddleError> {
        if let Some(joined) = &mut self.joined {
            self.config.band.fixed_atoms.clone_from(&fixed);
            return joined.set_fixed_atoms(fixed);
        }
        let mut candidate = self.config.clone();
        candidate.band.fixed_atoms = fixed;
        let combined = self.positions();
        check_fixed(&candidate.band, combined.view())?;
        self.config = candidate;
        self.pin_fragments();
        let dof = self.left_pos.ncols();
        self.left_fragment =
            FragmentRelaxer::new(&self.config.band, (self.left_pos.nrows() - 1) * dof);
        self.right_fragment =
            FragmentRelaxer::new(&self.config.band, (self.right_pos.nrows() - 1) * dof);
        Ok(())
    }

    fn reparametrization(&self) -> Reparametrization {
        match self.config.band.string {
            Some(cfg) => cfg.reparametrization,
            None => Reparametrization::NotAKnot,
        }
    }

    fn force_norm(&self, force: ArrayView1<'_, f64>) -> f64 {
        match self.config.force_gate {
            ForceGate::LinfNorm => force
                .iter()
                .fold(0.0_f64, |largest, value| largest.max(value.abs())),
            gate => gate.value(force),
        }
    }

    fn max_image_force(&self, force: &Array1<f64>, dof: usize) -> f64 {
        (0..force.len())
            .step_by(dof)
            .map(|start| self.force_norm(force.slice(s![start..start + dof])))
            .fold(0.0_f64, f64::max)
    }

    fn pin_fragments(&mut self) {
        if self.config.band.fixed_atoms.is_empty() {
            return;
        }
        let m_l = self.left_pos.nrows();
        let m_r = self.right_pos.nrows();
        for (atom, _) in self
            .config
            .band
            .fixed_atoms
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
        {
            for c in 3 * atom..3 * atom + 3 {
                let held = self.left_pos[(0, c)];
                for i in 1..m_l {
                    self.left_pos[(i, c)] = held;
                }
                for j in 0..m_r.saturating_sub(1) {
                    self.right_pos[(j, c)] = held;
                }
            }
        }
    }

    fn eval_batch<S: BandSurface>(
        &self,
        surface: &S,
        ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        match &self.pool {
            Some(pool) => pool.eval_with_ids(surface, ids, positions, energies, gradients)?,
            None => surface.eval_with_ids(ids, positions, energies, gradients)?,
        }
        if !energies.iter().all(|e| e.is_finite()) {
            return Err(SaddleError::NonFinite("growing string energies"));
        }
        if !gradients.iter().all(|g| g.is_finite()) {
            return Err(SaddleError::NonFinite("growing string gradients"));
        }
        if !self.config.band.fixed_atoms.is_empty() {
            for mut row in gradients.outer_iter_mut() {
                zero_fixed(&self.config.band.fixed_atoms, &mut row);
            }
        }
        Ok(())
    }

    /// Unwrap `chain` under the minimum-image convention and return
    /// `(unwrapped_chain, cumulative_chord_arcs)`.
    fn unwrap_chain(&self, chain: &Array2<f64>) -> (Array2<f64>, Vec<f64>) {
        let n = chain.nrows();
        let mut unwrapped = chain.clone();
        let mut s = vec![0.0; n];
        for i in 1..n {
            let mut d = &chain.row(i) - &chain.row(i - 1);
            if let Some(cell) = &self.config.band.cell {
                cell.minimum_image(&mut d);
            }
            let next = &unwrapped.row(i - 1) + &d;
            unwrapped.row_mut(i).assign(&next);
            let mut d_free = d;
            zero_fixed(&self.config.band.fixed_atoms, &mut d_free);
            s[i] = s[i - 1] + d_free.dot(&d_free).sqrt();
        }
        (unwrapped, s)
    }

    /// Compute the unit path tangents at every node `1..n-1` of `chain`.
    fn chain_tangents(
        &self,
        chain: &Array2<f64>,
        energies: &[f64],
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        let n = chain.nrows();
        let (unwrapped, s) = self.unwrap_chain(chain);
        let reparam = self.reparametrization();
        let spline = if reparam == Reparametrization::NotAKnot {
            self.fit_chain_spline(&unwrapped, &s)
        } else {
            None
        };

        let mut tangents = Vec::with_capacity(n);
        tangents.push(Array1::zeros(chain.ncols()));
        for i in 1..n - 1 {
            if let Some((_, sp)) = &spline {
                let mut deriv = sp.derivative(s[i]);
                zero_fixed(&self.config.band.fixed_atoms, &mut deriv);
                let norm = deriv.dot(&deriv).sqrt();
                if norm.is_finite() && norm > SAME_ROW_TOL {
                    deriv /= norm;
                    tangents.push(deriv);
                    continue;
                }
            }
            let mut diff_next = &unwrapped.row(i + 1) - &unwrapped.row(i);
            let mut diff_prev = &unwrapped.row(i) - &unwrapped.row(i - 1);
            zero_fixed(&self.config.band.fixed_atoms, &mut diff_next);
            zero_fixed(&self.config.band.fixed_atoms, &mut diff_prev);
            let t = compute_tangent(
                TangentKind::Improved,
                diff_next.view(),
                diff_prev.view(),
                energies[i],
                energies[i - 1],
                energies[i + 1],
            )?;
            tangents.push(t);
        }
        tangents.push(Array1::zeros(chain.ncols()));
        Ok(tangents)
    }

    /// Interpolate a point along the curve through `chain` at fraction
    /// `alpha` in `(0, 1)` of the gap between `left_frontier = m_l - 1`
    /// and `right_frontier = m_l`.
    fn sample_gap(&self, chain: &Array2<f64>, m_l: usize, alpha: f64) -> Array1<f64> {
        let (unwrapped, s) = self.unwrap_chain(chain);
        let s_lf = s[m_l - 1];
        let s_rf = s[m_l];
        let gap = s_rf - s_lf;
        let orig_lf = chain.row(m_l - 1);
        if !(gap.is_finite() && gap > SAME_ROW_TOL) {
            return orig_lf.to_owned();
        }
        if self.reparametrization() == Reparametrization::NotAKnot
            && let Some((kept, spline)) = self.fit_chain_spline(&unwrapped, &s)
        {
            let arcs = spline.knot_arcs();
            let arc_lf = Self::frontier_arc(&s, &kept, &arcs, m_l - 1);
            let arc_rf = Self::frontier_arc(&s, &kept, &arcs, m_l);
            let target_arc = arc_lf + alpha * (arc_rf - arc_lf);
            let t_param = spline.at_arc(&arcs, target_arc);
            let pt = spline.eval(t_param);
            let mut diff = &pt - &unwrapped.row(m_l - 1);
            if let Some(cell) = &self.config.band.cell {
                cell.minimum_image(&mut diff);
            }
            zero_fixed(&self.config.band.fixed_atoms, &mut diff);
            return &orig_lf + &diff;
        }
        let mut chord = &chain.row(m_l) - &orig_lf;
        if let Some(cell) = &self.config.band.cell {
            cell.minimum_image(&mut chord);
        }
        zero_fixed(&self.config.band.fixed_atoms, &mut chord);
        &orig_lf + &(&chord * alpha)
    }

    /// Euclidean distance between the left and right frontier nodes.
    fn frontier_gap(&self) -> f64 {
        let mut gap = &self.right_pos.row(0) - &self.left_pos.row(self.left_pos.nrows() - 1);
        if let Some(cell) = &self.config.band.cell {
            cell.minimum_image(&mut gap);
        }
        zero_fixed(&self.config.band.fixed_atoms, &mut gap);
        gap.dot(&gap).sqrt()
    }

    /// Hand off the joined fragments to [`StringSession`].
    fn hand_off_to_string(&mut self) -> Result<(), SaddleError> {
        let dof = self.left_pos.ncols();
        let reparam = self.reparametrization();
        while self.left_pos.nrows() + self.right_pos.nrows() < self.config.n_images {
            let m_l = self.left_pos.nrows();
            let m_r = self.right_pos.nrows();
            let n_rem = (self.config.n_images + 1 - m_l - m_r) as f64;
            let chain = self.positions();
            let new_node = self.sample_gap(&chain, m_l, 1.0 / n_rem);
            let mut next_left = Array2::zeros((m_l + 1, dof));
            next_left.slice_mut(s![..m_l, ..]).assign(&self.left_pos);
            next_left.row_mut(m_l).assign(&new_node);
            self.left_pos = next_left;
            self.left_ids.push(self.next_id);
            self.next_id += 1;
            reparametrize_fragment(&mut self.left_pos, self.config.band.cell.as_ref(), reparam);
        }
        self.pin_fragments();

        let mut joined_pos = self.positions();
        let joined_ids = self.ids();
        reparametrize(
            &mut joined_pos,
            None,
            self.config.band.cell.as_ref(),
            self.reparametrization(),
        );
        pin_fixed_rows(&self.config.band.fixed_atoms, &mut joined_pos);

        let mut string_session =
            StringSession::with_ids(self.config.band.clone(), joined_ids, joined_pos)?;
        if let Some(endpoints) = self.endpoint_energies {
            string_session.seed_endpoint_energies(endpoints)?;
        }
        if let Some(pool) = &self.pool {
            string_session.set_image_threads(pool.threads())?;
        }
        self.joined = Some(string_session);
        Ok(())
    }

    /// Take one step of the growing string or, once joined, of the
    /// simplified [`StringSession`].
    pub fn step<S: BandSurface>(
        &mut self,
        surface: &S,
    ) -> Result<GrowingStringReport, SaddleError> {
        if let Some(string_session) = &mut self.joined {
            let rep = string_session.step(surface)?;
            self.iteration += 1;
            self.total_surface_rows += rep.surface_rows;
            return Ok(GrowingStringReport {
                status: rep.status,
                phase: GrowingStringPhase::Joined,
                max_force: rep.max_force,
                left_frontier_force: 0.0,
                right_frontier_force: 0.0,
                left_nodes: 0,
                right_nodes: 0,
                n_images: self.config.n_images,
                nodes_added: 0,
                joined_this_step: false,
                ci_index: rep.ci_index,
                iteration: self.iteration,
                surface_rows: rep.surface_rows,
                total_surface_rows: self.total_surface_rows,
                saddle_index: rep.saddle_index,
            });
        }

        let m_l = self.left_pos.nrows();
        let m_r = self.right_pos.nrows();
        let total_active = m_l + m_r;
        let dof = self.left_pos.ncols();
        let chain_before = self.positions();
        let ids_before = self.ids();

        let (energies, gradients, surface_rows) = match self.endpoint_energies {
            None => {
                let mut e = Array1::zeros(total_active);
                let mut g = Array2::zeros((total_active, dof));
                self.eval_batch(surface, &ids_before, chain_before.view(), &mut e, &mut g)?;
                self.endpoint_energies = Some([e[0], e[total_active - 1]]);
                (e.to_vec(), g, total_active)
            }
            Some([e_first, e_last]) => {
                let interior_count = total_active - 2;
                let interior_pos = chain_before.slice(s![1..total_active - 1, ..]);
                let interior_ids = &ids_before[1..total_active - 1];
                let mut int_e = Array1::zeros(interior_count);
                let mut int_g = Array2::zeros((interior_count, dof));
                self.eval_batch(surface, interior_ids, interior_pos, &mut int_e, &mut int_g)?;
                let mut e = vec![0.0; total_active];
                e[0] = e_first;
                e[total_active - 1] = e_last;
                for i in 0..interior_count {
                    e[i + 1] = int_e[i];
                }
                let mut g = Array2::zeros((total_active, dof));
                g.slice_mut(s![1..total_active - 1, ..]).assign(&int_g);
                (e, g, interior_count)
            }
        };
        self.last_energies.clone_from(&energies);
        self.total_surface_rows += surface_rows;

        let tangents = self.chain_tangents(&chain_before, &energies)?;
        let t_lf = tangents[m_l - 1].clone();
        let t_rf = tangents[m_l].clone();
        self.last_frontier_tangents = Some((t_lf.clone(), t_rf.clone()));

        let n_left_mov = m_l - 1;
        let n_right_mov = m_r - 1;
        let mut f_left_perp = Array1::zeros(n_left_mov * dof);
        for (i, t_i) in tangents.iter().enumerate().take(m_l).skip(1) {
            let raw = -&gradients.row(i);
            let mut p = perp(raw.view(), t_i.view());
            zero_fixed(&self.config.band.fixed_atoms, &mut p);
            if self.config.band.drops_translation() {
                remove_net_translation(&mut p);
            }
            f_left_perp.slice_mut(s![(i - 1) * dof..i * dof]).assign(&p);
        }

        let mut f_right_perp = Array1::zeros(n_right_mov * dof);
        for r in 0..n_right_mov {
            let idx = m_l + r;
            let raw = -&gradients.row(idx);
            let mut p = perp(raw.view(), tangents[idx].view());
            zero_fixed(&self.config.band.fixed_atoms, &mut p);
            if self.config.band.drops_translation() {
                remove_net_translation(&mut p);
            }
            f_right_perp
                .slice_mut(s![r * dof..(r + 1) * dof])
                .assign(&p);
        }

        if !f_left_perp.dot(&f_left_perp).is_finite()
            || !f_right_perp.dot(&f_right_perp).is_finite()
        {
            return Err(SaddleError::NonFinite("growing string force norm"));
        }

        let lf_slice = f_left_perp.slice(s![(n_left_mov - 1) * dof..n_left_mov * dof]);
        let rf_slice = f_right_perp.slice(s![..dof]);
        let left_frontier_force = self.force_norm(lf_slice);
        let right_frontier_force = self.force_norm(rf_slice);
        let max_force = self
            .max_image_force(&f_left_perp, dof)
            .max(self.max_image_force(&f_right_perp, dof));

        let reparam = self.reparametrization();
        self.left_fragment
            .update_from_previous(&self.config.band, reparam, &f_left_perp);
        self.right_fragment
            .update_from_previous(&self.config.band, reparam, &f_right_perp);

        let before_left: Array1<f64> = self
            .left_pos
            .slice(s![1..m_l, ..])
            .iter()
            .copied()
            .collect();
        let before_right: Array1<f64> = self
            .right_pos
            .slice(s![..n_right_mov, ..])
            .iter()
            .copied()
            .collect();

        let evol_left =
            self.left_fragment
                .step_fragment(&self.config.band, &before_left, &f_left_perp)?;
        let evol_right =
            self.right_fragment
                .step_fragment(&self.config.band, &before_right, &f_right_perp)?;

        let mut next_left = self.left_pos.clone();
        for i in 1..m_l {
            let mut disp = &evol_left.slice(s![(i - 1) * dof..i * dof]) - &self.left_pos.row(i);
            if let Some(cell) = &self.config.band.cell {
                cell.minimum_image(&mut disp);
            }
            if i == m_l - 1 {
                disp = perp(disp.view(), t_lf.view());
            }
            zero_fixed(&self.config.band.fixed_atoms, &mut disp);
            let updated = &self.left_pos.row(i) + &disp;
            next_left.row_mut(i).assign(&updated);
        }
        reparametrize_fragment(&mut next_left, self.config.band.cell.as_ref(), reparam);

        let mut next_right = self.right_pos.clone();
        for r in 0..n_right_mov {
            let mut disp = &evol_right.slice(s![r * dof..(r + 1) * dof]) - &self.right_pos.row(r);
            if let Some(cell) = &self.config.band.cell {
                cell.minimum_image(&mut disp);
            }
            if r == 0 {
                disp = perp(disp.view(), t_rf.view());
            }
            zero_fixed(&self.config.band.fixed_atoms, &mut disp);
            let updated = &self.right_pos.row(r) + &disp;
            next_right.row_mut(r).assign(&updated);
        }
        reparametrize_fragment(&mut next_right, self.config.band.cell.as_ref(), reparam);

        let mut s_left = Array1::zeros(n_left_mov * dof);
        for i in 1..m_l {
            let mut d = &next_left.row(i) - &self.left_pos.row(i);
            if let Some(cell) = &self.config.band.cell {
                cell.minimum_image(&mut d);
            }
            zero_fixed(&self.config.band.fixed_atoms, &mut d);
            s_left.slice_mut(s![(i - 1) * dof..i * dof]).assign(&d);
        }

        let mut s_right = Array1::zeros(n_right_mov * dof);
        for r in 0..n_right_mov {
            let mut d = &next_right.row(r) - &self.right_pos.row(r);
            if let Some(cell) = &self.config.band.cell {
                cell.minimum_image(&mut d);
            }
            zero_fixed(&self.config.band.fixed_atoms, &mut d);
            s_right.slice_mut(s![r * dof..(r + 1) * dof]).assign(&d);
        }

        self.left_pos = next_left;
        self.right_pos = next_right;
        self.pin_fragments();

        self.left_fragment.prev_step = Some(s_left);
        self.left_fragment.prev_force = Some(f_left_perp);
        self.left_fragment.steps_at_size += 1;

        self.right_fragment.prev_step = Some(s_right);
        self.right_fragment.prev_force = Some(f_right_perp);
        self.right_fragment.steps_at_size += 1;

        let max_steps_per_node = self.max_steps_per_node();
        let left_ready = self.left_fragment.steps_at_size >= self.config.min_steps_per_node
            && (left_frontier_force <= self.config.frontier_tol
                || self.left_fragment.steps_at_size >= max_steps_per_node);
        let right_ready = self.right_fragment.steps_at_size >= self.config.min_steps_per_node
            && (right_frontier_force <= self.config.frontier_tol
                || self.right_fragment.steps_at_size >= max_steps_per_node);

        let mut nodes_added = 0;
        let evolved_chain = self.positions();
        let m_l_now = self.left_pos.nrows();
        let m_r_now = self.right_pos.nrows();
        let n_rem = self.config.n_images + 1 - m_l_now - m_r_now;

        if left_ready && self.left_pos.nrows() + self.right_pos.nrows() < self.config.n_images {
            let alpha = 1.0 / (n_rem as f64);
            let new_left = self.sample_gap(&evolved_chain, m_l_now, alpha);
            let old_l = self.left_pos.nrows();
            let mut grown = Array2::zeros((old_l + 1, dof));
            grown.slice_mut(s![..old_l, ..]).assign(&self.left_pos);
            grown.row_mut(old_l).assign(&new_left);
            self.left_pos = grown;
            self.left_ids.push(self.next_id);
            self.next_id += 1;
            reparametrize_fragment(&mut self.left_pos, self.config.band.cell.as_ref(), reparam);
            self.left_fragment
                .resize(&self.config.band, (self.left_pos.nrows() - 1) * dof);
            nodes_added += 1;
        }

        if right_ready && self.left_pos.nrows() + self.right_pos.nrows() < self.config.n_images {
            let alpha = 1.0 - 1.0 / (n_rem as f64);
            let new_right = self.sample_gap(&evolved_chain, m_l_now, alpha);
            let old_r = self.right_pos.nrows();
            let mut grown = Array2::zeros((old_r + 1, dof));
            grown.row_mut(0).assign(&new_right);
            grown.slice_mut(s![1.., ..]).assign(&self.right_pos);
            self.right_pos = grown;
            self.right_ids.insert(0, self.next_id);
            self.next_id += 1;
            reparametrize_fragment(&mut self.right_pos, self.config.band.cell.as_ref(), reparam);
            self.right_fragment
                .resize(&self.config.band, (self.right_pos.nrows() - 1) * dof);
            nodes_added += 1;
        }

        self.pin_fragments();
        self.iteration += 1;

        let count_reached = self.left_pos.nrows() + self.right_pos.nrows() >= self.config.n_images
            && (nodes_added > 0 || (left_ready && right_ready));
        let distance_reached = self
            .config
            .join_tol
            .is_some_and(|tol| self.frontier_gap() <= tol);

        let mut joined_this_step = false;
        if count_reached || distance_reached {
            self.hand_off_to_string()?;
            joined_this_step = true;
        }

        let phase = self.phase();
        let (left_nodes, right_nodes, n_images) = match phase {
            GrowingStringPhase::Growing => (
                self.left_pos.nrows(),
                self.right_pos.nrows(),
                self.left_pos.nrows() + self.right_pos.nrows(),
            ),
            GrowingStringPhase::Joined => (0, 0, self.config.n_images),
        };

        Ok(GrowingStringReport {
            status: BandStatus::Running,
            phase,
            max_force,
            left_frontier_force,
            right_frontier_force,
            left_nodes,
            right_nodes,
            n_images,
            nodes_added,
            joined_this_step,
            ci_index: None,
            iteration: self.iteration,
            surface_rows,
            total_surface_rows: self.total_surface_rows,
            saddle_index: None,
        })
    }

    /// Loop [`GrowingStringSession::step`] until the fragments join and
    /// the resulting [`StringSession`] converges, or until `max_steps`.
    pub fn run<S: BandSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<GrowingStringReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == BandStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}

fn reparametrize_fragment(
    fragment: &mut Array2<f64>,
    cell: Option<&Cell>,
    scheme: Reparametrization,
) {
    if fragment.nrows() < 3 {
        return;
    }
    reparametrize(fragment, None, cell, scheme);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct ConstantPush;

    impl BandSurface for ConstantPush {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, r) in positions.outer_iter().enumerate() {
                energies[i] = r[1];
                gradients[(i, 1)] = 1.0;
            }
            Ok(())
        }
    }

    #[test]
    fn repeated_nodes_do_not_break_tangent_and_gap_sampling() {
        let session = GrowingStringSession::new(
            GrowingStringConfig::default().with_n_images(7),
            array![-1.0, 0.0, 0.0],
            array![1.0, 0.0, 0.0],
        )
        .expect("session");
        let chain = array![
            [-1.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let energies = vec![0.0; chain.nrows()];
        let tangents = session
            .chain_tangents(&chain, &energies)
            .expect("tangents with repeated nodes");
        assert_eq!(tangents.len(), chain.nrows());
        for tangent in tangents.iter().skip(1).take(chain.nrows() - 2) {
            assert!(tangent.iter().all(|v| v.is_finite()));
        }
        let sampled = session.sample_gap(&chain, 3, 0.5);
        assert!(sampled.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn default_max_steps_per_node_forces_progress() {
        let config = GrowingStringConfig::default()
            .with_n_images(9)
            .with_frontier_tol(f64::MIN_POSITIVE)
            .with_remove_translation(false);
        let mut session =
            GrowingStringSession::new(config, array![-1.0, 0.0, 0.0], array![1.0, 0.0, 0.0])
                .expect("session");
        let cap = session.max_steps_per_node();
        for _ in 0..cap - 1 {
            let report = session.step(&ConstantPush).expect("step before cap");
            assert_eq!(report.nodes_added, 0);
        }
        let report = session.step(&ConstantPush).expect("step at cap");
        assert!(
            report.nodes_added > 0,
            "automatic cap must force growth when tolerance is never met"
        );
    }
}
