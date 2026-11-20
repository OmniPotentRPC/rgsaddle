//! Free-end adaptive nudged elastic band (FEA-NEB).
//!
//! Climbing on the final cut level reuses the climbing image force, and
//! when minimum-mode refinement is left on, the off-path climb of
//! Goswami, Gunde, and Jónsson, Front. Chem. 14 (2026),
//! doi:10.3389/fchem.2026.1807063. The endpoint force and the adaptive
//! window are those of Jiayong Zhang, Hongwu Zhang, Hongfei Ye, and
//! Yonggang Zheng, J. Chem. Phys. 145, 094104 (2016),
//! doi:10.1063/1.4962019, building on the free-end level-set endpoint
//! formulation of Ting Zhu, Ju Li, Amit Samanta, Hyoung Gyu Kim, and
//! Subra Suresh, Proc. Natl. Acad. Sci. U.S.A. 104, 3031 (2007),
//! doi:10.1073/pnas.0611097104. A nonzero endpoint gradient is required
//! and the endpoint residual is reported.
//!
//! One terminal image (`0` for [`FreeEndSide::First`] or `n_images - 1`
//! for [`FreeEndSide::Last`], the default) moves subject to the energy
//! level-set constraint \(E(x_{\mathrm{end}}) = E_{\mathrm{target}}\)
//! while the opposite terminal image is pinned:
//! 1. The free terminal image requires a finite, nonzero potential
//!    gradient \(\nabla E(x_{\mathrm{end}}) \ne 0\).
//! 2. Its path tangent \(\hat{\tau}_{\mathrm{end}}\) is the unit vector
//!    along the single adjacent band segment (`x_1 - x_0` for the first
//!    image, `x_{N-1} - x_{N-2}` for the last image). Zhang, Zhang, Ye,
//!    and Zheng (2016), Eq. (7), is only the perpendicular potential
//!    force
//!    \(F^p_{\mathrm{end},\perp} = -\nabla E(x_{\mathrm{end}}) +
//!    (\nabla E(x_{\mathrm{end}}) \cdot \hat{\tau}_{\mathrm{end}})
//!    \hat{\tau}_{\mathrm{end}}\).
//!    The endpoint force does not include a spring, so it does not
//!    depend on the spring constant.
//! 3. Net translation is removed, when the band drops it, before the
//!    force is projected onto \(\nabla E(x_{\mathrm{end}})^\perp\), so
//!    the force that steps the endpoint stays orthogonal to the
//!    gradient. Newton retraction onto
//!    \(E(x_{\mathrm{end}}) = E_{\mathrm{target}}\) is delegated to
//!    [`rgmin::Solver`] configured with [`rgmin::LevelSetConstraint`].
//! 4. When `max_cuts > 0`, once the highest-energy interior image holds
//!    its index for `stability` steps and the band force falls under the
//!    level's cut threshold while the peak rises above its neighbours by
//!    more than `energy_tol`, images outside the local window around
//!    that peak are dropped and all `n_images` images are remeshed at
//!    equal arc length (cubic Hermite or linear). The window keeps
//!    `window_offset` images on each side of the peak, the same local
//!    cut Zhang, Zhang, Ye, and Zheng use (one image on each side in
//!    their driver). A cut that would place the peak on an end, or that
//!    would remove no image, is not counted. Images outside the window
//!    are dropped. The remesh follows the window and, when the free
//!    terminal image lies outside it, the leg from the window to that
//!    image, so the free end stays an endpoint and the spacing stays
//!    even. A refused endpoint curvature pair is a stalled step and
//!    does not move the endpoint.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::{Accept, Control, LevelSetConstraint, Method, Oracle, Solver};

use super::zoom::resample;
use super::{
    BandConfig, BandReport, BandSession, BandStatus, BandSurface, SAME_ROW_TOL, build_solver,
    climb_target, image_force, max_abs, min_mode, pin_fixed_rows, remove_net_translation,
    zero_fixed,
};
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::saddle_index::SaddleIndex;
use crate::spring::SpringKind;

/// Which terminal image of the band moves on the target-energy level set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FreeEndSide {
    /// First image (`0`), coupled to interior neighbour `1`.
    First,
    /// Last image (`n_images - 1`), coupled to interior neighbour `n_images - 2`.
    #[default]
    Last,
}

impl FreeEndSide {
    /// Discriminant used by the C and Python interfaces (`0` for `First`, `1` for `Last`).
    #[must_use]
    pub fn to_abi(self) -> i32 {
        match self {
            Self::First => 0,
            Self::Last => 1,
        }
    }

    /// Inverse of [`Self::to_abi`].
    #[must_use]
    pub fn from_abi(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::First),
            1 => Some(Self::Last),
            _ => None,
        }
    }
}

/// Free-end adaptive NEB configuration on [`BandConfig::free_end`].
///
/// Jiayong Zhang, Hongwu Zhang, Hongfei Ye, and Yonggang Zheng, J. Chem.
/// Phys. 145, 094104 (2016), doi:10.1063/1.4962019; Ting Zhu, Ju Li,
/// Amit Samanta, Hyoung Gyu Kim, and Subra Suresh, Proc. Natl. Acad.
/// Sci. U.S.A. 104, 3031 (2007), doi:10.1073/pnas.0611097104.
///
/// Off (`None`) by default on [`BandConfig`], keeping fixed endpoints as
/// the default.
///
/// Non-exhaustive: start from [`FreeEndConfig::default`] and chain the
/// `with_*` setters.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2, ArrayView2};
/// use rgsaddle::{
///     BandConfig, BandStatus, BandSurface, CiConfig, DEFAULT_FORCE_TOL, FeaNebSession,
///     FreeEndConfig, FreeEndSide, SaddleError, linear_path,
/// };
///
/// struct TiltedBarrier;
///
/// impl BandSurface for TiltedBarrier {
///     fn eval(
///         &self,
///         positions: ArrayView2<f64>,
///         energies: &mut Array1<f64>,
///         gradients: &mut Array2<f64>,
///     ) -> Result<(), SaddleError> {
///         for (i, r) in positions.outer_iter().enumerate() {
///             let (x, y, z) = (r[0], r[1], r[2]);
///             let x2 = x * x;
///             let x4 = x2 * x2;
///             let g1 = (-1.5 * x2).exp();
///             let dx = x - 0.75;
///             let g2 = (-16.0 * dx * dx).exp();
///             let sw = x4 / (1.0 + x4);
///             let dsw = 4.0 * x * x2 / ((1.0 + x4) * (1.0 + x4));
///             let tilt = 0.12 * (x - 6.0).powi(2) - 3.0;
///             let dtilt = 0.24 * (x - 6.0);
///             let v_path = 2.0 * x2 * g1 + 1.6 * x2 * g2 + sw * tilt;
///             let dv_path = 2.0 * x * g1 * (2.0 - 3.0 * x2)
///                 + 1.6 * x * g2 * (2.0 - 32.0 * x * dx)
///                 + dsw * tilt
///                 + sw * dtilt;
///             let valley = y - 0.25 * x;
///             energies[i] = v_path + 4.0 * valley * valley + 2.0 * z * z;
///             gradients[(i, 0)] = dv_path - 2.0 * valley;
///             gradients[(i, 1)] = 8.0 * valley;
///             gradients[(i, 2)] = 4.0 * z;
///         }
///         Ok(())
///     }
/// }
///
/// let start = Array1::from(vec![0.0, 0.0, 0.0]);
/// let far_end = Array1::from(vec![2.0, 0.5, 0.0]);
/// let initial = linear_path(start.view(), far_end.view(), 7, None)?;
/// let free_end = FreeEndConfig::default()
///     .with_side(FreeEndSide::Last)
///     .with_target_energy(Some(-0.2))
///     .with_max_cuts(2);
/// let config = BandConfig::default()
///     .with_climbing(Some(CiConfig::after_rel(0.5)))
///     .with_min_mode(None)
///     .with_saddle_check(None)
///     .with_force_tol(DEFAULT_FORCE_TOL)
///     .with_max_move(0.08);
/// let mut session = FeaNebSession::new(config, free_end, initial)?;
/// let report = session.run(&TiltedBarrier, 600)?;
/// assert_eq!(report.status, BandStatus::Converged);
/// assert!(report.target_energy_residual.abs() < 0.001);
/// # Ok::<(), SaddleError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct FreeEndConfig {
    /// Which terminal image moves on the target-energy level set (`Last` by default).
    pub side: FreeEndSide,
    /// Target energy \(E_{\mathrm{target}}\) for the free terminal image.
    /// `None` uses the initial energy of that terminal image on the first evaluation.
    pub target_energy: Option<f64>,
    /// Maximum number of adaptive cut-and-remesh refinements around the
    /// detected saddle (`2` by default; `0` disables cutting).
    pub max_cuts: usize,
    /// Force threshold on the first cut level (`0.0` uses a geometric
    /// schedule ending at `4.0 * force_tol`).
    pub cut_force: f64,
    /// Minimum peak-to-neighbour energy difference
    /// `E[peak] - min(E[peak - 1], E[peak + 1])` required to cut
    /// (`etol` in Zhang, Zhang, Ye, and Zheng 2016; `0.001` by default).
    pub energy_tol: f64,
    /// Consecutive steps the peak image must hold its index before cutting (`2` by default).
    pub stability: usize,
    /// Half-width in images around the detected peak kept when cutting (`1` by default).
    pub window_offset: usize,
    /// Cubic Hermite remeshing when `true` (`true` by default); polyline when `false`.
    pub cubic: bool,
}

with_setters!(FreeEndConfig {
    with_side => side: val FreeEndSide,
    with_target_energy => target_energy: opt f64,
    with_max_cuts => max_cuts: val usize,
    with_cut_force => cut_force: val f64,
    with_energy_tol => energy_tol: val f64,
    with_stability => stability: val usize,
    with_window_offset => window_offset: val usize,
    with_cubic => cubic: val bool,
});

impl Default for FreeEndConfig {
    fn default() -> Self {
        Self {
            side: FreeEndSide::Last,
            target_energy: None,
            max_cuts: 2,
            cut_force: 0.0,
            energy_tol: 1.0 / 1000.0,
            stability: 2,
            window_offset: 1,
            cubic: true,
        }
    }
}

pub(super) fn check_free_end(
    config: &BandConfig,
    free_end: &FreeEndConfig,
) -> Result<(), SaddleError> {
    if config.solid.is_some() {
        return Err(SaddleError::Invalid(
            "free-end adaptive NEB does not carry a solid-state cell block".into(),
        ));
    }
    if config.string.is_some() {
        return Err(SaddleError::Invalid(
            "free-end adaptive NEB cannot be combined with the simplified string method".into(),
        ));
    }
    if config.dynamic.is_some() {
        return Err(SaddleError::Invalid(
            "free-end adaptive NEB cannot be combined with dynamic NEB; \
             the free terminal image and the level-set retraction need every image evaluated"
                .into(),
        ));
    }
    if !matches!(config.spring, SpringKind::Uniform { .. }) {
        return Err(SaddleError::Invalid(
            "free-end adaptive NEB requires a uniform spring".into(),
        ));
    }
    if let Some(target) = free_end.target_energy
        && !target.is_finite()
    {
        return Err(SaddleError::NonFinite("free-end target energy"));
    }
    if !(free_end.cut_force.is_finite() && free_end.cut_force >= 0.0) {
        return Err(SaddleError::Invalid(
            "free-end cut_force must be finite and non-negative".into(),
        ));
    }
    if !(free_end.energy_tol.is_finite() && free_end.energy_tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "free-end energy_tol must be finite and non-negative".into(),
        ));
    }
    if free_end.max_cuts > 0 && (free_end.stability == 0 || free_end.window_offset == 0) {
        return Err(SaddleError::Invalid(
            "free-end cut-and-remesh requires stability >= 1 and window_offset >= 1".into(),
        ));
    }
    Ok(())
}

/// Diagnostics reported after one [`FeaNebSession::step`] or [`BandSession::step_free_end`].
#[derive(Clone, Debug)]
pub struct FreeEndReport {
    /// Whether the band is still running or has converged.
    pub status: BandStatus,
    /// Largest convergence force norm across the interior images and the free endpoint.
    pub max_force: f64,
    /// Convergence force norm on the free terminal image (projected onto the energy level set).
    pub endpoint_force: f64,
    /// Target energy `E_target` of the free terminal image.
    pub target_energy: f64,
    /// Potential energy `E(x_end)` at the accepted free terminal image.
    pub endpoint_energy: f64,
    /// Signed target-energy residual `E(x_end) - E_target` after level-set retraction.
    pub target_energy_residual: f64,
    /// Total adaptive cut-and-remesh refinements performed so far.
    pub cuts: usize,
    /// Whether an adaptive cut-and-remesh triggered on this step.
    pub cut_this_step: bool,
    /// Climbing-image index when armed, if any.
    pub ci_index: Option<usize>,
    /// Completed steps in this session.
    pub iteration: usize,
    /// Surface image rows evaluated during this step.
    pub surface_rows: usize,
    /// Certified saddle index of the climbing image at convergence, if enabled.
    pub saddle_index: Option<SaddleIndex>,
}

/// Alias for [`FreeEndReport`].
pub type FeaNebReport = FreeEndReport;

struct FreeEndMeasurement {
    energies: Array1<f64>,
    gradients: Array2<f64>,
    projected: Array1<f64>,
    max_force: f64,
    endpoint_force: f64,
}

#[derive(Clone, Debug, Default)]
pub(super) struct FreeEndState {
    pub(super) resolved_target: Option<f64>,
    pub(super) fixed_endpoint_energy: Option<f64>,
    pub(super) initial_fixed_energy: Option<f64>,
    pub(super) last_residual: Option<f64>,
    pub(super) last_endpoint_energy: Option<f64>,
    pub(super) last_endpoint_force: Option<f64>,
    pub(super) prev_peak: Option<usize>,
    pub(super) stable: usize,
    pub(super) cuts: usize,
    pub(super) cut_windows: Vec<(usize, usize)>,
    pub(super) cut_baseline_force: Option<f64>,
    pub(super) endpoint_dt: Option<f64>,
    pub(super) prev_end_pos: Option<Array1<f64>>,
    pub(super) prev_end_force: Option<Array1<f64>>,
}

fn lock_guard<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Keep the curvature pair that makes the next level-set step the
/// tangent projection of the endpoint direction. A pair `push_pair`
/// drops, including one it reports as stored while `pair_count` stays
/// at zero, is a stalled step.
fn accept_endpoint_pair(
    solver: &mut Solver,
    step: ArrayView1<f64>,
    slope: ArrayView1<f64>,
) -> Result<(), SaddleError> {
    if solver.push_pair(step, slope) && solver.pair_count() > 0 {
        return Ok(());
    }
    Err(SaddleError::Stalled(
        "free-end endpoint step skipped: the curvature pair was refused".into(),
    ))
}

fn cut_force_limit(first_cut_tol: f64, final_cut_tol: f64, cut: usize, max_cuts: usize) -> f64 {
    let first_cut_tol = first_cut_tol.max(final_cut_tol);
    if max_cuts <= 1 {
        first_cut_tol
    } else {
        let frac = cut as f64 / (max_cuts - 1) as f64;
        first_cut_tol * (final_cut_tol / first_cut_tol).powf(frac)
    }
}

fn first_cut_tolerance(cut_force: f64, baseline_force: Option<f64>, final_cut_tol: f64) -> f64 {
    let requested = if cut_force > 0.0 {
        cut_force
    } else {
        baseline_force
            .map(|baseline| baseline * 0.5)
            .unwrap_or(final_cut_tol * 4.0)
            .max(final_cut_tol * 2.0)
    };
    requested.max(final_cut_tol)
}

/// Project `force` exactly onto the tangent hyperplane orthogonal to
/// `grad_end`.
fn project_onto_level_set(
    grad_end: &Array1<f64>,
    force: &Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let gn2 = grad_end.dot(grad_end);
    if !(gn2.is_finite() && gn2 > 0.0) {
        return Err(SaddleError::Invalid(
            "free-end terminal image requires a nonzero potential gradient".into(),
        ));
    }
    if !force.iter().all(|component| component.is_finite()) {
        return Err(SaddleError::NonFinite("free-end force"));
    }
    let parallel_scale = force.dot(grad_end) / gn2;
    Ok(force - &(grad_end * parallel_scale))
}

/// Step `x_end` by `step_dir` (projected onto the level-set tangent space
/// and capped per atom by `max_move`) and retract the accepted point onto
/// `E(x_end) = target_energy` using `rgmin::Solver` with
/// `LevelSetConstraint::new(target_energy)`.
#[allow(clippy::too_many_arguments)]
fn step_and_retract_endpoint_via_rgmin<S: BandSurface>(
    config: &BandConfig,
    surface: &S,
    rows_counter: &std::sync::atomic::AtomicUsize,
    end_id: Option<crate::stencil::ImageId>,
    target_energy: f64,
    x_end: &mut Array1<f64>,
    energy_end: &mut f64,
    grad_end: &mut Array1<f64>,
    step_dir: Option<&Array1<f64>>,
) -> Result<(), SaddleError> {
    let dof = x_end.len();
    let mut check_grad = grad_end.clone();
    zero_fixed(&config.fixed_atoms, &mut check_grad);
    let gn2 = check_grad.dot(&check_grad);
    if !(gn2.is_finite() && gn2 > 0.0) {
        return Err(SaddleError::Invalid(
            "free-end terminal image requires a nonzero potential gradient".into(),
        ));
    }

    let control = Control::default()
        .with_maxiter(usize::MAX)
        .with_gtol(0.0)
        .with_istep(1.0)
        .with_maxmove(None);
    let mut solver = Solver::new(Method::lbfgs_with_memory(1), control, dof);
    solver.set_accept(Accept::Step);
    solver.set_lbfgs_neb_guards(false);
    solver.set_atom_maxmove(config.max_move);
    solver
        .set_level_set(LevelSetConstraint::new(target_energy))
        .map_err(|e| SaddleError::Solver(e.to_string()))?;

    let cached_pos = Mutex::new(x_end.clone());
    let cached_e = Mutex::new(*energy_end);
    let cached_g = Mutex::new(check_grad.clone());
    let cached_raw_g = Mutex::new(grad_end.clone());
    let surface_err: Mutex<Option<SaddleError>> = Mutex::new(None);

    let eval_point = |xv: ArrayView1<f64>| -> (f64, Array1<f64>) {
        {
            let c_pos = lock_guard(&cached_pos);
            if xv == c_pos.view() {
                return (*lock_guard(&cached_e), lock_guard(&cached_g).clone());
            }
        }
        let mut row = Array2::zeros((1, dof));
        row.row_mut(0).assign(&xv);
        pin_fixed_rows(&config.fixed_atoms, &mut row);
        let mut e_buf = Array1::zeros(1);
        let mut g_buf = Array2::zeros((1, dof));
        rows_counter.fetch_add(1, Ordering::Relaxed);
        let res = match end_id {
            Some(id) => surface.eval_with_ids(&[id], row.view(), &mut e_buf, &mut g_buf),
            None => surface.eval(row.view(), &mut e_buf, &mut g_buf),
        };
        match res {
            Ok(()) => {
                let e_val = e_buf[0];
                let raw_g = g_buf.row(0).to_owned();
                let mut g_val = raw_g.clone();
                zero_fixed(&config.fixed_atoms, &mut g_val);
                if !e_val.is_finite() || !g_val.iter().all(|v| v.is_finite()) {
                    let mut slot = lock_guard(&surface_err);
                    if slot.is_none() {
                        *slot = Some(SaddleError::NonFinite("free-end surface evaluation"));
                    }
                    return (f64::INFINITY, Array1::zeros(dof));
                }
                let gn = g_val.dot(&g_val);
                if !(gn.is_finite() && gn > 0.0) {
                    let mut slot = lock_guard(&surface_err);
                    if slot.is_none() {
                        *slot = Some(SaddleError::Invalid(
                            "free-end terminal image requires a nonzero potential gradient".into(),
                        ));
                    }
                    return (e_val, Array1::zeros(dof));
                }
                *lock_guard(&cached_pos) = xv.to_owned();
                *lock_guard(&cached_e) = e_val;
                *lock_guard(&cached_g) = g_val.clone();
                *lock_guard(&cached_raw_g) = raw_g;
                (e_val, g_val)
            }
            Err(err) => {
                let mut slot = lock_guard(&surface_err);
                if slot.is_none() {
                    *slot = Some(err);
                }
                (f64::INFINITY, Array1::zeros(dof))
            }
        }
    };

    // First ensure the starting endpoint lies on E(x_end) = target_energy.
    {
        let oracle = Oracle::unbounded(dof, &eval_point);
        let res = solver.step(&oracle, x_end);
        drop(oracle);
        if let Some(err) = lock_guard(&surface_err).take() {
            return Err(err);
        }
        res.map_err(|e| SaddleError::Solver(e.to_string()))?;
    }

    if let Some(dir) = step_dir {
        let dir_norm = dir.dot(dir).sqrt();
        if dir_norm.is_finite() && dir_norm > 0.0 {
            let g_now = lock_guard(&cached_g).clone();
            let gn2_now = g_now.dot(&g_now);
            if !(gn2_now.is_finite() && gn2_now > 0.0) {
                return Err(SaddleError::Invalid(
                    "free-end terminal image requires a nonzero potential gradient".into(),
                ));
            }
            let gn_now = gn2_now.sqrt();
            let k = (dir.dot(&g_now) / gn2_now) + (dir_norm + 1.0) / gn_now;
            let s_pair = &g_now * k - dir;
            let y_pair = g_now;
            solver.forget();
            accept_endpoint_pair(&mut solver, s_pair.view(), y_pair.view())?;
            let oracle = Oracle::unbounded(dof, &eval_point);
            let res = solver.step(&oracle, x_end);
            drop(oracle);
            if let Some(err) = lock_guard(&surface_err).take() {
                return Err(err);
            }
            res.map_err(|e| SaddleError::Solver(e.to_string()))?;
        }
    }

    *energy_end = *lock_guard(&cached_e);
    *grad_end = lock_guard(&cached_raw_g).clone();
    Ok(())
}

/// Zhang, Zhang, Ye, and Zheng (2016), Eq. (7): the potential force
/// perpendicular to the free-end tangent. The spring constant does not
/// enter.
fn eq7_endpoint_force(
    config: &BandConfig,
    side: FreeEndSide,
    pos_end: ArrayView1<f64>,
    pos_neigh: ArrayView1<f64>,
    grad_end: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    if !matches!(config.spring, SpringKind::Uniform { .. }) {
        return Err(SaddleError::Invalid(
            "free-end adaptive NEB requires a uniform spring".into(),
        ));
    }
    let mut inward = &pos_neigh - &pos_end;
    if let Some(cell) = &config.cell {
        cell.minimum_image(&mut inward);
    }
    zero_fixed(&config.fixed_atoms, &mut inward);
    let dist2 = inward.dot(&inward);
    let dist = dist2.sqrt();
    if !(dist.is_finite() && dist > SAME_ROW_TOL) {
        return Err(SaddleError::Invalid(
            "free-end terminal image coincides with its interior neighbour".into(),
        ));
    }
    let tangent = match side {
        FreeEndSide::First => &inward / dist,
        FreeEndSide::Last => -&inward / dist,
    };
    let mut potential = -&grad_end;
    zero_fixed(&config.fixed_atoms, &mut potential);
    let potential_parallel = potential.dot(&tangent);
    let mut perpendicular = potential - &tangent * potential_parallel;
    zero_fixed(&config.fixed_atoms, &mut perpendicular);
    Ok(perpendicular)
}

/// Eq. (7), then net translation removal, then the level-set projection.
/// Translation is removed first so the force that is returned stays
/// orthogonal to `grad_end`.
fn assembled_free_end_force(
    config: &BandConfig,
    side: FreeEndSide,
    pos_end: ArrayView1<f64>,
    pos_neigh: ArrayView1<f64>,
    grad_end: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let mut raw = eq7_endpoint_force(config, side, pos_end, pos_neigh, grad_end)?;
    if config.drops_translation() {
        remove_net_translation(&mut raw);
    }
    let mut projected = project_onto_level_set(&grad_end.to_owned(), &raw)?;
    zero_fixed(&config.fixed_atoms, &mut projected);
    Ok(projected)
}

/// Images kept around an interior peak. `None` when the peak would be a
/// window endpoint or when the window is already the whole band.
fn peak_window(n_images: usize, max_i: usize, offset: usize) -> Option<(usize, usize)> {
    if offset == 0 || max_i == 0 || max_i + 1 >= n_images {
        return None;
    }
    let lo = max_i.saturating_sub(offset);
    let hi = max_i.saturating_add(offset).min(n_images - 1);
    if lo >= max_i || hi <= max_i {
        return None;
    }
    if lo == 0 && hi + 1 == n_images {
        return None;
    }
    Some((lo, hi))
}

/// Window rows for a cut, with the free end kept when the window does
/// not already contain it. `resample` then spaces that polyline evenly
/// and leaves both of its ends in place.
fn cut_path(positions: ArrayView2<f64>, lo: usize, hi: usize, end_idx: usize) -> Array2<f64> {
    let window = positions.slice(s![lo..=hi, ..]);
    if end_idx == lo || end_idx == hi {
        return window.to_owned();
    }
    let span = hi - lo + 1;
    let mut rows = Array2::zeros((span + 1, positions.ncols()));
    if end_idx > hi {
        rows.slice_mut(s![..span, ..]).assign(&window);
        rows.row_mut(span).assign(&positions.row(end_idx));
    } else {
        rows.row_mut(0).assign(&positions.row(end_idx));
        rows.slice_mut(s![1.., ..]).assign(&window);
    }
    rows
}

impl BandSession {
    /// Install, replace, or clear the free-end adaptive NEB configuration
    /// ([`BandConfig::free_end`]). Optimizer history and cut counters
    /// restart.
    pub fn set_free_end(&mut self, free_end: Option<FreeEndConfig>) -> Result<(), SaddleError> {
        if let Some(fe) = &free_end {
            check_free_end(&self.config, fe)?;
        }
        self.config.free_end = free_end;
        self.free_end_state = FreeEndState::default();
        self.restart();
        Ok(())
    }

    /// Free-end adaptive NEB configuration, if enabled (`None` by default).
    #[must_use]
    pub fn free_end(&self) -> Option<FreeEndConfig> {
        self.config.free_end
    }

    /// Signed target-energy residual `E(x_end) - E_target` from the most
    /// recent free-end step, or `None` when endpoints are fixed or before
    /// the first step.
    #[must_use]
    pub fn target_energy_residual(&self) -> Option<f64> {
        self.free_end_state.last_residual
    }

    /// Number of adaptive cut-and-remesh refinements performed so far.
    #[must_use]
    pub fn free_end_cuts(&self) -> usize {
        self.free_end_state.cuts
    }

    /// Window index pairs `(lo, hi)` of each adaptive cut-and-remesh
    /// performed so far.
    #[must_use]
    pub fn free_end_windows(&self) -> &[(usize, usize)] {
        &self.free_end_state.cut_windows
    }

    /// Initial energy of the fixed endpoint before any adaptive cuts.
    #[must_use]
    pub fn initial_fixed_endpoint_energy(&self) -> Option<f64> {
        self.free_end_state.initial_fixed_energy
    }

    /// Energies, gradients, and projected forces at the positions the
    /// step is about to report.
    fn measure_free_end<S: BandSurface>(
        &mut self,
        surface: &S,
        side: FreeEndSide,
        end_idx: usize,
        neigh_idx: usize,
        ci_at: Option<usize>,
    ) -> Result<FreeEndMeasurement, SaddleError> {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let mut energies = Array1::zeros(n_images);
        let mut gradients = Array2::zeros((n_images, dof));
        self.state.eval_rows(
            surface,
            0..n_images,
            self.positions.view(),
            &mut energies,
            &mut gradients,
        )?;
        self.state.rows.fetch_add(n_images, Ordering::Relaxed);
        if !energies.iter().all(|energy| energy.is_finite()) {
            return Err(SaddleError::NonFinite("band energies"));
        }
        if !gradients.iter().all(|component| component.is_finite()) {
            return Err(SaddleError::NonFinite("band gradients"));
        }
        if !self.state.fixed.is_empty() {
            for mut row in gradients.outer_iter_mut() {
                zero_fixed(&self.state.fixed, &mut row);
            }
        }
        let mut grad_end = gradients.row(end_idx).to_owned();
        zero_fixed(&self.config.fixed_atoms, &mut grad_end);
        let endpoint = assembled_free_end_force(
            &self.config,
            side,
            self.positions.row(end_idx),
            self.positions.row(neigh_idx),
            grad_end.view(),
        )?;
        let endpoint_force = match self.state.force_gate {
            ForceGate::LinfNorm => max_abs(&endpoint),
            gate => gate.value(endpoint.view()),
        };
        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let interior = self.interior_flat();
        let resolved_spring = self
            .config
            .spring
            .at_band(energies.view(), gradients.view());
        let metrics = self
            .exp
            .as_ref()
            .and_then(|exp| exp.fresh_metrics(interior.view(), n_images - 2));
        let mut projected = Array1::zeros((n_images - 2) * dof);
        for image in 1..n_images - 1 {
            let rows = s![(image - 1) * dof..image * dof];
            let metric =
                metrics.and_then(|band| band.per_image.get(image - 1).map(|p| (p, band.free)));
            let (force, _) = image_force(
                &self.config,
                &self.state,
                &resolved_spring,
                image,
                ci_at == Some(image),
                endpoint_first.view(),
                endpoint_last.view(),
                interior.view(),
                &energies,
                &gradients,
                metric,
            )?;
            projected.slice_mut(rows).assign(&force);
        }
        let max_force = self.state.convergence_force(&projected).max(endpoint_force);
        Ok(FreeEndMeasurement {
            energies,
            gradients,
            projected,
            max_force,
            endpoint_force,
        })
    }

    /// Advance a free-end adaptive NEB session by one step and return the
    /// detailed [`FreeEndReport`] including `target_energy_residual`.
    /// When climbing is on, a latched minimum-mode refinement runs first,
    /// the same way [`BandSession::step`] does.
    pub fn step_free_end<S: BandSurface>(
        &mut self,
        surface: &S,
    ) -> Result<FreeEndReport, SaddleError> {
        let Some(fe_cfg) = self.config.free_end else {
            return Err(SaddleError::Invalid(
                "BandSession::step_free_end requires BandConfig::free_end to be set".into(),
            ));
        };
        if let Some(mut band) = self.climbing_refine_step(surface)? {
            if band.status == BandStatus::Converged {
                self.saddle_check(surface, &mut band)?;
            }
            return Ok(self.free_end_report_from_band(band));
        }
        let mut rep = self.step_free_end_inner(surface, fe_cfg)?;
        if rep.status == BandStatus::Converged {
            let mut band_rep = BandReport {
                status: rep.status,
                max_force: rep.max_force,
                ci_index: rep.ci_index,
                iteration: rep.iteration,
                surface_rows: rep.surface_rows,
                curvature: None,
                saddle_index: None,
            };
            self.saddle_check(surface, &mut band_rep)?;
            rep.status = band_rep.status;
            rep.surface_rows = band_rep.surface_rows;
            rep.saddle_index = band_rep.saddle_index;
        }
        Ok(rep)
    }

    /// Loop [`BandSession::step_free_end`] until convergence or `max_steps`.
    pub fn run_free_end<S: BandSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<FreeEndReport, SaddleError> {
        let mut report = self.step_free_end(surface)?;
        while report.status == BandStatus::Running && self.iteration < max_steps {
            report = self.step_free_end(surface)?;
        }
        Ok(report)
    }

    pub(super) fn step_free_end_inner<S: BandSurface>(
        &mut self,
        surface: &S,
        fe_cfg: FreeEndConfig,
    ) -> Result<FreeEndReport, SaddleError> {
        check_free_end(&self.config, &fe_cfg)?;
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let interior_dof = (n_images - 2) * dof;
        let (end_idx, neigh_idx, fixed_idx) = match fe_cfg.side {
            FreeEndSide::First => (0, 1, n_images - 1),
            FreeEndSide::Last => (n_images - 1, n_images - 2, 0),
        };

        self.state.take_error();
        self.state.rows.store(0, Ordering::Relaxed);
        self.refresh_exp(surface)?;

        let mut energies = Array1::zeros(n_images);
        let mut gradients = Array2::zeros((n_images, dof));
        match self.free_end_state.fixed_endpoint_energy {
            None => {
                self.state.eval_rows(
                    surface,
                    0..n_images,
                    self.positions.view(),
                    &mut energies,
                    &mut gradients,
                )?;
                self.state.rows.fetch_add(n_images, Ordering::Relaxed);
                self.free_end_state.fixed_endpoint_energy = Some(energies[fixed_idx]);
                if self.free_end_state.initial_fixed_energy.is_none() {
                    self.free_end_state.initial_fixed_energy = Some(energies[fixed_idx]);
                }
            }
            Some(e_fixed) => {
                let range = match fe_cfg.side {
                    FreeEndSide::First => 0..n_images - 1,
                    FreeEndSide::Last => 1..n_images,
                };
                let count = range.len();
                let mut sub_e = Array1::zeros(count);
                let mut sub_g = Array2::zeros((count, dof));
                let view = self.positions.slice(s![range.clone(), ..]);
                self.state
                    .eval_rows(surface, range.clone(), view, &mut sub_e, &mut sub_g)?;
                self.state.rows.fetch_add(count, Ordering::Relaxed);
                energies[fixed_idx] = e_fixed;
                energies.slice_mut(s![range.clone()]).assign(&sub_e);
                gradients.slice_mut(s![range, ..]).assign(&sub_g);
            }
        }
        if !energies.iter().all(|e| e.is_finite()) {
            return Err(SaddleError::NonFinite("band energies"));
        }
        if !gradients.iter().all(|g| g.is_finite()) {
            return Err(SaddleError::NonFinite("band gradients"));
        }
        if !self.state.fixed.is_empty() {
            for mut row in gradients.outer_iter_mut() {
                zero_fixed(&self.state.fixed, &mut row);
            }
        }

        let target_energy = match fe_cfg.target_energy {
            Some(t) => t,
            None => *self
                .free_end_state
                .resolved_target
                .get_or_insert(energies[end_idx]),
        };
        self.free_end_state.resolved_target = Some(target_energy);

        let end_id = self.state.ids.get(end_idx).copied();
        let mut x_end = self.positions.row(end_idx).to_owned();
        let x_end_before_retract = x_end.clone();
        let mut e_end = energies[end_idx];
        let mut g_end = gradients.row(end_idx).to_owned();

        // Retract the starting terminal image onto E(x_end) = target_energy
        // using rgmin::Solver with LevelSetConstraint::new(target_energy).
        step_and_retract_endpoint_via_rgmin(
            &self.config,
            surface,
            &self.state.rows,
            end_id,
            target_energy,
            &mut x_end,
            &mut e_end,
            &mut g_end,
            None,
        )?;
        self.positions.row_mut(end_idx).assign(&x_end);
        self.pin_fixed();
        energies[end_idx] = e_end;
        gradients.row_mut(end_idx).assign(&g_end);

        // If initial level-set retraction moved the terminal image across a
        // large span (e.g. from a distant product minimum up to the target
        // contour), remesh the interior images between the two ends so the
        // chain does not fold back over itself.
        let mut retract_shift = &x_end - &x_end_before_retract;
        if let Some(cell) = &self.config.cell {
            cell.minimum_image(&mut retract_shift);
        }
        let mut neigh_chord = &self.positions.row(neigh_idx) - &x_end_before_retract;
        if let Some(cell) = &self.config.cell {
            cell.minimum_image(&mut neigh_chord);
        }
        if retract_shift.dot(&retract_shift) > neigh_chord.dot(&neigh_chord) && self.iteration == 0
        {
            let mut endpoints = Array2::zeros((2, dof));
            endpoints.row_mut(0).assign(&self.positions.row(0));
            endpoints
                .row_mut(1)
                .assign(&self.positions.row(n_images - 1));
            self.positions = resample(endpoints.view(), n_images, false, self.config.cell.as_ref());
            self.pin_fixed();
            let mut int_e = Array1::zeros(n_images - 2);
            let mut int_g = Array2::zeros((n_images - 2, dof));
            let int_view = self.positions.slice(s![1..n_images - 1, ..]);
            self.state
                .eval_rows(surface, 1..n_images - 1, int_view, &mut int_e, &mut int_g)?;
            self.state.rows.fetch_add(n_images - 2, Ordering::Relaxed);
            if !int_e.iter().all(|e| e.is_finite()) || !int_g.iter().all(|g| g.is_finite()) {
                return Err(SaddleError::NonFinite("band interior evaluation"));
            }
            if !self.state.fixed.is_empty() {
                for mut row in int_g.outer_iter_mut() {
                    zero_fixed(&self.state.fixed, &mut row);
                }
            }
            energies.slice_mut(s![1..n_images - 1]).assign(&int_e);
            gradients.slice_mut(s![1..n_images - 1, ..]).assign(&int_g);
        }

        // Eq. (7), translation removal, then the level-set projection.
        let mut g_end_free = gradients.row(end_idx).to_owned();
        zero_fixed(&self.config.fixed_atoms, &mut g_end_free);
        let f_end_level = assembled_free_end_force(
            &self.config,
            fe_cfg.side,
            self.positions.row(end_idx),
            self.positions.row(neigh_idx),
            g_end_free.view(),
        )?;

        // Assemble interior NEB forces.
        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let before = self.interior_flat();
        let mut max_i = 1;
        for i in 2..n_images - 1 {
            if energies[i] > energies[max_i] {
                max_i = i;
            }
        }
        let target_ci = climb_target(
            &self.config,
            max_i,
            endpoint_first.view(),
            endpoint_last.view(),
            before.view(),
            energies.view(),
            gradients.view(),
        );
        let climbing = self.config.climbing.is_some();
        let only_climb_final_level =
            fe_cfg.max_cuts == 0 || self.free_end_state.cuts >= fe_cfg.max_cuts;
        let armed = climbing && only_climb_final_level && self.state.armed.load(Ordering::Relaxed);
        if armed {
            let code = match target_ci {
                Some(idx) => idx as i64,
                None => -1,
            };
            self.state.ci_index.store(code, Ordering::Relaxed);
        } else {
            self.state.ci_index.store(-1, Ordering::Relaxed);
        }
        let ci_at = if armed { target_ci } else { None };

        let resolved_spring = self
            .config
            .spring
            .at_band(energies.view(), gradients.view());
        let metrics = self
            .exp
            .as_ref()
            .and_then(|exp| exp.fresh_metrics(before.view(), n_images - 2));
        let metric_of =
            |i: usize| metrics.and_then(|m| m.per_image.get(i - 1).map(|p| (p, m.free)));

        let mut interior_force = Array1::zeros(interior_dof);
        let mut step_force = Array1::zeros(interior_dof);
        for i in 1..n_images - 1 {
            let rows = s![(i - 1) * dof..i * dof];
            let (proj, drv) = image_force(
                &self.config,
                &self.state,
                &resolved_spring,
                i,
                ci_at == Some(i),
                endpoint_first.view(),
                endpoint_last.view(),
                before.view(),
                &energies,
                &gradients,
                metric_of(i),
            )?;
            interior_force.slice_mut(rows).assign(&proj);
            match drv {
                Some(d) => step_force.slice_mut(rows).assign(&d),
                None => step_force.slice_mut(rows).assign(&proj),
            }
        }

        let interior_norm = self.state.convergence_force(&interior_force);
        let endpoint_force_norm = match self.state.force_gate {
            ForceGate::LinfNorm => max_abs(&f_end_level),
            gate => gate.value(f_end_level.view()),
        };
        let mut max_force = interior_norm.max(endpoint_force_norm);

        let baseline = {
            let mut slot = lock_guard(&self.state.baseline);
            *slot.get_or_insert(max_force)
        };
        self.free_end_state
            .cut_baseline_force
            .get_or_insert(max_force);
        lock_guard(&self.state.atom_baseline)
            .get_or_insert_with(|| min_mode::max_atom(interior_force.view()));

        if let Some(ci_cfg) = &self.config.climbing
            && only_climb_final_level
            && !armed
            && (max_force < baseline * ci_cfg.trigger_factor || max_force < ci_cfg.trigger_force)
        {
            self.state.armed.store(true, Ordering::Relaxed);
            let code = match target_ci {
                Some(idx) => idx as i64,
                None => -1,
            };
            self.state.ci_index.store(code, Ordering::Relaxed);
            if let Some(ci_i) = target_ci {
                let rows = s![(ci_i - 1) * dof..ci_i * dof];
                let (proj, drv) = image_force(
                    &self.config,
                    &self.state,
                    &resolved_spring,
                    ci_i,
                    true,
                    endpoint_first.view(),
                    endpoint_last.view(),
                    before.view(),
                    &energies,
                    &gradients,
                    metric_of(ci_i),
                )?;
                interior_force.slice_mut(rows).assign(&proj);
                match drv {
                    Some(d) => step_force.slice_mut(rows).assign(&d),
                    None => step_force.slice_mut(rows).assign(&proj),
                }
                max_force = self
                    .state
                    .convergence_force(&interior_force)
                    .max(endpoint_force_norm);
            }
        }

        // Advance interior images using the session's solver.
        self.solver.forget_evaluation();
        let start_grad = -&step_force;
        let pseudo_e: f64 = energies.slice(s![1..n_images - 1]).sum();
        let config_ref = &self.config;
        let state_ref = &self.state;
        let e_first = energies[0];
        let e_last = energies[n_images - 1];
        let mut x_interior = before.clone();
        if interior_norm > 0.0 {
            let oracle = Oracle::unbounded(interior_dof, |xv: ArrayView1<f64>| {
                if xv == before.view() {
                    return (pseudo_e, start_grad.clone());
                }
                let Ok(trial_int) = xv.to_shape((n_images - 2, dof)) else {
                    state_ref.record_error(SaddleError::Shape("interior trial shape".into()));
                    return (f64::INFINITY, Array1::zeros(interior_dof));
                };
                let mut trial_e = Array1::zeros(n_images - 2);
                let mut trial_g = Array2::zeros((n_images - 2, dof));
                if let Err(err) = state_ref.eval_rows(
                    surface,
                    1..n_images - 1,
                    trial_int.view(),
                    &mut trial_e,
                    &mut trial_g,
                ) {
                    state_ref.record_error(err);
                    return (f64::INFINITY, Array1::zeros(interior_dof));
                }
                state_ref.rows.fetch_add(n_images - 2, Ordering::Relaxed);
                if !trial_e.iter().all(|v| v.is_finite()) || !trial_g.iter().all(|v| v.is_finite())
                {
                    state_ref.record_error(SaddleError::NonFinite("band trial evaluation"));
                    return (f64::INFINITY, Array1::zeros(interior_dof));
                }
                let mut full_e = Array1::zeros(n_images);
                full_e[0] = e_first;
                full_e[n_images - 1] = e_last;
                full_e.slice_mut(s![1..n_images - 1]).assign(&trial_e);
                let mut full_g = Array2::zeros((n_images, dof));
                full_g.row_mut(0).assign(&gradients.row(0));
                full_g
                    .row_mut(n_images - 1)
                    .assign(&gradients.row(n_images - 1));
                full_g.slice_mut(s![1..n_images - 1, ..]).assign(&trial_g);
                if !state_ref.fixed.is_empty() {
                    for mut row in full_g.outer_iter_mut() {
                        zero_fixed(&state_ref.fixed, &mut row);
                    }
                }
                let tr_spring = config_ref.spring.at_band(full_e.view(), full_g.view());
                let mut tr_step = Array1::zeros(interior_dof);
                for i in 1..n_images - 1 {
                    let rows = s![(i - 1) * dof..i * dof];
                    match image_force(
                        config_ref,
                        state_ref,
                        &tr_spring,
                        i,
                        ci_at == Some(i),
                        endpoint_first.view(),
                        endpoint_last.view(),
                        xv,
                        &full_e,
                        &full_g,
                        metric_of(i),
                    ) {
                        Ok((proj, drv)) => match drv {
                            Some(d) => tr_step.slice_mut(rows).assign(&d),
                            None => tr_step.slice_mut(rows).assign(&proj),
                        },
                        Err(err) => {
                            state_ref.record_error(err);
                            return (f64::INFINITY, Array1::zeros(interior_dof));
                        }
                    }
                }
                let tr_pseudo: f64 = trial_e.sum();
                (tr_pseudo, -tr_step)
            });
            let stepped = self.solver.step(&oracle, &mut x_interior);
            drop(oracle);
            if let Some(err) = self.state.take_error() {
                self.solver.forget_evaluation();
                return Err(err);
            }
            stepped.map_err(|e| SaddleError::Solver(e.to_string()))?;
        }
        self.scatter_interior(x_interior.view());

        // Advance the free terminal image along the level-set projected
        // force and retract onto E(x_end) = target_energy via rgmin::Solver.
        let mut dt_end = match self.free_end_state.endpoint_dt {
            Some(dt) => dt,
            None => 1.0 / (self.config.time_step.max(1.0) * 5.0),
        };
        if let (Some(prev_pos), Some(prev_f)) = (
            self.free_end_state.prev_end_pos.take(),
            self.free_end_state.prev_end_force.take(),
        ) {
            let mut s_end = &x_end - &prev_pos;
            if let Some(cell) = &self.config.cell {
                cell.minimum_image(&mut s_end);
            }
            let y_end = &prev_f - &f_end_level;
            let sy = s_end.dot(&y_end);
            let yy = y_end.dot(&y_end);
            if sy > 0.0 && yy > 0.0 {
                let bb = sy / yy;
                if bb.is_finite() && bb > 0.0 {
                    dt_end = bb.min(self.config.time_step);
                }
            } else if s_end.dot(&f_end_level) < 0.0 {
                dt_end *= 0.5;
            }
        }
        self.free_end_state.endpoint_dt = Some(dt_end);
        self.free_end_state.prev_end_pos = Some(x_end.clone());
        self.free_end_state.prev_end_force = Some(f_end_level.clone());

        let end_step = &f_end_level * dt_end;
        step_and_retract_endpoint_via_rgmin(
            &self.config,
            surface,
            &self.state.rows,
            end_id,
            target_energy,
            &mut x_end,
            &mut e_end,
            &mut g_end,
            Some(&end_step),
        )?;
        self.positions.row_mut(end_idx).assign(&x_end);
        self.pin_fixed();

        let residual = e_end - target_energy;
        self.free_end_state.last_residual = Some(residual);
        self.free_end_state.last_endpoint_energy = Some(e_end);
        self.free_end_state.last_endpoint_force = Some(endpoint_force_norm);
        self.state.drop_last();
        *lock_guard(&self.state.last_max_force) = max_force;
        self.iteration += 1;

        // Track peak stability for adaptive cut-and-remesh.
        if self.free_end_state.prev_peak == Some(max_i) {
            self.free_end_state.stable += 1;
        } else {
            self.free_end_state.prev_peak = Some(max_i);
            self.free_end_state.stable = 1;
        }

        let mut cut_this_step = false;
        if fe_cfg.max_cuts > 0
            && self.free_end_state.cuts < fe_cfg.max_cuts
            && self.free_end_state.stable >= fe_cfg.stability
        {
            let c_idx = self.free_end_state.cuts;
            let final_cut_tol = 4.0 * self.config.force_tol;
            let first_cut_tol = first_cut_tolerance(
                fe_cfg.cut_force,
                self.free_end_state.cut_baseline_force,
                final_cut_tol,
            );
            let cut_limit = cut_force_limit(first_cut_tol, final_cut_tol, c_idx, fe_cfg.max_cuts);
            let ediff = energies[max_i] - energies[max_i - 1].min(energies[max_i + 1]);

            if max_force <= cut_limit
                && ediff > fe_cfg.energy_tol
                && let Some((lo, hi)) = peak_window(n_images, max_i, fe_cfg.window_offset)
            {
                // The free end stays an endpoint of the remeshed chain.
                // When the window does not contain it, the leg from the
                // window to that image is part of the polyline, so the
                // equal-arc images do not bunch inside the window.
                self.positions.row_mut(end_idx).assign(&x_end);
                let path = cut_path(self.positions.view(), lo, hi, end_idx);
                self.positions = resample(
                    path.view(),
                    n_images,
                    fe_cfg.cubic,
                    self.config.cell.as_ref(),
                );
                self.pin_fixed();
                x_end.assign(&self.positions.row(end_idx));
                self.free_end_state.cuts += 1;
                self.free_end_state.cut_windows.push((lo, hi));
                self.free_end_state.prev_peak = None;
                self.free_end_state.stable = 0;
                self.forget_optimizer();
                let method = if self.config.quickmin {
                    Method::QuickMin
                } else {
                    self.config.method.clone()
                };
                self.solver = build_solver(&self.config, &method, interior_dof);
                self.solver.set_highs(self.highs);
                cut_this_step = true;
            }
        }

        let report_ci = if cut_this_step { None } else { ci_at };
        let measured =
            self.measure_free_end(surface, fe_cfg.side, end_idx, neigh_idx, report_ci)?;
        self.free_end_state.fixed_endpoint_energy = Some(measured.energies[fixed_idx]);
        let residual = measured.energies[end_idx] - target_energy;
        let e_end = measured.energies[end_idx];
        let max_force = measured.max_force;
        let endpoint_force_norm = measured.endpoint_force;
        self.free_end_state.last_residual = Some(residual);
        self.free_end_state.last_endpoint_energy = Some(e_end);
        self.free_end_state.last_endpoint_force = Some(endpoint_force_norm);
        *lock_guard(&self.state.last_max_force) = max_force;
        self.cache_free_end_evaluation(measured.energies, measured.gradients, measured.projected);

        let all_cuts_done = fe_cfg.max_cuts == 0 || self.free_end_state.cuts >= fe_cfg.max_cuts;
        let status = if !cut_this_step && all_cuts_done && max_force <= self.config.force_tol {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        self.refine_latch(status)?;

        Ok(FreeEndReport {
            status,
            max_force,
            endpoint_force: endpoint_force_norm,
            target_energy,
            endpoint_energy: e_end,
            target_energy_residual: residual,
            cuts: self.free_end_state.cuts,
            cut_this_step,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
            saddle_index: None,
        })
    }

    fn free_end_report_from_band(&self, band: BandReport) -> FreeEndReport {
        let state = &self.free_end_state;
        let (target_energy, endpoint_energy, residual, endpoint_force) =
            if let (Some(target), Some(energy), Some(residual), Some(force)) = (
                state.resolved_target,
                state.last_endpoint_energy,
                state.last_residual,
                state.last_endpoint_force,
            ) {
                (target, energy, residual, force)
            } else {
                (0.0, 0.0, 0.0, band.max_force)
            };
        FreeEndReport {
            status: band.status,
            max_force: band.max_force,
            endpoint_force,
            target_energy,
            endpoint_energy,
            target_energy_residual: residual,
            cuts: self.free_end_state.cuts,
            cut_this_step: false,
            ci_index: band.ci_index,
            iteration: band.iteration,
            surface_rows: band.surface_rows,
            saddle_index: band.saddle_index,
        }
    }
}

/// Dedicated session for the free-end adaptive nudged elastic band
/// (FEA-NEB) of Jiayong Zhang, Hongwu Zhang, Hongfei Ye, and Yonggang
/// Zheng, J. Chem. Phys. 145, 094104 (2016), doi:10.1063/1.4962019.
pub struct FeaNebSession {
    session: BandSession,
}

impl FeaNebSession {
    /// Open a free-end adaptive NEB session over `positions` (`n_images >= 3`
    /// rows of `3 * n_atoms`).
    pub fn new(
        config: BandConfig,
        free_end: FreeEndConfig,
        positions: Array2<f64>,
    ) -> Result<Self, SaddleError> {
        let session = BandSession::new(config.with_free_end(free_end), positions)?;
        Ok(Self { session })
    }

    /// Take one FEA-NEB step and return the [`FreeEndReport`] with
    /// `target_energy_residual`.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<FreeEndReport, SaddleError> {
        self.session.step_free_end(surface)
    }

    /// Loop [`FeaNebSession::step`] until convergence or `max_steps`.
    pub fn run<S: BandSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<FreeEndReport, SaddleError> {
        self.session.run_free_end(surface, max_steps)
    }

    /// Current band positions (`n_images x 3N`).
    #[must_use]
    pub fn positions(&self) -> ndarray::ArrayView2<'_, f64> {
        self.session.positions()
    }

    /// Signed target-energy residual `E(x_end) - E_target` from the last step.
    #[must_use]
    pub fn target_energy_residual(&self) -> Option<f64> {
        self.session.target_energy_residual()
    }

    /// Number of adaptive cut-and-remesh refinements performed so far.
    #[must_use]
    pub fn cuts(&self) -> usize {
        self.session.free_end_cuts()
    }

    /// Window index pairs `(lo, hi)` of each adaptive cut-and-remesh.
    #[must_use]
    pub fn cut_windows(&self) -> &[(usize, usize)] {
        self.session.free_end_windows()
    }

    /// Initial energy of the fixed endpoint before any adaptive cuts.
    #[must_use]
    pub fn initial_fixed_endpoint_energy(&self) -> Option<f64> {
        self.session.initial_fixed_endpoint_energy()
    }

    /// Index of the armed climbing image, if any.
    #[must_use]
    pub fn climbing_image(&self) -> Option<usize> {
        self.session.climbing_image()
    }

    /// Certified saddle index of the climbing image at convergence, if any.
    #[must_use]
    pub fn saddle_index(&self) -> Option<&SaddleIndex> {
        self.session.saddle_index()
    }

    /// Completed steps in this session.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.session.iteration()
    }

    /// Borrow the underlying [`BandSession`].
    #[must_use]
    pub fn band_session(&self) -> &BandSession {
        &self.session
    }

    /// Consume into the underlying [`BandSession`].
    #[must_use]
    pub fn into_band_session(self) -> BandSession {
        self.session
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, ArrayView2, array};

    use super::{
        SAME_ROW_TOL, accept_endpoint_pair, assembled_free_end_force, cut_force_limit,
        eq7_endpoint_force, first_cut_tolerance, peak_window, project_onto_level_set,
    };
    use crate::{
        BandConfig, BandSession, BandSurface, FeaNebSession, FreeEndConfig, FreeEndSide,
        SaddleError, SpringKind,
    };

    #[test]
    fn cut_force_schedule_never_increases() {
        let final_cut_tol = 0.4;
        for sample in 0..=100 {
            let cut_force = final_cut_tol * sample as f64 / 100.0;
            let first_cut_tol =
                first_cut_tolerance(cut_force, Some(4.0 * final_cut_tol), final_cut_tol);
            let limits: Vec<_> = (0..4)
                .map(|cut| cut_force_limit(first_cut_tol, final_cut_tol, cut, 4))
                .collect();
            assert_eq!(limits[3], final_cut_tol);
            assert!(limits.windows(2).all(|pair| pair[0] >= pair[1]));
            if cut_force > 0.0 {
                assert_eq!(limits, vec![final_cut_tol; 4]);
            }
        }

        let above_final = f64::from_bits(final_cut_tol.to_bits() + 1);
        let above_final =
            first_cut_tolerance(above_final, Some(4.0 * final_cut_tol), final_cut_tol);
        let limits: Vec<_> = (0..4)
            .map(|cut| cut_force_limit(above_final, final_cut_tol, cut, 4))
            .collect();
        assert_eq!(limits[0], above_final);
        assert_eq!(limits[3], final_cut_tol);
        assert!(limits.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    struct CutSurface;

    impl BandSurface for CutSurface {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (image, row) in positions.outer_iter().enumerate() {
                let delta = row[0] - 1.0;
                energies[image] = -(delta * delta);
                gradients[(image, 0)] = -2.0 * delta;
            }
            Ok(())
        }
    }

    #[test]
    fn cut_forgets_the_endpoint_step_state() {
        let positions = Array2::from_shape_fn(
            (5, 3),
            |(image, axis)| if axis == 0 { image as f64 } else { 0.0 },
        );
        let config = BandConfig::default()
            .with_free_end(
                FreeEndConfig::default()
                    .with_max_cuts(1)
                    .with_cut_force(100.0)
                    .with_stability(1)
                    .with_cubic(false),
            )
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false);
        let mut session = BandSession::new(config, positions).expect("session");
        session.free_end_state.endpoint_dt = Some(0.0125);

        let report = session.step_free_end(&CutSurface).expect("cut step");

        assert!(report.cut_this_step);
        assert_eq!(session.free_end_state.endpoint_dt, None);
        assert_eq!(session.free_end_state.prev_end_pos, None);
        assert_eq!(session.free_end_state.prev_end_force, None);
    }

    #[test]
    fn endpoint_force_matches_analytic_surface_and_eq_7_is_orthogonal() {
        // E(x, y, z) = 3 x^2 / 2 + 2 y^2, so grad E(1, 1, 0) = (3, 4, 0).
        let config = BandConfig::default()
            .with_remove_translation(false)
            .with_spring(SpringKind::Uniform { k: 2.5 });
        let end = array![2.0, 1.0, 0.0];
        let neighbour = array![0.0, 1.0, 0.0];
        let gradient = array![3.0, 4.0, 0.0];
        let force = eq7_endpoint_force(
            &config,
            FreeEndSide::Last,
            end.view(),
            neighbour.view(),
            gradient.view(),
        )
        .expect("endpoint force");

        // Eq. (7) is the perpendicular potential force (0, -4, 0).
        let expected = array![0.0, -4.0, 0.0];
        assert_eq!(force, expected);
        let tangent = array![1.0, 0.0, 0.0];
        assert!(force.dot(&tangent).abs() < SAME_ROW_TOL);
        let stiffer = config.clone().with_spring(SpringKind::Uniform { k: 9.0 });
        assert_eq!(
            eq7_endpoint_force(
                &stiffer,
                FreeEndSide::Last,
                end.view(),
                neighbour.view(),
                gradient.view(),
            )
            .expect("stiffer spring"),
            expected
        );

        let first_end = array![0.0, 1.0, 0.0];
        let first_neighbour = array![2.0, 1.0, 0.0];
        let first_force = eq7_endpoint_force(
            &config,
            FreeEndSide::First,
            first_end.view(),
            first_neighbour.view(),
            gradient.view(),
        )
        .expect("first endpoint force");
        assert_eq!(first_force, expected);

        assert!(matches!(
            eq7_endpoint_force(
                &config,
                FreeEndSide::First,
                first_end.view(),
                first_end.view(),
                gradient.view(),
            ),
            Err(SaddleError::Invalid(_))
        ));
        let tolerance_neighbour = array![SAME_ROW_TOL, 1.0, 0.0];
        assert!(matches!(
            eq7_endpoint_force(
                &config,
                FreeEndSide::First,
                first_end.view(),
                tolerance_neighbour.view(),
                gradient.view(),
            ),
            Err(SaddleError::Invalid(_))
        ));
    }

    #[test]
    fn level_set_force_keeps_the_downhill_sign_without_a_spring() {
        // Last end, neighbour to the left, tangent +x. The parallel
        // gradient sits where a one-sided spring would flip F·n.
        let config = BandConfig::default()
            .with_remove_translation(false)
            .with_spring(SpringKind::Uniform { k: 2.5 });
        let end = array![2.0, 0.0, 0.0];
        let neighbour = array![0.0, 0.0, 0.0];
        let gradient = array![1.0, 4.0, 0.0];
        let force = assembled_free_end_force(
            &config,
            FreeEndSide::Last,
            end.view(),
            neighbour.view(),
            gradient.view(),
        )
        .expect("assembled force");
        let downhill = array![0.0, -1.0, 0.0];
        assert!(force.dot(&downhill) > 0.0);
        assert!(force.dot(&gradient).abs() < SAME_ROW_TOL);
    }

    #[test]
    fn translation_removal_before_projection_stays_orthogonal() {
        let config = BandConfig::default().with_spring(SpringKind::Uniform { k: 2.5 });
        let end = array![2.0, 0.0, 0.0, 2.0, 1.0, 0.0];
        let neighbour = array![0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let gradient = array![1.0, 0.0, 0.0, 3.0, 4.0, 0.0];
        let force = assembled_free_end_force(
            &config,
            FreeEndSide::Last,
            end.view(),
            neighbour.view(),
            gradient.view(),
        )
        .expect("assembled force");
        assert!(force.dot(&gradient).abs() < SAME_ROW_TOL);

        let perpendicular = eq7_endpoint_force(
            &config,
            FreeEndSide::Last,
            end.view(),
            neighbour.view(),
            gradient.view(),
        )
        .expect("eq 7");
        let mut projected_first =
            project_onto_level_set(&gradient, &perpendicular).expect("project first");
        super::remove_net_translation(&mut projected_first);
        assert!(projected_first.dot(&gradient).abs() > SAME_ROW_TOL);
    }

    #[test]
    fn level_set_projection_matches_the_analytic_tangent_component() {
        let gradient = array![3.0, 4.0, 0.0];
        let force = array![5.0, -2.0, 1.0];
        let projected = project_onto_level_set(&gradient, &force).expect("level-set projection");
        let expected = array![4.16, -3.12, 1.0];
        for (actual, target) in projected.iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 1e-12);
        }
        assert!(projected.dot(&gradient).abs() < 1e-12);

        assert_eq!(
            project_onto_level_set(&gradient, &Array1::<f64>::zeros(3)).expect("zero force"),
            Array1::<f64>::zeros(3)
        );
        assert!(matches!(
            project_onto_level_set(&Array1::zeros(3), &force),
            Err(SaddleError::Invalid(_))
        ));
        assert!(matches!(
            project_onto_level_set(&array![f64::NAN, 0.0, 0.0], &force),
            Err(SaddleError::Invalid(_))
        ));
        assert!(matches!(
            project_onto_level_set(&gradient, &array![f64::NAN, 0.0, 0.0]),
            Err(SaddleError::NonFinite(_))
        ));
    }

    #[test]
    fn host_position_change_and_reset_invalidate_fixed_endpoint_energy() {
        let positions = Array2::from_shape_fn(
            (3, 3),
            |(image, axis)| if axis == 0 { image as f64 } else { 0.0 },
        );
        let config = BandConfig::default()
            .with_free_end(FreeEndConfig::default())
            .with_min_mode(None)
            .with_saddle_check(None);
        let mut session = BandSession::new(config, positions.clone()).expect("session");
        session.free_end_state.fixed_endpoint_energy = Some(7.0);

        let mut moved = positions;
        moved[(0, 0)] = -1.0;
        session.set_positions(moved).expect("replace positions");
        assert_eq!(session.free_end_state.fixed_endpoint_energy, None);
        assert_eq!(session.initial_fixed_endpoint_energy(), None);

        session.free_end_state.fixed_endpoint_energy = Some(8.0);
        session.reset();
        assert_eq!(session.free_end_state.fixed_endpoint_energy, None);

        let facade_positions =
            Array2::from_shape_fn(
                (3, 3),
                |(image, axis)| if axis == 0 { image as f64 } else { 0.0 },
            );
        let facade = FeaNebSession::new(
            BandConfig::default()
                .with_min_mode(None)
                .with_saddle_check(None),
            FreeEndConfig::default(),
            facade_positions,
        )
        .expect("free-end facade");
        assert_eq!(facade.initial_fixed_endpoint_energy(), None);
    }

    #[test]
    fn peak_window_stays_off_the_ends_and_skips_a_full_band() {
        assert_eq!(peak_window(5, 1, 1), Some((0, 2)));
        assert_eq!(peak_window(5, 3, 1), Some((2, 4)));
        assert_eq!(peak_window(3, 1, 1), None);
        assert_eq!(peak_window(5, 0, 1), None);
        assert_eq!(peak_window(5, 4, 1), None);
    }

    #[test]
    fn refused_curvature_pair_is_reported_for_both_sides() {
        use rgmin::{Accept, Control, LevelSetConstraint, Method, Solver};
        for side in [FreeEndSide::First, FreeEndSide::Last] {
            let mut solver = Solver::new(
                Method::lbfgs_with_memory(1),
                Control::default()
                    .with_maxiter(usize::MAX)
                    .with_gtol(0.0)
                    .with_istep(1.0),
                3,
            );
            solver.set_accept(Accept::Step);
            solver.set_lbfgs_neb_guards(false);
            solver
                .set_level_set(LevelSetConstraint::new(0.0))
                .expect("level set");
            solver.forget();
            let step = array![1.0, 0.0, 0.0];
            let slope = array![-1.0, 0.0, 0.0];
            let err = accept_endpoint_pair(&mut solver, step.view(), slope.view())
                .expect_err("refused curvature pair");
            assert!(
                matches!(err, SaddleError::Stalled(_)),
                "{side:?} reported {err}"
            );
            assert_eq!(solver.pair_count(), 0, "{side:?}");
            solver.forget();
            let rise = array![1.0, 0.0, 0.0];
            accept_endpoint_pair(&mut solver, rise.view(), rise.view())
                .expect("kept curvature pair");
            assert!(solver.pair_count() > 0, "{side:?}");
        }
    }

    #[test]
    fn step_free_end_runs_min_mode_refinement_on_both_sides() {
        use crate::{BandSurface, CiConfig};

        /// `E = (x^2 - 1)^2 + 4 (y - 0.3 x)^2`. The interior peak sits
        /// above both ends, and the images start off the valley, so the
        /// band force stays above the tolerance while climbing arms.
        struct Barrier;
        impl BandSurface for Barrier {
            fn eval(
                &self,
                positions: ArrayView2<f64>,
                energies: &mut Array1<f64>,
                gradients: &mut Array2<f64>,
            ) -> Result<(), SaddleError> {
                for (i, row) in positions.outer_iter().enumerate() {
                    let (x, y) = (row[0], row[1]);
                    let delta = x * x - 1.0;
                    let off = y - 0.3 * x;
                    energies[i] = delta * delta + 4.0 * off * off;
                    gradients[(i, 0)] = 4.0 * x * delta - 2.4 * off;
                    gradients[(i, 1)] = 8.0 * off;
                    gradients[(i, 2)] = 0.0;
                }
                Ok(())
            }
        }

        let mut misses = Vec::new();
        for side in [FreeEndSide::First, FreeEndSide::Last] {
            let positions = Array2::from_shape_fn((5, 3), |(image, axis)| match axis {
                0 => -1.2 + 0.6 * image as f64,
                _ => 0.0,
            });
            let config = BandConfig::default()
                .with_free_end(FreeEndConfig::default().with_side(side).with_max_cuts(0))
                .with_climbing(Some(CiConfig::always()))
                .with_saddle_check(None)
                .with_remove_translation(false);
            let mut session = BandSession::new(config, positions).expect("session");
            let mut saw_refine = false;
            for _ in 0..4 {
                session.step_free_end(&Barrier).expect("free-end step");
                saw_refine |= session.refine_last().is_some();
            }
            if !saw_refine {
                misses.push(format!(
                    "{side:?} step_free_end never ran min-mode refinement"
                ));
            }
        }
        assert!(misses.is_empty(), "{misses:#?}");
    }
}
