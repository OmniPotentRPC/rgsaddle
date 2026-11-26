//! Solution-landscape C entry points. Layouts live in `include/rgsaddle.h`.

use std::ffi::c_void;
use std::slice;

use ndarray::Array1;

use super::indexk::{HostHessian, HostSurface, RgsaddleHessianAtFn, read_modes, write_modes};
use super::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NO_EVALUATION,
    RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SESSION, RGSADDLE_NULL_SURFACE, RGSADDLE_OK,
    RGSADDLE_SHAPE, RgsaddleSurfaceFn, RgsaddleVersion, status_of,
};
use crate::Cell;
use crate::error::SaddleError;
use crate::indexk::{BlockEigenKind, IndexKConfig};
use crate::landscape::{LandscapeConfig, LandscapeSession, PathwayMap};
use crate::minmode::{MinModeConfig, MinModeKind};
use crate::relax::RelaxConfig;

#[repr(C)]
/// Solution-landscape configuration. The layout matches
/// `rgsaddle_landscape_config_t`.
pub struct RgsaddleLandscapeConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. Reserved; write zero.
    pub flags: u64,
    /// Euclidean separation at or below which two positions are one node.
    pub duplicate_distance: f64,
    /// Euclidean length of the push along one unit unstable eigenvector.
    pub push: f64,
    /// Most nodes, including the parent. At least 1.
    pub max_nodes: i64,
    /// Most [`crate::PointSurface::eval`] calls. At least 1.
    pub max_evaluations: i64,
    /// Most translations of one saddle leg, and the step cap of one minimum.
    pub max_steps: i64,
    /// Infinity-norm force stop of a saddle leg, energy per length.
    pub force_tol: f64,
    /// Infinity-norm force stop of a minimum, energy per length.
    pub relax_force_tol: f64,
    /// Largest per-atom translation of one saddle step, in length units.
    pub max_move: f64,
    /// Largest per-atom step of the minimizer, in length units.
    pub max_atom_move: f64,
    /// Opening step of the minimizer, in length units.
    pub initial_step: f64,
    /// Finite-difference displacement of a Hessian action, in length units.
    pub dr: f64,
    /// Dimer rotational-force stop, or Lanczos Ritz residual.
    pub rotation_tol: f64,
    /// Block-eigen residual used when a leg has index greater than 1.
    pub residual_tol: f64,
    /// Cap on dimer rotations or Lanczos actions in one estimate.
    pub max_rotations: i64,
    /// Cap on block eigen iterations. Positive.
    pub max_block_iterations: i32,
    /// Estimator, a `rgsaddle_minmode_t`, as [`super::RgsaddleIndexKConfig::method`].
    pub method: i32,
    /// Row-major 3x3 cell, or NULL for Euclidean distance.
    pub cell: *const f64,
}

#[repr(C)]
/// Counts from [`rgsaddle_landscape_run`].
pub struct RgsaddleLandscapeReport {
    /// ABI version.
    pub version: RgsaddleVersion,
    /// Bit flags. Reserved; the library writes zero.
    pub flags: u64,
    /// Nodes accepted. On [`super::RGSADDLE_LANDSCAPE_BOUND`] this is the
    /// count in the error, and the nodes themselves are not stored.
    pub n_nodes: i64,
    /// Edges stored. Zero when the search stopped on a limit.
    pub n_edges: i64,
    /// Which limit stopped the search: 0 when it finished, 1 for nodes,
    /// 2 for force evaluations.
    pub bound: i32,
    /// Force evaluations performed.
    pub evaluations: i64,
}

/// Opaque solution-landscape search.
pub struct RgsaddleLandscape {
    session: LandscapeSession,
    n_atoms: i64,
    map: Option<PathwayMap>,
}

fn positive_count(value: i64) -> Result<usize, i32> {
    usize::try_from(value).map_err(|_| RGSADDLE_INVALID_PARAMETER)
}

fn read_cell(cell: *const f64) -> Result<Option<Cell>, i32> {
    if cell.is_null() {
        return Ok(None);
    }
    let raw = unsafe { slice::from_raw_parts(cell, 9) };
    if !raw.iter().copied().all(f64::is_finite) {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    Ok(Some(Cell([
        [raw[0], raw[1], raw[2]],
        [raw[3], raw[4], raw[5]],
        [raw[6], raw[7], raw[8]],
    ])))
}

fn config_of(config: &RgsaddleLandscapeConfig) -> Result<LandscapeConfig, i32> {
    if config.version.major != RGSADDLE_ABI_MAJOR {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    if config.flags != 0 {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    if config.max_block_iterations <= 0 {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    let (kind, block) = match config.method {
        0 => (MinModeKind::Dimer, BlockEigenKind::Lobpcg),
        1 => (MinModeKind::Lanczos, BlockEigenKind::BlockLanczos),
        2 => (MinModeKind::Lanczos, BlockEigenKind::Lobpcg),
        _ => return Err(RGSADDLE_INVALID_PARAMETER),
    };
    let max_nodes = positive_count(config.max_nodes)?;
    let max_evaluations = positive_count(config.max_evaluations)?;
    let max_steps = positive_count(config.max_steps)?;
    let max_rotations = positive_count(config.max_rotations)?;
    let search = MinModeConfig::default()
        .with_kind(kind)
        .with_dr(config.dr)
        .with_rotation_tol(config.rotation_tol)
        .with_max_rotations(max_rotations)
        .with_force_tol(config.force_tol)
        .with_max_move(config.max_move)
        .with_saddle_check(None);
    let search = IndexKConfig::new(1, search)
        .with_residual_tol(config.residual_tol)
        .with_max_block_iterations(config.max_block_iterations as usize)
        .with_block(block);
    let relax = RelaxConfig::default()
        .with_force_tol(config.relax_force_tol)
        .with_max_steps(max_steps)
        .with_max_atom_move(config.max_atom_move)
        .with_initial_step(config.initial_step);
    Ok(LandscapeConfig::default()
        .with_duplicate_distance(config.duplicate_distance)
        .with_push(config.push)
        .with_max_nodes(max_nodes)
        .with_max_evaluations(max_evaluations)
        .with_max_steps(max_steps)
        .with_search(search)
        .with_relax(relax)
        .with_cell(read_cell(config.cell)?))
}

fn stamp(report: &mut RgsaddleLandscapeReport) {
    report.version = RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    };
    report.flags = 0;
}

/// Fill `out` with the Rust defaults. `cell` is NULL.
///
/// # Safety
/// `out` is writable, or NULL (which returns [`RGSADDLE_INVALID_PARAMETER`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_config_default(
    out: *mut RgsaddleLandscapeConfig,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let defaults = LandscapeConfig::default();
    let search = &defaults.search.search;
    unsafe {
        *out = RgsaddleLandscapeConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            duplicate_distance: defaults.duplicate_distance,
            push: defaults.push,
            max_nodes: defaults.max_nodes as i64,
            max_evaluations: defaults.max_evaluations as i64,
            max_steps: defaults.max_steps as i64,
            force_tol: search.force_tol,
            relax_force_tol: defaults.relax.force_tol,
            max_move: search.max_move,
            max_atom_move: defaults.relax.max_atom_move,
            initial_step: defaults.relax.initial_step,
            dr: search.dr,
            rotation_tol: search.rotation_tol,
            residual_tol: defaults.search.residual_tol,
            max_rotations: search.max_rotations as i64,
            max_block_iterations: defaults.search.max_block_iterations as i32,
            method: 0,
            cell: std::ptr::null(),
        };
    }
    RGSADDLE_OK
}

/// Create a search at `position` (`3 * n_atoms` doubles).
///
/// `modes` is NULL when `k` is 0. Otherwise it is a row-major
/// `(3 * n_atoms)` by `k` block, copied, and `k` must equal the index
/// later certified at the parent. Returns NULL when the config, the
/// shape, or the position is refused.
///
/// # Safety
/// `config` is readable. `position` holds `3 * n_atoms` doubles. `modes`
/// is NULL or holds `(3 * n_atoms) * k` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_create(
    config: *const RgsaddleLandscapeConfig,
    n_atoms: i64,
    position: *const f64,
    modes: *const f64,
    k: i32,
) -> *mut RgsaddleLandscape {
    if config.is_null() || position.is_null() || n_atoms < 1 || k < 0 {
        return std::ptr::null_mut();
    }
    if modes.is_null() != (k == 0) {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    let Ok(parsed) = config_of(cfg) else {
        return std::ptr::null_mut();
    };
    let Some(n) = usize::try_from(n_atoms)
        .ok()
        .and_then(|atoms| atoms.checked_mul(3))
    else {
        return std::ptr::null_mut();
    };
    let raw = unsafe { slice::from_raw_parts(position, n) };
    let position = Array1::from(raw.to_vec());
    let modes = if k == 0 {
        None
    } else {
        let Ok(Some(modes)) = read_modes(n, k as usize, modes) else {
            return std::ptr::null_mut();
        };
        Some(modes)
    };
    let Ok(session) = LandscapeSession::new(parsed, position, modes) else {
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(RgsaddleLandscape {
        session,
        n_atoms,
        map: None,
    }))
}

/// Search downward. `hessian` NULL estimates each action from `surface`.
/// A non-NULL `hessian` is a `rgsaddle_hessian_at_fn` and receives the
/// geometry at which the action is requested.
///
/// On [`super::RGSADDLE_LANDSCAPE_BOUND`], `out` still receives the accepted
/// node count and the evaluations performed, `n_edges` is 0, and the
/// pathway is discarded.
///
/// # Safety
/// `session` comes from [`rgsaddle_landscape_create`]. `out` is writable.
/// `surface` and `hessian` are called with `user` and must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_run(
    session: *mut RgsaddleLandscape,
    surface: Option<RgsaddleSurfaceFn>,
    hessian: Option<RgsaddleHessianAtFn>,
    user: *mut c_void,
    out: *mut RgsaddleLandscapeReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(function) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let host = HostSurface::new(
        function,
        user,
        session.n_atoms,
        hessian.map(HostHessian::AtPosition),
    );
    let out = unsafe { &mut *out };
    stamp(out);
    match session.session.run(&host) {
        Ok(map) => {
            out.n_nodes = map.nodes.len() as i64;
            out.n_edges = map.edges.len() as i64;
            out.bound = 0;
            out.evaluations = map.evaluations as i64;
            session.map = Some(map);
            RGSADDLE_OK
        }
        Err(SaddleError::LandscapeBound {
            kind,
            nodes,
            evaluations,
            ..
        }) => {
            session.map = None;
            out.n_nodes = nodes as i64;
            out.n_edges = 0;
            out.bound = match kind {
                crate::LandscapeBoundKind::Nodes => 1,
                crate::LandscapeBoundKind::Evaluations => 2,
            };
            out.evaluations = evaluations as i64;
            super::RGSADDLE_LANDSCAPE_BOUND
        }
        Err(err) => {
            session.map = None;
            out.n_nodes = 0;
            out.n_edges = 0;
            out.bound = 0;
            out.evaluations = 0;
            status_of(&err)
        }
    }
}

/// One stored node. `curvatures` may be NULL to query `n_curvatures`.
///
/// A short curvature buffer returns [`RGSADDLE_SHAPE`] and writes nothing
/// into it. [`RGSADDLE_NO_EVALUATION`] means `run` has not stored a pathway.
///
/// # Safety
/// `position`, when the call can succeed, holds `3 * n_atoms` doubles.
/// `curvatures` is NULL or holds `curvature_capacity` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_node(
    session: *const RgsaddleLandscape,
    node: i64,
    position: *mut f64,
    energy: *mut f64,
    index: *mut i32,
    curvatures: *mut f64,
    curvature_capacity: i32,
    n_curvatures: *mut i32,
    evaluations: *mut i64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if position.is_null()
        || energy.is_null()
        || index.is_null()
        || n_curvatures.is_null()
        || evaluations.is_null()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let Some(map) = session.map.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let Some(node) = usize::try_from(node)
        .ok()
        .and_then(|node| map.nodes.get(node))
    else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(count) = i32::try_from(node.curvatures.len()) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe {
        *n_curvatures = count;
    }
    if !curvatures.is_null() {
        if curvature_capacity < count {
            return RGSADDLE_SHAPE;
        }
        unsafe {
            slice::from_raw_parts_mut(curvatures, node.curvatures.len())
                .copy_from_slice(&node.curvatures);
        }
    }
    let n = node.position.len();
    unsafe {
        for (slot, value) in slice::from_raw_parts_mut(position, n)
            .iter_mut()
            .zip(node.position.iter())
        {
            *slot = *value;
        }
        *energy = node.energy;
        *index = match i32::try_from(node.index) {
            Ok(index) => index,
            Err(_) => return RGSADDLE_INVALID_PARAMETER,
        };
        *evaluations = node.evaluations as i64;
    }
    RGSADDLE_OK
}

/// Unstable modes of one node, row-major, `3 * n_atoms` by the saddle index.
///
/// `modes` NULL writes `n_columns` and returns. A minimum has no columns.
/// A non-NULL `modes` pointer is left unread when the column count is zero.
///
/// # Safety
/// `modes`, when non-NULL, holds `(3 * n_atoms) * n_columns` doubles after
/// a successful query of `n_columns`, or at least that many on this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_modes(
    session: *const RgsaddleLandscape,
    node: i64,
    modes: *mut f64,
    n_columns: *mut i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if n_columns.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let Some(map) = session.map.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let Some(node) = usize::try_from(node)
        .ok()
        .and_then(|node| map.nodes.get(node))
    else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let Ok(columns) = i32::try_from(node.modes.ncols()) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe {
        *n_columns = columns;
    }
    if modes.is_null() {
        return RGSADDLE_OK;
    }
    let count = node.modes.len();
    if count == 0 {
        return RGSADDLE_OK;
    }
    let out = unsafe { slice::from_raw_parts_mut(modes, count) };
    write_modes(node.modes.view(), out);
    RGSADDLE_OK
}

/// One stored edge. Every pointer is required.
///
/// # Safety
/// Each pointer is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_edge(
    session: *const RgsaddleLandscape,
    edge: i64,
    parent: *mut i64,
    child: *mut i64,
    direction: *mut i32,
    sign: *mut i32,
    evaluations: *mut i64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if parent.is_null()
        || child.is_null()
        || direction.is_null()
        || sign.is_null()
        || evaluations.is_null()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let Some(map) = session.map.as_ref() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let Some(edge) = usize::try_from(edge)
        .ok()
        .and_then(|edge| map.edges.get(edge))
    else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe {
        *parent = edge.parent as i64;
        *child = edge.child as i64;
        *direction = match i32::try_from(edge.direction) {
            Ok(direction) => direction,
            Err(_) => return RGSADDLE_INVALID_PARAMETER,
        };
        *sign = i32::from(edge.sign);
        *evaluations = edge.evaluations as i64;
    }
    RGSADDLE_OK
}

/// Release a search. NULL is a no-op.
///
/// # Safety
/// `session` came from [`rgsaddle_landscape_create`] and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_landscape_free(session: *mut RgsaddleLandscape) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

#[cfg(test)]
mod layout_tests {
    use super::{
        RGSADDLE_ABI_MINOR, RgsaddleLandscapeConfig, RgsaddleLandscapeReport, RgsaddleVersion,
        stamp,
    };
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn landscape_wire_layouts_match_c_header() {
        assert_eq!(size_of::<RgsaddleLandscapeConfig>(), 144);
        assert_eq!(align_of::<RgsaddleLandscapeConfig>(), 8);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, version), 0);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, flags), 8);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, duplicate_distance), 16);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, push), 24);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, max_nodes), 32);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, max_evaluations), 40);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, max_steps), 48);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, force_tol), 56);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, relax_force_tol), 64);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, max_move), 72);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, max_atom_move), 80);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, initial_step), 88);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, dr), 96);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, rotation_tol), 104);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, residual_tol), 112);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, max_rotations), 120);
        assert_eq!(
            offset_of!(RgsaddleLandscapeConfig, max_block_iterations),
            128
        );
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, method), 132);
        assert_eq!(offset_of!(RgsaddleLandscapeConfig, cell), 136);

        assert_eq!(size_of::<RgsaddleLandscapeReport>(), 48);
        assert_eq!(align_of::<RgsaddleLandscapeReport>(), 8);
        assert_eq!(offset_of!(RgsaddleLandscapeReport, version), 0);
        assert_eq!(offset_of!(RgsaddleLandscapeReport, flags), 8);
        assert_eq!(offset_of!(RgsaddleLandscapeReport, n_nodes), 16);
        assert_eq!(offset_of!(RgsaddleLandscapeReport, n_edges), 24);
        assert_eq!(offset_of!(RgsaddleLandscapeReport, bound), 32);
        assert_eq!(offset_of!(RgsaddleLandscapeReport, evaluations), 40);
    }

    #[test]
    fn a_stamped_report_has_zero_flags() {
        let mut report = RgsaddleLandscapeReport {
            version: RgsaddleVersion { major: 0, minor: 0 },
            flags: u64::MAX,
            n_nodes: 0,
            n_edges: 0,
            bound: 0,
            evaluations: 0,
        };
        stamp(&mut report);
        assert_eq!(report.flags, 0);
        assert_eq!(report.version.minor, RGSADDLE_ABI_MINOR);
    }
}

#[cfg(test)]
mod guard_tests {
    use super::{
        RGSADDLE_OK, RgsaddleLandscape, rgsaddle_landscape_config_default,
        rgsaddle_landscape_create, rgsaddle_landscape_free, rgsaddle_landscape_modes,
    };
    use crate::landscape::{CriticalPoint, LandscapeConfig, LandscapeSession, PathwayMap};
    use ndarray::{Array1, Array2};

    #[test]
    fn create_rejects_nonzero_reserved_flags() {
        let mut config = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { rgsaddle_landscape_config_default(&mut config) },
            RGSADDLE_OK
        );
        assert_eq!(config.flags, 0);
        let position = [0.0_f64; 3];
        let session = unsafe {
            rgsaddle_landscape_create(&config, 1, position.as_ptr(), std::ptr::null(), 0)
        };
        assert!(!session.is_null());
        unsafe { rgsaddle_landscape_free(session) };
        config.flags = 1;
        let rejected = unsafe {
            rgsaddle_landscape_create(&config, 1, position.as_ptr(), std::ptr::null(), 0)
        };
        assert!(rejected.is_null());
    }

    #[test]
    fn modes_of_a_minimum_leave_a_non_null_pointer_unread() {
        let session = LandscapeSession::new(LandscapeConfig::default(), Array1::zeros(3), None)
            .expect("default landscape");
        let map = PathwayMap {
            nodes: vec![CriticalPoint {
                position: Array1::zeros(3),
                energy: 0.0,
                index: 0,
                curvatures: vec![1.0, 2.0, 3.0],
                modes: Array2::zeros((3, 0)),
                evaluations: 1,
            }],
            edges: Vec::new(),
            evaluations: 1,
        };
        let session = Box::into_raw(Box::new(RgsaddleLandscape {
            session,
            n_atoms: 1,
            map: Some(map),
        }));
        let mut columns = -1;
        let dangling = std::ptr::dangling_mut();
        let status = unsafe { rgsaddle_landscape_modes(session, 0, dangling, &mut columns) };
        assert_eq!(status, RGSADDLE_OK);
        assert_eq!(columns, 0);
        unsafe { rgsaddle_landscape_free(session) };
    }
}
