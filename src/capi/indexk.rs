//! Index-k C entry points. Layouts live in `include/rgsaddle.h`.

use std::ffi::c_void;
use std::slice;

use ndarray::{Array1, Array2, ArrayView1};

use super::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NO_EVALUATION,
    RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SESSION, RGSADDLE_NULL_SURFACE, RGSADDLE_OK,
    RgsaddleReport, RgsaddleSurfaceFn, RgsaddleSurfaceRequest, RgsaddleVersion, stamp_report,
    status_of,
};
use crate::capi_kappa::RgsaddleHessianFn;
use crate::error::SaddleError;
use crate::indexk::{BlockEigenKind, IndexKConfig, IndexKSession};
use crate::minmode::{MinModeConfig, MinModeKind, MinModeStatus, PointSurface};

#[repr(C)]
/// Index-k minimum-mode configuration. The layout matches `rgsaddle_indexk_config_t`.
pub struct RgsaddleIndexKConfig {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. Reserved; write zero.
    pub flags: u64,
    /// Number of lowest modes, `k >= 1`.
    pub index: i32,
    /// Estimator: dimer, Lanczos, or block LOBPCG.
    pub method: i32,
    /// Finite-difference displacement.
    pub dr: f64,
    /// Rotation or Ritz residual.
    pub rotation_tol: f64,
    /// Cap on dimer rotations.
    pub max_rotations: i64,
    /// Force norm at which a negative-curvature point has converged.
    pub force_tol: f64,
    /// Largest per-atom translation.
    pub max_move: f64,
    /// Residual of the block eigenproblem when `k > 1`.
    pub residual_tol: f64,
    /// Cap on block iterations when `k > 1`.
    pub max_block_iterations: i32,
}

/// Opaque index-k session. Create it with [`rgsaddle_indexk_create`].
pub struct RgsaddleIndexK {
    session: IndexKSession,
    n_atoms: i64,
    k: usize,
}

struct HostSurface {
    function: RgsaddleSurfaceFn,
    user: *mut c_void,
    n_atoms: i64,
    hessian: Option<RgsaddleHessianFn>,
}

unsafe impl Send for HostSurface {}
unsafe impl Sync for HostSurface {}

impl PointSurface for HostSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let flat = x.to_vec();
        let mut energy = [0.0];
        let mut gradient = vec![0.0; flat.len()];
        let mut request = RgsaddleSurfaceRequest {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            n_images: 1,
            n_atoms: self.n_atoms,
            positions: flat.as_ptr(),
            energies: energy.as_mut_ptr(),
            gradients: gradient.as_mut_ptr(),
            image: -1,
        };
        let rc = (self.function)(self.user, &mut request);
        if rc != 0 {
            return Err(SaddleError::Surface(format!("host callback rc={rc}")));
        }
        Ok((energy[0], Array1::from(gradient)))
    }

    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let Some(hessian) = self.hessian else {
            return Ok(None);
        };
        let mut action = vec![0.0; vector.len()];
        let rc = hessian(
            self.user,
            vector.len() as i64,
            vector.as_ptr(),
            action.as_mut_ptr(),
        );
        if rc != 0 {
            return Err(SaddleError::Surface(format!("hessian callback rc={rc}")));
        }
        Ok(Some(Array1::from(action)))
    }
}

fn search_of(config: &RgsaddleIndexKConfig) -> Result<(MinModeConfig, BlockEigenKind), i32> {
    if config.version.major != RGSADDLE_ABI_MAJOR {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    if config.index < 1 || config.max_rotations < 0 || config.max_block_iterations <= 0 {
        return Err(RGSADDLE_INVALID_PARAMETER);
    }
    let (kind, block) = match config.method {
        0 => (MinModeKind::Dimer, BlockEigenKind::Lobpcg),
        1 => (MinModeKind::Lanczos, BlockEigenKind::BlockLanczos),
        2 => (MinModeKind::Lanczos, BlockEigenKind::Lobpcg),
        _ => return Err(RGSADDLE_INVALID_PARAMETER),
    };
    let mut search = MinModeConfig {
        kind,
        dr: config.dr,
        rotation_tol: config.rotation_tol,
        max_rotations: config.max_rotations as usize,
        force_tol: config.force_tol,
        max_move: config.max_move,
        ..MinModeConfig::default()
    };
    search.kind = kind;
    Ok((search, block))
}

fn read_modes(n: usize, k: usize, modes: *const f64) -> Result<Option<Array2<f64>>, i32> {
    if modes.is_null() {
        return Ok(None);
    }
    let count = n.checked_mul(k).ok_or(RGSADDLE_INVALID_PARAMETER)?;
    let raw = unsafe { slice::from_raw_parts(modes, count) };
    let mut block = Array2::zeros((n, k));
    for row in 0..n {
        for column in 0..k {
            block[(row, column)] = raw[row * k + column];
        }
    }
    Ok(Some(block))
}

fn write_modes(modes: ndarray::ArrayView2<f64>, out: &mut [f64]) {
    let (rows, cols) = modes.dim();
    for row in 0..rows {
        for column in 0..cols {
            out[row * cols + column] = modes[(row, column)];
        }
    }
}

/// # Safety
/// `position` holds `3 * n_atoms` doubles. `modes` is NULL or row-major
/// `(3 * n_atoms)` by `config.index`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_create(
    config: *const RgsaddleIndexKConfig,
    n_atoms: i64,
    position: *const f64,
    modes: *const f64,
) -> *mut RgsaddleIndexK {
    if config.is_null() || position.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    let Ok((search, block)) = search_of(cfg) else {
        return std::ptr::null_mut();
    };
    let k = cfg.index as usize;
    let n = (3 * n_atoms) as usize;
    if k > n {
        return std::ptr::null_mut();
    }
    let x = Array1::from(unsafe { slice::from_raw_parts(position, n) }.to_vec());
    let modes = match read_modes(n, k, modes) {
        Ok(modes) => modes,
        Err(_) => return std::ptr::null_mut(),
    };
    let mut options = IndexKConfig::new(k, search);
    options.block = block;
    options.residual_tol = cfg.residual_tol;
    options.max_block_iterations = cfg.max_block_iterations as usize;
    match IndexKSession::new(options, x, modes) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleIndexK {
            session,
            n_atoms,
            k,
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` comes from [`rgsaddle_indexk_create`]. `out` is writable.
/// `surface` and `hessian` are called with `user` and must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_step(
    session: *mut RgsaddleIndexK,
    surface: Option<RgsaddleSurfaceFn>,
    hessian: Option<RgsaddleHessianFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
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
    let host = HostSurface {
        function,
        user,
        n_atoms: session.n_atoms,
        hessian,
    };
    match session.session.step(&host) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == MinModeStatus::Converged {
                1
            } else {
                0
            };
            out.evaluations = i32::try_from(report.evaluations).unwrap_or(i32::MAX);
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = report.iteration as i64;
            out.curvature = report.curvatures.first().copied().unwrap_or(f64::NAN);
            out.rotations = report.rotations as i64;
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}

/// # Safety
/// `out` holds `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_position(
    session: *const RgsaddleIndexK,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dst = unsafe { slice::from_raw_parts_mut(out, (3 * session.n_atoms) as usize) };
    for (slot, value) in dst.iter_mut().zip(session.session.position().iter()) {
        *slot = *value;
    }
    RGSADDLE_OK
}

/// # Safety
/// `modes` holds `(3 * n_atoms) * k` doubles, row-major. `curvatures` holds `k`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_modes(
    session: *const RgsaddleIndexK,
    modes: *mut f64,
    curvatures: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if modes.is_null() || curvatures.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let n = (3 * session.n_atoms) as usize;
    let mode_out = unsafe { slice::from_raw_parts_mut(modes, n * session.k) };
    let curve_out = unsafe { slice::from_raw_parts_mut(curvatures, session.k) };
    write_modes(session.session.modes(), mode_out);
    for (slot, value) in curve_out
        .iter_mut()
        .zip(session.session.curvatures().iter())
    {
        *slot = *value;
    }
    RGSADDLE_OK
}

/// # Safety
/// `out` holds `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_translated(
    session: *const RgsaddleIndexK,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let force = session.session.translated_force();
    if force.is_empty() {
        return RGSADDLE_NO_EVALUATION;
    }
    let dst = unsafe { slice::from_raw_parts_mut(out, force.len()) };
    for (slot, value) in dst.iter_mut().zip(force.iter()) {
        *slot = *value;
    }
    RGSADDLE_OK
}

/// # Safety
/// `position` holds `3 * n_atoms` doubles. `gradient` is NULL or the same length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_set_position(
    session: *mut RgsaddleIndexK,
    position: *const f64,
    gradient: *const f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if position.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &mut *session };
    let n = (3 * session.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, n) }.to_vec());
    let gradient = if gradient.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(gradient, n) }.to_vec(),
        ))
    };
    match session.session.set_position(x, gradient) {
        Ok(()) => RGSADDLE_OK,
        Err(err) => status_of(&err),
    }
}

/// # Safety
/// `session` comes from [`rgsaddle_indexk_create`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_free(session: *mut RgsaddleIndexK) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// # Safety
/// `force` and `out` each hold `n` doubles and do not alias. `modes` is
/// row-major `n` by `k`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_reflect_force(
    n: i64,
    k: i64,
    force: *const f64,
    modes: *const f64,
    out: *mut f64,
) -> i32 {
    if n < 1 || k < 1 || force.is_null() || modes.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if std::ptr::eq(force, out as *const f64) {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let n = n as usize;
    let k = k as usize;
    let Some(count) = n.checked_mul(k) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let force_raw = unsafe { slice::from_raw_parts(force, n) };
    let mode_raw = unsafe { slice::from_raw_parts(modes, count) };
    let mut block = Array2::zeros((n, k));
    for row in 0..n {
        for column in 0..k {
            block[(row, column)] = mode_raw[row * k + column];
        }
    }
    match crate::indexk::reflect_force(ArrayView1::from(force_raw), block.view()) {
        Ok(reflected) => {
            let dst = unsafe { slice::from_raw_parts_mut(out, n) };
            dst.copy_from_slice(reflected.as_slice().expect("reflected force is contiguous"));
            RGSADDLE_OK
        }
        Err(err) => status_of(&err),
    }
}
