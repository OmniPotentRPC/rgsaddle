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
    /// Number of lowest modes `k`, the saddle index sought; `k >= 1` and at
    /// most `3 * n_atoms`.
    pub index: i32,
    /// Estimator, a `rgsaddle_minmode_t`. `0` is the dimer with a LOBPCG block
    /// for `k > 1`; `1` is Lanczos with a block Lanczos iteration for `k > 1`;
    /// `2` is Lanczos for `k = 1` and LOBPCG for `k > 1`. Any other value is
    /// refused when the session is created.
    pub method: i32,
    /// Dimer separation, or finite-difference displacement of the Hessian
    /// action, in Angstrom.
    pub dr: f64,
    /// Rotation stop. For the dimer this is the rotational force in
    /// eV/Angstrom^2. For Lanczos it is dimensionless: the lowest Ritz
    /// residual divided by the absolute Ritz value, and, with no host
    /// relative tolerance, the relative change of that value. A Ritz
    /// value near zero compares the residual with this bound directly.
    /// The core crate's default is 0.0001.
    pub rotation_tol: f64,
    /// Cap on dimer rotations, or Lanczos actions, in one step; zero or more.
    pub max_rotations: i64,
    /// Force stop in eV/Angstrom: the largest force component under which a
    /// point with every curvature negative has converged.
    pub force_tol: f64,
    /// Largest per-atom translation of one step, in Angstrom.
    pub max_move: f64,
    /// Residual of the block eigenproblem in eV/Angstrom^2, used when `k > 1`.
    pub residual_tol: f64,
    /// Cap on iterations of the block eigensolver, positive, used when `k > 1`.
    pub max_block_iterations: i32,
}

/// Opaque index-k session: the position, the `k` lowest modes and the last
/// translated force of a saddle search of index `k`.
///
/// Create it with [`rgsaddle_indexk_create`] and release it with
/// [`rgsaddle_indexk_free`]. The session copies what it is given and calls the
/// host's functions only inside [`rgsaddle_indexk_step`]. Do not use one
/// session from two threads at once.
pub struct RgsaddleIndexK {
    session: IndexKSession,
    n_atoms: i64,
    k: usize,
}

/// Geometry-aware Hessian action, `rgsaddle_hessian_at_fn`: the callback
/// receives `(user, n_dof, position, vector, action)`.
pub type RgsaddleHessianAtFn =
    extern "C" fn(*mut c_void, i64, *const f64, *const f64, *mut f64) -> i32;

#[derive(Clone, Copy)]
pub(crate) enum HostHessian {
    Vector(RgsaddleHessianFn),
    AtPosition(RgsaddleHessianAtFn),
}

pub(crate) struct HostSurface {
    function: RgsaddleSurfaceFn,
    user: *mut c_void,
    n_atoms: i64,
    hessian: Option<HostHessian>,
}

impl HostSurface {
    pub(crate) fn new(
        function: RgsaddleSurfaceFn,
        user: *mut c_void,
        n_atoms: i64,
        hessian: Option<HostHessian>,
    ) -> Self {
        Self {
            function,
            user,
            n_atoms,
            hessian,
        }
    }
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
        x: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let Some(hessian) = self.hessian else {
            return Ok(None);
        };
        if x.len() != vector.len() {
            return Err(SaddleError::Shape(
                "hessian vector must match the position".into(),
            ));
        }
        let flat = vector.to_vec();
        let mut action = vec![0.0; flat.len()];
        let rc = match hessian {
            HostHessian::Vector(callback) => callback(
                self.user,
                flat.len() as i64,
                flat.as_ptr(),
                action.as_mut_ptr(),
            ),
            HostHessian::AtPosition(callback) => {
                let position = x.to_vec();
                callback(
                    self.user,
                    flat.len() as i64,
                    position.as_ptr(),
                    flat.as_ptr(),
                    action.as_mut_ptr(),
                )
            }
        };
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
    let search = MinModeConfig::default()
        .with_kind(kind)
        .with_dr(config.dr)
        .with_rotation_tol(config.rotation_tol)
        .with_max_rotations(config.max_rotations as usize)
        .with_force_tol(config.force_tol)
        .with_max_move(config.max_move);
    Ok((search, block))
}

pub(crate) fn read_modes(
    n: usize,
    k: usize,
    modes: *const f64,
) -> Result<Option<Array2<f64>>, i32> {
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

pub(crate) fn write_modes(modes: ndarray::ArrayView2<f64>, out: &mut [f64]) {
    let (rows, cols) = modes.dim();
    for row in 0..rows {
        for column in 0..cols {
            out[row * cols + column] = modes[(row, column)];
        }
    }
}

/// Create an index-k session at `position`.
///
/// `position` holds the `3 * n_atoms` coordinates in Angstrom, atom-major
/// (`x0 y0 z0 x1 ...`). `modes` is NULL to seed the modes from the
/// coordinates, or a row-major `(3 * n_atoms)` by `config.index` block of
/// starting directions. Both are copied. The index `k` is `config.index`.
///
/// Returns NULL, with no other diagnostic, for a NULL `config` or `position`,
/// `n_atoms < 1`, a config of another ABI major, `index < 1` or above
/// `3 * n_atoms`, a negative `max_rotations`, a non-positive
/// `max_block_iterations`, an unknown `method`, or seed modes the core crate
/// rejects. Free the result with [`rgsaddle_indexk_free`].
///
/// # Safety
/// `config` is valid for reads. `position` holds `3 * n_atoms` doubles. `modes`
/// is NULL or row-major `(3 * n_atoms)` by `config.index`.
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
    let options = IndexKConfig::new(k, search)
        .with_block(block)
        .with_residual_tol(cfg.residual_tol)
        .with_max_block_iterations(cfg.max_block_iterations as usize);
    match IndexKSession::new(options, x, modes) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleIndexK {
            session,
            n_atoms,
            k,
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// One index-k step: refresh the `k` lowest modes, translate the force and
/// move the geometry.
///
/// `surface` is called with one geometry (`n_images = 1`, `image = -1`) and
/// writes the energy in eV and the `3 * n_atoms` gradient in eV/Angstrom.
/// `hessian`, when not NULL, is called with the Hessian-vector product request
/// `(user, n, vector, out)` and writes the product `H v` into `out`, in
/// eV/Angstrom^2 for a unit `vector`; NULL estimates the action by finite
/// differences of `surface`. A non-zero return from either callback fails the
/// step with `RGSADDLE_SURFACE_FAILED`.
///
/// On success `out` receives `status` 1 when converged and 0 otherwise,
/// `evaluations`, the largest force component `max_force` in eV/Angstrom, the
/// lowest of the `k` curvatures in eV/Angstrom^2 as `curvature`, `rotations`,
/// `iteration`, and `ci_index` of -1. Read the modes with
/// [`rgsaddle_indexk_modes`] and the translated force with
/// [`rgsaddle_indexk_translated`].
///
/// Returns `RGSADDLE_NULL_SESSION`, `RGSADDLE_NULL_REPORT` or
/// `RGSADDLE_NULL_SURFACE` for a NULL `session`, `out` or `surface`, and
/// otherwise the status of the core error (shape, surface, non-finite, solver,
/// invalid, stalled).
///
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
    unsafe {
        step_with(
            session,
            surface,
            hessian.map(HostHessian::Vector),
            user,
            out,
        )
    }
}

/// One index-k step with a geometry-aware Hessian action, ABI minor 68.
///
/// Identical to [`rgsaddle_indexk_step`] except that `hessian`, when not
/// NULL, is called with `(user, n, position, vector, out)`. `position` is
/// the geometry at which the action is requested and holds `n` doubles.
///
/// # Safety
/// `session` comes from [`rgsaddle_indexk_create`]. `out` is writable.
/// `surface` and `hessian` are called with `user` and must not unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_step_at(
    session: *mut RgsaddleIndexK,
    surface: Option<RgsaddleSurfaceFn>,
    hessian: Option<RgsaddleHessianAtFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    unsafe {
        step_with(
            session,
            surface,
            hessian.map(HostHessian::AtPosition),
            user,
            out,
        )
    }
}

unsafe fn step_with(
    session: *mut RgsaddleIndexK,
    surface: Option<RgsaddleSurfaceFn>,
    hessian: Option<HostHessian>,
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
    let host = HostSurface::new(function, user, session.n_atoms, hessian);
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

/// Copy the current geometry, `3 * n_atoms` doubles in Angstrom, atom-major,
/// into `out`.
///
/// Returns `RGSADDLE_NULL_SESSION` for a NULL `session` and
/// `RGSADDLE_INVALID_PARAMETER` for a NULL `out`.
///
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

/// Copy the `k` modes of the last step and their curvatures.
///
/// `modes` receives a row-major `(3 * n_atoms)` by `k` block, so mode `j` is
/// column `j`; the columns are orthonormal. `curvatures` receives `k` values in
/// eV/Angstrom^2, one per column. Before the first step the modes are the seed
/// and the curvatures are NaN.
///
/// Returns `RGSADDLE_NULL_SESSION` for a NULL `session` and
/// `RGSADDLE_INVALID_PARAMETER` for a NULL `modes` or `curvatures`.
///
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

/// Copy the translated force of the last step, `3 * n_atoms` doubles in
/// eV/Angstrom, into `out`.
///
/// This is the effective gradient the translation stepper used: the reflection
/// `F - 2 sum_i v_i (v_i . F)` when every curvature is negative, and
/// `-sum_i v_i (v_i . F)` otherwise, with `F` the force.
///
/// Returns `RGSADDLE_NULL_SESSION` for a NULL `session`,
/// `RGSADDLE_INVALID_PARAMETER` for a NULL `out`, and `RGSADDLE_NO_EVALUATION`
/// before the first successful step.
///
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

/// Move the session to `position`, `3 * n_atoms` doubles in Angstrom.
///
/// `gradient` is NULL, or the energy gradient at `position` in eV/Angstrom, the
/// same length; the session then skips the evaluation it would make there.
///
/// Returns `RGSADDLE_NULL_SESSION` for a NULL `session`,
/// `RGSADDLE_INVALID_PARAMETER` for a NULL `position`, and the status of the
/// core error when it refuses the move.
///
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

/// Release a session. NULL is accepted and ignored.
///
/// # Safety
/// `session` comes from [`rgsaddle_indexk_create`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_indexk_free(session: *mut RgsaddleIndexK) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// Householder reflection of `force` through the columns of `modes`,
/// `F - 2 sum_i v_i (v_i . F)`.
///
/// `n` is the number of coordinates and `k` the number of modes. `force` holds
/// `n` doubles in eV/Angstrom; `modes` is an orthonormal row-major `n` by `k`
/// block; `out` receives the `n` reflected components in the units of `force`.
///
/// Returns `RGSADDLE_INVALID_PARAMETER` for `n < 1`, `k < 1`, a NULL pointer,
/// an `out` equal to `force`, or an `n * k` that overflows, and the status of the
/// core error for a non-finite `force`. The orthonormality of `modes` is not
/// checked.
///
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Seen {
        position: Vec<f64>,
        vector: Vec<f64>,
    }

    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);

    extern "C" fn idle_surface(_user: *mut c_void, _request: *mut RgsaddleSurfaceRequest) -> i32 {
        RGSADDLE_OK
    }

    extern "C" fn recording_hessian(
        _user: *mut c_void,
        n: i64,
        position: *const f64,
        vector: *const f64,
        action: *mut f64,
    ) -> i32 {
        if position.is_null() || vector.is_null() || action.is_null() || n != 3 {
            return -1;
        }
        let count = n as usize;
        let position = unsafe { slice::from_raw_parts(position, count) }.to_vec();
        let vector = unsafe { slice::from_raw_parts(vector, count) }.to_vec();
        for (slot, value) in unsafe { slice::from_raw_parts_mut(action, count) }
            .iter_mut()
            .zip(vector.iter())
        {
            *slot = -value;
        }
        *SEEN.lock().expect("hessian record") = Some(Seen { position, vector });
        0
    }

    #[test]
    fn host_hessian_receives_the_position() {
        *SEEN.lock().expect("clear") = None;
        let host = HostSurface::new(
            idle_surface,
            std::ptr::null_mut(),
            1,
            Some(HostHessian::AtPosition(recording_hessian)),
        );
        let position = Array1::from_vec(vec![0.2, -0.4, 0.6]);
        let vector = Array1::from_vec(vec![1.0, 0.0, -1.0]);
        let action = host
            .hessian_vector(position.view(), vector.view())
            .expect("callback")
            .expect("analytic action");
        assert_eq!(action, Array1::from_vec(vec![-1.0, 0.0, 1.0]));
        let seen = SEEN
            .lock()
            .expect("read")
            .take()
            .expect("position was recorded");
        assert_eq!(seen.position, position.to_vec());
        assert_eq!(seen.vector, vector.to_vec());
        assert_ne!(seen.position, seen.vector);
    }

    extern "C" fn vector_only_hessian(
        _user: *mut c_void,
        n: i64,
        vector: *const f64,
        action: *mut f64,
    ) -> i32 {
        if vector.is_null() || action.is_null() || n != 3 {
            return -1;
        }
        let count = n as usize;
        let vector = unsafe { slice::from_raw_parts(vector, count) };
        for (slot, value) in unsafe { slice::from_raw_parts_mut(action, count) }
            .iter_mut()
            .zip(vector.iter())
        {
            *slot = 2.0 * value;
        }
        0
    }

    #[test]
    fn the_original_hessian_callback_gets_the_vector_alone() {
        let host = HostSurface::new(
            idle_surface,
            std::ptr::null_mut(),
            1,
            Some(HostHessian::Vector(vector_only_hessian)),
        );
        let position = Array1::from_vec(vec![0.2, -0.4, 0.6]);
        let vector = Array1::from_vec(vec![1.0, 0.0, -1.0]);
        let action = host
            .hessian_vector(position.view(), vector.view())
            .expect("callback")
            .expect("analytic action");
        assert_eq!(action, Array1::from_vec(vec![2.0, 0.0, -2.0]));
    }
}
