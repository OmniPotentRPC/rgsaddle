//! Opaque stencil handles. Point presets only.
//!
//! Band presets return [`crate::stencil::BAND_PRESET_SEAM`].
//! Every entry catches a panic and records [`rgsaddle_last_error`].

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::panic::AssertUnwindSafe;

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::stencil::{BAND_PRESET_SEAM, Preset, Stencil, StencilReport, StencilStatus};

use super::{
    RGSADDLE_INVALID_PARAMETER, RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SESSION, RGSADDLE_NULL_SURFACE,
    RGSADDLE_OK, RGSADDLE_SHAPE, RGSADDLE_SOLVER, status_of,
};

pub const RGSADDLE_STENCIL_KEY_DR: i32 = 1;
pub const RGSADDLE_STENCIL_KEY_FORCE_TOL: i32 = 2;
pub const RGSADDLE_STENCIL_KEY_MAX_MOVE: i32 = 3;
pub const RGSADDLE_STENCIL_KEY_ROTATION_TOL: i32 = 4;
pub const RGSADDLE_STENCIL_KEY_MAX_ROTATIONS: i32 = 5;
pub const RGSADDLE_STENCIL_KEY_KRYLOV_DIM: i32 = 6;
pub const RGSADDLE_STENCIL_KEY_THREADS: i32 = 7;
pub const RGSADDLE_STENCIL_KEY_ART_THRESHOLD: i32 = 8;
pub const RGSADDLE_STENCIL_KEY_ART_PUSH: i32 = 9;
pub const RGSADDLE_STENCIL_KEY_MODE: i32 = 10;

pub const RGSADDLE_STENCIL_QUERY_POSITIONS: i32 = 1;
pub const RGSADDLE_STENCIL_QUERY_MODE: i32 = 2;
pub const RGSADDLE_STENCIL_QUERY_CURVATURE: i32 = 3;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(c_string(""));
}

fn c_string(text: &str) -> CString {
    let bytes: Vec<u8> = text.bytes().filter(|byte| *byte != 0).collect();
    // The filter removed every interior nul.
    unsafe { CString::from_vec_unchecked(bytes) }
}

fn set_last_error(text: &str) {
    let owned = c_string(text);
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = owned;
    });
}

fn clear_last_error() {
    set_last_error("");
}

#[repr(C)]
pub struct RgsaddleStencilReport {
    pub struct_size: u64,
    pub status: i32,
    pub reserved: i32,
    pub max_force: f64,
    pub curvature: f64,
    pub iteration: u64,
    pub evaluations: u64,
}

struct Config {
    preset: i32,
    dr: f64,
    force_tol: f64,
    max_move: f64,
    rotation_tol: f64,
    max_rotations: usize,
    krylov_dim: usize,
    threads: usize,
    art_threshold: f64,
    art_push: usize,
    mode: Vec<f64>,
}

impl Config {
    fn new(preset: i32) -> Self {
        Self {
            preset,
            dr: 1e-3,
            force_tol: 1e-3,
            max_move: 0.2,
            rotation_tol: 1e-4,
            max_rotations: 20,
            krylov_dim: 12,
            threads: 1,
            art_threshold: 0.0,
            art_push: 20,
            mode: Vec::new(),
        }
    }
}

struct Handle {
    stencil: Stencil,
    dof: usize,
}

pub type StencilEvalFn = unsafe extern "C" fn(
    *mut c_void,
    usize,
    usize,
    *const u64,
    *const f64,
    *mut f64,
    *mut f64,
) -> i32;

struct CbSurface {
    eval: StencilEvalFn,
    user: *mut c_void,
}

// SAFETY: `PointSurface` requires `Sync`. This value stays inside
// `rgsaddle_stencil_step` and `rgsaddle_stencil_run`, and those calls
// invoke the callback on the thread that entered the C ABI. The chain
// pool never sees it. The raw `user` pointer is therefore not shared.
unsafe impl Sync for CbSurface {}

impl PointSurface for CbSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let Some(row) = x.as_slice() else {
            return Err(SaddleError::Shape(
                "stencil position is not contiguous".into(),
            ));
        };
        let dof = row.len();
        let mut energy = [0.0];
        let mut gradient = vec![0.0; dof];
        let ids = [0_u64];
        let code = unsafe {
            (self.eval)(
                self.user,
                1,
                dof,
                ids.as_ptr(),
                row.as_ptr(),
                energy.as_mut_ptr(),
                gradient.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(SaddleError::Surface(format!(
                "stencil callback status {code}"
            )));
        }
        if !energy[0].is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("stencil callback"));
        }
        Ok((energy[0], Array1::from_vec(gradient)))
    }
}

fn guard(body: impl FnOnce() -> i32) -> i32 {
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(_) => {
            set_last_error("stencil call panicked");
            RGSADDLE_SOLVER
        }
    }
}

fn guard_ptr<T>(body: impl FnOnce() -> *mut T) -> *mut T {
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("stencil call panicked");
            std::ptr::null_mut()
        }
    }
}

fn fail(err: &SaddleError) -> i32 {
    set_last_error(&err.to_string());
    status_of(err)
}

fn apply(config: &Config, x: Array1<f64>) -> Result<Stencil, SaddleError> {
    let Some(preset) = Preset::from_abi(config.preset) else {
        return Err(SaddleError::Invalid("unknown stencil preset".into()));
    };
    if !preset.is_point() {
        return Err(SaddleError::Invalid(BAND_PRESET_SEAM.into()));
    }
    let mut builder = Stencil::point(x)
        .preset(preset)
        .dr(config.dr)
        .force_tol(config.force_tol)
        .max_move(config.max_move)
        .rotation_tol(config.rotation_tol)
        .max_rotations(config.max_rotations)
        .krylov_dim(config.krylov_dim)
        .threads(config.threads)
        .art_threshold(config.art_threshold)
        .art_push_steps(config.art_push);
    if !config.mode.is_empty() {
        builder = builder.mode(Array1::from_vec(config.mode.clone()));
    }
    builder.build()
}

fn write_report(out: *mut RgsaddleStencilReport, report: &StencilReport) -> i32 {
    if out.is_null() {
        set_last_error("stencil report is null");
        return RGSADDLE_NULL_REPORT;
    }
    let full = std::mem::size_of::<RgsaddleStencilReport>() as u64;
    let given = unsafe { (*out).struct_size };
    if given != 0 && given < full {
        set_last_error("stencil report struct_size is short");
        return RGSADDLE_INVALID_PARAMETER;
    }
    unsafe {
        (*out).struct_size = full;
        (*out).status = match report.status {
            StencilStatus::Running => 0,
            StencilStatus::Converged => 1,
        };
        (*out).reserved = 0;
        (*out).max_force = report.max_force;
        (*out).curvature = report.curvature;
        (*out).iteration = report.iteration as u64;
        (*out).evaluations = report.evaluations as u64;
    }
    RGSADDLE_OK
}

/// # Safety
/// The returned pointer is valid until the next stencil call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// # Safety
/// The pointer is opaque. `preset` is a `Preset::to_abi` value.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_stencil_config_new(preset: i32) -> *mut c_void {
    guard_ptr(|| {
        if Preset::from_abi(preset).is_none() {
            set_last_error("unknown stencil preset");
            return std::ptr::null_mut();
        }
        Box::into_raw(Box::new(Config::new(preset))).cast()
    })
}

/// # Safety
/// `config` comes from [`rgsaddle_stencil_config_new`] or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_set_f64(
    config: *mut c_void,
    key: i32,
    value: f64,
) -> i32 {
    guard(|| {
        if config.is_null() {
            set_last_error("stencil config is null");
            return RGSADDLE_NULL_SESSION;
        }
        let config = unsafe { &mut *(config as *mut Config) };
        match key {
            RGSADDLE_STENCIL_KEY_DR => config.dr = value,
            RGSADDLE_STENCIL_KEY_FORCE_TOL => config.force_tol = value,
            RGSADDLE_STENCIL_KEY_MAX_MOVE => config.max_move = value,
            RGSADDLE_STENCIL_KEY_ROTATION_TOL => config.rotation_tol = value,
            RGSADDLE_STENCIL_KEY_ART_THRESHOLD => config.art_threshold = value,
            RGSADDLE_STENCIL_KEY_MAX_ROTATIONS
            | RGSADDLE_STENCIL_KEY_KRYLOV_DIM
            | RGSADDLE_STENCIL_KEY_THREADS
            | RGSADDLE_STENCIL_KEY_ART_PUSH
            | RGSADDLE_STENCIL_KEY_MODE => {
                set_last_error("stencil key is not an f64");
                return RGSADDLE_INVALID_PARAMETER;
            }
            _ => {
                set_last_error("unknown stencil key");
                return RGSADDLE_INVALID_PARAMETER;
            }
        }
        RGSADDLE_OK
    })
}

/// # Safety
/// `config` comes from [`rgsaddle_stencil_config_new`] or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_set_i64(
    config: *mut c_void,
    key: i32,
    value: i64,
) -> i32 {
    guard(|| {
        if config.is_null() {
            set_last_error("stencil config is null");
            return RGSADDLE_NULL_SESSION;
        }
        if value < 0 {
            set_last_error("stencil integer key is negative");
            return RGSADDLE_INVALID_PARAMETER;
        }
        let config = unsafe { &mut *(config as *mut Config) };
        let value = value as usize;
        match key {
            RGSADDLE_STENCIL_KEY_MAX_ROTATIONS => config.max_rotations = value,
            RGSADDLE_STENCIL_KEY_KRYLOV_DIM => config.krylov_dim = value,
            RGSADDLE_STENCIL_KEY_THREADS => config.threads = value,
            RGSADDLE_STENCIL_KEY_ART_PUSH => config.art_push = value,
            RGSADDLE_STENCIL_KEY_DR
            | RGSADDLE_STENCIL_KEY_FORCE_TOL
            | RGSADDLE_STENCIL_KEY_MAX_MOVE
            | RGSADDLE_STENCIL_KEY_ROTATION_TOL
            | RGSADDLE_STENCIL_KEY_ART_THRESHOLD
            | RGSADDLE_STENCIL_KEY_MODE => {
                set_last_error("stencil key is not an i64");
                return RGSADDLE_INVALID_PARAMETER;
            }
            _ => {
                set_last_error("unknown stencil key");
                return RGSADDLE_INVALID_PARAMETER;
            }
        }
        RGSADDLE_OK
    })
}

/// # Safety
/// `config` comes from [`rgsaddle_stencil_config_new`] or is null.
/// `values` points at `n` finite f64s when `n > 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_set_array(
    config: *mut c_void,
    key: i32,
    values: *const f64,
    n: usize,
) -> i32 {
    guard(|| {
        if config.is_null() {
            set_last_error("stencil config is null");
            return RGSADDLE_NULL_SESSION;
        }
        if key != RGSADDLE_STENCIL_KEY_MODE {
            set_last_error("unknown stencil key");
            return RGSADDLE_INVALID_PARAMETER;
        }
        if n > 0 && values.is_null() {
            set_last_error("stencil mode pointer is null");
            return RGSADDLE_INVALID_PARAMETER;
        }
        let mode = if n == 0 {
            Vec::new()
        } else {
            let slice = unsafe { std::slice::from_raw_parts(values, n) };
            if slice.iter().any(|value| !value.is_finite()) {
                set_last_error("non-finite value in mode");
                return super::RGSADDLE_NON_FINITE;
            }
            slice.to_vec()
        };
        unsafe { (*(config as *mut Config)).mode = mode };
        RGSADDLE_OK
    })
}

/// # Safety
/// `config` comes from [`rgsaddle_stencil_config_new`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_config_free(config: *mut c_void) {
    if config.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(config as *mut Config) });
    }));
}

/// # Safety
/// `config` is live. `x` points at `rows * dof` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_new(
    config: *const c_void,
    rows: usize,
    dof: usize,
    x: *const f64,
) -> *mut c_void {
    guard_ptr(|| {
        if config.is_null() {
            set_last_error("stencil config is null");
            return std::ptr::null_mut();
        }
        let config_ref = unsafe { &*(config as *const Config) };
        if Preset::from_abi(config_ref.preset).is_some_and(|preset| !preset.is_point()) {
            set_last_error(BAND_PRESET_SEAM);
            return std::ptr::null_mut();
        }
        if rows != 1 {
            set_last_error("point presets take one row");
            return std::ptr::null_mut();
        }
        if dof == 0 || x.is_null() {
            set_last_error("stencil position is empty");
            return std::ptr::null_mut();
        }
        let values = unsafe { std::slice::from_raw_parts(x, dof) };
        let position = Array1::from_vec(values.to_vec());
        match apply(config_ref, position) {
            Ok(stencil) => Box::into_raw(Box::new(Handle { stencil, dof })).cast(),
            Err(err) => {
                let _ = fail(&err);
                std::ptr::null_mut()
            }
        }
    })
}

/// # Safety
/// `stencil` is live or null. `eval` is a callback the host keeps valid
/// for the call. `out` is a report whose `struct_size` is 0 or the full size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_step(
    stencil: *mut c_void,
    eval: Option<StencilEvalFn>,
    user: *mut c_void,
    out: *mut RgsaddleStencilReport,
) -> i32 {
    guard(|| {
        if stencil.is_null() {
            set_last_error("stencil is null");
            return RGSADDLE_NULL_SESSION;
        }
        let Some(eval) = eval else {
            set_last_error("stencil callback is null");
            return RGSADDLE_NULL_SURFACE;
        };
        let surface = CbSurface { eval, user };
        let stencil = unsafe { &mut *(stencil as *mut Handle) };
        match stencil.stencil.step(&surface) {
            Ok(report) => write_report(out, &report),
            Err(err) => fail(&err),
        }
    })
}

/// # Safety
/// Same as [`rgsaddle_stencil_step`]. `max_steps` is the step cap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_run(
    stencil: *mut c_void,
    eval: Option<StencilEvalFn>,
    user: *mut c_void,
    max_steps: usize,
    out: *mut RgsaddleStencilReport,
) -> i32 {
    guard(|| {
        if stencil.is_null() {
            set_last_error("stencil is null");
            return RGSADDLE_NULL_SESSION;
        }
        let Some(eval) = eval else {
            set_last_error("stencil callback is null");
            return RGSADDLE_NULL_SURFACE;
        };
        let surface = CbSurface { eval, user };
        let stencil = unsafe { &mut *(stencil as *mut Handle) };
        match stencil.stencil.run(&surface, max_steps) {
            Ok(report) => write_report(out, &report),
            Err(err) => fail(&err),
        }
    })
}

/// # Safety
/// `stencil` is live or null. `out` has room for `n` values, with `n` equal
/// to the degree of freedom.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_positions(
    stencil: *const c_void,
    out: *mut f64,
    n: usize,
) -> i32 {
    guard(|| {
        if stencil.is_null() {
            set_last_error("stencil is null");
            return RGSADDLE_NULL_SESSION;
        }
        let stencil = unsafe { &*(stencil as *const Handle) };
        if n != stencil.dof || out.is_null() {
            set_last_error("stencil position length does not match");
            return RGSADDLE_SHAPE;
        }
        let position = stencil.stencil.position();
        let Some(slice) = position.as_slice() else {
            set_last_error("stencil position is not contiguous");
            return RGSADDLE_SHAPE;
        };
        unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), out, n) };
        RGSADDLE_OK
    })
}

/// # Safety
/// `stencil` is live or null. `out` has room for `n` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_query(
    stencil: *const c_void,
    key: i32,
    out: *mut f64,
    n: usize,
) -> i32 {
    guard(|| {
        if stencil.is_null() {
            set_last_error("stencil is null");
            return RGSADDLE_NULL_SESSION;
        }
        if out.is_null() && n > 0 {
            set_last_error("stencil query pointer is null");
            return RGSADDLE_INVALID_PARAMETER;
        }
        let stencil = unsafe { &*(stencil as *const Handle) };
        match key {
            RGSADDLE_STENCIL_QUERY_POSITIONS => {
                if n != stencil.dof {
                    set_last_error("stencil position length does not match");
                    return RGSADDLE_SHAPE;
                }
                let position = stencil.stencil.position();
                let Some(slice) = position.as_slice() else {
                    set_last_error("stencil position is not contiguous");
                    return RGSADDLE_SHAPE;
                };
                unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), out, n) };
                RGSADDLE_OK
            }
            RGSADDLE_STENCIL_QUERY_MODE => {
                let Some(mode) = stencil.stencil.mode() else {
                    set_last_error("minimiser has no mode");
                    return RGSADDLE_INVALID_PARAMETER;
                };
                if n != mode.len() {
                    set_last_error("stencil mode length does not match");
                    return RGSADDLE_SHAPE;
                }
                let Some(slice) = mode.as_slice() else {
                    set_last_error("stencil mode is not contiguous");
                    return RGSADDLE_SHAPE;
                };
                unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), out, n) };
                RGSADDLE_OK
            }
            RGSADDLE_STENCIL_QUERY_CURVATURE => {
                if n < 1 {
                    set_last_error("stencil curvature needs one value");
                    return RGSADDLE_SHAPE;
                }
                unsafe { *out = stencil.stencil.curvature() };
                RGSADDLE_OK
            }
            _ => {
                set_last_error("unknown stencil key");
                RGSADDLE_INVALID_PARAMETER
            }
        }
    })
}

/// # Safety
/// `stencil` comes from [`rgsaddle_stencil_new`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_stencil_free(stencil: *mut c_void) {
    if stencil.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(stencil as *mut Handle) });
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    unsafe extern "C" fn bowl(
        _user: *mut c_void,
        rows: usize,
        dof: usize,
        _ids: *const u64,
        x: *const f64,
        e: *mut f64,
        g: *mut f64,
    ) -> i32 {
        if rows != 1 || dof != 2 || x.is_null() || e.is_null() || g.is_null() {
            return -1;
        }
        let x0 = unsafe { *x };
        let y = unsafe { *x.add(1) };
        unsafe {
            *e = 2.0 * x0 * x0 + 5.0 * y * y;
            *g = 4.0 * x0;
            *g.add(1) = 10.0 * y;
        }
        0
    }

    #[test]
    fn unknown_keys_and_band_presets_fail_closed() {
        let config = rgsaddle_stencil_config_new(99);
        assert!(config.is_null());
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        assert!(text.to_string_lossy().contains("unknown"));

        let config = rgsaddle_stencil_config_new(Preset::CiNeb.to_abi());
        assert!(!config.is_null());
        let status = unsafe { rgsaddle_stencil_config_set_f64(config, 99, 1.0) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        assert!(text.to_string_lossy().contains("unknown stencil key"));

        let x = [0.2_f64, 0.0];
        let stencil = unsafe { rgsaddle_stencil_new(config, 1, 2, x.as_ptr()) };
        assert!(stencil.is_null());
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        assert!(text.to_string_lossy().contains(BAND_PRESET_SEAM));
        let images = [0.0_f64; 6];
        let stencil = unsafe { rgsaddle_stencil_new(config, 3, 2, images.as_ptr()) };
        assert!(stencil.is_null());
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        assert!(text.to_string_lossy().contains(BAND_PRESET_SEAM));
        unsafe { rgsaddle_stencil_config_free(config) };
    }

    #[test]
    fn a_point_dimer_steps_through_the_callback() {
        let config = rgsaddle_stencil_config_new(Preset::Dimer.to_abi());
        let mode = [1.0_f64, 0.0];
        let status = unsafe {
            rgsaddle_stencil_config_set_array(config, RGSADDLE_STENCIL_KEY_MODE, mode.as_ptr(), 2)
        };
        assert_eq!(status, RGSADDLE_OK);
        let x = [0.4_f64, 0.3];
        let stencil = unsafe { rgsaddle_stencil_new(config, 1, 2, x.as_ptr()) };
        assert!(!stencil.is_null());
        let mut report = RgsaddleStencilReport {
            struct_size: 0,
            status: -1,
            reserved: 0,
            max_force: 0.0,
            curvature: 0.0,
            iteration: 0,
            evaluations: 0,
        };
        let status = unsafe {
            rgsaddle_stencil_step(stencil, Some(bowl), std::ptr::null_mut(), &mut report)
        };
        assert_eq!(status, RGSADDLE_OK);
        let mut out = [0.0; 2];
        let status = unsafe { rgsaddle_stencil_positions(stencil, out.as_mut_ptr(), 2) };
        assert_eq!(status, RGSADDLE_OK);
        assert_eq!(out[1].to_le_bytes(), (0.3_f64).to_le_bytes());
        assert_ne!(out[0].to_le_bytes(), (0.4_f64).to_le_bytes());
        unsafe {
            rgsaddle_stencil_free(stencil);
            rgsaddle_stencil_config_free(config);
        }
    }
}
