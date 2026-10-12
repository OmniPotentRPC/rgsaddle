//! Opaque stencil handles.
//!
//! Point presets and band presets share these symbols. IDPP and S-IDPP
//! relax the pair-potential path. OCI-NEB is named and refused.
//! Every entry catches a panic and
//! records [`rgsaddle_last_error`].

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::panic::AssertUnwindSafe;

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};

use crate::error::SaddleError;
use crate::stencil::{
    ImageId, Interp, Preset, Stencil, StencilReport, StencilStatus, Surface, TangentRule, ZoomEnds,
    ZoomForce, ZoomParams, ZoomWindow,
};

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

pub const RGSADDLE_STENCIL_KEY_SPRING: i32 = 11;
pub const RGSADDLE_STENCIL_KEY_K: i32 = 12;
pub const RGSADDLE_STENCIL_KEY_K_MIN: i32 = 13;
pub const RGSADDLE_STENCIL_KEY_K_MAX: i32 = 14;
pub const RGSADDLE_STENCIL_KEY_ZOOM_STABILITY: i32 = 15;
pub const RGSADDLE_STENCIL_KEY_ZOOM_ALPHA: i32 = 16;
pub const RGSADDLE_STENCIL_KEY_ZOOM_OFFSET: i32 = 17;
pub const RGSADDLE_STENCIL_KEY_ZOOM_INTERP: i32 = 18;
pub const RGSADDLE_STENCIL_KEY_ZOOM_ENDS: i32 = 19;
pub const RGSADDLE_STENCIL_KEY_ZOOM_FORCE: i32 = 20;
pub const RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND: i32 = 21;
pub const RGSADDLE_STENCIL_KEY_IMAGE_SCALE: i32 = 22;
pub const RGSADDLE_STENCIL_KEY_BAND_SLACK: i32 = 23;
pub const RGSADDLE_STENCIL_KEY_CI_TRIGGER: i32 = 24;
pub const RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE: i32 = 25;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL: i32 = 26;
pub const RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX: i32 = 27;
pub const RGSADDLE_STENCIL_KEY_TANGENT: i32 = 28;
pub const RGSADDLE_STENCIL_KEY_MAX_ITER_ZOOM: i32 = 29;

pub const RGSADDLE_STENCIL_QUERY_POSITIONS: i32 = 1;
pub const RGSADDLE_STENCIL_QUERY_MODE: i32 = 2;
pub const RGSADDLE_STENCIL_QUERY_CURVATURE: i32 = 3;
pub const RGSADDLE_STENCIL_QUERY_CLIMBING: i32 = 4;
pub const RGSADDLE_STENCIL_QUERY_IMAGE_IDS: i32 = 5;
pub const RGSADDLE_STENCIL_QUERY_ENERGIES: i32 = 6;
pub const RGSADDLE_STENCIL_QUERY_ZOOM_WINDOW: i32 = 7;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(c_string(""));
}

fn c_string(text: &str) -> CString {
    let bytes: Vec<u8> = text.bytes().filter(|byte| *byte != 0).collect();
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
    pub ci_index: i64,
    pub zoom_lo: i64,
    pub zoom_hi: i64,
    pub phase: i32,
    pub phase_pad: i32,
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
    spring: i64,
    k: f64,
    k_min: f64,
    k_max: f64,
    k_bounds_set: bool,
    zoom_stability: usize,
    zoom_alpha: f64,
    zoom_offset: usize,
    zoom_interp: i64,
    zoom_ends: i64,
    zoom_force: f64,
    zoom_force_kind: i64,
    zoom_set: bool,
    image_scale: Option<f64>,
    band_slack: f64,
    trigger_factor: f64,
    trigger_force: f64,
    n_simul: usize,
    n_max: usize,
    tangent: i64,
    zoom_max_iter: usize,
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
            spring: 0,
            k: 5.0,
            k_min: 0.972,
            k_max: 9.72,
            k_bounds_set: false,
            zoom_stability: 5,
            zoom_alpha: 0.5,
            zoom_offset: 1,
            zoom_interp: 1,
            zoom_ends: 0,
            zoom_force: 10.0,
            zoom_force_kind: 0,
            zoom_set: false,
            image_scale: None,
            band_slack: 10.0,
            trigger_factor: 0.5,
            trigger_force: 0.0,
            n_simul: 3,
            n_max: 10,
            tangent: 1,
            zoom_max_iter: 0,
        }
    }

    fn preset(&self) -> Option<Preset> {
        Preset::from_abi(self.preset)
    }

    fn is_point(&self) -> bool {
        self.preset().is_some_and(Preset::is_point)
    }
}

struct Handle {
    stencil: Stencil,
    n_coords: usize,
    band: bool,
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

unsafe impl Send for CbSurface {}
unsafe impl Sync for CbSurface {}

impl Surface for CbSurface {
    fn eval(
        &self,
        ids: &[ImageId],
        x: ArrayView2<f64>,
        e: &mut [f64],
        mut g: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        let rows = x.nrows();
        let dof = x.ncols();
        if ids.len() != rows || e.len() != rows || g.dim() != (rows, dof) {
            return Err(SaddleError::Shape(
                "stencil callback shape does not match the position".into(),
            ));
        }
        let Some(flat) = x.as_slice() else {
            return Err(SaddleError::Shape(
                "stencil position is not contiguous".into(),
            ));
        };
        let Some(gradient) = g.as_slice_mut() else {
            return Err(SaddleError::Shape(
                "stencil gradient is not contiguous".into(),
            ));
        };
        let code = unsafe {
            (self.eval)(
                self.user,
                rows,
                dof,
                ids.as_ptr(),
                flat.as_ptr(),
                e.as_mut_ptr(),
                gradient.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(SaddleError::Surface(format!(
                "stencil callback status {code}"
            )));
        }
        if e.iter().any(|value| !value.is_finite())
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("stencil callback"));
        }
        Ok(())
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

fn finite_key(value: f64) -> Result<(), i32> {
    if value.is_finite() {
        Ok(())
    } else {
        set_last_error("stencil key is not finite");
        Err(super::RGSADDLE_NON_FINITE)
    }
}

fn reject_family(point_key: bool, config: &Config) -> Option<i32> {
    if point_key && !config.is_point() {
        set_last_error("this key is for a point stencil");
        return Some(RGSADDLE_INVALID_PARAMETER);
    }
    if !point_key && config.is_point() {
        set_last_error("this key is for a band stencil");
        return Some(RGSADDLE_INVALID_PARAMETER);
    }
    None
}

fn zoom_from(config: &Config) -> ZoomParams {
    let force = if config.zoom_force_kind == 1 {
        ZoomForce::Abs(config.zoom_force)
    } else {
        ZoomForce::RelTol(config.zoom_force)
    };
    ZoomParams {
        stability: config.zoom_stability,
        force,
        window: ZoomWindow::Auto {
            alpha: config.zoom_alpha,
            offset: config.zoom_offset,
        },
        interpolation: if config.zoom_interp == 0 {
            Interp::Linear
        } else {
            Interp::Cubic
        },
        ends: if config.zoom_ends == 1 {
            ZoomEnds::FreePerp
        } else {
            ZoomEnds::Fixed
        },
        max_iterations: config.zoom_max_iter,
    }
}

fn apply(config: &Config, rows: usize, dof: usize, values: &[f64]) -> Result<Stencil, SaddleError> {
    let Some(preset) = config.preset() else {
        return Err(SaddleError::Invalid("unknown stencil preset".into()));
    };
    if preset.is_point() {
        let mut builder = Stencil::point(Array1::from_vec(values.to_vec()))
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
        return builder.build();
    }
    let x = Array2::from_shape_vec((rows, dof), values.to_vec())
        .map_err(|_| SaddleError::Shape("band position is not a matrix".into()))?;
    let mut builder = Stencil::band(x)
        .preset(preset)
        .threads(config.threads)
        .force_tol(config.force_tol)
        .max_move(config.max_move)
        .k(config.k)
        .tangent(if config.tangent == 0 {
            TangentRule::Simple
        } else {
            TangentRule::Improved
        })
        .climb_trigger(config.trigger_factor, config.trigger_force)
        .band_slack(config.band_slack)
        .autoneb_limits(config.n_simul, config.n_max);
    if config.spring == 1 || config.k_bounds_set {
        builder = builder.energy_weighted(config.k_min, config.k_max);
    }
    if let Some(scale) = config.image_scale {
        builder = builder.image_scale(Some(scale));
    }
    if preset == Preset::ZoomNeb || config.zoom_set {
        builder = builder.zoom(zoom_from(config));
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
    let (zoom_lo, zoom_hi) = report
        .zoom_window
        .map_or((-1, -1), |(lo, hi)| (lo as i64, hi as i64));
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
        (*out).ci_index = report.ci_index.map_or(-1, |index| index as i64);
        (*out).zoom_lo = zoom_lo;
        (*out).zoom_hi = zoom_hi;
        (*out).phase = report.phase;
        (*out).phase_pad = 0;
    }
    RGSADDLE_OK
}

fn copy_f64(out: *mut f64, values: &[f64]) {
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len()) };
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
        let point_only = matches!(
            key,
            RGSADDLE_STENCIL_KEY_DR
                | RGSADDLE_STENCIL_KEY_ROTATION_TOL
                | RGSADDLE_STENCIL_KEY_ART_THRESHOLD
        );
        let shared = matches!(
            key,
            RGSADDLE_STENCIL_KEY_FORCE_TOL | RGSADDLE_STENCIL_KEY_MAX_MOVE
        );
        let band_only = matches!(
            key,
            RGSADDLE_STENCIL_KEY_K
                | RGSADDLE_STENCIL_KEY_K_MIN
                | RGSADDLE_STENCIL_KEY_K_MAX
                | RGSADDLE_STENCIL_KEY_ZOOM_ALPHA
                | RGSADDLE_STENCIL_KEY_ZOOM_FORCE
                | RGSADDLE_STENCIL_KEY_IMAGE_SCALE
                | RGSADDLE_STENCIL_KEY_BAND_SLACK
                | RGSADDLE_STENCIL_KEY_CI_TRIGGER
                | RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE
        );
        if matches!(
            key,
            RGSADDLE_STENCIL_KEY_MAX_ROTATIONS
                | RGSADDLE_STENCIL_KEY_KRYLOV_DIM
                | RGSADDLE_STENCIL_KEY_THREADS
                | RGSADDLE_STENCIL_KEY_ART_PUSH
                | RGSADDLE_STENCIL_KEY_MODE
                | RGSADDLE_STENCIL_KEY_SPRING
                | RGSADDLE_STENCIL_KEY_ZOOM_STABILITY
                | RGSADDLE_STENCIL_KEY_ZOOM_OFFSET
                | RGSADDLE_STENCIL_KEY_ZOOM_INTERP
                | RGSADDLE_STENCIL_KEY_ZOOM_ENDS
                | RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND
                | RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL
                | RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX
                | RGSADDLE_STENCIL_KEY_TANGENT
                | RGSADDLE_STENCIL_KEY_MAX_ITER_ZOOM
        ) {
            set_last_error("this key takes an integer");
            return RGSADDLE_INVALID_PARAMETER;
        }
        if !(point_only || shared || band_only) {
            set_last_error("unknown stencil key");
            return RGSADDLE_INVALID_PARAMETER;
        }
        if !shared && let Some(code) = reject_family(point_only, config) {
            return code;
        }
        if let Err(code) = finite_key(value) {
            return code;
        }
        match key {
            RGSADDLE_STENCIL_KEY_DR => config.dr = value,
            RGSADDLE_STENCIL_KEY_FORCE_TOL => config.force_tol = value,
            RGSADDLE_STENCIL_KEY_MAX_MOVE => config.max_move = value,
            RGSADDLE_STENCIL_KEY_ROTATION_TOL => config.rotation_tol = value,
            RGSADDLE_STENCIL_KEY_ART_THRESHOLD => config.art_threshold = value,
            RGSADDLE_STENCIL_KEY_K => config.k = value,
            RGSADDLE_STENCIL_KEY_K_MIN => {
                config.k_min = value;
                config.k_bounds_set = true;
            }
            RGSADDLE_STENCIL_KEY_K_MAX => {
                config.k_max = value;
                config.k_bounds_set = true;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_ALPHA => {
                config.zoom_alpha = value;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_FORCE => {
                config.zoom_force = value;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_IMAGE_SCALE => config.image_scale = Some(value),
            RGSADDLE_STENCIL_KEY_BAND_SLACK => config.band_slack = value,
            RGSADDLE_STENCIL_KEY_CI_TRIGGER => config.trigger_factor = value,
            RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE => config.trigger_force = value,
            _ => unreachable!("key family was checked"),
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
        let point_only = matches!(
            key,
            RGSADDLE_STENCIL_KEY_MAX_ROTATIONS
                | RGSADDLE_STENCIL_KEY_KRYLOV_DIM
                | RGSADDLE_STENCIL_KEY_ART_PUSH
        );
        let shared = key == RGSADDLE_STENCIL_KEY_THREADS;
        let band_only = matches!(
            key,
            RGSADDLE_STENCIL_KEY_SPRING
                | RGSADDLE_STENCIL_KEY_ZOOM_STABILITY
                | RGSADDLE_STENCIL_KEY_ZOOM_OFFSET
                | RGSADDLE_STENCIL_KEY_ZOOM_INTERP
                | RGSADDLE_STENCIL_KEY_ZOOM_ENDS
                | RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND
                | RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL
                | RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX
                | RGSADDLE_STENCIL_KEY_TANGENT
                | RGSADDLE_STENCIL_KEY_MAX_ITER_ZOOM
        );
        if matches!(
            key,
            RGSADDLE_STENCIL_KEY_DR
                | RGSADDLE_STENCIL_KEY_FORCE_TOL
                | RGSADDLE_STENCIL_KEY_MAX_MOVE
                | RGSADDLE_STENCIL_KEY_ROTATION_TOL
                | RGSADDLE_STENCIL_KEY_ART_THRESHOLD
                | RGSADDLE_STENCIL_KEY_MODE
                | RGSADDLE_STENCIL_KEY_K
                | RGSADDLE_STENCIL_KEY_K_MIN
                | RGSADDLE_STENCIL_KEY_K_MAX
                | RGSADDLE_STENCIL_KEY_ZOOM_ALPHA
                | RGSADDLE_STENCIL_KEY_ZOOM_FORCE
                | RGSADDLE_STENCIL_KEY_IMAGE_SCALE
                | RGSADDLE_STENCIL_KEY_BAND_SLACK
                | RGSADDLE_STENCIL_KEY_CI_TRIGGER
                | RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE
        ) {
            set_last_error("this key takes a float");
            return RGSADDLE_INVALID_PARAMETER;
        }
        if !(point_only || shared || band_only) {
            set_last_error("unknown stencil key");
            return RGSADDLE_INVALID_PARAMETER;
        }
        if !shared && let Some(code) = reject_family(point_only, config) {
            return code;
        }
        let count = value as usize;
        match key {
            RGSADDLE_STENCIL_KEY_MAX_ROTATIONS => config.max_rotations = count,
            RGSADDLE_STENCIL_KEY_KRYLOV_DIM => config.krylov_dim = count,
            RGSADDLE_STENCIL_KEY_THREADS => config.threads = count,
            RGSADDLE_STENCIL_KEY_ART_PUSH => config.art_push = count,
            RGSADDLE_STENCIL_KEY_SPRING => {
                if value > 1 {
                    set_last_error("unknown spring");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                config.spring = value;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_STABILITY => {
                config.zoom_stability = count;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_OFFSET => {
                config.zoom_offset = count;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_INTERP => {
                if value > 1 {
                    set_last_error("unknown interpolation");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                config.zoom_interp = value;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_ENDS => {
                if value > 1 {
                    set_last_error("unknown zoom end");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                config.zoom_ends = value;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND => {
                if value > 1 {
                    set_last_error("unknown zoom force");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                config.zoom_force_kind = value;
                config.zoom_set = true;
            }
            RGSADDLE_STENCIL_KEY_AUTONEB_N_SIMUL => config.n_simul = count,
            RGSADDLE_STENCIL_KEY_AUTONEB_N_MAX => config.n_max = count,
            RGSADDLE_STENCIL_KEY_TANGENT => {
                if value > 1 {
                    set_last_error("unknown tangent");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                config.tangent = value;
            }
            RGSADDLE_STENCIL_KEY_MAX_ITER_ZOOM => {
                config.zoom_max_iter = count;
                config.zoom_set = true;
            }
            _ => unreachable!("key family was checked"),
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
        let config = unsafe { &mut *(config as *mut Config) };
        if key != RGSADDLE_STENCIL_KEY_MODE {
            if matches!(
                key,
                RGSADDLE_STENCIL_KEY_DR
                    | RGSADDLE_STENCIL_KEY_FORCE_TOL
                    | RGSADDLE_STENCIL_KEY_MAX_MOVE
                    | RGSADDLE_STENCIL_KEY_ROTATION_TOL
                    | RGSADDLE_STENCIL_KEY_ART_THRESHOLD
                    | RGSADDLE_STENCIL_KEY_K
                    | RGSADDLE_STENCIL_KEY_K_MIN
                    | RGSADDLE_STENCIL_KEY_K_MAX
                    | RGSADDLE_STENCIL_KEY_ZOOM_ALPHA
                    | RGSADDLE_STENCIL_KEY_ZOOM_FORCE
                    | RGSADDLE_STENCIL_KEY_IMAGE_SCALE
                    | RGSADDLE_STENCIL_KEY_BAND_SLACK
                    | RGSADDLE_STENCIL_KEY_CI_TRIGGER
                    | RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE
            ) {
                set_last_error("this key takes a float");
            } else if (1..=29).contains(&key) {
                set_last_error("this key takes an integer");
            } else {
                set_last_error("unknown stencil key");
            }
            return RGSADDLE_INVALID_PARAMETER;
        }
        if let Some(code) = reject_family(true, config) {
            return code;
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
        config.mode = mode;
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
        let Some(preset) = config_ref.preset() else {
            set_last_error("unknown stencil preset");
            return std::ptr::null_mut();
        };
        if preset.is_point() && rows != 1 {
            set_last_error("point presets take one row");
            return std::ptr::null_mut();
        }
        if !preset.is_point() && rows < 2 {
            set_last_error("a band needs at least two images");
            return std::ptr::null_mut();
        }
        if dof == 0 || x.is_null() {
            set_last_error("stencil position is empty");
            return std::ptr::null_mut();
        }
        let n = rows.saturating_mul(dof);
        let values = unsafe { std::slice::from_raw_parts(x, n) };
        match apply(config_ref, rows, dof, values) {
            Ok(stencil) => Box::into_raw(Box::new(Handle {
                stencil,
                n_coords: n,
                band: !preset.is_point(),
            }))
            .cast(),
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
    guard(|| step_or_run(stencil, eval, user, 1, out, false))
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
    guard(|| step_or_run(stencil, eval, user, max_steps, out, true))
}

fn step_or_run(
    stencil: *mut c_void,
    eval: Option<StencilEvalFn>,
    user: *mut c_void,
    max_steps: usize,
    out: *mut RgsaddleStencilReport,
    run_to_end: bool,
) -> i32 {
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
    let result = if run_to_end {
        if stencil.band {
            stencil.stencil.run_surface(&surface, max_steps)
        } else {
            stencil.stencil.run(&surface, max_steps)
        }
    } else if stencil.band {
        stencil.stencil.step_surface(&surface)
    } else {
        stencil.stencil.step(&surface)
    };
    match result {
        Ok(report) => write_report(out, &report),
        Err(err) => fail(&err),
    }
}

/// # Safety
/// `stencil` is live or null. `out` has room for `n` values, with `n` equal
/// to the number of coordinates.
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
        if n != stencil.n_coords || out.is_null() {
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
                if n != stencil.n_coords {
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
                    set_last_error("this stencil has no mode");
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
            RGSADDLE_STENCIL_QUERY_CLIMBING => {
                if !stencil.band {
                    set_last_error("this key is for a band stencil");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                if n < 1 {
                    set_last_error("climbing index needs one value");
                    return RGSADDLE_SHAPE;
                }
                let index = stencil
                    .stencil
                    .climbing_index()
                    .map_or(-1.0, |value| value as f64);
                unsafe { *out = index };
                RGSADDLE_OK
            }
            RGSADDLE_STENCIL_QUERY_IMAGE_IDS => {
                if !stencil.band {
                    set_last_error("this key is for a band stencil");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                let ids = stencil.stencil.image_ids();
                if n != ids.len() {
                    set_last_error("image id length does not match");
                    return RGSADDLE_SHAPE;
                }
                let values: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
                copy_f64(out, &values);
                RGSADDLE_OK
            }
            RGSADDLE_STENCIL_QUERY_ENERGIES => {
                if !stencil.band {
                    set_last_error("this key is for a band stencil");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                let Some(energies) = stencil.stencil.energies() else {
                    set_last_error("energies are not known yet");
                    return RGSADDLE_INVALID_PARAMETER;
                };
                if n != energies.len() {
                    set_last_error("energy length does not match");
                    return RGSADDLE_SHAPE;
                }
                copy_f64(out, &energies);
                RGSADDLE_OK
            }
            RGSADDLE_STENCIL_QUERY_ZOOM_WINDOW => {
                if !stencil.band {
                    set_last_error("this key is for a band stencil");
                    return RGSADDLE_INVALID_PARAMETER;
                }
                if n < 2 {
                    set_last_error("zoom window needs two values");
                    return RGSADDLE_SHAPE;
                }
                let Some((lo, hi)) = stencil.stencil.zoom_window() else {
                    set_last_error("zoom has not packed");
                    return RGSADDLE_INVALID_PARAMETER;
                };
                unsafe {
                    *out = lo as f64;
                    *out.add(1) = hi as f64;
                }
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

    fn blank_report() -> RgsaddleStencilReport {
        RgsaddleStencilReport {
            struct_size: 0,
            status: -1,
            reserved: 0,
            max_force: 0.0,
            curvature: 0.0,
            iteration: 0,
            evaluations: 0,
            ci_index: -1,
            zoom_lo: -1,
            zoom_hi: -1,
            phase: -1,
            phase_pad: 0,
        }
    }

    #[test]
    fn unknown_keys_and_unsupported_presets_fail_closed() {
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

        let status =
            unsafe { rgsaddle_stencil_config_set_f64(config, RGSADDLE_STENCIL_KEY_DR, 1e-3) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        assert!(text.to_string_lossy().contains("point stencil"));

        let images = [0.0_f64, 0.0, 0.5, 0.0, 1.0, 0.0];
        let idpp = rgsaddle_stencil_config_new(Preset::Idpp.to_abi());
        let stencil = unsafe { rgsaddle_stencil_new(idpp, 3, 2, images.as_ptr()) };
        assert!(stencil.is_null());
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        let message = text.to_string_lossy();
        assert!(message.contains("groups of three"));
        assert!(!message.contains("feat/"));
        unsafe { rgsaddle_stencil_config_free(idpp) };

        let stencil = unsafe { rgsaddle_stencil_new(config, 3, 2, images.as_ptr()) };
        assert!(!stencil.is_null());
        unsafe {
            rgsaddle_stencil_free(stencil);
            rgsaddle_stencil_config_free(config);
        }
    }

    #[test]
    fn a_point_dimer_steps_through_the_callback() {
        let config = rgsaddle_stencil_config_new(Preset::Dimer.to_abi());
        let mode = [1.0_f64, 0.0];
        let status = unsafe {
            rgsaddle_stencil_config_set_array(config, RGSADDLE_STENCIL_KEY_MODE, mode.as_ptr(), 2)
        };
        assert_eq!(status, RGSADDLE_OK);
        let status = unsafe {
            rgsaddle_stencil_config_set_f64(config, RGSADDLE_STENCIL_KEY_FORCE_TOL, 1e-3)
        };
        assert_eq!(status, RGSADDLE_OK);
        let status =
            unsafe { rgsaddle_stencil_config_set_i64(config, RGSADDLE_STENCIL_KEY_THREADS, 1) };
        assert_eq!(status, RGSADDLE_OK);
        let status =
            unsafe { rgsaddle_stencil_config_set_f64(config, RGSADDLE_STENCIL_KEY_K, 5.0) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        assert!(text.to_string_lossy().contains("band stencil"));
        let x = [0.4_f64, 0.3];
        let stencil = unsafe { rgsaddle_stencil_new(config, 1, 2, x.as_ptr()) };
        assert!(!stencil.is_null());
        let mut report = blank_report();
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
