//! C ABI over the stepping sessions. See `include/rgsaddle.h`.
//!
//! The host owns the loop here exactly as it does in Rust: create,
//! step until the report says converged, reset at a surface-epoch
//! boundary, free. No run-to-completion entry exists.

use std::ffi::c_void;
use std::os::raw::c_char;
use std::slice;

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Method};

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface, CiConfig};
use crate::irc::{IrcConfig, IrcDirection, IrcKind, IrcSession};
use crate::mic::Cell;
use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface};
use crate::projection::ProjectionKind;
use crate::spring::SpringKind;
use crate::tangent::TangentKind;

pub const RGSADDLE_ABI_MAJOR: u32 = 1;
pub const RGSADDLE_ABI_MINOR: u32 = 0;

pub const RGSADDLE_OK: i32 = 0;
pub const RGSADDLE_NULL_SESSION: i32 = -1;
pub const RGSADDLE_NULL_BAND: i32 = -2;
pub const RGSADDLE_NULL_SURFACE: i32 = -3;
pub const RGSADDLE_NULL_REPORT: i32 = -4;
pub const RGSADDLE_SHAPE: i32 = -5;
pub const RGSADDLE_SURFACE_FAILED: i32 = -6;
pub const RGSADDLE_NON_FINITE: i32 = -7;
pub const RGSADDLE_SOLVER: i32 = -8;
pub const RGSADDLE_ABI_MISMATCH: i32 = -9;
pub const RGSADDLE_INVALID_PARAMETER: i32 = -11;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RgsaddleVersion {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
pub struct RgsaddleSurfaceRequest {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub n_images: i64,
    pub n_atoms: i64,
    pub positions: *const f64,
    pub energies: *mut f64,
    pub gradients: *mut f64,
}

pub type RgsaddleSurfaceFn = extern "C" fn(*mut c_void, *mut RgsaddleSurfaceRequest) -> i32;

#[repr(C)]
pub struct RgsaddleBandConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub tangent: i32,
    pub spring: i32,
    pub projection: i32,
    pub method: i32,
    pub spring_k: f64,
    pub spring_ks: *const f64,
    pub ci_trigger_factor: f64,
    pub ci_trigger_force: f64,
    pub cell: *const f64,
    pub force_tol: f64,
    pub max_move: f64,
    pub memory: i64,
}

#[repr(C)]
pub struct RgsaddleIrcConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub dx: f64,
    pub force_tol: f64,
    pub max_move: f64,
    pub max_inner: i64,
    pub kind: i32,
    pub direction: i32,
}

#[repr(C)]
pub struct RgsaddleMinModeConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub kind: i32,
    pub method: i32,
    pub dr: f64,
    pub rotation_tol: f64,
    pub max_rotations: i64,
    pub krylov_dim: i64,
    pub force_tol: f64,
    pub max_move: f64,
}

#[repr(C)]
pub struct RgsaddleReport {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub status: i32,
    pub reserved: i32,
    pub max_force: f64,
    pub ci_index: i64,
    pub iteration: i64,
    pub curvature: f64,
    pub rotations: i64,
}

/// Host surface behind the C callback. The pointers live only for the
/// duration of one call, which is exactly the request's lifetime.
struct CSurface {
    f: RgsaddleSurfaceFn,
    user: *mut c_void,
    n_atoms: i64,
}

// The C host is responsible for its own thread safety; the sessions
// call the surface only from the thread that called step.
unsafe impl Sync for CSurface {}
unsafe impl Send for CSurface {}

impl CSurface {
    fn call(
        &self,
        n_images: i64,
        positions: &[f64],
        energies: &mut [f64],
        gradients: &mut [f64],
    ) -> Result<(), SaddleError> {
        let mut req = RgsaddleSurfaceRequest {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            n_images,
            n_atoms: self.n_atoms,
            positions: positions.as_ptr(),
            energies: energies.as_mut_ptr(),
            gradients: gradients.as_mut_ptr(),
        };
        let rc = (self.f)(self.user, &mut req);
        if rc != 0 {
            return Err(SaddleError::Surface(format!("host callback rc={rc}")));
        }
        Ok(())
    }
}

impl BandSurface for CSurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n_images = positions.nrows() as i64;
        let flat: Vec<f64> = positions.iter().copied().collect();
        let mut e = vec![0.0; energies.len()];
        let mut g = vec![0.0; gradients.len()];
        self.call(n_images, &flat, &mut e, &mut g)?;
        for (i, v) in e.iter().enumerate() {
            energies[i] = *v;
        }
        let dof = gradients.ncols();
        for i in 0..gradients.nrows() {
            for c in 0..dof {
                gradients[(i, c)] = g[i * dof + c];
            }
        }
        Ok(())
    }
}

impl PointSurface for CSurface {
    fn eval(&self, x: ndarray::ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let flat: Vec<f64> = x.iter().copied().collect();
        let mut e = vec![0.0; 1];
        let mut g = vec![0.0; flat.len()];
        self.call(1, &flat, &mut e, &mut g)?;
        Ok((e[0], Array1::from(g)))
    }
}

pub struct RgsaddleBand {
    session: BandSession,
    n_images: i64,
    n_atoms: i64,
}

pub struct RgsaddleMinMode {
    session: MinModeSession,
    n_atoms: i64,
}

pub struct RgsaddleIrc {
    session: IrcSession,
    n_atoms: i64,
}

fn method_of(v: i32) -> Method {
    match v {
        1 => Method::Lbfgs { memory: 20 },
        _ => Method::Fire { kind: FireKind::V2 },
    }
}

fn stamp_report(out: &mut RgsaddleReport) {
    out.version = RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    };
    out.flags = 0;
}

fn status_of(err: &SaddleError) -> i32 {
    match err {
        SaddleError::Shape(_) => RGSADDLE_SHAPE,
        SaddleError::Surface(_) => RGSADDLE_SURFACE_FAILED,
        SaddleError::NonFinite(_) => RGSADDLE_NON_FINITE,
        SaddleError::Solver(_) => RGSADDLE_SOLVER,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_abi_version() -> i32 {
    ((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR) as i32
}

/// # Safety
/// `out` must be a valid pointer to an `rgsaddle_version_t` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_abi_stamp(out: *mut RgsaddleVersion) -> i32 {
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    unsafe {
        (*out).major = RGSADDLE_ABI_MAJOR;
        (*out).minor = RGSADDLE_ABI_MINOR;
    }
    RGSADDLE_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_status_name(status: i32) -> *const c_char {
    let s: &'static str = match status {
        RGSADDLE_OK => "OK\0",
        RGSADDLE_NULL_SESSION => "NULL_SESSION\0",
        RGSADDLE_NULL_BAND => "NULL_BAND\0",
        RGSADDLE_NULL_SURFACE => "NULL_SURFACE\0",
        RGSADDLE_NULL_REPORT => "NULL_REPORT\0",
        RGSADDLE_SHAPE => "SHAPE\0",
        RGSADDLE_SURFACE_FAILED => "SURFACE_FAILED\0",
        RGSADDLE_NON_FINITE => "NON_FINITE\0",
        RGSADDLE_SOLVER => "SOLVER\0",
        RGSADDLE_ABI_MISMATCH => "ABI_MISMATCH\0",
        _ => "UNKNOWN\0",
    };
    s.as_ptr() as *const c_char
}

/// # Safety
/// `config` and `positions` must be valid for the declared shape.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_create(
    config: *const RgsaddleBandConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
) -> *mut RgsaddleBand {
    if config.is_null() || positions.is_null() || n_images < 3 || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let total = dof * n_images as usize;
    let flat = unsafe { slice::from_raw_parts(positions, total) };
    let initial = match Array2::from_shape_vec((n_images as usize, dof), flat.to_vec()) {
        Ok(a) => a,
        Err(_) => return std::ptr::null_mut(),
    };

    let spring = match cfg.spring {
        1 => {
            if cfg.spring_ks.is_null() {
                return std::ptr::null_mut();
            }
            let ks = unsafe { slice::from_raw_parts(cfg.spring_ks, (n_images - 1) as usize) };
            SpringKind::Weighted { ks: ks.to_vec() }
        }
        2 => SpringKind::OnsagerMachlup {
            k: cfg.spring_k,
            l_vecs: vec![Array1::zeros(dof); n_images as usize],
        },
        _ => SpringKind::Uniform { k: cfg.spring_k },
    };
    let cell = if cfg.cell.is_null() {
        None
    } else {
        let c = unsafe { slice::from_raw_parts(cfg.cell, 9) };
        Cell::from_vectors(
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
            [0.0, 0.0, 0.0],
        )
        .ok()
    };
    let band_config = BandConfig {
        tangent: if cfg.tangent == 0 {
            TangentKind::Simple
        } else {
            TangentKind::Improved
        },
        spring,
        projection: match cfg.projection {
            0 => ProjectionKind::PlainElasticBand,
            2 => ProjectionKind::DoublyNudged,
            _ => ProjectionKind::Neb,
        },
        climbing: (cfg.ci_trigger_factor > 0.0).then_some(CiConfig {
            trigger_factor: cfg.ci_trigger_factor,
            trigger_force: cfg.ci_trigger_force,
        }),
        cell,
        force_tol: cfg.force_tol,
        max_move: cfg.max_move,
        method: method_of(cfg.method),
    };
    match BandSession::new(band_config, initial) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleBand {
            session,
            n_images,
            n_atoms,
        })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `band` and `out` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_step(
    band: *mut RgsaddleBand,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let band = unsafe { &mut *band };
    let cs = CSurface {
        f,
        user,
        n_atoms: band.n_atoms,
    };
    match band.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == BandStatus::Converged {
                1
            } else {
                0
            };
            out.reserved = 0;
            out.max_force = report.max_force;
            out.ci_index = report.ci_index.map(|i| i as i64).unwrap_or(-1);
            out.iteration = report.iteration as i64;
            out.curvature = 0.0;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `n_images * 3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_positions(band: *const RgsaddleBand, out: *mut f64) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let band = unsafe { &*band };
    let pos = band.session.positions();
    let total = (band.n_images * 3 * band.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, total) };
    for (i, v) in pos.iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `positions` must hold `n_images * 3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_positions(
    band: *mut RgsaddleBand,
    positions: *const f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    if positions.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let band = unsafe { &mut *band };
    let dof = (3 * band.n_atoms) as usize;
    let total = dof * band.n_images as usize;
    let flat = unsafe { slice::from_raw_parts(positions, total) };
    let arr = match Array2::from_shape_vec((band.n_images as usize, dof), flat.to_vec()) {
        Ok(a) => a,
        Err(_) => return RGSADDLE_SHAPE,
    };
    match band.session.set_positions(arr) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `band` must be a live session pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_reset(band: *mut RgsaddleBand) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    unsafe { (*band).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `band` must come from [`rgsaddle_band_create`] and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_free(band: *mut RgsaddleBand) {
    if !band.is_null() {
        drop(unsafe { Box::from_raw(band) });
    }
}

/// # Safety
/// `config`, `position`, and `mode` must be valid for `3 * n_atoms`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_create(
    config: *const RgsaddleMinModeConfig,
    n_atoms: i64,
    position: *const f64,
    mode: *const f64,
) -> *mut RgsaddleMinMode {
    if config.is_null() || position.is_null() || mode.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(mode, dof) }.to_vec());
    let mm_config = MinModeConfig {
        kind: if cfg.kind == 1 {
            MinModeKind::Lanczos
        } else {
            MinModeKind::Dimer
        },
        dr: cfg.dr,
        rotation_tol: cfg.rotation_tol,
        max_rotations: cfg.max_rotations as usize,
        krylov_dim: cfg.krylov_dim as usize,
        eigen_kind: rgmin::EigensolverKind::Lanczos,
        force_tol: cfg.force_tol,
        max_move: cfg.max_move,
        method: method_of(cfg.method),
    };
    match MinModeSession::new(mm_config, x, m) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleMinMode { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_step(
    session: *mut RgsaddleMinMode,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let cs = CSurface {
        f,
        user,
        n_atoms: session.n_atoms,
    };
    match session.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == MinModeStatus::Converged {
                1
            } else {
                0
            };
            out.reserved = 0;
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = report.iteration as i64;
            out.curvature = report.curvature;
            out.rotations = report.rotations as i64;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_position(
    session: *const RgsaddleMinMode,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dof = (3 * session.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (i, v) in session.session.position().iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_mode(
    session: *const RgsaddleMinMode,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dof = (3 * session.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (i, v) in session.session.mode().iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `session` must be a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_reset(session: *mut RgsaddleMinMode) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `session` must come from [`rgsaddle_minmode_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_free(session: *mut RgsaddleMinMode) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// # Safety
/// `saddle` and `mode` are 3N, `masses` is N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_create(
    config: *const RgsaddleIrcConfig,
    n_atoms: i64,
    saddle: *const f64,
    masses: *const f64,
    mode: *const f64,
) -> *mut RgsaddleIrc {
    if config.is_null() || saddle.is_null() || masses.is_null() || mode.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(saddle, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let md = Array1::from(unsafe { slice::from_raw_parts(mode, dof) }.to_vec());
    let irc_cfg = IrcConfig {
        dx: cfg.dx,
        force_tol: cfg.force_tol,
        max_move: cfg.max_move,
        max_inner: cfg.max_inner.max(1) as usize,
        kind: if cfg.kind == 1 {
            IrcKind::Morokuma
        } else {
            IrcKind::Gs2
        },
        ..IrcConfig::default()
    };
    let dir = if cfg.direction == 1 {
        IrcDirection::Reverse
    } else {
        IrcDirection::Forward
    };
    match IrcSession::new(irc_cfg, x, m, md, dir) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleIrc { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `seed` is 3N. `surface` is called during the Lanczos kick.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_create_from_surface(
    config: *const RgsaddleIrcConfig,
    n_atoms: i64,
    saddle: *const f64,
    masses: *const f64,
    seed: *const f64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
) -> *mut RgsaddleIrc {
    if config.is_null()
        || saddle.is_null()
        || masses.is_null()
        || seed.is_null()
        || n_atoms < 1
    {
        return std::ptr::null_mut();
    }
    let Some(f) = surface else {
        return std::ptr::null_mut();
    };
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(saddle, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let sd = Array1::from(unsafe { slice::from_raw_parts(seed, dof) }.to_vec());
    let irc_cfg = IrcConfig {
        dx: cfg.dx,
        force_tol: cfg.force_tol,
        max_move: cfg.max_move,
        max_inner: cfg.max_inner.max(1) as usize,
        kind: if cfg.kind == 1 {
            IrcKind::Morokuma
        } else {
            IrcKind::Gs2
        },
        ..IrcConfig::default()
    };
    let dir = if cfg.direction == 1 {
        IrcDirection::Reverse
    } else {
        IrcDirection::Forward
    };
    let cs = CSurface {
        f,
        user,
        n_atoms,
    };
    match IrcSession::from_surface(irc_cfg, x, m, sd, dir, &cs) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleIrc { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_step(
    session: *mut RgsaddleIrc,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let cs = CSurface {
        f,
        user,
        n_atoms: session.n_atoms,
    };
    match session.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.at_minimum { 1 } else { 0 };
            out.reserved = 0;
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = report.inner_steps as i64;
            out.curvature = report.arc;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` holds 3N doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_position(
    session: *const RgsaddleIrc,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let dof = (3 * session.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (i, v) in session.session.position().iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// Live session or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_set_direction(
    session: *mut RgsaddleIrc,
    direction: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let dir = if direction == 1 {
        IrcDirection::Reverse
    } else {
        IrcDirection::Forward
    };
    unsafe { (*session).session.set_direction(dir) };
    RGSADDLE_OK
}

/// # Safety
/// Live session or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_reset(session: *mut RgsaddleIrc) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// Pointer from [`rgsaddle_irc_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_irc_free(session: *mut RgsaddleIrc) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

#[cfg(test)]
mod irc_abi_tests {
    use super::*;
    use ndarray::Array1;

    struct Well;
    impl PointSurface for Well {
        fn eval(
            &self,
            x: ndarray::ArrayView1<f64>,
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    extern "C" fn well_cb(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
        unsafe {
            let req = &mut *req;
            let n = (req.n_atoms * 3) as usize;
            let pos = slice::from_raw_parts(req.positions, n);
            let x = Array1::from(pos.to_vec());
            let (e, g) = Well.eval(x.view()).unwrap();
            *req.energies = e;
            let gout = slice::from_raw_parts_mut(req.gradients, n);
            for (i, v) in g.iter().enumerate() {
                gout[i] = *v;
            }
        }
        RGSADDLE_OK
    }

    #[test]
    fn irc_abi_morokuma_leaves_the_saddle() {
        let n_atoms = 2i64;
        let saddle = [0.0; 6];
        let masses = [1.0, 1.0];
        let mut mode = [0.0; 6];
        mode[0] = 1.0;
        let cfg = RgsaddleIrcConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            dx: 0.15,
            force_tol: 0.05,
            max_move: 0.2,
            max_inner: 10,
            kind: 1,
            direction: 0,
        };
        let sess = unsafe {
            rgsaddle_irc_create(
                &cfg,
                n_atoms,
                saddle.as_ptr(),
                masses.as_ptr(),
                mode.as_ptr(),
            )
        };
        assert!(!sess.is_null());
        let mut report = RgsaddleReport {
            version: RgsaddleVersion { major: 0, minor: 0 },
            flags: 0,
            status: 0,
            reserved: 0,
            max_force: 0.0,
            ci_index: 0,
            iteration: 0,
            curvature: 0.0,
            rotations: 0,
        };
        let rc = unsafe { rgsaddle_irc_step(sess, Some(well_cb), std::ptr::null_mut(), &mut report) };
        assert_eq!(rc, RGSADDLE_OK);
        let mut out = [0.0; 6];
        assert_eq!(unsafe { rgsaddle_irc_position(sess, out.as_mut_ptr()) }, RGSADDLE_OK);
        assert!(out[0].abs() > 1e-8, "kick must leave the saddle");
        unsafe { rgsaddle_irc_free(sess) };
    }
}
