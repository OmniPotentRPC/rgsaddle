//! C ABI over the stepping sessions. See `include/rgsaddle.h`.
//!
//! The host owns the loop here exactly as it does in Rust: create,
//! step until the report says converged, reset at a surface-epoch
//! boundary, free. No run-to-completion entry exists.

use std::ffi::c_void;
use std::os::raw::c_char;
use std::slice;

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Manifold, Method};

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface, CiConfig};
use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::irc::{IrcConfig, IrcDirection, IrcKind, IrcSession};
use crate::mic::Cell;
use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface};
use crate::pes::CartesianPes;
use crate::pes_internal::InternalPes;
use crate::projection::ProjectionKind;
use crate::samd::{SamdConfig, SamdSession};
use crate::sella_min::{SellaMinConfig, SellaMinSession};
use crate::sella_saddle::{SellaSaddleConfig, SellaSaddleSession};
use crate::spring::SpringKind;
use crate::tangent::TangentKind;

pub const RGSADDLE_ABI_MAJOR: u32 = 1;
pub const RGSADDLE_ABI_MINOR: u32 = 13;

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

fn force_gate_of(v: i32) -> Option<crate::ForceGate> {
    crate::ForceGate::try_from_abi(v)
}

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
    pub force_gate: i32,
    pub max_move: f64,
    pub memory: i64,
}

#[repr(C)]
pub struct RgsaddleIrcConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub dx: f64,
    pub force_tol: f64,
    pub force_gate: i32,
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
    pub force_gate: i32,
    pub max_move: f64,
}

#[repr(C)]
pub struct RgsaddleSellaMinConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub delta: f64,
    pub force_tol: f64,
    pub force_gate: i32,
}

#[repr(C)]
pub struct RgsaddleSellaSaddleConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub delta: f64,
    pub force_tol: f64,
    pub force_gate: i32,
    pub order: i64,
}

#[repr(C)]
pub struct RgsaddleConstraintsConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
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

pub struct RgsaddleSellaMin {
    session: SellaMinSession,
    n_atoms: i64,
}

pub struct RgsaddleSellaSaddle {
    session: SellaSaddleSession,
    n_atoms: i64,
}

pub struct RgsaddleConstraints {
    cons: Constraints,
    n_atoms: i64,
}

pub struct RgsaddleInternalPes {
    pes: InternalPes,
    n_atoms: i64,
}

pub struct RgsaddlePes {
    pes: CartesianPes,
    n_atoms: i64,
}

#[repr(C)]
pub struct RgsaddleSamdConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub dt: f64,
    pub tau: f64,
    pub t0: f64,
    pub tf: f64,
    pub ngen: i64,
    pub exponential: i32,
}

pub struct RgsaddleSamd {
    session: SamdSession,
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

/// C type is `rgsaddle_force_gate_t`. Unknown discriminants return NULL.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_force_gate_name(gate: i32) -> *const c_char {
    match crate::ForceGate::try_from_abi(gate) {
        Some(g) => match g {
            crate::ForceGate::L2Norm => c"L2".as_ptr(),
            crate::ForceGate::LinfNorm => c"LINF".as_ptr(),
            crate::ForceGate::MaxForceOnAtom => c"MAX_ATOM".as_ptr(),
        },
        None => std::ptr::null(),
    }
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
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
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
        force_gate,
        max_move: cfg.max_move,
        method: method_of(cfg.method),
        // method 2: Riemannian trust region on the band; the trust radius
        // is the per-atom cap max_move over the interior images in the
        // flattened metric, so an accepted step moves no image further on
        // average than the solver path allows.
        rtr: (cfg.method == 2).then(|| crate::rtr::RtrConfig {
            radius_max: cfg.max_move * ((n_images - 2) as f64).sqrt(),
            ..crate::rtr::RtrConfig::default()
        }),
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
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
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
        force_gate,
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
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let irc_cfg = IrcConfig {
        dx: cfg.dx,
        force_tol: cfg.force_tol,
        force_gate,
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
    if config.is_null() || saddle.is_null() || masses.is_null() || seed.is_null() || n_atoms < 1 {
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
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let irc_cfg = IrcConfig {
        dx: cfg.dx,
        force_tol: cfg.force_tol,
        force_gate,
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
    let cs = CSurface { f, user, n_atoms };
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
pub unsafe extern "C" fn rgsaddle_irc_position(session: *const RgsaddleIrc, out: *mut f64) -> i32 {
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
        fn eval(&self, x: ndarray::ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
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
            force_gate: crate::ForceGate::MaxForceOnAtom.to_abi(),
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
        let rc =
            unsafe { rgsaddle_irc_step(sess, Some(well_cb), std::ptr::null_mut(), &mut report) };
        assert_eq!(rc, RGSADDLE_OK);
        let mut out = [0.0; 6];
        assert_eq!(
            unsafe { rgsaddle_irc_position(sess, out.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(out[0].abs() > 1e-8, "kick must leave the saddle");
        unsafe { rgsaddle_irc_free(sess) };
    }

    #[test]
    fn sella_min_abi_steps_a_well() {
        let n_atoms = 2i64;
        let mut x = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        let masses = [1.0, 1.0];
        let cfg = RgsaddleSellaMinConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            delta: 0.2,
            force_tol: 0.05,
            force_gate: crate::ForceGate::MaxForceOnAtom.to_abi(),
        };
        let sess = unsafe { rgsaddle_sella_min_create(&cfg, n_atoms, x.as_ptr(), masses.as_ptr()) };
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
        let rc = unsafe {
            rgsaddle_sella_min_step(sess, Some(well_cb), std::ptr::null_mut(), &mut report)
        };
        assert_eq!(rc, RGSADDLE_OK);
        assert!(report.max_force.is_finite());
        assert_eq!(
            unsafe { rgsaddle_sella_min_position(sess, x.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(x.iter().all(|v| v.is_finite()));
        let bad = RgsaddleSellaMinConfig {
            force_gate: 99,
            ..cfg
        };
        let refused =
            unsafe { rgsaddle_sella_min_create(&bad, n_atoms, x.as_ptr(), masses.as_ptr()) };
        assert!(refused.is_null());
        assert_eq!(
            unsafe { rgsaddle_sella_min_set_hess_update(sess, crate::HessUpdate::TsBfgs.to_abi()) },
            RGSADDLE_OK
        );
        assert_eq!(
            unsafe { rgsaddle_sella_min_set_hess_update(sess, 99) },
            RGSADDLE_INVALID_PARAMETER
        );
        assert_eq!(
            unsafe {
                rgsaddle_sella_min_set_restricted(
                    sess,
                    crate::RestrictedKind::MaxInternalStep.to_abi(),
                )
            },
            RGSADDLE_OK
        );
        assert_eq!(
            unsafe {
                rgsaddle_sella_min_set_restricted(
                    sess,
                    crate::RestrictedKind::RestrictedAtomicStep.to_abi(),
                )
            },
            RGSADDLE_OK
        );
        assert_eq!(
            unsafe { rgsaddle_sella_min_set_restricted(sess, 99) },
            RGSADDLE_INVALID_PARAMETER
        );
        unsafe { rgsaddle_sella_min_free(sess) };
    }

    #[test]
    fn sella_saddle_abi_steps() {
        let n_atoms = 2i64;
        let mut x = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0];
        let masses = [1.0, 1.0];
        let cfg = RgsaddleSellaSaddleConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            delta: 0.1,
            force_tol: 0.05,
            force_gate: crate::ForceGate::MaxForceOnAtom.to_abi(),
            order: 1,
        };
        let sess =
            unsafe { rgsaddle_sella_saddle_create(&cfg, n_atoms, x.as_ptr(), masses.as_ptr()) };
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
        let rc = unsafe {
            rgsaddle_sella_saddle_step(sess, Some(well_cb), std::ptr::null_mut(), &mut report)
        };
        assert_eq!(rc, RGSADDLE_OK);
        assert!(report.max_force.is_finite());
        assert_eq!(
            unsafe { rgsaddle_sella_saddle_position(sess, x.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(x.iter().all(|v| v.is_finite()));
        let bad = RgsaddleSellaSaddleConfig {
            force_gate: 99,
            ..cfg
        };
        let refused =
            unsafe { rgsaddle_sella_saddle_create(&bad, n_atoms, x.as_ptr(), masses.as_ptr()) };
        assert!(refused.is_null());
        unsafe { rgsaddle_sella_saddle_free(sess) };
    }
}

/// # Safety
/// `position` is 3N, `masses` is N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_create(
    config: *const RgsaddleSellaMinConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
) -> *mut RgsaddleSellaMin {
    if config.is_null() || position.is_null() || masses.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let mut sella = SellaMinConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    sella.force_gate = force_gate;
    match SellaMinSession::new(sella, x, m) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaMin { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cons` is copied; the caller still owns it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_create_on(
    config: *const RgsaddleSellaMinConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
) -> *mut RgsaddleSellaMin {
    if config.is_null() || position.is_null() || masses.is_null() || cons.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let mut sella = SellaMinConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    sella.force_gate = force_gate;
    match SellaMinSession::with_chart(sella, x, m, chart.cons.clone()) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaMin { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cons` is copied; the caller still owns it. QN runs in internals.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_create_internal(
    config: *const RgsaddleSellaMinConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
) -> *mut RgsaddleSellaMin {
    if config.is_null() || position.is_null() || masses.is_null() || cons.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let mut sella = SellaMinConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    sella.force_gate = force_gate;
    match SellaMinSession::on_internal(sella, x, m, chart.cons.clone()) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaMin { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cell` is 9 doubles. `mask` is 9 ints or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_create_cell(
    config: *const RgsaddleSellaMinConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cell: *const f64,
    mask: *const i32,
) -> *mut RgsaddleSellaMin {
    if config.is_null() || position.is_null() || masses.is_null() || cell.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let c = unsafe { slice::from_raw_parts(cell, 9) };
    let Ok(lc) = Cell::from_vectors(
        [c[0], c[1], c[2]],
        [c[3], c[4], c[5]],
        [c[6], c[7], c[8]],
        [0.0, 0.0, 0.0],
    ) else {
        return std::ptr::null_mut();
    };
    let mut bits = [true; 9];
    if !mask.is_null() {
        let mv = unsafe { slice::from_raw_parts(mask, 9) };
        for i in 0..9 {
            bits[i] = mv[i] != 0;
        }
    }
    let mut sella = SellaMinConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    sella.force_gate = force_gate;
    match SellaMinSession::on_cell(sella, x, m, lc, bits) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaMin { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cons` is copied. `cell` is 9 doubles. `mask` is 9 ints or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_create_cell_internal(
    config: *const RgsaddleSellaMinConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
    cell: *const f64,
    mask: *const i32,
) -> *mut RgsaddleSellaMin {
    if config.is_null()
        || position.is_null()
        || masses.is_null()
        || cons.is_null()
        || cell.is_null()
        || n_atoms < 1
    {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let c = unsafe { slice::from_raw_parts(cell, 9) };
    let Ok(lc) = Cell::from_vectors(
        [c[0], c[1], c[2]],
        [c[3], c[4], c[5]],
        [c[6], c[7], c[8]],
        [0.0, 0.0, 0.0],
    ) else {
        return std::ptr::null_mut();
    };
    let mut bits = [true; 9];
    if !mask.is_null() {
        let mv = unsafe { slice::from_raw_parts(mask, 9) };
        for i in 0..9 {
            bits[i] = mv[i] != 0;
        }
    }
    let mut sella = SellaMinConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    sella.force_gate = force_gate;
    match SellaMinSession::on_cell_internal(sella, x, m, chart.cons.clone(), lc, bits) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaMin { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_step(
    session: *mut RgsaddleSellaMin,
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
            out.iteration = 0;
            out.curvature = report.rho;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_position(
    session: *const RgsaddleSellaMin,
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
/// `session` must be a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_reset(session: *mut RgsaddleSellaMin) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `session` must be a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_set_hess_update(
    session: *mut RgsaddleSellaMin,
    update: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::HessUpdate::try_from_abi(update) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*session).session.set_update(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `restricted` is RestrictedKind. Unknown refuses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_set_restricted(
    session: *mut RgsaddleSellaMin,
    restricted: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::RestrictedKind::try_from_abi(restricted) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*session).session.set_restricted(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `chart` is CellChart. Unknown refuses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_set_cell_chart(
    session: *mut RgsaddleSellaMin,
    chart: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::CellChart::try_from_abi(chart) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*session).session.set_cell_chart(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `applied` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_maybe_niggli(
    session: *mut RgsaddleSellaMin,
    angle_threshold: f64,
    applied: *mut i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    match unsafe { (*session).session.maybe_niggli_reduce(angle_threshold) } {
        Ok(did) => {
            if !applied.is_null() {
                unsafe { *applied = if did { 1 } else { 0 } };
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `session` must come from [`rgsaddle_sella_min_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_min_free(session: *mut RgsaddleSellaMin) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// # Safety
/// `position` is 3N, `masses` is N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_create(
    config: *const RgsaddleSellaSaddleConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
) -> *mut RgsaddleSellaSaddle {
    if config.is_null() || position.is_null() || masses.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let mut sella = SellaSaddleConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    if cfg.order > 0 {
        sella.order = cfg.order as usize;
    }
    sella.force_gate = force_gate;
    match SellaSaddleSession::new(sella, x, m) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaSaddle { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cons` is copied; the caller still owns it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_create_on(
    config: *const RgsaddleSellaSaddleConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
) -> *mut RgsaddleSellaSaddle {
    if config.is_null() || position.is_null() || masses.is_null() || cons.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let mut sella = SellaSaddleConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    if cfg.order > 0 {
        sella.order = cfg.order as usize;
    }
    sella.force_gate = force_gate;
    match SellaSaddleSession::with_chart(sella, x, m, chart.cons.clone()) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaSaddle { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cons` is copied; the caller still owns it. P-RFO runs in internals.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_create_internal(
    config: *const RgsaddleSellaSaddleConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
) -> *mut RgsaddleSellaSaddle {
    if config.is_null() || position.is_null() || masses.is_null() || cons.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let mut sella = SellaSaddleConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    if cfg.order > 0 {
        sella.order = cfg.order as usize;
    }
    sella.force_gate = force_gate;
    match SellaSaddleSession::on_internal(sella, x, m, chart.cons.clone()) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaSaddle { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cell` is 9 doubles. `mask` is 9 ints or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_create_cell(
    config: *const RgsaddleSellaSaddleConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cell: *const f64,
    mask: *const i32,
) -> *mut RgsaddleSellaSaddle {
    if config.is_null() || position.is_null() || masses.is_null() || cell.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let c = unsafe { slice::from_raw_parts(cell, 9) };
    let Ok(lc) = Cell::from_vectors(
        [c[0], c[1], c[2]],
        [c[3], c[4], c[5]],
        [c[6], c[7], c[8]],
        [0.0, 0.0, 0.0],
    ) else {
        return std::ptr::null_mut();
    };
    let mut bits = [true; 9];
    if !mask.is_null() {
        let mv = unsafe { slice::from_raw_parts(mask, 9) };
        for i in 0..9 {
            bits[i] = mv[i] != 0;
        }
    }
    let mut sella = SellaSaddleConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    if cfg.order > 0 {
        sella.order = cfg.order as usize;
    }
    sella.force_gate = force_gate;
    match SellaSaddleSession::on_cell(sella, x, m, lc, bits) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaSaddle { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `cons` is copied. `cell` is 9 doubles. `mask` is 9 ints or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_create_cell_internal(
    config: *const RgsaddleSellaSaddleConfig,
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
    cell: *const f64,
    mask: *const i32,
) -> *mut RgsaddleSellaSaddle {
    if config.is_null()
        || position.is_null()
        || masses.is_null()
        || cons.is_null()
        || cell.is_null()
        || n_atoms < 1
    {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(force_gate) = force_gate_of(cfg.force_gate) else {
        return std::ptr::null_mut();
    };
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let c = unsafe { slice::from_raw_parts(cell, 9) };
    let Ok(lc) = Cell::from_vectors(
        [c[0], c[1], c[2]],
        [c[3], c[4], c[5]],
        [c[6], c[7], c[8]],
        [0.0, 0.0, 0.0],
    ) else {
        return std::ptr::null_mut();
    };
    let mut bits = [true; 9];
    if !mask.is_null() {
        let mv = unsafe { slice::from_raw_parts(mask, 9) };
        for i in 0..9 {
            bits[i] = mv[i] != 0;
        }
    }
    let mut sella = SellaSaddleConfig::default();
    if cfg.delta > 0.0 {
        sella.delta = cfg.delta;
    }
    if cfg.force_tol > 0.0 {
        sella.force_tol = cfg.force_tol;
    }
    if cfg.order > 0 {
        sella.order = cfg.order as usize;
    }
    sella.force_gate = force_gate;
    match SellaSaddleSession::on_cell_internal(sella, x, m, chart.cons.clone(), lc, bits) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSellaSaddle { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_step(
    session: *mut RgsaddleSellaSaddle,
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
            out.status = if report.at_saddle { 1 } else { 0 };
            out.reserved = 0;
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = 0;
            out.curvature = report.rho;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_position(
    session: *const RgsaddleSellaSaddle,
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
/// `session` must be a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_reset(session: *mut RgsaddleSellaSaddle) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `session` must be a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_set_hess_update(
    session: *mut RgsaddleSellaSaddle,
    update: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::HessUpdate::try_from_abi(update) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*session).session.set_update(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `expand` is ExpandKind. Unknown refuses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_set_expand(
    session: *mut RgsaddleSellaSaddle,
    expand: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::ExpandKind::try_from_abi(expand) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*session).session.set_expand(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `restricted` is RestrictedKind. Unknown refuses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_set_restricted(
    session: *mut RgsaddleSellaSaddle,
    restricted: i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::RestrictedKind::try_from_abi(restricted) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*session).session.set_restricted(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `applied` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_maybe_niggli(
    session: *mut RgsaddleSellaSaddle,
    angle_threshold: f64,
    applied: *mut i32,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    match unsafe { (*session).session.maybe_niggli_reduce(angle_threshold) } {
        Ok(did) => {
            if !applied.is_null() {
                unsafe { *applied = if did { 1 } else { 0 } };
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `session` must come from [`rgsaddle_sella_saddle_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_sella_saddle_free(session: *mut RgsaddleSellaSaddle) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// # Safety
/// `config` must be a valid pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_create(
    config: *const RgsaddleConstraintsConfig,
    n_atoms: i64,
) -> *mut RgsaddleConstraints {
    if config.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    match Constraints::new(n_atoms as usize) {
        Ok(cons) => Box::into_raw(Box::new(RgsaddleConstraints { cons, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `x` is 3N. `cons` is a live chart or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_fix_com(
    cons: *mut RgsaddleConstraints,
    x: *const f64,
) -> i32 {
    if cons.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if x.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cons = unsafe { &mut *cons };
    let dof = (3 * cons.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(x, dof) }.to_vec());
    match cons.cons.fix_com(x.view()) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `x` is 3N. `target` is one double or NULL (current length).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_fix_bond(
    cons: *mut RgsaddleConstraints,
    i: i64,
    j: i64,
    x: *const f64,
    target: *const f64,
) -> i32 {
    if cons.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if x.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cons = unsafe { &mut *cons };
    if i < 0 || j < 0 || i >= cons.n_atoms || j >= cons.n_atoms {
        return RGSADDLE_SHAPE;
    }
    let dof = (3 * cons.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(x, dof) }.to_vec());
    let target = if target.is_null() {
        None
    } else {
        Some(unsafe { *target })
    };
    match cons
        .cons
        .fix_bond([i as usize, j as usize], x.view(), target)
    {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `x` is 3N. `out` is one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_residual_norm(
    cons: *const RgsaddleConstraints,
    x: *const f64,
    out: *mut f64,
) -> i32 {
    if cons.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if x.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cons = unsafe { &*cons };
    let dof = (3 * cons.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(x, dof) }.to_vec());
    match cons.cons.residual_norm(x.view()) {
        Ok(n) => {
            unsafe { *out = n };
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `x`, `v`, and `out` are 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_project(
    cons: *const RgsaddleConstraints,
    x: *const f64,
    v: *const f64,
    out: *mut f64,
) -> i32 {
    if cons.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if x.is_null() || v.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cons = unsafe { &*cons };
    let dof = (3 * cons.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(x, dof) }.to_vec());
    let v = Array1::from(unsafe { slice::from_raw_parts(v, dof) }.to_vec());
    let p = cons.cons.project(&x, &v);
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (k, val) in p.iter().enumerate() {
        dst[k] = *val;
    }
    RGSADDLE_OK
}

/// # Safety
/// `x`, `v`, and `out` are 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_retract(
    cons: *const RgsaddleConstraints,
    x: *const f64,
    v: *const f64,
    out: *mut f64,
) -> i32 {
    if cons.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if x.is_null() || v.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let cons = unsafe { &*cons };
    let dof = (3 * cons.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(x, dof) }.to_vec());
    let v = Array1::from(unsafe { slice::from_raw_parts(v, dof) }.to_vec());
    let y = cons.cons.retract(&x, &v);
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (k, val) in y.iter().enumerate() {
        dst[k] = *val;
    }
    RGSADDLE_OK
}

/// # Safety
/// `cons` must come from [`rgsaddle_constraints_create`] and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_constraints_free(cons: *mut RgsaddleConstraints) {
    if !cons.is_null() {
        drop(unsafe { Box::from_raw(cons) });
    }
}

/// # Safety
/// `x` and `v0` are 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_samd_create(
    config: *const RgsaddleSamdConfig,
    n_atoms: i64,
    x: *const f64,
    v0: *const f64,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
) -> *mut RgsaddleSamd {
    if config.is_null() || x.is_null() || v0.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let Some(f) = surface else {
        return std::ptr::null_mut();
    };
    let dof = (3 * n_atoms) as usize;
    let pos = Array1::from(unsafe { slice::from_raw_parts(x, dof) }.to_vec());
    let vel = Array1::from(unsafe { slice::from_raw_parts(v0, dof) }.to_vec());
    let mut samd = SamdConfig::default();
    if cfg.dt > 0.0 {
        samd.dt = cfg.dt;
    }
    if cfg.tau > 0.0 {
        samd.tau = cfg.tau;
    }
    if cfg.t0 > 0.0 {
        samd.t0 = cfg.t0;
    }
    if cfg.tf > 0.0 {
        samd.tf = cfg.tf;
    }
    if cfg.ngen > 0 {
        samd.ngen = cfg.ngen as usize;
    }
    samd.exponential = cfg.exponential != 0;
    let cs = CSurface { f, user, n_atoms };
    match SamdSession::new(samd, pos, vel, &cs) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleSamd { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `r` is 3N. `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_samd_step(
    session: *mut RgsaddleSamd,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    r: *const f64,
    out: *mut RgsaddleReport,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    if r.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    let session = unsafe { &mut *session };
    let dof = (3 * session.n_atoms) as usize;
    let draw = Array1::from(unsafe { slice::from_raw_parts(r, dof) }.to_vec());
    let cs = CSurface {
        f,
        user,
        n_atoms: session.n_atoms,
    };
    match session.session.step(&cs, draw.view()) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = 0;
            out.reserved = 0;
            out.max_force = report.kinetic;
            out.ci_index = -1;
            out.iteration = 0;
            out.curvature = report.temperature;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_samd_position(
    session: *const RgsaddleSamd,
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
/// `session` must come from [`rgsaddle_samd_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_samd_free(session: *mut RgsaddleSamd) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// # Safety
/// `position` is 3N, `masses` is N. Nonzero `proj_trans` / `proj_rot`
/// hang fix_translation / fix_rotation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_create(
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    proj_trans: i32,
    proj_rot: i32,
) -> *mut RgsaddlePes {
    if position.is_null() || masses.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    match CartesianPes::with_proj(x, m, proj_trans != 0, proj_rot != 0) {
        Ok(pes) => Box::into_raw(Box::new(RgsaddlePes { pes, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `out` is 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_position(pes: *const RgsaddlePes, out: *mut f64) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let dof = (3 * pes.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (i, v) in pes.pes.position().iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `d` is 3N. Surface is one image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_kick(
    pes: *mut RgsaddlePes,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    d: *const f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    if d.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &mut *pes };
    let dof = (3 * pes.n_atoms) as usize;
    let dx = Array1::from(unsafe { slice::from_raw_parts(d, dof) }.to_vec());
    let cs = CSurface {
        f,
        user,
        n_atoms: pes.n_atoms,
    };
    match pes.pes.kick(&cs, dx.view()) {
        Ok(_) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `update` is 0 = BFGS, 1 = TS-BFGS.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_set_hess_update(pes: *mut RgsaddlePes, update: i32) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::HessUpdate::try_from_abi(update) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*pes).pes.set_update(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `v` and `out` are 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_project(
    pes: *const RgsaddlePes,
    v: *const f64,
    out: *mut f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if v.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let dof = (3 * pes.n_atoms) as usize;
    let v = Array1::from(unsafe { slice::from_raw_parts(v, dof) }.to_vec());
    let p = pes.pes.project(&v);
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (k, val) in p.iter().enumerate() {
        dst[k] = *val;
    }
    RGSADDLE_OK
}

/// # Safety
/// `v` and `out` are 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_retract(
    pes: *const RgsaddlePes,
    v: *const f64,
    out: *mut f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if v.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let dof = (3 * pes.n_atoms) as usize;
    let v = Array1::from(unsafe { slice::from_raw_parts(v, dof) }.to_vec());
    let y = pes.pes.retract(&v);
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (k, val) in y.iter().enumerate() {
        dst[k] = *val;
    }
    RGSADDLE_OK
}

/// # Safety
/// `x_to`, `v`, and `out` are 3N.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_transport(
    pes: *const RgsaddlePes,
    x_to: *const f64,
    v: *const f64,
    out: *mut f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if x_to.is_null() || v.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let dof = (3 * pes.n_atoms) as usize;
    let x_from = pes.pes.position().to_owned();
    let x_to = Array1::from(unsafe { slice::from_raw_parts(x_to, dof) }.to_vec());
    let v = Array1::from(unsafe { slice::from_raw_parts(v, dof) }.to_vec());
    let t = pes.pes.transport(&x_from, &x_to, &v);
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (k, val) in t.iter().enumerate() {
        dst[k] = *val;
    }
    RGSADDLE_OK
}

/// # Safety
/// `pes` must come from [`rgsaddle_pes_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_pes_free(pes: *mut RgsaddlePes) {
    if !pes.is_null() {
        drop(unsafe { Box::from_raw(pes) });
    }
}

/// # Safety
/// `position` is 3N, `masses` is N. `cons` is copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_create(
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
) -> *mut RgsaddleInternalPes {
    if position.is_null() || masses.is_null() || cons.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    match InternalPes::new(x, m, chart.cons.clone()) {
        Ok(pes) => Box::into_raw(Box::new(RgsaddleInternalPes { pes, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `dummies` is 3 * n_dummy. `cons` covers n_atoms + n_dummy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_create_dummies(
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    cons: *const RgsaddleConstraints,
    n_dummy: i64,
    dummies: *const f64,
) -> *mut RgsaddleInternalPes {
    if position.is_null()
        || masses.is_null()
        || cons.is_null()
        || dummies.is_null()
        || n_atoms < 1
        || n_dummy < 1
    {
        return std::ptr::null_mut();
    }
    let chart = unsafe { &*cons };
    if chart.n_atoms != n_atoms + n_dummy {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n_atoms as usize) }.to_vec());
    let d =
        Array1::from(unsafe { slice::from_raw_parts(dummies, (3 * n_dummy) as usize) }.to_vec());
    match InternalPes::with_dummies(x, m, chart.cons.clone(), d) {
        Ok(pes) => Box::into_raw(Box::new(RgsaddleInternalPes { pes, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `position` is 3N, `masses` is N. `bonds` is 2 * n_bonds when
/// n_bonds > 0, else NULL. Same for angles (3) and dihedrals (4).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_create_from_find(
    n_atoms: i64,
    position: *const f64,
    masses: *const f64,
    n_bonds: i64,
    bonds: *const i64,
    n_angles: i64,
    angles: *const i64,
    n_dihedrals: i64,
    dihedrals: *const i64,
) -> *mut RgsaddleInternalPes {
    if position.is_null() || masses.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    if n_bonds < 0 || n_angles < 0 || n_dihedrals < 0 {
        return std::ptr::null_mut();
    }
    if n_bonds > 0 && bonds.is_null() {
        return std::ptr::null_mut();
    }
    if n_angles > 0 && angles.is_null() {
        return std::ptr::null_mut();
    }
    if n_dihedrals > 0 && dihedrals.is_null() {
        return std::ptr::null_mut();
    }
    let n = n_atoms as usize;
    let mut bond_v = Vec::with_capacity(n_bonds as usize);
    if n_bonds > 0 {
        let raw = unsafe { slice::from_raw_parts(bonds, 2 * n_bonds as usize) };
        for chunk in raw.as_chunks::<2>().0 {
            if chunk[0] < 0 || chunk[1] < 0 {
                return std::ptr::null_mut();
            }
            bond_v.push([chunk[0] as usize, chunk[1] as usize]);
        }
    }
    let mut angle_v = Vec::with_capacity(n_angles as usize);
    if n_angles > 0 {
        let raw = unsafe { slice::from_raw_parts(angles, 3 * n_angles as usize) };
        for chunk in raw.as_chunks::<3>().0 {
            if chunk.iter().any(|&a| a < 0) {
                return std::ptr::null_mut();
            }
            angle_v.push([chunk[0] as usize, chunk[1] as usize, chunk[2] as usize]);
        }
    }
    let mut dih_v = Vec::with_capacity(n_dihedrals as usize);
    if n_dihedrals > 0 {
        let raw = unsafe { slice::from_raw_parts(dihedrals, 4 * n_dihedrals as usize) };
        for chunk in raw.as_chunks::<4>().0 {
            if chunk.iter().any(|&a| a < 0) {
                return std::ptr::null_mut();
            }
            dih_v.push([
                chunk[0] as usize,
                chunk[1] as usize,
                chunk[2] as usize,
                chunk[3] as usize,
            ]);
        }
    }
    let found = crate::vocn::Found::from_parts(bond_v, angle_v, dih_v);
    let dof = 3 * n;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let m = Array1::from(unsafe { slice::from_raw_parts(masses, n) }.to_vec());
    match InternalPes::from_find(x, m, &found) {
        Ok(pes) => Box::into_raw(Box::new(RgsaddleInternalPes { pes, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `x` is 3N, `out` is 3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_place_perp_dummy(
    n_atoms: i64,
    x: *const f64,
    i: i64,
    j: i64,
    out: *mut f64,
) -> i32 {
    if x.is_null() || out.is_null() || n_atoms < 2 {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pos = Array1::from(unsafe { slice::from_raw_parts(x, (3 * n_atoms) as usize) }.to_vec());
    match crate::place_perp_dummy(pos.view(), i as usize, j as usize) {
        Ok(r) => {
            let dst = unsafe { slice::from_raw_parts_mut(out, 3) };
            dst[0] = r[0];
            dst[1] = r[1];
            dst[2] = r[2];
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` is one i64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_n_int(
    pes: *const RgsaddleInternalPes,
    out: *mut i64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    unsafe { *out = (*pes).pes.n_int() as i64 };
    RGSADDLE_OK
}

/// # Safety
/// `out` holds n_int doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_internals(
    pes: *const RgsaddleInternalPes,
    out: *mut f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let q = match pes.pes.internals() {
        Ok(q) => q,
        Err(e) => return status_of(&e),
    };
    let dst = unsafe { slice::from_raw_parts_mut(out, q.len()) };
    for (i, v) in q.iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `out` holds 3N doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_position(
    pes: *const RgsaddleInternalPes,
    out: *mut f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let dof = (3 * pes.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, dof) };
    for (i, v) in pes.pes.position().iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `dq` is n_int. Surface is one image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_kick(
    pes: *mut RgsaddleInternalPes,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    dq: *const f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    if dq.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &mut *pes };
    let nint = pes.pes.n_int();
    let q = Array1::from(unsafe { slice::from_raw_parts(dq, nint) }.to_vec());
    let cs = CSurface {
        f,
        user,
        n_atoms: pes.n_atoms,
    };
    match pes.pes.kick(&cs, q.view()) {
        Ok(_) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `update` is 0 = BFGS, 1 = TS-BFGS.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_set_hess_update(
    pes: *mut RgsaddleInternalPes,
    update: i32,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let Some(kind) = crate::HessUpdate::try_from_abi(update) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    unsafe { (*pes).pes.set_update(kind) };
    RGSADDLE_OK
}

/// # Safety
/// `g_cart` is 3N, `out` is n_int.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_grad(
    pes: *const RgsaddleInternalPes,
    g_cart: *const f64,
    out: *mut f64,
) -> i32 {
    if pes.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if g_cart.is_null() || out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let pes = unsafe { &*pes };
    let dof = (3 * pes.n_atoms) as usize;
    let g = Array1::from(unsafe { slice::from_raw_parts(g_cart, dof) }.to_vec());
    let gint = match pes.pes.internals_grad(g.view()) {
        Ok(q) => q,
        Err(e) => return status_of(&e),
    };
    let dst = unsafe { slice::from_raw_parts_mut(out, gint.len()) };
    for (i, v) in gint.iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// Freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_internal_pes_free(pes: *mut RgsaddleInternalPes) {
    if !pes.is_null() {
        drop(unsafe { Box::from_raw(pes) });
    }
}

/// # Safety
/// `cell` is 9 doubles, row-major. `applied` may be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_niggli_reduce(
    cell: *mut f64,
    angle_threshold: f64,
    applied: *mut i32,
) -> i32 {
    if cell.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let sl = unsafe { slice::from_raw_parts_mut(cell, 9) };
    let mut arr = [0.0; 9];
    arr.copy_from_slice(sl);
    match crate::niggli_reduce_cell(&mut arr, angle_threshold) {
        Ok(did) => {
            sl.copy_from_slice(&arr);
            if !applied.is_null() {
                unsafe { *applied = if did { 1 } else { 0 } };
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

#[cfg(test)]
mod constraints_abi_tests {
    use super::*;
    use rgmin::vecops::nrm2;

    fn water() -> [f64; 9] {
        [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]
    }

    fn stamped_cfg() -> RgsaddleConstraintsConfig {
        RgsaddleConstraintsConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
        }
    }

    #[test]
    fn constraints_abi_com_kills_a_translation() {
        let x = water();
        let cfg = stamped_cfg();
        let cons = unsafe { rgsaddle_constraints_create(&cfg, 3) };
        assert!(!cons.is_null());
        assert_eq!(
            unsafe { rgsaddle_constraints_fix_com(cons, x.as_ptr()) },
            RGSADDLE_OK
        );
        let mut r = f64::NAN;
        assert_eq!(
            unsafe { rgsaddle_constraints_residual_norm(cons, x.as_ptr(), &mut r) },
            RGSADDLE_OK
        );
        assert!(r < 1e-14, "r={r}");

        let shift = [0.1; 9];
        let mut out = [0.0; 9];
        assert_eq!(
            unsafe {
                rgsaddle_constraints_project(cons, x.as_ptr(), shift.as_ptr(), out.as_mut_ptr())
            },
            RGSADDLE_OK
        );
        let n = nrm2(Array1::from(out.to_vec()).view());
        assert!(n < 1e-12, "projected translation {n}");
        unsafe { rgsaddle_constraints_free(cons) };
    }

    extern "C" fn well_cb(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
        unsafe {
            let req = &mut *req;
            let n = (req.n_atoms * 3) as usize;
            let pos = slice::from_raw_parts(req.positions, n);
            let e = req.energies;
            let g = slice::from_raw_parts_mut(req.gradients, n);
            let t = pos[0];
            *e = (t * t - 1.0).powi(2);
            for gi in g.iter_mut() {
                *gi = 0.0;
            }
            g[0] = 4.0 * t * (t * t - 1.0);
        }
        0
    }

    #[test]
    fn sella_min_create_on_keeps_a_fixed_com() {
        let x = water();
        let masses = [1.0, 1.0, 1.0];
        let cfg_c = stamped_cfg();
        let cons = unsafe { rgsaddle_constraints_create(&cfg_c, 3) };
        assert!(!cons.is_null());
        assert_eq!(
            unsafe { rgsaddle_constraints_fix_com(cons, x.as_ptr()) },
            RGSADDLE_OK
        );
        let cfg = RgsaddleSellaMinConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            delta: 0.1,
            force_tol: 0.05,
            force_gate: crate::ForceGate::MaxForceOnAtom.to_abi(),
        };
        let sess =
            unsafe { rgsaddle_sella_min_create_on(&cfg, 3, x.as_ptr(), masses.as_ptr(), cons) };
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
        assert_eq!(
            unsafe {
                rgsaddle_sella_min_step(sess, Some(well_cb), std::ptr::null_mut(), &mut report)
            },
            RGSADDLE_OK
        );
        let mut y = [0.0; 9];
        assert_eq!(
            unsafe { rgsaddle_sella_min_position(sess, y.as_mut_ptr()) },
            RGSADDLE_OK
        );
        let c0 = [
            (x[0] + x[3] + x[6]) / 3.0,
            (x[1] + x[4] + x[7]) / 3.0,
            (x[2] + x[5] + x[8]) / 3.0,
        ];
        let c1 = [
            (y[0] + y[3] + y[6]) / 3.0,
            (y[1] + y[4] + y[7]) / 3.0,
            (y[2] + y[5] + y[8]) / 3.0,
        ];
        assert!(
            (c0[0] - c1[0]).abs() < 1e-8
                && (c0[1] - c1[1]).abs() < 1e-8
                && (c0[2] - c1[2]).abs() < 1e-8,
            "COM drifted {c0:?} -> {c1:?}"
        );
        unsafe {
            rgsaddle_sella_min_free(sess);
            rgsaddle_constraints_free(cons);
        }
    }

    #[test]
    fn internal_pes_abi_from_find_h2o_n_int_matches() {
        let x = water();
        let masses = [16.0, 1.0, 1.0];
        let bonds = [0i64, 1, 0, 2];
        let angles = [1i64, 0, 2];
        let pes = unsafe {
            rgsaddle_internal_pes_create_from_find(
                3,
                x.as_ptr(),
                masses.as_ptr(),
                2,
                bonds.as_ptr(),
                1,
                angles.as_ptr(),
                0,
                std::ptr::null(),
            )
        };
        assert!(!pes.is_null());
        let mut nint = 0i64;
        assert_eq!(
            unsafe { rgsaddle_internal_pes_n_int(pes, &mut nint) },
            RGSADDLE_OK
        );
        assert_eq!(nint, 3);
        let mut q = [0.0; 3];
        assert_eq!(
            unsafe { rgsaddle_internal_pes_internals(pes, q.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(q.iter().all(|v| v.is_finite()));
        unsafe { rgsaddle_internal_pes_free(pes) };
        let empty = unsafe {
            rgsaddle_internal_pes_create_from_find(
                3,
                x.as_ptr(),
                masses.as_ptr(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            )
        };
        assert!(empty.is_null());
    }

    #[test]
    fn internal_pes_abi_kick_moves_the_com() {
        let x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let masses = [1.0, 1.0];
        let cfg = stamped_cfg();
        let cons = unsafe { rgsaddle_constraints_create(&cfg, 2) };
        assert!(!cons.is_null());
        assert_eq!(
            unsafe { rgsaddle_constraints_fix_com(cons, x.as_ptr()) },
            RGSADDLE_OK
        );
        let pes = unsafe { rgsaddle_internal_pes_create(2, x.as_ptr(), masses.as_ptr(), cons) };
        assert!(!pes.is_null());
        let mut nint = 0i64;
        assert_eq!(
            unsafe { rgsaddle_internal_pes_n_int(pes, &mut nint) },
            RGSADDLE_OK
        );
        assert_eq!(nint, 3);
        let dq = [0.2, 0.0, 0.0];
        assert_eq!(
            unsafe {
                rgsaddle_internal_pes_kick(pes, Some(well_cb), std::ptr::null_mut(), dq.as_ptr())
            },
            RGSADDLE_OK
        );
        let mut y = [0.0; 6];
        assert_eq!(
            unsafe { rgsaddle_internal_pes_position(pes, y.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!((y[0] - 0.2).abs() < 1e-10, "{y:?}");
        assert!((y[3] - 0.2).abs() < 1e-10, "{y:?}");
        assert_eq!(
            unsafe { rgsaddle_internal_pes_set_hess_update(pes, 0) },
            RGSADDLE_OK
        );
        assert_eq!(
            unsafe { rgsaddle_internal_pes_set_hess_update(pes, 99) },
            RGSADDLE_INVALID_PARAMETER
        );
        let g_cart = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut gint = [0.0; 3];
        assert_eq!(
            unsafe { rgsaddle_internal_pes_grad(pes, g_cart.as_ptr(), gint.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(gint.iter().all(|v| v.is_finite()));
        let min_cfg = RgsaddleSellaMinConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            delta: 0.1,
            force_tol: 0.05,
            force_gate: crate::ForceGate::MaxForceOnAtom.to_abi(),
        };
        let sess = unsafe {
            rgsaddle_sella_min_create_internal(&min_cfg, 2, x.as_ptr(), masses.as_ptr(), cons)
        };
        assert!(!sess.is_null());
        let cell = [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0];
        let mask = [1, 0, 0, 0, 0, 0, 0, 0, 0];
        let cell_sess = unsafe {
            rgsaddle_sella_min_create_cell(
                &min_cfg,
                2,
                x.as_ptr(),
                masses.as_ptr(),
                cell.as_ptr(),
                mask.as_ptr(),
            )
        };
        assert!(!cell_sess.is_null());
        let cint = unsafe {
            rgsaddle_sella_min_create_cell_internal(
                &min_cfg,
                2,
                x.as_ptr(),
                masses.as_ptr(),
                cons,
                cell.as_ptr(),
                mask.as_ptr(),
            )
        };
        assert!(!cint.is_null());
        unsafe {
            rgsaddle_sella_min_free(sess);
            rgsaddle_sella_min_free(cell_sess);
            rgsaddle_sella_min_free(cint);
            rgsaddle_internal_pes_free(pes);
            rgsaddle_constraints_free(cons);
        }
        assert!(
            unsafe {
                rgsaddle_internal_pes_create(2, x.as_ptr(), masses.as_ptr(), std::ptr::null())
            }
            .is_null()
        );
        assert!(
            unsafe {
                rgsaddle_sella_min_create_cell(
                    &min_cfg,
                    2,
                    x.as_ptr(),
                    masses.as_ptr(),
                    std::ptr::null(),
                    mask.as_ptr(),
                )
            }
            .is_null()
        );
    }

    #[test]
    fn niggli_abi_rewrites_a_skewed_cell() {
        let mut cell = [1.0, 0.0, 0.0, 0.9, 0.15, 0.0, 0.4, 0.5, 1.0];
        let mut applied = 0i32;
        assert_eq!(
            unsafe { rgsaddle_niggli_reduce(cell.as_mut_ptr(), 20.0, &mut applied) },
            RGSADDLE_OK
        );
        assert_eq!(applied, 1);
        assert_eq!(
            unsafe { rgsaddle_niggli_reduce(std::ptr::null_mut(), 20.0, &mut applied) },
            RGSADDLE_INVALID_PARAMETER
        );
    }

    #[test]
    fn constraints_abi_unknown_or_null_refuses_create() {
        let cfg = stamped_cfg();
        assert!(unsafe { rgsaddle_constraints_create(std::ptr::null(), 3) }.is_null());
        assert!(unsafe { rgsaddle_constraints_create(&cfg, 0) }.is_null());
        let bad = RgsaddleConstraintsConfig {
            version: RgsaddleVersion {
                major: 99,
                minor: 0,
            },
            flags: 0,
        };
        assert!(unsafe { rgsaddle_constraints_create(&bad, 3) }.is_null());
    }

    #[test]
    fn pes_abi_proj_trans_kills_a_translation() {
        let x = [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0];
        let masses = [16.0, 1.0, 1.0];
        let pes = unsafe { rgsaddle_pes_create(3, x.as_ptr(), masses.as_ptr(), 1, 1) };
        assert!(!pes.is_null());
        let v = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut p = [0.0; 9];
        assert_eq!(
            unsafe { rgsaddle_pes_project(pes, v.as_ptr(), p.as_mut_ptr()) },
            RGSADDLE_OK
        );
        let n = p.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(n < 1e-10, "translation survived: {p:?}");
        let mut step = [0.0; 9];
        step[0] = 0.2;
        step[4] = -0.15;
        step[8] = 0.05;
        let mut y = [0.0; 9];
        assert_eq!(
            unsafe { rgsaddle_pes_retract(pes, step.as_ptr(), y.as_mut_ptr()) },
            RGSADDLE_OK
        );
        let c0 = [
            (x[0] + x[3] + x[6]) / 3.0,
            (x[1] + x[4] + x[7]) / 3.0,
            (x[2] + x[5] + x[8]) / 3.0,
        ];
        let c1 = [
            (y[0] + y[3] + y[6]) / 3.0,
            (y[1] + y[4] + y[7]) / 3.0,
            (y[2] + y[5] + y[8]) / 3.0,
        ];
        assert!((c0[0] - c1[0]).abs() < 1e-10);
        assert!((c0[1] - c1[1]).abs() < 1e-10);
        assert!((c0[2] - c1[2]).abs() < 1e-10);
        let mut t = [0.0; 9];
        assert_eq!(
            unsafe { rgsaddle_pes_transport(pes, y.as_ptr(), p.as_ptr(), t.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(t.iter().all(|v| v.is_finite()));
        let mut d = [0.0; 9];
        d[0] = 0.05;
        assert_eq!(
            unsafe { rgsaddle_pes_kick(pes, Some(well_cb), std::ptr::null_mut(), d.as_ptr()) },
            RGSADDLE_OK
        );
        let mut pos = [0.0; 9];
        assert_eq!(
            unsafe { rgsaddle_pes_position(pes, pos.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!((pos[0] - 0.05).abs() < 1e-12);
        assert_eq!(unsafe { rgsaddle_pes_set_hess_update(pes, 0) }, RGSADDLE_OK);
        assert_eq!(
            unsafe { rgsaddle_pes_set_hess_update(pes, 99) },
            RGSADDLE_INVALID_PARAMETER
        );
        unsafe { rgsaddle_pes_free(pes) };
        assert!(unsafe { rgsaddle_pes_create(3, x.as_ptr(), std::ptr::null(), 1, 0) }.is_null());
        assert!(unsafe { rgsaddle_pes_create(0, x.as_ptr(), masses.as_ptr(), 1, 1) }.is_null());
    }
}

#[cfg(test)]
mod samd_abi_tests {
    use super::*;

    extern "C" fn well_cb(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
        unsafe {
            let req = &mut *req;
            let n = (req.n_atoms * 3) as usize;
            let pos = slice::from_raw_parts(req.positions, n);
            let e = req.energies;
            let g = slice::from_raw_parts_mut(req.gradients, n);
            *e = pos[0] * pos[0];
            for gi in g.iter_mut() {
                *gi = 0.0;
            }
            g[0] = 2.0 * pos[0];
        }
        0
    }

    #[test]
    fn samd_abi_step_is_finite() {
        let n_atoms = 2i64;
        let x = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v0 = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = [0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
        let cfg = RgsaddleSamdConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: 0,
            dt: 0.1,
            tau: 1.0,
            t0: 1.0,
            tf: 0.1,
            ngen: 8,
            exponential: 0,
        };
        let sess = unsafe {
            rgsaddle_samd_create(
                &cfg,
                n_atoms,
                x.as_ptr(),
                v0.as_ptr(),
                Some(well_cb),
                std::ptr::null_mut(),
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
        assert_eq!(
            unsafe {
                rgsaddle_samd_step(
                    sess,
                    Some(well_cb),
                    std::ptr::null_mut(),
                    r.as_ptr(),
                    &mut report,
                )
            },
            RGSADDLE_OK
        );
        assert!(report.max_force.is_finite());
        let mut y = [0.0; 6];
        assert_eq!(
            unsafe { rgsaddle_samd_position(sess, y.as_mut_ptr()) },
            RGSADDLE_OK
        );
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(
            unsafe {
                rgsaddle_samd_create(
                    std::ptr::null(),
                    n_atoms,
                    x.as_ptr(),
                    v0.as_ptr(),
                    Some(well_cb),
                    std::ptr::null_mut(),
                )
            }
            .is_null()
        );
        unsafe { rgsaddle_samd_free(sess) };
    }
}
