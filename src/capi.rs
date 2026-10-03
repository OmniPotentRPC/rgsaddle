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

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig};
use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface};
use crate::nichols::{
    HessianUpdate, Index1Config, Index1Session, Index1Status, NicholsMode, bofill_update,
    cap_max_abs, nichols_step, powell_update,
};
use crate::prfo::{PrfoKind, restricted_prfo_displacement};
use crate::projection::ProjectionKind;
use crate::spring::SpringKind;
use crate::tangent::TangentKind;

pub const RGSADDLE_ABI_MAJOR: u32 = 1;
pub const RGSADDLE_ABI_MINOR: u32 = 5;

/// Band config bit 0. The C step calls the surface once per image
/// carried by the evaluation: every image on the first evaluation
/// after create or set_positions, the interior images afterwards.
pub const RGSADDLE_BAND_PER_IMAGE: u64 = 1;

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
pub const RGSADDLE_ALLOC: i32 = -10;
pub const RGSADDLE_INVALID_PARAMETER: i32 = -11;
pub const RGSADDLE_NO_EVALUATION: i32 = -12;

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
    /// -1 for the whole band. Otherwise the image this call fills.
    pub image: i64,
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
    /// ABI minor 5: dimer rotation angle tolerance in radians (zero
    /// disables it). Read only when the config's minor is 5 or more.
    pub rotation_angle_tol: f64,
}

#[repr(C)]
pub struct RgsaddleReport {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub status: i32,
    /// Geometries the surface evaluated during the step (band image
    /// rows, or single points for the min-mode and index-1 sessions).
    pub evaluations: i32,
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
    /// Length of the band (1 for a min-mode session). Maps the rows of
    /// an interior-only evaluation back to band image indices.
    n_images: i64,
    per_image: bool,
}

// The C host is responsible for its own thread safety; the sessions
// call the surface only from the thread that called step.
unsafe impl Sync for CSurface {}
unsafe impl Send for CSurface {}

impl CSurface {
    fn call(
        &self,
        n_images: i64,
        image: i64,
        positions: &[f64],
        energies: &mut [f64],
        gradients: &mut [f64],
    ) -> Result<(), SaddleError> {
        let mut req = RgsaddleSurfaceRequest {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags: if image >= 0 { 1 } else { 0 },
            n_images,
            n_atoms: self.n_atoms,
            positions: positions.as_ptr(),
            energies: energies.as_mut_ptr(),
            gradients: gradients.as_mut_ptr(),
            image,
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
        let n_rows = positions.nrows() as i64;
        let dof = gradients.ncols();
        // Row r is band image r on a whole-band evaluation and image
        // r + 1 on an interior-only one (see BandSurface).
        let first_image: i64 = if n_rows == self.n_images { 0 } else { 1 };
        if self.per_image {
            for i in 0..positions.nrows() {
                let flat: Vec<f64> = positions.row(i).iter().copied().collect();
                let mut e = [0.0];
                let mut g = vec![0.0; dof];
                self.call(self.n_images, first_image + i as i64, &flat, &mut e, &mut g)?;
                energies[i] = e[0];
                for c in 0..dof {
                    gradients[(i, c)] = g[c];
                }
            }
            return Ok(());
        }
        // Batched: n_images in the request counts the rows carried.
        let flat: Vec<f64> = positions.iter().copied().collect();
        let mut e = vec![0.0; energies.len()];
        let mut g = vec![0.0; gradients.len()];
        self.call(n_rows, -1, &flat, &mut e, &mut g)?;
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
        self.call(1, -1, &flat, &mut e, &mut g)?;
        Ok((e[0], Array1::from(g)))
    }
}

pub struct RgsaddleBand {
    session: BandSession,
    n_images: i64,
    n_atoms: i64,
    per_image: bool,
}

pub struct RgsaddleMinMode {
    session: MinModeSession,
    n_atoms: i64,
}

fn method_of(v: i32, memory: i64) -> Method {
    match v {
        1 => Method::Lbfgs {
            memory: if memory > 0 { memory as usize } else { 20 },
        },
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
        SaddleError::Invalid(_) => RGSADDLE_INVALID_PARAMETER,
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
        RGSADDLE_ALLOC => "ALLOC\0",
        RGSADDLE_INVALID_PARAMETER => "INVALID_PARAMETER\0",
        RGSADDLE_NO_EVALUATION => "NO_EVALUATION\0",
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
        Some(Cell([
            [c[0], c[1], c[2]],
            [c[3], c[4], c[5]],
            [c[6], c[7], c[8]],
        ]))
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
        method: method_of(cfg.method, cfg.memory),
    };
    match BandSession::new(band_config, initial) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleBand {
            session,
            n_images,
            n_atoms,
            per_image: cfg.flags & RGSADDLE_BAND_PER_IMAGE != 0,
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
        n_images: band.n_images,
        per_image: band.per_image,
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
            out.evaluations = i32::try_from(report.surface_rows).unwrap_or(i32::MAX);
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
/// `band` must be a live session pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_restart(band: *mut RgsaddleBand) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    unsafe { (*band).session.restart() };
    RGSADDLE_OK
}

/// # Safety
/// Each non-NULL output must hold its documented length:
/// `energies` n_images, `gradients` n_images * 3 * n_atoms,
/// `projected` (n_images - 2) * 3 * n_atoms.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_evaluation(
    band: *const RgsaddleBand,
    energies: *mut f64,
    gradients: *mut f64,
    projected: *mut f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let band = unsafe { &*band };
    let Some(eval) = band.session.evaluation() else {
        return RGSADDLE_NO_EVALUATION;
    };
    let copy = |dst: *mut f64, src: &mut dyn Iterator<Item = f64>, len: usize| {
        if !dst.is_null() {
            let out = unsafe { slice::from_raw_parts_mut(dst, len) };
            for (o, v) in out.iter_mut().zip(src) {
                *o = v;
            }
        }
    };
    copy(
        energies,
        &mut eval.energies.iter().copied(),
        eval.energies.len(),
    );
    copy(
        gradients,
        &mut eval.gradients.iter().copied(),
        eval.gradients.len(),
    );
    copy(
        projected,
        &mut eval.projected.iter().copied(),
        eval.projected.len(),
    );
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
        rotation_angle_tol: if cfg.version.minor >= 5 {
            cfg.rotation_angle_tol
        } else {
            0.0
        },
        max_rotations: cfg.max_rotations as usize,
        krylov_dim: cfg.krylov_dim as usize,
        force_tol: cfg.force_tol,
        max_move: cfg.max_move,
        method: method_of(cfg.method, 0),
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
        n_images: 1,
        per_image: false,
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
            out.evaluations = i32::try_from(report.evaluations).unwrap_or(i32::MAX);
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
/// `session` and `out` must be valid; `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_estimate(
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
        n_images: 1,
        per_image: false,
    };
    match session.session.estimate_mode(&cs) {
        Ok(est) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = 0;
            out.evaluations = i32::try_from(est.evaluations).unwrap_or(i32::MAX);
            out.max_force = f64::NAN;
            out.ci_index = -1;
            out.iteration = 0;
            out.curvature = est.curvature;
            out.rotations = est.rotations as i64;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `position` must hold `3 * n_atoms` doubles; `gradient` is NULL or
/// holds `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_set_position(
    session: *mut RgsaddleMinMode,
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
    let dof = (3 * session.n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let g = (!gradient.is_null())
        .then(|| Array1::from(unsafe { slice::from_raw_parts(gradient, dof) }.to_vec()));
    match session.session.set_position(x, g) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `mode` must hold `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_set_mode(
    session: *mut RgsaddleMinMode,
    mode: *const f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if mode.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &mut *session };
    let dof = (3 * session.n_atoms) as usize;
    let m = Array1::from(unsafe { slice::from_raw_parts(mode, dof) }.to_vec());
    match session.session.set_mode(m) {
        Ok(()) => RGSADDLE_OK,
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

#[repr(C)]
pub struct RgsaddleIndex1Config {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub update: i32,
    pub mode: i32,
    pub trust_radius: f64,
    pub force_tol: f64,
    pub fd_dr: f64,
}

pub struct RgsaddleIndex1 {
    session: Index1Session,
    n_atoms: i64,
}

fn nichols_mode(mode: i32) -> Option<NicholsMode> {
    match mode {
        0 => Some(NicholsMode::Minimize),
        1 => Some(NicholsMode::Index1),
        _ => None,
    }
}

/// # Safety
/// `gradient`, `evals`, and `evecs` are readable. `evecs` holds
/// `n * nmode` doubles, row-major, eigenvector `k` in column `k`.
/// `masses` is `n` doubles or NULL for unit mass. `displacement`
/// holds `n` doubles.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_nichols_step(
    n: i64,
    nmode: i64,
    gradient: *const f64,
    evals: *const f64,
    evecs: *const f64,
    masses: *const f64,
    trust_radius: f64,
    mode: i32,
    displacement: *mut f64,
) -> i32 {
    if n < 1
        || nmode < 1
        || gradient.is_null()
        || evals.is_null()
        || evecs.is_null()
        || displacement.is_null()
    {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let Some(mode) = nichols_mode(mode) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let n = n as usize;
    let nmode = nmode as usize;
    if n.checked_mul(nmode).is_none() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let g = Array1::from(unsafe { slice::from_raw_parts(gradient, n) }.to_vec());
    let ev = Array1::from(unsafe { slice::from_raw_parts(evals, nmode) }.to_vec());
    let vectors = unsafe { slice::from_raw_parts(evecs, n * nmode) }.to_vec();
    let vectors = match Array2::from_shape_vec((n, nmode), vectors) {
        Ok(m) => m,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let mass = if masses.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(masses, n) }.to_vec(),
        ))
    };
    match nichols_step(
        g.view(),
        ev.view(),
        vectors.view(),
        mass.as_ref().map(|m| m.view()),
        trust_radius,
        mode,
    ) {
        Ok(dx) => {
            let dst = unsafe { slice::from_raw_parts_mut(displacement, n) };
            for (i, v) in dx.iter().enumerate() {
                dst[i] = *v;
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `gradient` holds `n` doubles. `hessian` holds `n * n` row-major
/// doubles, the Cartesian energy Hessian. `masses` holds `n` doubles
/// or is NULL for unit mass. `displacement` holds `n` doubles.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_prfo_step(
    n: i64,
    gradient: *const f64,
    hessian: *const f64,
    masses: *const f64,
    trust_radius: f64,
    mode: i32,
    displacement: *mut f64,
) -> i32 {
    if n < 1 || gradient.is_null() || hessian.is_null() || displacement.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let kind = match mode {
        0 => PrfoKind::Minimize,
        1 => PrfoKind::Index1,
        _ => return RGSADDLE_INVALID_PARAMETER,
    };
    let n = n as usize;
    let Some(len) = n.checked_mul(n) else {
        return RGSADDLE_INVALID_PARAMETER;
    };
    let g = Array1::from(unsafe { slice::from_raw_parts(gradient, n) }.to_vec());
    let raw = unsafe { slice::from_raw_parts(hessian, len) }.to_vec();
    let h = match Array2::from_shape_vec((n, n), raw) {
        Ok(m) => m,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let mass = if masses.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(masses, n) }.to_vec(),
        ))
    };
    match restricted_prfo_displacement(
        h.view(),
        g.view(),
        mass.as_ref().map(|m| m.view()),
        trust_radius,
        kind,
    ) {
        Ok(dx) => {
            let dst = unsafe { slice::from_raw_parts_mut(displacement, n) };
            for (i, v) in dx.iter().enumerate() {
                dst[i] = *v;
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `step` and `dgradient` hold `n` doubles. `hessian` holds `n * n`
/// row-major doubles and is updated in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hessian_powell(
    n: i64,
    step: *const f64,
    dgradient: *const f64,
    hessian: *mut f64,
) -> i32 {
    hessian_update(n, step, dgradient, hessian, true)
}

/// # Safety
/// Same layout as [`rgsaddle_hessian_powell`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hessian_bofill(
    n: i64,
    step: *const f64,
    dgradient: *const f64,
    hessian: *mut f64,
) -> i32 {
    hessian_update(n, step, dgradient, hessian, false)
}

fn hessian_update(
    n: i64,
    step: *const f64,
    dgradient: *const f64,
    hessian: *mut f64,
    powell: bool,
) -> i32 {
    if n < 1 || step.is_null() || dgradient.is_null() || hessian.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let n = n as usize;
    let s = Array1::from(unsafe { slice::from_raw_parts(step, n) }.to_vec());
    let y = Array1::from(unsafe { slice::from_raw_parts(dgradient, n) }.to_vec());
    let raw = unsafe { slice::from_raw_parts(hessian, n * n) }.to_vec();
    let mut h = match Array2::from_shape_vec((n, n), raw) {
        Ok(m) => m,
        Err(_) => return RGSADDLE_SHAPE,
    };
    let result = if powell {
        powell_update(&mut h, s.view(), y.view())
    } else {
        bofill_update(&mut h, s.view(), y.view())
    };
    match result {
        Ok(()) => {
            let dst = unsafe { slice::from_raw_parts_mut(hessian, n * n) };
            for (i, v) in h.iter().enumerate() {
                dst[i] = *v;
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `step` holds `n` doubles and is scaled in place.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_cap_max_abs(n: i64, step: *mut f64, trust_radius: f64) -> i32 {
    if n < 1 || step.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let n = n as usize;
    let mut dx = Array1::from(unsafe { slice::from_raw_parts(step, n) }.to_vec());
    match cap_max_abs(&mut dx, trust_radius) {
        Ok(()) => {
            let dst = unsafe { slice::from_raw_parts_mut(step, n) };
            for (i, v) in dx.iter().enumerate() {
                dst[i] = *v;
            }
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `config` and `position` are readable. `position` holds
/// `3 * n_atoms` doubles. `hessian` is `dof * dof` row-major or NULL
/// to build it by finite differences. `masses` is `dof` doubles or
/// NULL for unit mass.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_create(
    config: *const RgsaddleIndex1Config,
    n_atoms: i64,
    position: *const f64,
    hessian: *const f64,
    masses: *const f64,
) -> *mut RgsaddleIndex1 {
    if config.is_null() || position.is_null() || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let cfg = unsafe { &*config };
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return std::ptr::null_mut();
    }
    let dof = (3 * n_atoms) as usize;
    let x = Array1::from(unsafe { slice::from_raw_parts(position, dof) }.to_vec());
    let hess = if hessian.is_null() {
        None
    } else {
        let raw = unsafe { slice::from_raw_parts(hessian, dof * dof) }.to_vec();
        match Array2::from_shape_vec((dof, dof), raw) {
            Ok(m) => Some(m),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let mass = if masses.is_null() {
        None
    } else {
        Some(Array1::from(
            unsafe { slice::from_raw_parts(masses, dof) }.to_vec(),
        ))
    };
    let Some(mode) = nichols_mode(cfg.mode) else {
        return std::ptr::null_mut();
    };
    let update = match cfg.update {
        0 => HessianUpdate::Powell,
        1 => HessianUpdate::Bofill,
        _ => return std::ptr::null_mut(),
    };
    let session_config = Index1Config {
        update,
        mode,
        trust_radius: cfg.trust_radius,
        force_tol: cfg.force_tol,
        fd_dr: cfg.fd_dr,
    };
    match Index1Session::new(session_config, x, hess, mass) {
        Ok(session) => Box::into_raw(Box::new(RgsaddleIndex1 { session, n_atoms })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// # Safety
/// `session` and `out` are valid. `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_step(
    session: *mut RgsaddleIndex1,
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
        n_images: 1,
        per_image: false,
    };
    match session.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp_report(out);
            out.status = if report.status == Index1Status::Converged {
                1
            } else {
                0
            };
            out.evaluations = 0;
            out.max_force = report.max_force;
            out.ci_index = -1;
            out.iteration = report.iteration as i64;
            out.curvature = report.curvature;
            out.rotations = 0;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` holds `3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_position(
    session: *const RgsaddleIndex1,
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
/// `out` holds `dof * dof` doubles. Fails with `RGSADDLE_SHAPE` when
/// the Hessian has not been built yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_hessian(
    session: *const RgsaddleIndex1,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let Some(h) = session.session.hessian() else {
        return RGSADDLE_SHAPE;
    };
    let dof = (3 * session.n_atoms) as usize;
    let dst = unsafe { slice::from_raw_parts_mut(out, dof * dof) };
    for (i, v) in h.iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `session` is a live pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_reset(session: *mut RgsaddleIndex1) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// # Safety
/// `session` comes from [`rgsaddle_index1_create`] and is freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_index1_free(session: *mut RgsaddleIndex1) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}
