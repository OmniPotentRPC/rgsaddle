//! C entry points for the climbing-image hand-off.

use std::ffi::c_void;
use std::slice;

use ndarray::Array2;

use super::{
    CSurface, RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_INVALID_PARAMETER,
    RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SESSION, RGSADDLE_NULL_SURFACE, RGSADDLE_OK,
    RgsaddleBandConfig, RgsaddleMinModeConfig, RgsaddleVersion, minmode_config_from,
    parse_band_config, status_of,
};
use crate::ocineb::{OcinebConfig, OcinebPhase, OcinebSession, OcinebStatus};

#[repr(C)]
pub struct RgsaddleOcinebConfig {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub trigger_factor: f64,
    pub trigger_force: f64,
    pub angle_tol: f64,
    pub stability_count: i64,
    pub max_mmf_steps: i64,
    pub restore_unhelpful: i32,
    pub reserved: i32,
}

#[repr(C)]
pub struct RgsaddleOcinebReport {
    pub version: RgsaddleVersion,
    pub flags: u64,
    pub status: i32,
    pub phase: i32,
    pub fell_back: i32,
    pub evaluations: i32,
    pub max_force: f64,
    pub curvature: f64,
    pub alignment: f64,
    pub threshold: f64,
    pub ci_index: i64,
    pub iteration: i64,
    pub rotations: i64,
}

pub struct RgsaddleOcineb {
    session: OcinebSession,
    n_images: i64,
    n_atoms: i64,
    per_image: bool,
}

fn stamp(out: &mut RgsaddleOcinebReport) {
    out.version = RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    };
    out.flags = 0;
}

/// # Safety
/// The three configs must be live. `positions` holds
/// `n_images * 3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_create(
    band: *const RgsaddleBandConfig,
    minmode: *const RgsaddleMinModeConfig,
    config: *const RgsaddleOcinebConfig,
    n_images: i64,
    n_atoms: i64,
    positions: *const f64,
) -> *mut RgsaddleOcineb {
    if band.is_null() || minmode.is_null() || config.is_null() || positions.is_null() {
        return std::ptr::null_mut();
    }
    if n_images < 3 || n_atoms < 1 {
        return std::ptr::null_mut();
    }
    let band = unsafe { &*band };
    let minmode = unsafe { &*minmode };
    let config = unsafe { &*config };
    if config.version.major != RGSADDLE_ABI_MAJOR || minmode.method == 3 {
        return std::ptr::null_mut();
    }
    let Some(dof) = super::f64_len(&[3, n_atoms]) else {
        return std::ptr::null_mut();
    };
    let Some(parsed) = (unsafe { parse_band_config(band, n_images, dof, None) }) else {
        return std::ptr::null_mut();
    };
    let Some(minmode_config) = minmode_config_from(minmode) else {
        return std::ptr::null_mut();
    };
    let Some(total) = super::f64_len(&[n_images, 3, n_atoms]) else {
        return std::ptr::null_mut();
    };
    let Some(n_rows) = usize::try_from(n_images).ok() else {
        return std::ptr::null_mut();
    };
    let flat = unsafe { slice::from_raw_parts(positions, total) };
    let Ok(initial) = Array2::from_shape_vec((n_rows, dof), flat.to_vec()) else {
        return std::ptr::null_mut();
    };
    let per_image = parsed.per_image;
    let method = parsed.method;
    let max_move = parsed.config.max_move;
    let ocineb = OcinebConfig::default()
        .with_band(parsed.config)
        .with_minmode(minmode_config)
        .with_trigger_factor(config.trigger_factor)
        .with_trigger_force(config.trigger_force)
        .with_angle_tol(config.angle_tol)
        .with_stability_count(config.stability_count)
        .with_max_mmf_steps(config.max_mmf_steps as usize)
        .with_restore_unhelpful(config.restore_unhelpful != 0);
    let Ok(mut session) = OcinebSession::new(ocineb, initial) else {
        return std::ptr::null_mut();
    };
    if method == 2
        && session
            .set_rtr(Some(
                crate::rtr::RtrConfig::default()
                    .with_radius_max(max_move * ((n_images - 2) as f64).sqrt())
                    .with_reparametrization(super::rtr_reparametrization(band)),
            ))
            .is_err()
    {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(RgsaddleOcineb {
        session,
        n_images,
        n_atoms,
        per_image,
    }))
}

/// # Safety
/// `session` and `out` must be valid. `surface` is called with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_step(
    session: *mut RgsaddleOcineb,
    surface: Option<super::RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleOcinebReport,
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
        n_images: session.n_images,
        per_image: session.per_image,
        climbing_request: false,
        rows: Default::default(),
    };
    match session.session.step(&cs) {
        Ok(report) => {
            let out = unsafe { &mut *out };
            stamp(out);
            out.status = if report.status == OcinebStatus::Converged {
                1
            } else {
                0
            };
            out.phase = match report.phase {
                OcinebPhase::Band => 0,
                OcinebPhase::Align => 1,
                OcinebPhase::MinMode => 2,
            };
            out.fell_back = i32::from(report.fell_back);
            out.evaluations = i32::try_from(report.evaluations).unwrap_or(i32::MAX);
            out.max_force = report.max_force;
            out.curvature = report.curvature;
            out.alignment = report.alignment;
            out.threshold = report.threshold;
            out.ci_index = report.ci_index.map(|i| i as i64).unwrap_or(-1);
            out.iteration = report.iteration as i64;
            out.rotations = report.rotations as i64;
            RGSADDLE_OK
        }
        Err(e) => status_of(&e),
    }
}

/// # Safety
/// `out` must hold `n_images * 3 * n_atoms` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_positions(
    session: *const RgsaddleOcineb,
    out: *mut f64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    if out.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    let session = unsafe { &*session };
    let pos = session.session.positions();
    let n = pos.len();
    let dst = unsafe { slice::from_raw_parts_mut(out, n) };
    for (i, v) in pos.iter().enumerate() {
        dst[i] = *v;
    }
    RGSADDLE_OK
}

/// # Safety
/// `session` must be a live session pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_reset(session: *mut RgsaddleOcineb) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    unsafe { (*session).session.reset() };
    RGSADDLE_OK
}

/// Replace the OCI-NEB session's saddle check; null turns it off.
///
/// # Safety
/// `session` is a live session or null; `check` is null or a live config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_set_saddle_check(
    session: *mut RgsaddleOcineb,
    check: *const super::RgsaddleSaddleCheck,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let check = match unsafe { super::saddle_index::optional_check(check) } {
        Ok(check) => check,
        Err(rc) => return rc,
    };
    match unsafe { &mut *session }.session.set_saddle_check(check) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// The certificate of the OCI-NEB climbing image.
///
/// # Safety
/// `session` is a live session or null; `out` is null or writable and
/// `curvatures` addresses `capacity` doubles when `capacity` is positive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_saddle_index(
    session: *const RgsaddleOcineb,
    out: *mut super::RgsaddleSaddleIndex,
    curvatures: *mut f64,
    capacity: i64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let session = &unsafe { &*session }.session;
    unsafe {
        super::saddle_index::write_index(
            session.saddle_index(),
            session.saddle_evaluations(),
            session.saddle_escapes(),
            out,
            curvatures,
            capacity,
        )
    }
}

/// # Safety
/// `session` must come from [`rgsaddle_ocineb_create`] and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_ocineb_free(session: *mut RgsaddleOcineb) {
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}
