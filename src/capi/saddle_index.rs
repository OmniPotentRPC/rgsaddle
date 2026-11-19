//! C entry points for the saddle-index certification (ABI minor 40).

use super::*;
use crate::saddle_index::{SaddleEscape, SaddleIndex, saddle_index};

#[repr(C)]
/// The RgsaddleSaddleCheck.
pub struct RgsaddleSaddleCheck {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// Vectors that enter the Krylov space per expansion.
    pub block: i32,
    /// Lowest curvatures resolved.
    pub lowest: i32,
    /// Hessian actions one certification may spend.
    pub max_actions: i32,
    /// 0 forward differences, 1 central.
    pub difference: i32,
    /// Finite-difference displacement of the actions.
    pub dr: f64,
    /// Ritz residual each resolved pair meets.
    pub residual_tol: f64,
    /// A curvature under `-negative_tol` counts as negative.
    pub negative_tol: f64,
    /// Seed of the pseudo-random start vectors.
    pub seed: u64,
    /// Pushes off a saddle of index two or more; 0 for none.
    pub escape_attempts: i32,
    /// The reserved.
    pub reserved: i32,
    /// Largest per-atom displacement of the first push.
    pub escape_push: f64,
}

#[repr(C)]
/// The RgsaddleSaddleIndex.
pub struct RgsaddleSaddleIndex {
    /// ABI version this value was built against.
    pub version: RgsaddleVersion,
    /// Bit flags. The header names each bit.
    pub flags: u64,
    /// 1 when a certificate holds for the current point.
    pub certified: i32,
    /// Curvatures under `-negative_tol`.
    pub index: i32,
    /// Every resolved pair met the residual, or the space filled.
    pub converged: i32,
    /// Curvatures resolved.
    pub n_curvatures: i32,
    /// Surface evaluations of this certificate.
    pub evaluations: i64,
    /// Surface evaluations of every certificate and push of the session.
    pub total_evaluations: i64,
    /// Escape pushes of the session.
    pub escapes: i64,
}

fn stamp() -> RgsaddleVersion {
    RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    }
}

/// The check a C config asks for, or the status that refuses it.
fn check_from(cfg: &RgsaddleSaddleCheck) -> Result<SaddleIndexCheck, i32> {
    if cfg.version.major != RGSADDLE_ABI_MAJOR {
        return Err(RGSADDLE_ABI_MISMATCH);
    }
    let count = |v: i32| usize::try_from(v).map_err(|_| RGSADDLE_INVALID_PARAMETER);
    let difference = match cfg.difference {
        0 => FiniteDifference::Forward,
        1 => FiniteDifference::Central,
        _ => return Err(RGSADDLE_INVALID_PARAMETER),
    };
    let attempts = count(cfg.escape_attempts)?;
    let escape = (attempts > 0).then(|| {
        SaddleEscape::default()
            .with_push(cfg.escape_push)
            .with_max_attempts(attempts)
    });
    let check = SaddleIndexCheck::default()
        .with_block(count(cfg.block)?)
        .with_lowest(count(cfg.lowest)?)
        .with_max_actions(count(cfg.max_actions)?)
        .with_difference(difference)
        .with_dr(cfg.dr)
        .with_residual_tol(cfg.residual_tol)
        .with_negative_tol(cfg.negative_tol)
        .with_seed(cfg.seed)
        .with_escape(escape);
    check.validate().map_err(|_| RGSADDLE_INVALID_PARAMETER)?;
    Ok(check)
}

/// # Safety
/// `check` is null or a live config.
pub(super) unsafe fn optional_check(
    check: *const RgsaddleSaddleCheck,
) -> Result<Option<SaddleIndexCheck>, i32> {
    if check.is_null() {
        return Ok(None);
    }
    check_from(unsafe { &*check }).map(Some)
}

/// # Safety
/// `out` is null or writable; `curvatures` addresses `capacity` doubles
/// when `capacity` is positive.
pub(super) unsafe fn write_index(
    index: Option<&SaddleIndex>,
    total: usize,
    escapes: usize,
    out: *mut RgsaddleSaddleIndex,
    curvatures: *mut f64,
    capacity: i64,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    if capacity < 0 || (capacity > 0 && curvatures.is_null()) {
        return RGSADDLE_SHAPE;
    }
    let clamp = |v: usize| i32::try_from(v).unwrap_or(i32::MAX);
    let wide = |v: usize| i64::try_from(v).unwrap_or(i64::MAX);
    let report = RgsaddleSaddleIndex {
        version: stamp(),
        flags: 0,
        certified: i32::from(index.is_some()),
        index: index.map_or(0, |i| clamp(i.index)),
        converged: index.map_or(0, |i| i32::from(i.converged)),
        n_curvatures: index.map_or(0, |i| clamp(i.curvatures.len())),
        evaluations: index.map_or(0, |i| wide(i.evaluations)),
        total_evaluations: wide(total),
        escapes: wide(escapes),
    };
    unsafe { out.write(report) };
    if let Some(index) = index {
        let n = index.curvatures.len().min(capacity as usize);
        if n > 0 {
            unsafe { slice::from_raw_parts_mut(curvatures, n) }
                .copy_from_slice(&index.curvatures[..n]);
        }
    }
    RGSADDLE_OK
}

/// The default check, stamped, with the escape off.
///
/// # Safety
/// `out` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_saddle_check_default(out: *mut RgsaddleSaddleCheck) -> i32 {
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    let check = SaddleIndexCheck::default();
    let escape = SaddleEscape::default();
    let value = RgsaddleSaddleCheck {
        version: stamp(),
        flags: 0,
        block: check.block as i32,
        lowest: check.lowest as i32,
        max_actions: check.max_actions as i32,
        difference: 0,
        dr: check.dr,
        residual_tol: check.residual_tol,
        negative_tol: check.negative_tol,
        seed: check.seed,
        escape_attempts: 0,
        reserved: 0,
        escape_push: escape.push,
    };
    unsafe { out.write(value) };
    RGSADDLE_OK
}

/// Replace the band's saddle check; null turns it off.
///
/// # Safety
/// `band` is a live session or null; `check` is null or a live config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_saddle_check(
    band: *mut RgsaddleBand,
    check: *const RgsaddleSaddleCheck,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let check = match unsafe { optional_check(check) } {
        Ok(check) => check,
        Err(rc) => return rc,
    };
    match unsafe { &mut *band }.session.set_saddle_check(check) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// The certificate of the band's climbing image.
///
/// # Safety
/// `band` is a live session or null; `out` is null or writable and
/// `curvatures` addresses `capacity` doubles when `capacity` is positive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_saddle_index(
    band: *const RgsaddleBand,
    out: *mut RgsaddleSaddleIndex,
    curvatures: *mut f64,
    capacity: i64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let session = &unsafe { &*band }.session;
    unsafe {
        write_index(
            session.saddle_index(),
            session.saddle_evaluations(),
            session.saddle_escapes(),
            out,
            curvatures,
            capacity,
        )
    }
}

/// Replace the min-mode session's saddle check; null turns it off.
///
/// # Safety
/// `session` is a live session or null; `check` is null or a live config.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_set_saddle_check(
    session: *mut RgsaddleMinMode,
    check: *const RgsaddleSaddleCheck,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let check = match unsafe { optional_check(check) } {
        Ok(check) => check,
        Err(rc) => return rc,
    };
    match unsafe { &mut *session }.session.set_saddle_check(check) {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}

/// The certificate of the min-mode session's position.
///
/// # Safety
/// `session` is a live session or null; `out` is null or writable and
/// `curvatures` addresses `capacity` doubles when `capacity` is positive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_minmode_saddle_index(
    session: *const RgsaddleMinMode,
    out: *mut RgsaddleSaddleIndex,
    curvatures: *mut f64,
    capacity: i64,
) -> i32 {
    if session.is_null() {
        return RGSADDLE_NULL_SESSION;
    }
    let session = &unsafe { &*session }.session;
    unsafe {
        write_index(
            session.saddle_index(),
            session.saddle_evaluations(),
            session.saddle_escapes(),
            out,
            curvatures,
            capacity,
        )
    }
}

/// Certify one point.
///
/// # Safety
/// `position` and, when non-null, `gradient` and `mode` address
/// `3 * n_atoms` doubles, `fixed` `n_atoms` bytes; `check` is null or a
/// live config; `out` is null or writable and `curvatures` addresses
/// `capacity` doubles when `capacity` is positive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_saddle_index(
    check: *const RgsaddleSaddleCheck,
    n_atoms: i64,
    position: *const f64,
    gradient: *const f64,
    mode: *const f64,
    fixed: *const u8,
    surface: Option<RgsaddleSurfaceFn>,
    user: *mut c_void,
    out: *mut RgsaddleSaddleIndex,
    curvatures: *mut f64,
    capacity: i64,
) -> i32 {
    if out.is_null() {
        return RGSADDLE_NULL_REPORT;
    }
    if capacity < 0 || (capacity > 0 && curvatures.is_null()) {
        return RGSADDLE_SHAPE;
    }
    let Some(f) = surface else {
        return RGSADDLE_NULL_SURFACE;
    };
    if position.is_null() || n_atoms < 1 {
        return RGSADDLE_SHAPE;
    }
    let check = match unsafe { optional_check(check) } {
        Ok(check) => check.unwrap_or_default(),
        Err(rc) => return rc,
    };
    let dof = 3 * n_atoms as usize;
    let read = |p: *const f64| {
        (!p.is_null()).then(|| Array1::from(unsafe { slice::from_raw_parts(p, dof) }.to_vec()))
    };
    let x = read(position).expect("non-null position");
    let g = read(gradient);
    let m = read(mode);
    let mask = if fixed.is_null() {
        Vec::new()
    } else {
        match unsafe { fixed_mask(fixed, n_atoms) } {
            Some(mask) => mask,
            None => return RGSADDLE_INVALID_PARAMETER,
        }
    };
    let cs = CSurface::point(f, user, n_atoms);
    match saddle_index(
        &check,
        &cs,
        x.view(),
        g.as_ref().map(|g| g.view()),
        m.as_ref().map(|m| m.view()),
        &mask,
    ) {
        Ok(index) => unsafe {
            write_index(
                Some(&index),
                index.evaluations,
                0,
                out,
                curvatures,
                capacity,
            )
        },
        Err(e) => status_of(&e),
    }
}
