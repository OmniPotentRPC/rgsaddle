//! Quick-min variant of a band (ABI minor 16).

use super::*;

/// Quick-min variants for [`rgsaddle_band_set_quickmin_variant`].
pub const RGSADDLE_QUICKMIN_RGMIN: i32 = 0;
pub const RGSADDLE_QUICKMIN_EON: i32 = 1;
pub const RGSADDLE_QUICKMIN_PER_ATOM: i32 = 2;

/// # Safety
/// `band` is a live session or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_quickmin_variant(
    band: *mut RgsaddleBand,
    variant: i32,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let options = match variant {
        RGSADDLE_QUICKMIN_RGMIN => rgmin::QuickMinOptions::rgmin(),
        RGSADDLE_QUICKMIN_EON => rgmin::QuickMinOptions::eon(),
        RGSADDLE_QUICKMIN_PER_ATOM => rgmin::QuickMinOptions::per_atom(),
        _ => return RGSADDLE_INVALID_PARAMETER,
    };
    let band = unsafe { &mut *band };
    if !band.session.is_quickmin() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    band.session.set_quickmin_options(options);
    RGSADDLE_OK
}

/// # Safety
/// `band` is a live session or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_time_step(
    band: *mut RgsaddleBand,
    time_step: f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    match unsafe { (*band).session.set_time_step(time_step) } {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}
