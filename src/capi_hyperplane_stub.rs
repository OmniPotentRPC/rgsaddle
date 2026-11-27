//! Hyperplane C entry points when the `ohtst` feature is off.
//!
//! `include/rgsaddle.h` declares these four symbols for every build.
//! With `ohtst` they live in `capi_hyperplane`. Without it they are
//! still exported: creation cannot succeed, freeing is a no-op, the
//! integration reports an invalid parameter, and the observable name
//! stays the stable string `potential.full_minus_harmonic`.

use std::ffi::{CString, c_char, c_void};
use std::sync::OnceLock;

use crate::capi::RGSADDLE_INVALID_PARAMETER;

/// Same bytes as `ohtst::THERMODYNAMIC_DERIVATIVE`.
const THERMODYNAMIC_DERIVATIVE: &str = "potential.full_minus_harmonic";

/// Create a callback sampler. Always null without `ohtst`.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_hyperplane_sampler_create(
    _context: *mut c_void,
    _sample: Option<extern "C" fn()>,
) -> *mut c_void {
    std::ptr::null_mut()
}

/// Free a callback sampler. Null and non-null are both no-ops without `ohtst`.
///
/// # Safety
///
/// `sampler` is ignored. A non-null pointer is not freed, because this
/// build never allocates a sampler.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hyperplane_sampler_free(_sampler: *mut c_void) {}

/// Run fixed-plane lambda TI. Unavailable without `ohtst`.
///
/// # Safety
///
/// Every pointer is ignored. The call does not read or write them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_hyperplane_thermodynamic_integration(
    _sampler: *mut c_void,
    _point: *const f64,
    _normal: *const f64,
    _dimension: usize,
    _config: *const c_void,
    _result: *mut c_void,
) -> i32 {
    RGSADDLE_INVALID_PARAMETER
}

/// Stable observable name for `V_full - V_harmonic`.
#[unsafe(no_mangle)]
pub extern "C" fn rgsaddle_thermodynamic_derivative_observable() -> *const c_char {
    static POINTER: OnceLock<CString> = OnceLock::new();
    let name = POINTER.get_or_init(|| {
        CString::new(THERMODYNAMIC_DERIVATIVE).expect("thermodynamic derivative name")
    });
    name.as_ptr()
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn hyperplane_entry_points_stay_exported_without_ohtst() {
        assert!(rgsaddle_hyperplane_sampler_create(std::ptr::null_mut(), None).is_null());
        unsafe { rgsaddle_hyperplane_sampler_free(std::ptr::null_mut()) };
        let status = unsafe {
            rgsaddle_hyperplane_thermodynamic_integration(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let name = unsafe { CStr::from_ptr(rgsaddle_thermodynamic_derivative_observable()) };
        assert_eq!(name.to_str().unwrap(), THERMODYNAMIC_DERIVATIVE);
    }
}
