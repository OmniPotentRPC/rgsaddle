use rgsaddle::{capi::*, capi_kappa::*};
use std::{ffi::c_void, mem::MaybeUninit, ptr};

extern "C" fn hessian(_: *mut c_void, n: i64, v: *const f64, out: *mut f64) -> i32 {
    assert_eq!(n, 3);
    for (i, lambda) in [-4., 2., 6.].iter().enumerate() {
        unsafe {
            *out.add(i) = lambda * *v.add(i);
        }
    }
    0
}
extern "C" fn fail(_: *mut c_void, _: i64, _: *const f64, _: *mut f64) -> i32 {
    17
}

#[test]
fn c_force_matches_the_contour_definition_and_propagates_errors() {
    unsafe {
        let mut cfg = MaybeUninit::uninit();
        assert_eq!(rgsaddle_kappa_config_default(cfg.as_mut_ptr()), RGSADDLE_OK);
        let cfg = cfg.assume_init();
        let mut report = MaybeUninit::uninit();
        let mut out = [123.; 3];
        let g = [3., 0., 0.];
        let m = [1., 0., 0.];
        assert_eq!(
            rgsaddle_kappa_dimer_force(
                &cfg,
                3,
                g.as_ptr(),
                m.as_ptr(),
                0,
                ptr::null(),
                Some(hessian),
                ptr::null_mut(),
                out.as_mut_ptr(),
                report.as_mut_ptr()
            ),
            RGSADDLE_OK
        );
        let report = report.assume_init();
        assert!((report.kappa + 2. / 3.).abs() < 1e-8);
        assert_eq!(report.tangent_dimension, 2);
        assert!(out[0] > 0.);
        assert!(report.hessian_actions > 0);
        let expected = out;
        let mut error_report = MaybeUninit::uninit();
        assert_eq!(
            rgsaddle_kappa_dimer_force(
                &cfg,
                3,
                g.as_ptr(),
                m.as_ptr(),
                0,
                ptr::null(),
                Some(fail),
                ptr::null_mut(),
                out.as_mut_ptr(),
                error_report.as_mut_ptr()
            ),
            RGSADDLE_SURFACE_FAILED
        );
        assert_eq!(out, expected);
    }
}
