#![cfg(feature = "capi")]

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

#[cfg(feature = "ohtst")]
extern "C" fn quartic(
    _: *mut c_void,
    n: i64,
    x: *const f64,
    energy: *mut f64,
    gradient: *mut f64,
) -> i32 {
    assert_eq!(n, 1);
    unsafe {
        let q = *x;
        *energy = (q * q - 1.0).powi(2);
        *gradient = 4.0 * q * (q * q - 1.0);
    }
    0
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

#[test]
#[cfg(feature = "ohtst")]
fn c_pafi_transmission_selects_the_free_energy_maximum() {
    unsafe {
        let mut cfg = MaybeUninit::uninit();
        assert_eq!(
            rgsaddle_pafi_transmission_config_default(cfg.as_mut_ptr()),
            RGSADDLE_OK
        );
        let mut cfg = cfg.assume_init();
        cfg.thermal = 0.05;
        cfg.friction = 0.0;
        cfg.time_step = 0.002;
        cfg.steps = 100;
        cfg.check_every = 25;
        cfg.shots = 32;
        cfg.blocks = 4;
        cfg.seed = 7;
        let free = [0.0, 1.0];
        let points = [-1.0, 0.0];
        let normals = [1.0, 1.0];
        let offsets = [0_i64, 1, 3];
        let starts = [-1.0, 0.0, 0.0];
        let mut report = MaybeUninit::uninit();
        assert_eq!(
            rgsaddle_pafi_transmission(
                &cfg,
                1,
                2,
                free.as_ptr(),
                points.as_ptr(),
                normals.as_ptr(),
                offsets.as_ptr(),
                starts.as_ptr(),
                ptr::null(),
                Some(quartic),
                ptr::null_mut(),
                report.as_mut_ptr(),
            ),
            RGSADDLE_OK
        );
        let report = report.assume_init();
        assert_eq!(report.dividing, 1);
        assert_eq!(report.shots, 32);
        assert!((report.kappa - 1.0).abs() < 1.0e-12, "{:?}", report.kappa);
    }
}
