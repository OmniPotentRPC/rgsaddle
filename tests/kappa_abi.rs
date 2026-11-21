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

#[cfg(feature = "ohtst")]
mod pafi_transmission {
    use super::*;

    struct Band {
        free: Vec<f64>,
        points: Vec<f64>,
        normals: Vec<f64>,
        offsets: Vec<i64>,
        starts: Vec<f64>,
    }

    fn band() -> Band {
        Band {
            free: vec![0.0, 1.0, 0.0],
            points: vec![-1.0, 0.0, 1.0],
            normals: vec![1.0; 3],
            offsets: vec![0, 1, 3, 4],
            starts: vec![-1.0, 0.0, 0.0, 1.0],
        }
    }

    fn config() -> RgsaddlePafiTransmissionConfig {
        let mut cfg = MaybeUninit::uninit();
        unsafe {
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
            cfg
        }
    }

    fn shoot(
        cfg: &RgsaddlePafiTransmissionConfig,
        band: &Band,
    ) -> (i32, RgsaddlePafiTransmissionReport) {
        let mut report = MaybeUninit::<RgsaddlePafiTransmissionReport>::zeroed();
        let status = unsafe {
            rgsaddle_pafi_transmission(
                cfg,
                1,
                band.free.len() as i64,
                band.free.as_ptr(),
                band.points.as_ptr(),
                band.normals.as_ptr(),
                band.offsets.as_ptr(),
                band.starts.as_ptr(),
                ptr::null(),
                Some(quartic),
                ptr::null_mut(),
                report.as_mut_ptr(),
            )
        };
        (status, unsafe { report.assume_init() })
    }

    #[test]
    fn c_pafi_transmission_selects_the_free_energy_maximum() {
        let (status, report) = shoot(&config(), &band());
        assert_eq!(status, RGSADDLE_OK);
        assert_eq!(report.dividing, 1);
        assert_eq!(report.shots, 32);
        assert_eq!(report.version.major, RGSADDLE_ABI_MAJOR);
        assert_eq!(report.version.minor, RGSADDLE_ABI_MINOR);
        assert!((report.kappa - 1.0).abs() < 1.0e-12, "{:?}", report.kappa);
    }

    #[test]
    fn c_pafi_transmission_refuses_a_minor_newer_than_the_library() {
        let mut cfg = config();
        cfg.version.minor = RGSADDLE_ABI_MINOR + 1;
        assert_eq!(shoot(&cfg, &band()).0, RGSADDLE_ABI_MISMATCH);
    }

    #[test]
    fn c_pafi_transmission_refuses_a_minor_older_than_its_structs() {
        let mut cfg = config();
        cfg.version.minor = 51;
        assert_eq!(shoot(&cfg, &band()).0, RGSADDLE_ABI_MISMATCH);
    }

    #[test]
    fn c_pafi_transmission_reports_a_major_mismatch_before_other_checks() {
        let mut cfg = config();
        cfg.version.major = RGSADDLE_ABI_MAJOR + 1;
        assert_eq!(shoot(&cfg, &band()).0, RGSADDLE_ABI_MISMATCH);
        cfg.steps = 0;
        assert_eq!(shoot(&cfg, &band()).0, RGSADDLE_ABI_MISMATCH);
        cfg.version.major = RGSADDLE_ABI_MAJOR;
        assert_eq!(shoot(&cfg, &band()).0, RGSADDLE_INVALID_PARAMETER);
    }

    #[test]
    fn c_pafi_transmission_refuses_a_maximum_at_either_end() {
        let cfg = config();
        for free in [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]] {
            let mut band = band();
            band.free = free.to_vec();
            let (status, report) = shoot(&cfg, &band);
            assert_eq!(status, RGSADDLE_INVALID_PARAMETER, "{free:?}");
            assert_eq!(report.shots, 0, "{free:?}");
        }
    }
}
