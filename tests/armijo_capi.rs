//! The C step converges on Gould's saddle and shortens the dimer.

#![cfg(feature = "capi")]

use rgsaddle::capi::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_OK, RgsaddleArmijoConfig,
    RgsaddleArmijoReport, RgsaddleSurfaceRequest, RgsaddleVersion, rgsaddle_armijo_step,
};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;

extern "C" fn gould_well(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
    let req = unsafe { &*req };
    let x = unsafe { *req.positions };
    let y = unsafe { *req.positions.add(1) };
    unsafe {
        *req.energies = (x * x - 1.0).powi(2) + y * y;
        *req.gradients = 4.0 * x * (x * x - 1.0);
        *req.gradients.add(1) = 2.0 * y;
        *req.gradients.add(2) = 0.0;
    }
    RGSADDLE_OK
}

#[test]
fn c_armijo_step_converges_and_shrinks_the_dimer() {
    let mut position = [0.2, 1.0, 0.0];
    let mut mode = [1.0, 1.0, 0.0];
    let mut length = 1e-3;
    let mut alpha = 0.0;
    let mut beta = 0.0;
    let cfg = RgsaddleArmijoConfig {
        version: RgsaddleVersion {
            major: RGSADDLE_ABI_MAJOR,
            minor: RGSADDLE_ABI_MINOR,
        },
        theta: 0.0,
        psi: 0.0,
        tol_v: -1.0,
        alpha_max: 0.0,
        beta_max: 0.0,
        force_tol: 1e-5,
        length_floor: 1e-8,
        max_rotations: 0,
        max_shrinks: 0,
    };
    let mut report = MaybeUninit::<RgsaddleArmijoReport>::uninit();
    let mut converged = false;
    for _ in 0..20 {
        let status = unsafe {
            rgsaddle_armijo_step(
                &cfg,
                1,
                position.as_mut_ptr(),
                mode.as_mut_ptr(),
                &mut length,
                &mut alpha,
                &mut beta,
                Some(gould_well),
                ptr::null_mut(),
                report.as_mut_ptr(),
            )
        };
        assert_eq!(status, RGSADDLE_OK);
        let written = unsafe { report.assume_init_ref() };
        assert!(length < 1e-3, "dimer length {length}");
        assert!((written.length - length).abs() < 1e-15);
        if written.status == 1 {
            converged = true;
            break;
        }
    }
    assert!(converged, "position {position:?} length {length}");
    assert!(position[0].abs() < 1e-3 && position[1].abs() < 1e-3);
}
