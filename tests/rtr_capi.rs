#![cfg(feature = "capi")]

use rgsaddle::capi::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

extern "C" fn double_well(user: *mut c_void, request: *mut RgsaddleSurfaceRequest) -> i32 {
    let rows = unsafe { &*user.cast::<AtomicUsize>() };
    let request = unsafe { &mut *request };
    let n = request.n_images as usize;
    let x = unsafe { std::slice::from_raw_parts(request.positions, 3 * n) };
    let e = unsafe { std::slice::from_raw_parts_mut(request.energies, n) };
    let g = unsafe { std::slice::from_raw_parts_mut(request.gradients, 3 * n) };
    rows.fetch_add(n, Ordering::Relaxed);
    for i in 0..n {
        let (a, b, c) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        e[i] = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
        g[3 * i] = 4.0 * a * (a * a - 1.0);
        g[3 * i + 1] = 4.0 * b;
        g[3 * i + 2] = 4.0 * c;
    }
    RGSADDLE_OK
}

#[test]
fn rtr_selector_converges_and_reports_every_physical_row() {
    let n = 7usize;
    let mut x = vec![0.0; 3 * n];
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        x[3 * i] = -1.0 + 2.0 * t;
        x[3 * i + 1] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    x[3 * (n / 2)] += 0.1;
    let original = x.clone();
    let config = RgsaddleBandConfig {
        version: RgsaddleVersion {
            major: RGSADDLE_ABI_MAJOR,
            minor: RGSADDLE_ABI_MINOR,
        },
        flags: 0,
        tangent: 1,
        spring: 0,
        projection: 1,
        method: 2,
        spring_k: 5.0,
        spring_ks: std::ptr::null(),
        ci_trigger_factor: 0.5,
        ci_trigger_force: 0.0,
        cell: std::ptr::null(),
        force_tol: 1e-5,
        max_move: 0.2,
        memory: 20,
    };
    let session = unsafe { rgsaddle_band_create(&config, n as i64, 1, x.as_ptr()) };
    assert!(!session.is_null());
    let mut rows = AtomicUsize::new(0);
    let mut report = RgsaddleReport {
        version: RgsaddleVersion { major: 0, minor: 0 },
        flags: 0,
        status: 0,
        evaluations: 0,
        max_force: 0.0,
        ci_index: -1,
        iteration: 0,
        curvature: 0.0,
        rotations: 0,
    };
    let mut converged = false;
    for _ in 0..400 {
        let before = rows.load(Ordering::Relaxed);
        let rc = unsafe {
            rgsaddle_band_step(
                session,
                Some(double_well),
                (&mut rows as *mut AtomicUsize).cast(),
                &mut report,
            )
        };
        assert_eq!(rc, RGSADDLE_OK);
        if report.iteration == 1 {
            assert!(
                report.evaluations as usize > 2 * n - 2,
                "RTR must evaluate physical curvature probes"
            );
        }
        assert_eq!(
            report.evaluations as usize,
            rows.load(Ordering::Relaxed) - before
        );
        if report.status == 1 {
            converged = true;
            break;
        }
    }
    assert!(converged);
    assert!(rows.load(Ordering::Relaxed) > 0);
    assert_eq!(
        unsafe { rgsaddle_band_positions(session, x.as_mut_ptr()) },
        RGSADDLE_OK
    );
    assert_eq!(&x[..3], &original[..3]);
    assert_eq!(&x[3 * (n - 1)..], &original[3 * (n - 1)..]);
    for i in 1..n - 1 {
        assert!(x[3 * i + 1].abs() < 1e-4);
    }
    let ci = usize::try_from(report.ci_index).expect("climbing image");
    assert!(ci > 0 && ci < n - 1);
    assert!(x[3 * ci].abs() < 1e-3);
    unsafe { rgsaddle_band_free(session) };
}
