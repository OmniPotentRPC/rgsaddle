//! The OH-TST C entry points on a separable double well.
#![cfg(all(feature = "capi", feature = "ohtst"))]

use std::os::raw::c_void;
use std::ptr;

use rgsaddle::capi::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_OK, RgsaddleOhtstOptimizeConfig,
    RgsaddleOhtstOptimizeReport, RgsaddleOhtstRateIn, RgsaddleOhtstRateOut,
    RgsaddleOhtstSampleConfig, RgsaddleOhtstSampleReport, RgsaddleOhtstTransmissionConfig,
    RgsaddleOhtstTransmissionReport, RgsaddleOhtstWorkConfig, RgsaddleOhtstWorkReport,
    RgsaddleVersion, rgsaddle_ohtst_mbar, rgsaddle_ohtst_optimize, rgsaddle_ohtst_rate,
    rgsaddle_ohtst_sample, rgsaddle_ohtst_transmission, rgsaddle_ohtst_work,
};

extern "C" fn well(
    _user: *mut c_void,
    n: i64,
    q: *const f64,
    energy: *mut f64,
    gradient: *mut f64,
) -> i32 {
    if n != 2 || q.is_null() || energy.is_null() || gradient.is_null() {
        return -5;
    }
    let q = unsafe { std::slice::from_raw_parts(q, 2) };
    let (x, y) = (q[0], q[1]);
    let well = x * x - 1.0;
    unsafe {
        *energy = 0.25 * well * well + 0.5 * y * y;
        *gradient = x * well;
        *gradient.add(1) = y;
    }
    RGSADDLE_OK
}

fn version() -> RgsaddleVersion {
    RgsaddleVersion {
        major: RGSADDLE_ABI_MAJOR,
        minor: RGSADDLE_ABI_MINOR,
    }
}

#[test]
fn sample_and_rate_and_mbar_round_trip() {
    let config = RgsaddleOhtstSampleConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 2,
        stride: 1,
        records: 8,
        blocks: 2,
        seed: 1,
    };
    let point = [0.0, 0.0];
    let normal = [1.0, 0.0];
    let mut centroid = [0.0, 0.0];
    let mut grad_normal = [0.0, 0.0];
    let mut report = RgsaddleOhtstSampleReport {
        version: version(),
        flags: 0,
        normal_force: 0.0,
        normal_force_error: 0.0,
        virial: 0.0,
        virial_error: 0.0,
        acceptance: 0.0,
        records: 0,
        evaluations: 0,
    };
    let status = unsafe {
        rgsaddle_ohtst_sample(
            &config,
            2,
            point.as_ptr(),
            normal.as_ptr(),
            Some(well),
            ptr::null_mut(),
            centroid.as_mut_ptr(),
            grad_normal.as_mut_ptr(),
            &mut report,
        )
    };
    assert_eq!(status, RGSADDLE_OK);
    assert_eq!(report.records, 8);
    assert!(report.evaluations > 0);
    assert!(report.acceptance.is_finite());
    assert!(centroid.iter().all(|v| v.is_finite()));

    let input = RgsaddleOhtstRateIn {
        version: version(),
        flags: 0,
        thermal: 0.25,
        density: 1.0,
        density_error: 0.0,
        work: 0.0,
        work_error: 0.0,
    };
    let mut out = RgsaddleOhtstRateOut {
        version: version(),
        flags: 0,
        rate: 0.0,
        error: 0.0,
        log_rate: 0.0,
        log_rate_error: 0.0,
        flux: 0.0,
    };
    let status = unsafe { rgsaddle_ohtst_rate(&input, &mut out) };
    assert_eq!(status, RGSADDLE_OK);
    assert!(out.rate > 0.0);
    assert!((out.rate - out.flux).abs() <= out.flux / 1_000_000.0);

    let reduced = [0.0, 4.0, 0.1, 3.8, 3.9, 0.0, 4.2, 0.2];
    let counts = [2_i64, 2];
    let mut free = [0.0, 0.0];
    let mut covariance = [0.0; 4];
    let mut overlap = [0.0; 4];
    let status = unsafe {
        rgsaddle_ohtst_mbar(
            4,
            2,
            reduced.as_ptr(),
            counts.as_ptr(),
            ptr::null(),
            free.as_mut_ptr(),
            covariance.as_mut_ptr(),
            overlap.as_mut_ptr(),
        )
    };
    assert_eq!(status, RGSADDLE_OK);
    assert!(free[0].abs() <= 0.000_001);
    assert!((overlap[0] + overlap[1] - 1.0).abs() < 0.000_001);

    let mut point = [0.0, 0.0];
    let mut normal = [1.0, 0.0];
    let optimize = RgsaddleOhtstOptimizeConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 2,
        stride: 1,
        records: 8,
        blocks: 2,
        seed: 1,
        max_iterations: 6,
        max_translation: 0.2,
        max_rotation: 0.1,
        translate: 1,
        rotate: 0,
        patience: 1,
    };
    let mut optimum = RgsaddleOhtstOptimizeReport {
        version: version(),
        flags: 0,
        iterations: 0,
        evaluations: 0,
        normal_force: 0.0,
        normal_force_error: 0.0,
    };
    let status = unsafe {
        rgsaddle_ohtst_optimize(
            &optimize,
            2,
            point.as_mut_ptr(),
            normal.as_mut_ptr(),
            Some(well),
            ptr::null_mut(),
            &mut optimum,
        )
    };
    if status != RGSADDLE_OK {
        let text = unsafe { std::ffi::CStr::from_ptr(rgsaddle::capi::rgsaddle_last_error()) };
        panic!("{status}: {}", text.to_string_lossy());
    }
    assert!(optimum.evaluations > 0);
    let norm = normal[0] * normal[0] + normal[1] * normal[1];
    assert!((norm - 1.0).abs() < 0.000_001);

    let points = [0.0, 0.0, 0.3, 0.0];
    let normals = [1.0, 0.0, 1.0, 0.0];
    let work_config = RgsaddleOhtstWorkConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 2,
        stride: 1,
        records: 8,
        blocks: 2,
        seed: 1,
        nodes: 3,
        bar_samples: 2,
        mbar: 1,
        estimate: 2,
    };
    let mut work = RgsaddleOhtstWorkReport {
        version: version(),
        flags: 0,
        work: 0.0,
        error: 0.0,
        integrated_work: 0.0,
        integrated_error: 0.0,
        bar_work: 0.0,
        bar_error: 0.0,
        mbar_work: 0.0,
        mbar_error: 0.0,
        evaluations: 0,
    };
    let status = unsafe {
        rgsaddle_ohtst_work(
            &work_config,
            2,
            2,
            points.as_ptr(),
            normals.as_ptr(),
            ptr::null(),
            Some(well),
            ptr::null_mut(),
            &mut work,
        )
    };
    assert_eq!(status, RGSADDLE_OK);
    assert!(work.work.is_finite() && work.mbar_work.is_finite());
    assert!(work.evaluations > 0);

    let starts = [0.0, 0.0, 0.0, 0.2];
    let shooting = RgsaddleOhtstTransmissionConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 0.0,
        steps: 16,
        check_every: 4,
        shots: 4,
        blocks: 2,
        seed: 2,
    };
    let mut kappa = RgsaddleOhtstTransmissionReport {
        version: version(),
        flags: 0,
        kappa: 0.0,
        error: 0.0,
        shots: 0,
        unassigned: 0,
        reactive: 0,
        recrossed: 0,
        evaluations: 0,
    };
    let status = unsafe {
        rgsaddle_ohtst_transmission(
            &shooting,
            2,
            point.as_ptr(),
            normal.as_ptr(),
            2,
            starts.as_ptr(),
            Some(well),
            ptr::null_mut(),
            &mut kappa,
        )
    };
    assert_eq!(status, RGSADDLE_OK);
    assert!(kappa.kappa.is_finite());
    assert_eq!(kappa.shots + kappa.unassigned, 4);
}
