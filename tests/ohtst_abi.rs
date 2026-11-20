//! The OH-TST C entry points on a separable double well.
#![cfg(all(feature = "capi", feature = "ohtst"))]

use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use rgsaddle::PlaneOptimizerConfig;
use rgsaddle::capi::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_INVALID_PARAMETER, RGSADDLE_NON_FINITE,
    RGSADDLE_OK, RGSADDLE_SHAPE, RGSADDLE_SOLVER, RgsaddleOhtstOptimizeConfig,
    RgsaddleOhtstOptimizeReport, RgsaddleOhtstRateIn, RgsaddleOhtstRateOut,
    RgsaddleOhtstSampleConfig, RgsaddleOhtstSampleReport, RgsaddleOhtstTransmissionConfig,
    RgsaddleOhtstTransmissionReport, RgsaddleOhtstWorkConfig, RgsaddleOhtstWorkReport,
    RgsaddleVersion, rgsaddle_last_error, rgsaddle_ohtst_mbar, rgsaddle_ohtst_optimize,
    rgsaddle_ohtst_rate, rgsaddle_ohtst_sample, rgsaddle_ohtst_transmission, rgsaddle_ohtst_work,
    rgsaddle_status_name,
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
    assert_eq!(report.version.minor, RGSADDLE_ABI_MINOR);
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
    assert_eq!(out.version.minor, RGSADDLE_ABI_MINOR);
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

fn blank_rate_out() -> RgsaddleOhtstRateOut {
    RgsaddleOhtstRateOut {
        version: version(),
        flags: 0,
        rate: 0.0,
        error: 0.0,
        log_rate: 0.0,
        log_rate_error: 0.0,
        flux: 0.0,
    }
}

#[test]
fn unknown_flags_and_negative_rate_errors_are_rejected() {
    let mut input = RgsaddleOhtstRateIn {
        version: version(),
        flags: 1,
        thermal: 0.25,
        density: 1.0,
        density_error: 0.0,
        work: 0.0,
        work_error: 0.0,
    };
    let mut out = blank_rate_out();
    let status = unsafe { rgsaddle_ohtst_rate(&input, &mut out) };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    input.flags = 0;
    input.density_error = -0.01;
    let status = unsafe { rgsaddle_ohtst_rate(&input, &mut out) };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    input.density_error = 0.0;
    input.work_error = -0.01;
    let status = unsafe { rgsaddle_ohtst_rate(&input, &mut out) };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    input.work_error = 0.0;
    input.thermal = 1.0;
    input.work = -1000.0;
    let status = unsafe { rgsaddle_ohtst_rate(&input, &mut out) };
    assert_eq!(status, rgsaddle::capi::RGSADDLE_NON_FINITE);

    let mut sample = RgsaddleOhtstSampleConfig {
        version: version(),
        flags: 1,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
    };
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
    let point = [0.0, 0.0];
    let normal = [1.0, 0.0];
    let mut centroid = [0.0, 0.0];
    let mut grad_normal = [0.0, 0.0];
    let status = unsafe {
        rgsaddle_ohtst_sample(
            &sample,
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
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    sample.flags = 0;
    let status = unsafe {
        rgsaddle_ohtst_sample(
            &sample,
            -1,
            point.as_ptr(),
            normal.as_ptr(),
            Some(well),
            ptr::null_mut(),
            centroid.as_mut_ptr(),
            grad_normal.as_mut_ptr(),
            &mut report,
        )
    };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);

    let optimize = RgsaddleOhtstOptimizeConfig {
        version: version(),
        flags: 1,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
        max_iterations: 1,
        max_translation: 0.1,
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
    let mut point = [0.0, 0.0];
    let mut normal = [1.0, 0.0];
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
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);

    let work = RgsaddleOhtstWorkConfig {
        version: version(),
        flags: 1,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
        nodes: 3,
        bar_samples: 0,
        mbar: 0,
        estimate: 0,
    };
    let mut work_report = RgsaddleOhtstWorkReport {
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
    let planes = [0.0, 0.0, 0.2, 0.0];
    let status = unsafe {
        rgsaddle_ohtst_work(
            &work,
            2,
            2,
            planes.as_ptr(),
            planes.as_ptr(),
            ptr::null(),
            Some(well),
            ptr::null_mut(),
            &mut work_report,
        )
    };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    let status = unsafe {
        rgsaddle_ohtst_work(
            &RgsaddleOhtstWorkConfig { flags: 0, ..work },
            2,
            -3,
            planes.as_ptr(),
            planes.as_ptr(),
            ptr::null(),
            Some(well),
            ptr::null_mut(),
            &mut work_report,
        )
    };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);

    let shooting = RgsaddleOhtstTransmissionConfig {
        version: version(),
        flags: 1,
        thermal: 0.25,
        time_step: 0.05,
        friction: 0.0,
        steps: 4,
        check_every: 2,
        shots: 2,
        blocks: 2,
        seed: 1,
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
    let starts = [0.0, 0.0];
    let status = unsafe {
        rgsaddle_ohtst_transmission(
            &shooting,
            2,
            point.as_ptr(),
            normal.as_ptr(),
            1,
            starts.as_ptr(),
            Some(well),
            ptr::null_mut(),
            &mut kappa,
        )
    };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
}

#[test]
fn mbar_state_square_that_does_not_fit_is_rejected() {
    let dummy = 1_i64;
    let status = unsafe {
        rgsaddle_ohtst_mbar(
            1,
            1_i64 << 32,
            ptr::null(),
            &dummy,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, RGSADDLE_SHAPE);
}

#[test]
fn optimize_reports_not_converged_when_unconverged() {
    let mut point = [0.4, 0.0];
    let mut normal = [1.0, 0.3];
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
        max_iterations: 1,
        max_translation: 0.05,
        max_rotation: 0.05,
        translate: 1,
        rotate: 1,
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
    assert_eq!(status, rgsaddle::capi::RGSADDLE_SOLVER);
}

/// The first record count is only where the sample starts. The optimizer
/// keeps its default ceiling, so a plane that needs more than eight
/// records still converges.
#[test]
fn optimize_grows_the_sample_past_the_first_record_count() {
    let mut point = [0.4, 0.0];
    let mut normal = [1.0, 0.3];
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
        seed: 6,
        max_iterations: 100,
        max_translation: 0.05,
        max_rotation: 0.05,
        translate: 1,
        rotate: 1,
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
        let text = unsafe { CStr::from_ptr(rgsaddle_last_error()) };
        panic!("{status}: {}", text.to_string_lossy());
    }
    assert!(optimum.iterations > 0);
    assert!(optimum.evaluations > 0);
}

extern "C" fn noisy_zero_mean(
    user: *mut c_void,
    n: i64,
    q: *const f64,
    energy: *mut f64,
    gradient: *mut f64,
) -> i32 {
    if n != 2 || q.is_null() || energy.is_null() || gradient.is_null() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    if !user.is_null() {
        let count = unsafe { &*(user as *const AtomicU64) };
        count.fetch_add(1, Ordering::Relaxed);
    }
    let q = unsafe { std::slice::from_raw_parts(q, 2) };
    let (x, y) = (q[0], q[1]);
    unsafe {
        *energy = -0.5 * x * x + y * y + 0.2 * x * (y * y - 0.1);
        *gradient = -x + 0.2 * (y * y - 0.1);
        *gradient.add(1) = 2.0 * y + 0.4 * x * y;
    }
    RGSADDLE_OK
}

/// Sample growth stops at `PlaneOptimizerConfig::default().max_records`,
/// and a noisy plane held at that cap returns `RGSADDLE_SOLVER` rather than
/// reporting false convergence.
#[test]
fn optimize_caps_sample_growth_at_the_default_ceiling() {
    let default_cap = PlaneOptimizerConfig::default().max_records as u64;
    let start_records = 8_000_u64;
    assert!(start_records < default_cap);
    let max_iterations = 3_u64;
    let mut point = [0.0, 0.0];
    let mut normal = [1.0, 0.0];
    let optimize = RgsaddleOhtstOptimizeConfig {
        version: version(),
        flags: 0,
        thermal: 0.2,
        time_step: 0.1,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: start_records,
        blocks: 2,
        seed: 7,
        max_iterations,
        max_translation: 0.05,
        max_rotation: 0.05,
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
    let calls = AtomicU64::new(0);
    let status = unsafe {
        rgsaddle_ohtst_optimize(
            &optimize,
            2,
            point.as_mut_ptr(),
            normal.as_mut_ptr(),
            Some(noisy_zero_mean),
            (&calls as *const AtomicU64) as *mut c_void,
            &mut optimum,
        )
    };
    assert_eq!(status, RGSADDLE_SOLVER);
    let message = unsafe { CStr::from_ptr(rgsaddle_last_error()) }.to_string_lossy();
    assert!(
        message.contains("did not converge"),
        "expected non-convergence message, got: {message}"
    );
    let total_calls = calls.load(Ordering::Relaxed);
    // Growth happens past the starting record count, and each capped sample
    // evaluates at most `default_cap + 1` points (initial, grown re-sample,
    // and at most two trial evaluations per outer iteration).
    let per_capped_sample = default_cap + 1;
    let upper_bound = (2 + 2 * max_iterations) * per_capped_sample;
    assert!(
        total_calls > start_records + per_capped_sample,
        "expected sample growth to reach the cap, saw {total_calls} calls"
    );
    assert!(
        total_calls <= upper_bound,
        "sample growth exceeded the {default_cap}-record cap: {total_calls} > {upper_bound}"
    );
}

fn assert_non_finite_status(status: i32, expected_cause: &str) {
    assert_eq!(status, RGSADDLE_NON_FINITE);
    let name = unsafe { CStr::from_ptr(rgsaddle_status_name(status)) }.to_string_lossy();
    assert_eq!(name, "NON_FINITE");
    let detail = unsafe { CStr::from_ptr(rgsaddle_last_error()) }.to_string_lossy();
    assert!(
        detail.contains(expected_cause),
        "expected last_error to name '{expected_cause}', got: '{detail}'"
    );
}

#[test]
fn non_finite_inputs_return_named_non_finite_status() {
    let mut centroid = [0.0, 0.0];
    let mut grad_normal = [0.0, 0.0];
    let mut sample_report = RgsaddleOhtstSampleReport {
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
    let sample_cfg = RgsaddleOhtstSampleConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
    };
    let nan_point = [f64::NAN, 0.0];
    let normal = [1.0, 0.0];
    let status = unsafe {
        rgsaddle_ohtst_sample(
            &sample_cfg,
            2,
            nan_point.as_ptr(),
            normal.as_ptr(),
            Some(well),
            ptr::null_mut(),
            centroid.as_mut_ptr(),
            grad_normal.as_mut_ptr(),
            &mut sample_report,
        )
    };
    assert_non_finite_status(status, "point");

    let mut bad_sample_cfg = sample_cfg;
    bad_sample_cfg.thermal = f64::INFINITY;
    let point = [0.0, 0.0];
    let status = unsafe {
        rgsaddle_ohtst_sample(
            &bad_sample_cfg,
            2,
            point.as_ptr(),
            normal.as_ptr(),
            Some(well),
            ptr::null_mut(),
            centroid.as_mut_ptr(),
            grad_normal.as_mut_ptr(),
            &mut sample_report,
        )
    };
    assert_non_finite_status(status, "thermal");

    let mut opt_point = [0.0, 0.0];
    let mut opt_normal = [1.0, 0.0];
    let opt_cfg = RgsaddleOhtstOptimizeConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
        max_iterations: 2,
        max_translation: f64::NAN,
        max_rotation: 0.1,
        translate: 1,
        rotate: 0,
        patience: 1,
    };
    let mut opt_report = RgsaddleOhtstOptimizeReport {
        version: version(),
        flags: 0,
        iterations: 0,
        evaluations: 0,
        normal_force: 0.0,
        normal_force_error: 0.0,
    };
    let status = unsafe {
        rgsaddle_ohtst_optimize(
            &opt_cfg,
            2,
            opt_point.as_mut_ptr(),
            opt_normal.as_mut_ptr(),
            Some(well),
            ptr::null_mut(),
            &mut opt_report,
        )
    };
    assert_non_finite_status(status, "max_translation");

    let bad_reduced = [0.0, f64::NAN, 1.0, 0.0];
    let counts = [1_i64, 1];
    let mut free = [0.0; 2];
    let mut cov = [0.0; 4];
    let mut ov = [0.0; 4];
    let status = unsafe {
        rgsaddle_ohtst_mbar(
            2,
            2,
            bad_reduced.as_ptr(),
            counts.as_ptr(),
            ptr::null(),
            free.as_mut_ptr(),
            cov.as_mut_ptr(),
            ov.as_mut_ptr(),
        )
    };
    assert_non_finite_status(status, "reduced");

    let work_cfg = RgsaddleOhtstWorkConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
        nodes: 3,
        bar_samples: 0,
        mbar: 0,
        estimate: 0,
    };
    let mut work_report = RgsaddleOhtstWorkReport {
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
    let work_points = [0.0, 0.0, 0.3, 0.0];
    let bad_normals = [1.0, 0.0, f64::INFINITY, 0.0];
    let status = unsafe {
        rgsaddle_ohtst_work(
            &work_cfg,
            2,
            2,
            work_points.as_ptr(),
            bad_normals.as_ptr(),
            ptr::null(),
            Some(well),
            ptr::null_mut(),
            &mut work_report,
        )
    };
    assert_non_finite_status(status, "normals");

    let trans_cfg = RgsaddleOhtstTransmissionConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 0.0,
        steps: 4,
        check_every: 2,
        shots: 2,
        blocks: 2,
        seed: 1,
    };
    let mut trans_report = RgsaddleOhtstTransmissionReport {
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
    let bad_starts = [0.0, f64::NEG_INFINITY];
    let status = unsafe {
        rgsaddle_ohtst_transmission(
            &trans_cfg,
            2,
            point.as_ptr(),
            normal.as_ptr(),
            1,
            bad_starts.as_ptr(),
            Some(well),
            ptr::null_mut(),
            &mut trans_report,
        )
    };
    assert_non_finite_status(status, "starts");

    let bad_rate_in = RgsaddleOhtstRateIn {
        version: version(),
        flags: 0,
        thermal: f64::NAN,
        density: 1.0,
        density_error: 0.0,
        work: 0.0,
        work_error: 0.0,
    };
    let mut rate_out = blank_rate_out();
    let status = unsafe { rgsaddle_ohtst_rate(&bad_rate_in, &mut rate_out) };
    assert_non_finite_status(status, "thermal");
}

/// `1 << 32` does not fit in a 32-bit `usize`. The cast must fail before
/// any allocation.
#[cfg(target_pointer_width = "32")]
#[test]
fn dimension_above_usize_is_rejected() {
    let sample = RgsaddleOhtstSampleConfig {
        version: version(),
        flags: 0,
        thermal: 0.25,
        time_step: 0.05,
        friction: 1.0,
        burn_in: 0,
        stride: 1,
        records: 4,
        blocks: 2,
        seed: 1,
    };
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
    let point = [0.0_f64; 0];
    let status = unsafe {
        rgsaddle_ohtst_sample(
            &sample,
            1_i64 << 32,
            point.as_ptr(),
            point.as_ptr(),
            Some(well),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut report,
        )
    };
    assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
}
