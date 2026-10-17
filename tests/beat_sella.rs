//! Regressions for the single-ended search that optbench scores against Sella.
//!
//! Rigid modes leave the dimer, Lanczos, and Cartesian P-RFO. The dimer
//! stops once its trial angle is below what a forward difference can
//! resolve. FIRE does not report the geometry it started from when the
//! effective force vanishes. Bofill and TS-BFGS both move the stored Hessian.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::Method;
use rgsaddle::{
    CartesianPes, Constraints, HessUpdate, MinModeConfig, MinModeKind, MinModeSession,
    PointSurface, SaddleError, SellaSaddleConfig, SellaSaddleSession, prfo_trust_region,
    rigid_body_modes,
};

#[test]
fn cartesian_prfo_finite_difference_diag_finds_a_quadratic_saddle() {
    struct Quadratic {
        calls: std::sync::atomic::AtomicUsize,
    }
    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let curvature = [-1.0, 2.0, 3.0];
            let mut gradient = Array1::zeros(3);
            let mut energy = 0.0;
            for axis in 0..3 {
                energy += 0.5 * curvature[axis] * x[axis] * x[axis];
                gradient[axis] = curvature[axis] * x[axis];
            }
            Ok((energy, gradient))
        }
    }
    let surface = Quadratic {
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut session = SellaSaddleSession::new(
        SellaSaddleConfig {
            delta: 10.0,
            gamma: 0.1,
            force_tol: 1e-8,
            ..SellaSaddleConfig::default()
        },
        array![0.5, 0.4, 0.3],
        Array1::ones(1),
    )
    .unwrap();
    session.set_update(HessUpdate::TsBfgs);
    let mut report = session.step(&surface).unwrap();
    for _ in 0..4 {
        if report.at_saddle {
            break;
        }
        report = session.step(&surface).unwrap();
    }
    let calls = surface.calls.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        report.at_saddle,
        "force {} after {calls} calls",
        report.max_force
    );
    assert!(
        calls > 2,
        "the opening step did not finite-difference the Hessian"
    );
    assert!(
        calls < 20,
        "quadratic saddle took {calls} calls; the secant update did not install the curvature"
    );
}

fn water() -> Array1<f64> {
    array![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]
}

#[test]
fn prfo_trust_step_on_a_wide_spectrum_stays_under_a_second() {
    let n = 96;
    let evals = Array1::from_iter((0..n).map(|i| if i == 0 { -2.0 } else { 0.2 + i as f64 }));
    let evecs = Array2::<f64>::eye(n);
    let gradient = Array1::from_iter((0..n).map(|i| 0.05 * (1.0 + (i % 7) as f64)));
    let started = std::time::Instant::now();
    let step = prfo_trust_region(&evals, &evecs, &gradient, 1, 1.0e6);
    let elapsed = started.elapsed();
    assert!(step.iter().all(|value| value.is_finite()));
    assert!(step[0].abs() > 1e-8, "{step:?}");
    assert!(
        elapsed.as_millis() < 1000,
        "a {n}-dimensional P-RFO step took {elapsed:?}"
    );
}

fn overlap(mode: ArrayView1<f64>, row: ArrayView1<f64>) -> f64 {
    let mut acc = 0.0;
    let mut row_norm = 0.0;
    for (a, b) in mode.iter().zip(row.iter()) {
        acc += a * b;
        row_norm += b * b;
    }
    acc.abs() / row_norm.sqrt().max(1e-30)
}

struct BondCluster;

impl PointSurface for BondCluster {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let pairs = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.2)];
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for (i, j, target) in pairs {
            let mut delta = [0.0; 3];
            let mut dist2 = 0.0;
            for k in 0..3 {
                delta[k] = x[3 * j + k] - x[3 * i + k];
                dist2 += delta[k] * delta[k];
            }
            let dist = dist2.sqrt();
            let stretch = dist - target;
            energy += 0.5 * stretch * stretch;
            let scale = stretch / dist.max(1e-12);
            for k in 0..3 {
                gradient[3 * i + k] -= scale * delta[k];
                gradient[3 * j + k] += scale * delta[k];
            }
        }
        Ok((energy, gradient))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(rigid_body_modes(x))
    }
}

#[test]
fn rigid_modes_drop_the_null_rotation_of_a_linear_pair() {
    assert_eq!(rigid_body_modes(array![0.0, 0.0, 0.0].view()).nrows(), 0);
    let pair = array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0];
    assert_eq!(rigid_body_modes(pair.view()).nrows(), 5);
    let bent = water();
    assert_eq!(rigid_body_modes(bent.view()).nrows(), 6);
    let linear = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.2, 0.0, 0.0];
    assert_eq!(rigid_body_modes(linear.view()).nrows(), 5);
}

#[test]
fn dimer_and_lanczos_are_orthogonal_to_rigid_modes() {
    let x = water();
    let seed = array![0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.0, -0.3];
    for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
        let config = MinModeConfig {
            kind,
            max_rotations: 20,
            ..MinModeConfig::default()
        };
        let mut session = MinModeSession::new(config, x.clone(), seed.clone()).unwrap();
        let estimate = session.estimate_mode(&BondCluster).unwrap();
        assert!(
            estimate.rotations < 20,
            "{kind:?} used {} rotations, curvature {}",
            estimate.rotations,
            estimate.curvature
        );
        let modes = rigid_body_modes(x.view());
        for row in modes.rows() {
            let dot = overlap(estimate.mode.view(), row);
            assert!(dot < 1e-8, "{kind:?} overlap {dot}");
        }
        assert!(estimate.curvature > 0.05, "{kind:?} {}", estimate.curvature);
    }
}

#[test]
fn dimer_stops_before_the_rotation_budget_on_a_resolved_seed() {
    struct Valley;
    impl PointSurface for Valley {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (a, b, c) = (x[0], x[1], x[2]);
            let energy = 40.0 * a * a + 40.0 * b * b + 0.25 * c * c;
            Ok((energy, array![80.0 * a, 80.0 * b, 0.5 * c]))
        }
    }
    let config = MinModeConfig {
        rotation_tol: 1e-8,
        max_rotations: 40,
        ..MinModeConfig::default()
    };
    let mut session =
        MinModeSession::new(config, array![0.01, -0.02, 0.4], array![1.0e-3, 0.0, 1.0]).unwrap();
    let estimate = session.estimate_mode(&Valley).unwrap();
    assert!(
        estimate.rotations < 8,
        "rotations {} curvature {}",
        estimate.rotations,
        estimate.curvature
    );
    assert!(estimate.mode[2].abs() > 0.99, "{:?}", estimate.mode);
    assert!(
        (0.4..=0.7).contains(&estimate.curvature),
        "{}",
        estimate.curvature
    );
}

#[test]
fn prfo_step_on_a_euclidean_chart_drops_rigid_motion() {
    let x = water();
    let chart = Constraints::new(3).unwrap();
    let mut session = SellaSaddleSession::with_chart(
        SellaSaddleConfig {
            delta: 0.05,
            force_tol: 1e-8,
            ..SellaSaddleConfig::default()
        },
        x.clone(),
        Array1::ones(3),
        chart,
    )
    .unwrap();
    let before = session.position().to_owned();
    let report = session.step(&BondCluster).unwrap();
    assert!(report.max_force.is_finite());
    let step = &session.position().to_owned() - &before;
    assert!(step.dot(&step).sqrt() > 1e-8, "P-RFO did not move");
    for row in rigid_body_modes(before.view()).rows() {
        let dot = overlap(step.view(), row);
        assert!(dot < 1e-6, "rigid overlap {dot}");
    }
}

#[test]
fn fire_does_not_repeat_a_geometry_when_the_force_is_perpendicular() {
    struct Perpendicular;
    impl PointSurface for Perpendicular {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (a, b, c) = (x[0], x[1], x[2]);
            let energy = a * a + 5.0 * b * b + 5.0 * c * c + b;
            Ok((energy, array![2.0 * a, 10.0 * b + 1.0, 10.0 * c]))
        }
    }
    let mut session = MinModeSession::new(
        MinModeConfig {
            method: Method::Fire {
                kind: rgmin::FireKind::V2,
            },
            max_move: 0.2,
            ..MinModeConfig::default()
        },
        array![0.0, 0.0, 0.0],
        array![1.0, 0.0, 0.0],
    )
    .unwrap();
    let report = session.step(&Perpendicular).unwrap();
    assert!(report.evaluations > 0);
    assert!(
        (session.position()[0]).abs() > 1e-6,
        "position {:?}",
        session.position()
    );
    assert_ne!(session.position()[0], 0.0);
}

#[test]
fn nonlinear_cg_is_a_valid_min_mode_method() {
    let mut session = MinModeSession::new(
        MinModeConfig {
            method: Method::polak_ribiere(),
            ..MinModeConfig::default()
        },
        array![0.4, 0.1, -0.2],
        array![1.0, 0.0, 0.0],
    )
    .unwrap();
    let report = session.step(&BondClusterLike).expect("nonlinear CG step");
    assert!(report.max_force.is_finite());
    assert!(session.position().iter().all(|v| v.is_finite()));
}

struct BondClusterLike;

impl PointSurface for BondClusterLike {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        let energy = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
        Ok((energy, array![4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c]))
    }
}

#[test]
fn bofill_and_ts_bfgs_change_the_stored_hessian() {
    struct Quadratic;
    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut gradient = Array1::zeros(x.len());
            gradient[0] = 4.0 * t;
            Ok((2.0 * t * t, gradient))
        }
    }
    for update in [HessUpdate::TsBfgs, HessUpdate::Bofill] {
        let mut pes = CartesianPes::new(array![0.2, 0.0, 0.0], array![1.0]).unwrap();
        pes.set_update(update);
        let before = pes.hessian().hessian().clone();
        let mut step = Array1::zeros(3);
        step[0] = 0.05;
        pes.kick(&Quadratic, step.view()).unwrap();
        let mut diff = 0.0;
        for (left, right) in before.iter().zip(pes.hessian().hessian().iter()) {
            diff += (left - right).abs();
        }
        assert!(diff > 1e-6, "{update:?} left the Hessian unchanged");
        assert_eq!(update.to_abi() == 2, matches!(update, HessUpdate::Bofill));
    }
    assert!(HessUpdate::try_from_abi(99).is_none());
}

#[cfg(feature = "capi")]
mod abi {
    use rgsaddle::capi::{
        RGSADDLE_INVALID_PARAMETER, RGSADDLE_OK, RgsaddleBandConfig, RgsaddleMinModeConfig,
        RgsaddleReport, RgsaddleSurfaceRequest, RgsaddleVersion, rgsaddle_band_create,
        rgsaddle_band_free, rgsaddle_minmode_create, rgsaddle_minmode_free, rgsaddle_minmode_step,
    };
    use std::os::raw::c_void;

    fn version() -> RgsaddleVersion {
        RgsaddleVersion {
            major: 1,
            minor: 17,
        }
    }

    fn min_config(method: i32) -> RgsaddleMinModeConfig {
        RgsaddleMinModeConfig {
            version: version(),
            flags: 0,
            kind: 0,
            method,
            dr: 1e-3,
            rotation_tol: 1e-3,
            max_rotations: 4,
            krylov_dim: 4,
            force_tol: 1e-2,
            max_move: 0.2,
            rotation_angle_tol: 0.0,
            difference: 0,
            reserved: 0,
        }
    }

    extern "C" fn well(_user: *mut c_void, request: *mut RgsaddleSurfaceRequest) -> i32 {
        let request = unsafe { &mut *request };
        let n = request.n_atoms as usize * 3;
        let x = unsafe { std::slice::from_raw_parts(request.positions, n) };
        let gradient = unsafe { std::slice::from_raw_parts_mut(request.gradients, n) };
        let mut energy = 0.0;
        for value in x.iter().zip(gradient.iter_mut()) {
            energy += value.0 * value.0;
            *value.1 = 2.0 * value.0;
        }
        unsafe { *request.energies = energy };
        0
    }

    #[test]
    fn optimizer_codes_accept_nlcg_and_refuse_the_rest() {
        let x = [0.3, 0.0, 0.0];
        let mode = [1.0, 0.0, 0.0];
        for method in [2, 3, 99] {
            let cfg = min_config(method);
            let session = unsafe { rgsaddle_minmode_create(&cfg, 1, x.as_ptr(), mode.as_ptr()) };
            assert!(session.is_null(), "method {method} created a session");
        }
        for method in [0, 1, 4] {
            let cfg = min_config(method);
            let session = unsafe { rgsaddle_minmode_create(&cfg, 1, x.as_ptr(), mode.as_ptr()) };
            assert!(!session.is_null(), "method {method} was refused");
            let mut report = RgsaddleReport {
                version: RgsaddleVersion { major: 0, minor: 0 },
                flags: 0,
                status: 0,
                evaluations: 0,
                max_force: 0.0,
                ci_index: 0,
                iteration: 0,
                curvature: 0.0,
                rotations: 0,
            };
            let status = unsafe {
                rgsaddle_minmode_step(session, Some(well), std::ptr::null_mut(), &mut report)
            };
            assert_eq!(status, RGSADDLE_OK, "method {method}");
            assert!(report.max_force.is_finite());
            unsafe { rgsaddle_minmode_free(session) };
        }
    }

    #[test]
    fn an_unknown_band_method_is_not_fire() {
        let cfg = RgsaddleBandConfig {
            version: version(),
            flags: 0,
            tangent: 1,
            spring: 0,
            projection: 1,
            method: 99,
            spring_k: 0.1,
            spring_ks: std::ptr::null(),
            ci_trigger_factor: 0.0,
            ci_trigger_force: 0.0,
            cell: std::ptr::null(),
            force_tol: 1e-3,
            max_move: 0.2,
            memory: 0,
        };
        let positions = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0];
        let band = unsafe { rgsaddle_band_create(&cfg, 3, 1, positions.as_ptr()) };
        assert!(band.is_null());
        let mut nlcg = cfg;
        nlcg.method = 4;
        let band = unsafe { rgsaddle_band_create(&nlcg, 3, 1, positions.as_ptr()) };
        assert!(!band.is_null());
        unsafe { rgsaddle_band_free(band) };
        let _ = RGSADDLE_INVALID_PARAMETER;
    }
}
