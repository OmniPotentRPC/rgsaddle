//! Minimum-mode search on an analytic saddle: from a displaced start
//! the session must climb to the origin, where the Hessian has one
//! negative eigenvalue.

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

/// V = -x^2 + y^2 + z^2: saddle at the origin, unstable along x.
struct QuadraticSaddle;

impl PointSurface for QuadraticSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let e = -x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        let g = array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
        Ok((e, g))
    }
}

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at (+-1, 0, 0) where every
/// curvature is positive (8 along x, 4 along y and z).
struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        let e = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
        let g = array![4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c];
        Ok((e, g))
    }
}

fn converges(kind: MinModeKind) {
    let config = MinModeConfig {
        kind,
        force_tol: 1e-4,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let start = array![0.35, 0.4, -0.3];
    let seed = array![0.8, 0.5, 0.1];
    let mut session = MinModeSession::new(config, start, seed).unwrap();
    let report = session.run(&QuadraticSaddle, 4000).unwrap();
    assert_eq!(
        report.status,
        MinModeStatus::Converged,
        "{kind:?} max_force={}",
        report.max_force
    );
    let x = session.position();
    for c in 0..3 {
        assert!(x[c].abs() < 1e-3, "{kind:?} coord {c} = {}", x[c]);
    }
    // The lowest mode is the unstable x direction, curvature -2.
    let mode = session.mode();
    assert!(mode[0].abs() > 0.99, "{kind:?} mode={mode:?}");
    assert!(
        report.curvature < -1.0,
        "{kind:?} curvature={}",
        report.curvature
    );
}

#[test]
fn dimer_rotation_finds_the_saddle() {
    converges(MinModeKind::Dimer);
}

#[test]
fn lanczos_finds_the_saddle() {
    converges(MinModeKind::Lanczos);
}

/// The force vanishes at a minimum as it does at a saddle. A walker
/// sitting there must report Running while the lowest curvature is
/// positive, whatever the force says.
fn minimum_is_not_converged(kind: MinModeKind) {
    let config = MinModeConfig {
        kind,
        force_tol: 1e-3,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let start = array![1.0, 0.0, 0.0];
    let seed = array![1.0, 0.0, 0.0];
    let mut session = MinModeSession::new(config, start, seed).unwrap();
    for step in 0..50 {
        let report = session.step(&DoubleWell).unwrap();
        assert!(
            report.max_force <= 1e-3,
            "{kind:?} step {step}: max_force={} at a stationary point",
            report.max_force
        );
        assert!(
            report.curvature > 0.0,
            "{kind:?} step {step}: curvature={} at a minimum",
            report.curvature
        );
        assert_eq!(
            report.status,
            MinModeStatus::Running,
            "{kind:?} step {step}: max_force={} curvature={}",
            report.max_force,
            report.curvature
        );
    }
}

#[test]
fn dimer_does_not_converge_at_a_minimum() {
    minimum_is_not_converged(MinModeKind::Dimer);
}

#[test]
fn lanczos_does_not_converge_at_a_minimum() {
    minimum_is_not_converged(MinModeKind::Lanczos);
}
