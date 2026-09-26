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
