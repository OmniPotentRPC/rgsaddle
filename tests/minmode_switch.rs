//! The switched min-mode translation leaves a minimum basin: at a
//! nonnegative curvature only the reversed mode component drives the
//! walker, below zero the mode component is reversed and the
//! perpendicular force kept, and the walker stops at the saddle.

use std::sync::Mutex;

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

/// V = (x^2 - 1)^2 + 5 (y^2 + z^2) + 0.5 x y: minima near (+-1, 0, 0),
/// an index-1 saddle at the origin (curvature -4 along x), and a
/// coupling that bends the valley so the perpendicular force is not
/// zero along the escape. Every point it visits is recorded.
struct BentWell {
    visited: Mutex<Vec<Array1<f64>>>,
}

impl PointSurface for BentWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.visited.lock().unwrap().push(x.to_owned());
        let (a, b, c) = (x[0], x[1], x[2]);
        let e = (a * a - 1.0).powi(2) + 5.0 * (b * b + c * c) + 0.5 * a * b;
        let g = array![
            4.0 * a * (a * a - 1.0) + 0.5 * b,
            10.0 * b + 0.5 * a,
            10.0 * c
        ];
        Ok((e, g))
    }
}

/// From inside the basin of the minimum near x = 1 the walker leaves
/// the basin along the lowest mode and stops at the saddle.
fn escapes_the_basin(kind: MinModeKind, method: rgmin::Method) -> usize {
    let config = MinModeConfig {
        kind,
        method: method.clone(),
        force_tol: 1e-5,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let surface = BentWell {
        visited: Mutex::new(Vec::new()),
    };
    let start = array![0.85, 0.05, -0.04];
    let seed = array![1.0, 0.1, 0.0];
    let mut session = MinModeSession::new(config, start, seed).unwrap();
    let first = session.step(&surface).unwrap();
    assert!(
        first.curvature > 0.0,
        "{kind:?} {method:?}: start curvature {}",
        first.curvature
    );
    let report = session.run(&surface, 400).unwrap();
    assert_eq!(
        report.status,
        MinModeStatus::Converged,
        "{kind:?} {method:?}: {report:?}"
    );
    let x = session.position();
    for c in 0..3 {
        assert!(x[c].abs() < 1e-4, "{kind:?} {method:?}: {x:?}");
    }
    assert!(report.curvature < -3.0, "{report:?}");
    report.iteration
}

#[test]
fn dimer_fire_escapes_the_basin() {
    escapes_the_basin(
        MinModeKind::Dimer,
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
    );
}

#[test]
fn dimer_lbfgs_escapes_the_basin() {
    escapes_the_basin(MinModeKind::Dimer, rgmin::Method::Lbfgs { memory: 10 });
}

#[test]
fn lanczos_fire_escapes_the_basin() {
    escapes_the_basin(
        MinModeKind::Lanczos,
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
    );
}

#[test]
fn lanczos_lbfgs_escapes_the_basin() {
    escapes_the_basin(MinModeKind::Lanczos, rgmin::Method::Lbfgs { memory: 10 });
}

/// The dimer's estimate is a measured Rayleigh quotient, inside the
/// Hessian's spectrum [4.62, 10.05] at the basin start, also once its
/// rotational force falls under what forward differences resolve
/// (validation/dimer_extrapolation.py).
#[test]
fn dimer_estimate_stays_inside_the_spectrum() {
    let config = MinModeConfig {
        kind: MinModeKind::Dimer,
        ..MinModeConfig::default()
    };
    let surface = BentWell {
        visited: Mutex::new(Vec::new()),
    };
    let mut session =
        MinModeSession::new(config, array![0.85, 0.05, -0.04], array![1.0, 0.1, 0.0]).unwrap();
    let estimate = session.estimate_mode(&surface).unwrap();
    assert!(
        (4.62..=4.65).contains(&estimate.curvature),
        "curvature {} outside the lowest eigenvalue 4.6235 + dr/2 T",
        estimate.curvature
    );
    assert!(estimate.mode[0].abs() > 0.99, "{:?}", estimate.mode);
}
