//! The min-mode translation switches on the sign of the curvature: at
//! a nonnegative curvature only the inverted mode component drives the
//! walker, below it the mode component is inverted and the
//! perpendicular force kept.

use std::sync::Mutex;

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::minmode::effective_gradient;
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

#[test]
fn nonnegative_curvature_keeps_the_inverted_mode_component_alone() {
    let g = array![3.0, -4.0, 12.0];
    let tau = array![0.6, 0.8, 0.0];
    // g.tau = 1.8 - 3.2 = -1.4.
    let climb = effective_gradient(g.view(), tau.view(), true);
    for (got, want) in climb.iter().zip([0.84, 1.12, 0.0]) {
        assert!((got - want).abs() < 1e-15, "{climb:?}");
    }
    let reflected = effective_gradient(g.view(), tau.view(), false);
    for (got, want) in reflected.iter().zip([4.68, -1.76, 12.0]) {
        assert!((got - want).abs() < 1e-14, "{reflected:?}");
    }
    // Both branches reverse the mode component.
    assert!((climb.dot(&tau) - 1.4).abs() < 1e-15);
    assert!((reflected.dot(&tau) - 1.4).abs() < 1e-14);
}

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

/// In the convex region the step follows the mode alone: from a point
/// with a large perpendicular force the first translation leaves the
/// component orthogonal to the mode where it was.
#[test]
fn convex_step_moves_along_the_mode_only() {
    let config = MinModeConfig {
        kind: MinModeKind::Lanczos,
        method: rgmin::Method::Lbfgs { memory: 4 },
        rotation_tol: 1e-10,
        max_move: 0.05,
        ..MinModeConfig::default()
    };
    // V = 2 x^2 + 8 y^2: lowest mode x (curvature 4), the walker sits
    // at y = 0.3 where the perpendicular force is -4.8.
    struct Bowl;
    impl PointSurface for Bowl {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let e = 2.0 * x[0] * x[0] + 8.0 * x[1] * x[1] + 8.0 * x[2] * x[2];
            Ok((e, array![4.0 * x[0], 16.0 * x[1], 16.0 * x[2]]))
        }
    }
    let start = array![0.2, 0.3, 0.0];
    let mut session = MinModeSession::new(config, start.clone(), array![1.0, 0.0, 0.0]).unwrap();
    let report = session.step(&Bowl).unwrap();
    assert!(report.curvature > 3.9, "{report:?}");
    let x = session.position();
    assert!((x[1] - start[1]).abs() < 1e-8, "{x:?}");
    assert!((x[2] - start[2]).abs() < 1e-8, "{x:?}");
    // Uphill along the mode, away from the minimum at x = 0.
    assert!(x[0] > start[0], "{x:?}");
}
