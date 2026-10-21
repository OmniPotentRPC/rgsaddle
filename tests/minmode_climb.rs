//! Stock min-mode climb out of a convex well.
//!
//! The optbench saddle-search harness asks for `rotation_tol` 1e-4,
//! `rotation_angle_tol` 0, and `max_rotations` 20. A search that spends
//! the rotation cap on every step, and that grows an empty-memory step
//! by a fraction of the force while the curvature is positive, does not
//! reach the saddle inside a few hundred evaluations. The stock defaults
//! stop the dimer on the classic torque window, keep the translator's
//! memory across a small mode turn, and leave a non-negative curvature
//! by one full `max_move`.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

/// `V = (x^2 - 1)^2 + 2 y^2 + 2 z^2`. The index-1 saddle is the origin.
/// Along x the curvature `12 x^2 - 4` is positive on
/// `(1/sqrt(3), sqrt(2/3))` and is the lowest eigenvalue there, so a
/// start at x = 0.7 climbs the barrier, not the perpendicular well.
struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        let energy = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
        let gradient = array![4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c];
        Ok((energy, gradient))
    }
}

struct Counting {
    calls: AtomicUsize,
}

impl PointSurface for Counting {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        DoubleWell.eval(x)
    }
}

/// Harness values, not [`MinModeConfig::default`]: the lbfgs, fire, and
/// Lanczos templates pass an explicit angle tolerance of zero.
fn harness(kind: MinModeKind, method: rgmin::Method) -> MinModeConfig {
    MinModeConfig {
        kind,
        method,
        dr: 1e-3,
        rotation_tol: 1e-4,
        rotation_angle_tol: 0.0,
        max_rotations: 20,
        krylov_dim: 12,
        force_tol: 1e-3,
        max_move: 0.2,
        ..MinModeConfig::default()
    }
}

fn climb(kind: MinModeKind, method: rgmin::Method) {
    let label = format!("{kind:?} {method:?}");
    let surface = Counting {
        calls: AtomicUsize::new(0),
    };
    let start = array![0.70, 0.05, -0.04];
    let seed = array![0.3, 0.8, 0.2];
    let mut session = MinModeSession::new(harness(kind, method), start.clone(), seed).unwrap();
    let first = session.step(&surface).unwrap();
    assert_eq!(first.status, MinModeStatus::Running, "{label}");
    assert!(
        first.curvature > 0.0,
        "{label} curvature {}",
        first.curvature
    );
    let moved = &session.position() - &start;
    let travel = moved.dot(&moved).sqrt();
    assert!(
        (travel - 0.2).abs() < 1e-9,
        "{label} left the convex region by {travel}"
    );
    if kind == MinModeKind::Dimer {
        assert!(
            first.rotations <= 2,
            "{label} spent {} rotations on the first step",
            first.rotations
        );
    }
    let mut report = first;
    let mut steps = 1usize;
    while report.status == MinModeStatus::Running && steps < 40 {
        report = session.step(&surface).unwrap();
        steps += 1;
    }
    let calls = surface.calls.load(Ordering::Relaxed);
    assert_eq!(
        report.status,
        MinModeStatus::Converged,
        "{label} still running after {steps} steps, {calls} evals, x={:?}",
        session.position()
    );
    assert!(calls < 300, "{label} used {calls} evaluations");
    let x = session.position();
    for c in 0..3 {
        assert!(x[c].abs() < 2e-2, "{label} coord {c} = {}", x[c]);
    }
    assert!(
        report.curvature < 0.0,
        "{label} curvature {}",
        report.curvature
    );
}

#[test]
fn dimer_lbfgs_climbs_the_double_well() {
    climb(MinModeKind::Dimer, rgmin::Method::Lbfgs { memory: 20 });
}

#[test]
fn dimer_fire_climbs_the_double_well() {
    climb(
        MinModeKind::Dimer,
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
    );
}

#[test]
fn lanczos_lbfgs_climbs_the_double_well() {
    climb(MinModeKind::Lanczos, rgmin::Method::Lbfgs { memory: 20 });
}

#[test]
fn lanczos_fire_climbs_the_double_well() {
    climb(
        MinModeKind::Lanczos,
        rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
    );
}

#[test]
fn rust_default_angle_is_five_degrees() {
    let tol = MinModeConfig::default().rotation_angle_tol;
    let five = 5.0 * std::f64::consts::PI / 180.0;
    assert!((tol - five).abs() < 1e-15, "{tol}");
}
