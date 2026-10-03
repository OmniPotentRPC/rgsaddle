//! Minimum-mode search on an analytic saddle: from a displaced start
//! the session must climb to the origin, where the Hessian has one
//! negative eigenvalue.

use std::sync::atomic::{AtomicUsize, Ordering};

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

/// The quadratic saddle, failing at any point farther than `radius`
/// from `center`. The finite-difference shell (dr = 1e-3) stays
/// inside; the first FIRE trial (max_move = 0.1) lands outside, so the
/// failure happens inside the translation oracle.
struct FailingBeyond {
    center: Array1<f64>,
    radius: f64,
    calls: AtomicUsize,
}

impl PointSurface for FailingBeyond {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let d = &x - &self.center;
        if d.dot(&d).sqrt() > self.radius {
            return Err(SaddleError::Surface("scf did not converge".into()));
        }
        QuadraticSaddle.eval(x)
    }
}

#[test]
fn surface_error_inside_a_step_is_returned_and_the_walker_stays_put() {
    let config = MinModeConfig {
        force_tol: 1e-4,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let start = array![0.35, 0.4, -0.3];
    let seed = array![0.8, 0.5, 0.1];
    let mut session = MinModeSession::new(config, start.clone(), seed).unwrap();
    let surface = FailingBeyond {
        center: start.clone(),
        radius: 0.01,
        calls: AtomicUsize::new(0),
    };
    let err = session.step(&surface).unwrap_err();
    assert!(
        matches!(err, SaddleError::Surface(ref m) if m.contains("scf did not converge")),
        "{err}"
    );
    assert_eq!(session.position(), start.view());
    // Initial gradient, at least one rotation action, the oracle's
    // evaluations: the failure came after the rotation phase.
    assert!(surface.calls.load(Ordering::Relaxed) >= 3);
}

#[test]
fn dimer_does_not_converge_at_a_minimum() {
    minimum_is_not_converged(MinModeKind::Dimer);
}

#[test]
fn lanczos_does_not_converge_at_a_minimum() {
    minimum_is_not_converged(MinModeKind::Lanczos);
}

/// V = 1/2 x^T diag(d) x with d = (-1, 0.5, 1, 2, 4, 8): lowest mode e0,
/// curvature -1, and a spread of positive curvatures to rotate past.
struct Anisotropic;

const D: [f64; 6] = [-1.0, 0.5, 1.0, 2.0, 4.0, 8.0];

impl PointSurface for Anisotropic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let g = Array1::from_iter(x.iter().zip(D.iter()).map(|(xi, di)| di * xi));
        let e = 0.5 * x.iter().zip(g.iter()).map(|(a, b)| a * b).sum::<f64>();
        Ok((e, g))
    }
}

struct CountingPoint<'a, S: PointSurface> {
    inner: &'a S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for CountingPoint<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

fn estimate_from_bad_seed(kind: MinModeKind) -> (f64, usize, usize, Array1<f64>) {
    let config = MinModeConfig {
        kind,
        rotation_tol: 1e-6,
        max_rotations: 50,
        krylov_dim: 6,
        ..MinModeConfig::default()
    };
    let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.02]);
    let seed = Array1::from(vec![0.3, 0.5, 0.4, 0.4, 0.4, 0.4]);
    let mut session = MinModeSession::new(config, x, seed).unwrap();
    let surface = CountingPoint {
        inner: &Anisotropic,
        calls: AtomicUsize::new(0),
    };
    let est = session.estimate_mode(&surface).unwrap();
    assert_eq!(est.evaluations, surface.calls.load(Ordering::Relaxed));
    (est.curvature, est.rotations, est.evaluations, est.mode)
}

#[test]
fn dimer_rotation_reaches_the_lowest_mode_of_a_quadratic() {
    let (curvature, rotations, evaluations, mode) = estimate_from_bad_seed(MinModeKind::Dimer);
    // Forward differences are exact on a quadratic, up to rounding at
    // dr = 1e-3.
    assert!((curvature + 1.0).abs() < 1e-8, "curvature {curvature}");
    assert!(mode[0].abs() > 1.0 - 1e-9, "mode {mode:?}");
    // Conjugate rotation planes with the exact in-plane step reach the
    // 1e-6 rotational force in 16 rotations here (a numpy model of the
    // same iteration agrees); steepest-descent planes need 45.
    assert!(rotations <= 18, "rotations {rotations}");
    // Centre, the first dimer gradient, one trial per rotation.
    assert_eq!(evaluations, 2 + rotations);
}

#[test]
fn lanczos_stops_on_the_ritz_residual() {
    let (curvature, actions, evaluations, mode) = estimate_from_bad_seed(MinModeKind::Lanczos);
    assert!((curvature + 1.0).abs() < 1e-8, "curvature {curvature}");
    assert!(mode[0].abs() > 1.0 - 1e-9, "mode {mode:?}");
    assert!(actions <= 6);
    assert_eq!(evaluations, 1 + actions);
}

fn warm_session_costs_one_evaluation(kind: MinModeKind) {
    let config = MinModeConfig {
        kind,
        rotation_tol: 1e-6,
        ..MinModeConfig::default()
    };
    let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.02]);
    let mut seed = Array1::zeros(6);
    seed[0] = 1.0;
    let mut session = MinModeSession::new(config, x.clone(), seed).unwrap();
    let surface = CountingPoint {
        inner: &Anisotropic,
        calls: AtomicUsize::new(0),
    };
    // The seed is the mode: the centre plus one dimer gradient (for
    // Lanczos, one Hessian action, which `rotations` counts).
    let first_rotations = usize::from(kind == MinModeKind::Lanczos);
    let est = session.estimate_mode(&surface).unwrap();
    assert_eq!(
        (est.rotations, est.evaluations),
        (first_rotations, 2),
        "{kind:?}"
    );
    // The same session moved to a new point with the host's gradient:
    // one evaluation, the dimer endpoint.
    let x2 = &x * 0.5;
    let (_, g2) = Anisotropic.eval(x2.view()).unwrap();
    session.set_position(x2, Some(g2)).unwrap();
    surface.calls.store(0, Ordering::Relaxed);
    let est = session.estimate_mode(&surface).unwrap();
    assert_eq!(
        (est.rotations, est.evaluations),
        (first_rotations, 1),
        "{kind:?}"
    );
    assert_eq!(surface.calls.load(Ordering::Relaxed), 1);
    assert!((est.curvature + 1.0).abs() < 1e-8);
}

#[test]
fn dimer_session_is_reusable_across_points() {
    warm_session_costs_one_evaluation(MinModeKind::Dimer);
}

#[test]
fn lanczos_session_is_reusable_across_points() {
    warm_session_costs_one_evaluation(MinModeKind::Lanczos);
}

#[test]
fn a_step_reuses_its_accepted_point_as_the_next_centre() {
    let config = MinModeConfig {
        force_tol: 1e-6,
        max_move: 0.05,
        ..MinModeConfig::default()
    };
    let start = array![0.35, 0.4, -0.3];
    let seed = array![1.0, 0.0, 0.0];
    let mut session = MinModeSession::new(config, start, seed).unwrap();
    let surface = CountingPoint {
        inner: &QuadraticSaddle,
        calls: AtomicUsize::new(0),
    };
    session.step(&surface).unwrap();
    for k in 0..5 {
        surface.calls.store(0, Ordering::Relaxed);
        let report = session.step(&surface).unwrap();
        let calls = surface.calls.load(Ordering::Relaxed);
        assert_eq!(report.evaluations, calls, "step {k}");
        // No centre evaluation and no re-evaluation of the accepted
        // point: one dimer gradient, one per rotation, one trial.
        assert_eq!(calls, 1 + report.rotations + 1, "step {k}");
    }
}
