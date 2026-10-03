//! Curvature used for convergence belongs to the reported position.

use std::sync::Mutex;

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

struct Cubic {
    coefficient: f64,
    fail_accepted_probe: bool,
    points: Mutex<Vec<Array1<f64>>>,
}

impl PointSurface for Cubic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.points.lock().unwrap().push(x.to_owned());
        if self.fail_accepted_probe && x[0] > 1.0 {
            return Err(SaddleError::Surface(
                "accepted curvature unavailable".into(),
            ));
        }
        let a = self.coefficient;
        let value = a * x[0].powi(3) / 3.0 - x[0] * x[0] / 2.0
            + (1.0 - a) * x[0]
            + x[1] * x[1]
            + x[2] * x[2];
        let gradient = array![a * x[0] * x[0] - x[0] + 1.0 - a, 2.0 * x[1], 2.0 * x[2]];
        Ok((value, gradient))
    }
}

fn translated_stationary_point(
    kind: MinModeKind,
    coefficient: f64,
    expected_status: MinModeStatus,
) {
    let config = MinModeConfig {
        kind,
        method: rgmin::Method::Lbfgs { memory: 4 },
        dr: 1e-3,
        rotation_tol: 1e-8,
        force_tol: 1e-10,
        max_move: 2.0,
        ..MinModeConfig::default()
    };
    let surface = Cubic {
        coefficient,
        fail_accepted_probe: false,
        points: Mutex::new(Vec::new()),
    };
    let mut session =
        MinModeSession::new(config, array![-1.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
    let report = session.step(&surface).unwrap();
    let points = surface.points.lock().unwrap();
    eprintln!(
        "{kind:?}: x={:?}, report={report:?}, points={points:?}",
        session.position()
    );
    assert_eq!(session.position(), array![1.0, 0.0, 0.0].view());
    assert_eq!(report.max_force, 0.0);
    assert_eq!(report.status, expected_status);
    let expected_curvature = 2.0 * coefficient - 1.0 + coefficient * 1e-3;
    assert!(
        (report.curvature - expected_curvature).abs() < 1e-10,
        "{kind:?}: measured {} rather than {expected_curvature}",
        report.curvature
    );
    assert_eq!(session.curvature(), report.curvature);
    assert!(
        points.iter().any(|point| point[0] > 1.0),
        "{kind:?}: no Hessian probe at the accepted stationary point"
    );
    assert_eq!(report.evaluations, points.len());
    assert_eq!(report.evaluations, 4);
}

#[test]
fn dimer_translation_to_a_minimum_is_not_converged() {
    translated_stationary_point(MinModeKind::Dimer, 1.0, MinModeStatus::Running);
}

#[test]
fn lanczos_translation_to_a_minimum_is_not_converged() {
    translated_stationary_point(MinModeKind::Lanczos, 1.0, MinModeStatus::Running);
}

#[test]
fn dimer_translation_to_a_saddle_uses_its_negative_curvature() {
    translated_stationary_point(MinModeKind::Dimer, -1.0, MinModeStatus::Converged);
}

#[test]
fn lanczos_translation_to_a_saddle_uses_its_negative_curvature() {
    translated_stationary_point(MinModeKind::Lanczos, -1.0, MinModeStatus::Converged);
}

fn failed_confirmation_preserves_position(kind: MinModeKind) {
    let config = MinModeConfig {
        kind,
        method: rgmin::Method::Lbfgs { memory: 4 },
        dr: 1e-3,
        rotation_tol: 1e-8,
        force_tol: 1e-10,
        max_move: 2.0,
        ..MinModeConfig::default()
    };
    let surface = Cubic {
        coefficient: 1.0,
        fail_accepted_probe: true,
        points: Mutex::new(Vec::new()),
    };
    let start = array![-1.0, 0.0, 0.0];
    let mut session = MinModeSession::new(config, start.clone(), array![1.0, 0.0, 0.0]).unwrap();
    let error = session.step(&surface).unwrap_err();
    assert!(
        matches!(error, SaddleError::Surface(ref message)
            if message == "accepted curvature unavailable"),
        "{error}"
    );
    assert_eq!(session.position(), start.view());
    let points = surface.points.lock().unwrap();
    assert_eq!(points.len(), 4);
    assert_eq!(points[2], array![1.0, 0.0, 0.0]);
    assert!(points[3][0] > 1.0);
}

#[test]
fn dimer_failed_confirmation_preserves_position() {
    failed_confirmation_preserves_position(MinModeKind::Dimer);
}

#[test]
fn lanczos_failed_confirmation_preserves_position() {
    failed_confirmation_preserves_position(MinModeKind::Lanczos);
}
