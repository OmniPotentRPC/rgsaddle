//! A positive curvature climbs the mode and drops the perpendicular force.

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{MinModeConfig, MinModeKind, MinModeSession, PointSurface, SaddleError};

/// `E = 0.5 c x^2 + 0.5 y^2 + 0.5 z^2`. The lowest mode is `x` when
/// `|c|` is the extreme curvature.
struct Quadratic {
    curvature: f64,
}

impl PointSurface for Quadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let c = self.curvature;
        let value = 0.5 * c * x[0] * x[0] + 0.5 * x[1] * x[1] + 0.5 * x[2] * x[2];
        Ok((value, array![c * x[0], x[1], x[2]]))
    }
}

fn one_step(kind: MinModeKind, curvature: f64) -> Array1<f64> {
    let config = MinModeConfig {
        kind,
        max_move: 2.0,
        force_tol: 1e-12,
        rotation_tol: 1e-8,
        ..MinModeConfig::default()
    };
    let mut session =
        MinModeSession::new(config, array![1.0, 1.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
    session.step(&Quadratic { curvature }).unwrap();
    session.position().to_owned()
}

#[test]
fn dimer_positive_curvature_does_not_move_the_perpendicular_coordinate() {
    let position = one_step(MinModeKind::Dimer, 0.5);
    assert!((position[1] - 1.0).abs() < 1e-9, "{position:?}");
    assert!(position[0] > 1.0, "{position:?}");
}

#[test]
fn lanczos_positive_curvature_does_not_move_the_perpendicular_coordinate() {
    let position = one_step(MinModeKind::Lanczos, 0.5);
    assert!((position[1] - 1.0).abs() < 1e-9, "{position:?}");
    assert!(position[0] > 1.0, "{position:?}");
}

#[test]
fn dimer_negative_curvature_descends_the_perpendicular_coordinate() {
    let position = one_step(MinModeKind::Dimer, -2.0);
    assert!(position[1] < 1.0, "{position:?}");
}

#[test]
fn lanczos_negative_curvature_descends_the_perpendicular_coordinate() {
    let position = one_step(MinModeKind::Lanczos, -2.0);
    assert!(position[1] < 1.0, "{position:?}");
}
