//! A soft negative curvature is a Newton step along the mode, so the
//! saddle is reached in a handful of translations rather than a crawl
//! at the empty-memory scale.

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface, SaddleError};

/// V = -0.005 x² + 100 y². Curvature along x is -0.01; y is stiff.
struct SoftSaddle;

impl PointSurface for SoftSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let energy = -0.005 * x[0] * x[0] + 100.0 * x[1] * x[1];
        Ok((energy, array![-0.01 * x[0], 200.0 * x[1]]))
    }
}

#[test]
fn soft_negative_curvature_converges_in_few_translations() {
    let config = MinModeConfig {
        method: rgmin::Method::Lbfgs { memory: 8 },
        force_tol: 1e-3,
        max_move: 1.5,
        max_rotations: 4,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, array![1.0, 0.02], array![1.0, 0.0]).unwrap();
    let report = session.run(&SoftSaddle, 3).unwrap();
    assert_eq!(
        report.status,
        MinModeStatus::Converged,
        "position {:?} force {} curvature {} after {} steps",
        session.position(),
        report.max_force,
        report.curvature,
        report.iteration
    );
    assert!(session.position().iter().all(|c| c.abs() < 1e-3));
}
