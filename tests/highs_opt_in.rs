use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface, SaddleError};

struct Saddle;
impl PointSurface for Saddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((-x[0] * x[0] + x[1] * x[1] + x[2] * x[2], array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]))
    }
}

#[test]
fn optional_feasible_steps_preserve_the_saddle_and_atom_motion_cap() {
    for enabled in [false, true] {
        let config = MinModeConfig { method: rgmin::Method::Lbfgs { memory: 5 }, max_move: 0.05, force_tol: 1e-5, ..Default::default() };
        let mut session = MinModeSession::new(config, array![0.35, 0.4, -0.3], array![1.0, 0.0, 0.0]).unwrap();
        session.set_highs(enabled);
        let mut converged = false;
        for _ in 0..4000 {
            let previous = session.position().to_owned();
            let report = session.step(&Saddle).unwrap();
            let delta = &session.position() - &previous;
            assert!(delta.dot(&delta).sqrt() <= 0.05 * (1.0 + 1e-12));
            if report.status == MinModeStatus::Converged {
                assert!(report.curvature < -1.0);
                assert!(report.max_force <= 1e-5);
                assert!(session.position().iter().all(|x| x.abs() < 1e-5));
                converged = true;
                break;
            }
        }
        assert!(converged, "HiGHS control {enabled}");
    }
}
