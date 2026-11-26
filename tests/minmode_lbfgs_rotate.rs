//! Opt-in L-BFGS dimer rotation on a 2-D Müller-Brown saddle and on one
//! quadratic saddle where the rotation plane is not unique.

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    MinModeConfig, MinModeSession, MinModeStatus, PointSurface, SaddleError, SaddleIndexCheck,
};

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
/// Saddle between the two central Müller-Brown wells.
const MB_SADDLE: [f64; 2] = [0.212_486_582_000_662, 0.292_988_325_107_368];

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (px, py) = (x[0], x[1]);
        let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
        for i in 0..4 {
            let dx = px - MB_X0[i];
            let dy = py - MB_Y0[i];
            let term = MB_A[i] * (MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy).exp();
            energy += term;
            gx += term * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
            gy += term * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
        }
        Ok((energy, array![gx, gy]))
    }
}

/// Diagonal saddle `1/2 Σ λ_i x_i²` with one negative curvature.
struct QuadraticSaddle;

const QUADRATIC_LAMBDA: [f64; 8] = [-2.0, 0.3, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];

impl PointSurface for QuadraticSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for (i, &lambda) in QUADRATIC_LAMBDA.iter().enumerate() {
            energy += 0.5 * lambda * x[i] * x[i];
            gradient[i] = lambda * x[i];
        }
        Ok((energy, gradient))
    }
}

fn mb_config() -> MinModeConfig {
    MinModeConfig::default()
        .with_force_tol(1e-8)
        .with_max_move(0.15)
        .with_saddle_check(None::<SaddleIndexCheck>)
}

#[test]
fn muller_brown_lbfgs_rotation_matches_the_saddle() {
    let start = array![0.25, 0.25];
    let mode = array![1.0, -0.3];
    let mut lbfgs = MinModeSession::new(mb_config(), start.clone(), mode.clone()).unwrap();
    lbfgs.set_lbfgs_rotate(true);
    let lbfgs_report = lbfgs.run(&MullerBrown, 400).unwrap();
    assert_eq!(
        lbfgs_report.status,
        MinModeStatus::Converged,
        "status {:?} force {} curvature {}",
        lbfgs_report.status,
        lbfgs_report.max_force,
        lbfgs_report.curvature
    );
    let found = lbfgs.position();
    assert!(
        (found[0] - MB_SADDLE[0]).abs() < 1e-6 && (found[1] - MB_SADDLE[1]).abs() < 1e-6,
        "found ({}, {}), saddle ({}, {})",
        found[0],
        found[1],
        MB_SADDLE[0],
        MB_SADDLE[1]
    );

    let mut conjugate = MinModeSession::new(mb_config(), start, mode).unwrap();
    let conjugate_report = conjugate.run(&MullerBrown, 400).unwrap();
    assert_eq!(conjugate_report.status, MinModeStatus::Converged);
    let plain = conjugate.position();
    assert!(
        (plain[0] - MB_SADDLE[0]).abs() < 1e-6 && (plain[1] - MB_SADDLE[1]).abs() < 1e-6,
        "default ({}, {})",
        plain[0],
        plain[1]
    );
}

fn rotation_estimate(lbfgs_rotate: bool) -> (usize, f64) {
    let config = MinModeConfig::default()
        .with_rotation_tol(1e-6)
        .with_max_rotations(80)
        .with_saddle_check(None::<SaddleIndexCheck>);
    let start = array![0.2, -0.15, 0.1, 0.05, -0.08, 0.04, 0.03, -0.02];
    let mode = array![0.2, 0.5, -0.3, 0.4, 0.1, -0.6, 0.25, 0.35];
    let mut session = MinModeSession::new(config, start, mode).unwrap();
    session.set_lbfgs_rotate(lbfgs_rotate);
    let estimate = session.estimate_mode(&QuadraticSaddle).unwrap();
    (estimate.evaluations, estimate.curvature)
}

#[test]
fn lbfgs_rotation_saves_gradients_on_the_same_start() {
    let (conjugate_calls, conjugate_curvature) = rotation_estimate(false);
    let (lbfgs_calls, lbfgs_curvature) = rotation_estimate(true);
    let lowest = QUADRATIC_LAMBDA[0];
    assert!(
        (conjugate_curvature - lowest).abs() < 1e-6,
        "conjugate curvature {conjugate_curvature}"
    );
    assert!(
        (lbfgs_curvature - lowest).abs() < 1e-6,
        "lbfgs curvature {lbfgs_curvature}"
    );
    assert!(
        lbfgs_calls < conjugate_calls,
        "lbfgs {lbfgs_calls} conjugate {conjugate_calls}"
    );
}
