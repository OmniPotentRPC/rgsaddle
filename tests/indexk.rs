//! Index-k step against the one-mode translator, and an index-2 saddle.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{
    IndexKConfig, IndexKSession, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus,
    PointSurface, SaddleError, indexk_step, reflect_force, translated_force,
};

struct QuadraticSaddle;

impl PointSurface for QuadraticSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let energy = -x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        Ok((energy, array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]))
    }
}

struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        let energy = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
        Ok((energy, array![4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c]))
    }
}

const PREFACTOR: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
        for i in 0..4 {
            let (dx, dy) = (x[0] - X0[i], x[1] - Y0[i]);
            let term = PREFACTOR[i] * (XX[i] * dx * dx + XY[i] * dx * dy + YY[i] * dy * dy).exp();
            energy += term;
            gx += term * (2.0 * XX[i] * dx + XY[i] * dy);
            gy += term * (XY[i] * dx + 2.0 * YY[i] * dy);
        }
        Ok((energy + 2.0 * x[2] * x[2], array![gx, gy, 4.0 * x[2]]))
    }
}

/// V = -x² - ½ y² + z². Hessian eigenvalues -2, -1, 2. Index 2 at the origin.
struct IndexTwo;

impl PointSurface for IndexTwo {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y, z) = (r[0], r[1], r[2]);
        Ok((-x * x - 0.5 * y * y + z * z, array![-2.0 * x, -y, 2.0 * z]))
    }
}

fn same_bits(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "index {index}: {a} vs {b}");
    }
}

fn column(mode: ArrayView1<f64>) -> Array2<f64> {
    let mut modes = Array2::zeros((mode.len(), 1));
    modes.column_mut(0).assign(&mode);
    modes
}

fn one_matches<S: PointSurface>(
    kind: MinModeKind,
    surface: &S,
    start: Array1<f64>,
    seed: Array1<f64>,
    steps: usize,
) {
    let config = MinModeConfig {
        kind,
        force_tol: 1e-12,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let mut one = MinModeSession::new(config.clone(), start.clone(), seed.clone()).unwrap();
    let mut many = IndexKSession::new(
        IndexKConfig::new(1, config),
        start,
        Some(column(seed.view())),
    )
    .unwrap();
    for step in 0..steps {
        let left = one.step(surface).unwrap();
        let right = many.step(surface).unwrap();
        same_bits(
            one.position().as_slice().unwrap(),
            many.position().as_slice().unwrap(),
        );
        same_bits(
            one.mode().as_slice().unwrap(),
            &many.modes().column(0).to_vec(),
        );
        assert_eq!(
            left.curvature.to_bits(),
            right.curvatures[0].to_bits(),
            "{kind:?} step {step} curvature"
        );
        let translated = one.translated_force().expect("one-mode translated force");
        same_bits(
            translated.as_slice().unwrap(),
            right.translated_force.as_slice().unwrap(),
        );
    }
}

#[test]
fn k_equals_one_matches_dimer_and_lanczos_on_three_surfaces() {
    let start = array![0.35, 0.4, -0.3];
    let seed = array![0.8, 0.5, 0.1];
    for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
        one_matches(kind, &QuadraticSaddle, start.clone(), seed.clone(), 4);
        one_matches(kind, &DoubleWell, start.clone(), seed.clone(), 3);
        one_matches(
            kind,
            &MullerBrown,
            array![-0.5, 1.4, 0.2],
            array![1.0, -0.4, 0.2],
            3,
        );
    }
}

#[test]
fn one_shot_with_the_host_force_matches_set_position() {
    let surface = QuadraticSaddle;
    let start = array![0.35, 0.4, -0.3];
    let seed = array![0.8, 0.5, 0.1];
    let (_, force) = surface.eval(start.view()).unwrap();
    let config = MinModeConfig {
        kind: MinModeKind::Lanczos,
        force_tol: 1e-12,
        max_move: 0.1,
        ..MinModeConfig::default()
    };
    let mut one = MinModeSession::new(config.clone(), start.clone(), seed.clone()).unwrap();
    one.set_position(start.clone(), Some(force.clone()))
        .unwrap();
    let left = one.step(&surface).unwrap();
    let right = indexk_step(
        start,
        force,
        &surface,
        1,
        Some(column(seed.view())),
        &config,
    )
    .unwrap();
    same_bits(
        one.position().as_slice().unwrap(),
        right.position.as_slice().unwrap(),
    );
    same_bits(
        one.mode().as_slice().unwrap(),
        &right.modes.column(0).to_vec(),
    );
    assert_eq!(left.curvature.to_bits(), right.curvatures[0].to_bits());
    same_bits(
        one.translated_force().unwrap().as_slice().unwrap(),
        right.translated_force.as_slice().unwrap(),
    );
}

#[test]
fn index_two_quadratic_converges_from_nearby() {
    let mut config = IndexKConfig::new(2, MinModeConfig::default());
    config.search.force_tol = 1e-6;
    config.search.max_move = 0.1;
    config.residual_tol = 1e-8;
    let mut session = IndexKSession::new(config, array![0.3, -0.25, 0.2], None).unwrap();
    let report = session.run(&IndexTwo, 400).unwrap();
    assert_eq!(report.status, MinModeStatus::Converged);
    assert!(session.position().iter().all(|c| c.abs() < 1e-3));
    assert!(report.curvatures.iter().all(|c| *c < 0.0));
    assert!((report.curvatures[0] + 2.0).abs() < 1e-4);
    assert!((report.curvatures[1] + 1.0).abs() < 1e-4);
}

fn expect_err(result: Result<IndexKSession, SaddleError>, kind: &str) {
    match result {
        Err(SaddleError::Invalid(message)) if kind == "invalid" => assert!(!message.is_empty()),
        Err(SaddleError::Shape(message)) if kind == "shape" => assert!(!message.is_empty()),
        Err(err) => panic!("unexpected error {err}"),
        Ok(_) => panic!("index-k session accepted a bad k"),
    }
}

#[test]
fn bad_k_is_rejected() {
    let x = array![0.1, 0.2, 0.3];
    expect_err(
        IndexKSession::new(
            IndexKConfig::new(0, MinModeConfig::default()),
            x.clone(),
            None,
        ),
        "invalid",
    );
    expect_err(
        IndexKSession::new(
            IndexKConfig::new(4, MinModeConfig::default()),
            x.clone(),
            None,
        ),
        "invalid",
    );
    let modes = Array2::zeros((3, 1));
    expect_err(
        IndexKSession::new(
            IndexKConfig::new(2, MinModeConfig::default()),
            x,
            Some(modes),
        ),
        "shape",
    );
}

#[test]
fn reflect_matches_the_one_mode_spelling_and_the_block_sum() {
    let force = array![0.5, -1.0, 0.25];
    let mut one = Array2::zeros((3, 1));
    one[(1, 0)] = 1.0;
    let reflected = reflect_force(force.view(), one.view()).unwrap();
    assert_eq!(reflected[0].to_bits(), force[0].to_bits());
    assert_eq!(reflected[1].to_bits(), (-force[1]).to_bits());
    assert_eq!(reflected[2].to_bits(), force[2].to_bits());

    let mut modes = Array2::zeros((3, 2));
    modes[(0, 0)] = 1.0;
    modes[(1, 1)] = 1.0;
    let block = reflect_force(force.view(), modes.view()).unwrap();
    assert_eq!(block[0].to_bits(), (-force[0]).to_bits());
    assert_eq!(block[1].to_bits(), (-force[1]).to_bits());
    assert_eq!(block[2].to_bits(), force[2].to_bits());

    let curvatures = array![1.0, -0.5];
    let climbed = translated_force(force.view(), modes.view(), curvatures.view()).unwrap();
    assert_eq!(climbed[0].to_bits(), (-force[0]).to_bits());
    assert_eq!(climbed[1].to_bits(), (-force[1]).to_bits());
    assert_eq!(climbed[2].to_bits(), 0.0_f64.to_bits());
}
