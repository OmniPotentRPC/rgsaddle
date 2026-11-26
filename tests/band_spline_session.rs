//! A band session with a spline tangent steps on the double well and
//! stores the same interior force as the one-shot projection.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, ExpBandPrecon, FreeEndConfig,
    SaddleError, StringConfig, TangentKind, band_forces,
};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2. Minima at (+-1, 0, 0), saddle at
/// the origin.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

fn initial_band(n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    band
}

fn spline_config() -> BandConfig {
    BandConfig::default()
        .with_tangent(TangentKind::Spline)
        .with_force_tol(1e-3)
        .with_max_move(0.1)
        .with_min_mode(None)
        .with_saddle_check(None)
}

fn assert_matches(config: &BandConfig, session: &BandSession) {
    let eval = session.evaluation().expect("the step stored an evaluation");
    let projected = band_forces(
        config,
        &DoubleWell,
        session.positions(),
        session.climbing_image(),
    )
    .expect("one-shot spline projection")
    .force;
    let flat: Vec<f64> = eval.projected.iter().copied().collect();
    assert_eq!(flat.len(), projected.len());
    for (got, want) in flat.iter().zip(projected.iter()) {
        assert!(
            (got - want).abs() <= 1e-9,
            "session force {got} differs from the one-shot projection {want}"
        );
    }
}

#[test]
fn spline_session_steps_and_matches_the_one_shot_projection() {
    let n_images = 7;
    let config = spline_config().with_climbing(None);
    let mut session = BandSession::new(config.clone(), initial_band(n_images)).expect("session");
    let first = session.step(&DoubleWell).expect("first spline step");
    assert_eq!(first.iteration, 1);
    assert_matches(&config, &session);
    let mut report = first;
    for _ in 1..4000 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = session.step(&DoubleWell).expect("spline step");
    }
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "spline band still running at force {}",
        report.max_force
    );
    assert!(report.max_force <= 1e-3);
    assert_matches(&config, &session);
    let ends = session.positions();
    assert!((ends[(0, 0)] + 1.0).abs() < 1e-12);
    assert!((ends[(n_images - 1, 0)] - 1.0).abs() < 1e-12);
    for i in 1..n_images - 1 {
        assert!(
            session.positions()[(i, 1)].abs() < 5e-2,
            "image {i} left the valley"
        );
    }
}

#[test]
fn climbing_spline_session_converges_and_matches_the_projection() {
    let n_images = 7;
    let config = spline_config().with_climbing(Some(
        CiConfig::default()
            .with_trigger_factor(0.5)
            .with_trigger_force(0.0),
    ));
    let mut session = BandSession::new(config.clone(), initial_band(n_images)).expect("session");
    let mut report = session.step(&DoubleWell).expect("first climbing step");
    assert_matches(&config, &session);
    for _ in 1..4000 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = session.step(&DoubleWell).expect("climbing spline step");
    }
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "climbing spline band still running at force {}",
        report.max_force
    );
    let ci = session.climbing_image().expect("climbing image");
    assert!(
        session.positions()[(ci, 0)].abs() < 5e-2,
        "climber off the saddle"
    );
    assert_matches(&config, &session);
}

fn assert_refused(result: Result<BandSession, SaddleError>, needle: &str) {
    match result {
        Err(SaddleError::Invalid(msg)) => {
            assert!(msg.contains(needle), "{msg}");
        }
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("spline tangent was accepted with {needle}"),
    }
}

#[test]
fn spline_tangent_is_refused_with_the_unsupported_sessions() {
    let band = initial_band(5);
    assert_refused(
        BandSession::new(
            spline_config().with_string(Some(StringConfig::default())),
            band.clone(),
        ),
        "simplified string",
    );
    assert_refused(
        BandSession::new(
            spline_config().with_free_end(Some(FreeEndConfig::default())),
            band.clone(),
        ),
        "free-end",
    );
    assert_refused(
        BandSession::new(
            spline_config().with_exp_precon(Some(ExpBandPrecon::default())),
            band,
        ),
        "exponential metric",
    );
}
