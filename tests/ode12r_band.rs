//! ODE12r on the band: the double well converges to its saddle, a
//! rejected trial costs one band evaluation and moves nothing, and the
//! per-image Exp metric relaxes a stiff diatomic band in fewer trials
//! than the Euclidean integration.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, ExpBandPrecon, SaddleError, SpringKind,
};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at (+-1, 0, 0), saddle at
/// the origin with barrier 1.
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

fn relax<S: BandSurface>(
    session: &mut BandSession,
    surface: &S,
    max_steps: usize,
) -> (BandStatus, usize, usize) {
    let mut rows = 0;
    let mut steps = 0;
    let mut status = BandStatus::Running;
    for _ in 0..max_steps {
        let report = session.step(surface).unwrap();
        rows += report.surface_rows;
        steps += 1;
        status = report.status;
        if status == BandStatus::Converged {
            break;
        }
    }
    (status, steps, rows)
}

#[test]
fn ode12r_converges_the_double_well_band_to_the_saddle() {
    let n_images = 9;
    // The plain climbing band: the min-mode refinement of the climbing
    // image has its own evaluation count and is off here.
    let config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        method: rgmin::Method::Ode12r,
        min_mode: None,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial_band(n_images)).unwrap();
    let (status, steps, rows) = relax(&mut session, &DoubleWell, 3000);
    assert_eq!(
        status,
        BandStatus::Converged,
        "not converged in {steps} steps"
    );
    // The first step evaluates the whole band at the start and then its
    // trial; every later step is one interior evaluation, accepted or not.
    assert_eq!(
        rows,
        (n_images - 2) * (steps + 1) + 2,
        "rows {rows} for {steps} steps"
    );
    let ci = session.climbing_image().expect("climbing image armed");
    let saddle = session.positions().row(ci).to_owned();
    assert!(saddle[0].abs() < 2e-2, "saddle x = {}", saddle[0]);
    assert!(saddle[1].abs() < 2e-2, "saddle y = {}", saddle[1]);
    for i in 1..n_images - 1 {
        assert!(
            session.positions()[(i, 1)].abs() < 5e-2,
            "image {i} off the path"
        );
    }
    let ends = session.positions();
    assert_eq!(
        ends.row(0).to_owned(),
        initial_band(n_images).row(0).to_owned()
    );
}

/// `E = 1/2 y^2` at a symmetric peak: the band force on the middle
/// image is `(0, -y, 0)`, and an Euler step of length 1 lands on
/// `y = 0`, where the residual vanishes: accepted.
struct Valley;

impl BandSurface for Valley {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let y = row[1];
            energies[i] = 0.5 * y * y;
            gradients[(i, 0)] = 0.0;
            gradients[(i, 1)] = y;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn a_rejected_trial_leaves_the_band_in_place_and_costs_one_evaluation() {
    // `E = 1/2 k y^2` with k = 40 through a surface scale: the opening
    // step of length 1 overshoots y = 0.5 to y = -19.5, the residual
    // grows 39-fold, ODE12r rejects, and the band stays put.
    struct Stiff;
    impl BandSurface for Stiff {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                let y = row[1];
                energies[i] = 20.0 * y * y;
                gradients[(i, 0)] = 0.0;
                gradients[(i, 1)] = 40.0 * y;
                gradients[(i, 2)] = 0.0;
            }
            Ok(())
        }
    }
    let band = ndarray::array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
    let config = BandConfig {
        spring: SpringKind::Uniform { k: 0.0 },
        climbing: None,
        max_move: 0.0,
        method: rgmin::Method::Ode12r,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, band.clone()).unwrap();
    let report = session.step(&Stiff).unwrap();
    assert_eq!(
        report.surface_rows, 4,
        "first step: the whole band, then the trial"
    );
    assert_eq!(
        session.positions(),
        band.view(),
        "a rejected trial moved the band"
    );
    let report = session.step(&Stiff).unwrap();
    assert_eq!(
        report.surface_rows, 1,
        "the retry is one trial from the cache"
    );
    let (status, _, _) = relax(&mut session, &Stiff, 200);
    assert_eq!(status, BandStatus::Converged);
    assert!(session.positions()[(1, 1)].abs() < 1e-3);
}

#[test]
fn the_valley_step_of_length_one_lands_on_the_path() {
    let band = ndarray::array![[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]];
    let config = BandConfig {
        spring: SpringKind::Uniform { k: 0.0 },
        climbing: None,
        max_move: 0.0,
        method: rgmin::Method::Ode12r,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, band).unwrap();
    let report = session.step(&Valley).unwrap();
    assert!(
        session.positions()[(1, 1)].abs() < 1e-12,
        "{:?}",
        session.positions()
    );
    assert!((session.positions()[(1, 0)] - 1.0).abs() < 1e-12);
    assert_eq!(report.status, BandStatus::Converged);
}

/// Two atoms per image: a stiff bond of rest length 1 (k = 200) and a
/// double well in the centroid's x, a soft well in the centroid's y
/// and z. Minima at centroid (+-1, 0, 0); the saddle is at the origin.
struct StiffDiatomic;

impl BandSurface for StiffDiatomic {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let a = [row[0], row[1], row[2]];
            let b = [row[3], row[4], row[5]];
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let c = [
                0.5 * (a[0] + b[0]),
                0.5 * (a[1] + b[1]),
                0.5 * (a[2] + b[2]),
            ];
            let bond = 100.0 * (r - 1.0).powi(2);
            let well = (c[0] * c[0] - 1.0).powi(2) + 2.0 * c[1] * c[1] + 2.0 * c[2] * c[2];
            energies[i] = bond + well;
            let dbond = 200.0 * (r - 1.0) / r;
            let dwell = [4.0 * c[0] * (c[0] * c[0] - 1.0), 4.0 * c[1], 4.0 * c[2]];
            for k in 0..3 {
                gradients[(i, k)] = -dbond * d[k] + 0.5 * dwell[k];
                gradients[(i, 3 + k)] = dbond * d[k] + 0.5 * dwell[k];
            }
        }
        Ok(())
    }
}

fn diatomic_band(n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 6));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        let cx = -1.0 + 2.0 * t;
        let cy = 0.3 * (std::f64::consts::PI * t).sin();
        // Bond along z, slightly stretched on the interior images.
        let half = 0.5 + 0.05 * (std::f64::consts::PI * t).sin();
        band[(i, 0)] = cx;
        band[(i, 1)] = cy;
        band[(i, 2)] = -half;
        band[(i, 3)] = cx;
        band[(i, 4)] = cy;
        band[(i, 5)] = half;
    }
    band
}

fn diatomic_trials(precon: Option<ExpBandPrecon>) -> (usize, usize, Array1<f64>) {
    let n_images = 9;
    // The well acts on the centroid, a pure translation of the molecule,
    // so the band keeps the mean force: dropping it would leave only the
    // bond, which the plain integration solves in three trials.
    let config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        method: rgmin::Method::Ode12r,
        remove_translation: false,
        min_mode: None,
        exp_precon: precon,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, diatomic_band(n_images)).unwrap();
    let (status, steps, rows) = relax(&mut session, &StiffDiatomic, 4000);
    assert_eq!(
        status,
        BandStatus::Converged,
        "not converged in {steps} steps"
    );
    let ci = session.climbing_image().expect("climbing image armed");
    (steps, rows, session.positions().row(ci).to_owned())
}

#[test]
fn the_exp_metric_relaxes_the_stiff_diatomic_band_in_fewer_trials() {
    let (plain_steps, plain_rows, plain_saddle) = diatomic_trials(None);
    let (exp_steps, exp_rows, exp_saddle) = diatomic_trials(Some(ExpBandPrecon {
        mu: Some(200.0),
        ..ExpBandPrecon::default()
    }));
    eprintln!("plain {plain_steps} trials / {plain_rows} rows, exp {exp_steps} / {exp_rows}");
    for saddle in [&plain_saddle, &exp_saddle] {
        let cx = 0.5 * (saddle[0] + saddle[3]);
        let cy = 0.5 * (saddle[1] + saddle[4]);
        let r = ((saddle[3] - saddle[0]).powi(2)
            + (saddle[4] - saddle[1]).powi(2)
            + (saddle[5] - saddle[2]).powi(2))
        .sqrt();
        assert!(cx.abs() < 2e-2, "saddle centroid x = {cx}");
        assert!(cy.abs() < 2e-2, "saddle centroid y = {cy}");
        assert!((r - 1.0).abs() < 1e-2, "saddle bond {r}");
    }
    assert!(
        exp_steps < plain_steps,
        "Exp metric took {exp_steps} trials against {plain_steps} without it"
    );
}

#[test]
fn the_exp_metric_is_refused_where_it_has_no_pairs() {
    let config = BandConfig {
        method: rgmin::Method::Ode12r,
        exp_precon: Some(ExpBandPrecon::default()),
        ..BandConfig::default()
    };
    let err = match BandSession::new(config, initial_band(5)) {
        Err(err) => err,
        Ok(_) => panic!("a single-atom band accepted the exponential metric"),
    };
    assert!(matches!(err, SaddleError::Invalid(_)), "{err:?}");
}
