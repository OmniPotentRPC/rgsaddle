//! The climbing image handed off the band: one evaluation of that
//! image per step, convergence on its force, and the band back in
//! charge after a restart.

use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at (+-1, 0, 0), saddle at
/// the origin with curvature -4 along x. Records the images asked
/// for alone.
#[derive(Default)]
struct DoubleWell {
    alone: Mutex<Vec<usize>>,
    rows: Mutex<usize>,
}

fn well(row: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (x, y, z) = (row[0], row[1], row[2]);
    (
        (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z,
        Array1::from(vec![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]),
    )
}

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        *self.rows.lock().unwrap() += positions.nrows();
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = well(row);
            energies[i] = e;
            gradients.row_mut(i).assign(&g);
        }
        Ok(())
    }

    fn eval_image(
        &self,
        image: usize,
        position: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.alone.lock().unwrap().push(image);
        *self.rows.lock().unwrap() += 1;
        Ok(well(position))
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

fn config(handoff: bool) -> BandConfig {
    let mut config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        ..BandConfig::default()
    };
    if !handoff {
        config.climb_refine = None;
    }
    config
}

fn converge(session: &mut BandSession, surface: &DoubleWell) -> rgsaddle::BandReport {
    let mut report = session.step(surface).unwrap();
    for _ in 0..5000 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = session.step(surface).unwrap();
    }
    assert_eq!(report.status, BandStatus::Converged, "{report:?}");
    report
}

#[test]
fn handed_off_image_converges_on_the_saddle_one_evaluation_per_step() {
    let n_images = 9;
    let band = initial_band(n_images);
    let surface = DoubleWell::default();
    let mut session = BandSession::new(config(true), band.clone()).unwrap();
    let mut report = session.step(&surface).unwrap();
    while !session.climbing_refined() {
        assert_eq!(report.status, BandStatus::Running);
        report = session.step(&surface).unwrap();
    }
    let ci = report.ci_index.expect("climbing image armed");
    let frozen = session.positions().to_owned();
    while report.status == BandStatus::Running {
        report = session.step(&surface).unwrap();
        assert_eq!(report.surface_rows, 1);
        assert_eq!(report.ci_index, Some(ci));
    }
    assert!(surface.alone.lock().unwrap().iter().all(|&i| i == ci));

    let pos = session.positions();
    for i in (0..n_images).filter(|&i| i != ci) {
        assert_eq!(
            pos.row(i),
            frozen.row(i),
            "image {i} moved after the hand-off"
        );
    }
    assert_eq!(pos.row(0), band.row(0));
    assert_eq!(pos.row(n_images - 1), band.row(n_images - 1));
    let (_, g) = well(pos.row(ci));
    assert!(g.dot(&g).sqrt() <= 1e-3, "|g| {}", g.dot(&g).sqrt());
    assert!(pos[(ci, 0)].abs() < 1e-3, "ci x={}", pos[(ci, 0)]);
    let curvature = session.climbing_curvature().expect("followed mode");
    assert!((curvature + 4.0).abs() < 0.5, "curvature {curvature}");

    let eval = session.evaluation().expect("evaluation at the saddle");
    assert_eq!(eval.gradients.row(ci), g);
    assert!(report.max_force <= 1e-3);
}

#[test]
fn the_hand_off_spends_fewer_evaluations_than_the_band() {
    let banded = DoubleWell::default();
    let mut session = BandSession::new(config(false), initial_band(9)).unwrap();
    converge(&mut session, &banded);
    assert!(!session.climbing_refined());
    assert!(banded.alone.lock().unwrap().is_empty());

    let handed = DoubleWell::default();
    let mut session = BandSession::new(config(true), initial_band(9)).unwrap();
    converge(&mut session, &handed);
    let (band_rows, handed_rows) = (*banded.rows.lock().unwrap(), *handed.rows.lock().unwrap());
    assert!(handed_rows < band_rows, "{handed_rows} >= {band_rows}");
}

#[test]
fn restart_returns_the_climbing_image_to_the_band() {
    let surface = DoubleWell::default();
    let mut session = BandSession::new(config(true), initial_band(9)).unwrap();
    while !session.climbing_refined() {
        session.step(&surface).unwrap();
    }
    session.step(&surface).unwrap();
    session.restart();
    assert!(!session.climbing_refined());
    let report = session.step(&surface).unwrap();
    assert_eq!(report.surface_rows, 7);
}

#[test]
fn a_band_without_climbing_never_hands_off() {
    let surface = DoubleWell::default();
    let mut session = BandSession::new(
        BandConfig {
            climbing: None,
            ..config(true)
        },
        initial_band(9),
    )
    .unwrap();
    converge(&mut session, &surface);
    assert!(!session.climbing_refined());
    assert!(surface.alone.lock().unwrap().is_empty());
}
