//! Band relaxation on an analytic double well: interiors must fall
//! onto the reaction path, the climbing image must find the saddle,
//! and the endpoints must never move.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at (+-1, 0, 0), saddle at
/// the origin with barrier 1.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut ndarray::Array2<f64>,
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

#[test]
fn band_converges_to_the_double_well_saddle() {
    let n_images = 9;
    let config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial_band(n_images)).unwrap();
    let mut report = session.step(&DoubleWell).unwrap();
    for k in 0..3000 {
        if report.status == BandStatus::Converged {
            break;
        }
        if k % 200 == 0 {
            eprintln!(
                "step {k}: max_force={} ci={:?}",
                report.max_force, report.ci_index
            );
        }
        report = session.step(&DoubleWell).unwrap();
    }
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "max_force={}",
        report.max_force
    );

    let pos = session.positions();
    // Endpoints pinned exactly.
    assert_eq!(pos[(0, 0)], -1.0);
    assert_eq!(pos[(n_images - 1, 0)], 1.0);
    assert_eq!(pos[(0, 1)], 0.0);
    // Interiors on the reaction path.
    for i in 1..n_images - 1 {
        assert!(pos[(i, 1)].abs() < 5e-2, "image {i} y={}", pos[(i, 1)]);
        assert!(pos[(i, 2)].abs() < 5e-2, "image {i} z={}", pos[(i, 2)]);
    }
    // The climbing image sits at the barrier top.
    let ci = report.ci_index.expect("climbing image armed");
    assert!(pos[(ci, 0)].abs() < 0.15, "ci x={}", pos[(ci, 0)]);
}

/// Fails every evaluation once `fail_after` calls have gone through.
struct FailingWell {
    calls: AtomicUsize,
    fail_after: usize,
}

impl BandSurface for FailingWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut ndarray::Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        if n >= self.fail_after {
            return Err(SaddleError::Surface("scf did not converge".into()));
        }
        DoubleWell.eval(positions, energies, gradients)
    }
}

#[test]
fn surface_error_inside_a_step_is_returned_and_the_band_stays_put() {
    let n_images = 7;
    let mut session = BandSession::new(BandConfig::default(), initial_band(n_images)).unwrap();
    let surface = FailingWell {
        calls: AtomicUsize::new(0),
        fail_after: 1,
    };
    // The first step sees at least one good evaluation followed by a
    // failing one inside the solver step.
    let before = session.positions().to_owned();
    let err = session.step(&surface).unwrap_err();
    assert!(
        matches!(err, SaddleError::Surface(ref m) if m.contains("scf did not converge")),
        "{err}"
    );
    assert_eq!(session.positions(), before.view());
    assert!(surface.calls.load(Ordering::Relaxed) >= 2);

    // A surface that fails from the very first evaluation.
    let surface = FailingWell {
        calls: AtomicUsize::new(0),
        fail_after: 0,
    };
    let err = session.step(&surface).unwrap_err();
    assert!(matches!(err, SaddleError::Surface(_)), "{err}");
    assert_eq!(session.positions(), before.view());
}

/// Counts evaluations and the rows they carry.
struct Counting {
    calls: AtomicUsize,
    rows: AtomicUsize,
}

impl Counting {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            rows: AtomicUsize::new(0),
        }
    }
    fn take(&self) -> (usize, usize) {
        (
            self.calls.swap(0, Ordering::Relaxed),
            self.rows.swap(0, Ordering::Relaxed),
        )
    }
}

impl BandSurface for Counting {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut ndarray::Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        DoubleWell.eval(positions, energies, gradients)
    }
}

#[test]
fn endpoints_are_evaluated_once_and_each_step_costs_one_interior_eval() {
    let n_images = 7;
    let interior = n_images - 2;
    let mut session = BandSession::new(BandConfig::default(), initial_band(n_images)).unwrap();
    let surface = Counting::new();

    // First step: exactly one whole-band evaluation (the endpoint
    // energies come from it), every other evaluation interior-only.
    session.step(&surface).unwrap();
    let (calls, rows) = surface.take();
    assert!(calls >= 1);
    assert_eq!(rows, n_images + (calls - 1) * interior, "calls={calls}");

    // Every later step: one evaluation, interior rows only.
    for k in 0..5 {
        session.step(&surface).unwrap();
        let (calls, rows) = surface.take();
        assert_eq!(calls, 1, "step {k}");
        assert_eq!(rows, interior, "step {k}");
    }

    // reset marks a surface change: the endpoint energies are taken
    // again from one whole-band evaluation.
    session.reset();
    session.step(&surface).unwrap();
    let (calls, rows) = surface.take();
    assert!(calls >= 1);
    assert_eq!(rows, n_images + (calls - 1) * interior, "calls={calls}");

    // set_positions may change the endpoint rows: one whole-band
    // evaluation again.
    let current = session.positions().to_owned();
    session.set_positions(current).unwrap();
    session.step(&surface).unwrap();
    let (calls, rows) = surface.take();
    assert!(calls >= 1);
    assert_eq!(rows, n_images + (calls - 1) * interior, "calls={calls}");
}

#[test]
fn reset_clears_history_and_stepping_resumes() {
    let config = BandConfig::default();
    let mut session = BandSession::new(config, initial_band(7)).unwrap();
    let first = session.step(&DoubleWell).unwrap();
    assert!(first.max_force.is_finite());
    session.reset();
    let after = session.step(&DoubleWell).unwrap();
    assert!(after.max_force.is_finite());
}
