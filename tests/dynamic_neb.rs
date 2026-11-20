//! Dynamic NEB on a two-dimensional double well embedded in three
//! coordinates. The analytic gradient is the surface. A plain band and
//! a dynamic band at the same tolerance are compared by the force rows
//! each one asks for and by the path they stop on.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, DynamicNeb, SaddleError, StringConfig,
};

/// `V = (x^2 - 1)^2 + 2 y^2 + 2 z^2`. Minima at `(±1, 0, 0)`.
struct Well {
    rows: AtomicUsize,
}

impl BandSurface for Well {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
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

fn bowed(n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.4 * (std::f64::consts::PI * t).sin();
    }
    band
}

fn config(dynamic: Option<DynamicNeb>) -> BandConfig {
    BandConfig::default()
        .with_force_tol(0.05)
        .with_max_move(0.1)
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_dynamic(dynamic)
}

fn relax(dynamic: Option<DynamicNeb>, band: &Array2<f64>) -> (Array2<f64>, usize) {
    let surface = Well {
        rows: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(config(dynamic), band.clone()).expect("band");
    let mut report = session.step(&surface).expect("step");
    for _ in 0..800 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = session.step(&surface).expect("step");
    }
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "max force {}",
        report.max_force
    );
    let rows = surface.rows.load(Ordering::Relaxed);
    (session.positions().to_owned(), rows)
}

#[test]
fn dynamic_neb_skips_force_rows_and_matches_the_plain_path() {
    let band = bowed(9);
    let (plain, plain_rows) = relax(None, &band);
    let spec = DynamicNeb::default().with_fmax(0.05);
    let (dynamic, dynamic_rows) = relax(Some(spec), &band);
    println!("plain force rows {plain_rows}");
    println!("dynamic force rows {dynamic_rows}");
    assert!(
        dynamic_rows < plain_rows,
        "dynamic {dynamic_rows} plain {plain_rows}"
    );
    let mut gap = 0.0;
    for (left, right) in plain.iter().zip(dynamic.iter()) {
        let diff = (left - right).abs();
        if diff > gap {
            gap = diff;
        }
    }
    assert!(gap <= 0.05, "path gap {gap} above the band tolerance");
}

#[test]
fn scaled_dynamic_neb_converges_with_fewer_rows() {
    let band = bowed(9);
    let (_plain, plain_rows) = relax(None, &band);
    let spec = DynamicNeb::default().with_fmax(0.05).with_scale_fmax(6.0);
    let (_scaled, scaled_rows) = relax(Some(spec), &band);
    println!("plain force rows {plain_rows}");
    println!("scaled force rows {scaled_rows}");
    assert!(
        scaled_rows < plain_rows,
        "scaled {scaled_rows} plain {plain_rows}"
    );
}

#[test]
fn dynamic_default_converges_at_the_stock_tolerances() {
    let band = bowed(9);
    let surface = Well {
        rows: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(
        BandConfig::default()
            .with_max_move(0.1)
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_dynamic(Some(DynamicNeb::default())),
        band,
    )
    .expect("band");
    let mut report = session.step(&surface).expect("step");
    for _ in 0..800 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = session.step(&surface).expect("step");
    }
    println!(
        "default dynamic rows {} max force {}",
        surface.rows.load(Ordering::Relaxed),
        report.max_force
    );
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "max force {}",
        report.max_force
    );
}

#[test]
fn dynamic_neb_and_the_simplified_string_method_exclude_each_other() {
    let band = bowed(5);
    let spec = DynamicNeb::default();
    let string = StringConfig::default();
    assert!(matches!(
        BandSession::new(config(Some(spec)).with_string(Some(string)), band.clone()),
        Err(SaddleError::Invalid(_))
    ));
    let mut stringy = BandSession::new(config(None).with_string(Some(string)), band.clone())
        .expect("string band");
    assert!(matches!(
        stringy.set_dynamic(Some(spec)),
        Err(SaddleError::Invalid(_))
    ));
    assert!(stringy.string().is_some());
    let mut dynamic = BandSession::new(config(Some(spec)), band).expect("dynamic band");
    assert!(matches!(
        dynamic.set_string(Some(string)),
        Err(SaddleError::Invalid(_))
    ));
    assert!(dynamic.string().is_none());
}
