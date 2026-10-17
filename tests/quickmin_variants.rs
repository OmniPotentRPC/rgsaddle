//! The three quick-min variants relax a band over a double well to its
//! climbing-image saddle.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, CiConfig, SaddleError};

/// One atom on V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at x = +-1, the
/// saddle at the origin with energy 1.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (r, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            energies[r] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(r, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(r, 1)] = 4.0 * y;
            gradients[(r, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

fn relax(options: rgmin::QuickMinOptions, time_step: f64) -> (usize, f64) {
    let n = 7;
    let mut band = Array2::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    let config = BandConfig {
        quickmin: true,
        quickmin_options: options,
        time_step,
        climbing: Some(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        }),
        force_tol: 1e-4,
        max_move: 0.05,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, band).unwrap();
    let report = session.run(&DoubleWell, 5000).unwrap();
    assert_eq!(report.status, BandStatus::Converged, "{report:?}");
    let ci = report.ci_index.expect("climbing image");
    let x = session.positions()[(ci, 0)];
    (report.iteration, x)
}

#[test]
fn every_quick_min_variant_finds_the_saddle() {
    // The eOn variant keeps its time step fixed; 1 is too large for it
    // here as on the Al and Pt bands, 0.25 converges.
    for (options, time_step) in [
        (rgmin::QuickMinOptions::rgmin(), 1.0),
        (rgmin::QuickMinOptions::eon(), 0.25),
        (rgmin::QuickMinOptions::per_atom(), 1.0),
    ] {
        let (iterations, x) = relax(options.clone(), time_step);
        assert!(x.abs() < 1e-3, "{options:?}: climbing image at x = {x}");
        assert!(iterations > 1, "{options:?}");
    }
}

#[test]
fn a_band_on_another_method_ignores_the_variant() {
    let band = ndarray::array![[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.0, 0.0]];
    let mut session = BandSession::new(BandConfig::default(), band).unwrap();
    assert!(!session.is_quickmin());
    session.set_quickmin_options(rgmin::QuickMinOptions::per_atom());
    session.step(&DoubleWell).unwrap();
}

#[test]
fn the_time_step_is_validated_and_kept() {
    let band = ndarray::array![[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.0, 0.0]];
    let refused = BandSession::new(
        BandConfig {
            time_step: 0.0,
            ..BandConfig::default()
        },
        band.clone(),
    );
    assert!(matches!(refused, Err(SaddleError::Invalid(_))));
    let mut session = BandSession::new(
        BandConfig {
            quickmin: true,
            quickmin_options: rgmin::QuickMinOptions::eon(),
            ..BandConfig::default()
        },
        band,
    )
    .unwrap();
    assert!(session.set_time_step(f64::NAN).is_err());
    session.set_time_step(0.25).unwrap();
    session.step(&DoubleWell).unwrap();
}
