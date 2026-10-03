use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{BandConfig, BandRtr, BandSurface, RtrConfig, SaddleError};
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailingCurvatureProbe(AtomicUsize);

impl BandSurface for FailingCurvatureProbe {
    fn eval(&self, positions: ArrayView2<f64>, energies: &mut Array1<f64>, gradients: &mut Array2<f64>) -> Result<(), SaddleError> {
        if self.0.fetch_add(1, Ordering::Relaxed) == 1 {
            return Err(SaddleError::Surface("curvature probe failed".into()));
        }
        for (i, row) in positions.outer_iter().enumerate() {
            energies[i] = (row[0] * row[0] - 1.0).powi(2) + 2.0 * row[1] * row[1] + 2.0 * row[2] * row[2];
            gradients[(i, 0)] = 4.0 * row[0] * (row[0] * row[0] - 1.0);
            gradients[(i, 1)] = 4.0 * row[1];
            gradients[(i, 2)] = 4.0 * row[2];
        }
        Ok(())
    }
}

#[test]
fn failed_curvature_probe_preserves_positions_and_returns_the_surface_error() {
    let mut x = array![[-1.0, 0.0, 0.0], [0.0, 0.3, 0.0], [1.0, 0.0, 0.0]];
    let initial = x.clone();
    let surface = FailingCurvatureProbe(AtomicUsize::new(0));
    let mut rtr = BandRtr::new(RtrConfig::default(), false);
    let error = rtr.step(&BandConfig::default(), &surface, &mut x).unwrap_err();
    assert!(matches!(error, SaddleError::Surface(ref message) if message == "curvature probe failed"));
    assert_eq!(x, initial);
    assert!(surface.0.load(Ordering::Relaxed) >= 2);
}


struct TransverseWell {
    calls: AtomicUsize,
    fail_at: Option<usize>,
}

impl BandSurface for TransverseWell {
    fn eval(&self, positions: ArrayView2<f64>, energies: &mut Array1<f64>, gradients: &mut Array2<f64>) -> Result<(), SaddleError> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if self.fail_at == Some(call) {
            return Err(SaddleError::Surface("final trial evaluation failed".into()));
        }
        for (i, row) in positions.outer_iter().enumerate() {
            energies[i] = 0.5 * (row[1] * row[1] + row[2] * row[2]);
            gradients[(i, 0)] = 0.0;
            gradients[(i, 1)] = row[1];
            gradients[(i, 2)] = row[2];
        }
        Ok(())
    }
}

fn uneven_band() -> Array2<f64> {
    array![[-1.0, 0.0, 0.0], [-0.6, 0.2, 0.0], [0.1, 0.3, 0.0], [0.6, 0.2, 0.0], [1.0, 0.0, 0.0]]
}

#[test]
fn final_trial_failure_cannot_publish_an_accepted_position() {
    let config = BandConfig::default();
    let initial = uneven_band();
    let surface = TransverseWell { calls: AtomicUsize::new(0), fail_at: None };
    let mut control = BandRtr::new(RtrConfig::default(), false);
    let mut reference = initial.clone();
    let report = control.step(&config, &surface, &mut reference).unwrap();
    assert!(report.accepted, "fixture must exercise accepted position publication");
    let final_call = surface.calls.load(Ordering::Relaxed);
    let failing = TransverseWell { calls: AtomicUsize::new(0), fail_at: Some(final_call) };
    let mut session = BandRtr::new(RtrConfig::default(), false);
    let mut positions = initial.clone();
    let error = session.step(&config, &failing, &mut positions).unwrap_err();
    assert!(matches!(error, SaddleError::Surface(ref message) if message == "final trial evaluation failed"));
    assert_eq!(positions, initial);
}

#[test]
fn trust_report_uses_the_entire_reparameterized_displacement() {
    use rgsaddle::band_forces;
    let config = BandConfig::default();
    let initial = uneven_band();
    let surface = TransverseWell { calls: AtomicUsize::new(0), fail_at: None };
    let options = RtrConfig { radius_max: 0.5, ..RtrConfig::default() };
    let mut session = BandRtr::new(options, false);
    let radius = session.radius.radius;
    let mut positions = initial.clone();
    let report = session.step(&config, &surface, &mut positions).unwrap();
    assert!(report.accepted, "fixture must exercise the accepted displacement");
    let displacement: Array1<f64> = positions.slice(ndarray::s![1..4, ..]).iter()
        .zip(initial.slice(ndarray::s![1..4, ..]).iter()).map(|(a,b)| a-b).collect();
    let length = displacement.dot(&displacement).sqrt();
    assert!(length <= radius * (1.0 + 1e-12));
    let before = band_forces(&config, &surface, initial.view(), None).unwrap();
    let after = band_forces(&config, &surface, positions.view(), None).unwrap();
    let measured = 0.5 * (&before.force + &after.force).dot(&displacement);
    assert!((report.actual_decrease - measured).abs() < 1e-12 * (1.0 + measured.abs()));
    let mut plus = initial.clone();
    let mut minus = initial.clone();
    for row in 1..4 {
        let direction = displacement.slice(ndarray::s![3*(row-1)..3*row]);
        plus.row_mut(row).scaled_add(options.fd_step / length, &direction);
        minus.row_mut(row).scaled_add(-options.fd_step / length, &direction);
    }
    let fp = band_forces(&config, &surface, plus.view(), None).unwrap().force;
    let fm = band_forces(&config, &surface, minus.view(), None).unwrap().force;
    let action = (fm-fp) * (length / (2.0 * options.fd_step));
    let predicted = before.force.dot(&displacement) - 0.5 * displacement.dot(&action);
    assert!((report.model_decrease - predicted).abs() < 1e-12 * (1.0 + predicted.abs()));
}

#[test]
fn periodic_equal_arc_spacing_uses_short_chords_and_preserves_endpoints() {
    let cell = rgsaddle::Cell([[10.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]]);
    let mut positions = array![[9.6,0.0,0.0],[9.9,0.0,0.0],[0.5,0.0,0.0],[1.2,0.0,0.0],[1.6,0.0,0.0]];
    let original = positions.clone();
    rgsaddle::rtr::reparametrize_equal_arc_in_cell(&mut positions, None, &cell);
    assert_eq!(positions.row(0), original.row(0));
    assert_eq!(positions.row(4), original.row(4));
    for row in 1..5 {
        let mut difference = &positions.row(row) - &positions.row(row-1);
        cell.minimum_image(&mut difference);
        assert!((difference[0] - 0.5).abs() < 1e-12);
        assert_eq!(difference[1], 0.0);
        assert_eq!(difference[2], 0.0);
    }
}
