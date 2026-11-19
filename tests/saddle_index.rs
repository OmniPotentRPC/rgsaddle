//! Saddle-index certification and escape on a surface with a known
//! index-2 point.
//!
//! V = (x^4 - 2 x^2) / 4 + (y^4 - y^2) / 4 + z^2 / 2 has minima at
//! (+-1, +-sqrt(1/2), 0), index-1 saddles at (0, +-sqrt(1/2), 0) (V =
//! -1/16) and (+-1, 0, 0) (V = -1/4), and an index-2 saddle at the
//! origin with curvatures -1 and -1/2. A search that starts on a line
//! the inversion maps onto itself never leaves it.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, MinModeConfig, MinModeSession, MinModeStatus,
    OcinebConfig, OcinebSession, OcinebStatus, PointSurface, SaddleError, SaddleEscape,
    SaddleIndexCheck, saddle_index,
};

const SIDE: f64 = std::f64::consts::FRAC_1_SQRT_2;

struct Wells;

fn wells(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b, c) = (x[0], x[1], x[2]);
    let e = (a.powi(4) - 2.0 * a * a) / 4.0 + (b.powi(4) - b * b) / 4.0 + c * c / 2.0;
    (e, array![a.powi(3) - a, b.powi(3) - 0.5 * b, c])
}

impl PointSurface for Wells {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(wells(x))
    }
}

impl BandSurface for Wells {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = wells(row);
            energies[i] = e;
            gradients.row_mut(i).assign(&g);
        }
        Ok(())
    }
}

fn escape() -> SaddleIndexCheck {
    SaddleIndexCheck::default().with_escape(Some(SaddleEscape::default()))
}

fn min_mode(check: Option<SaddleIndexCheck>) -> (MinModeSession, Vec<rgsaddle::MinModeReport>) {
    let config = MinModeConfig::default()
        .with_force_tol(1e-6)
        .with_max_move(0.1)
        .with_saddle_check(check);
    let mut session =
        MinModeSession::new(config, array![0.2, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
    let mut reports = Vec::new();
    for _ in 0..5000 {
        let report = session.step(&Wells).unwrap();
        let done = report.status == MinModeStatus::Converged;
        reports.push(report);
        if done {
            break;
        }
    }
    (session, reports)
}

#[test]
fn the_min_mode_search_on_the_axis_stops_on_the_index_two_top() {
    let (session, reports) = min_mode(Some(SaddleIndexCheck::default()));
    let last = reports.last().unwrap();
    assert_eq!(last.status, MinModeStatus::Converged);
    let x = session.position();
    assert!(x.iter().all(|v| v.abs() < 1e-5), "{x}");
    let index = last.saddle_index.as_ref().expect("certified");
    assert_eq!(index.index, 2, "{index:?}");
    assert!(index.converged);
    assert!((index.curvatures[0] + 1.0).abs() < 1e-2, "{index:?}");
    assert!((index.curvatures[1] + 0.5).abs() < 1e-2, "{index:?}");
    assert!(index.curvatures[2] > 0.0, "{index:?}");
    assert!(index.evaluations > 0);
    assert_eq!(session.saddle_index(), Some(index));
    assert_eq!(session.saddle_evaluations(), index.evaluations);
    assert_eq!(session.saddle_escapes(), 0);
    assert!(
        reports[..reports.len() - 1]
            .iter()
            .all(|r| r.saddle_index.is_none())
    );
}

#[test]
fn the_min_mode_escape_reaches_the_index_one_saddle_beside_the_top() {
    let (session, reports) = min_mode(Some(escape()));
    let last = reports.last().unwrap();
    assert_eq!(last.status, MinModeStatus::Converged);
    let index = last.saddle_index.as_ref().expect("certified");
    assert_eq!(index.index, 1, "{index:?}");
    let x = session.position();
    assert!(x[0].abs() < 1e-4 && (x[1].abs() - SIDE).abs() < 1e-4, "{x}");
    assert!((wells(x).0 + 1.0 / 16.0).abs() < 1e-8);
    assert_eq!(session.saddle_escapes(), 1);
    let pushed: Vec<_> = reports
        .iter()
        .filter(|r| r.saddle_index.as_ref().is_some_and(|i| i.index == 2))
        .collect();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0].status, MinModeStatus::Running);
    assert!(session.saddle_evaluations() > pushed[0].saddle_index.as_ref().unwrap().evaluations);
}

#[test]
fn without_a_check_the_min_mode_search_reports_no_index() {
    let (session, reports) = min_mode(None);
    assert_eq!(reports.last().unwrap().status, MinModeStatus::Converged);
    assert!(reports.iter().all(|r| r.saddle_index.is_none()));
    assert!(session.saddle_index().is_none());
    assert_eq!(session.saddle_evaluations(), 0);
}

/// Seven images from (-1, -sqrt(1/2)) to (1, sqrt(1/2)), bowed off the
/// diagonal and mapped onto themselves by the inversion, so the middle
/// image sits on the index-2 top throughout.
fn symmetric_band() -> Array2<f64> {
    let mut band = Array2::zeros((7, 3));
    for i in 0..3 {
        let t = i as f64 / 6.0;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = -SIDE + 2.0 * SIDE * t + 0.3 * (std::f64::consts::PI * 2.0 * t).sin();
    }
    for i in 4..7 {
        for k in 0..3 {
            band[(i, k)] = -band[(6 - i, k)];
        }
    }
    band
}

fn band_config(check: Option<SaddleIndexCheck>) -> BandConfig {
    BandConfig::default()
        .with_force_tol(1e-5)
        .with_max_move(0.1)
        .with_saddle_check(check)
}

fn band(config: BandConfig) -> (BandSession, Vec<rgsaddle::BandReport>) {
    let mut session = BandSession::new(config, symmetric_band()).unwrap();
    let mut reports = Vec::new();
    for _ in 0..5000 {
        let report = session.step(&Wells).unwrap();
        let done = report.status == BandStatus::Converged;
        reports.push(report);
        if done {
            break;
        }
    }
    (session, reports)
}

#[test]
fn the_symmetric_band_climbs_to_the_index_two_top() {
    let (session, reports) = band(band_config(Some(SaddleIndexCheck::default())));
    let last = reports.last().unwrap();
    assert_eq!(last.status, BandStatus::Converged);
    let ci = last.ci_index.expect("climbing image");
    assert_eq!(ci, 3);
    let pos = session.positions();
    assert!(
        pos.row(ci).iter().all(|v| v.abs() < 1e-12),
        "{}",
        pos.row(ci)
    );
    let index = last.saddle_index.as_ref().expect("certified");
    assert_eq!(index.index, 2, "{index:?}");
    assert!((index.curvatures[0] + 1.0).abs() < 1e-2, "{index:?}");
    assert!((index.curvatures[1] + 0.5).abs() < 1e-2, "{index:?}");
    assert_eq!(session.saddle_index(), Some(index));
    assert_eq!(session.saddle_escapes(), 0);
}

#[cfg(feature = "faer")]
#[test]
fn the_band_escape_climbs_to_an_index_one_saddle() {
    let (session, reports) = band(band_config(Some(escape())));
    let last = reports.last().unwrap();
    assert_eq!(last.status, BandStatus::Converged);
    let index = last.saddle_index.as_ref().expect("certified");
    assert_eq!(index.index, 1, "{index:?}");
    let ci = last.ci_index.expect("climbing image");
    let x = session.positions().row(ci).to_owned();
    let (e, g) = wells(x.view());
    assert!(g.iter().all(|v| v.abs() < 1e-5), "{x}");
    let beside = x[0].abs() < 1e-4 && (x[1].abs() - SIDE).abs() < 1e-4;
    let across = (x[0].abs() - 1.0).abs() < 1e-4 && x[1].abs() < 1e-4;
    assert!(beside || across, "{x}, energy {e}");
    assert_eq!(session.saddle_escapes(), 1);
    assert!(reports.iter().any(|r| r.status == BandStatus::Running
        && r.saddle_index.as_ref().is_some_and(|i| i.index == 2)));
}

#[test]
fn the_ocineb_session_reports_the_index_of_its_climbing_image() {
    let config = OcinebConfig::default();
    let band = config.band.clone().with_force_tol(1e-5).with_max_move(0.1);
    let mut session = OcinebSession::new(config.with_band(band), symmetric_band()).unwrap();
    let last = session.run(&Wells, 5000).unwrap();
    assert_eq!(last.status, OcinebStatus::Converged);
    let index = last.saddle_index.as_ref().expect("certified");
    assert_eq!(index.index, 2, "{index:?}");
    assert_eq!(session.saddle_index(), Some(index));
    session.set_saddle_check(None).unwrap();
    assert!(session.saddle_index().is_none());
}

#[test]
fn the_point_check_finds_both_negative_modes_without_a_hint() {
    let check = SaddleIndexCheck::default();
    let top = saddle_index(
        &check,
        &Wells,
        array![0.0, 0.0, 0.0].view(),
        None,
        None,
        &[],
    )
    .unwrap();
    assert_eq!(top.index, 2, "{top:?}");
    let side = saddle_index(
        &check,
        &Wells,
        array![1.0, 0.0, 0.0].view(),
        None,
        Some(array![0.0, 1.0, 0.0].view()),
        &[],
    )
    .unwrap();
    assert_eq!(side.index, 1, "{side:?}");
    let minimum = saddle_index(
        &check,
        &Wells,
        array![1.0, SIDE, 0.0].view(),
        None,
        None,
        &[],
    )
    .unwrap();
    assert_eq!(minimum.index, 0, "{minimum:?}");
}
