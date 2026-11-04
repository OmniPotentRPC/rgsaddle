//! Zoom-NEB on analytic surfaces: the band is placed on a window around
//! the climbing image and keeps climbing to the saddle.

use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2, s};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, SaddleError, SpringKind, ZoomConfig,
};

type Pot = fn(f64, f64, f64) -> (f64, [f64; 3]);

fn well(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    (
        (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z,
        [4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z],
    )
}

/// A barrier at x = 0.3 that rises slowly from the left and falls
/// steeply to the right.
fn cliff(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let w = if x < 0.3 { 0.8 } else { 0.2 };
    let u = (x - 0.3) / w;
    let v = (-u * u).exp();
    (
        v + 2.0 * y * y + 2.0 * z * z,
        [-2.0 * u / w * v, 4.0 * y, 4.0 * z],
    )
}

/// A surface remembering the interior rows it was last given.
struct Recorded(Pot, Mutex<Option<Array2<f64>>>);

impl Recorded {
    fn new(pot: Pot) -> Self {
        Self(pot, Mutex::new(None))
    }

    fn last_band(&self) -> Array2<f64> {
        let mut old = band();
        let interior = self.1.lock().unwrap().clone().unwrap();
        old.slice_mut(s![1..8, ..]).assign(&interior);
        old
    }
}

impl BandSurface for Recorded {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (r, p) in positions.outer_iter().enumerate() {
            let (e, g) = (self.0)(p[0], p[1], p[2]);
            energies[r] = e;
            gradients.row_mut(r).assign(&Array1::from(g.to_vec()));
        }
        *self.1.lock().unwrap() = Some(positions.to_owned());
        Ok(())
    }
}

fn band() -> Array2<f64> {
    Array2::from_shape_fn((9, 3), |(i, c)| {
        let t = i as f64 / 8.0;
        [-1.0 + 2.0 * t, 0.3 * (std::f64::consts::PI * t).sin(), 0.0][c]
    })
}

fn config(zoom: Option<ZoomConfig>) -> BandConfig {
    BandConfig {
        max_move: 0.1,
        min_mode: None,
        zoom,
        ..BandConfig::default()
    }
}

fn eager(cubic: bool) -> ZoomConfig {
    let mut zoom = ZoomConfig::default();
    zoom.after = 10.0;
    zoom.stability = 2;
    zoom.cubic = cubic;
    zoom
}

/// Step until the band zooms; returns the climbing image before it.
fn step_to_zoom(session: &mut BandSession, surface: &Recorded) -> usize {
    let (mut ci, mut steps) = (None, 0);
    while session.zoom_window().is_none() {
        ci = session.step(surface).unwrap().ci_index;
        steps += 1;
        assert!(steps < 500);
    }
    ci.unwrap()
}

fn assert_saddle_at(session: &BandSession, ci: Option<usize>, x: f64) {
    let top = session.positions().row(ci.unwrap()).to_owned();
    assert!(
        (top[0] - x).abs() < 1e-3 && top[1].abs() < 1e-3,
        "saddle at {top}"
    );
}

#[test]
fn zoom_is_off_by_default_and_refuses_bad_settings() {
    assert!(BandConfig::default().zoom.is_none());
    let mut session = BandSession::new(config(None), band()).unwrap();
    for (alpha, offset, stability) in [(1.0, 1, 5), (0.5, 0, 5), (0.5, 1, 0)] {
        let mut zoom = ZoomConfig::default();
        (zoom.alpha, zoom.offset, zoom.stability) = (alpha, offset, stability);
        assert!(session.set_zoom(Some(zoom)).is_err());
    }
    let weighted = BandConfig {
        spring: SpringKind::Weighted { ks: vec![5.0; 8] },
        ..config(Some(ZoomConfig::default()))
    };
    assert!(BandSession::new(weighted, band()).is_err());
    session.set_zoom(Some(ZoomConfig::default())).unwrap();
    assert_eq!(session.zoom(), Some(ZoomConfig::default()));
    session.set_zoom(None).unwrap();
    let report = session.run(&Recorded::new(well), 2000).unwrap();
    assert_eq!(report.status, BandStatus::Converged);
    assert_eq!(session.zoom_window(), None);
    assert_eq!(session.positions().row(0), band().row(0));
}

#[test]
fn the_band_is_placed_inside_the_window_and_keeps_climbing() {
    for cubic in [true, false] {
        let surface = Recorded::new(well);
        let mut session = BandSession::new(config(Some(eager(cubic))), band()).unwrap();
        let ci = step_to_zoom(&mut session, &surface);
        let old = surface.last_band();
        let (lo, hi) = session.zoom_window().unwrap();
        assert!(lo < ci && ci < hi, "window ({lo}, {hi}) around {ci}");
        let placed = session.positions().to_owned();
        assert_eq!(placed.row(0), old.row(lo));
        assert_eq!(placed.row(8), old.row(hi));
        assert!((1..9).all(|i| placed[(i, 0)] > placed[(i - 1, 0)]));
        assert!(placed[(0, 0)] < old[(ci, 0)] && old[(ci, 0)] < placed[(8, 0)]);
        // The band is symmetric, so the middle image sits on the climber.
        let middle = &placed.row(4) - &old.row(ci);
        assert!(middle.iter().all(|d| d.abs() < 1e-9));

        let report = session.step(&surface).unwrap();
        // The placed interior and the moved one; no endpoint rows.
        assert_eq!(report.surface_rows, 2 * 7);
        assert!(report.ci_index.is_some(), "the climbing image stays armed");
        let report = session.run(&surface, 2000).unwrap();
        assert_eq!(report.status, BandStatus::Converged);
        assert_saddle_at(&session, report.ci_index, 0.0);
        assert_eq!(session.positions().row(0), old.row(lo));
    }
}

#[test]
fn a_one_sided_energy_window_keeps_the_climber_inside() {
    let surface = Recorded::new(cliff);
    let mut session = BandSession::new(config(Some(eager(true))), band()).unwrap();
    let ci = step_to_zoom(&mut session, &surface);
    let old = surface.last_band();
    let energy = |i: usize| cliff(old[(i, 0)], old[(i, 1)], old[(i, 2)]).0;
    let e_ref = energy(0).min(energy(8));
    let threshold = e_ref + 0.5 * (energy(ci) - e_ref);
    // eOn's window would end on the climbing image.
    assert!(energy(ci + 1) <= threshold && energy(ci - 1) > threshold);
    let (lo, hi) = session.zoom_window().unwrap();
    assert!(lo < ci && ci < hi, "window ({lo}, {hi}) around {ci}");
    assert!(session.positions()[(8, 0)] > 0.3);

    let report = session.run(&surface, 5000).unwrap();
    assert_eq!(report.status, BandStatus::Converged);
    assert_saddle_at(&session, report.ci_index, 0.3);
}

#[test]
fn the_zoom_decides_before_the_min_mode_refinement_starts() {
    let refined = |zoom| BandConfig {
        max_move: 0.1,
        zoom,
        ..BandConfig::default()
    };
    assert!(refined(None).min_mode.is_some());
    let mut zoom = ZoomConfig::default();
    zoom.after = 10.0;
    let surface = Recorded::new(well);

    let mut session = BandSession::new(refined(None), band()).unwrap();
    let mut first = 1;
    while session.step(&surface).unwrap().curvature.is_none() {
        first += 1;
        assert!(first < 50);
    }
    assert!(first <= zoom.stability, "refinement from step {first}");

    let mut session = BandSession::new(refined(Some(zoom)), band()).unwrap();
    let mut steps = 0;
    while session.zoom_window().is_none() {
        let report = session.step(&surface).unwrap();
        assert!(report.curvature.is_none(), "refinement before the zoom");
        steps += 1;
        assert!(steps < 50);
    }
    assert!(steps >= zoom.stability);
    let (mut refined_steps, mut steps) = (0, 0);
    let report = loop {
        let report = session.step(&surface).unwrap();
        refined_steps += usize::from(report.curvature.is_some());
        steps += 1;
        if report.status == BandStatus::Converged {
            break report;
        }
        assert!(steps < 2000);
    };
    assert!(refined_steps > 0, "the refinement runs after the zoom");
    assert_saddle_at(&session, report.ci_index, 0.0);
}
