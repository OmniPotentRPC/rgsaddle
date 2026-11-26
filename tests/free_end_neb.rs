//! Free-end adaptive nudged elastic band (FEA-NEB) of Jiayong Zhang,
//! Hongwu Zhang, Hongfei Ye, and Yonggang Zheng, J. Chem. Phys. 145,
//! 094104 (2016), doi:10.1063/1.4962019, and Ting Zhu, Ju Li, Amit
//! Samanta, Hyoung Gyu Kim, and Subra Suresh, Proc. Natl. Acad. Sci.
//! U.S.A. 104, 3031 (2007), doi:10.1073/pnas.0611097104:
//!
//! 1. Narrow barrier on a long tilted 2D path: 7-image FEA-NEB with
//!    level-set constraint E(x_end) = E_target and repeated cut-and-remesh
//!    recovers the same saddle and barrier height as 21-image fixed-end
//!    CI-NEB within tolerance, with 3x fewer images.
//! 2. Both FreeEndSide::Last and FreeEndSide::First move on E(x_end) = E_target
//!    and report target_energy_residual.
//! 3. Nonzero endpoint gradient requirement and fixed-endpoint default
//!    preservation.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, DEFAULT_FORCE_TOL, DynamicNeb,
    FeaNebSession, FreeEndConfig, FreeEndSide, SaddleError, SaddleIndexCheck, SpringKind,
    linear_path,
};

/// Narrow barrier near `x = 0.82` on a long tilted 2D valley `y = 0.25 x`
/// stretching from reactant `(0, 0)` (`E = 0`) to a distant product basin
/// near `(6.0, 1.5)` (`E = -3.0`).
#[derive(Default)]
struct TiltedNarrowBarrier2D {
    rows: AtomicUsize,
}

impl TiltedNarrowBarrier2D {
    fn jet(x: f64, y: f64) -> (f64, [f64; 2]) {
        let x2 = x * x;
        let x4 = x2 * x2;
        let g1 = (-1.5 * x2).exp();
        let dx = x - 0.75;
        let g2 = (-16.0 * dx * dx).exp();
        let sw = x4 / (1.0 + x4);
        let dsw = 4.0 * x * x2 / ((1.0 + x4) * (1.0 + x4));
        let tilt = 0.12 * (x - 6.0).powi(2) - 3.0;
        let dtilt = 0.24 * (x - 6.0);
        let v_path = 2.0 * x2 * g1 + 1.6 * x2 * g2 + sw * tilt;
        let dv_path = 2.0 * x * g1 * (2.0 - 3.0 * x2)
            + 1.6 * x * g2 * (2.0 - 32.0 * x * dx)
            + dsw * tilt
            + sw * dtilt;
        let valley = y - 0.25 * x;
        let energy = v_path + 4.0 * valley * valley;
        let gx = dv_path - 2.0 * valley;
        let gy = 8.0 * valley;
        (energy, [gx, gy])
    }
}

impl BandSurface for TiltedNarrowBarrier2D {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        for (i, r) in positions.outer_iter().enumerate() {
            let (e, [gx, gy]) = Self::jet(r[0], r[1]);
            energies[i] = e;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 4.0 * r[2];
        }
        Ok(())
    }
}

#[test]
fn free_end_adaptive_neb_on_narrow_barrier_matches_fixed_end_cineb_with_fewer_images() {
    let reactant = array![0.0, 0.0, 0.0];
    let product = array![6.0, 1.5, 0.0];
    let (e_reactant, [gx0, gy0]) = TiltedNarrowBarrier2D::jet(reactant[0], reactant[1]);
    assert_eq!(e_reactant, 0.0);
    assert_eq!(gx0, 0.0);
    assert_eq!(gy0, 0.0);

    // 1. Reference fixed-end CI-NEB with 21 images across the full [0, 6] path.
    let cineb_images = 21;
    let cineb_initial =
        linear_path(reactant.view(), product.view(), cineb_images, None).expect("cineb path");
    let cineb_surface = TiltedNarrowBarrier2D::default();
    let cineb_cfg = BandConfig::default()
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.08);
    assert!(
        cineb_cfg.free_end.is_none(),
        "fixed endpoints must stay the default"
    );
    let mut cineb = BandSession::new(cineb_cfg, cineb_initial).expect("cineb session");
    let cineb_rep = cineb.run(&cineb_surface, 800).expect("cineb run");
    assert_eq!(cineb_rep.status, BandStatus::Converged);
    let cineb_ci = cineb_rep.ci_index.expect("cineb ci");
    let cineb_saddle = cineb.positions().row(cineb_ci).to_owned();
    let (cineb_saddle_e, [cineb_gx, cineb_gy]) =
        TiltedNarrowBarrier2D::jet(cineb_saddle[0], cineb_saddle[1]);
    let cineb_barrier = cineb_saddle_e - e_reactant;
    let cineb_rows = cineb_surface.rows.load(Ordering::Relaxed);
    assert!(
        cineb_gx.hypot(cineb_gy) < 2e-3,
        "CI-NEB saddle gradient: {}",
        cineb_gx.hypot(cineb_gy)
    );

    // 2. FEA-NEB with only 7 images (3x fewer images than 21-image CI-NEB),
    // initialized on the long tilted slope and constrained to E(x_end) = -0.2
    // with 2 adaptive cut-and-remesh stages around the detected saddle.
    let fea_images = 7;
    assert!(fea_images < cineb_images);
    // On the valley y = 0.25 x, so the free-end tangent stays along the
    // gradient and Eq. (7) does not walk the endpoint down the basin.
    let far_product_slope = array![1.75, 0.4375, 0.0];
    let fea_initial =
        linear_path(reactant.view(), far_product_slope.view(), fea_images, None).expect("fea path");
    let target_energy = -0.2;
    let free_end_cfg = FreeEndConfig::default()
        .with_side(FreeEndSide::Last)
        .with_target_energy(Some(target_energy))
        .with_max_cuts(2)
        .with_cut_force(0.5)
        .with_energy_tol(0.01)
        .with_stability(2)
        .with_window_offset(1)
        .with_cubic(true);
    let fea_band_cfg = BandConfig::default()
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.08);
    let fea_surface = TiltedNarrowBarrier2D::default();
    let mut fea = FeaNebSession::new(fea_band_cfg, free_end_cfg, fea_initial).expect("fea session");

    let mut cut_events = 0;
    let mut fea_rep = fea.step(&fea_surface).expect("first step");
    if fea_rep.cut_this_step {
        cut_events += 1;
    }
    while fea_rep.status == BandStatus::Running && fea.iteration() < 800 {
        // Every step must keep the terminal image on the level set E(x_end) = target_energy.
        assert!(
            fea_rep.target_energy_residual.abs() < 1e-5,
            "step {} left level set: residual={:e}",
            fea_rep.iteration,
            fea_rep.target_energy_residual
        );
        fea_rep = fea.step(&fea_surface).expect("step");
        if fea_rep.cut_this_step {
            cut_events += 1;
        }
    }

    assert_eq!(fea_rep.status, BandStatus::Converged);
    assert_eq!(
        fea_rep.cuts, 2,
        "FEA-NEB must perform 2 cut-and-remesh refinements"
    );
    assert_eq!(cut_events, 2);
    assert_eq!(fea.cut_windows().len(), 2);
    assert!(
        fea_rep.target_energy_residual.abs() < 1e-6,
        "converged target_energy_residual={:e}",
        fea_rep.target_energy_residual
    );
    assert_eq!(
        fea.target_energy_residual(),
        Some(fea_rep.target_energy_residual)
    );

    let fea_ci = fea_rep.ci_index.expect("fea ci");
    let fea_saddle = fea.positions().row(fea_ci).to_owned();
    let (fea_saddle_e, [fea_gx, fea_gy]) = TiltedNarrowBarrier2D::jet(fea_saddle[0], fea_saddle[1]);
    let fea_initial_e0 = fea
        .initial_fixed_endpoint_energy()
        .expect("initial fixed endpoint energy");
    let fea_barrier = fea_saddle_e - fea_initial_e0;
    let fea_rows = fea_surface.rows.load(Ordering::Relaxed);

    // Check that the free terminal image sits on E(x_end) = target_energy AND
    // on the MEP where the level-set projected force vanishes.
    let end_pos = fea.positions().row(fea_images - 1).to_owned();
    let (end_e_check, _) = TiltedNarrowBarrier2D::jet(end_pos[0], end_pos[1]);
    assert!((end_e_check - target_energy).abs() < 1e-6);
    assert!(
        fea_rep.endpoint_force <= DEFAULT_FORCE_TOL,
        "free end must relax onto the MEP: endpoint_force={:e}",
        fea_rep.endpoint_force
    );

    eprintln!(
        "TiltedNarrowBarrier2D: FEA-NEB (7 images, cuts={}) saddle=({:.6}, {:.6}) barrier={:.6} residual={:.2e} rows={} steps={}; Fixed CI-NEB (21 images) saddle=({:.6}, {:.6}) barrier={:.6} rows={} steps={}",
        fea_rep.cuts,
        fea_saddle[0],
        fea_saddle[1],
        fea_barrier,
        fea_rep.target_energy_residual,
        fea_rows,
        fea_rep.iteration,
        cineb_saddle[0],
        cineb_saddle[1],
        cineb_barrier,
        cineb_rows,
        cineb_rep.iteration,
    );

    assert!(
        fea_gx.hypot(fea_gy) < 2e-3,
        "FEA-NEB saddle gradient: {}",
        fea_gx.hypot(fea_gy)
    );
    let saddle_dist = (&fea_saddle - &cineb_saddle)
        .dot(&(&fea_saddle - &cineb_saddle))
        .sqrt();
    assert!(
        saddle_dist < 1e-3,
        "FEA-NEB (7 images) and fixed-end CI-NEB (21 images) must locate the same saddle: dist={saddle_dist:e}"
    );
    assert!(
        (fea_barrier - cineb_barrier).abs() < 1e-5,
        "FEA-NEB and fixed-end CI-NEB barriers must agree: fea={fea_barrier}, cineb={cineb_barrier}"
    );
}

#[test]
fn free_end_first_side_and_zero_gradient_rejection() {
    // 1. FreeEndSide::First: image 0 moves on E(x_0) = E_target while image N-1 is fixed.
    let start_slope = array![1.8, 0.45, 0.0];
    let fixed_reactant = array![0.0, 0.0, 0.0];
    let reversed_initial =
        linear_path(start_slope.view(), fixed_reactant.view(), 7, None).expect("reversed");
    let fe_first = FreeEndConfig::default()
        .with_side(FreeEndSide::First)
        .with_target_energy(Some(-0.2))
        .with_max_cuts(1)
        .with_cut_force(0.4)
        .with_stability(2);
    let cfg = BandConfig::default()
        .with_free_end(fe_first)
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_force_tol(DEFAULT_FORCE_TOL)
        .with_max_move(0.08);
    let surface = TiltedNarrowBarrier2D::default();
    let mut session = BandSession::new(cfg, reversed_initial).expect("first-side session");
    let rep = session.run_free_end(&surface, 800).expect("run first-side");
    assert_eq!(rep.status, BandStatus::Converged);
    assert!(rep.target_energy_residual.abs() < 1e-6);
    assert_eq!(rep.cuts, 1);
    let ci = rep.ci_index.expect("ci");
    let saddle = session.positions().row(ci).to_owned();
    let (e_sad, [gx, gy]) = TiltedNarrowBarrier2D::jet(saddle[0], saddle[1]);
    assert!(gx.hypot(gy) < 2e-3);
    assert!((e_sad - 1.553_676).abs() < 1e-3);

    // 2. Nonzero endpoint gradient requirement: a free terminal image placed at
    // a stationary point (gradient = 0) must be rejected with SaddleError::Invalid.
    struct FlatEndpoint;
    impl BandSurface for FlatEndpoint {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let n = positions.nrows();
            for (i, r) in positions.outer_iter().enumerate() {
                energies[i] = r[0] * r[0];
                if i == n - 1 {
                    gradients.row_mut(i).fill(0.0);
                } else {
                    gradients[(i, 0)] = 2.0 * r[0];
                }
            }
            Ok(())
        }
    }

    let flat_band = Array2::from_shape_fn((5, 3), |(i, c)| if c == 0 { i as f64 } else { 0.0 });
    let mut bad_session = BandSession::new(
        BandConfig::default()
            .with_free_end(FreeEndConfig::default().with_target_energy(Some(16.0)))
            .with_min_mode(None)
            .with_saddle_check(None),
        flat_band,
    )
    .expect("session");
    let err = bad_session
        .step_free_end(&FlatEndpoint)
        .expect_err("zero endpoint gradient must be rejected");
    assert!(
        matches!(err, SaddleError::Invalid(_)),
        "expected SaddleError::Invalid for zero endpoint gradient, got {err:?}"
    );
}

#[test]
fn band_step_runs_free_end_min_mode_and_saddle_check() {
    let reactant = array![0.0, 0.0, 0.0];
    let slope = array![1.75, 0.4375, 0.0];
    let initial = linear_path(reactant.view(), slope.view(), 7, None).expect("path");
    let config = BandConfig::default()
        .with_free_end(
            FreeEndConfig::default()
                .with_target_energy(Some(-0.2))
                .with_max_cuts(0),
        )
        .with_remove_translation(false)
        .with_climbing(Some(CiConfig::after_rel(0.5)))
        .with_force_tol(1e-6)
        .with_max_move(0.08);
    let surface = TiltedNarrowBarrier2D::default();
    let mut session = BandSession::new(config, initial).expect("session");
    let mut refinement_seen = false;
    let report = loop {
        let report = session.step(&surface).expect("step");
        refinement_seen |= report.curvature.is_some();
        if report.status == BandStatus::Converged {
            break report;
        }
        assert!(session.iteration() < 800, "free-end band did not converge");
    };

    assert!(refinement_seen, "min-mode refinement never ran");
    let index = report.saddle_index.expect("saddle-index certificate");
    assert_eq!(index.index, 1);
}

/// `E = x^2 / 2 + 2 y`. The endpoint gradient is not parallel to the
/// band, so a one-sided spring would change the level-set step.
struct TiltedQuadratic;

impl BandSurface for TiltedQuadratic {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            energies[i] = 0.5 * row[0] * row[0] + 2.0 * row[1];
            gradients[(i, 0)] = row[0];
            gradients[(i, 1)] = 2.0;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

fn tilted_band() -> Array2<f64> {
    Array2::from_shape_fn((4, 3), |(image, axis)| match axis {
        0 => 1.0 + 0.4 * image as f64,
        1 => 0.2,
        _ => 0.0,
    })
}

#[test]
fn endpoint_step_does_not_depend_on_the_spring_constant() {
    let step_end = |k: f64| {
        let config = BandConfig::default()
            .with_spring(SpringKind::Uniform { k })
            .with_free_end(FreeEndConfig::default().with_max_cuts(0))
            .with_climbing(None)
            .with_min_mode(None)
            .with_saddle_check(None)
            .with_remove_translation(false);
        let mut session = BandSession::new(config, tilted_band()).expect("session");
        session
            .step_free_end(&TiltedQuadratic)
            .expect("one free-end step");
        session.positions().row(3).to_owned()
    };
    let soft = step_end(1.0);
    let stiff = step_end(9.0);
    assert_eq!(soft, stiff);
}

/// Energy is the constant `0`. On the starting geometry the free-end
/// gradient is nearly parallel to the path, so the level-set projection
/// leaves a force under the tolerance even when a one-sided spring is
/// still added. Any other geometry has a large transverse gradient.
struct FlatThenSteep {
    home: Array2<f64>,
}

impl BandSurface for FlatThenSteep {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let last = positions.nrows() - 1;
        for (i, row) in positions.outer_iter().enumerate() {
            energies[i] = 0.0;
            let at_home = row
                .iter()
                .zip(self.home.row(i).iter())
                .all(|(left, right)| left == right);
            gradients[(i, 2)] = 0.0;
            let transverse = 1.0e-4;
            let (gx, gy) = if at_home && i == last {
                (1.0, transverse)
            } else if at_home {
                (0.0, 0.0)
            } else {
                (4.0, 4.0)
            };
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
        }
        Ok(())
    }
}

#[test]
fn convergence_uses_the_force_at_the_reported_positions() {
    let positions = Array2::from_shape_fn(
        (4, 3),
        |(image, axis)| {
            if axis == 0 { image as f64 } else { 0.0 }
        },
    );
    let surface = FlatThenSteep {
        home: positions.clone(),
    };
    let config = BandConfig::default()
        .with_free_end(FreeEndConfig::default().with_max_cuts(0))
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false)
        .with_force_tol(DEFAULT_FORCE_TOL);
    let mut session = BandSession::new(config, positions).expect("session");
    let start = session.positions().row(3).to_owned();
    let report = session.step_free_end(&surface).expect("step");
    let positions = session.positions();
    let end = positions.row(3);
    let moved = start
        .iter()
        .zip(end.iter())
        .any(|(left, right)| left != right);
    assert!(moved, "the endpoint has to leave the flat point");
    assert_eq!(report.status, BandStatus::Running);
}

/// Interior energy peak plus a transverse gradient, so the climbing
/// image arms and the projected force stays above the tolerance.
struct InteriorPeak;

impl BandSurface for InteriorPeak {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let delta = row[0] - 1.5;
            energies[i] = -(delta * delta) + 2.0 * row[1];
            gradients[(i, 0)] = -2.0 * delta;
            gradients[(i, 1)] = 2.0;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn climbing_image_arming_does_not_add_evaluations() {
    let rows_of = |check: Option<SaddleIndexCheck>| {
        let positions = Array2::from_shape_fn(
            (4, 3),
            |(image, axis)| {
                if axis == 0 { image as f64 } else { 0.0 }
            },
        );
        let config = BandConfig::default()
            .with_free_end(FreeEndConfig::default().with_max_cuts(0))
            .with_climbing(Some(CiConfig::always()))
            .with_min_mode(None)
            .with_saddle_check(check)
            .with_remove_translation(false)
            .with_force_tol(DEFAULT_FORCE_TOL);
        let mut session = BandSession::new(config, positions).expect("session");
        let report = session.step_free_end(&InteriorPeak).expect("step");
        assert_eq!(report.status, BandStatus::Running);
        assert!(report.ci_index.is_some(), "climbing image should arm");
        report.surface_rows
    };
    assert_eq!(rows_of(Some(SaddleIndexCheck::default())), rows_of(None));
}

struct DownhillParabola {
    center: f64,
}

impl BandSurface for DownhillParabola {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let delta = row[0] - self.center;
            energies[i] = -(delta * delta);
            gradients[(i, 0)] = -2.0 * delta;
            gradients[(i, 1)] = 0.0;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

fn cut_once(side: FreeEndSide, n_images: usize, center: f64) -> (BandSession, usize) {
    let positions =
        Array2::from_shape_fn(
            (n_images, 3),
            |(image, axis)| {
                if axis == 0 { image as f64 } else { 0.0 }
            },
        );
    let config = BandConfig::default()
        .with_free_end(
            FreeEndConfig::default()
                .with_side(side)
                .with_max_cuts(1)
                .with_cut_force(100.0)
                .with_stability(1)
                .with_window_offset(1)
                .with_cubic(false),
        )
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false);
    let mut session = BandSession::new(config, positions).expect("session");
    session
        .step_free_end(&DownhillParabola { center })
        .expect("cut step");
    let windows = session.free_end_windows().len();
    (session, windows)
}

fn fresh_peak(session: &BandSession, center: f64) -> usize {
    let positions = session.positions();
    let mut best = 0usize;
    let mut best_energy = f64::NEG_INFINITY;
    for (image, row) in positions.outer_iter().enumerate() {
        let delta = row[0] - center;
        let energy = -(delta * delta);
        if energy > best_energy {
            best = image;
            best_energy = energy;
        }
    }
    best
}

#[test]
fn last_cut_near_the_fixed_end_keeps_the_peak_interior() {
    let (session, windows) = cut_once(FreeEndSide::Last, 5, 1.0);
    assert_eq!(windows, 1);
    assert_eq!(session.free_end_windows(), &[(0, 2)]);
    let peak = fresh_peak(&session, 1.0);
    assert_ne!(peak, 0);
    assert_ne!(peak, 4);
    assert_cached_matches(&session, &DownhillParabola { center: 1.0 });
}

#[test]
fn first_cut_near_the_free_end_drops_images() {
    let (session, windows) = cut_once(FreeEndSide::First, 5, 3.0);
    assert_eq!(windows, 1);
    assert_eq!(session.free_end_windows(), &[(2, 4)]);
    assert_cached_matches(&session, &DownhillParabola { center: 3.0 });
}

#[test]
fn full_band_window_does_not_count_a_cut() {
    let (session, windows) = cut_once(FreeEndSide::Last, 3, 1.0);
    assert_eq!(windows, 0);
    assert_eq!(session.free_end_cuts(), 0);
    assert!(session.free_end_windows().is_empty());
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2]) {
    let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
    for term in 0..4 {
        let (dx, dy) = (x - MB_X0[term], y - MB_Y0[term]);
        let q = MB_XX[term] * dx * dx + MB_XY[term] * dx * dy + MB_YY[term] * dy * dy;
        let value = MB_A[term] * q.exp();
        let qx = 2.0 * MB_XX[term] * dx + MB_XY[term] * dy;
        let qy = MB_XY[term] * dx + 2.0 * MB_YY[term] * dy;
        energy += value;
        gx += value * qx;
        gy += value * qy;
    }
    (energy, [gx, gy])
}

struct MullerBrown;

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, [gx, gy]) = muller_brown(row[0], row[1]);
            energies[i] = energy;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

#[test]
fn adaptive_cut_keeps_a_local_window_on_muller_brown() {
    let start = [-0.558_223_634_633, 1.441_725_841_805];
    let end = [0.623_499_404_931 + 0.15, 0.028_037_758_529 + 0.05];
    let n_images = 9usize;
    let offset = 1usize;
    let positions = Array2::from_shape_fn((n_images, 3), |(image, axis)| {
        let t = image as f64 / (n_images - 1) as f64;
        match axis {
            0 => start[0] + (end[0] - start[0]) * t,
            1 => start[1] + (end[1] - start[1]) * t,
            _ => 0.0,
        }
    });
    let chord_before = {
        let a = positions.row(0);
        let b = positions.row(n_images - 1);
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        dx.hypot(dy)
    };
    let config = BandConfig::default()
        .with_free_end(
            FreeEndConfig::default()
                .with_max_cuts(1)
                .with_cut_force(10_000.0)
                .with_stability(1)
                .with_window_offset(offset)
                .with_cubic(false),
        )
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false)
        .with_max_move(0.05);
    let mut session = BandSession::new(config, positions).expect("session");
    session.step_free_end(&MullerBrown).expect("cut");
    let (lo, hi) = session.free_end_windows()[0];
    assert!(lo > 0, "window {lo}..={hi} still includes the fixed end");
    assert!(
        hi + 1 < n_images,
        "window {lo}..={hi} still runs to the free end"
    );
    assert_eq!(hi - lo, 2 * offset);
    let chord_after = {
        let positions = session.positions();
        let a = positions.row(0);
        let b = positions.row(n_images - 2);
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        dx.hypot(dy)
    };
    assert!(chord_after < chord_before);
    assert_cached_matches(&session, &MullerBrown);
}

fn segment_lengths(positions: ndarray::ArrayView2<f64>) -> Vec<f64> {
    let mut lengths = Vec::new();
    for image in 1..positions.nrows() {
        let step = &positions.row(image) - &positions.row(image - 1);
        lengths.push(step.dot(&step).sqrt());
    }
    lengths
}

fn assert_even_monotone_free_end(session: &BandSession, side: FreeEndSide, anchor: f64) {
    let positions = session.positions();
    let n_images = positions.nrows();
    let end_idx = match side {
        FreeEndSide::First => 0,
        FreeEndSide::Last => n_images - 1,
    };
    let end = positions.row(end_idx)[0];
    let xs: Vec<f64> = positions.column(0).iter().copied().collect();
    let lengths = segment_lengths(positions);
    assert!(
        (end - anchor).abs() < 1e-8,
        "{side:?} dropped the free end: end={end} anchor={anchor} xs={xs:?}"
    );
    assert!(
        xs.windows(2).all(|pair| pair[1] > pair[0]),
        "{side:?} arc length is not monotone: xs={xs:?}"
    );
    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    assert!(
        lengths.iter().all(|len| (len - mean).abs() <= 1e-6),
        "{side:?} spacing is not even: lengths={lengths:?} xs={xs:?}"
    );
}

#[test]
fn last_cut_off_the_free_end_keeps_even_monotone_spacing() {
    let (session, windows) = cut_once(FreeEndSide::Last, 5, 1.0);
    assert_eq!(windows, 1);
    assert_eq!(session.free_end_windows(), &[(0, 2)]);
    assert_even_monotone_free_end(&session, FreeEndSide::Last, 4.0);
}

#[test]
fn first_cut_off_the_free_end_keeps_even_monotone_spacing() {
    let (session, windows) = cut_once(FreeEndSide::First, 5, 3.0);
    assert_eq!(windows, 1);
    assert_eq!(session.free_end_windows(), &[(2, 4)]);
    assert_even_monotone_free_end(&session, FreeEndSide::First, 0.0);
}

#[test]
fn edge_window_cut_keeps_the_free_end_and_even_spacing() {
    // n = 5, offset = 1. Last peaks at image 3, so the window is (2, 4)
    // and touches the free end. First peaks at image 1, so the window is
    // (0, 2) and touches the free end.
    let (last, last_windows) = cut_once(FreeEndSide::Last, 5, 3.0);
    assert_eq!(last_windows, 1);
    assert_eq!(last.free_end_windows(), &[(2, 4)]);
    assert_even_monotone_free_end(&last, FreeEndSide::Last, 4.0);

    let (first, first_windows) = cut_once(FreeEndSide::First, 5, 1.0);
    assert_eq!(first_windows, 1);
    assert_eq!(first.free_end_windows(), &[(0, 2)]);
    assert_even_monotone_free_end(&first, FreeEndSide::First, 0.0);
}

fn assert_cached_matches(session: &BandSession, surface: &impl BandSurface) {
    let held = session.evaluation().expect("free-end evaluation");
    let positions = session.positions().to_owned();
    let n_images = positions.nrows();
    let dof = positions.ncols();
    let mut energies = Array1::zeros(n_images);
    let mut gradients = Array2::zeros((n_images, dof));
    surface
        .eval(positions.view(), &mut energies, &mut gradients)
        .expect("direct evaluation");
    assert_eq!(held.energies, energies);
    assert_eq!(held.gradients, gradients);
    assert_eq!(held.projected.dim(), (n_images - 2, dof));
}

/// Records the size of every surface call and every row's coordinates.
#[derive(Default)]
struct CallLog {
    batches: Mutex<Vec<usize>>,
    first_coordinate: Mutex<Vec<f64>>,
}

impl CallLog {
    fn take_batches(&self) -> Vec<usize> {
        std::mem::take(&mut *self.batches.lock().unwrap())
    }

    fn evaluations_at(&self, x: f64) -> usize {
        self.first_coordinate
            .lock()
            .unwrap()
            .iter()
            .filter(|&&seen| seen == x)
            .count()
    }
}

impl BandSurface for CallLog {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.batches.lock().unwrap().push(positions.nrows());
        self.first_coordinate
            .lock()
            .unwrap()
            .extend(positions.outer_iter().map(|row| row[0]));
        TiltedQuadratic.eval(positions, energies, gradients)
    }
}

fn caching_band(n_images: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_images, 3), |(image, axis)| match axis {
        0 => 1.0 + 0.4 * image as f64,
        1 => 0.2,
        _ => 0.0,
    })
}

#[test]
fn free_end_steps_do_not_evaluate_the_fixed_endpoint_or_repeat_the_last_band() {
    let n_images = 6;
    let steps = 6;
    let positions = caching_band(n_images);
    let fixed_x = positions[(0, 0)];
    let config = BandConfig::default()
        .with_free_end(FreeEndConfig::default().with_max_cuts(0))
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false);
    let mut session = BandSession::new(config, positions).expect("session");
    let log = CallLog::default();

    let mut per_step = Vec::new();
    for _ in 0..steps {
        let report = session.step_free_end(&log).expect("step");
        let batches = log.take_batches();
        assert_eq!(
            report.surface_rows,
            batches.iter().sum::<usize>(),
            "the report counts the rows the surface saw"
        );
        per_step.push(batches);
    }

    // Step 1 meets the whole band once. The batch sizes are 1 (the
    // level-set retraction), n - 2 (the interior trials), n - 1 (the
    // band without its fixed endpoint) and n (the whole band).
    assert_eq!(
        per_step[0].iter().filter(|&&n| n == n_images).count(),
        1,
        "the first step evaluates the whole band: {:?}",
        per_step[0]
    );
    for (index, batches) in per_step.iter().enumerate().skip(1) {
        assert_eq!(
            batches.iter().filter(|&&n| n == n_images).count(),
            0,
            "step {} evaluated the fixed endpoint again: {batches:?}",
            index + 1
        );
        assert_eq!(
            batches.iter().filter(|&&n| n == n_images - 1).count(),
            1,
            "step {} evaluated the band at its start as well as its end: {batches:?}",
            index + 1
        );
        assert_ne!(
            batches.first(),
            Some(&(n_images - 1)),
            "step {} began with a band evaluation the previous step already held: {batches:?}",
            index + 1
        );
    }
    assert_eq!(
        log.evaluations_at(fixed_x),
        1,
        "the fixed endpoint is evaluated once for the whole run"
    );
}

#[test]
fn free_end_session_evaluates_the_band_again_after_a_host_move() {
    let n_images = 6;
    let positions = caching_band(n_images);
    let config = BandConfig::default()
        .with_free_end(FreeEndConfig::default().with_max_cuts(0))
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false);
    let mut session = BandSession::new(config, positions).expect("session");
    let log = CallLog::default();
    session.step_free_end(&log).expect("first step");
    session.step_free_end(&log).expect("second step");
    log.take_batches();

    let mut moved = session.positions().to_owned();
    moved[(2, 1)] += 0.05;
    session.set_positions(moved).expect("host move");
    session.step_free_end(&log).expect("step after the move");
    let batches = log.take_batches();
    assert_eq!(
        batches.first(),
        Some(&(n_images - 1)),
        "a moved band is evaluated at the start of the step: {batches:?}"
    );

    session.reset();
    session.step_free_end(&log).expect("step after reset");
    let batches = log.take_batches();
    assert_eq!(
        batches.first(),
        Some(&n_images),
        "a reset evaluates the whole band, fixed endpoint included: {batches:?}"
    );
}

#[test]
fn free_end_refuses_dynamic_neb() {
    let free_end = FreeEndConfig::default();
    let dynamic = DynamicNeb::default();

    let both = BandConfig::default()
        .with_free_end(free_end)
        .with_dynamic(dynamic);
    let err = match BandSession::new(both.clone(), tilted_band()) {
        Err(err) => err,
        Ok(_) => panic!("a band with free_end and dynamic was accepted"),
    };
    assert!(matches!(&err, SaddleError::Invalid(_)), "{err:?}");
    assert!(
        err.to_string().contains("dynamic NEB"),
        "the error names the conflict: {err}"
    );
    assert!(matches!(
        FeaNebSession::new(
            BandConfig::default().with_dynamic(dynamic),
            free_end,
            tilted_band()
        ),
        Err(SaddleError::Invalid(_))
    ));

    let mut dynamic_band =
        BandSession::new(BandConfig::default().with_dynamic(dynamic), tilted_band())
            .expect("dynamic session");
    assert!(matches!(
        dynamic_band.set_free_end(Some(free_end)),
        Err(SaddleError::Invalid(_))
    ));
    assert_eq!(dynamic_band.free_end(), None);

    let mut free_end_band =
        BandSession::new(BandConfig::default().with_free_end(free_end), tilted_band())
            .expect("free-end session");
    let err = free_end_band
        .set_dynamic(Some(dynamic))
        .expect_err("dynamic on a free-end band");
    assert!(matches!(&err, SaddleError::Invalid(_)), "{err:?}");
    assert!(err.to_string().contains("free-end"), "{err}");
    free_end_band
        .set_dynamic(None)
        .expect("clearing is allowed");
}

#[test]
fn reset_then_step_matches_a_fresh_session_after_free_end_progress() {
    let n_images = 5;
    let positions =
        Array2::from_shape_fn(
            (n_images, 3),
            |(image, axis)| {
                if axis == 0 { image as f64 } else { 0.0 }
            },
        );
    let config = BandConfig::default()
        .with_free_end(
            FreeEndConfig::default()
                .with_side(FreeEndSide::Last)
                .with_max_cuts(2)
                .with_cut_force(100.0)
                .with_stability(5)
                .with_window_offset(1)
                .with_cubic(false),
        )
        .with_climbing(None)
        .with_min_mode(None)
        .with_saddle_check(None)
        .with_remove_translation(false);
    let surface = DownhillParabola { center: 2.0 };
    let mut aged = BandSession::new(config.clone(), positions).expect("session");
    for _ in 0..4 {
        let report = aged.step_free_end(&surface).expect("age the free end");
        assert!(!report.cut_this_step);
        assert_eq!(report.cuts, 0);
    }
    assert!(aged.initial_fixed_endpoint_energy().is_some());
    aged.reset();
    let mut fresh = BandSession::new(config, aged.positions().to_owned()).expect("fresh");
    assert_eq!(
        aged.initial_fixed_endpoint_energy(),
        fresh.initial_fixed_endpoint_energy()
    );
    assert_eq!(aged.free_end_cuts(), fresh.free_end_cuts());
    assert_eq!(aged.free_end_windows(), fresh.free_end_windows());
    let aged_report = aged.step_free_end(&surface).expect("step after reset");
    let fresh_report = fresh.step_free_end(&surface).expect("fresh step");
    assert_eq!(aged.positions(), fresh.positions());
    assert_eq!(aged_report.cuts, fresh_report.cuts);
    assert_eq!(aged_report.cut_this_step, fresh_report.cut_this_step);
    assert_eq!(aged_report.status, fresh_report.status);
    assert_eq!(
        aged.initial_fixed_endpoint_energy(),
        fresh.initial_fixed_endpoint_energy()
    );
    assert!(aged.initial_fixed_endpoint_energy().is_some());
}
