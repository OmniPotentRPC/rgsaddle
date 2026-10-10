//! CI-NEB to minimum-mode hand-off on analytic surfaces.

use std::sync::atomic::{AtomicUsize, Ordering};

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, OcinebConfig, OcinebPhase, OcinebSession,
    OcinebStatus, PointSurface, SaddleError,
};

struct Well {
    y_k: f64,
    z_k: f64,
    /// Quartic barrier when set. The quadratic barrier is `(x^2 - 1)^2`.
    quartic: bool,
}

impl Well {
    fn smooth() -> Self {
        Self {
            y_k: 2.0,
            z_k: 2.0,
            quartic: false,
        }
    }

    fn stiff() -> Self {
        Self {
            y_k: 25.0,
            z_k: 25.0,
            quartic: false,
        }
    }

    fn flat() -> Self {
        Self {
            y_k: 4.0,
            z_k: 4.0,
            quartic: true,
        }
    }

    fn energy_grad(&self, x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
        if self.quartic {
            let u = x * x - 1.0;
            let e = u.powi(4) + self.y_k * y * y + self.z_k * z * z;
            let g = [8.0 * x * u.powi(3), 2.0 * self.y_k * y, 2.0 * self.z_k * z];
            (e, g)
        } else {
            let e = (x * x - 1.0).powi(2) + self.y_k * y * y + self.z_k * z * z;
            let g = [
                4.0 * x * (x * x - 1.0),
                2.0 * self.y_k * y,
                2.0 * self.z_k * z,
            ];
            (e, g)
        }
    }
}

impl BandSurface for Well {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = self.energy_grad(row[0], row[1], row[2]);
            energies[i] = e;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = g[2];
        }
        Ok(())
    }
}

impl PointSurface for Well {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (e, g) = self.energy_grad(x[0], x[1], x[2]);
        Ok((e, Array1::from(g.to_vec())))
    }
}

/// Band on the double well, minimum-mode on a surface whose lowest
/// mode is perpendicular to the path.
struct Split;

impl BandSurface for Split {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        BandSurface::eval(&Well::smooth(), positions, energies, gradients)
    }
}

impl PointSurface for Split {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (px, py, pz) = (x[0], x[1], x[2]);
        let energy = 2.0 * px * px - 3.0 * py * py + 2.0 * pz * pz;
        let gradient = Array1::from(vec![4.0 * px, -6.0 * py, 4.0 * pz]);
        Ok((energy, gradient))
    }
}

/// Convex bowl, `V = (x² + y² + z²) / 2`. Every Hessian eigenvalue is 1,
/// so the climbing-image curvature is positive wherever the image sits.
struct Bowl;

impl Bowl {
    fn energy_grad(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
        (0.5 * (x * x + y * y + z * z), [x, y, z])
    }
}

impl BandSurface for Bowl {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = Self::energy_grad(row[0], row[1], row[2]);
            energies[i] = e;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = g[2];
        }
        Ok(())
    }
}

impl PointSurface for Bowl {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (e, g) = Self::energy_grad(x[0], x[1], x[2]);
        Ok((e, Array1::from(g.to_vec())))
    }
}

struct Counted<S> {
    inner: S,
    geometries: AtomicUsize,
}

impl<S> Counted<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            geometries: AtomicUsize::new(0),
        }
    }

    fn geometries(&self) -> usize {
        self.geometries.load(Ordering::Relaxed)
    }
}

impl<S: BandSurface> BandSurface for Counted<S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.geometries
            .fetch_add(positions.nrows(), Ordering::Relaxed);
        self.inner.eval(positions, energies, gradients)
    }
}

impl<S: PointSurface> PointSurface for Counted<S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.geometries.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

fn initial_band(n_images: usize, y_amp: f64) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = y_amp * (std::f64::consts::PI * t).sin();
    }
    band
}

fn ocineb_config(force_tol: f64) -> OcinebConfig {
    let mut config = OcinebConfig::default();
    config.band.force_tol = force_tol;
    config.band.max_move = 0.1;
    config.minmode.force_tol = force_tol;
    config.minmode.max_move = 0.1;
    config.minmode.rotation_tol = 1e-4;
    config
}

fn run_ocineb(well: &Well, n_images: usize, y_amp: f64, force_tol: f64) -> (OcinebSession, usize) {
    let surface = Counted::new(Well {
        y_k: well.y_k,
        z_k: well.z_k,
        quartic: well.quartic,
    });
    let mut session =
        OcinebSession::new(ocineb_config(force_tol), initial_band(n_images, y_amp)).unwrap();
    let mut seen = 0usize;
    let report = loop {
        let report = session.step(&surface).unwrap();
        seen += report.evaluations;
        if report.status == OcinebStatus::Converged || report.iteration > 4000 {
            break report;
        }
    };
    assert_eq!(
        report.status,
        OcinebStatus::Converged,
        "oci max_force={} iter={}",
        report.max_force,
        report.iteration
    );
    assert_eq!(seen, surface.geometries());
    (session, surface.geometries())
}

fn run_band(well: &Well, n_images: usize, y_amp: f64, force_tol: f64) -> usize {
    let surface = Counted::new(Well {
        y_k: well.y_k,
        z_k: well.z_k,
        quartic: well.quartic,
    });
    let config = BandConfig {
        force_tol,
        max_move: 0.1,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial_band(n_images, y_amp)).unwrap();
    for _ in 0..4000 {
        let report = session.step(&surface).unwrap();
        if report.status == BandStatus::Converged {
            return surface.geometries();
        }
    }
    panic!("plain CI-NEB did not converge");
}

#[test]
fn switch_waits_for_the_stability_latch() {
    let mut config = ocineb_config(1e-3);
    config.stability_count = 4;
    config.trigger_factor = 10.0;
    let mut session = OcinebSession::new(config, initial_band(9, 0.3)).unwrap();
    let mut switched_at = None;
    for _ in 0..400 {
        let report = session.step(&Well::smooth()).unwrap();
        if session.phase() == OcinebPhase::Align {
            switched_at = Some(report.iteration);
            assert!(session.stability() > 4, "stability={}", session.stability());
            assert_eq!(report.phase, OcinebPhase::Band);
            break;
        }
        assert_eq!(report.phase, OcinebPhase::Band);
        assert!(!report.fell_back);
    }
    let switched_at = switched_at.expect("hand-off did not arm");
    assert!(switched_at > 4, "switched at {switched_at}");

    let mut held = ocineb_config(1e-3);
    held.stability_count = 1000;
    held.trigger_factor = 10.0;
    let mut session = OcinebSession::new(held, initial_band(9, 0.3)).unwrap();
    for _ in 0..40 {
        let report = session.step(&Well::smooth()).unwrap();
        assert_eq!(session.phase(), OcinebPhase::Band);
        assert_eq!(report.phase, OcinebPhase::Band);
    }
}

#[test]
fn alignment_accepts_the_path_tangent_on_the_double_well() {
    let mut session = OcinebSession::new(ocineb_config(1e-3), initial_band(9, 0.3)).unwrap();
    let mut saw = false;
    for _ in 0..400 {
        let report = session.step(&Well::smooth()).unwrap();
        if report.phase == OcinebPhase::Align && !report.fell_back {
            assert!(report.alignment >= 0.85, "alignment={}", report.alignment);
            assert!(report.curvature < 0.0, "curvature={}", report.curvature);
            assert_eq!(session.phase(), OcinebPhase::MinMode);
            saw = true;
            break;
        }
    }
    assert!(saw, "no accepted alignment");
}

#[test]
fn misaligned_mode_falls_back_without_moving_the_band() {
    let mut config = ocineb_config(1e-3);
    config.trigger_factor = 10.0;
    config.stability_count = 2;
    let mut session = OcinebSession::new(config, initial_band(9, 0.3)).unwrap();
    let mut saw = false;
    for _ in 0..400 {
        if session.phase() == OcinebPhase::Align {
            let before = session.positions().to_owned();
            let report = session.step(&Split).unwrap();
            assert_eq!(report.phase, OcinebPhase::Align);
            assert!(report.fell_back, "alignment={}", report.alignment);
            assert!(report.alignment < 0.85, "alignment={}", report.alignment);
            assert_eq!(session.phase(), OcinebPhase::Band);
            let now = session.positions();
            let delta = (&now - &before).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(delta <= 1e-12, "band moved by {delta}");
            saw = true;
            break;
        }
        session.step(&Split).unwrap();
    }
    assert!(saw, "alignment fallback did not run");
}

#[test]
fn positive_curvature_restores_the_climbing_image() {
    let mut config = ocineb_config(1e-3);
    config.trigger_factor = 10.0;
    config.stability_count = 2;
    // Arm on the first assembly, while the bowed images still sit
    // above the endpoints. The bowl has no saddle, so the hand-off
    // must restore the band.
    if let Some(ci) = config.band.climbing.as_mut() {
        ci.trigger_factor = 2.0;
    }
    // y amplitude above 1 puts the middle image above both endpoints.
    let mut session = OcinebSession::new(config, initial_band(9, 2.0)).unwrap();
    let mut saw = false;
    for _ in 0..400 {
        if session.phase() == OcinebPhase::Align {
            let before = session.positions().to_owned();
            let threshold = session.threshold().unwrap();
            let report = session.step(&Bowl).unwrap();
            assert!(report.fell_back);
            assert!(report.curvature > 0.0, "curvature={}", report.curvature);
            assert_eq!(session.phase(), OcinebPhase::Band);
            let now = session.positions();
            let delta = (&now - &before).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(delta <= 1e-12, "band moved by {delta}");
            let after = session.threshold().unwrap();
            assert!(after < threshold, "threshold {threshold} -> {after}");
            saw = true;
            break;
        }
        session.step(&Bowl).unwrap();
    }
    assert!(saw, "positive-curvature fallback did not run");
}

#[test]
fn handoff_finds_the_same_saddle_as_plain_ci_neb() {
    let n_images = 9;
    let tol = 1e-3;
    let (session, _) = run_ocineb(&Well::smooth(), n_images, 0.3, tol);
    let pos = session.positions();
    assert_eq!(pos[(0, 0)], -1.0);
    assert_eq!(pos[(n_images - 1, 0)], 1.0);
    let ci = session.climbing_image().expect("climbing image");
    assert!(pos[(ci, 0)].abs() < 0.05, "ci x={}", pos[(ci, 0)]);
    assert!(pos[(ci, 1)].abs() < 0.05, "ci y={}", pos[(ci, 1)]);
    assert!(pos[(ci, 2)].abs() < 0.05, "ci z={}", pos[(ci, 2)]);

    let config = BandConfig {
        force_tol: tol,
        max_move: 0.1,
        ..BandConfig::default()
    };
    let mut plain = BandSession::new(config, initial_band(n_images, 0.3)).unwrap();
    let mut plain_report = plain.step(&Well::smooth()).unwrap();
    for _ in 0..4000 {
        if plain_report.status == BandStatus::Converged {
            break;
        }
        plain_report = plain.step(&Well::smooth()).unwrap();
    }
    assert_eq!(plain_report.status, BandStatus::Converged);
    let plain_ci = plain_report.ci_index.expect("plain climbing image");
    let plain_pos = plain.positions();
    assert_abs_diff_eq!(pos[(ci, 0)], plain_pos[(plain_ci, 0)], epsilon = 0.05);
    assert_abs_diff_eq!(pos[(ci, 1)], plain_pos[(plain_ci, 1)], epsilon = 0.05);
}

#[test]
fn benchmark_force_calls_against_plain_ci_neb() {
    let tol = 1e-3;
    let systems = [
        ("smooth double well", Well::smooth(), 9usize, 0.3f64),
        ("stiff perpendicular", Well::stiff(), 9, 0.4),
        ("quartic barrier", Well::flat(), 9, 0.3),
    ];
    eprintln!("| system | CI-NEB calls | OCI-NEB calls | OCI/CI |");
    eprintln!("| --- | ---: | ---: | ---: |");
    for (name, well, n_images, y_amp) in systems {
        let plain = run_band(&well, n_images, y_amp, tol);
        let (_session, hybrid) = run_ocineb(&well, n_images, y_amp, tol);
        let ratio = hybrid as f64 / plain as f64;
        eprintln!("| {name} | {plain} | {hybrid} | {ratio:.3} |");
        assert!(plain > 0 && hybrid > 0);
    }
}
