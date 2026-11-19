//! CI-NEB to minimum-mode hand-off on analytic surfaces.

#[cfg(feature = "faer")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "faer")]
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
#[cfg(feature = "faer")]
use rgsaddle::{BandConfig, BandSession, BandStatus, OcinebStatus};
use rgsaddle::{BandSurface, OcinebConfig, OcinebPhase, OcinebSession, PointSurface, SaddleError};

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

    #[cfg(feature = "faer")]
    fn stiff() -> Self {
        Self {
            y_k: 25.0,
            z_k: 25.0,
            quartic: false,
        }
    }

    #[cfg(feature = "faer")]
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

#[cfg(feature = "faer")]
struct Counted<S> {
    inner: S,
    geometries: AtomicUsize,
}

#[cfg(feature = "faer")]
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

#[cfg(feature = "faer")]
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

#[cfg(feature = "faer")]
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
    let config = OcinebConfig::default();
    let band = config
        .band
        .clone()
        .with_force_tol(force_tol)
        .with_max_move(0.1);
    let minmode = config
        .minmode
        .clone()
        .with_force_tol(force_tol)
        .with_max_move(0.1)
        .with_rotation_tol(1e-4);
    config.with_band(band).with_minmode(minmode)
}

#[cfg(feature = "faer")]
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

#[cfg(feature = "faer")]
fn run_band(
    well: &Well,
    n_images: usize,
    y_amp: f64,
    force_tol: f64,
) -> (BandSession, BandStatus, usize) {
    let surface = Counted::new(Well {
        y_k: well.y_k,
        z_k: well.z_k,
        quartic: well.quartic,
    });
    let config = BandConfig::default()
        .with_force_tol(force_tol)
        .with_max_move(0.1)
        .with_min_mode(None);
    let mut session = BandSession::new(config, initial_band(n_images, y_amp)).unwrap();
    let mut status = BandStatus::Running;
    for _ in 0..4000 {
        status = session.step(&surface).unwrap().status;
        if status == BandStatus::Converged {
            break;
        }
    }
    (session, status, surface.geometries())
}

/// Position and energy of the climbing image.
#[cfg(feature = "faer")]
fn climbing_point(well: &Well, positions: ArrayView2<f64>, ci: usize) -> ([f64; 3], f64) {
    let x = [positions[(ci, 0)], positions[(ci, 1)], positions[(ci, 2)]];
    (x, well.energy_grad(x[0], x[1], x[2]).0)
}

#[test]
fn switch_waits_for_the_stability_latch() {
    let config = ocineb_config(1e-3)
        .with_stability_count(4)
        .with_trigger_factor(10.0);
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

    let held = ocineb_config(1e-3)
        .with_stability_count(1000)
        .with_trigger_factor(10.0);
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
    let config = ocineb_config(1e-3)
        .with_trigger_factor(10.0)
        .with_stability_count(2);
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
    let config = ocineb_config(1e-3)
        .with_trigger_factor(10.0)
        .with_stability_count(2);
    // Arm on the first assembly, while the bowed images still sit
    // above the endpoints. The bowl has no saddle, so the hand-off
    // must restore the band.
    let climbing = config.band.climbing.map(|ci| ci.with_trigger_factor(2.0));
    let band = config.band.clone().with_climbing(climbing);
    let config = config.with_band(band);
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

#[cfg(feature = "faer")]
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

    let config = BandConfig::default()
        .with_force_tol(tol)
        .with_max_move(0.1)
        .with_min_mode(None);
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

#[cfg(not(feature = "faer"))]
#[test]
fn handoff_climb_without_faer_returns_solver_error() {
    let mut session = OcinebSession::new(ocineb_config(1e-3), initial_band(9, 0.3)).unwrap();
    let err = session.run(&Well::smooth(), 400).unwrap_err();
    assert_eq!(session.phase(), OcinebPhase::MinMode);
    match err {
        SaddleError::Solver(msg) => assert!(msg.contains("faer"), "{msg}"),
        other => panic!("expected SaddleError::Solver, got {other:?}"),
    }
}

#[cfg(all(feature = "capi", not(feature = "faer")))]
#[test]
fn c_abi_handoff_climb_without_faer_returns_solver_status() {
    use rgsaddle::capi::*;
    use std::ffi::c_void;

    extern "C" fn double_well(_user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
        let req = unsafe { &mut *req };
        let rows = req.n_images as usize;
        let x = unsafe { std::slice::from_raw_parts(req.positions, 3 * rows) };
        let e = unsafe { std::slice::from_raw_parts_mut(req.energies, rows) };
        let g = unsafe { std::slice::from_raw_parts_mut(req.gradients, 3 * rows) };
        for i in 0..rows {
            let (px, py, pz) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            e[i] = (px * px - 1.0).powi(2) + 2.0 * py * py + 2.0 * pz * pz;
            g[3 * i] = 4.0 * px * (px * px - 1.0);
            g[3 * i + 1] = 4.0 * py;
            g[3 * i + 2] = 4.0 * pz;
        }
        RGSADDLE_OK
    }

    let n_images = 9usize;
    let pos: Vec<f64> = initial_band(n_images, 0.3).into_iter().collect();
    let band = RgsaddleBandConfig {
        version: RgsaddleVersion {
            major: RGSADDLE_ABI_MAJOR,
            minor: RGSADDLE_ABI_MINOR,
        },
        flags: 0,
        tangent: 1,
        spring: 0,
        projection: 1,
        method: 0,
        spring_k: 5.0,
        spring_ks: std::ptr::null(),
        ci_trigger_factor: 0.5,
        ci_trigger_force: 0.0,
        cell: std::ptr::null(),
        force_tol: 1e-3,
        max_move: 0.1,
        memory: 0,
    };
    let mm = RgsaddleMinModeConfig {
        version: RgsaddleVersion {
            major: RGSADDLE_ABI_MAJOR,
            minor: RGSADDLE_ABI_MINOR,
        },
        flags: 0,
        kind: 0,
        method: 0,
        dr: 1e-3,
        rotation_tol: 1e-4,
        max_rotations: 20,
        krylov_dim: 8,
        force_tol: 1e-3,
        max_move: 0.1,
        rotation_angle_tol: 5f64.to_radians(),
        difference: 0,
        reserved: 0,
    };
    let cfg = RgsaddleOcinebConfig {
        version: RgsaddleVersion {
            major: RGSADDLE_ABI_MAJOR,
            minor: RGSADDLE_ABI_MINOR,
        },
        flags: 0,
        trigger_factor: 0.31,
        trigger_force: 0.0,
        angle_tol: 0.85,
        stability_count: 5,
        max_mmf_steps: 1000,
        restore_unhelpful: 0,
        reserved: 0,
    };
    let session =
        unsafe { rgsaddle_ocineb_create(&band, &mm, &cfg, n_images as i64, 1, pos.as_ptr()) };
    assert!(!session.is_null());
    let mut report: RgsaddleOcinebReport = unsafe { std::mem::zeroed() };
    let mut saw_align = false;
    let mut final_rc = RGSADDLE_OK;
    for _ in 0..400 {
        let rc = unsafe {
            rgsaddle_ocineb_step(
                session,
                Some(double_well),
                std::ptr::null_mut(),
                &mut report,
            )
        };
        if rc != RGSADDLE_OK {
            final_rc = rc;
            break;
        }
        if report.phase == 1 {
            saw_align = true;
        }
    }
    unsafe { rgsaddle_ocineb_free(session) };
    assert!(saw_align, "expected alignment step before climb failure");
    assert_eq!(final_rc, RGSADDLE_SOLVER);
}

/// OCI-NEB (Goswami, Gunde, and Jónsson, "Enhanced climbing image
/// nudged elastic band method with Hessian eigenmode alignment",
/// Front. Chem. 14 (2026), doi:10.3389/fchem.2026.1807063) against
/// plain CI-NEB on three wells.
///
/// Every well is separable with its saddle at the origin, energy 1, and
/// no Hessian eigenvalue there smaller than 4 in magnitude. A climbing
/// image whose largest force component is under `tol` is then, to first
/// order, within `tol / 4` of the saddle in each coordinate and within
/// `3 tol^2 / 8` of its energy, so two converged climbing images agree
/// to `tol` in position and `tol^2` in energy.
///
/// The OCI/CI force-call ratios are 0.170, 0.167 and 0.201 on these
/// starts and at most 0.332 over initial bows of 0.2 to 0.5, so the
/// bound is half the CI-NEB count.
#[cfg(feature = "faer")]
#[test]
fn benchmark_force_calls_against_plain_ci_neb() {
    let tol = 1e-3;
    let position_tol = tol;
    let energy_tol = tol * tol;
    let max_ratio = 0.5;
    let systems = [
        ("smooth double well", Well::smooth(), 9usize, 0.3f64),
        ("stiff perpendicular", Well::stiff(), 9, 0.4),
        ("quartic barrier", Well::flat(), 9, 0.3),
    ];
    eprintln!("| system | CI-NEB calls | OCI-NEB calls | OCI/CI |");
    eprintln!("| --- | ---: | ---: | ---: |");
    for (name, well, n_images, y_amp) in systems {
        let (band, status, plain) = run_band(&well, n_images, y_amp, tol);
        assert_eq!(status, BandStatus::Converged, "{name}: CI-NEB");
        let (session, hybrid) = run_ocineb(&well, n_images, y_amp, tol);
        let ratio = hybrid as f64 / plain as f64;
        eprintln!("| {name} | {plain} | {hybrid} | {ratio:.3} |");

        let band_ci = band.climbing_image().expect("CI-NEB climbing image");
        let oci_ci = session.climbing_image().expect("OCI-NEB climbing image");
        let (band_x, band_e) = climbing_point(&well, band.positions(), band_ci);
        let (oci_x, oci_e) = climbing_point(&well, session.positions(), oci_ci);
        for k in 0..3 {
            assert_abs_diff_eq!(band_x[k], oci_x[k], epsilon = position_tol);
            assert_abs_diff_eq!(band_x[k], 0.0, epsilon = position_tol);
            assert_abs_diff_eq!(oci_x[k], 0.0, epsilon = position_tol);
        }
        assert_abs_diff_eq!(band_e, oci_e, epsilon = energy_tol);
        assert_abs_diff_eq!(band_e, 1.0, epsilon = energy_tol);
        assert_abs_diff_eq!(oci_e, 1.0, epsilon = energy_tol);

        assert!(
            ratio <= max_ratio,
            "{name}: OCI-NEB {hybrid} calls, CI-NEB {plain}, ratio {ratio:.3} > {max_ratio}"
        );
    }
}
