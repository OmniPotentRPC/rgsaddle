//! OCI-NEB min-mode refinement of the climbing image on an analytic
//! double well, through the Rust session and the C band step.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, MinModeRefine, SaddleError};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at (+-1, 0, 0), saddle at
/// the origin with curvature -4 along x.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
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
    band[(n_images / 2, 0)] += 0.1;
    band
}

fn config(min_mode: Option<MinModeRefine>) -> BandConfig {
    BandConfig {
        force_tol: 1e-6,
        max_move: 0.1,
        min_mode,
        ..BandConfig::default()
    }
}

fn run(config: BandConfig) -> (BandSession, Vec<rgsaddle::BandReport>) {
    let mut session = BandSession::new(config, initial_band(7)).unwrap();
    let mut reports = Vec::new();
    for _ in 0..2000 {
        let report = session.step(&DoubleWell).unwrap();
        let done = report.status == BandStatus::Converged;
        reports.push(report);
        if done {
            break;
        }
    }
    (session, reports)
}

#[test]
fn the_refined_climbing_image_lands_on_the_saddle() {
    let (session, reports) = run(config(Some(MinModeRefine::default())));
    let last = reports.last().unwrap();
    assert_eq!(last.status, BandStatus::Converged);
    let curvatures: Vec<f64> = reports.iter().filter_map(|r| r.curvature).collect();
    assert!(!curvatures.is_empty(), "the refinement never ran");
    assert!(
        curvatures.iter().all(|&c| c < 0.0),
        "the refinement climbed a positive mode: {curvatures:?}"
    );
    let ci = last.ci_index.expect("climbing image");
    let pos = session.positions();
    for k in 0..3 {
        assert!(
            pos[(ci, k)].abs() < 1e-5,
            "coordinate {k}: {}",
            pos[(ci, k)]
        );
    }
    let gradient = session.evaluation().expect("evaluation").gradients;
    let norm = gradient.row(ci).dot(&gradient.row(ci)).sqrt();
    assert!(norm <= 1e-6, "climbing image gradient {norm}");
    assert_eq!(pos[(0, 0)], -1.0);
    assert_eq!(pos[(6, 0)], 1.0);
}

#[test]
fn the_refinement_counts_its_one_image_evaluations() {
    let (_, reports) = run(config(Some(MinModeRefine::default())));
    let refined = reports
        .iter()
        .filter(|r| r.curvature.is_some() && r.status == BandStatus::Running);
    for report in refined {
        assert!(
            report.surface_rows <= 9,
            "a refinement step is one image's Lanczos and step: {}",
            report.surface_rows
        );
    }
}

#[test]
fn without_min_mode_the_band_reports_no_curvature() {
    let (_, reports) = run(config(None));
    assert_eq!(reports.last().unwrap().status, BandStatus::Converged);
    assert!(reports.iter().all(|r| r.curvature.is_none()));
}

#[test]
fn an_invalid_refinement_is_refused() {
    let mut refine = MinModeRefine::default();
    refine.alignment_tol = 1.5;
    assert!(BandSession::new(config(Some(refine)), initial_band(7)).is_err());
    let mut refine = MinModeRefine::default();
    refine.curvature_retained = 1.5;
    assert!(BandSession::new(config(Some(refine)), initial_band(7)).is_err());
    let mut refine = MinModeRefine::default();
    refine.min_overlap = -0.1;
    assert!(BandSession::new(config(Some(refine)), initial_band(7)).is_err());
    let mut refine = MinModeRefine::default();
    refine.retry_factor = Some(0.0);
    assert!(BandSession::new(config(Some(refine)), initial_band(7)).is_err());
}

#[test]
fn an_infinite_trigger_needs_a_retry_factor() {
    let mut refine = MinModeRefine::default();
    assert!(refine.trigger_factor.is_infinite());
    refine.retry_factor = None;
    assert!(BandSession::new(config(Some(refine)), initial_band(7)).is_err());
    refine.trigger_factor = 0.5;
    assert!(BandSession::new(config(Some(refine)), initial_band(7)).is_ok());
}

#[test]
fn the_alignment_exit_still_lands_on_the_saddle() {
    let mut refine = MinModeRefine::default();
    refine.climb_alignment = true;
    let (session, reports) = run(config(Some(refine)));
    let last = reports.last().unwrap();
    assert_eq!(last.status, BandStatus::Converged);
    let ci = last.ci_index.expect("climbing image");
    assert!(session.positions()[(ci, 0)].abs() < 1e-5);
}

/// The double well, recording the band index of every one-image call.
struct Indexed(std::sync::Mutex<Vec<usize>>);

impl BandSurface for Indexed {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        DoubleWell.eval(positions, energies, gradients)
    }

    fn eval_image(
        &self,
        image: usize,
        position: ndarray::ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.0.lock().unwrap().push(image);
        let mut e = Array1::zeros(1);
        let mut g = Array2::zeros((1, position.len()));
        DoubleWell.eval(position.insert_axis(ndarray::Axis(0)), &mut e, &mut g)?;
        Ok((e[0], g.row(0).to_owned()))
    }
}

#[test]
fn the_refinement_evaluates_the_climbing_image_by_index() {
    let surface = Indexed(std::sync::Mutex::new(Vec::new()));
    let mut session =
        BandSession::new(config(Some(MinModeRefine::default())), initial_band(7)).unwrap();
    let mut last = None;
    for _ in 0..2000 {
        let report = session.step(&surface).unwrap();
        let done = report.status == BandStatus::Converged;
        last = Some(report);
        if done {
            break;
        }
    }
    let last = last.unwrap();
    assert_eq!(last.status, BandStatus::Converged);
    let ci = last.ci_index.expect("climbing image");
    let images = surface.0.into_inner().unwrap();
    assert!(!images.is_empty(), "the refinement never ran");
    assert!(images.iter().all(|&i| i == ci), "{images:?}, ci {ci}");
}

#[cfg(feature = "capi")]
mod capi {
    use rgsaddle::capi::*;
    use std::ffi::c_void;

    extern "C" fn double_well(_user: *mut c_void, request: *mut RgsaddleSurfaceRequest) -> i32 {
        let request = unsafe { &mut *request };
        let n = request.n_images as usize;
        let x = unsafe { std::slice::from_raw_parts(request.positions, 3 * n) };
        let e = unsafe { std::slice::from_raw_parts_mut(request.energies, n) };
        let g = unsafe { std::slice::from_raw_parts_mut(request.gradients, 3 * n) };
        for i in 0..n {
            let (a, b, c) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            e[i] = (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c;
            g[3 * i] = 4.0 * a * (a * a - 1.0);
            g[3 * i + 1] = 4.0 * b;
            g[3 * i + 2] = 4.0 * c;
        }
        RGSADDLE_OK
    }

    /// Converge a seven-image C band; the curvatures it reported.
    fn curvatures(flags: u64) -> Vec<f64> {
        let n = 7usize;
        let mut x = vec![0.0; 3 * n];
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            x[3 * i] = -1.0 + 2.0 * t;
            x[3 * i + 1] = 0.3 * (std::f64::consts::PI * t).sin();
        }
        let config = RgsaddleBandConfig {
            version: RgsaddleVersion {
                major: RGSADDLE_ABI_MAJOR,
                minor: RGSADDLE_ABI_MINOR,
            },
            flags,
            tangent: 1,
            spring: 0,
            projection: 1,
            method: 1,
            spring_k: 5.0,
            spring_ks: std::ptr::null(),
            ci_trigger_factor: 0.5,
            ci_trigger_force: 0.0,
            cell: std::ptr::null(),
            force_tol: 1e-5,
            max_move: 0.1,
            memory: 20,
        };
        let session = unsafe { rgsaddle_band_create(&config, n as i64, 1, x.as_ptr()) };
        assert!(!session.is_null());
        let mut report = RgsaddleReport {
            version: RgsaddleVersion { major: 0, minor: 0 },
            flags: 0,
            status: 0,
            evaluations: 0,
            max_force: 0.0,
            ci_index: -1,
            iteration: 0,
            curvature: 0.0,
            rotations: 0,
        };
        let mut seen = Vec::new();
        let mut converged = false;
        for _ in 0..2000 {
            let rc = unsafe {
                rgsaddle_band_step(
                    session,
                    Some(double_well),
                    std::ptr::null_mut(),
                    &mut report,
                )
            };
            assert_eq!(rc, RGSADDLE_OK);
            seen.push(report.curvature);
            if report.status == 1 {
                converged = true;
                break;
            }
        }
        assert!(converged);
        assert_eq!(
            unsafe { rgsaddle_band_positions(session, x.as_mut_ptr()) },
            RGSADDLE_OK
        );
        let ci = usize::try_from(report.ci_index).expect("climbing image");
        assert!(x[3 * ci].abs() < 1e-3, "saddle x {}", x[3 * ci]);
        unsafe { rgsaddle_band_free(session) };
        seen
    }

    #[test]
    fn the_c_band_refines_unless_the_flag_is_set() {
        assert!(curvatures(0).iter().any(|&c| c < 0.0));
        assert!(
            curvatures(RGSADDLE_BAND_NO_MIN_MODE)
                .iter()
                .all(|&c| c == 0.0)
        );
    }
}
