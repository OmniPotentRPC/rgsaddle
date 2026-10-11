//! AutoNEB on the double-well band fixture and on Müller-Brown.
//!
//! Both searches stop at the same force tolerance. The fixed CI-NEB
//! starts with every image in place. AutoNEB starts from three of
//! those images and inserts the rest. The printed row counts are the
//! force calls.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::autoneb::worst_segment;
use rgsaddle::{
    AutoNeb, AutoNebConfig, AutoNebReport, BandConfig, BandReport, BandSession, BandStatus,
    BandSurface, Cell, CiConfig, ImageInterpolation, LinearInterpolation, SaddleError, eval_images,
    image_block,
};

struct Rows<S> {
    inner: S,
    rows: AtomicUsize,
}

impl<S> Rows<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            rows: AtomicUsize::new(0),
        }
    }

    fn rows(&self) -> usize {
        self.rows.load(Ordering::Relaxed)
    }
}

impl<S: BandSurface> BandSurface for Rows<S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        self.inner.eval(positions, energies, gradients)
    }
}

struct Calls {
    n: Arc<AtomicUsize>,
}

impl ImageInterpolation for Calls {
    fn interpolate(
        &self,
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
        n_between: usize,
        cell: Option<&Cell>,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        self.n.fetch_add(1, Ordering::Relaxed);
        LinearInterpolation.interpolate(left, right, n_between, cell)
    }
}

fn lbfgs_band(force_tol: f64) -> BandConfig {
    BandConfig {
        force_tol,
        max_move: 0.1,
        method: rgmin::Method::Lbfgs { memory: 10 },
        climbing: Some(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        }),
        ..BandConfig::default()
    }
}

fn fixed_cineb<S: BandSurface>(
    config: BandConfig,
    band: Array2<f64>,
    surface: &S,
    max_steps: usize,
) -> (BandReport, Array2<f64>) {
    let mut session = BandSession::new(config, band).unwrap();
    let mut report = session.step(surface).unwrap();
    while report.status == BandStatus::Running && report.iteration < max_steps {
        report = session.step(surface).unwrap();
    }
    (report, session.positions().to_owned())
}

fn ends_of(band: &Array2<f64>, count: usize) -> Array2<f64> {
    let n = band.nrows();
    let dof = band.ncols();
    let mut out = Array2::zeros((count, dof));
    for k in 0..count {
        let index = if count == 1 {
            0
        } else {
            k * (n - 1) / (count - 1)
        };
        out.row_mut(k).assign(&band.row(index));
    }
    out
}

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2. Minima at (+-1, 0, 0), saddle at
/// the origin, barrier 1. Same surface as tests/double_well.rs.
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

fn double_well_band(n: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    band
}

fn assert_origin_saddle(pos: ArrayView2<f64>, ci: usize) {
    let n = pos.nrows();
    assert_eq!(pos[(0, 0)], -1.0);
    assert_eq!(pos[(n - 1, 0)], 1.0);
    assert!(pos[(ci, 0)].abs() < 0.05, "ci x={}", pos[(ci, 0)]);
    assert!(pos[(ci, 1)].abs() < 0.05, "ci y={}", pos[(ci, 1)]);
    assert!(pos[(ci, 2)].abs() < 0.05, "ci z={}", pos[(ci, 2)]);
}

#[test]
fn double_well_autoneb_matches_fixed_cineb_accuracy() {
    let n_images = 7;
    let force_tol = 1e-3;
    let band = lbfgs_band(force_tol);
    let full = double_well_band(n_images);
    let inserted = Arc::new(AtomicUsize::new(0));
    let interp = Calls {
        n: Arc::clone(&inserted),
    };
    let mut auto = AutoNeb::with_interpolation(
        AutoNebConfig {
            band: band.clone(),
            n_simul: 3,
            n_max: n_images,
            local_force_tol: 5e-2,
            local_max_steps: 40,
            climb_max_steps: 500,
            space_energy_ratio: 0.5,
            climb: true,
            threads: 1,
        },
        ends_of(&full, 3),
        interp,
    )
    .unwrap();
    let auto_surface = Rows::new(DoubleWell);
    let auto_report = auto.run(&auto_surface).unwrap();
    assert_eq!(auto_report.status, BandStatus::Converged, "{auto_report:?}");
    assert_eq!(auto_report.n_images, n_images);
    assert_eq!(auto_report.surface_rows, auto_surface.rows());
    let ci = auto_report.ci_index.expect("climbing image");
    assert_origin_saddle(auto.positions(), ci);
    // Three supplied images, four inserted: two while the first window
    // is filled, two during the run.
    assert_eq!(inserted.load(Ordering::Relaxed), n_images - 3);

    let fixed_surface = Rows::new(DoubleWell);
    let (fixed_report, fixed_pos) = fixed_cineb(band, full, &fixed_surface, 2000);
    assert_eq!(
        fixed_report.status,
        BandStatus::Converged,
        "fixed max_force={}",
        fixed_report.max_force
    );
    let fixed_ci = fixed_report.ci_index.expect("fixed climbing image");
    assert_origin_saddle(fixed_pos.view(), fixed_ci);

    // This well converges in a handful of L-BFGS steps, so the extra
    // windows can spend more calls than one fixed band.
    eprintln!(
        "double_well autoneb_rows={} fixed_cineb_rows={} auto_force={:.3e} fixed_force={:.3e} tol={force_tol} n_images={n_images} method=lbfgs",
        auto_report.surface_rows,
        fixed_surface.rows(),
        auto_report.max_force,
        fixed_report.max_force
    );
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let mut energy = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    let mut hxx = 0.0;
    let mut hyy = 0.0;
    let mut hxy = 0.0;
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy;
        let weight = MB_A[i] * exponent.exp();
        let px = 2.0 * MB_AA[i] * dx + MB_B[i] * dy;
        let py = MB_B[i] * dx + 2.0 * MB_C[i] * dy;
        energy += weight;
        gx += weight * px;
        gy += weight * py;
        hxx += weight * (px * px + 2.0 * MB_AA[i]);
        hyy += weight * (py * py + 2.0 * MB_C[i]);
        hxy += weight * (px * py + MB_B[i]);
    }
    (energy, [gx, gy], [[hxx, hxy], [hxy, hyy]])
}

fn mb_index(x: f64, y: f64) -> i32 {
    let (_, _, h) = muller_brown(x, y);
    let tr = h[0][0] + h[1][1];
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let low = 0.5 * (tr - disc);
    let high = 0.5 * (tr + disc);
    i32::from(low < 0.0) + i32::from(high < 0.0)
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
            let (energy, g, _) = muller_brown(row[0], row[1]);
            energies[i] = energy;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

/// Minima B and C of Müller-Brown, with a perpendicular sine bump on
/// the interior images. The saddle between them is TS2.
fn muller_brown_band(n: usize) -> Array2<f64> {
    let start = [-0.05001082_f64, 0.46669410];
    let end = [0.62349940_f64, 0.02803776];
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let norm = (dx * dx + dy * dy).sqrt();
    let (px, py) = (-dy / norm, dx / norm);
    let mut band = Array2::zeros((n, 3));
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let bump = 0.15 * (std::f64::consts::PI * t).sin();
        band[(i, 0)] = start[0] + dx * t + bump * px;
        band[(i, 1)] = start[1] + dy * t + bump * py;
    }
    band
}

fn assert_ts2(pos: ArrayView2<f64>, ci: usize) {
    let ts = [0.21248658, 0.29298833];
    let x = pos[(ci, 0)];
    let y = pos[(ci, 1)];
    let dist = ((x - ts[0]).powi(2) + (y - ts[1]).powi(2)).sqrt();
    assert!(dist < 0.05, "ci at ({x}, {y}), distance {dist}");
    assert_eq!(mb_index(x, y), 1, "index at ({x}, {y})");
    assert!(pos[(ci, 2)].abs() < 1e-8, "z={}", pos[(ci, 2)]);
}

#[test]
fn muller_brown_autoneb_matches_fixed_cineb_accuracy() {
    let n_images = 7;
    let force_tol = 1e-2;
    let (_, g_start, _) = muller_brown(-0.05001082, 0.46669410);
    let (_, g_end, _) = muller_brown(0.62349940, 0.02803776);
    assert!(
        g_start[0].abs() < 1e-3 && g_start[1].abs() < 1e-3,
        "start gradient {g_start:?}"
    );
    assert!(
        g_end[0].abs() < 1e-3 && g_end[1].abs() < 1e-3,
        "end gradient {g_end:?}"
    );

    let band = lbfgs_band(force_tol);
    let full = muller_brown_band(n_images);
    let mut auto = AutoNeb::new(
        AutoNebConfig {
            band: band.clone(),
            n_simul: 3,
            n_max: n_images,
            local_force_tol: 1.0,
            local_max_steps: 30,
            climb_max_steps: 800,
            space_energy_ratio: 0.5,
            climb: true,
            threads: 1,
        },
        ends_of(&full, 3),
    )
    .unwrap();
    let auto_surface = Rows::new(MullerBrown);
    let auto_report = auto.run(&auto_surface).unwrap();
    assert_eq!(auto_report.status, BandStatus::Converged, "{auto_report:?}");
    assert_eq!(auto_report.n_images, n_images);
    assert_eq!(auto_report.surface_rows, auto_surface.rows());
    let ci = auto_report.ci_index.expect("climbing image");
    assert_ts2(auto.positions(), ci);
    assert_eq!(auto.positions()[(0, 0)], full[(0, 0)]);
    assert_eq!(auto.positions()[(n_images - 1, 0)], full[(n_images - 1, 0)]);

    let fixed_surface = Rows::new(MullerBrown);
    let (fixed_report, fixed_pos) = fixed_cineb(band, full, &fixed_surface, 2000);
    assert_eq!(
        fixed_report.status,
        BandStatus::Converged,
        "fixed max_force={}",
        fixed_report.max_force
    );
    let fixed_ci = fixed_report.ci_index.expect("fixed climbing image");
    assert_ts2(fixed_pos.view(), fixed_ci);
    assert!(
        auto_report.surface_rows < fixed_surface.rows(),
        "autoneb rows {} fixed rows {}",
        auto_report.surface_rows,
        fixed_surface.rows()
    );

    eprintln!(
        "muller_brown autoneb_rows={} fixed_cineb_rows={} auto_force={:.3e} fixed_force={:.3e} tol={force_tol} n_images={n_images} method=lbfgs",
        auto_report.surface_rows,
        fixed_surface.rows(),
        auto_report.max_force,
        fixed_report.max_force
    );
}

fn same_array_bits(left: &Array2<f64>, right: &Array2<f64>) {
    assert_eq!(left.dim(), right.dim());
    for (a, b) in left.iter().zip(right.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}");
    }
}

fn same_energy_bits(left: &[Option<f64>], right: &[Option<f64>]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right.iter()) {
        match (a, b) {
            (Some(a), Some(b)) => assert_eq!(a.to_bits(), b.to_bits(), "{a} vs {b}"),
            (None, None) => {}
            _ => panic!("energy presence {a:?} vs {b:?}"),
        }
    }
}

struct RankedRun {
    report: AutoNebReport,
    positions: Array2<f64>,
    energies: Vec<Option<f64>>,
}

fn ranked_double_well(threads: usize) -> RankedRun {
    let n_images = 7;
    let mut auto = AutoNeb::new(
        AutoNebConfig {
            band: lbfgs_band(1e-3),
            n_simul: 3,
            n_max: n_images,
            local_force_tol: 5e-2,
            local_max_steps: 40,
            climb_max_steps: 500,
            space_energy_ratio: 0.5,
            climb: true,
            threads,
        },
        ends_of(&double_well_band(n_images), 3),
    )
    .unwrap();
    let surface = Rows::new(DoubleWell);
    let report = auto.run(&surface).unwrap();
    assert_eq!(report.surface_rows, surface.rows());
    RankedRun {
        report,
        positions: auto.positions().to_owned(),
        energies: auto.energies().to_vec(),
    }
}

fn assert_same_run(label: &str, left: &RankedRun, right: &RankedRun) {
    assert_eq!(left.report.status, right.report.status, "{label}");
    assert_eq!(
        left.report.max_force.to_bits(),
        right.report.max_force.to_bits(),
        "{label}"
    );
    assert_eq!(left.report.ci_index, right.report.ci_index, "{label}");
    assert_eq!(
        left.report.surface_rows, right.report.surface_rows,
        "{label}"
    );
    assert_eq!(left.report.n_images, right.report.n_images, "{label}");
    same_array_bits(&left.positions, &right.positions);
    same_energy_bits(&left.energies, &right.energies);
}

#[test]
fn autoneb_is_bit_identical_at_one_two_and_n_threads() {
    let n_images = 7;
    let one = ranked_double_well(1);
    let two = ranked_double_well(2);
    let per_image = ranked_double_well(n_images);
    assert_eq!(one.report.status, BandStatus::Converged, "{:?}", one.report);
    assert_eq!(one.report.n_images, n_images);
    assert!(one.report.surface_rows > 0);
    assert_same_run("2 threads", &one, &two);
    assert_same_run("one thread per image", &one, &per_image);
}

struct DelayLeft {
    finished: Mutex<Vec<f64>>,
}

impl BandSurface for DelayLeft {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let tag = positions[(0, 0)];
        if tag < 0.0 {
            std::thread::sleep(Duration::from_millis(40));
        }
        for (i, row) in positions.outer_iter().enumerate() {
            let mut energy = 0.0;
            for (k, &x) in row.iter().enumerate() {
                energy += x * x;
                gradients[(i, k)] = 2.0 * x;
            }
            energies[i] = energy;
        }
        self.finished.lock().unwrap().push(tag);
        Ok(())
    }
}

#[test]
fn a_late_block_does_not_change_the_insertion_index() {
    let positions = ndarray::array![
        [-2.0, 0.0, 0.0],
        [-0.5, 0.1, 0.0],
        [0.4, -0.2, 0.0],
        [1.5, 0.3, 0.0],
    ];
    let n = positions.nrows();
    let dof = positions.ncols();
    let serial_surface = DelayLeft {
        finished: Mutex::new(Vec::new()),
    };
    let mut serial_e = Array1::zeros(n);
    let mut serial_g = Array2::zeros((n, dof));
    eval_images(
        &serial_surface,
        positions.view(),
        &mut serial_e,
        &mut serial_g,
        1,
    )
    .unwrap();

    let delayed = DelayLeft {
        finished: Mutex::new(Vec::new()),
    };
    let mut parallel_e = Array1::from_elem(n, f64::NAN);
    let mut parallel_g = Array2::from_elem((n, dof), f64::NAN);
    eval_images(
        &delayed,
        positions.view(),
        &mut parallel_e,
        &mut parallel_g,
        2,
    )
    .unwrap();
    for (a, b) in serial_e.iter().zip(parallel_e.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in serial_g.iter().zip(parallel_g.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let left = image_block(n, 0, 2);
    let right = image_block(n, 1, 2);
    assert_eq!(
        *delayed.finished.lock().unwrap(),
        vec![positions[(right.start, 0)], positions[(left.start, 0)]]
    );

    let mut lengths = Vec::with_capacity(n - 1);
    for j in 0..n - 1 {
        let step = &positions.row(j + 1) - &positions.row(j);
        lengths.push(step.dot(&step).sqrt());
    }
    let span_step = &positions.row(n - 1) - &positions.row(0);
    let span = span_step.dot(&span_step).sqrt();
    let serial: Vec<f64> = serial_e.iter().copied().collect();
    let parallel: Vec<f64> = parallel_e.iter().copied().collect();
    assert_eq!(
        worst_segment(&lengths, &serial, span, 0.5).unwrap(),
        worst_segment(&lengths, &parallel, span, 0.5).unwrap()
    );
}

/// Per-image polynomial. The value of one row does not depend on any
/// other row, and the loop is long enough to time.
struct HeavyPoly {
    passes: usize,
}

impl BandSurface for HeavyPoly {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let mut energy = 0.0;
            for (k, &x) in row.iter().enumerate() {
                let mut acc = 0.0;
                let mut deriv = 0.0;
                for pass in 0..self.passes {
                    let z = x + (pass as f64) * 1e-4;
                    acc += z.sin();
                    deriv += z.cos();
                }
                energy += acc;
                gradients[(i, k)] = deriv;
            }
            energies[i] = energy;
        }
        Ok(())
    }
}

fn median_and_range(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    assert_eq!(samples.len(), 5);
    samples.sort();
    (samples[2], samples[0], samples[4])
}

#[test]
fn parallel_image_eval_strong_scaling() {
    let n_images = 16;
    let dof = 32 * 3;
    let passes = 1600;
    let mut positions = Array2::zeros((n_images, dof));
    for i in 0..n_images {
        for k in 0..dof {
            positions[(i, k)] = (i + 1) as f64 * 1e-3 + k as f64 * 1e-4;
        }
    }
    let surface = HeavyPoly { passes };
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut counts = vec![1_usize, 2, 4];
    if !counts.contains(&max_threads) {
        counts.push(max_threads);
    }

    let mut serial_e = Array1::zeros(n_images);
    let mut serial_g = Array2::zeros((n_images, dof));
    eval_images(&surface, positions.view(), &mut serial_e, &mut serial_g, 1).unwrap();
    for &threads in &counts {
        let mut energies = Array1::from_elem(n_images, f64::NAN);
        let mut gradients = Array2::from_elem((n_images, dof), f64::NAN);
        eval_images(
            &surface,
            positions.view(),
            &mut energies,
            &mut gradients,
            threads,
        )
        .unwrap();
        for (a, b) in serial_e.iter().zip(energies.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "threads={threads}");
        }
        for (a, b) in serial_g.iter().zip(gradients.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "threads={threads}");
        }
    }

    for &threads in &counts {
        let mut energies = Array1::zeros(n_images);
        let mut gradients = Array2::zeros((n_images, dof));
        eval_images(
            &surface,
            positions.view(),
            &mut energies,
            &mut gradients,
            threads,
        )
        .unwrap();
        std::hint::black_box((&energies, &gradients));
    }

    let mut buffers: Vec<(Array1<f64>, Array2<f64>)> = counts
        .iter()
        .map(|_| (Array1::zeros(n_images), Array2::zeros((n_images, dof))))
        .collect();
    let mut samples = vec![Vec::with_capacity(5); counts.len()];
    for _rep in 0..5 {
        for (index, &threads) in counts.iter().enumerate() {
            let (energies, gradients) = &mut buffers[index];
            let started = Instant::now();
            eval_images(&surface, positions.view(), energies, gradients, threads).unwrap();
            let elapsed = started.elapsed();
            std::hint::black_box((&*energies, &*gradients));
            samples[index].push(elapsed);
        }
    }

    let mut rows: Vec<(usize, f64, Duration, Duration, f64)> = Vec::with_capacity(counts.len());
    for (index, &threads) in counts.iter().enumerate() {
        let (median, low, high) = median_and_range(&mut samples[index]);
        assert!(median > Duration::ZERO, "threads={threads}");
        let efficiency = if threads == 1 {
            1.0
        } else {
            let t1 = rows[0].1;
            t1 / (threads as f64 * median.as_secs_f64())
        };
        rows.push((threads, median.as_secs_f64(), low, high, efficiency));
    }

    // E(p) = median(T_1) / (p * median(T_p)).
    // Spawn and join dominates when E(2) is below 0.6.
    // Idle ranks dominate when E(max) is more than 0.15 below E(2).
    let efficiency_of = |threads: usize| -> f64 {
        rows.iter()
            .find(|(count, ..)| *count == threads)
            .map(|(_, _, _, _, efficiency)| *efficiency)
            .unwrap()
    };
    let limiter = if max_threads <= 1 {
        "single hardware thread"
    } else if efficiency_of(2) < 0.6 {
        "thread spawn and join"
    } else if max_threads > 2 && efficiency_of(max_threads) + 0.15 < efficiency_of(2) {
        "load imbalance and idle threads"
    } else {
        "arithmetic in the per-image kernel"
    };

    eprintln!(
        "strong scaling image evaluation: {n_images} images, {dof} dof, {passes} passes, median of 5 interleaved runs, hardware_threads={max_threads}"
    );
    eprintln!("threads  median_ms  min_ms  max_ms  efficiency");
    for (threads, median, low, high, efficiency) in &rows {
        eprintln!(
            "{threads:<7}  {:>9.3}  {:>6.3}  {:>6.3}  {efficiency:.3}",
            median * 1e3,
            low.as_secs_f64() * 1e3,
            high.as_secs_f64() * 1e3,
        );
    }
    eprintln!("limiter: {limiter}");
}
