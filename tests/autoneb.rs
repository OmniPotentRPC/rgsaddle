//! AutoNEB against a fixed CI-NEB, and against ASE's AutoNEB.
//!
//! Both methods use the same climbing trigger and the same force
//! tolerance. The printed CSV is one row per start.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    AutoNeb, AutoNebReport, AutoNebTuning, BandConfig, BandReport, BandSession, BandStatus,
    BandSurface, CiConfig, SaddleError,
};

struct Rows<S> {
    inner: S,
    rows: AtomicUsize,
    samples: Mutex<Vec<[f64; 3]>>,
}

impl<S> Rows<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            rows: AtomicUsize::new(0),
            samples: Mutex::new(Vec::new()),
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
        let mut samples = self.samples.lock().unwrap();
        for row in positions.outer_iter() {
            if row.len() >= 3 {
                samples.push([row[0], row[1], row[2]]);
            }
        }
        drop(samples);
        self.inner.eval(positions, energies, gradients)
    }
}

fn tuning(n_max: usize, threads: usize) -> AutoNebTuning {
    AutoNebTuning {
        n_simul: 3,
        n_max,
        local_force_tol: 1e-1,
        local_max_steps: 40,
        climb_max_steps: 800,
        space_energy_ratio: 0.5,
        climb: true,
        threads,
    }
}

fn lbfgs_band(force_tol: f64, n_max: usize, threads: usize) -> BandConfig {
    BandConfig {
        force_tol,
        max_move: 0.1,
        method: rgmin::Method::Lbfgs { memory: 10 },
        climbing: Some(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        }),
        autoneb: Some(tuning(n_max, threads)),
        ..BandConfig::default()
    }
}

fn fixed_cineb<S: BandSurface>(
    config: BandConfig,
    band: Array2<f64>,
    surface: &S,
    max_steps: usize,
) -> (BandReport, Array2<f64>, Option<f64>) {
    let mut session = BandSession::new(config, band).unwrap();
    let mut report = session.step(surface).unwrap();
    while report.status == BandStatus::Running && report.iteration < max_steps {
        report = session.step(surface).unwrap();
    }
    let energy = report
        .ci_index
        .and_then(|index| session.evaluation().map(|eval| eval.energies[index]));
    (report, session.positions().to_owned(), energy)
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

fn rattle(band: &Array2<f64>, seed: u64) -> Array2<f64> {
    let mut out = band.clone();
    if seed == 0 {
        return out;
    }
    for i in 1..out.nrows() - 1 {
        out[(i, 1)] += (seed as f64) * 0.01 * ((i as f64) * 0.17).sin();
    }
    out
}

#[derive(Clone, Copy)]
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

fn near_origin(pos: ArrayView2<f64>, ci: usize) -> bool {
    pos[(ci, 0)].abs() < 0.05 && pos[(ci, 1)].abs() < 0.05 && pos[(ci, 2)].abs() < 0.05
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

#[derive(Clone, Copy)]
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

fn near_ts2(pos: ArrayView2<f64>, ci: usize) -> bool {
    let ts = [0.21248658, 0.29298833];
    let dist = ((pos[(ci, 0)] - ts[0]).powi(2) + (pos[(ci, 1)] - ts[1]).powi(2)).sqrt();
    dist < 0.05 && mb_index(pos[(ci, 0)], pos[(ci, 1)]) == 1 && pos[(ci, 2)].abs() < 1e-8
}

struct SeedRow {
    calls: f64,
    wall: f64,
    ok: bool,
}

struct CsvLine<'a> {
    system: &'a str,
    method: &'a str,
    seed: u64,
    converged: bool,
    iters: usize,
    calls: usize,
    energy: f64,
    fmax: f64,
    wall: f64,
}

fn csv_line(line: CsvLine<'_>) {
    eprintln!(
        "{system},{method},{seed},{converged},{iters},{calls},{energy:.8},{fmax:.8e},{wall:.6}",
        system = line.system,
        method = line.method,
        seed = line.seed,
        converged = i32::from(line.converged),
        iters = line.iters,
        calls = line.calls,
        energy = line.energy,
        fmax = line.fmax,
        wall = line.wall,
    );
}

fn median_range(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    (values[mid], values[0], *values.last().unwrap())
}

fn compare_surface(
    system: &str,
    force_tol: f64,
    full: Array2<f64>,
    near: fn(ArrayView2<f64>, usize) -> bool,
    surface: impl BandSurface + Copy,
) {
    let n_images = full.nrows();
    let mut auto_calls = Vec::new();
    let mut fixed_calls = Vec::new();
    eprintln!("system,method,seed,converged,iters,calls,energy,fmax,wall_s");
    for seed in 0..5 {
        let rattled = rattle(&full, seed);
        let band = lbfgs_band(force_tol, n_images, 1);
        let mut auto = AutoNeb::new(band.clone(), ends_of(&rattled, 3)).unwrap();
        let auto_surface = Rows::new(surface);
        let started = Instant::now();
        let auto_report = auto.run(&auto_surface).unwrap();
        let auto_wall = started.elapsed().as_secs_f64();
        assert_eq!(auto_report.surface_rows, auto_surface.rows());
        let auto_ci = auto_report.ci_index.expect("climbing image");
        let auto_ok =
            auto_report.status == BandStatus::Converged && near(auto.positions(), auto_ci);
        let auto_energy = auto.energies()[auto_ci].unwrap_or(f64::NAN);
        csv_line(CsvLine {
            system,
            method: "autoneb",
            seed,
            converged: auto_report.status == BandStatus::Converged,
            iters: auto_report.iteration,
            calls: auto_report.surface_rows,
            energy: auto_energy,
            fmax: auto_report.max_force,
            wall: auto_wall,
        });
        if seed == 0 {
            let left = rattled.row(0);
            let right = rattled.row(rattled.nrows() - 1);
            let hits = auto_surface.samples.lock().unwrap();
            let left_hits = hits
                .iter()
                .filter(|row| {
                    row[0].to_bits() == left[0].to_bits() && row[1].to_bits() == left[1].to_bits()
                })
                .count();
            let right_hits = hits
                .iter()
                .filter(|row| {
                    row[0].to_bits() == right[0].to_bits() && row[1].to_bits() == right[1].to_bits()
                })
                .count();
            assert_eq!(left_hits, 1, "{system} left endpoint evaluations");
            assert_eq!(right_hits, 1, "{system} right endpoint evaluations");
        }
        auto_calls.push(SeedRow {
            calls: auto_report.surface_rows as f64,
            wall: auto_wall,
            ok: auto_ok,
        });

        let fixed_band = BandConfig {
            autoneb: None,
            ..band
        };
        let fixed_surface = Rows::new(surface);
        let started = Instant::now();
        let (fixed_report, fixed_pos, fixed_energy) =
            fixed_cineb(fixed_band, rattled, &fixed_surface, 2000);
        let fixed_wall = started.elapsed().as_secs_f64();
        let fixed_ci = fixed_report.ci_index.expect("fixed climbing image");
        let fixed_ok =
            fixed_report.status == BandStatus::Converged && near(fixed_pos.view(), fixed_ci);
        csv_line(CsvLine {
            system,
            method: "cineb",
            seed,
            converged: fixed_report.status == BandStatus::Converged,
            iters: fixed_report.iteration,
            calls: fixed_surface.rows(),
            energy: fixed_energy.unwrap_or(f64::NAN),
            fmax: fixed_report.max_force,
            wall: fixed_wall,
        });
        fixed_calls.push(SeedRow {
            calls: fixed_surface.rows() as f64,
            wall: fixed_wall,
            ok: fixed_ok,
        });
    }
    for (method, rows) in [("autoneb", &auto_calls), ("cineb", &fixed_calls)] {
        assert!(rows.iter().all(|row| row.ok), "{system} {method}");
        let mut calls: Vec<f64> = rows.iter().map(|row| row.calls).collect();
        let mut walls: Vec<f64> = rows.iter().map(|row| row.wall).collect();
        let (call_med, call_lo, call_hi) = median_range(&mut calls);
        let (wall_med, wall_lo, wall_hi) = median_range(&mut walls);
        eprintln!(
            "{system} {method} calls median={call_med:.1} range={call_lo:.1}..{call_hi:.1} wall_s median={wall_med:.6} range={wall_lo:.6}..{wall_hi:.6}"
        );
    }
}

#[test]
fn fair_double_well_and_muller_brown_starts() {
    compare_surface(
        "double_well",
        1e-3,
        double_well_band(7),
        near_origin,
        DoubleWell,
    );
    compare_surface(
        "muller_brown",
        1e-2,
        muller_brown_band(7),
        near_ts2,
        MullerBrown,
    );
}

struct RankedRun {
    report: AutoNebReport,
    positions: Array2<f64>,
    energies: Vec<Option<f64>>,
}

fn ranked_double_well(threads: usize) -> RankedRun {
    let n_images = 9;
    let mut auto = AutoNeb::new(
        lbfgs_band(1e-3, n_images, threads),
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

fn hash_run(run: &RankedRun) -> u64 {
    let mut hasher = DefaultHasher::new();
    for value in run.positions.iter() {
        value.to_le_bytes().hash(&mut hasher);
    }
    for energy in &run.energies {
        match energy {
            Some(value) => {
                1u8.hash(&mut hasher);
                value.to_le_bytes().hash(&mut hasher);
            }
            None => 0u8.hash(&mut hasher),
        }
    }
    run.report.max_force.to_le_bytes().hash(&mut hasher);
    hasher.finish()
}

#[test]
fn autoneb_is_bit_identical_at_one_two_three_and_eight_threads() {
    let one = ranked_double_well(1);
    assert_eq!(
        one.report.status,
        BandStatus::Converged,
        "{:?}",
        one.report.max_force
    );
    assert_eq!(one.report.n_images, 9);
    let digest = hash_run(&one);
    for threads in [2, 3, 8] {
        let other = ranked_double_well(threads);
        assert_eq!(hash_run(&other), digest, "threads={threads}");
        assert_eq!(other.report.surface_rows, one.report.surface_rows);
        assert_eq!(other.report.ci_index, one.report.ci_index);
    }
    eprintln!("bit identity threads=1,2,3,8 hash={digest:#x} simd=none");
}

fn json_field<'a>(text: &'a str, key: &str) -> &'a str {
    let pat = format!("\"{key}\": ");
    let start = text
        .find(&pat)
        .unwrap_or_else(|| panic!("missing {key} in {text}"))
        + pat.len();
    let rest = &text[start..];
    if let Some(stripped) = rest.strip_prefix('[') {
        let end = stripped.find(']').unwrap();
        &stripped[..end]
    } else {
        let end = rest.find([',', '}']).unwrap();
        &rest[..end]
    }
}

fn json_ints(text: &str, key: &str) -> Vec<usize> {
    json_field(text, key)
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse().unwrap())
        .collect()
}

fn json_floats(text: &str, key: &str) -> Vec<f64> {
    json_field(text, key)
        .split(',')
        .map(|part| part.trim().parse().unwrap())
        .collect()
}

#[test]
fn muller_brown_matches_an_ase_autoneb_run() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ase_autoneb_mb.py");
    let run = Command::new("python3")
        .arg(&script)
        .output()
        .expect("run ASE AutoNEB");
    assert!(
        run.status.success(),
        "ASE AutoNEB failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let text = String::from_utf8(run.stdout).unwrap();
    let dynamic = json_ints(&text, "dynamic");
    let ase_ci = json_floats(&text, "ci");
    let ts = [0.21248658_f64, 0.29298833];
    let ase_dist = ((ase_ci[0] - ts[0]).powi(2) + (ase_ci[1] - ts[1]).powi(2)).sqrt();
    assert!(ase_dist < 0.05, "ASE ci {ase_ci:?} distance {ase_dist}");
    assert!(ase_ci[2].abs() < 1e-8, "ASE z {}", ase_ci[2]);

    let band = BandConfig {
        force_tol: 0.05,
        max_move: 0.05,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V2,
        },
        spring: rgsaddle::SpringKind::Uniform { k: 5.0 },
        climbing: Some(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        }),
        autoneb: Some(AutoNebTuning {
            n_simul: 3,
            n_max: 7,
            local_force_tol: 0.5,
            local_max_steps: 80,
            climb_max_steps: 800,
            space_energy_ratio: 0.5,
            climb: true,
            threads: 1,
        }),
        ..BandConfig::default()
    };
    let initial = array![
        [-0.05001082, 0.46669410, 0.0],
        [0.62349940, 0.02803776, 0.0]
    ];
    let mut search = AutoNeb::new(band, initial).unwrap();
    let report = search.run(&MullerBrown).unwrap();
    let ci = report.ci_index;
    eprintln!(
        "rust status={:?} force={:.4} insertions={:?} ci={ci:?}",
        report.status,
        report.max_force,
        search.insertions(),
    );
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "force {}",
        report.max_force
    );
    let ci = report.ci_index.expect("climbing image");
    assert!(near_ts2(search.positions(), ci), "{:?}", search.positions());
    // Two endpoints and n_simul = 3: three repeated midpoints. That
    // fill is the equal spacing ASE gets from one interpolate, which
    // ASE logs as a single "max length" at 0. The later indices are
    // the dynamic inserts. ASE 3.29 FIRE and this FIRE do not share a
    // trajectory, so those indices differ; both climbing images are TS2.
    assert_eq!(search.insertions()[..3], [0, 0, 2]);
    let rust_dynamic = search.insertions()[3..].to_vec();
    assert_eq!(dynamic, vec![2, 1], "ASE 3.29 dynamic inserts");
    assert_eq!(rust_dynamic, vec![1, 4]);
    let dx = search.positions()[(ci, 0)] - ase_ci[0];
    let dy = search.positions()[(ci, 1)] - ase_ci[1];
    let ci_gap = (dx * dx + dy * dy).sqrt();
    eprintln!("ASE dynamic={dynamic:?} rust dynamic={rust_dynamic:?} ci_gap={ci_gap:.6e}");
    assert!(ci_gap < 1e-3, "climbing images differ by {ci_gap}");
}
