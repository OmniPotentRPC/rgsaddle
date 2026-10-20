//! COMPASS connect: a two-barrier toy, the C ABI, and LJ38 pairs.
//!
//! The Lennard-Jones kernel is the unshifted 12-6 potential with
//! cutoff 40, the optbench lj38_2ts_separation convention.

use std::io::Write;
use std::os::raw::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::Method;
use rgsaddle::capi::{
    RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR, RGSADDLE_OK, RgsaddleConnectConfig,
    RgsaddleConnectReport, RgsaddleSurfaceRequest, RgsaddleVersion,
};
use rgsaddle::{
    BandConfig, BandReport, BandSession, BandStatus, BandSurface, CiConfig, ConnectConfig,
    ForceGate, ProjectionKind, SaddleError, SpringKind, TangentKind, connect,
};

fn toy(x: &[f64]) -> Result<(f64, Vec<f64>), SaddleError> {
    let (a, b, c) = (x[0], x[1], x[2]);
    let u = a * a * a - 4.0 * a;
    let energy = u * u + 2.0 * b * b + 2.0 * c * c;
    let gradient = vec![2.0 * u * (3.0 * a * a - 4.0), 4.0 * b, 4.0 * c];
    Ok((energy, gradient))
}

#[test]
fn two_barrier_toy_returns_an_index_1_saddle() {
    let left = [-2.0, 0.0, 0.0];
    let right = [2.0, 0.0, 0.0];
    let report = connect(toy, &left, &right, &ConnectConfig::optbench(2_000)).unwrap();
    assert_eq!(report.kind, 's');
    assert_eq!(report.hessian_index, 1);
    assert!(report.force_norm <= 1e-3, "|g| {}", report.force_norm);
    assert!(report.evaluations < 4_000, "calls {}", report.evaluations);
    let saddle = (256.0_f64) / 27.0;
    assert!(
        (report.energy - saddle).abs() < 1e-6,
        "energy {} calls {}",
        report.energy,
        report.evaluations
    );
    assert!((report.position[0].abs() - (4.0_f64 / 3.0).sqrt()).abs() < 1e-4);
    assert!(report.position[1].abs() < 1e-4 && report.position[2].abs() < 1e-4);
}

fn lj(x: &[f64]) -> Result<(f64, Vec<f64>), SaddleError> {
    let n = x.len() / 3;
    let rc = 40.0;
    let mut energy = 0.0;
    let mut gradient = vec![0.0; x.len()];
    for i in 0..n {
        for j in i + 1..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 >= rc * rc || r2 < 1e-24 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let sr6 = inv_r2 * inv_r2 * inv_r2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * (sr12 - sr6);
            let inv_r = inv_r2.sqrt();
            let dedr = 24.0 * inv_r * (sr6 - 2.0 * sr12);
            let scale = dedr * inv_r;
            gradient[3 * i] += scale * dx;
            gradient[3 * i + 1] += scale * dy;
            gradient[3 * i + 2] += scale * dz;
            gradient[3 * j] -= scale * dx;
            gradient[3 * j + 1] -= scale * dy;
            gradient[3 * j + 2] -= scale * dz;
        }
    }
    Ok((energy, gradient))
}

fn read_xyz(path: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut coords = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i < 2 {
            continue;
        }
        let mut parts = line.split_whitespace();
        let _ = parts.next();
        coords.push(parts.next().unwrap().parse().unwrap());
        coords.push(parts.next().unwrap().parse().unwrap());
        coords.push(parts.next().unwrap().parse().unwrap());
    }
    coords
}

fn pair(index: usize) -> (Vec<f64>, Vec<f64>) {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lj38");
    (
        read_xyz(&format!("{root}/start_{index}.xyz")),
        read_xyz(&format!("{root}/end_{index}.xyz")),
    )
}

#[test]
fn lj38_pair_00_matches_the_compass_saddle() {
    let (left, right) = pair(0);
    let report = connect(lj, &left, &right, &ConnectConfig::optbench(250_000)).unwrap();
    assert_eq!(report.kind, 's');
    assert_eq!(report.hessian_index, 1);
    assert!(report.force_norm <= 1e-3);
    assert!(report.evaluations <= 250_000);
    assert!((report.energy - (-170.047586900225)).abs() < 1e-8);
}

#[test]
fn lj38_pair_39_is_not_the_index_3_point() {
    let (left, right) = pair(39);
    let report = connect(lj, &left, &right, &ConnectConfig::optbench(250_000)).unwrap();
    assert_eq!(report.kind, 's');
    assert_eq!(report.hessian_index, 1, "index {}", report.hessian_index);
    assert!(report.force_norm <= 1e-3, "|g| {}", report.force_norm);
    assert!(report.evaluations < 20_000, "calls {}", report.evaluations);
    assert!(
        (report.energy - (-168.105889260981)).abs() < 1e-6,
        "energy {}",
        report.energy
    );
    assert!((report.energy - (-167.541922837218)).abs() > 0.1);
}

extern "C" fn toy_surface(user: *mut c_void, req: *mut RgsaddleSurfaceRequest) -> i32 {
    if user.is_null() || req.is_null() {
        return -6;
    }
    let req = unsafe { &mut *req };
    let n = (req.n_images as usize) * (req.n_atoms as usize) * 3;
    let pos = unsafe { std::slice::from_raw_parts(req.positions, n) };
    let (energy, gradient) = toy(pos).unwrap();
    unsafe {
        *req.energies = energy;
        std::ptr::copy_nonoverlapping(gradient.as_ptr(), req.gradients, n);
    }
    let calls = unsafe { &mut *(user as *mut usize) };
    *calls += 1;
    RGSADDLE_OK
}

#[test]
fn c_abi_connects_the_toy() {
    let mut calls = 0usize;
    let config = RgsaddleConnectConfig {
        version: RgsaddleVersion {
            major: RGSADDLE_ABI_MAJOR,
            minor: RGSADDLE_ABI_MINOR,
        },
        flags: 0,
        force_tol: 1e-3,
        max_calls: 2_000,
        quench_nodes: 1,
    };
    let left = [-2.0, 0.0, 0.0];
    let right = [2.0, 0.0, 0.0];
    let mut position = [0.0; 3];
    let mut report = RgsaddleConnectReport {
        version: RgsaddleVersion { major: 0, minor: 0 },
        flags: 0,
        energy: 0.0,
        force_norm: 0.0,
        kind: 0,
        hessian_index: -1,
        evaluations: 0,
    };
    let status = unsafe {
        rgsaddle::capi::rgsaddle_connect(
            &config,
            1,
            left.as_ptr(),
            right.as_ptr(),
            Some(toy_surface),
            (&mut calls) as *mut usize as *mut c_void,
            position.as_mut_ptr(),
            &mut report,
        )
    };
    assert_eq!(status, RGSADDLE_OK);
    assert_eq!(report.kind, 's' as i32);
    assert_eq!(report.hessian_index, 1);
    assert_eq!(report.evaluations as usize, calls);
    assert!((report.energy - 256.0 / 27.0).abs() < 1e-6);
    assert!((position[0].abs() - (4.0 / 3.0_f64).sqrt()).abs() < 1e-4);
}

/// Full optbench sample. Release: `cargo test --release --features capi --test connect_driver -- --ignored --nocapture`.
#[test]
#[ignore]
fn lj38_fifty_one_par2() {
    let cap = 250_000usize;
    let mut scores = Vec::new();
    let mut failures = Vec::new();
    for index in 0..51 {
        let (left, right) = pair(index);
        let report = connect(lj, &left, &right, &ConnectConfig::optbench(cap)).unwrap();
        let ok = report.kind == 's'
            && report.hessian_index == 1
            && report.force_norm <= 1e-3
            && report.evaluations <= cap;
        println!(
            "pair_{index:02} calls {} energy {:.12} |g| {:.6e} kind {} index {}",
            report.evaluations, report.energy, report.force_norm, report.kind, report.hessian_index
        );
        if ok {
            scores.push(report.evaluations);
        } else {
            failures.push(index);
            scores.push(2 * cap);
        }
    }
    let mean = scores.iter().sum::<usize>() as f64 / scores.len() as f64;
    let mut ordered = scores.clone();
    ordered.sort_unstable();
    println!(
        "n {} failures {} mean {:.2} median {} max {} PAR2 {mean:.6}",
        scores.len(),
        failures.len(),
        scores.iter().filter(|c| **c <= cap).sum::<usize>() as f64
            / (scores.len() - failures.len()).max(1) as f64,
        ordered[ordered.len() / 2],
        ordered
            .iter()
            .copied()
            .filter(|c| *c <= cap)
            .max()
            .unwrap_or(0),
    );
    assert!(failures.is_empty(), "{failures:?}");
    assert!(mean < 17530.0, "PAR-2 {mean}");
}

/// Henkelman segment constants, frozen from one energy pass.
/// `k` is `k_max` at the path maximum and `k_min` at or below the
/// higher endpoint. The optbench templates use 0.97 and 9.7.
fn energy_weighted(energies: &[f64], k_min: f64, k_max: f64) -> Vec<f64> {
    let reference = energies[0].max(*energies.last().unwrap_or(&energies[0]));
    let highest = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut constants = vec![k_min; energies.len() - 1];
    let span = highest - reference;
    if span < 1e-10 {
        return constants;
    }
    for index in 1..energies.len() {
        let ei = energies[index].max(energies[index - 1]);
        if ei <= reference {
            continue;
        }
        let alpha = ((highest - ei) / span).clamp(0.0, 1.0);
        constants[index - 1] = (1.0 - alpha) * k_max + alpha * k_min;
    }
    constants
}

#[test]
fn energy_weighted_springs_match_the_segment_rule() {
    let constants = energy_weighted(&[0.0, 2.0, 5.0, 1.0], 0.97, 9.7);
    assert!((constants[0] - 3.1525).abs() < 1e-12);
    assert!((constants[1] - 9.7).abs() < 1e-12);
    assert!((constants[2] - 9.7).abs() < 1e-12);
}

fn linear_images(left: &[f64], right: &[f64], n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, left.len()));
    for image in 0..n_images {
        let t = image as f64 / (n_images - 1) as f64;
        for (col, (a, b)) in left.iter().zip(right).enumerate() {
            band[(image, col)] = (1.0 - t) * a + t * b;
        }
    }
    band
}

struct CountedLj {
    calls: AtomicUsize,
    cap: usize,
    hit_cap: AtomicBool,
}

impl CountedLj {
    fn new(cap: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            cap,
            hit_cap: AtomicBool::new(false),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    fn hit_cap(&self) -> bool {
        self.hit_cap.load(Ordering::Relaxed)
    }

    fn evaluate(&self, x: &[f64]) -> Result<(f64, Vec<f64>), SaddleError> {
        if self.calls() >= self.cap {
            self.hit_cap.store(true, Ordering::Relaxed);
            return Err(SaddleError::Surface("cap".into()));
        }
        let answer = lj(x)?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(answer)
    }
}

impl BandSurface for CountedLj {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (row, image) in positions.outer_iter().enumerate() {
            let flat = image.to_vec();
            let (energy, gradient) = self.evaluate(&flat)?;
            energies[row] = energy;
            for (col, value) in gradient.iter().enumerate() {
                gradients[(row, col)] = *value;
            }
        }
        Ok(())
    }
}

fn row_l2(row: ArrayView1<f64>) -> f64 {
    row.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn bench_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn bench_norm(v: &[f64]) -> f64 {
    bench_dot(v, v).sqrt()
}

fn bench_rigid_basis(pos: &[f64]) -> Vec<Vec<f64>> {
    let n = pos.len() / 3;
    if n < 2 {
        return Vec::new();
    }
    let mut center = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            center[axis] += pos[3 * atom + axis];
        }
    }
    for axis in 0..3 {
        center[axis] /= n as f64;
    }
    let mut seeds = Vec::new();
    for axis in 0..3 {
        let mut translation = vec![0.0; pos.len()];
        for atom in 0..n {
            translation[3 * atom + axis] = 1.0;
        }
        seeds.push(translation);
        let mut rotation = vec![0.0; pos.len()];
        for atom in 0..n {
            let p = [
                pos[3 * atom] - center[0],
                pos[3 * atom + 1] - center[1],
                pos[3 * atom + 2] - center[2],
            ];
            let rot = match axis {
                0 => [0.0, p[2], -p[1]],
                1 => [-p[2], 0.0, p[0]],
                _ => [p[1], -p[0], 0.0],
            };
            for k in 0..3 {
                rotation[3 * atom + k] = rot[k];
            }
        }
        seeds.push(rotation);
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for seed in seeds {
        let mut mode = seed;
        for kept in &basis {
            let scale = bench_dot(kept, &mode);
            for (value, kept_value) in mode.iter_mut().zip(kept) {
                *value -= scale * kept_value;
            }
        }
        let norm = bench_norm(&mode);
        if norm > 1e-8 {
            for value in &mut mode {
                *value /= norm;
            }
            basis.push(mode);
        }
    }
    basis
}

fn bench_project_out(v: &[f64], modes: &[Vec<f64>]) -> Vec<f64> {
    let mut out = v.to_vec();
    for mode in modes {
        let scale = bench_dot(mode, &out);
        for (value, mode_value) in out.iter_mut().zip(mode) {
            *value -= scale * mode_value;
        }
    }
    out
}

fn bench_project_rigid(hess: &[Vec<f64>], modes: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = hess.len();
    if modes.is_empty() {
        return hess.to_vec();
    }
    let mut out = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut col = vec![0.0; n];
        for (row, value) in col.iter_mut().enumerate() {
            *value = hess[row][column];
        }
        let projected = bench_project_out(&col, modes);
        for (row, value) in projected.iter().enumerate() {
            out[row][column] = *value;
        }
    }
    for row in 0..n {
        out[row] = bench_project_out(&out[row], modes);
    }
    for mode in modes {
        for i in 0..n {
            for j in 0..n {
                out[i][j] += mode[i] * mode[j];
            }
        }
    }
    out
}

/// Verification index. These finite-difference calls are not added to
/// the band's force-call count, matching the optbench harness.
fn verification_index(pos: &[f64]) -> i32 {
    let n = pos.len();
    let step = 1e-5;
    let mut hess = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut plus = pos.to_vec();
        let mut minus = pos.to_vec();
        plus[i] += step;
        minus[i] -= step;
        let gp = lj(&plus).expect("plus gradient").1;
        let gm = lj(&minus).expect("minus gradient").1;
        for (j, column) in hess.iter_mut().enumerate() {
            column[i] = (gp[j] - gm[j]) / (2.0 * step);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = mean;
            hess[j][i] = mean;
        }
    }
    let projected = bench_project_rigid(&hess, &bench_rigid_basis(pos));
    let mut matrix = faer::Mat::<f64>::zeros(n, n);
    for (i, row) in projected.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            matrix[(i, j)] = *value;
        }
    }
    let eig = matrix
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("verification eigh");
    let mut index = 0;
    for k in 0..n {
        if eig.S()[k] < -1e-6 {
            index += 1;
        }
    }
    index
}

fn candidate_index(report: &BandReport, energies: ArrayView1<f64>) -> usize {
    if let Some(ci) = report.ci_index {
        return ci;
    }
    let mut best = 1;
    let mut energy = energies[1];
    for index in 2..energies.len() - 1 {
        if energies[index] > energy {
            energy = energies[index];
            best = index;
        }
    }
    best
}

struct BandRow {
    index: usize,
    template: &'static str,
    calls: usize,
    energy: f64,
    force_norm: f64,
    hessian_index: i32,
    failure: &'static str,
}

impl BandRow {
    fn unfinished(
        index: usize,
        template: &'static str,
        calls: usize,
        failure: &'static str,
    ) -> Self {
        Self {
            index,
            template,
            calls,
            energy: f64::NAN,
            force_norm: f64::NAN,
            hessian_index: -1,
            failure,
        }
    }
}

const BAND_CAP: usize = 250_000;

fn run_band(pair_index: usize, weighted: bool) -> BandRow {
    let template = if weighted {
        "ew-ci-neb-lbfgs"
    } else {
        "ci-neb-lbfgs"
    };
    let (left, right) = pair(pair_index);
    let images = linear_images(&left, &right, 7);
    let surface = CountedLj::new(BAND_CAP);
    let spring = if weighted {
        let mut energies = Vec::with_capacity(images.nrows());
        for row in images.outer_iter() {
            match surface.evaluate(&row.to_vec()) {
                Ok((energy, _)) => energies.push(energy),
                Err(_) => {
                    return BandRow::unfinished(pair_index, template, surface.calls(), "cap");
                }
            }
        }
        SpringKind::Weighted {
            ks: energy_weighted(&energies, 0.97, 9.7),
        }
    } else {
        SpringKind::Uniform { k: 5.0 }
    };
    let config = BandConfig {
        tangent: TangentKind::Improved,
        spring,
        projection: ProjectionKind::Neb,
        climbing: Some(CiConfig::after_rel(0.5)),
        force_tol: 1e-3,
        max_move: 0.05,
        method: Method::Lbfgs { memory: 100 },
        ..BandConfig::default()
    };
    let mut session = match BandSession::new(config, images) {
        Ok(session) => session,
        Err(_) => return BandRow::unfinished(pair_index, template, surface.calls(), "error"),
    };
    session.set_force_gate(ForceGate::L2Norm);
    let mut last = None;
    let mut crashed = false;
    loop {
        if surface.calls() >= BAND_CAP || surface.hit_cap() {
            break;
        }
        if last
            .as_ref()
            .is_some_and(|report: &BandReport| report.iteration >= BAND_CAP)
        {
            break;
        }
        match session.step(&surface) {
            Ok(report) => {
                let early = report.ci_index.is_some_and(|ci| {
                    session
                        .evaluation()
                        .is_some_and(|evaluation| row_l2(evaluation.gradients.row(ci)) <= 1e-3)
                });
                let converged = report.status == BandStatus::Converged;
                last = Some(report);
                if converged || early {
                    break;
                }
            }
            Err(SaddleError::Stalled(_)) => break,
            Err(_) => {
                crashed = !surface.hit_cap() && surface.calls() < BAND_CAP;
                break;
            }
        }
    }
    let calls = surface.calls();
    if crashed {
        return BandRow::unfinished(pair_index, template, calls, "error");
    }
    if calls >= BAND_CAP || surface.hit_cap() {
        return scored_or_cap(pair_index, template, calls, &session, last.as_ref(), "cap");
    }
    let Some(report) = last else {
        return BandRow::unfinished(pair_index, template, calls, "error");
    };
    score_band(pair_index, template, calls, &session, &report)
}

fn scored_or_cap(
    pair_index: usize,
    template: &'static str,
    calls: usize,
    session: &BandSession,
    report: Option<&BandReport>,
    failure: &'static str,
) -> BandRow {
    let Some(report) = report else {
        return BandRow::unfinished(pair_index, template, calls, failure);
    };
    let Some(evaluation) = session.evaluation() else {
        return BandRow::unfinished(pair_index, template, calls, failure);
    };
    let ci = candidate_index(report, evaluation.energies.view());
    BandRow {
        index: pair_index,
        template,
        calls,
        energy: evaluation.energies[ci],
        force_norm: row_l2(evaluation.gradients.row(ci)),
        hessian_index: -1,
        failure,
    }
}

fn score_band(
    pair_index: usize,
    template: &'static str,
    calls: usize,
    session: &BandSession,
    report: &BandReport,
) -> BandRow {
    let Some(evaluation) = session.evaluation() else {
        return BandRow::unfinished(pair_index, template, calls, "error");
    };
    let ci = candidate_index(report, evaluation.energies.view());
    let force_norm = row_l2(evaluation.gradients.row(ci));
    let energy = evaluation.energies[ci];
    if !force_norm.is_finite() || force_norm > 1e-3 {
        return BandRow {
            index: pair_index,
            template,
            calls,
            energy,
            force_norm,
            hessian_index: -1,
            failure: "force",
        };
    }
    let point = session.positions().row(ci).to_vec();
    let hessian_index = verification_index(&point);
    BandRow {
        index: pair_index,
        template,
        calls,
        energy,
        force_norm,
        hessian_index,
        failure: if hessian_index == 1 { "ok" } else { "index" },
    }
}

fn emit(line: String) {
    static PRINT: Mutex<()> = Mutex::new(());
    let _guard = PRINT.lock().unwrap_or_else(|err| err.into_inner());
    println!("{line}");
    let _ = std::io::stdout().flush();
}

fn sample_template(weighted: bool) -> Vec<BandRow> {
    let threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let rows = Mutex::new((0..20).map(|_| None).collect::<Vec<Option<BandRow>>>());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= 20 {
                        break;
                    }
                    let started = std::time::Instant::now();
                    let row = run_band(index, weighted);
                    emit(format!(
                        "{} pair_{:02} calls {} energy {:.12} |g| {:.6e} index {} failure {} ms {}",
                        row.template,
                        row.index,
                        row.calls,
                        row.energy,
                        row.force_norm,
                        row.hessian_index,
                        row.failure,
                        started.elapsed().as_millis()
                    ));
                    rows.lock().unwrap_or_else(|err| err.into_inner())[index] = Some(row);
                }
            });
        }
    });
    rows.into_inner()
        .unwrap_or_else(|err| err.into_inner())
        .into_iter()
        .map(|row| row.expect("pair"))
        .collect()
}

fn summarize(rows: &[BandRow]) {
    let n = rows.len();
    let failures = rows.iter().filter(|row| row.failure != "ok").count();
    let mut success: Vec<usize> = rows
        .iter()
        .filter(|row| row.failure == "ok")
        .map(|row| row.calls)
        .collect();
    success.sort_unstable();
    let penalty = 2 * BAND_CAP;
    let par2 = rows
        .iter()
        .map(|row| {
            if row.failure == "ok" {
                row.calls
            } else {
                penalty
            }
        })
        .sum::<usize>() as f64
        / n as f64;
    let (min, median, mean, max) = if success.is_empty() {
        (0, 0, 0.0, 0)
    } else {
        (
            success[0],
            success[success.len() / 2],
            success.iter().sum::<usize>() as f64 / success.len() as f64,
            *success.last().unwrap_or(&0),
        )
    };
    emit(format!(
        "{} n {n} failures {failures} success_min {min} success_median {median} success_mean {mean:.2} success_max {max} PAR2 {par2:.6}",
        rows.first().map(|row| row.template).unwrap_or("band")
    ));
}

/// Optbench CI-NEB templates on pairs 0..19. Five interior images,
/// linear path, L-BFGS memory 100, L2 gate, max move 0.05, climbing
/// armed at half the first force. The weighted template freezes
/// Henkelman springs from the linear images. A call at the cap is a
/// failure. The verification Hessian is not counted.
/// Release: `cargo test --release --features capi --test connect_driver lj38_band_templates_twenty -- --ignored --nocapture`.
#[test]
#[ignore]
fn lj38_band_templates_twenty() {
    for weighted in [false, true] {
        let rows = sample_template(weighted);
        assert_eq!(rows.len(), 20);
        summarize(&rows);
    }
}
