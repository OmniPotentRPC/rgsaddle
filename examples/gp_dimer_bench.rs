// Copyright (c) rgsaddle developers
//!
//! Force calls and wall time of the plain dimer against the opt-in
//! GP dimer. The translator on both arms is FIRE, the min-mode default.
//! Each system is started from at least ten seeds. A pair lands on the
//! same saddle when both searches converge, the energies agree, and the
//! inverse-distance gap between the geometries is small. That gap is
//! blind to a shared rotation or translation.
//!
//! ```text
//! cargo run --release --example gp_dimer_bench
//! ```
//!
//! `RGSADDLE_BENCH_SYSTEM` selects one system. `RGSADDLE_BENCH_SEEDS`
//! sets the seed count (default 10).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    GpDimerConfig, GpDimerSession, GpDimerStatus, GpStepTiming, MinModeConfig, MinModeKind,
    MinModeSession, MinModeStatus, PointSurface, SaddleError, finite_difference_index,
    max_log_pair_distance,
};

const SEEDS: usize = 10;
const ENERGY_TOL: f64 = 1e-4;
const GAP_TOL: f64 = 1e-3;
const BASE_SEED: u64 = 20261111;

fn main() {
    let seeds = std::env::var("RGSADDLE_BENCH_SEEDS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(SEEDS);
    let only = std::env::var("RGSADDLE_BENCH_SYSTEM").ok();
    println!("seed\tsystem\tmethod\tcalls\tmillis\tconverged\tenergy\tmax_force\tcurvature");
    for system in systems() {
        if only.as_ref().is_some_and(|name| name != system.name) {
            continue;
        }
        let mut plain_calls = Vec::new();
        let mut gp_calls = Vec::new();
        let mut plain_ms = Vec::new();
        let mut gp_ms = Vec::new();
        let mut matched_plain = Vec::new();
        let mut matched_gp = Vec::new();
        let mut matched_plain_ms = Vec::new();
        let mut matched_gp_ms = Vec::new();
        let mut same = 0usize;
        let mut both = 0usize;
        for index in 0..seeds {
            let seed = mix_seed(index);
            let plain = search_plain(&system, seed);
            let (gp, timings) = search_gp(&system, seed);
            print_row(system.name, seed, "dimer", &plain);
            print_row(system.name, seed, "gp-dimer", &gp);
            if let Some(certificate) = certify(&system, &gp) {
                println!(
                    "certificate\t{name}\t{seed}\tgp-dimer\tindex {index}\tknown {known}\tlowest {lowest:.6}\tgap {gap:.6e}",
                    name = system.name,
                    index = certificate.0,
                    known = certificate.1,
                    lowest = certificate.2,
                    gap = certificate.3,
                );
            }
            let mut sizes: Vec<usize> = timings.iter().map(|timing| timing.observations).collect();
            sizes.sort_unstable();
            sizes.dedup();
            for n in sizes {
                let samples: Vec<f64> = timings
                    .iter()
                    .filter(|timing| timing.observations == n)
                    .map(|timing| timing.fit_nanos as f64 / 1.0e3)
                    .collect();
                let rank_one = timings
                    .iter()
                    .filter(|timing| timing.observations == n && timing.rank_one)
                    .count();
                let width = timings
                    .iter()
                    .find(|timing| timing.observations == n)
                    .map(|timing| timing.width)
                    .unwrap_or(0);
                let count = samples.len();
                println!(
                    "timing\t{name}\tseed {seed}\tn {n}\twidth {width}\tfit_us {}\trank_one {rank_one}/{count}",
                    band_owned(samples),
                    name = system.name,
                );
            }
            if plain.converged {
                plain_calls.push(plain.calls as f64);
                plain_ms.push(plain.millis);
            }
            if gp.converged {
                gp_calls.push(gp.calls as f64);
                gp_ms.push(gp.millis);
            }
            if plain.converged && gp.converged {
                both += 1;
                let gap = geometry_gap(&system, &plain.position, &gp.position);
                let close = (plain.energy - gp.energy).abs() <= ENERGY_TOL && gap <= GAP_TOL;
                if close {
                    same += 1;
                    matched_plain.push(plain.calls as f64);
                    matched_gp.push(gp.calls as f64);
                    matched_plain_ms.push(plain.millis);
                    matched_gp_ms.push(gp.millis);
                } else {
                    eprintln!(
                        "miss {name} seed {seed} dE {:.6e} gap {gap:.6e} calls {pc}/{gc}",
                        (plain.energy - gp.energy).abs(),
                        name = system.name,
                        pc = plain.calls,
                        gc = gp.calls,
                    );
                }
            } else {
                eprintln!(
                    "unconverged {name} seed {seed} dimer {dp} gp {gp}",
                    name = system.name,
                    dp = plain.converged,
                    gp = gp.converged,
                );
            }
        }
        println!(
            "summary\t{name}\tseeds {seeds}\tboth {both}\tsame {same}\tdimer_calls {dc}\tgp_calls {gc}\tdimer_ms {dm}\tgp_ms {gm}\tmatched_dimer_calls {mdc}\tmatched_gp_calls {mgc}\tmatched_dimer_ms {mdm}\tmatched_gp_ms {mgm}",
            name = system.name,
            dc = band(&mut plain_calls),
            gc = band(&mut gp_calls),
            dm = band(&mut plain_ms),
            gm = band(&mut gp_ms),
            mdc = band(&mut matched_plain),
            mgc = band(&mut matched_gp),
            mdm = band(&mut matched_plain_ms),
            mgm = band(&mut matched_gp_ms),
        );
    }
}

fn print_row(system: &str, seed: u64, method: &str, outcome: &Outcome) {
    println!(
        "{seed}\t{system}\t{method}\t{calls}\t{millis:.3}\t{converged}\t{energy:.8}\t{force:.6e}\t{curvature:.6}",
        calls = outcome.calls,
        millis = outcome.millis,
        converged = outcome.converged,
        energy = outcome.energy,
        force = outcome.max_force,
        curvature = outcome.curvature,
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

/// Median, then the interquartile spread `q3 - q1`.
fn band(values: &mut [f64]) -> String {
    if values.is_empty() {
        return "n=0".to_string();
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    let q = |p: f64| values[((n as f64 - 1.0) * p).round() as usize];
    format!("n={n} median {:.4} spread {:.4}", q(0.5), q(0.75) - q(0.25))
}

struct Outcome {
    calls: usize,
    millis: f64,
    converged: bool,
    energy: f64,
    position: Array1<f64>,
    max_force: f64,
    curvature: f64,
}

struct System {
    name: &'static str,
    steps: usize,
    max_move: f64,
    start: fn(u64) -> (Array1<f64>, Array1<f64>),
    kind: BenchSurface,
}

enum BenchSurface {
    LennardJones { cutoff: f64, cell: Option<f64> },
    Muller,
    Morse(MorseSlab),
}

fn step_budget(default: usize) -> usize {
    std::env::var("RGSADDLE_BENCH_STEPS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn systems() -> Vec<System> {
    let vacancy = vacancy_cell();
    let morse = MorseSlab::load();
    vec![
        System {
            name: "lj13",
            steps: step_budget(600),
            max_move: 0.2,
            start: lj13_start,
            kind: BenchSurface::LennardJones {
                cutoff: 15.0,
                cell: None,
            },
        },
        System {
            name: "muller-brown",
            steps: step_budget(600),
            max_move: 0.2,
            start: muller_start,
            kind: BenchSurface::Muller,
        },
        System {
            name: "morse-adatom",
            steps: step_budget(400),
            max_move: 0.15,
            start: muller_start,
            kind: BenchSurface::Morse(morse),
        },
        System {
            name: "vacancy-hop",
            steps: step_budget(400),
            max_move: 0.2,
            start: vacancy_start,
            kind: BenchSurface::LennardJones {
                cutoff: 2.5,
                cell: Some(vacancy),
            },
        },
    ]
}

fn search_plain(system: &System, seed: u64) -> Outcome {
    let (start, mode) = initial(system, seed);
    let config = MinModeConfig {
        kind: MinModeKind::Dimer,
        force_tol: 1e-3,
        max_move: system.max_move,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, start, mode).expect("session");
    let surface = Counting {
        kind: &system.kind,
        calls: AtomicUsize::new(0),
    };
    let started = Instant::now();
    let mut converged = false;
    let mut max_force = f64::NAN;
    let mut curvature = f64::NAN;
    for _ in 0..system.steps {
        match session.step(&surface) {
            Ok(report) => {
                max_force = report.max_force;
                curvature = report.curvature;
                if report.status == MinModeStatus::Converged {
                    converged = true;
                    break;
                }
            }
            Err(err) => {
                eprintln!("dimer step failed: {err}");
                break;
            }
        }
    }
    finish(
        &surface,
        session.position().to_owned(),
        started,
        converged,
        max_force,
        curvature,
    )
}

fn band_owned(mut values: Vec<f64>) -> String {
    band(&mut values)
}

fn certify(system: &System, outcome: &Outcome) -> Option<(usize, bool, f64, f64)> {
    if !outcome.converged {
        return None;
    }
    let project = system.name == "lj13";
    let surface = RawSurface { kind: &system.kind };
    let index = finite_difference_index(&surface, outcome.position.view(), project).ok()?;
    let reference = reference_geometry(system);
    let known = matches_known(system, &outcome.position, outcome.energy, &reference);
    let gap = geometry_gap(system, &outcome.position, &reference);
    Some((index.negative, known, index.lowest, gap))
}

fn matches_known(
    system: &System,
    position: &Array1<f64>,
    energy: f64,
    reference: &Array1<f64>,
) -> bool {
    let Ok((reference_energy, _)) = RawSurface { kind: &system.kind }.eval(reference.view()) else {
        return false;
    };
    if (energy - reference_energy).abs() > ENERGY_TOL {
        return false;
    }
    geometry_gap(system, position, reference) <= GAP_TOL
}

fn reference_geometry(system: &System) -> Array1<f64> {
    match &system.kind {
        BenchSurface::Muller => ndarray::array![-0.8220015587, 0.6243128025, 0.0],
        BenchSurface::Morse(slab) => slab.saddle.clone(),
        BenchSurface::LennardJones { cell: None, .. } => lj13_reference(),
        BenchSurface::LennardJones { cell: Some(_), .. } => vacancy_reference(),
    }
}

struct RawSurface<'a> {
    kind: &'a BenchSurface,
}

impl PointSurface for RawSurface<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(match self.kind {
            BenchSurface::LennardJones { cutoff, cell } => lj_energy_gradient(x, *cutoff, *cell),
            BenchSurface::Muller => muller_energy_gradient(x),
            BenchSurface::Morse(slab) => slab.energy_gradient(x),
        })
    }
}

fn search_gp(system: &System, seed: u64) -> (Outcome, Vec<GpStepTiming>) {
    let (start, mode) = initial(system, seed);
    let (cell, fixed) = match &system.kind {
        BenchSurface::LennardJones { cell, .. } => {
            (cell.map(|length| [length, length, length]), None)
        }
        BenchSurface::Muller => (None, None),
        BenchSurface::Morse(slab) => (Some(slab.cell), Some(slab.fixed.clone())),
    };
    let config = GpDimerConfig {
        inner: MinModeConfig {
            force_tol: 1e-3,
            max_move: system.max_move,
            ..MinModeConfig::default()
        },
        cell,
        fixed,
        ..GpDimerConfig::default()
    };
    let mut session = GpDimerSession::new(config, start, mode).expect("gp session");
    let surface = Counting {
        kind: &system.kind,
        calls: AtomicUsize::new(0),
    };
    let started = Instant::now();
    let mut converged = false;
    let mut max_force = f64::NAN;
    let mut curvature = f64::NAN;
    for _ in 0..system.steps {
        match session.step(&surface) {
            Ok(report) => {
                max_force = report.max_force;
                curvature = report.curvature;
                if report.status == GpDimerStatus::Converged {
                    converged = true;
                    break;
                }
            }
            Err(err) => {
                eprintln!("gp-dimer step failed: {err}");
                break;
            }
        }
    }
    let outcome = finish(
        &surface,
        session.position().to_owned(),
        started,
        converged,
        max_force,
        curvature,
    );
    (outcome, session.step_timings().to_vec())
}

fn initial(system: &System, seed: u64) -> (Array1<f64>, Array1<f64>) {
    if let BenchSurface::Morse(slab) = &system.kind {
        slab.start_at(seed)
    } else {
        (system.start)(seed)
    }
}

fn finish(
    surface: &Counting<'_>,
    position: Array1<f64>,
    started: Instant,
    converged: bool,
    max_force: f64,
    curvature: f64,
) -> Outcome {
    let millis = started.elapsed().as_secs_f64() * 1.0e3;
    let energy = match surface.energy(position.view()) {
        Ok(energy) => energy,
        Err(err) => {
            eprintln!("final energy failed: {err}");
            f64::NAN
        }
    };
    // The trailing energy read is not a search call.
    let calls = surface.calls.load(Ordering::Relaxed).saturating_sub(1);
    Outcome {
        calls,
        millis,
        converged,
        energy,
        position,
        max_force,
        curvature,
    }
}

fn geometry_gap(system: &System, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let (cell, fixed) = match &system.kind {
        BenchSurface::LennardJones { cell, .. } => (cell.map(|l| [l, l, l]), None),
        BenchSurface::Muller => (None, None),
        BenchSurface::Morse(slab) => (Some(slab.cell), Some(slab.fixed.view())),
    };
    max_log_pair_distance(a.view(), b.view(), cell, fixed).unwrap_or(f64::INFINITY)
}

struct Counting<'a> {
    kind: &'a BenchSurface,
    calls: AtomicUsize,
}

impl Counting<'_> {
    fn energy(&self, x: ArrayView1<f64>) -> Result<f64, SaddleError> {
        Ok(self.eval(x)?.0)
    }
}

impl PointSurface for Counting<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(match self.kind {
            BenchSurface::LennardJones { cutoff, cell } => lj_energy_gradient(x, *cutoff, *cell),
            BenchSurface::Muller => muller_energy_gradient(x),
            BenchSurface::Morse(slab) => slab.energy_gradient(x),
        })
    }
}

struct MorseSlab {
    cell: [f64; 3],
    rest: Vec<[f64; 3]>,
    free_index: usize,
    fixed: Array1<f64>,
    saddle: Array1<f64>,
    mode: Array1<f64>,
}

impl MorseSlab {
    fn load() -> Self {
        let path = format!(
            "{}/tests/fixtures/irc/pt_saddle.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(path).expect("pt saddle");
        let mut lines = text.lines();
        let head: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let mut rest = Vec::new();
        let mut free_index = 0usize;
        let mut saddle = Vec::new();
        let mut mode = Vec::new();
        let mut fixed = Vec::new();
        for line in lines {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|value| value.parse().unwrap())
                .collect();
            let xyz = [v[2], v[3], v[4]];
            if v[0] == 0.0 {
                free_index = rest.len();
                saddle.extend_from_slice(&xyz);
                mode.extend_from_slice(&v[5..8]);
            } else {
                fixed.extend_from_slice(&xyz);
            }
            rest.push(xyz);
        }
        Self {
            cell: [head[2], head[3], head[4]],
            rest,
            free_index,
            fixed: Array1::from(fixed),
            saddle: Array1::from(saddle),
            mode: Array1::from(mode),
        }
    }

    fn start_at(&self, seed: u64) -> (Array1<f64>, Array1<f64>) {
        let (position, mode) = offset_start(
            self.saddle.as_slice().unwrap(),
            self.mode.as_slice().unwrap(),
            0.2,
            0.05,
            seed,
        );
        (Array1::from(position), Array1::from(mode))
    }

    fn energy_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut atoms = self.rest.clone();
        atoms[self.free_index] = [x[0], x[1], x[2]];
        let rc = 9.5;
        let (shift, _) = morse_pair(rc * rc);
        let mut energy = 0.0;
        let mut gradient = [0.0; 3];
        let i = self.free_index;
        for (j, other) in atoms.iter().enumerate() {
            if j == i {
                continue;
            }
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = atoms[i][k] - other[k];
                d[k] -= self.cell[k] * (d[k] / self.cell[k]).round();
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 >= rc * rc {
                continue;
            }
            let (u, fscale) = morse_pair(r2);
            energy += u - shift;
            for k in 0..3 {
                gradient[k] += -fscale * d[k];
            }
        }
        (energy, Array1::from_vec(gradient.to_vec()))
    }
}

fn morse_pair(r2: f64) -> (f64, f64) {
    let (de, alpha, re) = (0.7102, 1.6047, 2.897);
    let r = r2.sqrt();
    let d = 1.0 - (-alpha * (r - re)).exp();
    (de * d * d - de, 2.0 * de * alpha * d * (d - 1.0) / r)
}

fn lj_energy_gradient(x: ArrayView1<f64>, cutoff: f64, cell: Option<f64>) -> (f64, Array1<f64>) {
    let (shift, _) = lj_pair(cutoff * cutoff);
    let n = x.len() / 3;
    let mut energy = 0.0;
    let mut gradient = vec![0.0; x.len()];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = x[3 * i + k] - x[3 * j + k];
                if let Some(cell) = cell {
                    d[k] -= cell * (d[k] / cell).round();
                }
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 >= cutoff * cutoff || r2 == 0.0 {
                continue;
            }
            let (potential, fscale) = lj_pair(r2);
            energy += potential - shift;
            for k in 0..3 {
                let g = -fscale * d[k];
                gradient[3 * i + k] += g;
                gradient[3 * j + k] -= g;
            }
        }
    }
    (energy, Array1::from(gradient))
}

fn lj_pair(r2: f64) -> (f64, f64) {
    let a = (1.0 / r2).powi(3);
    let b = 4.0 * a;
    (b * (a - 1.0), 6.0 * b / r2 * (2.0 * a - 1.0))
}

const PREFACTOR: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const XX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const XY: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const YY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, f64, f64) {
    let (mut energy, mut gx, mut gy) = (0.0, 0.0, 0.0);
    for i in 0..4 {
        let (dx, dy) = (x - X0[i], y - Y0[i]);
        let term = PREFACTOR[i] * (XX[i] * dx * dx + XY[i] * dx * dy + YY[i] * dy * dy).exp();
        energy += term;
        gx += term * (2.0 * XX[i] * dx + XY[i] * dy);
        gy += term * (XY[i] * dx + 2.0 * YY[i] * dy);
    }
    (energy, gx, gy)
}

fn muller_energy_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (energy_xy, gx, gy) = muller_brown(x[0], x[1]);
    let energy = energy_xy + 2.0 * x[2] * x[2];
    (energy, Array1::from_vec(vec![gx, gy, 4.0 * x[2]]))
}

fn lj13_reference() -> Array1<f64> {
    let path = format!(
        "{}/tests/fixtures/irc/lj13_saddle.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("lj13 saddle");
    let mut position = Vec::new();
    for line in text.lines().skip(1) {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        if v[0] == 0.0 {
            position.extend_from_slice(&v[2..5]);
        }
    }
    Array1::from(position)
}

fn vacancy_reference() -> Array1<f64> {
    let (position, _) = vacancy_geometry();
    Array1::from(position)
}

fn lj13_start(seed: u64) -> (Array1<f64>, Array1<f64>) {
    let path = format!(
        "{}/tests/fixtures/irc/lj13_saddle.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("lj13 saddle");
    let mut lines = text.lines();
    let _head = lines.next();
    let mut position = Vec::new();
    let mut mode = Vec::new();
    for line in lines {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        if v[0] == 0.0 {
            position.extend_from_slice(&v[2..5]);
            mode.extend_from_slice(&v[5..8]);
        }
    }
    let (mut position, mode) = offset_start(&position, &mode, 0.12, 0.04, BASE_SEED);
    let kick = mix(position.len(), seed);
    for (coord, kick) in position.iter_mut().zip(kick) {
        *coord += 0.002 * kick;
    }
    (Array1::from(position), Array1::from(mode))
}

fn vacancy_cell() -> f64 {
    let n = 4.0_f64;
    let spacing = 2.0_f64.powf(1.0 / 6.0) * 2.0_f64.sqrt();
    n * spacing
}

fn vacancy_geometry() -> (Vec<f64>, Vec<f64>) {
    let n = 4usize;
    let spacing = 2.0_f64.powf(1.0 / 6.0) * 2.0_f64.sqrt();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut atoms = Vec::new();
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for site in basis {
                    atoms.push([
                        (ix as f64 + site[0]) * spacing,
                        (iy as f64 + site[1]) * spacing,
                        (iz as f64 + site[2]) * spacing,
                    ]);
                }
            }
        }
    }
    atoms.remove(0);
    let neighbor = atoms[0];
    let mut mode = Vec::new();
    let mut position = Vec::new();
    for atom in &atoms {
        if *atom == neighbor {
            position.extend_from_slice(&[neighbor[0] * 0.5, neighbor[1] * 0.5, neighbor[2] * 0.5]);
            mode.extend_from_slice(&neighbor);
        } else {
            position.extend_from_slice(atom);
            mode.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }
    (position, mode)
}

fn vacancy_start(seed: u64) -> (Array1<f64>, Array1<f64>) {
    let (position, mode) = vacancy_geometry();
    let (position, mode) = offset_start(&position, &mode, 0.0, 0.008, seed);
    (Array1::from(position), Array1::from(mode))
}

fn muller_start(seed: u64) -> (Array1<f64>, Array1<f64>) {
    let kick = mix(3, seed);
    let position = vec![-0.6 + 0.08 * kick[0], 0.8 + 0.08 * kick[1], 0.05 * kick[2]];
    let mode = unit(&[-1.0 + 0.3 * kick[1], 0.4 + 0.3 * kick[2], 0.05 * kick[0]]);
    (Array1::from(position), Array1::from(mode))
}

fn offset_start(
    origin: &[f64],
    mode: &[f64],
    along: f64,
    aside: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mode = unit(mode);
    let perp = unit(&perpendicular(&mix(mode.len(), seed), &mode));
    let position = origin
        .iter()
        .zip(mode.iter().zip(perp.iter()))
        .map(|(x, (m, p))| x + along * m + aside * p)
        .collect();
    (position, mode)
}

fn mix_seed(index: usize) -> u64 {
    BASE_SEED.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn mix(n: usize, mut seed: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = ((seed >> 33) as u32) as f64 / u32::MAX as f64;
        out.push(unit - 0.5);
    }
    out
}

fn perpendicular(v: &[f64], mode: &[f64]) -> Vec<f64> {
    let dot = v.iter().zip(mode.iter()).map(|(a, b)| a * b).sum::<f64>();
    v.iter()
        .zip(mode.iter())
        .map(|(a, b)| a - dot * b)
        .collect()
}

fn unit(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|c| c * c).sum::<f64>().sqrt();
    v.iter().map(|c| c / norm).collect()
}
