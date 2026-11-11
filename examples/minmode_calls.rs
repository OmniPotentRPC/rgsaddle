// Copyright (c) rgsaddle developers
//!
//! Force calls and wall time of one minimum-mode search with the
//! stiff-mode FIRE step off and on. Off is FIRE at its opening
//! timestep. On, after two thousand steps in which neither the force
//! nor the energy drops and the curvature is steeper than -25, a
//! probe along the reflected force takes a step of length
//! `|F|/slope`, at most 0.05, and keeps doing so while the curvature
//! stays that steep. L-BFGS does not read the switch. Each system is
//! started from twelve
//! geometry seeds. A pair is comparable only when both searches converge
//! and land on the same saddle: the energies agree to 1e-6 and the
//! aligned root-mean-square distance is under 1e-3. The printed calls
//! and milliseconds are the median of those matched pairs, with the
//! interquartile range beside it. The call count leaves out the energy
//! read taken after the search. A full run writes
//! `validation/minmode_translator.csv`.
//!
//! ```text
//! cargo run --release --example minmode_calls
//! ```
//!
//! `RGSADDLE_BENCH_SYSTEM`, `RGSADDLE_BENCH_KIND`, and
//! `RGSADDLE_BENCH_METHOD` select one row. `RGSADDLE_BENCH_SEEDS` sets
//! how many seeds to run. The study below uses every system and twelve
//! seeds.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

const BASE_SEED: u64 = 20261111;
const SEEDS: usize = 12;
/// Same saddle: energies within this, and the aligned distance below
/// [`RMSD_TOL`].
const ENERGY_TOL: f64 = 1e-6;
const RMSD_TOL: f64 = 1e-3;

fn main() {
    kabsch_self_check();
    let seeds = std::env::var("RGSADDLE_BENCH_SEEDS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(SEEDS);
    let only = std::env::var("RGSADDLE_BENCH_SYSTEM").ok();
    let only_kind = std::env::var("RGSADDLE_BENCH_KIND").ok();
    let only_method = std::env::var("RGSADDLE_BENCH_METHOD").ok();
    let translator = std::env::var("RGSADDLE_BENCH_TRANSLATOR").ok();
    let skip_off = std::env::var("RGSADDLE_BENCH_SKIP_OFF").ok().as_deref() == Some("1");
    let systems = systems();
    let mut csv = vec![
        "system,kind,method,seed,off_converged,on_converged,off_calls,on_calls,off_ms,on_ms,off_energy,on_energy,off_force,on_force,off_curvature,on_curvature,rmsd,matched"
            .to_string(),
    ];
    println!("seeds {seeds} base {}", bench_base());
    println!("system\tkind\tmethod\tmatched\tcalls off\tcalls on\tms off\tms on\tdiverged");
    for system in &systems {
        if only.as_ref().is_some_and(|name| name != system.name) {
            continue;
        }
        for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
            if only_kind
                .as_ref()
                .is_some_and(|name| name != kind_label(kind))
            {
                continue;
            }
            if let Some(name) = &translator {
                compare_translator(system, kind, name, seeds);
                continue;
            }
            for method in [MethodKind::Fire, MethodKind::Lbfgs] {
                if only_method
                    .as_ref()
                    .is_some_and(|name| name != method.label())
                {
                    continue;
                }
                let mut matched_calls_off = Vec::new();
                let mut matched_calls_on = Vec::new();
                let mut matched_ms_off = Vec::new();
                let mut matched_ms_on = Vec::new();
                let mut diverged = 0usize;
                for index in 0..seeds {
                    let seed = mix_seed(index);
                    let (start, mode) = (system.start)(seed);
                    let off = if skip_off {
                        Outcome {
                            calls: 0,
                            millis: 0.0,
                            converged: false,
                            energy: f64::NAN,
                            position: Vec::new(),
                            max_force: f64::NAN,
                            curvature: f64::NAN,
                        }
                    } else {
                        search(
                            system,
                            kind,
                            Arm {
                                method,
                                curvature: false,
                                cg: false,
                                project: false,
                            },
                            &start,
                            &mode,
                        )
                    };
                    let on = search(
                        system,
                        kind,
                        Arm {
                            method,
                            curvature: true,
                            cg: false,
                            project: false,
                        },
                        &start,
                        &mode,
                    );
                    if skip_off {
                        println!(
                            "on {nc} {name} {kind} {method} seed {seed} calls {ncalls} e {ne:.6} f {nf:.3e} c {ncv:.4}",
                            nc = on.converged,
                            name = system.name,
                            kind = kind_label(kind),
                            method = method.label(),
                            ncalls = on.calls,
                            ne = on.energy,
                            nf = on.max_force,
                            ncv = on.curvature,
                        );
                        continue;
                    }
                    let rmsd = aligned_rmsd(&off.position, &on.position, system.align);
                    let same = off.converged
                        && on.converged
                        && (off.energy - on.energy).abs() <= ENERGY_TOL
                        && rmsd <= RMSD_TOL;
                    csv.push(format!(
                        "{system},{kind},{method},{seed},{oc},{nc},{ocalls},{ncalls},{oms:.6},{nms:.6},{oe:.8},{ne:.8},{of:.6e},{nf:.6e},{ocv:.6},{ncv:.6},{rmsd:.8},{same}",
                        system = system.name,
                        kind = kind_label(kind),
                        method = method.label(),
                        oc = off.converged,
                        nc = on.converged,
                        ocalls = off.calls,
                        ncalls = on.calls,
                        oms = off.millis,
                        nms = on.millis,
                        oe = off.energy,
                        ne = on.energy,
                        of = off.max_force,
                        nf = on.max_force,
                        ocv = off.curvature,
                        ncv = on.curvature,
                    ));
                    if same {
                        matched_calls_off.push(off.calls as f64);
                        matched_calls_on.push(on.calls as f64);
                        matched_ms_off.push(off.millis);
                        matched_ms_on.push(on.millis);
                    } else {
                        diverged += 1;
                        eprintln!(
                            "miss {name} {kind:?} {method} seed {seed} \
                             off {oc}/{oe:.6}/{ocalls} on {nc}/{ne:.6}/{ncalls} \
                             rmsd {rmsd:.6} off_ext {oext:.3} on_ext {next:.3} \
                             f {of:.3e}/{nf:.3e} c {ocv:.4}/{ncv:.4}",
                            name = system.name,
                            method = method.label(),
                            oc = off.converged,
                            oe = off.energy,
                            ocalls = off.calls,
                            nc = on.converged,
                            ne = on.energy,
                            ncalls = on.calls,
                            oext = extent(&off.position),
                            next = extent(&on.position),
                            of = off.max_force,
                            nf = on.max_force,
                            ocv = off.curvature,
                            ncv = on.curvature,
                        );
                    }
                }
                let n = matched_calls_off.len();
                println!(
                    "{}\t{}\t{}\t{}/{}\t{}\t{}\t{}\t{}\t{}",
                    system.name,
                    kind_label(kind),
                    method.label(),
                    n,
                    seeds,
                    band(&mut matched_calls_off, true),
                    band(&mut matched_calls_on, true),
                    band(&mut matched_ms_off, false),
                    band(&mut matched_ms_on, false),
                    diverged,
                );
            }
        }
    }
    if translator.is_none()
        && only.is_none()
        && only_kind.is_none()
        && only_method.is_none()
        && seeds == SEEDS
        && !skip_off
    {
        let path = format!(
            "{}/validation/minmode_translator.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::write(&path, csv.join("\n") + "\n").expect("write the csv");
        println!("wrote {path}");
    }
}

fn kind_label(kind: MinModeKind) -> &'static str {
    match kind {
        MinModeKind::Dimer => "dimer",
        MinModeKind::Lanczos => "lanczos",
    }
}

fn band(values: &mut [f64], calls: bool) -> String {
    if values.is_empty() {
        return "-".to_string();
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mid = if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    };
    let low = values[n / 4];
    let high = values[n - 1 - n / 4];
    if calls {
        format!("{mid:.0} [{low:.0}, {high:.0}]")
    } else {
        format!("{mid:.3} [{low:.3}, {high:.3}]")
    }
}

struct System {
    name: &'static str,
    align: Align,
    steps: usize,
    max_move: f64,
    start: fn(u64) -> (Array1<f64>, Array1<f64>),
    surface: Surface,
}

#[derive(Clone, Copy)]
enum Align {
    /// Rotate and translate, then the root-mean-square distance.
    Cluster,
    /// Minimum image after the centre-of-mass shift, in a cubic cell.
    Periodic { cell: f64 },
    /// The coordinates are the potential's own. No alignment.
    Model,
}

fn systems() -> Vec<System> {
    let vacancy_cell = vacancy_cell();
    vec![
        System {
            name: "lj13",
            align: Align::Cluster,
            steps: 2500,
            max_move: 0.2,
            start: lj13_start,
            surface: Surface::Cluster { cutoff: 15.0 },
        },
        System {
            name: "lj38",
            align: Align::Cluster,
            steps: 8000,
            max_move: 0.2,
            start: lj38_start,
            surface: Surface::Cluster { cutoff: 15.0 },
        },
        System {
            name: "vacancy-255",
            align: Align::Periodic { cell: vacancy_cell },
            steps: 8000,
            max_move: 0.2,
            start: vacancy_start,
            surface: Surface::Periodic {
                cutoff: 2.5,
                cell: vacancy_cell,
            },
        },
        System {
            name: "muller-brown",
            align: Align::Model,
            steps: 2500,
            max_move: 0.2,
            start: muller_start,
            surface: Surface::Muller,
        },
    ]
}

struct Outcome {
    calls: usize,
    millis: f64,
    converged: bool,
    energy: f64,
    position: Vec<f64>,
    max_force: f64,
    curvature: f64,
}

struct Arm {
    method: MethodKind,
    curvature: bool,
    cg: bool,
    project: bool,
}

fn search(
    system: &System,
    kind: MinModeKind,
    arm: Arm,
    start: &Array1<f64>,
    mode: &Array1<f64>,
) -> Outcome {
    let config = MinModeConfig {
        kind,
        method: arm.method.method(),
        force_tol: 1e-3,
        max_move: system.max_move,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, start.clone(), mode.clone()).expect("session");
    session.set_fire_curvature(arm.curvature);
    session.set_cg_translate(arm.cg);
    session.set_lbfgs_project(arm.project);
    let surface = Counting {
        inner: system.surface,
        calls: AtomicUsize::new(0),
    };
    let started = Instant::now();
    let mut converged = false;
    let mut max_force = f64::NAN;
    let mut curvature = f64::NAN;
    for _ in 0..system.steps {
        let report = match session.step(&surface) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("step failed: {err}");
                break;
            }
        };
        max_force = report.max_force;
        curvature = report.curvature;
        if report.status == MinModeStatus::Converged {
            converged = true;
            break;
        }
    }
    let millis = started.elapsed().as_secs_f64() * 1.0e3;
    let energy = match surface.eval(session.position()) {
        Ok((energy, _)) => energy,
        Err(err) => {
            eprintln!("final energy failed: {err}");
            f64::NAN
        }
    };
    Outcome {
        calls: surface.calls.load(Ordering::Relaxed) - 1,
        millis,
        converged,
        energy,
        position: session.position().to_vec(),
        max_force,
        curvature,
    }
}

#[derive(Clone, Copy)]
enum MethodKind {
    Fire,
    Lbfgs,
    Bb,
    QuickMin,
}

impl MethodKind {
    fn method(self) -> rgmin::Method {
        match self {
            Self::Fire => rgmin::Method::Fire {
                kind: rgmin::FireKind::V2,
            },
            Self::Lbfgs => rgmin::Method::Lbfgs { memory: 10 },
            Self::Bb => rgmin::Method::Bb,
            Self::QuickMin => rgmin::Method::QuickMin,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fire => "fire",
            Self::Lbfgs => "lbfgs",
            Self::Bb => "bb",
            Self::QuickMin => "quickmin",
        }
    }
}

/// FIRE with the stiff-mode probe against one other translator.
/// `name` is `cg`, `project`, `lbfgs`, `bb`, or `quickmin`.
fn compare_translator(system: &System, kind: MinModeKind, name: &str, seeds: usize) {
    let mut matched_calls_off = Vec::new();
    let mut matched_calls_on = Vec::new();
    let mut matched_ms_off = Vec::new();
    let mut matched_ms_on = Vec::new();
    let mut diverged = 0usize;
    let mut on_converged = 0usize;
    for index in 0..seeds {
        let seed = mix_seed(index);
        let (start, mode) = (system.start)(seed);
        let fire = Arm {
            method: MethodKind::Fire,
            curvature: true,
            cg: false,
            project: false,
        };
        let off = search(system, kind, fire, &start, &mode);
        let on = search(
            system,
            kind,
            match name {
                "cg" => Arm {
                    method: MethodKind::Fire,
                    curvature: true,
                    cg: true,
                    project: false,
                },
                "project" => Arm {
                    method: MethodKind::Lbfgs,
                    curvature: true,
                    cg: false,
                    project: true,
                },
                "lbfgs" => Arm {
                    method: MethodKind::Lbfgs,
                    curvature: true,
                    cg: false,
                    project: false,
                },
                "bb" => Arm {
                    method: MethodKind::Bb,
                    curvature: true,
                    cg: false,
                    project: false,
                },
                "quickmin" => Arm {
                    method: MethodKind::QuickMin,
                    curvature: true,
                    cg: false,
                    project: false,
                },
                other => panic!("unknown translator {other}"),
            },
            &start,
            &mode,
        );
        if on.converged {
            on_converged += 1;
        }
        let rmsd = aligned_rmsd(&off.position, &on.position, system.align);
        let same = off.converged
            && on.converged
            && (off.energy - on.energy).abs() <= ENERGY_TOL
            && rmsd <= RMSD_TOL;
        println!(
            "seed {seed} fire {oc}/{oe:.6}/{ocalls} cand {nc}/{ne:.6}/{ncalls} rmsd {rmsd:.6} same {same}",
            oc = off.converged,
            oe = off.energy,
            ocalls = off.calls,
            nc = on.converged,
            ne = on.energy,
            ncalls = on.calls,
        );
        if same {
            matched_calls_off.push(off.calls as f64);
            matched_calls_on.push(on.calls as f64);
            matched_ms_off.push(off.millis);
            matched_ms_on.push(on.millis);
        } else {
            diverged += 1;
            eprintln!(
                "miss {name_sys} {kind:?} {name} seed {seed} \
                 fire {oc}/{oe:.6}/{ocalls} cand {nc}/{ne:.6}/{ncalls} \
                 rmsd {rmsd:.6} f {of:.3e}/{nf:.3e}",
                name_sys = system.name,
                oc = off.converged,
                oe = off.energy,
                ocalls = off.calls,
                nc = on.converged,
                ne = on.energy,
                ncalls = on.calls,
                of = off.max_force,
                nf = on.max_force,
            );
        }
    }
    let n = matched_calls_off.len();
    println!(
        "{}\t{}\t{}\t{}/{}\t{}\t{}\t{}\t{}\t{}\tconverged {on_converged}/{seeds}",
        system.name,
        kind_label(kind),
        name,
        n,
        seeds,
        band(&mut matched_calls_off, true),
        band(&mut matched_calls_on, true),
        band(&mut matched_ms_off, false),
        band(&mut matched_ms_on, false),
        diverged,
    );
}

#[derive(Clone, Copy)]
enum Surface {
    Cluster { cutoff: f64 },
    Periodic { cutoff: f64, cell: f64 },
    Muller,
}

struct Counting {
    inner: Surface,
    calls: AtomicUsize,
}

impl PointSurface for Counting {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(match self.inner {
            Surface::Cluster { cutoff } => lj_energy_gradient(x, cutoff, None),
            Surface::Periodic { cutoff, cell } => lj_energy_gradient(x, cutoff, Some(cell)),
            Surface::Muller => muller_energy_gradient(x),
        })
    }
}

/// Shifted 12-6 Lennard-Jones, epsilon 1, sigma 1. The shift is the
/// unshifted pair energy at the cutoff, so a pair outside the cutoff
/// contributes nothing. `cell` is a cubic minimum image.
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
            let (u, fscale) = lj_pair(r2);
            energy += u - shift;
            for k in 0..3 {
                let g = -fscale * d[k];
                gradient[3 * i + k] += g;
                gradient[3 * j + k] -= g;
            }
        }
    }
    (energy, Array1::from(gradient))
}

/// Unshifted pair energy and `-dU/dr / r`.
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

/// Müller-Brown reaction surface, with a convex third coordinate so the
/// search has one unstable direction in the plane of the reaction.
fn muller_energy_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (energy_xy, gx, gy) = muller_brown(x[0], x[1]);
    let energy = energy_xy + 2.0 * x[2] * x[2];
    (energy, Array1::from_vec(vec![gx, gy, 4.0 * x[2]]))
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
    // The saddle's unstable mode is a ridge. A displacement of 0.12 along
    // it and 0.04 across, from this seed, stays in the cluster. The twelve
    // starts are that point plus a 0.002 jitter, which is still on the
    // cluster side of the ridge. A fresh direction of size 0.02 often
    // evaporates an atom under the dimer.
    let (mut position, mode) = offset_start(&position, &mode, 0.12, 0.04, BASE_SEED);
    let kick = mix(position.len(), seed);
    for (coord, kick) in position.iter_mut().zip(kick) {
        *coord += 0.002 * kick;
    }
    (Array1::from(position), Array1::from(mode))
}

fn lj38_start(seed: u64) -> (Array1<f64>, Array1<f64>) {
    let root = format!("{}/tests/fixtures/lj38", env!("CARGO_MANIFEST_DIR"));
    let reactant = read_xyz(&format!("{root}/start_0.xyz"));
    let product = read_xyz(&format!("{root}/end_0.xyz"));
    let mode: Vec<f64> = product
        .iter()
        .zip(reactant.iter())
        .map(|(p, r)| p - r)
        .collect();
    let position: Vec<f64> = reactant
        .iter()
        .zip(product.iter())
        .map(|(r, p)| 0.6 * r + 0.4 * p)
        .collect();
    let (position, mode) = offset_start(&position, &mode, 0.0, 0.008, seed);
    (Array1::from(position), Array1::from(mode))
}

fn vacancy_cell() -> f64 {
    let n = 4.0_f64;
    let spacing = 2.0_f64.powf(1.0 / 6.0) * 2.0_f64.sqrt();
    n * spacing
}

/// 4x4x4 face-centred cubic cell with one vacancy. The migrating atom
/// sits just off the bond midpoint, which is the hop saddle.
fn vacancy_start(seed: u64) -> (Array1<f64>, Array1<f64>) {
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
    let (position, mode) = offset_start(&position, &mode, 0.0, 0.008, seed);
    (Array1::from(position), Array1::from(mode))
}

/// A point between the Müller-Brown minima, nudged by `seed`. The third
/// coordinate is the convex direction.
fn muller_start(seed: u64) -> (Array1<f64>, Array1<f64>) {
    let kick = mix(3, seed);
    let position = vec![-0.6 + 0.08 * kick[0], 0.8 + 0.08 * kick[1], 0.05 * kick[2]];
    let mode = vec![-1.0 + 0.3 * kick[1], 0.4 + 0.3 * kick[2], 0.05 * kick[0]];
    let mode = unit(&mode);
    (Array1::from(position), Array1::from(mode))
}

fn read_xyz(path: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut coords = Vec::new();
    for line in text.lines().skip(2) {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace().skip(1);
        for _ in 0..3 {
            coords.push(parts.next().unwrap().parse().unwrap());
        }
    }
    coords
}

/// Move off `origin` by `along` times the unit mode and `aside` times a
/// direction from `seed` perpendicular to that mode.
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

fn bench_base() -> u64 {
    std::env::var("RGSADDLE_BENCH_BASE")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(BASE_SEED)
}

fn mix_seed(index: usize) -> u64 {
    bench_base().wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
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

fn aligned_rmsd(a: &[f64], b: &[f64], align: Align) -> f64 {
    match align {
        Align::Cluster => kabsch_rmsd(a, b),
        Align::Periodic { cell } => periodic_rmsd(a, b, cell),
        Align::Model => euclidean_rmsd(a, b),
    }
}

fn extent(p: &[f64]) -> f64 {
    p.iter().copied().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn euclidean_rmsd(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (left, right) in a.iter().zip(b.iter()) {
        let d = left - right;
        sum += d * d;
    }
    (sum / a.len() as f64).sqrt()
}

fn periodic_rmsd(a: &[f64], b: &[f64], cell: f64) -> f64 {
    let n = a.len() / 3;
    let mut shift = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            let mut d = b[3 * i + k] - a[3 * i + k];
            d -= cell * (d / cell).round();
            shift[k] += d;
        }
    }
    for item in &mut shift {
        *item /= n as f64;
    }
    let mut sum = 0.0;
    for i in 0..n {
        for k in 0..3 {
            let mut d = b[3 * i + k] - a[3 * i + k] - shift[k];
            d -= cell * (d / cell).round();
            sum += d * d;
        }
    }
    (sum / a.len() as f64).sqrt()
}

/// Rotate and translate `b` onto `a`. The singular vectors come from the
/// eigenpairs of the 3x3 Gram matrix.
#[allow(clippy::needless_range_loop)]
fn kabsch_rmsd(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() / 3;
    let mut ca = [0.0; 3];
    let mut cb = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            ca[k] += a[3 * i + k];
            cb[k] += b[3 * i + k];
        }
    }
    for k in 0..3 {
        ca[k] /= n as f64;
        cb[k] /= n as f64;
    }
    let mut cov = [[0.0; 3]; 3];
    for i in 0..n {
        let pa = [a[3 * i] - ca[0], a[3 * i + 1] - ca[1], a[3 * i + 2] - ca[2]];
        let pb = [b[3 * i] - cb[0], b[3 * i + 1] - cb[1], b[3 * i + 2] - cb[2]];
        for row in 0..3 {
            for col in 0..3 {
                cov[row][col] += pa[row] * pb[col];
            }
        }
    }
    let mut gram = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                gram[i][j] += cov[k][i] * cov[k][j];
            }
        }
    }
    let vectors = jacobi_3(&mut gram);
    let mut sigma = [0.0; 3];
    let mut order = [0, 1, 2];
    for i in 0..3 {
        sigma[i] = gram[i][i].max(0.0).sqrt();
    }
    order.sort_by(|&i, &j| sigma[j].total_cmp(&sigma[i]));
    let mut right = [[0.0; 3]; 3];
    let mut left = [[0.0; 3]; 3];
    for (slot, &axis) in order.iter().enumerate() {
        for k in 0..3 {
            right[k][slot] = vectors[k][axis];
        }
        if sigma[axis] > 1e-12 {
            for row in 0..3 {
                let mut acc = 0.0;
                for col in 0..3 {
                    acc += cov[row][col] * vectors[col][axis];
                }
                left[row][slot] = acc / sigma[axis];
            }
        }
    }
    complete_left(&mut left);
    if determinant_3(&mat_mul(&left, &transpose_3(&right))) < 0.0 {
        for row in 0..3 {
            left[row][2] = -left[row][2];
        }
    }
    let rotation = mat_mul(&left, &transpose_3(&right));
    let mut sum = 0.0;
    for i in 0..n {
        let pb = [b[3 * i] - cb[0], b[3 * i + 1] - cb[1], b[3 * i + 2] - cb[2]];
        for row in 0..3 {
            let mut mapped = 0.0;
            for col in 0..3 {
                mapped += rotation[row][col] * pb[col];
            }
            let d = a[3 * i + row] - ca[row] - mapped;
            sum += d * d;
        }
    }
    (sum / a.len() as f64).sqrt()
}

#[allow(clippy::needless_range_loop)]
fn complete_left(left: &mut [[f64; 3]; 3]) {
    let mut norms = [0.0; 3];
    for slot in 0..3 {
        norms[slot] = (0..3)
            .map(|row| left[row][slot].powi(2))
            .sum::<f64>()
            .sqrt();
    }
    if norms[0] < 1e-8 {
        left[0][0] = 1.0;
        left[1][0] = 0.0;
        left[2][0] = 0.0;
        norms[0] = 1.0;
    }
    for slot in 0..3 {
        if norms[slot] > 1e-8 {
            for row in 0..3 {
                left[row][slot] /= norms[slot];
            }
        }
    }
    if norms[1] < 1e-8 {
        let ax = [left[0][0], left[1][0], left[2][0]];
        let trial = if ax[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let cross = [
            ax[1] * trial[2] - ax[2] * trial[1],
            ax[2] * trial[0] - ax[0] * trial[2],
            ax[0] * trial[1] - ax[1] * trial[0],
        ];
        let scale = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        for (row, value) in cross.iter().enumerate() {
            left[row][1] = value / scale;
        }
    }
    if norms[2] < 1e-8 {
        let cross = [
            left[1][0] * left[2][1] - left[2][0] * left[1][1],
            left[2][0] * left[0][1] - left[0][0] * left[2][1],
            left[0][0] * left[1][1] - left[1][0] * left[0][1],
        ];
        for (row, value) in cross.iter().enumerate() {
            left[row][2] = *value;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn jacobi_3(m: &mut [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..8 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let app = m[p][p];
            let aqq = m[q][q];
            let apq = m[p][q];
            if apq.abs() <= 1e-15 * (app.abs() + aqq.abs() + 1.0) {
                continue;
            }
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            m[p][p] = app - t * apq;
            m[q][q] = aqq + t * apq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;
            for r in 0..3 {
                if r == p || r == q {
                    continue;
                }
                let arp = m[r][p];
                let arq = m[r][q];
                m[r][p] = c * arp - s * arq;
                m[p][r] = m[r][p];
                m[r][q] = s * arp + c * arq;
                m[q][r] = m[r][q];
            }
            for r in 0..3 {
                let vip = v[r][p];
                let viq = v[r][q];
                v[r][p] = c * vip - s * viq;
                v[r][q] = s * vip + c * viq;
            }
        }
    }
    v
}

#[allow(clippy::needless_range_loop)]
fn transpose_3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn determinant_3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn kabsch_self_check() {
    let a = vec![0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.1, 1.0, 0.3, 0.0, 0.2, 1.1];
    let mut b = Vec::new();
    for i in 0..4 {
        let (x, y, z) = (a[3 * i], a[3 * i + 1], a[3 * i + 2]);
        b.extend_from_slice(&[-y + 0.4, x - 0.2, z + 0.7]);
    }
    let rmsd = kabsch_rmsd(&a, &b);
    assert!(rmsd < 1e-8, "kabsch rmsd {rmsd}");
}
