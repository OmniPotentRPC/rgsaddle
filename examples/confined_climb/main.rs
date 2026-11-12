//! Confined convex climb on Mishin copper point defects.
//!
//! Same potential, same seeds. Each seed is a random mode on every atom
//! or a displacement inside `displace_radius` of the defect, the eOn
//! initial kick. Confine is none, top-force, the eOn threshold, the
//! LOGN+RAS sigmoid, or participation in the dimer's own mode.
//! Odd seeds are held out and are not used to choose a default.
//!
//! ```text
//! cargo run --release --example confined_climb
//! ```
//!
//! `RGSADDLE_BENCH_SEEDS`, `RGSADDLE_BENCH_STEPS`, `RGSADDLE_BENCH_SYSTEM`,
//! and `RGSADDLE_BENCH_CONFINE` narrow the matrix. The rows land in
//! `benchmarks/confined_climb.csv`.

mod eam;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use eam::Eam;
use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{
    Confine, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

const A: f64 = 3.615;
const BASE: u64 = 20261112;
const RADIUS: f64 = 3.8;

struct Metal<'a> {
    pot: &'a Eam,
    box_len: f64,
    calls: AtomicUsize,
}

impl PointSurface for Metal<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.pot.energy_gradient(x, self.box_len))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let n = x.len() / 3;
        let mut modes = Array2::zeros((3, x.len()));
        let scale = (n as f64).sqrt();
        for axis in 0..3 {
            for atom in 0..n {
                modes[(axis, 3 * atom + axis)] = 1.0 / scale;
            }
        }
        Ok(modes)
    }
}

fn main() {
    let path = format!(
        "{}/examples/data/Cu01.eam.alloy",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("Mishin Cu table");
    let pot = Eam::load(&text);
    let (eco, force) = perfect_check(&pot);
    println!("fcc E/atom {eco:.6} max|F| {force:.3e}");
    if (eco + 3.54).abs() > 2e-3 || force > 1e-3 {
        panic!("Mishin fcc check failed");
    }
    let fd = force_check(&pot);
    println!("force finite difference {fd:.3e}");
    if fd > 1e-3 {
        panic!("Mishin force does not match a finite difference");
    }

    let seeds: usize = env_usize("RGSADDLE_BENCH_SEEDS", 4);
    let steps: usize = env_usize("RGSADDLE_BENCH_STEPS", 180);
    let only = std::env::var("RGSADDLE_BENCH_SYSTEM").ok();
    let only_confine = std::env::var("RGSADDLE_BENCH_CONFINE").ok();
    let systems = [
        prepare(&pot, Kind::Vacancy, 8),
        prepare(&pot, Kind::Sia, 8),
        prepare(&pot, Kind::Sia, 3),
    ];
    let confines = [
        ("none", Confine::None),
        ("top_force", Confine::top_force(20)),
        ("threshold", Confine::eon_threshold()),
        ("sigmoid", Confine::log_n_ras([0.0; 3])),
        ("participation", Confine::own_mode(0.0, 1.0)),
    ];
    let mut rows = vec![
        "system,init,confine,seed,held_out,converged,calls,ms,energy,max_force,curvature,barrier,connected,disp_rms,disp_max,cluster"
            .to_string(),
    ];
    println!("seeds {seeds} steps {steps} base {BASE}");
    for system in &systems {
        if only.as_ref().is_some_and(|name| name != &system.name) {
            continue;
        }
        println!(
            "{} atoms {} residual {:.3e} E {:.4}",
            system.name,
            system.positions.len() / 3,
            system.residual,
            system.energy
        );
        for init in ["random", "local"] {
            let mut group = Vec::new();
            for index in 0..seeds {
                let seed = mix(BASE ^ (index as u64).wrapping_mul(0x9E37) ^ system.tag);
                let (start, mode) = initial(system, init, seed);
                for (name, confine) in &confines {
                    if only_confine.as_ref().is_some_and(|want| want != *name) {
                        continue;
                    }
                    let confine = confine.clone_kind(system.centre);
                    let outcome = search(system, &pot, &start, &mode, confine, steps);
                    println!(
                        "{}\t{}\t{}\t{}\tconv {}\tcalls {}\t{:.1} ms\tE {:.4}\tF {:.3e}",
                        system.name,
                        init,
                        name,
                        index,
                        outcome.converged,
                        outcome.calls,
                        outcome.ms,
                        outcome.energy,
                        outcome.max_force
                    );
                    group.push((index, name.to_string(), outcome));
                }
            }
            let clusters = clusters_of(&group);
            for (index, name, outcome) in &group {
                let cluster = clusters
                    .iter()
                    .find(|(i, n, _)| i == index && n == name)
                    .map(|(_, _, c)| c.to_string())
                    .unwrap_or_default();
                rows.push(format!(
                    "{},{},{},{},{},{},{},{:.3},{:.8},{:.6e},{:.6e},{:.8},{},{:.6},{:.6},{}",
                    system.name,
                    init,
                    name,
                    index,
                    if index % 2 == 1 { 1 } else { 0 },
                    if outcome.converged { 1 } else { 0 },
                    outcome.calls,
                    outcome.ms,
                    outcome.energy,
                    outcome.max_force,
                    outcome.curvature,
                    outcome.barrier,
                    if outcome.connected { 1 } else { 0 },
                    outcome.disp_rms,
                    outcome.disp_max,
                    cluster
                ));
            }
            report_group(&system.name, init, &group, &clusters);
        }
    }
    let out = format!(
        "{}/benchmarks/confined_climb.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(&out, rows.join("\n") + "\n").unwrap();
    println!("wrote {out}");
}

struct System {
    name: String,
    tag: u64,
    box_len: f64,
    positions: Array1<f64>,
    energy: f64,
    residual: f64,
    centre: [f64; 3],
}

enum Kind {
    Vacancy,
    Sia,
}

fn prepare(pot: &Eam, kind: Kind, n: usize) -> System {
    let (atoms, box_len, centre, name) = match kind {
        Kind::Vacancy => {
            let (atoms, box_len) = fcc(n);
            let centre = [box_len * 0.5; 3];
            let drop = closest(&atoms, centre);
            let site = atoms[drop];
            let atoms: Vec<_> = atoms
                .into_iter()
                .enumerate()
                .filter(|(i, _)| *i != drop)
                .map(|(_, atom)| atom)
                .collect();
            (
                atoms,
                box_len,
                site,
                format!("cu-vacancy-{}", n * n * n * 4 - 1),
            )
        }
        Kind::Sia => {
            let (mut atoms, box_len) = fcc(n);
            let centre = [box_len * 0.5; 3];
            let host = closest(&atoms, centre);
            let site = atoms[host];
            atoms.remove(host);
            atoms.push([site[0] - 0.95, site[1], site[2]]);
            atoms.push([site[0] + 0.95, site[1], site[2]]);
            let count = atoms.len();
            (atoms, box_len, site, format!("cu-sia100-{count}"))
        }
    };
    let flat = flatten(&atoms);
    let (positions, energy, residual) = quench(pot, flat, box_len, 1e-2, 800);
    System {
        tag: n as u64 * 16 + matches!(kind, Kind::Sia) as u64,
        name,
        box_len,
        positions,
        energy,
        residual,
        centre,
    }
}

fn fcc(n: usize) -> (Vec<[f64; 3]>, f64) {
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
                        (ix as f64 + site[0]) * A,
                        (iy as f64 + site[1]) * A,
                        (iz as f64 + site[2]) * A,
                    ]);
                }
            }
        }
    }
    (atoms, n as f64 * A)
}

fn flatten(atoms: &[[f64; 3]]) -> Array1<f64> {
    let mut x = Array1::zeros(atoms.len() * 3);
    for (atom, p) in atoms.iter().enumerate() {
        x[3 * atom] = p[0];
        x[3 * atom + 1] = p[1];
        x[3 * atom + 2] = p[2];
    }
    x
}

fn closest(atoms: &[[f64; 3]], centre: [f64; 3]) -> usize {
    atoms
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| dist2(a, &centre).partial_cmp(&dist2(b, &centre)).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

fn quench(
    pot: &Eam,
    mut x: Array1<f64>,
    box_len: f64,
    tol: f64,
    steps: usize,
) -> (Array1<f64>, f64, f64) {
    let mut dt = 0.01;
    let mut energy = 0.0;
    let mut residual = f64::MAX;
    for _ in 0..steps {
        let (e, g) = pot.energy_gradient(x.view(), box_len);
        energy = e;
        residual = max_atom(&g);
        if residual < tol {
            break;
        }
        let mut trial = x.clone();
        for (slot, &grad) in trial.iter_mut().zip(g.iter()) {
            *slot -= dt * grad;
        }
        cap_step(x.view(), &mut trial, 0.04);
        let (e2, g2) = pot.energy_gradient(trial.view(), box_len);
        if e2 <= e + 1e-8 {
            x = trial;
            energy = e2;
            residual = max_atom(&g2);
            dt = (dt * 1.15).min(0.05);
            if residual < tol {
                break;
            }
        } else {
            dt *= 0.5;
            if dt < 1e-4 {
                dt = 0.01;
            }
        }
    }
    (x, energy, residual)
}

fn cap_step(origin: ArrayView1<f64>, trial: &mut Array1<f64>, cap: f64) {
    let mut max_atom = 0.0_f64;
    let n = origin.len() / 3;
    for atom in 0..n {
        let mut r2 = 0.0;
        for k in 0..3 {
            let d = trial[3 * atom + k] - origin[3 * atom + k];
            r2 += d * d;
        }
        max_atom = max_atom.max(r2.sqrt());
    }
    if max_atom > cap && max_atom > 0.0 {
        let scale = cap / max_atom;
        for (slot, &start) in trial.iter_mut().zip(origin.iter()) {
            *slot = start + scale * (*slot - start);
        }
    }
}

fn max_atom(gradient: &Array1<f64>) -> f64 {
    let n = gradient.len() / 3;
    let mut max_force = 0.0_f64;
    for atom in 0..n {
        let r = (gradient[3 * atom].powi(2)
            + gradient[3 * atom + 1].powi(2)
            + gradient[3 * atom + 2].powi(2))
        .sqrt();
        max_force = max_force.max(r);
    }
    max_force
}

struct Outcome {
    converged: bool,
    calls: usize,
    ms: f64,
    energy: f64,
    max_force: f64,
    curvature: f64,
    barrier: f64,
    connected: bool,
    disp_rms: f64,
    disp_max: f64,
}

fn search(
    system: &System,
    pot: &Eam,
    start: &Array1<f64>,
    mode: &Array1<f64>,
    confine: Confine,
    steps: usize,
) -> Outcome {
    let config = MinModeConfig {
        kind: MinModeKind::Dimer,
        force_tol: 1e-2,
        max_move: 0.2,
        max_rotations: 12,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, start.clone(), mode.clone()).unwrap();
    session.set_confine(confine).unwrap();
    let surface = Metal {
        pot,
        box_len: system.box_len,
        calls: AtomicUsize::new(0),
    };
    let started = Instant::now();
    let mut converged = false;
    let mut max_force = f64::NAN;
    let mut curvature = f64::NAN;
    for _ in 0..steps {
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
    let ms = started.elapsed().as_secs_f64() * 1.0e3;
    let calls = surface.calls.load(Ordering::Relaxed);
    let (energy, gradient) = pot.energy_gradient(session.position(), system.box_len);
    let full_force = gradient.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if converged {
        max_force = full_force;
    }
    let (barrier, connected, disp_rms, disp_max) = if converged {
        connect(system, pot, session.position(), session.mode(), energy)
    } else {
        (f64::NAN, false, f64::NAN, f64::NAN)
    };
    Outcome {
        converged,
        calls,
        ms,
        energy,
        max_force,
        curvature,
        barrier,
        connected,
        disp_rms,
        disp_max,
    }
}

fn connect(
    system: &System,
    pot: &Eam,
    saddle: ArrayView1<f64>,
    mode: ArrayView1<f64>,
    saddle_energy: f64,
) -> (f64, bool, f64, f64) {
    let (rms, max_disp) = displacement(system, system.positions.view(), saddle);
    let mut peak = 0.0_f64;
    let n_atoms = mode.len() / 3;
    for atom in 0..n_atoms {
        let r = (mode[3 * atom].powi(2) + mode[3 * atom + 1].powi(2) + mode[3 * atom + 2].powi(2))
            .sqrt();
        peak = peak.max(r);
    }
    let kick = if peak > 0.0 { 0.8 / peak } else { 0.0 };
    let mut back = [0.0; 2];
    for (index, sign) in [1.0, -1.0].into_iter().enumerate() {
        let mut x = saddle.to_owned();
        for (slot, &component) in x.iter_mut().zip(mode.iter()) {
            *slot += sign * kick * component;
        }
        let (relaxed, _, _) = quench(pot, x, system.box_len, 2e-2, 160);
        let (distance, _) = displacement(system, system.positions.view(), relaxed.view());
        back[index] = distance;
    }
    let connected = (back[0] < 0.03) ^ (back[1] < 0.03);
    let barrier = saddle_energy - system.energy;
    (barrier, connected, rms, max_disp)
}

fn displacement(system: &System, reference: ArrayView1<f64>, other: ArrayView1<f64>) -> (f64, f64) {
    let n = reference.len() / 3;
    let mut shift = [0.0; 3];
    for atom in 0..n {
        for k in 0..3 {
            shift[k] += mic(
                other[3 * atom + k] - reference[3 * atom + k],
                system.box_len,
            );
        }
    }
    for s in &mut shift {
        *s /= n as f64;
    }
    let mut sum = 0.0;
    let mut max_disp = 0.0_f64;
    for atom in 0..n {
        let mut r2 = 0.0;
        for k in 0..3 {
            let d = mic(
                other[3 * atom + k] - reference[3 * atom + k],
                system.box_len,
            ) - shift[k];
            r2 += d * d;
        }
        sum += r2;
        max_disp = max_disp.max(r2.sqrt());
    }
    ((sum / n as f64).sqrt(), max_disp)
}

fn mic(d: f64, box_len: f64) -> f64 {
    d - box_len * (d / box_len).round()
}

fn initial(system: &System, init: &str, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let n = system.positions.len() / 3;
    let mut mode = Array1::zeros(system.positions.len());
    let mut start = system.positions.clone();
    if init == "local" {
        for atom in 0..n {
            let p = [
                system.positions[3 * atom],
                system.positions[3 * atom + 1],
                system.positions[3 * atom + 2],
            ];
            let d = [
                mic(p[0] - system.centre[0], system.box_len),
                mic(p[1] - system.centre[1], system.box_len),
                mic(p[2] - system.centre[2], system.box_len),
            ];
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > RADIUS * RADIUS {
                continue;
            }
            for k in 0..3 {
                let kick = 0.18 * gaussian(mix(seed ^ (atom as u64) << 2 ^ k as u64));
                mode[3 * atom + k] = kick;
                start[3 * atom + k] += kick;
            }
        }
    } else {
        for component in 0..mode.len() {
            mode[component] = gaussian(mix(seed ^ component as u64));
        }
    }
    remove_mean(&mut mode);
    let norm = mode.dot(&mode).sqrt();
    if norm > 1e-8 {
        mode /= norm;
    } else {
        mode[0] = 1.0;
    }
    (start, mode)
}

fn remove_mean(mode: &mut Array1<f64>) {
    let n = mode.len() / 3;
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for k in 0..3 {
            mean[k] += mode[3 * atom + k];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    for atom in 0..n {
        for k in 0..3 {
            mode[3 * atom + k] -= mean[k];
        }
    }
}

fn clusters_of(group: &[(usize, String, Outcome)]) -> Vec<(usize, String, usize)> {
    let mut assigned = Vec::new();
    let mut next = std::collections::HashMap::<String, usize>::new();
    for (index, name, outcome) in group {
        if !outcome.converged {
            continue;
        }
        let mut found = None;
        for (other_index, other_name, other) in group {
            if other_name != name || other_index == index || !other.converged {
                continue;
            }
            if (outcome.energy - other.energy).abs() < 0.02
                && (outcome.disp_rms - other.disp_rms).abs() < 0.08
            {
                if let Some((_, _, id)) = assigned
                    .iter()
                    .find(|(i, n, _)| i == other_index && n == other_name)
                {
                    found = Some(*id);
                    break;
                }
            }
        }
        let id = found.unwrap_or_else(|| {
            let slot = next.entry(name.clone()).or_insert(0);
            *slot += 1;
            *slot
        });
        assigned.push((*index, name.clone(), id));
    }
    assigned
}

fn report_group(
    system: &str,
    init: &str,
    group: &[(usize, String, Outcome)],
    clusters: &[(usize, String, usize)],
) {
    let mut names: Vec<&str> = group.iter().map(|(_, name, _)| name.as_str()).collect();
    names.sort();
    names.dedup();
    for name in &names {
        let rows: Vec<_> = group
            .iter()
            .filter(|(_, n, _)| n.as_str() == *name)
            .collect();
        let converged = rows.iter().filter(|(_, _, o)| o.converged).count();
        let connected = rows.iter().filter(|(_, _, o)| o.connected).count();
        let mut calls: Vec<f64> = rows.iter().map(|(_, _, o)| o.calls as f64).collect();
        let mut millis: Vec<f64> = rows.iter().map(|(_, _, o)| o.ms).collect();
        let distinct = clusters
            .iter()
            .filter(|(index, n, _)| n.as_str() == *name && rows.iter().any(|(i, _, _)| i == index))
            .map(|(_, _, id)| id)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let lowest = rows
            .iter()
            .filter(|(_, _, o)| o.connected)
            .map(|(_, _, o)| o.barrier)
            .fold(f64::INFINITY, f64::min);
        println!(
            "summary {system} {init} {name} converged {converged}/{} connected {connected} distinct {distinct} calls {} ms {} lowest {}",
            rows.len(),
            band(&mut calls),
            band(&mut millis),
            if lowest.is_finite() {
                format!("{lowest:.4}")
            } else {
                "-".into()
            }
        );
    }
    let none = barriers(group, "none");
    for name in &names {
        if *name == "none" {
            continue;
        }
        let (u, p) = mann_whitney(&none, &barriers(group, name));
        println!("bias {system} {init} {name} U {u:.1} p {p:.4}");
    }
}

fn barriers(group: &[(usize, String, Outcome)], name: &str) -> Vec<f64> {
    group
        .iter()
        .filter(|(_, n, outcome)| n == name && outcome.connected && outcome.barrier.is_finite())
        .map(|(_, _, outcome)| outcome.barrier)
        .collect()
}

/// Mann-Whitney U on the first sample, and a two-sided normal p.
fn mann_whitney(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.is_empty() || b.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut all: Vec<(f64, bool)> = a
        .iter()
        .copied()
        .map(|v| (v, true))
        .chain(b.iter().copied().map(|v| (v, false)))
        .collect();
    all.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());
    let mut ranks = vec![0.0; all.len()];
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && (all[j + 1].0 - all[i].0).abs() <= 1e-9 {
            j += 1;
        }
        let rank = 0.5 * (i + j) as f64 + 1.0;
        for slot in ranks.iter_mut().take(j + 1).skip(i) {
            *slot = rank;
        }
        i = j + 1;
    }
    let rank_sum: f64 = ranks
        .iter()
        .zip(all.iter())
        .filter(|(_, item)| item.1)
        .map(|(rank, _)| *rank)
        .sum();
    let n1 = a.len() as f64;
    let n2 = b.len() as f64;
    let u = rank_sum - n1 * (n1 + 1.0) / 2.0;
    let mut tie = 0.0;
    i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && (all[j + 1].0 - all[i].0).abs() <= 1e-9 {
            j += 1;
        }
        let t = (j - i + 1) as f64;
        tie += t * t * t - t;
        i = j + 1;
    }
    let n = n1 + n2;
    let sigma2 = if n > 1.0 {
        n1 * n2 / 12.0 * ((n + 1.0) - tie / (n * (n - 1.0)))
    } else {
        0.0
    };
    if sigma2 <= 0.0 {
        return (u, 1.0);
    }
    let z = ((u - n1 * n2 / 2.0).abs() - 0.5).max(0.0) / sigma2.sqrt();
    (u, normal_two_sided(z))
}

fn normal_two_sided(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = 0.3989422804014327 * (-0.5 * z * z).exp() * poly;
    (2.0 * tail).clamp(0.0, 1.0)
}

fn band(values: &mut [f64]) -> String {
    if values.is_empty() {
        return "-".into();
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values[values.len() / 2];
    format!("{mid:.1}")
}

fn perfect_check(pot: &Eam) -> (f64, f64) {
    let (atoms, box_len) = fcc(4);
    let x = flatten(&atoms);
    let (energy, gradient) = pot.energy_gradient(x.view(), box_len);
    (energy / atoms.len() as f64, max_atom(&gradient))
}

fn force_check(pot: &Eam) -> f64 {
    let (atoms, box_len) = fcc(2);
    let mut x = flatten(&atoms);
    x[0] += 0.04;
    let (_, g) = pot.energy_gradient(x.view(), box_len);
    let h = 1e-4;
    let mut plus = x.clone();
    plus[0] += h;
    let mut minus = x.clone();
    minus[0] -= h;
    let (ep, _) = pot.energy_gradient(plus.view(), box_len);
    let (em, _) = pot.energy_gradient(minus.view(), box_len);
    let fd = (ep - em) / (2.0 * h);
    (fd - g[0]).abs()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gaussian(seed: u64) -> f64 {
    let u = (mix(seed) >> 11) as f64 / (1u64 << 53) as f64;
    let v = (mix(seed ^ 0xA5A5) >> 11) as f64 / (1u64 << 53) as f64;
    let u = u.clamp(1e-12, 1.0);
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

trait ConfineBuild {
    fn clone_kind(&self, centre: [f64; 3]) -> Confine;
}

impl ConfineBuild for Confine {
    fn clone_kind(&self, centre: [f64; 3]) -> Confine {
        match self {
            Confine::None => Confine::None,
            Confine::TopForce { n } => Confine::top_force(*n),
            Confine::Threshold {
                min_force,
                scale,
                min_active,
                boost,
            } => Confine::threshold(*min_force, *scale, *min_active, *boost),
            Confine::Sigmoid { .. } => Confine::log_n_ras(centre),
            Confine::Participation { floor, power, .. } => Confine::own_mode(*floor, *power),
        }
    }
}
