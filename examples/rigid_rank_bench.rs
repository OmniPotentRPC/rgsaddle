//! Force calls for dimer, Lanczos, and CI-NEB on geometries that enter
//! the rigid rank test.
//!
//! Three systems, each with a translation/rotation-invariant potential
//! except the frozen slab, whose adatom sits in a lab-frame double well:
//!
//! * `lj-far` — a Lennard-Jones trimer (masses 16, 1, 1) on its linear
//!   saddle, shifted by `10^5` length units.
//! * `near-linear` — the same trimer with a bend of `1e-8`.
//! * `frozen-slab` — one adatom in a double well over three frozen atoms.
//!
//! `RGSADDLE_RIGID_LEGACY=1` selects the develop Gram–Schmidt (geometric
//! centroid, cutoff `1e-8`, frozen atoms ignored). Unset, the run uses
//! the column-pivoted QR. Rows go to stdout and, when
//! `RGSADDLE_BENCH_OUT` is set, to that file.
//!
//! ```text
//! cargo run --release --example rigid_rank_bench
//! RGSADDLE_RIGID_LEGACY=1 cargo run --release --example rigid_rank_bench
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, MinModeConfig, MinModeKind,
    MinModeSession, MinModeStatus, PointSurface, SaddleError, project_rigid_force,
    rigid_directions,
};

const SHIFT: [f64; 3] = [1.0e5, -4.0e4, 2.5e4];
const MASSES: [f64; 3] = [16.0, 1.0, 1.0];
/// Lennard-Jones 12-6, σ = 1, ε = 1. The linear trimer is stationary at
/// this nearest-neighbour distance and is the index-1 saddle between
/// the two equilateral wells.
const LJ_SADDLE: f64 = 1.1210299382816957;
const LJ_MIN: f64 = 1.122462048309373;

#[derive(Clone, Copy)]
enum System {
    Far,
    Near,
    Slab,
}

impl System {
    fn name(self) -> &'static str {
        match self {
            Self::Far => "lj-far",
            Self::Near => "near-linear",
            Self::Slab => "frozen-slab",
        }
    }

    fn masses(self) -> Option<&'static [f64]> {
        match self {
            Self::Slab => None,
            _ => Some(&MASSES),
        }
    }

    fn frozen(self) -> Option<&'static [bool]> {
        match self {
            Self::Slab => Some(&[true, true, true, false]),
            _ => None,
        }
    }
}

fn legacy() -> bool {
    std::env::var_os("RGSADDLE_RIGID_LEGACY").is_some_and(|v| v == "1")
}

fn ranker() -> &'static str {
    if legacy() { "develop" } else { "pivoted" }
}

fn lj_energy(x: &[f64]) -> f64 {
    let n = x.len() / 3;
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r2 = {
                let mut s = 0.0;
                for c in 0..3 {
                    let d = x[3 * i + c] - x[3 * j + c];
                    s += d * d;
                }
                s
            };
            let inv = 1.0 / r2;
            let s6 = inv * inv * inv;
            energy += 4.0 * (s6 * s6 - s6);
        }
    }
    energy
}

/// Sin of the bond angle at the middle atom. Zero when the trimer is linear.
fn bend_sin(x: &[f64]) -> f64 {
    let u = [x[0] - x[3], x[1] - x[4], x[2] - x[5]];
    let v = [x[6] - x[3], x[7] - x[4], x[8] - x[5]];
    let u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let cross2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    if u2 * v2 == 0.0 {
        0.0
    } else {
        (cross2 / (u2 * v2)).sqrt()
    }
}

fn grad_fd(energy: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let eps = 1e-6;
    let mut g = vec![0.0; x.len()];
    let mut plus = x.to_vec();
    let mut minus = x.to_vec();
    for i in 0..x.len() {
        plus[i] = x[i] + eps;
        minus[i] = x[i] - eps;
        g[i] = (energy(&plus) - energy(&minus)) / (2.0 * eps);
        plus[i] = x[i];
        minus[i] = x[i];
    }
    g
}

fn slab_energy(x: &[f64]) -> f64 {
    let (ax, ay, az) = (x[9], x[10], x[11]);
    (ax * ax - 1.0).powi(2) + 8.0 * ay * ay + 8.0 * (az - 1.2).powi(2)
}

fn energy_of(system: System, x: &[f64]) -> f64 {
    match system {
        System::Slab => slab_energy(x),
        _ => lj_energy(x),
    }
}

fn raw_grad(system: System, x: &[f64]) -> Vec<f64> {
    grad_fd(|y| energy_of(system, y), x)
}

struct Mol {
    system: System,
    calls: AtomicUsize,
    /// Project inside the point surface. The band projects on its own.
    project: bool,
}

impl Mol {
    fn new(system: System, project: bool) -> Self {
        Self {
            system,
            calls: AtomicUsize::new(0),
            project,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PointSurface for Mol {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let slice = x.as_slice().expect("contiguous");
        let e = energy_of(self.system, slice);
        let mut g = raw_grad(self.system, slice);
        if self.project {
            project_rigid_force(
                &mut g,
                slice,
                self.system.masses(),
                self.system.frozen(),
                true,
            );
        }
        Ok((e, Array1::from(g)))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let slice = x.as_slice().expect("contiguous");
        let mut modes = rigid_directions(slice, self.system.masses(), self.system.frozen(), true);
        if let Some(mask) = self.system.frozen() {
            let mut extra = Vec::new();
            for (atom, &frozen) in mask.iter().enumerate() {
                if !frozen {
                    continue;
                }
                for axis in 0..3 {
                    let mut row = vec![0.0; slice.len()];
                    row[3 * atom + axis] = 1.0;
                    extra.push(row);
                }
            }
            if !extra.is_empty() {
                let mut stacked = Array2::zeros((modes.nrows() + extra.len(), slice.len()));
                if modes.nrows() > 0 {
                    stacked
                        .slice_mut(ndarray::s![..modes.nrows(), ..])
                        .assign(&modes);
                }
                for (i, row) in extra.iter().enumerate() {
                    stacked
                        .row_mut(modes.nrows() + i)
                        .assign(&Array1::from_vec(row.clone()));
                }
                modes = stacked;
            }
        }
        Ok(modes)
    }
}

struct BandMol {
    system: System,
    calls: AtomicUsize,
}

impl BandSurface for BandMol {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let slice = row.as_slice().expect("contiguous row");
            energies[i] = energy_of(self.system, slice);
            let g = raw_grad(self.system, slice);
            gradients.row_mut(i).assign(&Array1::from(g));
        }
        Ok(())
    }
}

fn place_linear(bend: f64, shift: [f64; 3]) -> Array1<f64> {
    let mut x = Array1::zeros(9);
    let s = LJ_SADDLE;
    x[0] = -s + shift[0];
    x[1] = shift[1];
    x[2] = shift[2];
    x[3] = shift[0];
    x[4] = bend + shift[1];
    x[5] = shift[2];
    x[6] = s + shift[0];
    x[7] = shift[1];
    x[8] = shift[2];
    x
}

fn place_equilateral(sign: f64, shift: [f64; 3]) -> Array1<f64> {
    let a = LJ_MIN;
    let h = (a * a - 0.25 * a * a).sqrt();
    let mut x = Array1::zeros(9);
    x[0] = shift[0];
    x[1] = shift[1];
    x[2] = shift[2];
    x[3] = a + shift[0];
    x[4] = shift[1];
    x[5] = shift[2];
    x[6] = 0.5 * a + shift[0];
    x[7] = sign * h + shift[1];
    x[8] = shift[2];
    x
}

fn place_slab(adatom_x: f64) -> Array1<f64> {
    Array1::from(vec![
        0.0, 0.0, 0.0, 2.2, 0.0, 0.0, 0.4, 1.7, 0.0, adatom_x, 0.05, 1.2,
    ])
}

fn start(system: System) -> Array1<f64> {
    match system {
        // Between the bent well (half ≈ 0.806) and the linear saddle (half = 1).
        System::Far => place_linear(1.0e-4, SHIFT),
        System::Near => place_linear(1.0e-8, [0.0; 3]),
        System::Slab => place_slab(0.35),
    }
}

fn slab_shift(system: System, x: &[f64]) -> f64 {
    let Some(mask) = system.frozen() else {
        return 0.0;
    };
    let origin = start(system);
    let mut worst = 0.0_f64;
    for (atom, &frozen) in mask.iter().enumerate() {
        if !frozen {
            continue;
        }
        for c in 0..3 {
            worst = worst.max((x[3 * atom + c] - origin[3 * atom + c]).abs());
        }
    }
    worst
}

/// Linear saddle, or the adatom on the barrier and the slab unmoved.
fn saddle_ok(system: System, x: &[f64], converged: bool) -> bool {
    if !converged {
        return false;
    }
    match system {
        System::Slab => x[9].abs() < 0.05 && slab_shift(system, x) < 1e-6,
        _ => bend_sin(x) < 0.02,
    }
}

fn barrier(system: System, energy: f64) -> f64 {
    let reactant = match system {
        System::Slab => slab_energy(place_slab(1.0).as_slice().unwrap()),
        _ => lj_energy(place_equilateral(1.0, [0.0; 3]).as_slice().unwrap()),
    };
    energy - reactant
}

fn run_minmode(system: System, kind: MinModeKind) -> Row {
    let surface = Mol::new(system, system.frozen().is_some());
    let config = MinModeConfig {
        kind,
        force_tol: 1e-3,
        max_move: 0.05,
        max_rotations: 12,
        krylov_dim: 8,
        rotation_tol: 1e-3,
        ..MinModeConfig::default()
    };
    let x0 = start(system);
    let rank = rigid_directions(
        x0.as_slice().unwrap(),
        system.masses(),
        system.frozen(),
        true,
    )
    .nrows();
    let mut mode = Array1::zeros(x0.len());
    match system {
        System::Slab => mode[9] = 1.0,
        _ => mode[4] = 1.0,
    }
    let mut session = MinModeSession::new(config, x0.clone(), mode).unwrap();
    if session.estimate_mode(&surface).is_err() {
        return failed_row(system, method_name(kind), surface.calls(), rank);
    }
    let origin = x0.clone();
    let mut report = None;
    let mut worst_shift = 0.0_f64;
    for _ in 0..200 {
        let step = match session.step(&surface) {
            Ok(step) => step,
            Err(_) => return failed_row(system, method_name(kind), surface.calls(), rank),
        };
        worst_shift = worst_shift.max(slab_shift(system, session.position().as_slice().unwrap()));
        if let Some(mask) = system.frozen() {
            let mut pinned = session.position().to_owned();
            for (atom, &frozen) in mask.iter().enumerate() {
                if frozen {
                    for c in 0..3 {
                        pinned[3 * atom + c] = origin[3 * atom + c];
                    }
                }
            }
            session.set_position(pinned, None).unwrap();
        }
        let done = step.status == MinModeStatus::Converged;
        report = Some(step);
        if done {
            break;
        }
    }
    let report = report.unwrap();
    let pos = session.position().to_owned();
    let energy = energy_of(system, pos.as_slice().unwrap());
    let converged = report.status == MinModeStatus::Converged;
    Row {
        system: system.name(),
        method: method_name(kind),
        converged,
        force_calls: surface.calls(),
        curvature: report.curvature,
        barrier: barrier(system, energy),
        max_force: report.max_force,
        slab_shift: worst_shift,
        rigid_rank: rank,
        saddle_ok: saddle_ok(system, pos.as_slice().unwrap(), converged),
    }
}

fn band_ends(system: System) -> Array2<f64> {
    let (left, right) = match system {
        System::Slab => (place_slab(-1.0), place_slab(1.0)),
        System::Far => (
            place_equilateral(1.0, SHIFT),
            place_equilateral(-1.0, SHIFT),
        ),
        System::Near => (
            place_linear(1.0e-8, [0.0; 3]),
            place_linear(-1.0e-8, [0.0; 3]),
        ),
    };
    let n = 5;
    let mut band = Array2::zeros((n, left.len()));
    for image in 0..n {
        let t = image as f64 / (n - 1) as f64;
        for i in 0..left.len() {
            band[(image, i)] = (1.0 - t) * left[i] + t * right[i];
        }
    }
    band
}

fn method_name(kind: MinModeKind) -> &'static str {
    match kind {
        MinModeKind::Dimer => "dimer",
        MinModeKind::Lanczos => "lanczos",
    }
}

fn failed_row(system: System, method: &'static str, calls: usize, rank: usize) -> Row {
    Row {
        system: system.name(),
        method,
        converged: false,
        force_calls: calls,
        curvature: f64::NAN,
        barrier: f64::NAN,
        max_force: f64::NAN,
        slab_shift: f64::NAN,
        rigid_rank: rank,
        saddle_ok: false,
    }
}

fn run_neb(system: System) -> Row {
    let surface = BandMol {
        system,
        calls: AtomicUsize::new(0),
    };
    let images = band_ends(system);
    let rank = rigid_directions(
        images.row(0).as_slice().unwrap(),
        system.masses(),
        system.frozen(),
        true,
    )
    .nrows();
    let config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        min_mode: None,
        project_rotation: true,
        masses: system.masses().map(|m| m.to_vec()),
        frozen: system.frozen().map(|m| m.to_vec()),
        climbing: Some(CiConfig {
            trigger_factor: 1.0,
            trigger_force: f64::INFINITY,
        }),
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, images).unwrap();
    let mut report = None;
    for _ in 0..80 {
        let step = match session.step(&surface) {
            Ok(step) => step,
            Err(_) => {
                return failed_row(
                    system,
                    "ci-neb",
                    surface.calls.load(Ordering::Relaxed),
                    rank,
                );
            }
        };
        let done = step.status == BandStatus::Converged;
        report = Some(step);
        if done {
            break;
        }
    }
    let report = report.unwrap();
    let pos = session.positions().to_owned();
    let ci = report.ci_index.unwrap_or(2);
    let row = pos.row(ci).to_owned();
    let energy = energy_of(system, row.as_slice().unwrap());
    let converged = report.status == BandStatus::Converged;
    Row {
        system: system.name(),
        method: "ci-neb",
        converged,
        force_calls: surface.calls.load(Ordering::Relaxed),
        curvature: f64::NAN,
        barrier: barrier(system, energy),
        max_force: report.max_force,
        slab_shift: slab_shift(system, row.as_slice().unwrap()),
        rigid_rank: rank,
        saddle_ok: saddle_ok(system, row.as_slice().unwrap(), converged),
    }
}

struct Row {
    system: &'static str,
    method: &'static str,
    converged: bool,
    force_calls: usize,
    curvature: f64,
    barrier: f64,
    max_force: f64,
    slab_shift: f64,
    rigid_rank: usize,
    saddle_ok: bool,
}

fn fmt_row(row: &Row) -> String {
    format!(
        "{system},{method},{ranker},{converged},{calls},{curvature:.8},{barrier:.8},{force:.6e},{shift:.6e},{rank},{ok}",
        system = row.system,
        method = row.method,
        ranker = ranker(),
        converged = row.converged,
        calls = row.force_calls,
        curvature = row.curvature,
        barrier = row.barrier,
        force = row.max_force,
        shift = row.slab_shift,
        rank = row.rigid_rank,
        ok = row.saddle_ok,
    )
}

fn check_gradient() {
    let x = place_linear(0.12, [0.0; 3]);
    let g = raw_grad(System::Far, x.as_slice().unwrap());
    let eps = 1e-4;
    let mut plus = x.clone();
    plus[4] += eps;
    let mut minus = x.clone();
    minus[4] -= eps;
    let fd =
        (lj_energy(plus.as_slice().unwrap()) - lj_energy(minus.as_slice().unwrap())) / (2.0 * eps);
    assert!((g[4] - fd).abs() < 1e-6, "bend gradient {} vs {fd}", g[4]);
}

fn main() {
    check_gradient();
    let header = "system,method,ranker,converged,force_calls,curvature,barrier,max_force,slab_shift,rigid_rank,saddle_ok";
    let mut lines = vec![header.to_string()];
    for system in [System::Far, System::Near, System::Slab] {
        for kind in [MinModeKind::Dimer, MinModeKind::Lanczos] {
            let row = run_minmode(system, kind);
            let text = fmt_row(&row);
            println!("{text}");
            lines.push(text);
        }
        let row = run_neb(system);
        let text = fmt_row(&row);
        println!("{text}");
        lines.push(text);
    }
    if let Ok(path) = std::env::var("RGSADDLE_BENCH_OUT") {
        std::fs::write(&path, lines.join("\n") + "\n").expect("write csv");
    }
}
