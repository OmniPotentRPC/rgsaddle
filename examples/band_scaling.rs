// Copyright (c) rgsaddle developers

//! Timed band and min-mode runs on an adatom hop of a Lennard-Jones
//! cluster, for the scaling figures of `howto/band` and
//! `howto/minmode`.
//!
//! ```text
//! cargo run --release --example band_scaling -- \
//!     band <atoms> <interior> <threads> <steps> [fire|ode12r|rtr]
//! cargo run --release --example band_scaling -- \
//!     minmode <atoms> <dimer|lanczos> <steps>
//! ```
//!
//! The cluster is the `<atoms>` sites of a face-centred cubic lattice
//! nearest the origin, at the pair-minimum spacing. The atom farthest
//! from the origin hops to the free nearest-neighbour site with the
//! most neighbours. Both states are relaxed by capped steepest descent
//! before the clock starts.
//!
//! `band` runs `<steps>` steps of the climbing image NEB on the linear
//! path of `<interior>` interior images, its surface split over
//! `<threads>` image threads. The method is FIRE (the default),
//! ODE12r, or the projected trust-region band method (RTR). `minmode`
//! runs up to `<steps>` steps of the min-mode search from the midpoint
//! of that path, seeded with the hop direction. Each run prints one
//! line: the inputs, the steps taken, the surface rows or evaluations,
//! the largest force, the wall-clock seconds of the steps and a hash of
//! the forces and the final positions.

use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, MinModeConfig, MinModeKind,
    MinModeSession, MinModeStatus, PointSurface, SaddleError,
};

/// The Lennard-Jones pair minimum, `2^(1/6)` in reduced units.
fn pair_minimum() -> f64 {
    2f64.powf(1.0 / 6.0)
}

/// Steepest-descent step per unit force, below `2 / lambda_max` of a
/// close-packed Lennard-Jones solid.
const RELAX_RATE: f64 = 1e-3;
/// Largest per-atom move of one relaxation step.
const RELAX_CAP: f64 = 0.02;
const RELAX_STEPS: usize = 500;

/// All-pairs Lennard-Jones cluster in reduced units.
struct Cluster;

fn lennard_jones(x: ArrayView1<f64>, mut g: ndarray::ArrayViewMut1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut energy = 0.0;
    g.fill(0.0);
    for i in 0..n {
        for j in i + 1..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let inv_r2 = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            let sr6 = inv_r2 * inv_r2 * inv_r2;
            energy += 4.0 * (sr6 * sr6 - sr6);
            let s = 24.0 * inv_r2 * (sr6 - 2.0 * sr6 * sr6);
            for c in 0..3 {
                g[3 * i + c] += s * d[c];
                g[3 * j + c] -= s * d[c];
            }
        }
    }
    energy
}

impl BandSurface for Cluster {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (r, row) in positions.outer_iter().enumerate() {
            energies[r] = lennard_jones(row, gradients.row_mut(r));
        }
        Ok(())
    }
}

impl PointSurface for Cluster {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut g = Array1::zeros(x.len());
        let e = lennard_jones(x, g.view_mut());
        Ok((e, g))
    }
}

fn fcc_cluster(atoms: usize) -> Vec<[f64; 3]> {
    let a = pair_minimum() * 2f64.sqrt();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let span = (atoms as f64 / 4.0).cbrt().ceil() as i64 + 1;
    let mut sites = Vec::new();
    for i in -span..=span {
        for j in -span..=span {
            for k in -span..=span {
                for b in basis {
                    sites.push([
                        a * (i as f64 + b[0]),
                        a * (j as f64 + b[1]),
                        a * (k as f64 + b[2]),
                    ]);
                }
            }
        }
    }
    let r2 = |p: &[f64; 3]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    sites.sort_by(|p, q| {
        r2(p)
            .total_cmp(&r2(q))
            .then(p[0].total_cmp(&q[0]))
            .then(p[1].total_cmp(&q[1]))
            .then(p[2].total_cmp(&q[2]))
    });
    sites.truncate(atoms);
    sites
}

fn distance(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

/// The end state: the last (outermost) site moved to the free
/// nearest-neighbour site with the most neighbours, the first such
/// site in a fixed order on a tie.
fn hop(sites: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let r0 = pair_minimum();
    let half = r0 / 2f64.sqrt();
    let mover = sites.len() - 1;
    let from = sites[mover];
    let others = &sites[..mover];
    let mut best: Option<([f64; 3], usize)> = None;
    for (p, q) in [(0, 1), (0, 2), (1, 2)] {
        for (sp, sq) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let mut to = from;
            to[p] += sp * half;
            to[q] += sq * half;
            let nearest = others
                .iter()
                .map(|o| distance(o, &to))
                .fold(f64::INFINITY, f64::min);
            if nearest < 0.5 * r0 {
                continue;
            }
            let neighbours = others
                .iter()
                .filter(|o| distance(o, &to) < 1.1 * r0)
                .count();
            if best.is_none_or(|(_, n)| neighbours > n) {
                best = Some((to, neighbours));
            }
        }
    }
    let mut end = sites.to_vec();
    end[mover] = best.expect("an outermost site has a free neighbour site").0;
    end
}

fn relax(sites: &[[f64; 3]]) -> Array1<f64> {
    let mut x: Array1<f64> = sites.iter().flatten().copied().collect();
    let mut g = Array1::zeros(x.len());
    for _ in 0..RELAX_STEPS {
        lennard_jones(x.view(), g.view_mut());
        let largest = g
            .exact_chunks(3)
            .into_iter()
            .map(|f| f.dot(&f).sqrt())
            .fold(0.0, f64::max);
        let rate = RELAX_RATE.min(RELAX_CAP / largest.max(f64::MIN_POSITIVE));
        Zip::from(&mut x).and(&g).for_each(|x, &g| *x -= rate * g);
    }
    x
}

fn endpoints(atoms: usize) -> (Array1<f64>, Array1<f64>) {
    let sites = fcc_cluster(atoms);
    (relax(&sites), relax(&hop(&sites)))
}

fn fnv(hash: &mut u64, bits: u64) {
    for byte in bits.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn arg(args: &[String], i: usize, what: &str) -> usize {
    args.get(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("argument {i} is {what}"))
}

fn band(
    atoms: usize,
    interior: usize,
    threads: usize,
    steps: usize,
    method: &str,
) -> Result<(), SaddleError> {
    let (start, end) = endpoints(atoms);
    let images = interior + 2;
    let mut path = Array2::zeros((images, start.len()));
    for (i, mut row) in path.outer_iter_mut().enumerate() {
        let t = i as f64 / (images - 1) as f64;
        Zip::from(&mut row)
            .and(&start)
            .and(&end)
            .for_each(|x, &a, &b| *x = (1.0 - t) * a + t * b);
    }
    let mut config = BandConfig {
        climbing: Some(CiConfig::after(0.5)),
        min_mode: None,
        force_tol: f64::MIN_POSITIVE,
        ..BandConfig::default()
    };
    match method {
        "fire" | "rtr" => {}
        "ode12r" => config.method = rgmin::Method::Ode12r,
        other => panic!("unknown band method {other}"),
    }
    let mut session = BandSession::new(config, path)?;
    if method == "rtr" {
        session.set_rtr(Some(rgsaddle::rtr::RtrConfig::default()))?;
    }
    if threads > 1 {
        session.set_image_threads(threads)?;
    }
    let mut hash = FNV_OFFSET;
    let mut rows = 0;
    let mut taken = 0;
    let mut force = f64::NAN;
    let clock = Instant::now();
    while taken < steps {
        let report = session.step(&Cluster)?;
        rows += report.surface_rows;
        taken = report.iteration;
        force = report.max_force;
        fnv(&mut hash, force.to_bits());
        if report.status != BandStatus::Running {
            break;
        }
    }
    let seconds = clock.elapsed().as_secs_f64();
    session
        .positions()
        .iter()
        .for_each(|v| fnv(&mut hash, v.to_bits()));
    println!(
        "band atoms={atoms} interior={interior} threads={threads} method={method} steps={taken} \
         rows={rows} fmax={force:.6e} seconds={seconds:.6} hash={hash:016x}"
    );
    Ok(())
}

fn minmode(atoms: usize, kind: &str, steps: usize) -> Result<(), SaddleError> {
    let (start, end) = endpoints(atoms);
    let kind = match kind {
        "dimer" => MinModeKind::Dimer,
        "lanczos" => MinModeKind::Lanczos,
        other => panic!("unknown min-mode kind {other}"),
    };
    let x0 = (&start + &end) * 0.5;
    let config = MinModeConfig {
        kind,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, x0, &end - &start)?;
    let mut hash = FNV_OFFSET;
    let mut evaluations = 0;
    let mut taken = 0;
    let mut force = f64::NAN;
    let clock = Instant::now();
    while taken < steps {
        let report = session.step(&Cluster)?;
        evaluations += report.evaluations;
        taken = report.iteration;
        force = report.max_force;
        fnv(&mut hash, force.to_bits());
        fnv(&mut hash, report.curvature.to_bits());
        if report.status != MinModeStatus::Running {
            break;
        }
    }
    let seconds = clock.elapsed().as_secs_f64();
    session
        .position()
        .iter()
        .for_each(|v| fnv(&mut hash, v.to_bits()));
    println!(
        "minmode atoms={atoms} kind={kind:?} steps={taken} evaluations={evaluations} \
         fmax={force:.6e} seconds={seconds:.6} hash={hash:016x}"
    );
    Ok(())
}

fn main() -> Result<(), SaddleError> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("band") => band(
            arg(&args, 2, "the atom count"),
            arg(&args, 3, "the interior image count"),
            arg(&args, 4, "the image thread count"),
            arg(&args, 5, "the step count"),
            args.get(6).map_or("fire", String::as_str),
        ),
        Some("minmode") => minmode(
            arg(&args, 2, "the atom count"),
            args.get(3).map_or("dimer", String::as_str),
            arg(&args, 4, "the step count"),
        ),
        _ => panic!("usage: band_scaling band|minmode ..."),
    }
}
