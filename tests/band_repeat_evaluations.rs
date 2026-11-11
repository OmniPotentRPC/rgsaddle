// Copyright (c) rgsaddle developers

//! A band the session already evaluated is not handed to the surface
//! again: the trust-region stepper reads the products a rejected step
//! took around the same base, the base itself comes back from the
//! cache after the rejection, and a stepper that returns to the point
//! before its trial finds it in the cache. The trajectories are those
//! of the parent, pinned by hashes taken on it.

use std::collections::HashMap;
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::rtr::RtrConfig;
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, CiConfig, SaddleError};

const DOF: usize = 6;

/// Coupled double wells; counts each band it is handed, by bits.
#[derive(Default)]
struct Wells {
    seen: Mutex<HashMap<Vec<u64>, usize>>,
}

impl Wells {
    fn repeated_calls(&self) -> usize {
        self.seen.lock().unwrap().values().map(|n| n - 1).sum()
    }
}

impl BandSurface for Wells {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let key = positions.iter().map(|v| v.to_bits()).collect();
        *self.seen.lock().unwrap().entry(key).or_insert(0) += 1;
        for (r, row) in positions.outer_iter().enumerate() {
            let mut e = 0.0;
            for j in 0..DOF {
                let w = row[j] * row[j] - 1.0;
                e += w * w;
                gradients[(r, j)] = 4.0 * row[j] * w;
            }
            for j in 1..DOF {
                e += 0.3 * row[j] * row[j - 1];
                gradients[(r, j)] += 0.3 * row[j - 1];
                gradients[(r, j - 1)] += 0.3 * row[j];
            }
            energies[r] = e;
        }
        Ok(())
    }
}

fn band(n_images: usize) -> Array2<f64> {
    let start: [f64; DOF] = [-1.0, -0.9, -1.1, -1.0, -0.95, -1.05];
    let end: [f64; DOF] = [1.0, 1.1, 0.9, 1.05, 1.0, 0.95];
    Array2::from_shape_fn((n_images, DOF), |(i, c)| {
        let t = i as f64 / (n_images - 1) as f64;
        let bow = 0.2 * (std::f64::consts::PI * t).sin() * if c % 2 == 0 { 1.0 } else { -1.0 };
        (1.0 - t) * start[c] + t * end[c] + bow
    })
}

fn fnv1a(hash: &mut u64, value: f64) {
    for byte in value.to_bits().to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Run to convergence or `steps`; the trajectory hash, the summed
/// surface rows, and the surface.
fn run(config: BandConfig, rtr: Option<RtrConfig>, steps: usize) -> (u64, usize, Wells) {
    let mut session = BandSession::new(config, band(7)).unwrap();
    session.set_rtr(rtr).unwrap();
    let surface = Wells::default();
    let mut hash = 0xcbf2_9ce4_8422_2325;
    let mut rows = 0;
    for _ in 0..steps {
        let report = match session.step(&surface) {
            Ok(report) => report,
            Err(_) => break,
        };
        rows += report.surface_rows;
        fnv1a(&mut hash, report.max_force);
        session
            .positions()
            .iter()
            .for_each(|&v| fnv1a(&mut hash, v));
        if report.status != BandStatus::Running {
            break;
        }
    }
    (hash, rows, surface)
}

fn base() -> BandConfig {
    BandConfig {
        climbing: Some(CiConfig::after(0.5)),
        min_mode: None,
        max_move: 0.1,
        force_tol: 1e-6,
        ..BandConfig::default()
    }
}

#[test]
fn the_trust_region_stepper_evaluates_each_band_once() {
    let (hash, rows, surface) = run(base(), Some(RtrConfig::default()), 200);
    assert_eq!(surface.repeated_calls(), 0, "rows {rows}");
    assert_eq!(
        hash, 0x2093_81b9_2590_5c88,
        "rtr trajectory {hash:#018x}, rows {rows}"
    );
}

/// LJ38 with cutoff 40, the kernel of `connect_driver.rs`; counts
/// each band it is handed, by bits.
#[derive(Default)]
struct Lj38 {
    seen: Mutex<HashMap<Vec<u64>, usize>>,
}

fn lj(x: &[f64], gradient: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    let rc = 40.0;
    let mut energy = 0.0;
    gradient.fill(0.0);
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
    energy
}

impl BandSurface for Lj38 {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let key = positions.iter().map(|v| v.to_bits()).collect();
        *self.seen.lock().unwrap().entry(key).or_insert(0) += 1;
        for (r, row) in positions.outer_iter().enumerate() {
            let x = row.as_slice().expect("contiguous rows");
            let g = gradients.row_mut(r).into_slice().expect("contiguous rows");
            energies[r] = lj(x, g);
        }
        Ok(())
    }
}

fn read_xyz(path: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut coords = Vec::new();
    for line in text.lines().skip(2) {
        let mut parts = line.split_whitespace().skip(1);
        for _ in 0..3 {
            coords.push(parts.next().unwrap().parse().unwrap());
        }
    }
    coords
}

/// ODE12r on the LJ38 pair 0 band ends rejecting trial after trial
/// from one point until it reports the stall; each rejection returns
/// to that point.
#[test]
fn ode12r_evaluates_each_band_once() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lj38");
    let a = read_xyz(&format!("{root}/start_0.xyz"));
    let b = read_xyz(&format!("{root}/end_0.xyz"));
    let n_images = 9;
    let positions = Array2::from_shape_fn((n_images, a.len()), |(i, c)| {
        let t = i as f64 / (n_images - 1) as f64;
        (1.0 - t) * a[c] + t * b[c]
    });
    let config = BandConfig {
        method: rgmin::Method::Ode12r,
        climbing: Some(CiConfig::after(0.5)),
        min_mode: None,
        max_move: 0.1,
        force_tol: 1e-3,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, positions).unwrap();
    let surface = Lj38::default();
    let mut hash = 0xcbf2_9ce4_8422_2325;
    let mut rows = 0;
    let mut stalled = false;
    for _ in 0..3000 {
        match session.step(&surface) {
            Ok(report) => {
                rows += report.surface_rows;
                fnv1a(&mut hash, report.max_force);
                session
                    .positions()
                    .iter()
                    .for_each(|&v| fnv1a(&mut hash, v));
                if report.status != BandStatus::Running {
                    break;
                }
            }
            Err(_) => {
                stalled = true;
                break;
            }
        }
    }
    let repeated: usize = surface.seen.lock().unwrap().values().map(|n| n - 1).sum();
    assert!(stalled, "the fixture ends in the stall");
    assert_eq!(repeated, 0, "rows {rows}");
    assert_eq!(
        hash, 0xca90_c1d4_6f6f_bbf3,
        "ode12r trajectory {hash:#018x}, rows {rows}"
    );
}
