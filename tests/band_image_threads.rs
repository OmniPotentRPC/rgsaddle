// Copyright (c) rgsaddle developers

//! Image threads split each band evaluation into contiguous image
//! blocks. On a surface whose rows are independent the trajectory is
//! bitwise that of one thread, and the first failing image is the one
//! reported.

use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandSurface, CiConfig, ProjectionKind, SaddleError, SpringKind,
};

const DOF: usize = 6;

/// Coupled double wells per coordinate. Records the row count of every
/// call and fails on any row whose first coordinate lies in `fail`.
#[derive(Default)]
struct Wells {
    calls: Mutex<Vec<usize>>,
    fail: Option<(f64, f64)>,
}

impl BandSurface for Wells {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.calls.lock().unwrap().push(positions.nrows());
        for (r, row) in positions.outer_iter().enumerate() {
            if let Some((low, high)) = self.fail
                && (low..high).contains(&row[0])
            {
                return Err(SaddleError::Invalid(format!("row at {}", row[0])));
            }
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

fn configs() -> Vec<(&'static str, BandConfig)> {
    let base = BandConfig {
        climbing: Some(CiConfig::after(0.5)),
        min_mode: None,
        max_move: 0.05,
        force_tol: 1e-6,
        ..BandConfig::default()
    };
    vec![
        ("cineb-uniform-fire", base.clone()),
        (
            "dneb-weighted-lbfgs",
            BandConfig {
                projection: ProjectionKind::DoublyNudged,
                spring: SpringKind::EnergyWeighted {
                    k_min: 1.0,
                    k_max: 5.0,
                },
                method: rgmin::Method::Lbfgs { memory: 5 },
                ..base.clone()
            },
        ),
        (
            "neb-om-live-fire",
            BandConfig {
                spring: SpringKind::OnsagerMachlupLive { k: 5.0 },
                climbing: None,
                ..base.clone()
            },
        ),
        (
            "ocineb-fire",
            BandConfig {
                min_mode: BandConfig::default().min_mode,
                ..base.clone()
            },
        ),
    ]
}

/// Every step's band and force bits, and the surface rows.
fn trajectory(config: &BandConfig, threads: usize, steps: usize) -> (Vec<u64>, usize) {
    let mut session = BandSession::new(config.clone(), band(11)).unwrap();
    session.set_image_threads(threads).unwrap();
    assert_eq!(session.image_threads(), threads);
    let surface = Wells::default();
    let mut bits = Vec::new();
    let mut rows = 0;
    for _ in 0..steps {
        let report = session.step(&surface).unwrap();
        rows += report.surface_rows;
        bits.push(report.max_force.to_bits());
        bits.extend(session.positions().iter().map(|v| v.to_bits()));
        if report.status != rgsaddle::BandStatus::Running {
            break;
        }
    }
    (bits, rows)
}

#[test]
fn image_threads_keep_the_trajectory_bitwise() {
    for (label, config) in configs() {
        let (serial, serial_rows) = trajectory(&config, 1, 80);
        for threads in [2, 3, 8, 16] {
            let (split, rows) = trajectory(&config, threads, 80);
            assert_eq!(split.len(), serial.len(), "{label} threads={threads}");
            assert!(
                split == serial,
                "{label} threads={threads}: trajectory bits differ"
            );
            assert_eq!(rows, serial_rows, "{label} threads={threads}");
        }
    }
}

#[test]
fn image_threads_call_the_surface_once_per_block() {
    let (_, config) = &configs()[0];
    let mut session = BandSession::new(config.clone(), band(11)).unwrap();
    session.set_image_threads(4).unwrap();
    let surface = Wells::default();
    session.step(&surface).unwrap();
    session.step(&surface).unwrap();
    let mut calls = surface.calls.lock().unwrap().clone();
    // Whole band of 11 rows in blocks of 3, 3, 3, 2, then each call on
    // the nine interior rows in blocks of 3, 2, 2, 2, in any finishing
    // order.
    let (whole, interior) = calls.split_at_mut(4);
    whole.sort_unstable();
    assert_eq!(whole, [2, 3, 3, 3]);
    assert!(!interior.is_empty() && interior.len() % 4 == 0);
    for call in interior.chunks_mut(4) {
        call.sort_unstable();
        assert_eq!(call, [2, 2, 2, 3]);
    }
}

#[test]
fn image_threads_report_the_first_failing_image() {
    let (_, config) = &configs()[0];
    // Images 3 to 7 of 11 have first coordinates in this window;
    // one thread meets image 3 first.
    let positions = band(11);
    let low = positions[(3, 0)] - 1e-12;
    let high = positions[(7, 0)] + 1e-12;
    let mut serial = BandSession::new(config.clone(), positions.clone()).unwrap();
    let surface = Wells {
        fail: Some((low, high)),
        ..Wells::default()
    };
    let expected = serial.step(&surface).unwrap_err().to_string();
    for threads in [2, 3, 8] {
        let mut split = BandSession::new(config.clone(), positions.clone()).unwrap();
        split.set_image_threads(threads).unwrap();
        let err = split.step(&surface).unwrap_err().to_string();
        assert_eq!(err, expected, "threads={threads}");
    }
}

#[test]
fn zero_image_threads_are_refused() {
    let mut session = BandSession::new(BandConfig::default(), band(5)).unwrap();
    assert!(session.set_image_threads(0).is_err());
    assert_eq!(session.image_threads(), 1);
}
