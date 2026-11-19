//! The default band stepper is FIRE until the band's force is at or under
//! the hand-over force, then the capped eOn-style L-BFGS. A band that
//! starts under that force never runs FIRE and spends the rows of an
//! explicit L-BFGS, and a band that starts over it still beats FIRE
//! alone while finding the same saddle.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Method};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, CiConfig, Finish, SaddleError};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2 over four atoms that do not touch.
struct Wells {
    rows: AtomicUsize,
}

impl BandSurface for Wells {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        for (i, row) in positions.outer_iter().enumerate() {
            let mut e = 0.0;
            for atom in 0..row.len() / 3 {
                let (x, y, z) = (row[3 * atom], row[3 * atom + 1], row[3 * atom + 2]);
                e += (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
                gradients[(i, 3 * atom)] = 4.0 * x * (x * x - 1.0);
                gradients[(i, 3 * atom + 1)] = 4.0 * y;
                gradients[(i, 3 * atom + 2)] = 4.0 * z;
            }
            energies[i] = e;
        }
        Ok(())
    }
}

fn initial_band(n_images: usize, bow_height: f64) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 12));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        for atom in 0..4 {
            let bow = bow_height * (std::f64::consts::PI * t).sin();
            band[(i, 3 * atom)] = -1.0 + 2.0 * t;
            band[(i, 3 * atom + 1)] = bow * (1.0 + atom as f64);
            band[(i, 3 * atom + 2)] = -0.5 * bow;
        }
    }
    band
}

struct Run {
    rows: usize,
    barrier: f64,
}

fn run(config: BandConfig, bow_height: f64) -> Run {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(config, initial_band(9, bow_height)).unwrap();
    for _ in 0..4000 {
        let report = session.step(&surface).unwrap();
        if report.status == BandStatus::Converged {
            let barrier = session
                .positions()
                .outer_iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(k, v)| match k % 3 {
                            0 => (v * v - 1.0).powi(2),
                            _ => 2.0 * v * v,
                        })
                        .sum::<f64>()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            return Run {
                rows: surface.rows.load(Ordering::Relaxed),
                barrier,
            };
        }
    }
    panic!("the band did not converge");
}

fn plain(climbing: Option<CiConfig>) -> BandConfig {
    BandConfig::default()
        .with_force_tol(1e-2)
        .with_climbing(climbing)
        .with_min_mode(None)
        .with_saddle_check(None)
}

const TAME: f64 = 0.3;
const STIFF: f64 = 3.0;

fn fire() -> Method {
    Method::fire(FireKind::V2)
}

#[test]
fn the_default_stepper_hands_over_from_fire_to_lbfgs() {
    let finish = BandConfig::default().finish.expect("a finish by default");
    assert!(matches!(finish.method, Method::Lbfgs { memory: 20, .. }));
    assert_eq!(finish.force.to_bits(), 5.0_f64.to_bits());
    assert!(matches!(BandConfig::default().method, Method::Fire { .. }));
}

#[test]
fn a_band_under_the_hand_over_force_spends_the_rows_of_lbfgs() {
    for climbing in [None, Some(CiConfig::after(0.5))] {
        let default = run(plain(climbing), TAME);
        let explicit = run(
            plain(climbing)
                .with_method(Method::lbfgs_with_memory(20))
                .with_finish(None),
            TAME,
        );
        assert_eq!(default.rows, explicit.rows, "climbing {climbing:?}");
    }
}

#[test]
fn a_band_under_the_hand_over_force_needs_under_half_the_rows_of_fire() {
    for climbing in [None, Some(CiConfig::after(0.5))] {
        let default = run(plain(climbing), TAME);
        let fire_only = run(plain(climbing).with_method(fire()).with_finish(None), TAME);
        assert!(
            2 * default.rows < fire_only.rows,
            "default {} rows, FIRE {} rows, climbing {climbing:?}",
            default.rows,
            fire_only.rows
        );
    }
}

#[test]
fn a_band_over_the_hand_over_force_beats_fire_and_finds_its_saddle() {
    for climbing in [None, Some(CiConfig::after(0.5))] {
        let default = run(plain(climbing), STIFF);
        let fire_only = run(plain(climbing).with_finish(None), STIFF);
        assert!(
            default.rows < fire_only.rows,
            "default {} rows, FIRE {} rows, climbing {climbing:?}",
            default.rows,
            fire_only.rows
        );
        assert!(
            (default.barrier - fire_only.barrier).abs() < 1e-3,
            "default {} eV, FIRE {} eV, climbing {climbing:?}",
            default.barrier,
            fire_only.barrier
        );
    }
}

#[test]
fn a_finish_force_the_band_never_reaches_leaves_fire_to_the_end() {
    let never = Finish::default().with_force(1e-9);
    let hybrid = run(plain(None).with_finish(never), STIFF);
    let fire_only = run(plain(None).with_finish(None), STIFF);
    assert_eq!(hybrid.rows, fire_only.rows);
}

#[test]
fn a_finish_force_that_is_not_positive_is_rejected() {
    let bad = plain(None).with_finish(Finish::default().with_force(0.0));
    assert!(BandSession::new(bad, initial_band(9, TAME)).is_err());
}
