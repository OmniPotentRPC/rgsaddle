//! The band refine stepper mirrors eOn's `[Refine]` section. The
//! default is no refine stepper. [`StepRefine::default`] uses L-BFGS
//! with memory 20 at a threshold of 0.5, and each step uses that
//! stepper when the convergence force is at or under the threshold
//! and the primary stepper otherwise.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::{FireKind, Method};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, RtrConfig, SaddleError, StepRefine,
    StringConfig,
};

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
            let mut energy = 0.0;
            for atom in 0..row.len() / 3 {
                let (x, y, z) = (row[3 * atom], row[3 * atom + 1], row[3 * atom + 2]);
                energy += (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
                gradients[(i, 3 * atom)] = 4.0 * x * (x * x - 1.0);
                gradients[(i, 3 * atom + 1)] = 4.0 * y;
                gradients[(i, 3 * atom + 2)] = 4.0 * z;
            }
            energies[i] = energy;
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

fn lbfgs() -> StepRefine {
    StepRefine::default().with_method(Method::lbfgs_with_memory(20))
}

fn fire() -> Method {
    Method::fire(FireKind::V2)
}

#[test]
fn the_defaults_are_no_refine_stepper_and_a_threshold_of_one_half() {
    let config = BandConfig::default();
    assert!(config.refine.is_none());
    assert_eq!(StepRefine::default().threshold.to_bits(), 0.5_f64.to_bits());
    assert!(matches!(StepRefine::default().method, Method::Lbfgs { .. }));
    assert!(matches!(config.method, Method::Fire { .. }));
}

#[test]
fn without_a_refine_stepper_the_band_is_pure_fire() {
    for climbing in [None, Some(CiConfig::after(0.5))] {
        let default = run(plain(climbing), STIFF);
        let explicit = run(plain(climbing).with_method(fire()).with_refine(None), STIFF);
        assert_eq!(default.rows, explicit.rows, "climbing {climbing:?}");
    }
}

#[test]
fn a_threshold_over_every_force_steps_with_the_refine_stepper_alone() {
    for climbing in [None, Some(CiConfig::after(0.5))] {
        let refined = run(
            plain(climbing).with_refine(lbfgs().with_threshold(f64::MAX)),
            STIFF,
        );
        let alone = run(
            plain(climbing).with_method(Method::lbfgs_with_memory(20)),
            STIFF,
        );
        assert_eq!(refined.rows, alone.rows, "climbing {climbing:?}");
        assert_eq!(refined.barrier.to_bits(), alone.barrier.to_bits());
    }
}

#[test]
fn a_threshold_under_every_force_steps_with_the_primary_stepper_alone() {
    let never = run(
        plain(None).with_refine(lbfgs().with_threshold(f64::MIN_POSITIVE)),
        STIFF,
    );
    let fire_only = run(plain(None), STIFF);
    assert_eq!(never.rows, fire_only.rows);
    assert_eq!(never.barrier.to_bits(), fire_only.barrier.to_bits());
}

#[test]
fn a_force_exactly_at_the_threshold_uses_the_refine_stepper() {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let mut probe = BandSession::new(plain(None), initial_band(9, STIFF)).unwrap();
    let threshold = probe.step(&surface).unwrap().max_force;
    for (threshold, refining) in [(threshold, true), (threshold.next_down(), false)] {
        let config = plain(None).with_refine(lbfgs().with_threshold(threshold));
        let mut session = BandSession::new(config, initial_band(9, STIFF)).unwrap();
        session.step(&surface).unwrap();
        assert!(!session.is_refining());
        session.step(&surface).unwrap();
        assert_eq!(session.is_refining(), refining, "threshold {threshold}");
    }
}

#[test]
fn a_force_rising_back_over_the_threshold_returns_to_fire() {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let config = plain(None).with_refine(lbfgs());
    let mut session = BandSession::new(config, initial_band(9, TAME)).unwrap();
    let mut refined = false;
    for _ in 0..4000 {
        session.step(&surface).unwrap();
        if session.is_refining() {
            refined = true;
            break;
        }
    }
    assert!(refined, "the relaxing band never reached the threshold");

    session.set_positions(initial_band(9, STIFF)).unwrap();
    let report = session.step(&surface).unwrap();
    assert!(report.max_force > 0.5);
    assert!(
        !session.is_refining(),
        "a force over the threshold uses FIRE"
    );

    let mut again = false;
    for _ in 0..4000 {
        session.step(&surface).unwrap();
        if session.is_refining() {
            again = true;
            break;
        }
    }
    assert!(again, "the band never came back under the threshold");
}

#[test]
fn each_stepper_keeps_its_own_state_across_switches() {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let config = plain(None).with_refine(lbfgs());
    let mut session = BandSession::new(config.clone(), initial_band(9, TAME)).unwrap();
    for _ in 0..4000 {
        session.step(&surface).unwrap();
        if session.is_refining() {
            break;
        }
    }
    for _ in 0..3 {
        session.step(&surface).unwrap();
    }
    let resting = session.positions().to_owned();
    session.set_positions(initial_band(9, STIFF)).unwrap();
    session.step(&surface).unwrap();
    session.set_positions(resting.clone()).unwrap();
    session.step(&surface).unwrap();
    assert!(session.is_refining());
    let kept = session.positions().to_owned();

    let mut fresh = BandSession::new(config, resting).unwrap();
    fresh.step(&surface).unwrap();
    assert!(fresh.is_refining());
    assert_ne!(kept, fresh.positions().to_owned());
}

#[test]
fn restart_returns_the_band_to_the_primary_stepper_state() {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let config = plain(None).with_refine(lbfgs().with_threshold(f64::MAX));
    let mut session = BandSession::new(config, initial_band(9, TAME)).unwrap();
    session.step(&surface).unwrap();
    assert!(session.is_refining());
    session.restart();
    assert!(!session.is_refining());
}

#[test]
fn a_refined_band_beats_fire_and_finds_the_same_saddle() {
    for climbing in [None, Some(CiConfig::after(0.5))] {
        let refined = run(plain(climbing).with_refine(lbfgs()), STIFF);
        let fire_only = run(plain(climbing), STIFF);
        assert!(
            refined.rows < fire_only.rows,
            "refined {} rows, FIRE {} rows, climbing {climbing:?}",
            refined.rows,
            fire_only.rows
        );
        assert!(
            (refined.barrier - fire_only.barrier).abs() < 1e-3,
            "refined {} , FIRE {} , climbing {climbing:?}",
            refined.barrier,
            fire_only.barrier
        );
    }
}

#[test]
fn a_tame_band_needs_under_half_the_rows_of_fire() {
    let refined = run(plain(None).with_refine(lbfgs()), TAME);
    let fire_only = run(plain(None), TAME);
    assert!(
        2 * refined.rows < fire_only.rows,
        "refined {} rows, FIRE {} rows",
        refined.rows,
        fire_only.rows
    );
}

#[test]
fn a_refine_threshold_that_is_not_positive_is_rejected() {
    for bad in [0.0, -1.0, f64::NAN] {
        let config = plain(None).with_refine(lbfgs().with_threshold(bad));
        assert!(BandSession::new(config, initial_band(9, TAME)).is_err());
    }
}

#[test]
fn a_band_without_a_refine_stepper_has_nothing_to_reject() {
    let config = plain(None).with_refine(None);
    assert!(BandSession::new(config, initial_band(9, TAME)).is_ok());
}

#[test]
fn a_refine_stepper_that_needs_more_than_the_force_is_rejected() {
    let config = plain(None).with_refine(lbfgs().with_method(Method::Bfgs));
    assert!(BandSession::new(config, initial_band(9, TAME)).is_err());
}

fn largest_difference(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    (&a - &b).iter().map(|v| v.abs()).fold(0.0, f64::max)
}

#[test]
fn a_band_moved_over_the_threshold_takes_the_step_of_a_fresh_fire_band() {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let mut fresh =
        BandSession::new(plain(None).with_max_move(10.0), initial_band(9, STIFF)).unwrap();
    fresh.step(&surface).unwrap();

    let moved = |threshold: Option<f64>| {
        let spec = match threshold {
            Some(threshold) => lbfgs().with_threshold(threshold),
            None => lbfgs(),
        };
        let mut session = BandSession::new(
            plain(None).with_max_move(10.0).with_refine(spec),
            initial_band(9, 0.01),
        )
        .unwrap();
        loop {
            if session.step(&surface).unwrap().status == BandStatus::Converged {
                break;
            }
        }
        assert!(session.is_refining());
        session.set_positions(initial_band(9, STIFF)).unwrap();
        session.step(&surface).unwrap();
        session
    };

    let back = moved(None);
    assert!(!back.is_refining());
    assert!(largest_difference(back.positions(), fresh.positions()) < 1e-12);

    let kept = moved(Some(f64::MAX));
    assert!(kept.is_refining());
    assert!(largest_difference(kept.positions(), fresh.positions()) > 1e-6);
}

fn after_steps(config: BandConfig, rtr: bool, steps: usize) -> BandSession {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(config, initial_band(9, STIFF)).unwrap();
    if rtr {
        session
            .set_rtr(Some(RtrConfig::default().with_radius_max(0.2)))
            .unwrap();
    }
    for _ in 0..steps {
        session.step(&surface).unwrap();
        assert!(!session.is_refining());
    }
    session
}

#[test]
fn string_and_trust_region_bands_ignore_the_refine_stepper() {
    let refined = |config: BandConfig| config.with_refine(lbfgs().with_threshold(f64::MAX));
    let string = plain(None).with_string(StringConfig::linear());
    let with = after_steps(refined(string), false, 12);
    let without = after_steps(plain(None).with_string(StringConfig::linear()), false, 12);
    assert_eq!(with.positions(), without.positions());

    let with = after_steps(refined(plain(None)), true, 12);
    let without = after_steps(plain(None), true, 12);
    assert_eq!(with.positions(), without.positions());
}

/// One step under a threshold that covers every force is the L-BFGS
/// step, not the FIRE step. Removing the refine stepper makes these
/// positions equal.
#[test]
fn a_step_under_the_threshold_is_not_the_primary_fire_step() {
    let surface = Wells {
        rows: AtomicUsize::new(0),
    };
    let start = initial_band(9, TAME);
    let mut fire_only = BandSession::new(plain(None).with_max_move(0.1), start.clone()).unwrap();
    let mut refined = BandSession::new(
        plain(None)
            .with_max_move(0.1)
            .with_refine(lbfgs().with_threshold(f64::MAX)),
        start,
    )
    .unwrap();
    fire_only.step(&surface).unwrap();
    refined.step(&surface).unwrap();
    assert!(refined.is_refining());
    assert!(
        largest_difference(fire_only.positions(), refined.positions()) > 1e-8,
        "the refine stepper did not change the step"
    );
}
