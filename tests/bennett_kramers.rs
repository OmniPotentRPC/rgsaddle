//! Bennett–Chandler shooting against a transmission coefficient that is known.
//!
//! Chandler, J. Chem. Phys. 68, 2959 (1978), doi:10.1063/1.436049.
//! The high-friction comparison is Kramers, Physica 7, 284 (1940):
//! `κ = sqrt(1+α²) - α` with `α = γ / (2 ω_b)`.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{PointSurface, SaddleError, ShootingConfig, bennett_chandler};

struct Quartic;

impl PointSurface for Quartic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let coordinate = x[0];
        let energy = (coordinate * coordinate - 1.0).powi(2);
        let gradient = 4.0 * coordinate * (coordinate * coordinate - 1.0);
        Ok((energy, Array1::from_vec(vec![gradient])))
    }
}

fn shoot(friction: f64, steps: usize, shots: usize, plateau: usize) -> rgsaddle::Transmission {
    let config = ShootingConfig {
        thermal: 0.05,
        mass: 1.0,
        friction,
        time_step: 0.002,
        steps,
        record_every: 25,
        shots,
        blocks: 4,
        seed: 7,
        plateau_record: plateau,
    };
    let starts = vec![Array1::zeros(1); 4];
    bennett_chandler(
        &Quartic,
        &starts,
        Array1::zeros(1).view(),
        Array1::from_vec(vec![1.0]).view(),
        &config,
    )
    .expect("shooting")
}

#[test]
fn newtonian_crossing_does_not_recross_immediately() {
    let transmission = shoot(0.0, 200, 80, 2);
    assert!(
        (transmission.kappa - 1.0).abs() < 1.0e-9,
        "kappa {} at t {}",
        transmission.kappa,
        transmission.time[2]
    );
    assert!(transmission.forward > 20, "forward {}", transmission.forward);
}

#[test]
fn high_friction_matches_kramers() {
    // ω_b = 2, γ = 2, α = γ / (2 ω_b) = 0.5, κ = sqrt(5)/2 - 1/2.
    let alpha: f64 = 2.0 / 4.0;
    let kramers = (1.0 + alpha * alpha).sqrt() - alpha;
    let transmission = shoot(2.0, 4000, 280, 120);
    let window = &transmission.kappa_of_t[100..140];
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    assert!(
        (mean - kramers).abs() < 0.06,
        "plateau {mean} kramers {kramers} curve {window:?}"
    );
    assert!(transmission.error < 0.15, "error {}", transmission.error);
}
