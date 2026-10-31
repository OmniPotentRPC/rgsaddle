//! The getting-started example: a band on the double well
//! V = (x² − 1)² + 2y² + 2z², which has minima at (±1, 0, 0) and a
//! saddle at the origin with barrier 1.
//!
//! ```text
//! cargo run --example double_well
//! ```

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError};

/// Energy and gradient of the double well at (x, y, z).
fn double_well(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
    (energy, [4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z])
}

struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, gradient) = double_well(row[0], row[1], row[2]);
            energies[i] = energy;
            for (k, g) in gradient.into_iter().enumerate() {
                gradients[(i, k)] = g;
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), SaddleError> {
    // Nine images from (-1, 0, 0) to (1, 0, 0), bowed off the path in y.
    let initial = Array2::from_shape_fn((9, 3), |(i, k)| {
        let t = i as f64 / 8.0;
        match k {
            0 => 2.0 * t - 1.0,
            1 => 0.3 * (std::f64::consts::PI * t).sin(),
            _ => 0.0,
        }
    });

    let config = BandConfig {
        force_tol: 1e-3,
        max_move: 0.1,
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, initial)?;
    let report = loop {
        let report = session.step(&DoubleWell)?;
        if report.status == BandStatus::Converged {
            break report;
        }
    };

    let ci = report.ci_index.expect("a climbing image");
    let positions = session.positions();
    let saddle = positions.row(ci);
    let (energy, _) = double_well(saddle[0], saddle[1], saddle[2]);
    println!(
        "{:?} after {} steps, max force {:.1e}",
        report.status, report.iteration, report.max_force
    );
    println!(
        "climbing image {ci} at ({:.4}, {:.4}, {:.4}), energy {energy:.6}",
        saddle[0], saddle[1], saddle[2]
    );
    Ok(())
}
