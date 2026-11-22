//! Sample the pressure of an ideal gas with `npt_lattice`.
//!
//! The atoms do not interact, so the stress is zero and the pressure comes
//! from the kinetic term alone. With the centre-of-mass momentum removed it
//! is `(N - 1) k T / V`. A near-zero compressibility holds the volume, so
//! the zero target pressure is never reached and `pressure_converged`
//! stays false: the run reads the gas pressure, it does not equilibrate it.
//!
//! `cargo run --release --example npt_lattice`

use ndarray::Array1;
use rgsaddle::{BOLTZMANN_EV_PER_K, Cell, NptConfig, SaddleError, npt_lattice};

fn main() -> Result<(), SaddleError> {
    let per_side = 5;
    let spacing = 4.0;
    let mut positions = Vec::new();
    for i in 0..per_side {
        for j in 0..per_side {
            for k in 0..per_side {
                positions.extend([i, j, k].map(|index| index as f64 * spacing));
            }
        }
    }
    let atoms = positions.len() / 3;
    let mut positions = Array1::from_vec(positions);
    let edge = per_side as f64 * spacing;
    let mut cell = Cell([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]);

    let temperature_k = 300.0;
    let config = NptConfig::default()
        .with_repeats(per_side)
        .with_steps(4_000)
        .with_maximum_steps(8_000)
        .with_extension_steps(2_000)
        .with_equilibration_steps(1_000)
        .with_compressibility_per_ev_a3(1.0e-12)
        .with_seed(3);
    let report = npt_lattice(
        &mut positions,
        &mut cell,
        temperature_k,
        63.546,
        config,
        |x, _| Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3])),
    )?;

    let ideal = (atoms - 1) as f64 * BOLTZMANN_EV_PER_K * temperature_k / edge.powi(3);
    println!(
        "{} steps, pressure {:.4e} +- {:.1e} eV/A^3 (resolved {}), ideal gas {ideal:.4e}",
        report.steps,
        report.mean_pressure,
        report.pressure_standard_error,
        report.pressure_error_resolved
    );
    Ok(())
}
