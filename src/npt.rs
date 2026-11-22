//! Isotropic NPT sampling of a crystal lattice: Langevin dynamics with a
//! Berendsen barostat.
//!
//! The force evaluator supplies a tension-positive Cauchy stress.  The
//! reported pressure is therefore minus one third of its trace.

use ndarray::{Array1, ArrayView1};

use crate::{BOLTZMANN_EV_PER_K, Cell, SaddleError, angular_scale_rad_s};

/// Controls an isotropic NPT lattice run.
///
/// Non-exhaustive: start from [`NptConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct NptConfig {
    /// Number of conventional cells along each lattice vector.
    pub repeats: usize,
    /// Initial number of integration steps.
    pub steps: usize,
    /// Largest number of integration steps after extensions.
    pub maximum_steps: usize,
    /// Number of steps added when the pressure criterion is not met.
    pub extension_steps: usize,
    /// Steps discarded before block statistics are accumulated.
    pub equilibration_steps: usize,
    /// Number of production steps in one statistics block.
    pub block_steps: usize,
    /// Velocity-Verlet time step in seconds.
    pub timestep_s: f64,
    /// Langevin friction in inverse seconds.
    pub langevin_gamma_per_s: f64,
    /// Berendsen pressure-coupling time in seconds.
    pub berendsen_tau_s: f64,
    /// Isotropic compressibility in inverse eV per cubic Angstrom.
    pub compressibility_per_ev_a3: f64,
    /// Target thermodynamic pressure in eV per cubic Angstrom.
    pub target_pressure_ev_a3: f64,
    /// Seed for the deterministic thermostat stream. Every seed, including
    /// consecutive ones, starts an unrelated stream.
    pub seed: u64,
}

with_setters!(NptConfig {
    with_repeats => repeats: val usize,
    with_steps => steps: val usize,
    with_maximum_steps => maximum_steps: val usize,
    with_extension_steps => extension_steps: val usize,
    with_equilibration_steps => equilibration_steps: val usize,
    with_block_steps => block_steps: val usize,
    with_timestep_s => timestep_s: val f64,
    with_langevin_gamma_per_s => langevin_gamma_per_s: val f64,
    with_berendsen_tau_s => berendsen_tau_s: val f64,
    with_compressibility_per_ev_a3 => compressibility_per_ev_a3: val f64,
    with_target_pressure_ev_a3 => target_pressure_ev_a3: val f64,
    with_seed => seed: val u64,
});

impl Default for NptConfig {
    fn default() -> Self {
        Self {
            repeats: 6,
            steps: 4_000,
            maximum_steps: 4_000,
            extension_steps: 1_000,
            equilibration_steps: 2_000,
            block_steps: 250,
            timestep_s: 2.0e-15,
            langevin_gamma_per_s: 1.0e12,
            berendsen_tau_s: 0.5e-12,
            compressibility_per_ev_a3: 1.0 / 0.87,
            target_pressure_ev_a3: 0.0,
            seed: 11,
        }
    }
}

/// Block-averaged result of an NPT lattice run.
#[derive(Clone, Copy, Debug)]
pub struct NptReport {
    /// Total number of integration steps.
    pub steps: usize,
    /// Number of complete production blocks.
    pub blocks: usize,
    /// Mean conventional lattice parameter in Angstrom.
    pub mean_lattice_a: f64,
    /// Standard error of the lattice block means in Angstrom.
    pub lattice_standard_error: f64,
    /// Mean thermodynamic pressure in eV per cubic Angstrom.
    pub mean_pressure: f64,
    /// Standard error of the pressure block means.
    pub pressure_standard_error: f64,
    /// Whether `|mean pressure - target| <= 2 SE`, including exact zero.
    pub pressure_converged: bool,
}

/// Seed expansion of Steele, Lea and Flood (the SplitMix64 generator).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** (Blackman and Vigna) with SplitMix64 seeding, the
/// initialisation its authors prescribe. Nearby seeds give unrelated
/// states, so no warm-up is needed.
#[derive(Clone, Copy)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut mixer = seed;
        let mut state = [0_u64; 4];
        for word in &mut state {
            *word = splitmix64(&mut mixer);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)` with 53 random bits.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn zero_mean(values: &mut Array1<f64>) {
    let atoms = values.len() / 3;
    let mut mean = [0.0; 3];
    for atom in 0..atoms {
        for c in 0..3 {
            mean[c] += values[3 * atom + c] / atoms as f64;
        }
    }
    for atom in 0..atoms {
        for c in 0..3 {
            values[3 * atom + c] -= mean[c];
        }
    }
}

fn determinant(cell: Cell) -> f64 {
    let h = cell.0;
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

fn block_statistics(values: &[f64], block_steps: usize) -> (f64, f64, usize) {
    let blocks = values.len() / block_steps;
    let means: Vec<f64> = values[..blocks * block_steps]
        .chunks_exact(block_steps)
        .map(|block| block.iter().sum::<f64>() / block.len() as f64)
        .collect();
    let mean = means.iter().sum::<f64>() / blocks as f64;
    let variance = if blocks > 1 {
        means
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (blocks - 1) as f64
    } else {
        0.0
    };
    (mean, (variance / blocks as f64).sqrt(), blocks)
}

fn summarize(lattice: &[f64], pressure: &[f64], config: NptConfig, steps: usize) -> NptReport {
    let (mean_lattice_a, lattice_standard_error, blocks) =
        block_statistics(lattice, config.block_steps);
    let (mean_pressure, pressure_standard_error, _) =
        block_statistics(pressure, config.block_steps);
    let residual = (mean_pressure - config.target_pressure_ev_a3).abs();
    let pressure_converged = if pressure_standard_error == 0.0 {
        residual == 0.0
    } else {
        residual <= 2.0 * pressure_standard_error
    };
    NptReport {
        steps,
        blocks,
        mean_lattice_a,
        lattice_standard_error,
        mean_pressure,
        pressure_standard_error,
        pressure_converged,
    }
}

/// Run isotropic NPT dynamics and return block standard errors.
///
/// `evaluate` returns `(energy, gradient, tension-positive stress)` for
/// Cartesian `positions` and the row-major `cell`.  Energy is accepted for
/// evaluator compatibility but is not used by the integrator.  A run extends
/// by `extension_steps` until the zero-pressure rule is met or
/// `maximum_steps` is reached.
pub fn npt_lattice<F>(
    positions: &mut Array1<f64>,
    cell: &mut Cell,
    temperature_k: f64,
    mass_amu: f64,
    config: NptConfig,
    mut evaluate: F,
) -> Result<NptReport, SaddleError>
where
    F: FnMut(ArrayView1<'_, f64>, Cell) -> Result<(f64, Array1<f64>, [[f64; 3]; 3]), SaddleError>,
{
    let production = config.steps.saturating_sub(config.equilibration_steps);
    if positions.is_empty()
        || !positions.len().is_multiple_of(3)
        || config.repeats == 0
        || config.steps > config.maximum_steps
        || config.extension_steps == 0
        || config.block_steps == 0
        || production / config.block_steps < 2
        || !(temperature_k >= 0.0 && temperature_k.is_finite())
        || !(mass_amu > 0.0 && mass_amu.is_finite())
        || !(config.timestep_s > 0.0
            && config.langevin_gamma_per_s > 0.0
            && config.berendsen_tau_s > 0.0
            && config.compressibility_per_ev_a3 > 0.0)
    {
        return Err(SaddleError::Invalid(
            "NPT lattice sampling needs finite physical controls and at least two production blocks"
                .into(),
        ));
    }

    let thermal = BOLTZMANN_EV_PER_K * temperature_k;
    let omega = angular_scale_rad_s();
    let dt = config.timestep_s * omega;
    let gamma = config.langevin_gamma_per_s / omega;
    let tau = config.berendsen_tau_s * omega;
    let mut rng = Rng::new(config.seed);
    let mut velocity = Array1::zeros(positions.len());
    let sigma = (thermal / mass_amu).sqrt();
    for value in velocity.iter_mut() {
        *value = sigma * rng.normal();
    }
    zero_mean(&mut velocity);

    let mut lattice_samples = Vec::with_capacity(
        config
            .maximum_steps
            .saturating_sub(config.equilibration_steps),
    );
    let mut pressure_samples = Vec::with_capacity(lattice_samples.capacity());
    let mut next_check = config.steps;
    let mut report = None;

    for step in 0..config.maximum_steps {
        let (_, gradient, _) = evaluate(positions.view(), *cell)?;
        if gradient.len() != positions.len() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("NPT gradient"));
        }
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass_amu));
        *positions = &*positions + &(&velocity * (0.5 * dt));

        let damp = (-gamma * dt).exp();
        let noise = (thermal / mass_amu * (1.0 - damp * damp)).sqrt();
        for value in velocity.iter_mut() {
            *value = damp * *value + noise * rng.normal();
        }
        zero_mean(&mut velocity);
        *positions = &*positions + &(&velocity * (0.5 * dt));

        let (_, gradient, stress) = evaluate(positions.view(), *cell)?;
        if gradient.len() != positions.len()
            || gradient.iter().any(|value| !value.is_finite())
            || stress.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("NPT force or stress"));
        }
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass_amu));

        let configurational_stress = (stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        let kinetic_energy = velocity
            .iter()
            .map(|value| 0.5 * mass_amu * value * value)
            .sum::<f64>();
        let volume = determinant(*cell).abs();
        let kinetic_pressure = 2.0 * kinetic_energy / (3.0 * volume);
        let hydrostatic_stress = configurational_stress - kinetic_pressure;
        let drive = hydrostatic_stress + config.target_pressure_ev_a3;
        let volume_factor =
            (1.0 - config.compressibility_per_ev_a3 * dt / tau * drive).clamp(0.995, 1.005);
        let length_factor = volume_factor.cbrt();
        *positions *= length_factor;
        for row in &mut cell.0 {
            for value in row {
                *value *= length_factor;
            }
        }

        let completed = step + 1;
        if completed > config.equilibration_steps {
            lattice_samples.push(determinant(*cell).abs().cbrt() / config.repeats as f64);
            pressure_samples.push(-hydrostatic_stress);
        }
        if completed == next_check {
            let current = summarize(&lattice_samples, &pressure_samples, config, completed);
            if current.pressure_converged || completed == config.maximum_steps {
                return Ok(current);
            }
            report = Some(current);
            next_check = (next_check + config.extension_steps).min(config.maximum_steps);
        }
    }
    report.ok_or_else(|| SaddleError::Invalid("NPT run produced no report".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(seed: u64, length: usize) -> Vec<u64> {
        let mut rng = Rng::new(seed);
        (0..length).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn splitmix64_matches_the_published_stream() {
        let mut state = 0_u64;
        assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(&mut state), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(splitmix64(&mut state), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn xoshiro_stream_is_pinned_for_a_zero_seed() {
        assert_eq!(
            stream(0, 3),
            [
                0x99EC_5F36_CB75_F2B4,
                0xBF6E_1F78_4956_452A,
                0x1A5F_849D_4933_E6E0
            ]
        );
    }

    #[test]
    fn distinct_seeds_give_distinct_streams() {
        let streams: Vec<Vec<u64>> = (0..64).map(|seed| stream(seed, 8)).collect();
        for (a, first) in streams.iter().enumerate() {
            for second in &streams[a + 1..] {
                assert_ne!(first, second);
                assert!(first.iter().zip(second).all(|(x, y)| x != y));
            }
        }
        for pair in (0..32_u64).map(|k| (stream(2 * k, 8), stream(2 * k + 1, 8))) {
            assert_ne!(pair.0, pair.1);
        }
    }

    #[test]
    fn nearby_seeds_are_uncorrelated_in_their_first_draw() {
        let draws: Vec<f64> = (0..4096_u64).map(|seed| Rng::new(seed).uniform()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance =
            draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
        assert!((variance - 1.0 / 12.0).abs() < 0.005, "{variance}");
        let lag = draws
            .windows(2)
            .map(|pair| (pair[0] - mean) * (pair[1] - mean))
            .sum::<f64>()
            / (draws.len() - 1) as f64
            / variance;
        assert!(lag.abs() < 0.05, "{lag}");
    }

    #[test]
    fn normal_draws_have_unit_variance() {
        let mut rng = Rng::new(7);
        let draws: Vec<f64> = (0..200_000).map(|_| rng.normal()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance =
            draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((variance - 1.0).abs() < 0.02, "{variance}");
    }
}
